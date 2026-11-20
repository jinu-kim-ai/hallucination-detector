"""Load source documents and split them into sentences with character offsets."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import Config
from .schemas import Sentence, SourceDocument

MAX_SENTENCE_WORDS = Config.max_sentence_words    # the embedder truncates around 256 word pieces; split before that
MIN_SOURCE_WORDS = Config.min_source_words


def as_documents(sources) -> list[SourceDocument]:
    """Accept a string, a list of strings, dicts ({"id", "text", "title"}) or SourceDocuments."""
    if sources is None:
        return []
    if isinstance(sources, (str, SourceDocument, dict)):
        sources = [sources]
    sources = list(sources)
    given = [g for g in map(_given_id, sources) if g is not None]
    duplicates = sorted({g for g in given if given.count(g) > 1})
    if duplicates:
        raise ValueError(f"source ids must be unique, got duplicates {duplicates}")
    taken = set(given)          # generated ids must not take an id the caller chose for a later source
    return [_as_document(s, i, taken) for i, s in enumerate(sources, 1)]


def _given_id(source) -> str | None:
    """The caller's id, or None when it is missing or blank (a 'source-N' id is generated then)."""
    if isinstance(source, SourceDocument):
        given = source.id
    elif isinstance(source, dict):
        given = source.get("id")
    else:
        return None
    if given is None or str(given) == "":
        return None
    return str(given)                     # str() keeps a falsy id such as 0


def _as_document(source, position: int, taken: set[str]) -> SourceDocument:
    doc_id = _given_id(source)
    if doc_id is None:
        doc_id = _free_id(f"source-{position}", taken)
    # every form gets the same checks: a SourceDocument with text=None is blank, not a crash later
    if isinstance(source, SourceDocument):
        text, title = source.text, source.title
    elif isinstance(source, dict):
        text, title = source.get("text"), source.get("title")
    else:
        text, title = source, None
    return SourceDocument(doc_id, _check_text(text, doc_id, "text"), _check_text(title, doc_id, "title"))


def _check_text(value, doc_id: str, field: str) -> str:
    """Missing / None text is blank; anything else that is not a string would crash deep in the pipeline."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"source '{doc_id}': field '{field}' must be a string, got {type(value).__name__}")
    return value


def _free_id(base: str, taken: set[str]) -> str:
    doc_id, n = base, 2
    while doc_id in taken:
        doc_id, n = f"{base}#{n}", n + 1
    taken.add(doc_id)
    return doc_id


def load_request(path: str | Path) -> tuple[str, list[SourceDocument]]:
    """Read {"response": ..., "sources": [{"id", "title", "text"}, ...]} from a JSON file."""
    try:
        data = json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object with 'response' and 'sources'")
    for field, kind in (("response", str), ("sources", list)):
        if field not in data:
            raise ValueError(f"{path}: missing required field '{field}'")
        if not isinstance(data[field], kind):
            raise ValueError(f"{path}: field '{field}' must be a {'string' if kind is str else 'list'}, "
                             f"got {type(data[field]).__name__}")
    try:
        return data["response"], as_documents(data["sources"])
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_text_sources(paths: list[str | Path]) -> list[SourceDocument]:
    """One document per text file; the file name is the id (the path, then a #n suffix, if that is taken)."""
    docs, used = [], set()
    for p in map(Path, paths):
        doc_id = _free_id(p.name if p.name not in used else str(p), used)
        docs.append(SourceDocument(doc_id, read_text(p)))
    return docs


def read_text(path: str | Path) -> str:
    """A text file exactly as written: spans computed on the result index the file itself."""
    # utf-8-sig drops a BOM (PowerShell and Notepad write one); newline="" keeps \r\n so spans index the real file
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return fh.read()


def split_sentences(doc: SourceDocument, nlp, min_words: int = MIN_SOURCE_WORDS,
                    max_words: int = MAX_SENTENCE_WORDS) -> tuple[list[Sentence], list[str]]:
    """Sentences of one document (with spans into doc.text) and any warnings."""
    warnings: list[str] = []
    if not re.search(r"\w", doc.text):              # blank, or only a BOM / zero-width spaces / punctuation
        return [], warnings
    found = [(a, b) for a, b in (_trim(doc.text, s, e) for s, e in _sentence_bounds(doc.text, nlp)) if a < b]
    spans: list[tuple[int, int]] = []
    for start, end in found:
        words = len(doc.text[start:end].split())
        if words < min_words:
            continue
        if words > max_words:
            warnings.append(f"source '{doc.id}': a sentence of over {max_words} words was split into parts")
            spans.extend(_split_long(doc.text, start, end, max_words))
        else:
            spans.append((start, end))
    if not spans:                                   # e.g. a two-word document: keep it as one unit
        spans = _group_short(doc.text, found or [_trim(doc.text, 0, len(doc.text))], max_words)
        if len(spans) > 1:
            warnings.append(f"source '{doc.id}': a passage of over {max_words} words was split into parts")
    return [Sentence(doc.id, i, doc.text[a:b], a, b) for i, (a, b) in enumerate(spans)], warnings


def _sentence_bounds(text: str, nlp) -> list[tuple[int, int]]:
    """spaCy sentence spans; text over nlp.max_length is parsed in chunks, offsets shifted back into `text`."""
    bounds = []
    for offset, chunk in _chunks(text, getattr(nlp, "max_length", 1_000_000)):
        bounds.extend((offset + s.start_char, offset + s.end_char) for s in nlp(chunk).sents)
    return bounds


def _chunks(text: str, limit: int) -> list[tuple[int, str]]:
    """(offset, chunk) pieces of at most `limit` characters, cut at the best boundary available."""
    out, start = [], 0
    while len(text) - start > limit:
        cut = _cut_point(text, start, start + limit)
        out.append((start, text[start:cut]))
        start = cut
    out.append((start, text[start:]))
    return out


def _cut_point(text: str, lo: int, hi: int) -> int:
    """Last paragraph break, line break, sentence end or space in the back half of text[lo:hi]."""
    mid = lo + (hi - lo) // 2                       # cutting in the back half keeps the chunk count linear
    window = text[mid:hi]
    for pattern in (r"\n\s*\n", r"\n", r"[.!?][\"')\]]*\s", r"\s"):
        last = None
        for last in re.finditer(pattern, window):
            pass
        if last is not None:
            return mid + last.end()
    return hi                                       # one enormous token: a hard cut is all that is left


def _group_short(text: str, spans: list[tuple[int, int]], max_words: int) -> list[tuple[int, int]]:
    """Join consecutive short sentences into units of at most `max_words` words."""
    pieces = [p for a, b in spans
              for p in (_split_long(text, a, b, max_words) if len(text[a:b].split()) > max_words else [(a, b)])]
    groups: list[tuple[int, int]] = []
    words = 0
    for a, b in pieces:
        n = len(text[a:b].split())
        if groups and words + n <= max_words:
            groups[-1], words = (groups[-1][0], b), words + n
        else:
            groups.append((a, b))
            words = n
    return groups


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_long(text: str, start: int, end: int, max_words: int = MAX_SENTENCE_WORDS) -> list[tuple[int, int]]:
    """Cut an over-long sentence at semicolons / commas into pieces of at most `max_words` words."""
    pieces, piece_start, words = [], start, 0
    for m in re.finditer(r"\S+", text[start:end]):
        words += 1
        token_end = start + m.end()
        if words >= max_words // 2 and m.group().endswith((";", ",")) or words >= max_words:
            pieces.append(_trim(text, piece_start, token_end))
            piece_start, words = token_end, 0
    if piece_start < end and text[piece_start:end].strip():
        pieces.append(_trim(text, piece_start, end))
    return pieces
