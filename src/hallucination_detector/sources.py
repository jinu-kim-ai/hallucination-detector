"""Load source documents and split them into sentences with character offsets."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .schemas import Sentence, SourceDocument

MAX_SENTENCE_WORDS = 120      # the embedder truncates around 256 word pieces; split before that


def as_documents(sources) -> list[SourceDocument]:
    """Accept a string, a list of strings, dicts ({"id", "text", "title"}) or SourceDocuments."""
    if sources is None:
        return []
    if isinstance(sources, (str, SourceDocument, dict)):
        sources = [sources]
    docs: list[SourceDocument] = []
    for i, s in enumerate(sources, 1):
        if isinstance(s, SourceDocument):
            docs.append(s)
        elif isinstance(s, dict):
            docs.append(SourceDocument(str(s.get("id") or f"source-{i}"), s.get("text") or "", s.get("title") or ""))
        else:
            docs.append(SourceDocument(f"source-{i}", str(s)))
    ids = [d.id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"source ids must be unique, got {ids}")
    return docs


def load_request(path: str | Path) -> tuple[str, list[SourceDocument]]:
    """Read {"response": ..., "sources": [{"id", "title", "text"}, ...]} from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "response" not in data:
        raise ValueError(f"{path}: missing required field 'response'")
    return data["response"], as_documents(data.get("sources", []))


def load_text_sources(paths: list[str | Path]) -> list[SourceDocument]:
    """One document per text file; the file name is the id."""
    docs, seen = [], set()
    for p in map(Path, paths):
        doc_id = p.name if p.name not in seen else str(p)
        seen.add(p.name)
        docs.append(SourceDocument(doc_id, p.read_text(encoding="utf-8")))
    return docs


def split_sentences(doc: SourceDocument, nlp, min_words: int = 3) -> tuple[list[Sentence], list[str]]:
    """Sentences of one document (with spans into doc.text) and any warnings."""
    warnings: list[str] = []
    if not doc.text.strip():
        return [], warnings
    spans: list[tuple[int, int]] = []
    for sent in nlp(doc.text).sents:
        start, end = _trim(doc.text, sent.start_char, sent.end_char)
        if len(doc.text[start:end].split()) < min_words:
            continue
        if len(doc.text[start:end].split()) > MAX_SENTENCE_WORDS:
            warnings.append(f"source '{doc.id}': a sentence of over {MAX_SENTENCE_WORDS} words was split into parts")
            spans.extend(_split_long(doc.text, start, end))
        else:
            spans.append((start, end))
    if not spans:                                   # e.g. a two-word document: keep it as one unit
        spans = [_trim(doc.text, 0, len(doc.text))]
    return [Sentence(doc.id, i, doc.text[a:b], a, b) for i, (a, b) in enumerate(spans)], warnings


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_long(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Cut an over-long sentence at semicolons / commas into pieces of at most MAX_SENTENCE_WORDS words."""
    pieces, piece_start, words = [], start, 0
    for m in re.finditer(r"\S+", text[start:end]):
        words += 1
        token_end = start + m.end()
        if words >= MAX_SENTENCE_WORDS // 2 and m.group().endswith((";", ",")) or words >= MAX_SENTENCE_WORDS:
            pieces.append(_trim(text, piece_start, token_end))
            piece_start, words = token_end, 0
    if piece_start < end and text[piece_start:end].strip():
        pieces.append(_trim(text, piece_start, end))
    return pieces
