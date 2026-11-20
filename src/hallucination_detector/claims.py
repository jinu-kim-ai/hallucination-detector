"""Claim decomposition: split a model response into atomic factual assertions.

A sentence is cut into several claims when it packs several facts:

* coordinated verbs     "Curie discovered polonium and won two Nobel Prizes."
                        -> "Curie discovered polonium." / "Curie won two Nobel Prizes."
* non-restrictive       "Curie, who was born in Warsaw, won the prize in 1903."
  relative clauses      -> "Curie won the prize in 1903." / "Curie was born in Warsaw."

Sentences with nothing to check (questions, pleasantries, headings) are dropped, markdown list
markers are ignored, and a leading pronoun is resolved to the subject it refers to, so that
"She died in 1934." is checked as "Marie Curie died in 1934.".
"""
from __future__ import annotations

import re

from .schemas import Claim

_NON_FACTUAL = re.compile(
    r"^(i hope|hope this|let me know|feel free|in summary|in conclusion|overall,|"
    r"great question|sure[,!]|certainly[,!]|of course[,!])",
    re.IGNORECASE,
)
_LIST_ITEM = re.compile(r"^(\s*(?:[-*+•]|\d+[.)])\s+)")
_HEADING = re.compile(r"^\s*#{1,6}\s")
_EMPHASIS = re.compile(r"\*\*|__|`")
_SENTENCE_END = (".", "!", "?", ";")


def load_nlp(model: str = "en_core_web_sm"):
    import spacy

    try:
        return spacy.load(model)
    except OSError as exc:  # model not downloaded
        raise RuntimeError(f"spaCy model '{model}' is missing. Run: python -m spacy download {model}") from exc


class ClaimExtractor:
    def __init__(self, nlp=None, min_words: int = 3):
        self.nlp = nlp or load_nlp()
        self.min_words = min_words

    def extract(self, response: str) -> list[Claim]:
        # Blank out inline markdown without moving any character, so spans still index the original text.
        clean = _EMPHASIS.sub(lambda m: " " * len(m.group()), response)
        claims: list[Claim] = []
        referents = _Referents()
        seen: dict[str, int] = {}
        for block_start, block_end in _blocks(clean):
            doc = self.nlp(clean[block_start:block_end])
            for sent in doc.sents:
                text = sent.text.strip()
                if not text or text.endswith("?") or _NON_FACTUAL.match(text):
                    continue
                pronoun, referent = referents.resolve(sent)
                for piece, start, end in _decompose(sent):
                    piece = piece.strip(" ,;")
                    if len(piece.split()) < self.min_words:
                        continue
                    claim = Claim(len(claims) + 1, _finish(piece), text, block_start + start, block_start + end)
                    if referent and piece.lower().startswith(pronoun + " "):
                        claim.resolved_text = _finish(referent + piece[len(pronoun):])
                    key = re.sub(r"[^a-z0-9 ]", "", claim.resolved_text.lower())
                    claim.duplicate_of = seen.get(key)
                    seen.setdefault(key, claim.id)
                    claims.append(claim)
                referents.update(sent)
        return claims


# ------------------------------------------------------------------ text blocks
def _blocks(text: str) -> list[tuple[int, int]]:
    """Character ranges to parse separately: paragraphs, and each markdown list item on its own."""
    blocks: list[tuple[int, int]] = []
    para_start = para_end = None
    pos = 0
    for line in text.splitlines(keepends=True):
        start, end = pos, pos + len(line.rstrip("\r\n"))
        pos += len(line)
        item = _LIST_ITEM.match(line)
        if not line.strip() or _HEADING.match(line) or item:
            if para_start is not None:
                blocks.append((para_start, para_end))
                para_start = None
            if item:
                blocks.append((start + len(item.group(1)), end))
            continue
        if para_start is None:
            para_start = start
        para_end = end
    if para_start is not None:
        blocks.append((para_start, para_end))
    return blocks


# ------------------------------------------------------------------ sentence -> claims
def _decompose(sent) -> list[tuple[str, int, int]]:
    """Split one sentence into (claim text, start char, end char) pieces."""
    doc, root = sent.doc, sent.root
    excluded: set[int] = set()
    pieces: list[tuple[str, int, int]] = []

    # Non-restrictive relative clauses with a subject pronoun: ", who was born in Warsaw,"
    relative: list[tuple[str, int, int]] = []
    for verb in sent:
        if verb.dep_ != "relcl" or verb.pos_ not in ("VERB", "AUX"):
            continue
        clause = list(verb.subtree)
        left, right = clause[0], clause[-1]
        pronoun = next((c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")
                        and c.lower_ in ("who", "which", "that")), None)
        set_off = left.i > sent.start and doc[left.i - 1].text == ","
        if pronoun is None or not set_off:      # restrictive ("the man who ...") defines the noun: keep it whole
            continue
        clause_ids = {t.i for t in clause}
        head_words = [t.text for t in verb.head.subtree if t.i not in clause_ids and not t.is_punct]
        body = [t.text for t in clause if t.i != pronoun.i and t.text not in _SENTENCE_END]
        relative.append((_detokenize(" ".join(head_words + body)), left.idx, right.idx + len(right)))
        excluded |= clause_ids | {left.i - 1}
        if right.i + 1 < sent.end and doc[right.i + 1].text == ",":
            excluded.add(right.i + 1)

    # Coordinated verbs on the main clause: "S V1 O1 and V2 O2"
    conj_verbs = [t for t in root.children if t.dep_ == "conj" and t.pos_ in ("VERB", "AUX") and t.i not in excluded]
    if not conj_verbs and not relative:
        return [(re.sub(r"\s{2,}", " ", sent.text), sent.start_char, sent.end_char)]

    subject = next((t for t in root.children if t.dep_ in ("nsubj", "nsubjpass")), None)
    subj_text = " ".join(t.text for t in subject.subtree if t.i not in excluded) if subject is not None else ""

    main_excluded = set(excluded)
    for verb in conj_verbs:
        main_excluded.update(t.i for t in verb.subtree)
        for t in root.children:                 # the "and" / comma right before the conjunct
            if t.dep_ in ("cc", "punct") and root.i < t.i < verb.i:
                main_excluded.add(t.i)

    def keep(t) -> bool:
        return t.text not in _SENTENCE_END

    main_tokens = [t for t in sent if t.i not in main_excluded and keep(t)]
    if main_tokens:
        pieces.append((_detokenize(" ".join(t.text for t in main_tokens)),
                       main_tokens[0].idx, main_tokens[-1].idx + len(main_tokens[-1])))
    for verb in conj_verbs:
        own_subject = any(c.dep_ in ("nsubj", "nsubjpass") for c in verb.children)
        nested = {t.i for v in verb.children if v.dep_ == "conj" for t in v.subtree}
        tokens = [t for t in verb.subtree if t.i not in nested and t.i not in excluded and keep(t)]
        if not tokens:
            continue
        body = " ".join(t.text for t in tokens)
        text = body if own_subject or not subj_text else f"{subj_text} {body}"
        pieces.append((_detokenize(text), tokens[0].idx, tokens[-1].idx + len(tokens[-1])))
    return pieces + relative


# ------------------------------------------------------------------ pronouns
class _Referents:
    """Remembers recent sentence subjects so a leading pronoun can be replaced by what it refers to."""

    def __init__(self):
        self.person = self.thing = self.plural = None

    @staticmethod
    def _subject(sent):
        return next((t for t in sent.root.children if t.dep_ in ("nsubj", "nsubjpass")), None)

    def resolve(self, sent) -> tuple[str, str | None]:
        """(pronoun, referent) when the sentence's subject is he / she / it / they and a referent is known."""
        subj = self._subject(sent)
        if subj is None or subj.pos_ != "PRON":
            return "", None
        word = subj.lower_
        referent = {"he": self.person, "she": self.person, "it": self.thing, "they": self.plural}.get(word)
        return word, referent

    def update(self, sent) -> None:
        subj = self._subject(sent)
        if subj is None or subj.pos_ == "PRON":
            return
        ent = next((e for e in sent.ents if e.start <= subj.i < e.end), None)
        chunk = next((c for c in sent.noun_chunks if c.start <= subj.i < c.end), None)
        phrase = ent.text if ent is not None else (chunk.text if chunk is not None else subj.text)
        if ent is not None and ent.label_ == "PERSON":
            self.person = phrase
        elif ent is None and subj.tag_ in ("NNS", "NNPS"):      # "the astronauts"; a company name is an "it"
            self.plural = phrase
        else:
            self.thing = phrase


# ------------------------------------------------------------------ text helpers
def _detokenize(text: str) -> str:
    text = re.sub(r"\s+([,.;:%)])", r"\1", text)
    text = re.sub(r"([($])\s+", r"\1", text)
    text = re.sub(r"\s+'s\b", "'s", text)
    text = re.sub(r"(\w) - (\w)", r"\1-\2", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _finish(text: str) -> str:
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!")) else text + "."
