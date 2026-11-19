"""Claim decomposition: split a model answer into atomic factual assertions.

Uses spaCy's dependency parse to split sentences on coordinated verb clauses
("Curie discovered polonium and won two Nobel Prizes" -> two claims), and drops
sentences that carry no checkable content (questions, hedges, pleasantries).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NON_FACTUAL = re.compile(
    r"^(i hope|hope this|let me know|feel free|in summary|in conclusion|overall,|"
    r"great question|sure[,!]|certainly[,!]|of course[,!])",
    re.IGNORECASE,
)


@dataclass
class Claim:
    text: str
    sentence: str        # the sentence the claim came from
    start: int           # char offset of the sentence in the answer
    end: int


def _load_nlp(model: str = "en_core_web_sm"):
    import spacy

    try:
        return spacy.load(model)
    except OSError as exc:  # model not downloaded
        raise RuntimeError(
            f"spaCy model '{model}' is missing. Run: python -m spacy download {model}"
        ) from exc


class ClaimExtractor:
    def __init__(self, nlp=None):
        self.nlp = nlp or _load_nlp()

    def extract(self, answer: str) -> list[Claim]:
        doc = self.nlp(answer)
        claims: list[Claim] = []
        for sent in doc.sents:
            text = sent.text.strip()
            if not text or text.endswith("?") or _NON_FACTUAL.match(text):
                continue
            for piece in self._split_coordinated(sent):
                piece = piece.strip(" ,;")
                if len(piece.split()) >= 3:
                    claims.append(Claim(_finish(piece), text, sent.start_char, sent.end_char))
        return claims

    @staticmethod
    def _split_coordinated(sent) -> list[str]:
        """Split 'S V1 O1 and V2 O2' into 'S V1 O1' and 'S V2 O2'."""
        root = sent.root
        conj_verbs = [t for t in root.children if t.dep_ == "conj" and t.pos_ in ("VERB", "AUX")]
        if not conj_verbs:
            return [sent.text]

        subject = next((t for t in root.children if t.dep_ in ("nsubj", "nsubjpass")), None)
        subj_text = " ".join(t.text for t in subject.subtree) if subject is not None else ""

        excluded: set[int] = set()
        for verb in conj_verbs:
            excluded.update(t.i for t in verb.subtree)
            # drop the coordinating conjunction / comma right before the conjunct
            for t in root.children:
                if t.dep_ in ("cc", "punct") and t.i < verb.i and t.i > root.i:
                    excluded.add(t.i)

        def keep(t) -> bool:
            return t.text not in (".", "!", "?", ";")

        main = " ".join(t.text for t in sent if t.i not in excluded and keep(t))
        pieces = [main]
        for verb in conj_verbs:
            own_subject = any(c.dep_ in ("nsubj", "nsubjpass") for c in verb.children)
            nested = {t.i for v in verb.children if v.dep_ == "conj" for t in v.subtree}
            body = " ".join(t.text for t in verb.subtree if t.i not in nested and keep(t))
            pieces.append(body if own_subject or not subj_text else f"{subj_text} {body}")
        return [_detokenize(p) for p in pieces]


def _detokenize(text: str) -> str:
    text = re.sub(r"\s+([,.;:%)])", r"\1", text)
    text = re.sub(r"([($])\s+", r"\1", text)
    text = re.sub(r"\s+'s\b", "'s", text)
    text = re.sub(r"(\w) - (\w)", r"\1-\2", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _finish(text: str) -> str:
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!")) else text + "."
