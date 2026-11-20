"""Data objects passed between pipeline stages. No model code lives here."""
from __future__ import annotations

from dataclasses import dataclass, field

SUPPORTED, CONTRADICTED, FABRICATED, UNSUPPORTED = "SUPPORTED", "CONTRADICTED", "FABRICATED", "UNSUPPORTED"
LABELS = [SUPPORTED, CONTRADICTED, FABRICATED, UNSUPPORTED]
HALLUCINATION_LABELS = {CONTRADICTED, FABRICATED, UNSUPPORTED}


@dataclass(frozen=True)
class SourceDocument:
    id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Sentence:
    """One sentence of a source document, with its character span in that document."""
    source_id: str
    index: int
    text: str
    start: int
    end: int


@dataclass
class Claim:
    id: int
    text: str                    # the claim as written (after splitting compound sentences)
    sentence: str                # the response sentence it came from
    start: int                   # character span in the response
    end: int
    resolved_text: str = ""      # text used for checking (leading pronoun replaced by its referent)
    duplicate_of: int | None = None
    ambiguous_pronoun: str | None = None   # "she" when two people could be meant, so it was left unresolved
    pronoun_referent: str | None = None    # with ambiguous_pronoun: the remembered subject ("Marie Curie")
    pronoun_rival: str | None = None       # with ambiguous_pronoun: the other candidate, as a full name ("Pierre Curie")

    def __post_init__(self):
        self.resolved_text = self.resolved_text or self.text


@dataclass
class Citation:
    source_id: str
    sentence_index: int
    source_span: tuple[int, int]
    quote: str
    similarity: float
    nli: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "sentence_index": self.sentence_index,
            "source_span": list(self.source_span),
            "quote": self.quote,
            "similarity": round(self.similarity, 3),
            "nli": {k: round(v, 3) for k, v in self.nli.items()} if self.nli else None,
        }


@dataclass
class ClaimVerdict:
    claim: Claim
    label: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    best_similarity: float = 0.0     # closest source sentence, cited or not (none is cited below related_threshold)

    @property
    def is_hallucination(self) -> bool:
        return self.label in HALLUCINATION_LABELS

    def to_dict(self) -> dict:
        out = {
            "id": self.claim.id,
            "text": self.claim.text,
            "response_span": [self.claim.start, self.claim.end],
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "best_similarity": round(self.best_similarity, 3),
            "reasons": self.reasons,
            "citations": [c.to_dict() for c in self.citations],
        }
        if self.claim.resolved_text != self.claim.text:
            out["checked_as"] = self.claim.resolved_text
        if self.claim.duplicate_of is not None:
            out["duplicate_of"] = self.claim.duplicate_of
        return out


@dataclass
class Report:
    response: str
    verdicts: list[ClaimVerdict]
    sources: list[SourceDocument] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    unscorable: bool = False      # e.g. non-English text: verdicts are not meaningful

    @property
    def results(self) -> list[ClaimVerdict]:
        """v1 name for `verdicts`."""
        return self.verdicts

    @property
    def scored(self) -> list[ClaimVerdict]:
        """Verdicts that count toward the score (a repeated claim counts once)."""
        return [v for v in self.verdicts if v.claim.duplicate_of is None]

    @property
    def reliability_score(self) -> float | None:
        """100 x confidence mass of supported claims / confidence mass of all claims. None if nothing to check."""
        scored = self.scored
        if not scored or self.unscorable:
            return None
        total = sum(v.confidence for v in scored)
        good = sum(v.confidence for v in scored if not v.is_hallucination)
        return 100.0 * good / total if total else None

    @property
    def hallucination_rate(self) -> float | None:
        scored = self.scored
        if not scored or self.unscorable:
            return None
        return sum(v.is_hallucination for v in scored) / len(scored)

    def counts(self) -> dict[str, int]:
        return {lab: sum(v.label == lab for v in self.scored) for lab in LABELS}

    def to_dict(self) -> dict:
        score, rate = self.reliability_score, self.hallucination_rate
        return {
            "reliability_score": None if score is None else round(score, 1),
            "hallucination_rate": None if rate is None else round(rate, 3),
            "counts": self.counts(),
            "warnings": self.warnings,
            "claims": [v.to_dict() for v in self.verdicts],
            "config": self.config,
        }
