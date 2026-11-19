"""Claim-level hallucination detection against a source document.

Pipeline per claim:
  1. Retrieve the most similar source sentences with sentence embeddings.
  2. Extract checkable facts from the claim (numbers, named entities, negation).
  3. Compare those facts with the retrieved evidence and the whole source.
  4. Assign a label + confidence and keep the reasons as an audit trail.

Labels
  SUPPORTED     evidence closely matches and every fact in the claim is grounded
  CONTRADICTED  related evidence exists but a number / entity / polarity disagrees
  FABRICATED    the claim introduces specific facts that appear nowhere in the source
  UNSUPPORTED   nothing in the source backs the claim (no specific conflict either)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

import numpy as np

from .claims import Claim, ClaimExtractor, _load_nlp
from .embeddings import get_embedder

SUPPORTED, CONTRADICTED, FABRICATED, UNSUPPORTED = "SUPPORTED", "CONTRADICTED", "FABRICATED", "UNSUPPORTED"
LABELS = [SUPPORTED, CONTRADICTED, FABRICATED, UNSUPPORTED]
HALLUCINATION_LABELS = {CONTRADICTED, FABRICATED, UNSUPPORTED}

# Entity types that carry checkable facts. CARDINAL/PERCENT/MONEY are handled as numbers.
FACT_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "NORP", "FAC", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW"}
WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "hundred": 100,
    "thousand": 1000, "million": 1e6, "billion": 1e9, "half": 0.5, "dozen": 12,
}
NEGATION_WORDS = {"not", "no", "never", "n't", "none", "neither", "nor", "without", "cannot"}


@dataclass
class Evidence:
    sentence: str
    similarity: float


@dataclass
class ClaimResult:
    claim: str
    label: str
    confidence: float
    evidence: list[Evidence]
    reasons: list[str] = field(default_factory=list)
    sentence: str = ""

    @property
    def is_hallucination(self) -> bool:
        return self.label in HALLUCINATION_LABELS

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectionReport:
    answer: str
    results: list[ClaimResult]

    @property
    def hallucination_rate(self) -> float:
        return sum(r.is_hallucination for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def reliability_score(self) -> float:
        """Confidence-weighted share of claims that are grounded (0-100)."""
        if not self.results:
            return 100.0
        good = sum(r.confidence for r in self.results if not r.is_hallucination)
        bad = sum(r.confidence for r in self.results if r.is_hallucination)
        return 100.0 * good / (good + bad) if good + bad else 100.0

    def counts(self) -> dict[str, int]:
        return {lab: sum(r.label == lab for r in self.results) for lab in LABELS}


@dataclass
class _Facts:
    numbers: set[float]
    entities: list[tuple[str, str]]  # (text, label)
    negated: bool


class HallucinationDetector:
    def __init__(
        self,
        embedder: str = "all-MiniLM-L6-v2",
        support_threshold: float = 0.65,
        related_threshold: float = 0.45,
        contradiction_threshold: float = 0.7,
        top_k: int = 3,
        use_nli: bool = False,
        nlp=None,
    ):
        self.nlp = nlp or _load_nlp()
        self.extractor = ClaimExtractor(self.nlp)
        self.embedder = get_embedder(embedder)
        self.support_threshold = support_threshold
        self.related_threshold = related_threshold
        self.contradiction_threshold = contradiction_threshold
        self.top_k = top_k
        self.nli = _NLI() if use_nli else None
        self._source_cache: dict[str, tuple[list[str], np.ndarray, str]] = {}

    # ------------------------------------------------------------------ public API
    def detect(self, answer: str, source: str) -> DetectionReport:
        claims = self.extractor.extract(answer)
        return DetectionReport(answer, [self.check_claim(c, source) for c in claims])

    def check_claim(self, claim: Claim | str, source: str) -> ClaimResult:
        text = claim.text if isinstance(claim, Claim) else claim
        sentence = claim.sentence if isinstance(claim, Claim) else text
        sents, sent_emb, source_lower = self._prepare_source(source)

        sims = sent_emb @ self.embedder.encode([text])[0]
        order = np.argsort(-sims)[: self.top_k]
        evidence = [Evidence(sents[i], float(sims[i])) for i in order]
        best_sim = evidence[0].similarity if evidence else 0.0
        related = [e for e in evidence if e.similarity >= self.related_threshold]

        facts = self._facts(text)
        # Only evidence about the *same* fact can contradict; weakly related text cannot.
        conflicts = self._conflicts(facts, related, source_lower, text) if best_sim >= self.contradiction_threshold else []
        novel = self._novel_facts(facts, source_lower)

        nli_scores = self.nli.score(evidence[0].sentence, text) if self.nli and evidence else None
        label, conf, reasons = self._decide(best_sim, conflicts, novel, nli_scores)
        return ClaimResult(text, label, round(conf, 3), evidence, reasons, sentence)

    # ------------------------------------------------------------------ decision
    def _decide(self, sim, conflicts, novel, nli):
        s, r = self.support_threshold, self.related_threshold
        reasons = [f"best evidence similarity {sim:.2f}"]

        if nli and sim >= self.contradiction_threshold and nli["contradiction"] >= 0.7:
            conflicts = conflicts + [f"NLI contradiction p={nli['contradiction']:.2f}"]
        if conflicts:
            conf = min(1.0, 0.55 + 0.15 * len(conflicts) + 0.3 * (sim - r) / (1 - r))
            return CONTRADICTED, conf, reasons + conflicts
        if novel:
            conf = min(1.0, 0.55 + 0.1 * len(novel) + 0.35 * max(0.0, 1 - sim))
            return FABRICATED, conf, reasons + [f"not found in source: {', '.join(novel)}"]
        if sim >= s or (nli and nli["entailment"] >= 0.7 and sim >= r):
            conf = 0.5 + 0.5 * min(1.0, (sim - r) / (1 - r))
            if nli:
                conf = max(conf, nli["entailment"])
            return SUPPORTED, conf, reasons
        conf = 0.5 + 0.5 * (s - sim) / s
        return UNSUPPORTED, conf, reasons + [f"no source sentence reaches support threshold {s:.2f}"]

    # ------------------------------------------------------------------ fact checks
    def _facts(self, text: str) -> _Facts:
        doc = self.nlp(text)
        # Relative dates without numbers ("last year", "the century") are too paraphrase-prone to check.
        ents = [(e.text, e.label_) for e in doc.ents
                if e.label_ in FACT_ENTITY_LABELS and not (e.label_ == "DATE" and not _numbers(e.text))]
        negated = any(t.dep_ == "neg" or t.lower_ in NEGATION_WORDS for t in doc)
        return _Facts(_numbers(text), ents, negated)

    def _conflicts(self, facts: _Facts, related: list[Evidence], source_lower: str, claim_text: str) -> list[str]:
        ev_text = " ".join(e.sentence for e in related)
        ev_facts = self._facts(ev_text)
        out = []
        # Numbers are compared with the single best-matching sentence: it states the same fact.
        best_nums = _numbers(related[0].sentence) or ev_facts.numbers
        missing_nums = {n for n in facts.numbers if not _has_number(best_nums, n)}
        if missing_nums and best_nums:
            out.append(f"number mismatch: claim says {_fmt(missing_nums)}, evidence says {_fmt(best_nums)}")
        ev_lower = ev_text.lower()
        claim_ent_text = " ".join(_norm(e) for e, _ in facts.entities)
        for ent, label in facts.entities:
            # grounded somewhere in the source -> not a conflict
            if _norm(ent) in source_lower or _norm(ent) in ev_lower or (_numbers(ent) and _numbers(ent) <= ev_facts.numbers):
                continue
            rivals = [t for t, l in ev_facts.entities if l == label and _norm(t) not in claim_ent_text]
            if rivals:
                # prefer the rival that shares a word with the claim's entity ("Atlantic Ocean" -> "Pacific Ocean")
                words = set(_norm(ent).split())
                rival = max(rivals, key=lambda r: len(words & set(_norm(r).split())))
                out.append(f"{label} mismatch: claim says '{ent}', evidence says '{rival}'")
        if self._polarity_flip(claim_text, related[0].sentence):
            out.append("polarity mismatch: negation differs from evidence")
        return out

    def _polarity_flip(self, claim: str, evidence: str) -> bool:
        """Negation attached to a verb that both texts share (e.g. 'was not compatible' vs 'was compatible')."""
        c, e = self.nlp(claim), self.nlp(evidence)

        def negated_heads(doc):
            return {t.head.lemma_.lower() for t in doc if t.dep_ == "neg"}

        c_lemmas, e_lemmas = {t.lemma_.lower() for t in c}, {t.lemma_.lower() for t in e}
        c_neg, e_neg = negated_heads(c), negated_heads(e)
        return bool((e_neg - c_neg) & c_lemmas) or bool((c_neg - e_neg) & e_lemmas)

    @staticmethod
    def _novel_facts(facts: _Facts, source_lower: str) -> list[str]:
        src_nums = _numbers(source_lower)
        novel = [_fmt({n}) for n in facts.numbers if not _has_number(src_nums, n)]
        for ent, _ in facts.entities:
            if _norm(ent) not in source_lower and not (_numbers(ent) and _numbers(ent) <= src_nums):
                novel.append(f"'{ent}'")
        return novel

    def _prepare_source(self, source: str):
        if source not in self._source_cache:
            sents = [s.text.strip() for s in self.nlp(source).sents if len(s.text.split()) >= 3]
            sents = sents or [source]
            self._source_cache[source] = (sents, self.embedder.encode(sents), source.lower())
        return self._source_cache[source]


class _NLI:
    """Optional cross-encoder NLI scorer (premise=evidence, hypothesis=claim)."""

    def __init__(self, model: str = "cross-encoder/nli-deberta-v3-xsmall"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)
        self.labels = ["contradiction", "entailment", "neutral"]

    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        logits = np.asarray(self.model.predict([(premise, hypothesis)]))[0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return dict(zip(self.labels, map(float, probs)))


# ---------------------------------------------------------------------- helpers
_NUM_RE = re.compile(r"(?<![\w.])\$?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)")


def _numbers(text: str) -> set[float]:
    nums = {float(m.group(1).replace(",", "")) for m in _NUM_RE.finditer(text)}
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in WORD_NUMBERS:
            nums.add(float(WORD_NUMBERS[word]))
    return nums


def _has_number(pool: set[float], n: float) -> bool:
    return any(abs(p - n) < 1e-9 for p in pool)


def _fmt(nums: set[float]) -> str:
    return ", ".join(f"{n:g}" for n in sorted(nums))


def _norm(ent: str) -> str:
    return re.sub(r"^(the|a|an)\s+", "", ent.lower()).strip(" .,'\"")
