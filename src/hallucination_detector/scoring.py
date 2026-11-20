"""Turn evidence into a label and a confidence. Pure functions: no models, easy to test.

Rules, first match wins (s = best similarity, E / C = decisive NLI entailment / contradiction):

  1. CONTRADICTED  s >= contradiction_threshold and (fact conflicts or C >= nli_threshold)
  2. FABRICATED    the claim has numbers or names that appear in no source
  3. SUPPORTED     s >= support_threshold, or (E >= nli_threshold and s >= related_threshold)
  4. UNSUPPORTED   otherwise

Confidences are heuristics, not calibrated probabilities.
"""
from __future__ import annotations

from .config import Config
from .schemas import CONTRADICTED, FABRICATED, SUPPORTED, UNSUPPORTED, Citation


def decisive_nli(citations: list[Citation], cfg: Config) -> dict[str, float] | None:
    """The NLI result that counts.

    Contradiction only counts from the single best-matching sentence: that sentence states the same
    fact, while a neighbor about a *different* fact ("the 1903 Physics prize" vs a claim about the 1911
    Chemistry prize) would contradict a true claim. Entailment may come from any related sentence.
    """
    scored = [c for c in citations if c.nli]
    if not scored:
        return None
    best = citations[0]
    if best.nli and best.nli["contradiction"] >= cfg.nli_threshold and best.similarity >= cfg.contradiction_threshold:
        return best.nli
    for c in scored:                                    # citations are ordered best-first
        if c.nli["entailment"] >= cfg.nli_threshold and c.similarity >= cfg.related_threshold:
            return c.nli
    return best.nli or scored[0].nli


def decide(similarity: float, conflicts: list[str], novel: list[str], nli: dict[str, float] | None,
           cfg: Config) -> tuple[str, float, list[str]]:
    """Return (label, confidence, reasons)."""
    s, r = cfg.support_threshold, cfg.related_threshold
    reasons = [f"best evidence similarity {similarity:.2f}"]

    # Rule 1 needs close evidence for fact conflicts too: weakly related text is about another fact.
    close = similarity >= cfg.contradiction_threshold
    if nli and close and nli["contradiction"] >= cfg.nli_threshold:
        conflicts = conflicts + [f"NLI contradiction p={nli['contradiction']:.2f}"]
    if conflicts and close:
        conf = min(1.0, 0.55 + 0.15 * len(conflicts) + 0.3 * (similarity - r) / (1 - r))
        return CONTRADICTED, conf, reasons + conflicts
    if novel:
        conf = min(1.0, 0.55 + 0.1 * len(novel) + 0.35 * max(0.0, 1 - similarity))
        return FABRICATED, conf, reasons + [f"not found in source: {', '.join(novel)}"]
    if similarity >= s or (nli and nli["entailment"] >= cfg.nli_threshold and similarity >= r):
        conf = 0.5 + 0.5 * min(1.0, (similarity - r) / (1 - r))
        if nli:
            conf = max(conf, nli["entailment"])
            if similarity < s:
                reasons.append(f"NLI entailment p={nli['entailment']:.2f}")
        return SUPPORTED, conf, reasons
    conf = 0.5 + 0.5 * (s - similarity) / s
    return UNSUPPORTED, max(0.0, min(1.0, conf)), reasons + [f"no source sentence reaches support threshold {s:.2f}"]
