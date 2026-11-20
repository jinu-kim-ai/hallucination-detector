"""Sample cases: realistic AI answers with ground-truth labels, for demos and regression checks.

A case file (examples/cases/*.json) holds one question, the AI's response, the source documents,
and the true label of every factual sentence or clause in the response:

  {"name", "domain", "question", "description", "response",
   "sources": [{"id", "title", "text"}],
   "expected": [{"claim": "<verbatim text from the response>", "label": "...", "why": "..."}]}

Each expected claim is matched to the detector claim whose character span overlaps it most.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import HALLUCINATION_LABELS, LABELS, ClaimVerdict, Report
from .sources import as_documents

REQUIRED_KEYS = {"name", "domain", "question", "description", "response", "sources", "expected"}


@dataclass
class Case:
    path: Path
    name: str
    domain: str
    question: str
    description: str
    response: str
    sources: list
    expected: list[dict]

    @property
    def kind(self) -> str:
        return "hallucinated" if any(e["label"] in HALLUCINATION_LABELS for e in self.expected) else "factual"


@dataclass
class ClaimCheck:
    claim: str
    expected: str
    predicted: str | None           # None: the detector produced no claim covering this text
    confidence: float | None
    reasons: list[str] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        return self.predicted == self.expected

    @property
    def flag_agrees(self) -> bool:
        """Both call it a hallucination, or both call it supported."""
        return self.predicted is not None and \
            (self.expected in HALLUCINATION_LABELS) == (self.predicted in HALLUCINATION_LABELS)


@dataclass
class CaseResult:
    case: Case
    report: Report
    checks: list[ClaimCheck]

    @property
    def exact_rate(self) -> float:
        return sum(c.exact for c in self.checks) / len(self.checks) if self.checks else 0.0

    @property
    def flag_rate(self) -> float:
        return sum(c.flag_agrees for c in self.checks) / len(self.checks) if self.checks else 0.0


def load_case(path: str | Path) -> Case:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    for e in data["expected"]:
        if e.get("label") not in LABELS:
            raise ValueError(f"{path}: unknown label {e.get('label')!r} for claim {e.get('claim')!r}")
        if e.get("claim") not in data["response"]:
            raise ValueError(f"{path}: expected claim is not verbatim text from the response: {e.get('claim')!r}")
    as_documents(data["sources"])                  # validates ids
    return Case(path, **{k: data[k] for k in REQUIRED_KEYS})


def load_cases(directory: str | Path) -> list[Case]:
    paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no case files (*.json) in {directory}")
    return [load_case(p) for p in paths]


def _best_verdict(response: str, claim_text: str, verdicts: list[ClaimVerdict]) -> ClaimVerdict | None:
    start = response.find(claim_text)
    end = start + len(claim_text)
    best, best_overlap = None, 0
    for v in verdicts:
        overlap = min(end, v.claim.end) - max(start, v.claim.start)
        if overlap > best_overlap:
            best, best_overlap = v, overlap
    return best


def run_case(detector, case: Case) -> CaseResult:
    report = detector.detect(case.response, case.sources)
    checks = []
    for e in case.expected:
        v = _best_verdict(case.response, e["claim"], report.verdicts)
        checks.append(ClaimCheck(e["claim"], e["label"], v.label if v else None,
                                 v.confidence if v else None, v.reasons if v else []))
    return CaseResult(case, report, checks)


def format_results(results: list[CaseResult], verbose: bool = False) -> str:
    lines = [f"{'case':<28} {'kind':<13} {'score':>7}  {'flag agree':>10}  {'exact label':>11}", "-" * 76]
    for r in results:
        score = r.report.reliability_score
        lines.append(f"{r.case.path.stem:<28} {r.case.kind:<13} {('n/a' if score is None else f'{score:.1f}'):>7}  "
                     f"{r.flag_rate:>10.0%}  {r.exact_rate:>11.0%}")
    checks = [c for r in results for c in r.checks]
    if checks:
        flag = sum(c.flag_agrees for c in checks) / len(checks)
        exact = sum(c.exact for c in checks) / len(checks)
        lines += ["-" * 76, f"{'all ' + str(len(checks)) + ' claims':<50} {flag:>10.0%}  {exact:>11.0%}"]
    misses = [(r, c) for r in results for c in r.checks if not c.exact]
    if misses:
        lines += ["", f"Disagreements with the ground truth ({len(misses)}):"]
        for r, c in misses:
            marker = "  " if c.flag_agrees else "!!"
            lines.append(f" {marker} {r.case.path.stem}: expected {c.expected}, got {c.predicted}  \"{c.claim}\"")
            if verbose and c.reasons:
                lines.append(f"      {' | '.join(c.reasons)}")
        lines.append("    (!! = the detector got hallucination-vs-supported wrong, not just the label)")
    return "\n".join(lines)
