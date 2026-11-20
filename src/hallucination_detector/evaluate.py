"""Evaluation harness: precision / recall / F1 over a labeled claim dataset.

Dataset layout (a directory):
  sources.json          {"<source_id>": "<source document text>", ...}
  labeled_claims.jsonl  one object per line:
                        {"id", "domain", "source_id" | "source_ids": [...], "claim", "label"}
                        label is SUPPORTED / CONTRADICTED / FABRICATED / UNSUPPORTED
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from .pipeline import HallucinationDetector
from .schemas import HALLUCINATION_LABELS, LABELS, SourceDocument


@dataclass
class BinaryMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, gold_bad: bool, pred_bad: bool) -> None:
        if gold_bad and pred_bad:
            self.tp += 1
        elif pred_bad:
            self.fp += 1
        elif gold_bad:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    def as_dict(self) -> dict:
        return {"n": self.n, "precision": round(self.precision, 3), "recall": round(self.recall, 3),
                "f1": round(self.f1, 3), "accuracy": round(self.accuracy, 3)}


def load_dataset(data_dir: str | Path) -> tuple[dict[str, str], list[dict]]:
    """Read and validate a dataset directory. Every problem is a ValueError that names the file and line."""
    data_dir = Path(data_dir)
    sources_path, claims_path = data_dir / "sources.json", data_dir / "labeled_claims.jsonl"
    try:
        sources = json.loads(sources_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sources_path}: not valid JSON ({exc})") from exc
    if not isinstance(sources, dict) or not all(isinstance(v, str) for v in sources.values()):
        raise ValueError(f"{sources_path}: expected an object mapping source ids to document text")

    rows = []
    for number, line in enumerate(claims_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{claims_path}:{number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}: not valid JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{where}: expected a JSON object")
        missing = [k for k in ("id", "domain", "claim", "label") if k not in row]
        if missing:
            raise ValueError(f"{where}: missing field(s) {missing}")
        if not isinstance(row["claim"], str):
            raise ValueError(f"{where}: 'claim' must be a string")
        if row["label"] not in LABELS:
            raise ValueError(f"{where}: unknown label {row['label']!r}; use one of {LABELS}")
        ids = row.get("source_ids") or ([row["source_id"]] if "source_id" in row else [])
        if not ids or not isinstance(ids, list):
            raise ValueError(f"{where}: needs 'source_id' or a non-empty 'source_ids' list")
        unknown = [i for i in ids if i not in sources]
        if unknown:
            raise ValueError(f"{where}: unknown source id(s) {unknown}; {sources_path.name} has {sorted(sources)}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{claims_path}: no labeled claims")
    return sources, rows


def predict_all(detector: HallucinationDetector, sources: dict[str, str], rows: list[dict]) -> list[dict]:
    preds = []
    for row in rows:
        ids = row.get("source_ids") or [row["source_id"]]
        verdict = detector.check_claim(row["claim"], [SourceDocument(i, sources[i]) for i in ids])
        preds.append({**row, "pred": verdict.label, "confidence": verdict.confidence,
                      "similarity": verdict.best_similarity,
                      "reasons": verdict.reasons})
    return preds


def score(preds: list[dict]) -> dict:
    overall = BinaryMetrics()
    by_domain: dict[str, BinaryMetrics] = defaultdict(BinaryMetrics)
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for p in preds:
        gold_bad, pred_bad = p["label"] in HALLUCINATION_LABELS, p["pred"] in HALLUCINATION_LABELS
        overall.add(gold_bad, pred_bad)
        by_domain[p["domain"]].add(gold_bad, pred_bad)
        confusion[p["label"]][p["pred"]] += 1

    per_label = {}
    for lab in LABELS:
        tp = confusion[lab][lab]
        pred_n = sum(confusion[g][lab] for g in LABELS)
        gold_n = sum(confusion[lab].values())
        per_label[lab] = {"precision": round(tp / pred_n, 3) if pred_n else 0.0,
                          "recall": round(tp / gold_n, 3) if gold_n else 0.0, "support": gold_n}
    exact = sum(confusion[lab][lab] for lab in LABELS) / len(preds) if preds else 0.0
    return {
        "binary_hallucination_detection": overall.as_dict(),
        "by_domain": {d: m.as_dict() for d, m in sorted(by_domain.items())},
        "four_way_accuracy": round(exact, 3),
        "per_label": per_label,
        "confusion": confusion,
    }


def sweep_thresholds(detector: HallucinationDetector, sources, rows, values=None) -> list[dict]:
    """Re-score at several support thresholds; returns rows sorted by binary F1.

    Values below the related threshold are skipped: Config rejects support < related.
    """
    original = detector.config
    values = [v for v in (values or [round(0.45 + 0.025 * i, 3) for i in range(13)])
              if original.related_threshold <= v <= 1.0]
    out = []
    try:
        for v in values:
            detector.config = replace(original, support_threshold=v)
            m = score(predict_all(detector, sources, rows))["binary_hallucination_detection"]
            out.append({"support_threshold": v, **m})
    finally:
        detector.config = original
    return sorted(out, key=lambda r: -r["f1"])


def format_report(metrics: dict) -> str:
    b = metrics["binary_hallucination_detection"]
    lines = [
        "Binary hallucination detection (positive = CONTRADICTED/FABRICATED/UNSUPPORTED)",
        f"  n={b['n']}  precision={b['precision']:.3f}  recall={b['recall']:.3f}  F1={b['f1']:.3f}  acc={b['accuracy']:.3f}",
        "",
        "By domain:",
    ]
    for d, m in metrics["by_domain"].items():
        lines.append(f"  {d:<12} n={m['n']:<3} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    lines += ["", f"4-way label accuracy: {metrics['four_way_accuracy']:.3f}", "", "Per label:"]
    for lab, m in metrics["per_label"].items():
        lines.append(f"  {lab:<13} P={m['precision']:.3f}  R={m['recall']:.3f}  support={m['support']}")
    lines += ["", "Confusion (rows = gold, cols = predicted):",
              "  " + " " * 13 + "".join(f"{lab[:5]:>7}" for lab in LABELS)]
    for g in LABELS:
        lines.append(f"  {g:<13}" + "".join(f"{metrics['confusion'][g][p]:>7}" for p in LABELS))
    return "\n".join(lines)
