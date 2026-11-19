"""Command line interface.

  python -m hallucination_detector detect --source examples/source.txt --answer examples/answer.txt --html report.html
  python -m hallucination_detector evaluate --data data --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detector import HallucinationDetector
from .evaluate import format_report, load_dataset, predict_all, score, sweep_thresholds
from .report import to_html, to_text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hallucination_detector")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2", help="sentence-transformers model, or 'hashing'")
    ap.add_argument("--support-threshold", type=float, default=0.65)
    ap.add_argument("--nli", action="store_true", help="also use a cross-encoder NLI model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="check one answer against one source document")
    d.add_argument("--source", required=True)
    d.add_argument("--answer", required=True)
    d.add_argument("--html", help="write a highlighted HTML report here")
    d.add_argument("--json", help="write claim results as JSON here")

    e = sub.add_parser("evaluate", help="precision/recall over a labeled dataset")
    e.add_argument("--data", default="data")
    e.add_argument("--sweep", action="store_true", help="sweep the support threshold")
    e.add_argument("--out", help="write metrics + predictions JSON here")

    args = ap.parse_args(argv)
    det = HallucinationDetector(embedder=args.embedder, support_threshold=args.support_threshold, use_nli=args.nli)

    if args.cmd == "detect":
        source = Path(args.source).read_text(encoding="utf-8")
        answer = Path(args.answer).read_text(encoding="utf-8")
        report = det.detect(answer, source)
        print(to_text(report, color=sys.stdout.isatty()))
        if args.html:
            Path(args.html).write_text(to_html(report), encoding="utf-8")
            print(f"\nHTML report written to {args.html}")
        if args.json:
            Path(args.json).write_text(json.dumps([r.to_dict() for r in report.results], indent=2), encoding="utf-8")
        return 0

    sources, rows = load_dataset(args.data)
    preds = predict_all(det, sources, rows)
    metrics = score(preds)
    print(format_report(metrics))
    if args.sweep:
        print("\nSupport-threshold sweep (top 5 by F1):")
        for r in sweep_thresholds(det, sources, rows)[:5]:
            print(f"  t={r['support_threshold']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")
    errors = [p for p in preds if p["label"] != p["pred"]]
    if errors:
        print(f"\nMisclassified ({len(errors)}):")
        for p in errors:
            print(f"  [{p['id']}] gold={p['label']:<12} pred={p['pred']:<12} {p['claim']}")
            print(f"        {' | '.join(p['reasons'])}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"metrics": metrics, "predictions": preds}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
