"""Command line interface.

  hallucheck detect --request examples/request.json --html report.html
  hallucheck detect --answer examples/answer.txt --source examples/source.txt
  hallucheck evaluate --data data --sweep

(`python -m hallucination_detector ...` works the same way.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .evaluate import format_report, load_dataset, predict_all, score, sweep_thresholds
from .pipeline import HallucinationDetector
from .report import to_html, to_json, to_text
from .sources import load_request, load_text_sources


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    ap = argparse.ArgumentParser(prog="hallucheck", description=__doc__.split("\n")[0])
    ap.add_argument("--embedder", default=defaults.embedder, help="sentence-transformers model, or 'hashing'")
    ap.add_argument("--no-nli", action="store_true", help="skip the NLI cross-encoder (faster, smaller download)")
    ap.add_argument("--support-threshold", type=float, default=defaults.support_threshold)
    ap.add_argument("--top-k", type=int, default=defaults.top_k, help="evidence sentences cited per claim")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="check one response against its source documents")
    d.add_argument("--request", help='JSON file: {"response": ..., "sources": [{"id", "title", "text"}]}')
    d.add_argument("--answer", help="text file with the response (use with --source)")
    d.add_argument("--source", action="append", default=[], help="text file with a source document (repeatable)")
    d.add_argument("--html", help="write a highlighted HTML report here")
    d.add_argument("--json", help="write the JSON result here ('-' prints it instead of the text report)")

    e = sub.add_parser("evaluate", help="precision / recall / F1 over a labeled dataset")
    e.add_argument("--data", default="data")
    e.add_argument("--sweep", action="store_true", help="sweep the support threshold")
    e.add_argument("--out", help="write metrics + predictions JSON here")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    detector = HallucinationDetector(Config(embedder=args.embedder, use_nli=not args.no_nli,
                                            support_threshold=args.support_threshold, top_k=args.top_k))
    if args.cmd == "detect":
        if args.request:
            response, sources = load_request(args.request)
        elif args.answer:
            response, sources = Path(args.answer).read_text(encoding="utf-8"), load_text_sources(args.source)
        else:
            ap.error("detect needs --request FILE, or --answer FILE with --source FILE")
        report = detector.detect(response, sources)
        if args.json == "-":
            print(to_json(report))
        else:
            print(to_text(report, color=sys.stdout.isatty()))
            if args.json:
                Path(args.json).write_text(to_json(report), encoding="utf-8")
        if args.html:
            Path(args.html).write_text(to_html(report), encoding="utf-8")
            print(f"\nHTML report written to {args.html}", file=sys.stderr)
        return 0

    sources, rows = load_dataset(args.data)
    preds = predict_all(detector, sources, rows)
    metrics = score(preds)
    print(format_report(metrics))
    if args.sweep:
        print("\nSupport-threshold sweep (top 5 by F1):")
        for r in sweep_thresholds(detector, sources, rows)[:5]:
            print(f"  t={r['support_threshold']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")
    errors = [p for p in preds if p["label"] != p["pred"]]
    if errors:
        print(f"\nMisclassified ({len(errors)}):")
        for p in errors:
            print(f"  [{p['id']}] gold={p['label']:<12} pred={p['pred']:<12} {p['claim']}")
            print(f"        {' | '.join(p['reasons'])}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"config": detector.config.public(), "metrics": metrics,
                                              "predictions": preds}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
