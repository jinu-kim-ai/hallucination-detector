"""Command line interface.

  hallucheck detect --request examples/request.json --html report.html
  hallucheck detect --answer examples/answer.txt --source examples/source.txt
  hallucheck evaluate --data data --sweep
  hallucheck cases --dir examples/cases --verbose

(`python -m hallucination_detector ...` works the same way.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, TypeVar

from .cases import format_results, load_cases, run_case
from .config import Config
from .evaluate import format_report, load_dataset, predict_all, score, sweep_thresholds
from .pipeline import HallucinationDetector
from .report import to_html, to_json, to_text
from .sources import load_request, load_text_sources

T = TypeVar("T")

# Bad input files and bad settings: reported as one line, not a traceback. JSON and UTF-8 decode
# errors are ValueErrors; a missing file is an OSError.
INPUT_ERRORS = (OSError, ValueError)

# Config fields that the CLI exposes as --flags (dest name == Config field name).
CONFIG_FLAGS = ("embedder", "support_threshold", "related_threshold", "contradiction_threshold",
                "nli_threshold", "top_k", "nli_top_k", "nli_context_sentences", "min_claim_words")


# ---------------------------------------------------------------------- argument types
def _unit_float(text: str) -> float:
    """A threshold: a similarity or a probability, so it must lie in [0, 1]."""
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {text}")
    return value


def _int_at_least(minimum: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {text}")
        return value
    parse.__name__ = "int"                          # argparse names the type in "invalid int value"
    return parse


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    ap = argparse.ArgumentParser(prog="hallucheck", description=__doc__.split("\n")[0])
    ap.add_argument("--embedder", default=defaults.embedder, help="sentence-transformers model, or 'hashing'")
    ap.add_argument("--no-nli", action="store_true", help="skip the NLI cross-encoder (faster, smaller download)")
    t = ap.add_argument_group("thresholds (defaults from config.py)")
    t.add_argument("--support-threshold", type=_unit_float, default=defaults.support_threshold,
                   help="similarity that counts as support on its own (default %(default)s)")
    t.add_argument("--related-threshold", type=_unit_float, default=defaults.related_threshold,
                   help="below this similarity a sentence is not evidence at all (default %(default)s)")
    t.add_argument("--contradiction-threshold", type=_unit_float, default=defaults.contradiction_threshold,
                   help="only evidence this similar can contradict a claim (default %(default)s)")
    t.add_argument("--nli-threshold", type=_unit_float, default=defaults.nli_threshold,
                   help="NLI entailment / contradiction probability that is decisive (default %(default)s)")
    t.add_argument("--top-k", type=_int_at_least(1), default=defaults.top_k,
                   help="evidence sentences cited per claim (default %(default)s)")
    t.add_argument("--nli-top-k", type=_int_at_least(1), default=defaults.nli_top_k,
                   help="cited sentences the NLI model reads (default %(default)s)")
    t.add_argument("--nli-context-sentences", type=_int_at_least(0), default=defaults.nli_context_sentences,
                   help="neighbor sentences on each side added to the NLI premise (default %(default)s)")
    t.add_argument("--min-claim-words", type=_int_at_least(1), default=defaults.min_claim_words,
                   help="shorter sentences are not treated as claims (default %(default)s)")
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

    c = sub.add_parser("cases", help="run the sample cases (realistic answers with true labels) and compare")
    c.add_argument("--dir", default="examples/cases", help="folder of case files (default %(default)s)")
    c.add_argument("--verbose", action="store_true", help="show the detector's reasons for each disagreement")
    return ap


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(use_nli=not args.no_nli, **{name: getattr(args, name) for name in CONFIG_FLAGS})


# ---------------------------------------------------------------------- input
def _naming(path: str, load: Callable[[], T]) -> T:
    """Run load(), naming the file in decode errors (JSON and UTF-8 errors do not say which file)."""
    try:
        return load()
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"{path}: {e}") from e


def _load_detect_input(args: argparse.Namespace) -> tuple[str, list]:
    if args.request:
        try:
            response, sources = _naming(args.request, lambda: load_request(args.request))
        except TypeError as e:                      # e.g. the JSON is a list, or "sources" is a number
            raise ValueError(f"{args.request}: not a valid request ({e})") from e
        if not isinstance(response, str):
            raise ValueError(f"{args.request}: 'response' must be a string")
        return response, sources
    # utf-8-sig: Windows tools (PowerShell 5.1 Out-File, Notepad) often start the file with a BOM
    response = _naming(args.answer, lambda: Path(args.answer).read_text(encoding="utf-8-sig"))
    return response, _naming(", ".join(args.source), lambda: load_text_sources(args.source))


def _error_message(e: Exception) -> str:
    if isinstance(e, OSError) and e.filename and e.strerror:
        return f"{e.filename}: {e.strerror}"
    return str(e) or type(e).__name__


def _fail(ap: argparse.ArgumentParser, e: Exception) -> int:
    """One line on stderr and exit status 2, the way argparse reports bad arguments."""
    print(f"{ap.prog}: error: {_error_message(e)}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------- output
def _utf8_streams() -> None:
    """A redirected stdout on Windows defaults to cp1252, which cannot print most non-English names."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):           # e.g. a stream that was already read from
                pass


def _write(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def _run_detect(args: argparse.Namespace, detector: HallucinationDetector, inputs: tuple[str, list]) -> int:
    report = detector.detect(*inputs)
    # Write the files first, so a bad output path fails before the report has been printed.
    if args.json and args.json != "-":
        _write(args.json, to_json(report))
    if args.html:
        _write(args.html, to_html(report))
    if args.json == "-":
        print(to_json(report))
    else:
        print(to_text(report, color=sys.stdout.isatty()))
    if args.html:
        print(f"\nHTML report written to {args.html}", file=sys.stderr)
    return 0


def _run_evaluate(args: argparse.Namespace, detector: HallucinationDetector, inputs: tuple[dict, list]) -> int:
    sources, rows = inputs
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
        _write(args.out, json.dumps({"config": detector.config.public(), "metrics": metrics,
                                     "predictions": preds}, indent=2))
    return 0


def _run_cases(args: argparse.Namespace, detector: HallucinationDetector, cases: list) -> int:
    print(format_results([run_case(detector, case) for case in cases], verbose=args.verbose))
    return 0


def _load_input(args: argparse.Namespace):
    if args.cmd == "detect":
        return _load_detect_input(args)
    if args.cmd == "cases":
        return load_cases(args.dir)
    return load_dataset(args.data)


def main(argv=None) -> int:
    _utf8_streams()
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd == "detect" and not (args.request or args.answer):
        ap.error("detect needs --request FILE, or --answer FILE with --source FILE")
    # Input is read before the models load, so a typo in a path fails in a second, not after a minute.
    try:
        inputs = _load_input(args)
        detector = HallucinationDetector(config_from_args(args))
    except INPUT_ERRORS as e:
        return _fail(ap, e)
    run = {"detect": _run_detect, "evaluate": _run_evaluate, "cases": _run_cases}[args.cmd]
    try:
        return run(args, detector, inputs)
    except OSError as e:                            # an output file could not be written
        return _fail(ap, e)


if __name__ == "__main__":
    raise SystemExit(main())
