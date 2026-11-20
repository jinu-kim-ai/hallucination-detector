import json
from pathlib import Path

import pytest

from hallucination_detector import CONTRADICTED, FABRICATED, SUPPORTED, UNSUPPORTED
from hallucination_detector.cli import main
from hallucination_detector.evaluate import BinaryMetrics, load_dataset, predict_all, score, sweep_thresholds

DATA = Path(__file__).parent.parent / "data"


def test_binary_metrics():
    m = BinaryMetrics(tp=8, fp=2, fn=2, tn=8)
    assert m.precision == pytest.approx(0.8) and m.recall == pytest.approx(0.8)
    assert m.f1 == pytest.approx(0.8) and m.accuracy == pytest.approx(0.8)
    assert BinaryMetrics().f1 == 0.0                      # no division by zero on empty input


def test_score_shapes():
    preds = [
        {"domain": "a", "label": SUPPORTED, "pred": SUPPORTED},
        {"domain": "a", "label": FABRICATED, "pred": UNSUPPORTED},       # still caught as a hallucination
        {"domain": "b", "label": CONTRADICTED, "pred": SUPPORTED},       # missed
    ]
    m = score(preds)
    assert m["binary_hallucination_detection"]["recall"] == pytest.approx(0.5)
    assert m["four_way_accuracy"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["confusion"][FABRICATED][UNSUPPORTED] == 1
    assert set(m["by_domain"]) == {"a", "b"}


def test_rows_may_cite_several_sources(detector_no_nli):
    sources = {"a": "Apollo 11 launched on July 16, 1969.", "b": "The crew splashed down in the Pacific Ocean."}
    rows = [{"id": "1", "domain": "d", "source_ids": ["a", "b"], "label": SUPPORTED,
             "claim": "The crew splashed down in the Pacific Ocean."},
            {"id": "2", "domain": "d", "source_id": "a", "label": UNSUPPORTED,
             "claim": "The crew splashed down in the Pacific Ocean."}]
    preds = predict_all(detector_no_nli, sources, rows)
    assert preds[0]["pred"] == SUPPORTED and preds[1]["pred"] != SUPPORTED


def test_dev_set_results_do_not_regress(detector, detector_no_nli):
    """The published numbers on data/ (the development set the thresholds were tuned on)."""
    sources, rows = load_dataset(DATA)
    assert len(rows) == 50
    without = score(predict_all(detector_no_nli, sources, rows))["binary_hallucination_detection"]
    with_nli = score(predict_all(detector, sources, rows))["binary_hallucination_detection"]
    assert without["f1"] == pytest.approx(0.931, abs=1e-3)
    assert with_nli["f1"] == pytest.approx(1.0)


def test_threshold_sweep_restores_the_config(detector_no_nli):
    sources, rows = load_dataset(DATA)
    before = detector_no_nli.config
    results = sweep_thresholds(detector_no_nli, sources, rows[:10], values=[0.5, 0.65, 0.8])
    assert len(results) == 3 and results[0]["f1"] >= results[-1]["f1"]
    assert detector_no_nli.config == before


def test_cli_detect_prints_json(tmp_path, capsys):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"response": "Apollo 11 launched on July 16, 1969.",
                                   "sources": [{"id": "a", "text": "Apollo 11 launched on July 16, 1969."}]}),
                       encoding="utf-8")
    html_path = tmp_path / "report.html"
    assert main(["--no-nli", "detect", "--request", str(request), "--json", "-", "--html", str(html_path)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["claims"][0]["label"] == SUPPORTED and data["config"]["use_nli"] is False
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_cli_detect_from_text_files(tmp_path, capsys):
    (tmp_path / "answer.txt").write_text("The crew collected 45 kilograms of samples.", encoding="utf-8")
    (tmp_path / "source.txt").write_text("The crew collected 21.5 kilograms of samples.", encoding="utf-8")
    assert main(["--no-nli", "detect", "--answer", str(tmp_path / "answer.txt"),
                 "--source", str(tmp_path / "source.txt")]) == 0
    out = capsys.readouterr().out
    assert "CONTRADICTED" in out and "[source.txt #0]" in out
