"""The sample cases in examples/cases: realistic factual and hallucinated answers with true labels."""
import json
from pathlib import Path

import pytest

from hallucination_detector.cases import format_results, load_case, load_cases, run_case
from hallucination_detector.cli import main
from hallucination_detector.schemas import HALLUCINATION_LABELS

CASES = Path(__file__).parent.parent / "examples" / "cases"


@pytest.fixture(scope="module")
def results(detector):
    return {r.case.path.stem: r for r in (run_case(detector, c) for c in load_cases(CASES))}


def test_every_domain_has_a_factual_and_a_hallucinated_case():
    cases = load_cases(CASES)
    domains = {c.domain for c in cases}
    assert len(domains) >= 4
    for d in domains:
        assert sorted(c.kind for c in cases if c.domain == d) == ["factual", "hallucinated"]


def test_hallucinated_cases_cover_every_kind_of_error():
    for case in load_cases(CASES):
        labels = {e["label"] for e in case.expected}
        if case.kind == "hallucinated":
            assert labels >= {"SUPPORTED", "CONTRADICTED", "FABRICATED", "UNSUPPORTED"}, case.path.name
        else:
            assert labels == {"SUPPORTED"}, case.path.name


def test_factual_answers_score_higher_than_hallucinated_ones(results):
    for name, r in results.items():
        if r.case.kind == "factual":
            twin = results[name.replace("_factual", "_hallucinated")]
            assert r.report.reliability_score > twin.report.reliability_score + 15, name


def test_agreement_with_the_ground_truth_does_not_regress(results):
    """Measured when the cases were added: 86% flag agreement, 78% exact labels over 72 claims."""
    checks = [c for r in results.values() for c in r.checks]
    assert len(checks) == 72
    assert sum(c.flag_agrees for c in checks) / len(checks) >= 0.85
    assert sum(c.exact for c in checks) / len(checks) >= 0.75


def test_every_planted_hallucination_type_is_caught_somewhere(results):
    caught = {c.expected for r in results.values() for c in r.checks
              if c.expected in HALLUCINATION_LABELS and c.predicted in HALLUCINATION_LABELS}
    assert caught == HALLUCINATION_LABELS


def test_format_results_marks_wrong_flags(results):
    text = format_results(list(results.values()), verbose=True)
    assert "all 72 claims" in text and "!!" in text and "best evidence similarity" in text


@pytest.mark.parametrize("change, message", [
    (lambda d: d.pop("sources"), "missing keys"),
    (lambda d: d["expected"][0].update(label="WRONG"), "unknown label"),
    (lambda d: d["expected"][0].update(claim="text that is not in the response"), "not verbatim"),
])
def test_invalid_case_files_are_rejected(tmp_path, change, message):
    data = json.loads(next(CASES.glob("*_factual.json")).read_text(encoding="utf-8"))
    change(data)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_case(path)


def test_cli_cases_command(capsys):
    assert main(["--no-nli", "cases", "--dir", str(CASES)]) == 0
    out = capsys.readouterr().out
    assert "medication_factual" in out and "all 72 claims" in out


def test_cli_cases_with_an_empty_folder_is_a_clean_error(tmp_path, capsys):
    assert main(["--no-nli", "cases", "--dir", str(tmp_path)]) == 2
    assert "no case files" in capsys.readouterr().err
