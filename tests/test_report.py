import json

import pytest

from hallucination_detector.report import to_html, to_json, to_text

RESPONSE = ("Apollo 11 launched from Kennedy Space Center on July 16, 1969. "
            "The astronauts collected about 45 kilograms of lunar samples <b>really</b>.")


@pytest.fixture(scope="module")
def report(detector, apollo):
    return detector.detect(RESPONSE, [{"id": "nasa", "title": "NASA summary", "text": apollo}])


def test_json_follows_the_documented_schema(report):
    data = json.loads(to_json(report))
    assert set(data) == {"reliability_score", "hallucination_rate", "counts", "warnings", "claims", "config"}
    claim = data["claims"][1]
    assert set(claim) >= {"id", "text", "response_span", "label", "confidence", "reasons", "citations"}
    a, b = claim["response_span"]
    assert "45 kilograms" in RESPONSE[a:b]
    citation = claim["citations"][0]
    assert set(citation) == {"source_id", "sentence_index", "source_span", "quote", "similarity", "nli"}
    assert citation["source_id"] == "nasa" and set(citation["nli"]) == {"entailment", "contradiction", "neutral"}
    assert 0 <= data["reliability_score"] <= 100 and sum(data["counts"].values()) == 2


def test_text_report_lists_every_claim_with_evidence(report):
    text = to_text(report, color=False)
    assert "CONTRADICTED" in text and "SUPPORTED" in text
    assert "[nasa #" in text and "number mismatch" in text and "Reliability score" in text
    assert "\033[" not in text


def test_html_highlights_response_and_sources_and_escapes(report):
    page = to_html(report)
    assert page.count("<mark") >= 3                     # two claims in the response + cited source sentences
    assert "NASA summary" in page and "cited by claim" in page
    assert "&lt;b&gt;really&lt;/b&gt;" in page and "<b>really</b>" not in page


def test_reports_handle_a_response_with_nothing_to_check(detector, apollo):
    empty = detector.detect("I hope this helps!", apollo)
    assert "n/a" in to_text(empty, color=False) and "n/a" in to_html(empty)
    assert json.loads(to_json(empty))["claims"] == []
