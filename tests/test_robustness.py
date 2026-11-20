"""Wrong types, empty input and very long input: every case works or fails with one clear error."""
import json
import re
import time

import pytest

from hallucination_detector import SUPPORTED, UNSUPPORTED, ClaimExtractor, HallucinationDetector
from hallucination_detector.cli import main
from hallucination_detector.evaluate import load_dataset
from hallucination_detector.sources import as_documents


def run_on(clauses: int) -> str:
    """One sentence of many joined clauses, with no punctuation to split at."""
    return "Apollo " + " and ".join(f"mission {i} launched in {1960 + i % 40}" for i in range(clauses)) + "."


# ---------------------------------------------------------------- wrong types
@pytest.mark.parametrize("response", [None, 123, b"Apollo 11 launched in 1969.", ["Apollo 11 launched in 1969."]])
def test_response_of_the_wrong_type_is_a_clear_error(detector_no_nli, apollo, nlp, response):
    with pytest.raises(TypeError, match="response must be a string"):
        detector_no_nli.detect(response, apollo)
    with pytest.raises(TypeError, match="response must be a string"):
        ClaimExtractor(nlp).extract(response)


@pytest.mark.parametrize("sources", [5, 3.2, b"raw bytes", object()])
def test_sources_of_the_wrong_type_are_a_clear_error(detector_no_nli, sources):
    message = "sources must be a string, a dict, a SourceDocument, or a list of them"
    with pytest.raises(TypeError, match=message):
        as_documents(sources)
    with pytest.raises(TypeError, match=message):
        detector_no_nli.detect("Apollo 11 launched in 1969.", sources)


def test_claim_of_the_wrong_type_is_a_clear_error(detector_no_nli, apollo):
    with pytest.raises(TypeError, match="claim must be a string, got NoneType"):
        detector_no_nli.check_claim(None, apollo)


# ---------------------------------------------------------------- empty and degenerate input
@pytest.mark.parametrize("response", ["... !!! ???", "\U0001F680\U0001F680 \U0001F315", "Yes", "\n\n\t  "])
def test_response_with_nothing_to_check(detector_no_nli, apollo, response):
    report = detector_no_nli.detect(response, apollo)
    assert report.verdicts == [] and report.reliability_score is None
    assert report.warnings == ["no checkable claims found in the response"]


@pytest.mark.parametrize("sources", ["\n\n\n", [""] * 100, [None], [{"id": "a"}]])
def test_sources_with_no_text(detector_no_nli, sources):
    report = detector_no_nli.detect("Apollo 11 launched in 1969.", sources)
    assert [v.label for v in report.verdicts] == [UNSUPPORTED] and report.reliability_score == 0.0
    assert report.warnings == ["no source text provided; every claim is unsupported"]


def test_empty_and_blank_claims_are_unsupported(detector_no_nli, apollo):
    assert detector_no_nli.check_claim("", apollo).label == UNSUPPORTED
    assert detector_no_nli.check_claim("   ", apollo).label == UNSUPPORTED


def test_control_characters_do_not_break_checking(detector_no_nli, apollo):
    report = detector_no_nli.detect("Apollo 11 launched\x00 from Kennedy Space Center on July 16, 1969.\x00", apollo)
    assert [v.label for v in report.verdicts] == [SUPPORTED]


def test_top_k_larger_than_the_number_of_sentences(nlp, apollo):
    detector = HallucinationDetector(nlp=nlp, use_nli=False, embedder="hashing", top_k=50)
    assert len(detector.detect("Apollo 11 launched in 1969.", apollo).verdicts) == 1


# ---------------------------------------------------------------- very long input
def test_run_on_sentence_is_checked_in_bounded_parts(nlp):
    extractor = ClaimExtractor(nlp, max_sentence_words=40)
    response = "Apollo 11 launched in 1969. " + run_on(60)
    claims = extractor.extract(response)
    assert claims[0].text == "Apollo 11 launched in 1969."
    parts = claims[1:]
    assert len(parts) > 1 and all(len(c.text.split()) <= 40 for c in parts)
    for c in parts:                                         # spans index the original response
        written = re.sub(r"\s+", " ", response[c.start:c.end]).strip(" ,;").rstrip(".")
        assert written.lower() == c.text.rstrip(".").lower()        # a part is shown with a capital first letter
    assert parts[0].start >= len("Apollo 11 launched in 1969.") and parts[-1].end <= len(response)


def test_sentence_at_the_limit_is_still_split_by_grammar(nlp):
    claims = ClaimExtractor(nlp, max_sentence_words=9).extract("Marie Curie discovered polonium and won two Nobel Prizes.")
    assert [c.text for c in claims] == ["Marie Curie discovered polonium.", "Marie Curie won two Nobel Prizes."]


def test_run_on_response_is_reported_and_takes_bounded_time(nlp, apollo):
    """A single sentence of 600 joined clauses took minutes before claims were bounded."""
    detector = HallucinationDetector(nlp=nlp, use_nli=False, embedder="hashing")
    start = time.perf_counter()
    report = detector.detect(run_on(600), apollo)
    assert time.perf_counter() - start < 30
    assert any("were checked in parts" in w for w in report.warnings)
    limit = detector.config.max_sentence_words
    assert len(report.verdicts) > 10 and all(len(v.claim.text.split()) <= limit for v in report.verdicts)


def test_response_over_the_parser_limit_is_parsed_in_chunks(nlp, monkeypatch):
    monkeypatch.setattr(nlp, "max_length", 200)             # far below the response's length
    sentences = [f"Mission {i} launched in {1960 + i}." for i in range(30)]
    response = " ".join(sentences)
    claims = ClaimExtractor(nlp).extract(response)
    assert [c.text for c in claims] == sentences
    assert all(response[c.start:c.end] == c.text for c in claims)


def test_many_sentences_scale_linearly(nlp, apollo):
    detector = HallucinationDetector(nlp=nlp, use_nli=False, embedder="hashing")
    response = " ".join(f"Mission {i} launched in {1960 + i % 40}." for i in range(400))
    start = time.perf_counter()
    report = detector.detect(response, apollo)
    assert len(report.verdicts) == 400 and time.perf_counter() - start < 30


def test_repeated_source_warning_is_shown_once(detector_no_nli):
    long_sentence = "The report lists " + ", ".join(f"item number {i} of the inventory" for i in range(120)) + "."
    report = detector_no_nli.detect("The report lists item number 7 of the inventory.",
                                    long_sentence + " " + long_sentence)
    assert len([w for w in report.warnings if "split into parts" in w]) == 1


# ---------------------------------------------------------------- evaluation datasets
GOOD_ROW = {"id": "1", "domain": "d", "source_id": "a", "claim": "Apollo 11 launched in 1969.", "label": SUPPORTED}


def dataset(tmp_path, sources, lines):
    (tmp_path / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
    (tmp_path / "labeled_claims.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return tmp_path


def without(key: str) -> dict:
    return {k: v for k, v in GOOD_ROW.items() if k != key}


@pytest.mark.parametrize("sources, row, message", [
    ({"a": "text"}, {**GOOD_ROW, "source_id": "zzz"}, r"labeled_claims.jsonl:1: unknown source id\(s\) \['zzz'\]"),
    ({"a": "text"}, without("label"), r"missing field\(s\) \['label'\]"),
    ({"a": "text"}, {**GOOD_ROW, "label": "TRUE"}, "unknown label 'TRUE'"),
    ({"a": "text"}, {**GOOD_ROW, "claim": 7}, "'claim' must be a string"),
    ({"a": "text"}, without("source_id"), "needs 'source_id'"),
    ({"a": 5}, GOOD_ROW, "expected an object mapping source ids to document text"),
])
def test_invalid_dataset_is_a_value_error_naming_the_line(tmp_path, sources, row, message):
    with pytest.raises(ValueError, match=message):
        load_dataset(dataset(tmp_path, sources, [json.dumps(row)]))


def test_dataset_problems_in_the_files_themselves(tmp_path):
    with pytest.raises(ValueError, match="no labeled claims"):
        load_dataset(dataset(tmp_path, {"a": "text"}, [""]))
    with pytest.raises(ValueError, match=r"labeled_claims.jsonl:2: not valid JSON"):
        load_dataset(dataset(tmp_path, {"a": "text"}, [json.dumps(GOOD_ROW), "{broken"]))
    (tmp_path / "sources.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="sources.json: not valid JSON"):
        load_dataset(tmp_path)


def test_valid_dataset_with_source_ids_loads(tmp_path):
    row = {**without("source_id"), "source_ids": ["a", "b"]}
    sources, rows = load_dataset(dataset(tmp_path, {"a": "x", "b": "y"}, [json.dumps(row), ""]))
    assert set(sources) == {"a", "b"} and rows == [row]


def test_cli_evaluate_with_a_bad_dataset_is_a_one_line_error(tmp_path, capsys):
    dataset(tmp_path, {"a": "text"}, [json.dumps({**GOOD_ROW, "source_id": "zzz"})])
    assert main(["--no-nli", "--embedder", "hashing", "evaluate", "--data", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("hallucheck: error: ") and "unknown source id" in err and "Traceback" not in err
