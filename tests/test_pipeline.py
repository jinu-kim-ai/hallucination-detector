"""End-to-end behaviour, including the edge cases listed in PLAN.md section 4."""
import pytest

from hallucination_detector import CONTRADICTED, FABRICATED, SUPPORTED, UNSUPPORTED

MISSION = {"id": "mission", "title": "Mission summary",
           "text": "Apollo 11 launched from Kennedy Space Center on July 16, 1969. "
                   "Armstrong became the first person to walk on the Moon, followed by Aldrin about 19 minutes later."}
RETURN = {"id": "return", "title": "Return to Earth",
          "text": "The astronauts collected about 21.5 kilograms of lunar samples. "
                  "The mission returned to Earth on July 24, 1969, splashing down in the Pacific Ocean."}


# ---------------------------------------------------------------- the four labels
def test_supported_paraphrase(detector, apollo):
    v = detector.check_claim("The Eagle lunar module touched down in the Sea of Tranquility on July 20, 1969.", apollo)
    assert v.label == SUPPORTED


def test_number_contradiction(detector, apollo):
    v = detector.check_claim("The astronauts collected about 45 kilograms of lunar samples.", apollo)
    assert v.label == CONTRADICTED
    assert any("number mismatch" in r for r in v.reasons)


def test_entity_contradiction(detector, apollo):
    v = detector.check_claim("The mission splashed down in the Atlantic Ocean on July 24, 1969.", apollo)
    assert v.label == CONTRADICTED


def test_fabricated_details(detector, apollo):
    v = detector.check_claim("The landing site was picked by 300 geologists at Stanford University.", apollo)
    assert v.label == FABRICATED


def test_unsupported_opinion(detector, apollo):
    v = detector.check_claim("The astronauts felt very nervous during the descent.", apollo)
    assert v.label == UNSUPPORTED


# ---------------------------------------------------------------- negation and relations
def test_negated_claim_against_positive_source(detector):
    source = "Python 3.0 was released in 2008 and was not backward compatible with Python 2."
    assert detector.check_claim("Python 3.0 was fully backward compatible with Python 2.", source).label == CONTRADICTED
    assert detector.check_claim("Python 3.0 was not backward compatible with Python 2.", source).label == SUPPORTED


def test_negation_on_another_verb_does_not_contradict(detector):
    source = "People with type 1 diabetes do not produce enough insulin and require insulin therapy."
    assert detector.check_claim("People with type 1 diabetes require insulin therapy.", source).label == SUPPORTED


def test_right_entities_wrong_relation_needs_nli(detector, detector_no_nli, apollo):
    claim = "Aldrin was the first person to walk on the Moon."
    assert detector.check_claim(claim, apollo).label == CONTRADICTED
    # documented limit: without NLI every name and number checks out, so the error is missed
    assert detector_no_nli.check_claim(claim, apollo).label == SUPPORTED


# ---------------------------------------------------------------- multiple documents and citations
def test_citations_point_to_the_right_document_and_span(detector):
    report = detector.detect("Apollo 11 launched on July 16, 1969. The crew collected about 21.5 kilograms of samples.",
                             [MISSION, RETURN])
    first, second = report.verdicts
    assert first.citations[0].source_id == "mission" and second.citations[0].source_id == "return"
    for verdict, doc in ((first, MISSION), (second, RETURN)):
        c = verdict.citations[0]
        assert doc["text"][c.source_span[0]:c.source_span[1]] == c.quote
        assert c.nli is not None and 0 <= c.similarity <= 1
    sims = [c.similarity for c in first.citations]
    assert sims == sorted(sims, reverse=True)


def test_fact_in_a_second_document_is_not_fabricated(detector):
    claim = "The mission splashed down in the Pacific Ocean."
    assert detector.check_claim(claim, [MISSION]).label != SUPPORTED
    assert detector.check_claim(claim, [MISSION, RETURN]).label == SUPPORTED


def test_conflicting_sources_raise_a_warning(detector):
    docs = [{"id": "encyclopedia", "text": "Insulin was first isolated in 1921 by Banting and Best."},
            {"id": "blog", "text": "Insulin was first isolated in 1925 by Banting and Best."}]
    report = detector.detect("Insulin was first isolated in 1925 by Banting and Best.", docs)
    assert any("sources conflict" in w for w in report.warnings)
    assert {c.source_id for c in report.verdicts[0].citations} == {"encyclopedia", "blog"}


def test_one_source_as_plain_string(detector, apollo):
    report = detector.detect("Apollo 11 launched from Kennedy Space Center on July 16, 1969.", apollo)
    assert report.verdicts[0].citations[0].source_id == "source-1"


# ---------------------------------------------------------------- degenerate input
@pytest.mark.parametrize("sources", [[], "", "   ", [{"id": "blank", "text": ""}], None])
def test_empty_sources_never_crash(detector, sources):
    report = detector.detect("Apollo 11 launched in 1969. The crew collected samples.", sources)
    assert [v.label for v in report.verdicts] == [UNSUPPORTED, UNSUPPORTED]
    assert report.reliability_score == 0.0
    assert any("no source text" in w for w in report.warnings)
    assert all(v.citations == [] for v in report.verdicts)


@pytest.mark.parametrize("response", ["", "   ", "I hope this helps!", "Is that right?"])
def test_response_without_claims_has_no_score(detector, apollo, response):
    report = detector.detect(response, apollo)
    assert report.verdicts == [] and report.reliability_score is None
    assert any("no checkable claims" in w for w in report.warnings)


def test_non_english_response_is_flagged_not_scored(detector, apollo):
    report = detector.detect("아폴로 11호는 1969년 7월 16일 케네디 우주 센터에서 발사되었습니다.", apollo)
    assert report.reliability_score is None
    assert any("English" in w for w in report.warnings)


# ---------------------------------------------------------------- claim handling inside detect()
def test_pronoun_claim_is_checked_with_its_referent(detector):
    source = "Marie Curie was born in Warsaw in 1867. Marie Curie died in 1934 from aplastic anemia."
    report = detector.detect("Marie Curie was born in Warsaw. She died in 1934.", source)
    second = report.verdicts[1]
    assert second.claim.text == "She died in 1934." and "Marie Curie" in second.claim.resolved_text
    assert second.label == SUPPORTED
    assert second.to_dict()["checked_as"] == "Marie Curie died in 1934."


def test_compound_sentence_gets_one_verdict_per_fact(detector, apollo):
    report = detector.detect("The astronauts collected about 21.5 kilograms of lunar samples "
                             "and splashed down in the Atlantic Ocean.", apollo)
    assert [v.label for v in report.verdicts][0] == SUPPORTED
    assert report.verdicts[1].is_hallucination


def test_repeated_claim_is_scored_once(detector, apollo):
    once = detector.detect("The astronauts collected about 45 kilograms of lunar samples.", apollo)
    twice = detector.detect("The astronauts collected about 45 kilograms of lunar samples. "
                            "Apollo 11 launched from Kennedy Space Center on July 16, 1969. "
                            "The astronauts collected about 45 kilograms of lunar samples.", apollo)
    assert twice.verdicts[2].claim.duplicate_of == 1 and twice.verdicts[2].label == once.verdicts[0].label
    assert len(twice.scored) == 2 and twice.hallucination_rate == 0.5


def test_report_records_its_configuration(detector, detector_no_nli, apollo):
    assert detector.detect("Apollo 11 launched in 1969.", apollo).config["use_nli"] is True
    report = detector_no_nli.detect("Apollo 11 launched in 1969.", apollo)
    assert report.config["use_nli"] is False and report.config["nli_model"] is None
    assert report.verdicts[0].citations[0].nli is None
