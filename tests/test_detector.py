import pytest

from hallucination_detector import (
    CONTRADICTED, FABRICATED, SUPPORTED, UNSUPPORTED, ClaimExtractor, HallucinationDetector,
)
from hallucination_detector.evaluate import BinaryMetrics, score

SOURCE = (
    "Apollo 11 launched from Kennedy Space Center on July 16, 1969. "
    "On July 20, 1969, the lunar module Eagle landed in the Sea of Tranquility. "
    "The astronauts collected about 21.5 kilograms of lunar samples. "
    "The mission returned to Earth on July 24, 1969, splashing down in the Pacific Ocean."
)


@pytest.fixture(scope="module")
def detector():
    return HallucinationDetector()


def test_claim_splitting_on_coordinated_verbs(detector):
    claims = ClaimExtractor(detector.nlp).extract(
        "Marie Curie discovered polonium and won two Nobel Prizes. Is that right? I hope this helps!"
    )
    texts = [c.text for c in claims]
    assert texts == ["Marie Curie discovered polonium.", "Marie Curie won two Nobel Prizes."]


def test_supported_paraphrase(detector):
    r = detector.check_claim("The Eagle lunar module touched down in the Sea of Tranquility on July 20, 1969.", SOURCE)
    assert r.label == SUPPORTED


def test_number_contradiction(detector):
    r = detector.check_claim("The astronauts collected about 45 kilograms of lunar samples.", SOURCE)
    assert r.label == CONTRADICTED
    assert any("number mismatch" in x for x in r.reasons)


def test_entity_contradiction(detector):
    r = detector.check_claim("The mission splashed down in the Atlantic Ocean on July 24, 1969.", SOURCE)
    assert r.label == CONTRADICTED


def test_fabricated_details(detector):
    r = detector.check_claim("The landing site was picked by 300 geologists at Stanford University.", SOURCE)
    assert r.label == FABRICATED


def test_unsupported_opinion(detector):
    r = detector.check_claim("The astronauts felt very nervous during the descent.", SOURCE)
    assert r.label in (UNSUPPORTED, FABRICATED)
    assert r.is_hallucination


def test_report_scores(detector):
    report = detector.detect("Apollo 11 launched from Kennedy Space Center on July 16, 1969. "
                             "The crew collected 45 kilograms of samples.", SOURCE)
    assert len(report.results) == 2
    assert 0 < report.hallucination_rate < 1
    assert 0 < report.reliability_score < 100


def test_binary_metrics():
    m = BinaryMetrics(tp=8, fp=2, fn=2, tn=8)
    assert m.precision == pytest.approx(0.8)
    assert m.recall == pytest.approx(0.8)
    assert m.f1 == pytest.approx(0.8)


def test_score_shapes():
    preds = [
        {"domain": "a", "label": SUPPORTED, "pred": SUPPORTED},
        {"domain": "a", "label": FABRICATED, "pred": UNSUPPORTED},
        {"domain": "b", "label": CONTRADICTED, "pred": SUPPORTED},
    ]
    m = score(preds)
    assert m["binary_hallucination_detection"]["recall"] == pytest.approx(0.5)
    assert m["confusion"][FABRICATED][UNSUPPORTED] == 1
