"""Two deliberate design decisions, pinned so they are not undone by accident."""
import pytest

from hallucination_detector import CONTRADICTED, ClaimExtractor


@pytest.mark.parametrize("response, claim", [
    ("I think Apollo 11 probably launched in 1970.", "I think Apollo 11 probably launched in 1970."),
    ("Apollo 11 might have launched from Florida.", "Apollo 11 might have launched from Florida."),
])
def test_hedged_sentences_are_claims(nlp, response, claim):
    """A hedge does not make a false statement true, so a hedged sentence is checked rather than dropped."""
    assert [c.text for c in ClaimExtractor(nlp).extract(response)] == [claim]


def test_a_hedged_wrong_year_is_still_flagged(detector, apollo):
    verdict = detector.detect("I think Apollo 11 probably launched from Kennedy Space Center on July 16, 1970.",
                              apollo).verdicts[0]
    assert verdict.is_hallucination
    assert verdict.label == CONTRADICTED


def test_questions_and_pleasantries_are_still_dropped(nlp):
    assert ClaimExtractor(nlp).extract("Did Apollo 11 launch in 1969? I hope this helps!") == []


def test_uncited_claim_reports_its_closest_match(detector, apollo):
    """No citation below the related threshold, but the closest similarity is still reported."""
    verdict = detector.detect("The astronauts felt very nervous during the descent.", apollo).verdicts[0]
    data = verdict.to_dict()
    assert data["citations"] == [] and 0 < data["best_similarity"] < detector.config.related_threshold
