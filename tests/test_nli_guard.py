"""The NLI notation guard: a contradiction caused only by how equal numbers are written is dropped,
one about anything else is kept. Uses the real models."""
import pytest

from hallucination_detector.schemas import CONTRADICTED, SUPPORTED

NOTE = "written differently"


@pytest.mark.parametrize("claim, source", [
    ("The company reported revenue of $2.1 billion for the year.",
     "The company reported revenue of $2,100 million for the year."),
    ("The tower has twenty-one floors.", "The tower has 21 floors."),
    ("The bridge is 1280.5 meters long.", "The bridge is 1,280.5 meters long."),
    ("The company added 2000 jobs in 2024.", "The company added 2,000 jobs in 2024."),
])
def test_equal_amounts_in_other_notation_are_supported(detector, claim, source):
    verdict = detector.check_claim(claim, source)
    assert verdict.label == SUPPORTED
    assert any(NOTE in r for r in verdict.reasons)


@pytest.mark.parametrize("claim, source", [
    # reversed meaning, same amounts in another notation
    ("Revenue rose to $2.1 billion in 2023.", "Revenue fell to $2,100 million in 2023."),
    ("The company added 1200 jobs to its workforce last year.", "The company cut its workforce by 1,200 jobs last year."),
    ("The museum closed two days a week in winter.", "The museum is open 2 days a week in winter."),
    ("Acme sold its stake in Beta Corp for $2.1 billion.", "Acme bought a stake in Beta Corp for $2,100 million."),
    ("The company cut 2,000 jobs in 2024.", "The company added 2000 jobs in 2024."),
    ("Net loss widened to $1.2 billion in 2025.", "Net loss narrowed to $1,200 million in 2025."),
    ("Exports to China decreased by twelve percent.", "Exports to China increased by 12 percent."),
    # a wrong name, same amounts in another notation. The claimed name appears elsewhere in the source,
    # so the fact checks pass it and only NLI sees the swap.
    ("Microsoft reported revenue of $2.1 billion for the year.",
     "Apple reported revenue of $2,100 million for the year. Microsoft reported a loss for the year."),
    ("The bridge designed by Gustave Eiffel is 1280.5 meters long.",
     "The bridge designed by John Roebling is 1,280.5 meters long. Gustave Eiffel designed a tower."),
])
def test_other_contradiction_with_other_notation_stays_contradicted(detector, claim, source):
    verdict = detector.detect(claim, [source]).verdicts[0]
    assert verdict.label == CONTRADICTED
    assert not any(NOTE in r for r in verdict.reasons)


def test_different_amount_stays_contradicted(detector):
    verdict = detector.check_claim("The company reported revenue of $3.1 billion for the year.",
                                   "The company reported revenue of $2,100 million for the year.")
    assert verdict.label == CONTRADICTED
    assert not any(NOTE in r for r in verdict.reasons)
