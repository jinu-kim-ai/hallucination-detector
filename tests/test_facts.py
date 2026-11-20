import pytest

from hallucination_detector import facts


@pytest.mark.parametrize("text, expected", [
    ("about 1,200 people", {1200.0}),
    ("revenue grew 12%", {12.0}),
    ("guidance of $1.6 billion", {1.6, 1e9}),
    ("won two Nobel Prizes", {2.0}),
    ("21.5 kilograms on July 20, 1969", {21.5, 20.0, 1969.0}),
    ("no numbers here", set()),
])
def test_number_formats(text, expected):
    assert facts.numbers(text) == expected


def test_number_mismatch_with_the_best_sentence(nlp):
    claim = "The astronauts collected about 45 kilograms of lunar samples."
    evidence = ["The astronauts collected about 21.5 kilograms of lunar samples."]
    out = facts.conflicts(nlp, claim, facts.extract(nlp, claim), evidence, evidence[0].lower())
    assert out == ["number mismatch: claim says 45, evidence says 21.5"]


def test_entity_mismatch_names_the_closest_rival(nlp):
    claim = "The mission splashed down in the Atlantic Ocean."
    evidence = ["The mission returned to Earth, splashing down in the Pacific Ocean."]
    out = facts.conflicts(nlp, claim, facts.extract(nlp, claim), evidence, evidence[0].lower())
    assert any("Atlantic Ocean" in r and "Pacific Ocean" in r for r in out)


def test_entity_found_elsewhere_in_the_sources_is_not_a_conflict(nlp):
    claim = "Neil Armstrong launched from Kennedy Space Center."
    evidence = ["Apollo 11 launched from Kennedy Space Center."]
    everything = evidence[0].lower() + " the crew consisted of neil armstrong and buzz aldrin."
    assert facts.conflicts(nlp, claim, facts.extract(nlp, claim), evidence, everything) == []


def test_negation_on_a_shared_verb_is_a_polarity_flip(nlp):
    assert facts.polarity_flip(nlp, "Python 3.0 was backward compatible with Python 2.",
                               "Python 3.0 was not backward compatible with Python 2.")
    assert not facts.polarity_flip(nlp, "Python 3.0 was not backward compatible.",
                                   "Python 3.0 was not backward compatible with Python 2.")


def test_negation_on_a_different_verb_is_not_a_flip(nlp):
    # v1 regression: the negated verb ("produce") does not appear in the claim
    evidence = "People with type 1 diabetes do not produce enough insulin and require insulin therapy."
    assert not facts.polarity_flip(nlp, "People with type 1 diabetes require insulin therapy.", evidence)


def test_novel_facts_appear_in_no_source(nlp):
    source = "apollo 11 launched from kennedy space center on july 16, 1969."
    claim = facts.extract(nlp, "The landing site was chosen by 300 geologists at Stanford University.")
    out = facts.novel(claim, source)
    assert "300" in out and any("Stanford University" in o for o in out)
    grounded = facts.extract(nlp, "Apollo 11 launched on July 16, 1969.")
    assert facts.novel(grounded, source) == []


def test_relative_dates_are_not_checkable_facts(nlp):
    found = facts.extract(nlp, "Revenue grew compared with the same quarter last year.")
    assert all(label != "DATE" for _, label in found.entities)
