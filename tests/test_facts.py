import pytest

from hallucination_detector import facts
from hallucination_detector.schemas import CONTRADICTED, FABRICATED, SUPPORTED


@pytest.mark.parametrize("text, expected", [
    ("about 1,200 people", {1200.0}),
    ("revenue grew 12%", {12.0}),
    ("guidance of $1.6 billion", {1.6e9}),                  # one value, not 1.6 and 1e9
    ("won two Nobel Prizes", {2.0}),
    ("21.5 kilograms on July 20, 1969", {21.5, 20.0, 1969.0}),
    ("no numbers here", set()),
    # comma groups keep their decimal part
    ("the bridge is 1,280.5 meters long", {1280.5}),
    ("revenue was 1,234.56 dollars", {1234.56}),
    # scale words and suffixes join their numeral
    ("a population of 2 million", {2e6}),
    ("a population of 2,000,000", {2e6}),
    ("revenue of $2,100 million", {2.1e9}),
    ("1.2M users", {1.2e6}),
    ("a $1.6bn deal", {1.6e9}),
    ("a 5 m wall", {5.0}),                                  # meters, not millions
    ("a million people", {1e6}),
    ("two hundred people", {200.0}),
    # word numbers: teens, tens and hyphenated compounds
    ("seventy-five hospitals", {75.0}),
    ("twenty-one floors", {21.0}),
    ("twenty one floors", {21.0}),
    ("fifteen hospitals", {15.0}),
    ("thirty days", {30.0}),
    # signs
    ("-40 degrees", {-40.0}),
    ("−40 degrees", {-40.0}),
    ("from 1990-1995", {1990.0, 1995.0}),
    ("COVID-19 cases", {19.0}),
    # ordinals, words and digits alike
    ("finished third", {3.0}),
    ("finished 3rd", {3.0}),
    ("the twenty-first century", {21.0}),
    ("the second quarter", {2.0}),
    ("wait a second", set()),
    # pronoun 'one' counts nothing
    ("one of the largest cities", set()),
    ("no one knows", set()),
    ("won one award", {1.0}),
])
def test_number_formats(text, expected):
    assert facts.numbers(text) == expected


def test_equal_values_match_with_relative_tolerance():
    assert facts.has_number(facts.numbers("$2.01 billion"), 2_010_000_000)
    assert not facts.has_number({2e9}, 2.01e9)


def test_reasons_print_large_numbers_in_words():
    assert facts.fmt_numbers({73e6, 1.6e9, 21.5}) == "21.5, 73 million, 1.6 billion"


@pytest.mark.parametrize("text, kinds", [
    ("grew 14 percent", {"%"}),
    ("grew 14%", {"%"}),
    ("$412 million", {"money"}),
    ("fiscal 2026", {"calendar year"}),
    ("45 kilograms", {"kilogram"}),
    ("a 12-year-old boy", {"year"}),
    ("type 1 patients", {"type", "patient"}),
    ("on July 20", {"day"}),
    ("at 1:30", {"time"}),
    ("18 to 24", set()),
])
def test_quantities_know_what_they_count(text, kinds):
    assert set(facts.quantities(text)[0].kinds) == kinds


def test_number_mismatch_with_the_best_sentence(nlp):
    claim = "The astronauts collected about 45 kilograms of lunar samples."
    evidence = ["The astronauts collected about 21.5 kilograms of lunar samples."]
    out = facts.conflicts(nlp, claim, facts.extract(nlp, claim), evidence, evidence[0].lower())
    assert out == ["number mismatch: claim says 45, evidence says 21.5"]


@pytest.mark.parametrize("claim, evidence, expected", [
    # the dev set depends on these staying contradictions
    ("Marie Curie moved to Paris in 1895.", "She moved to Paris in 1891 to study at the University of Paris.",
     "claim says 1895, evidence says 1891"),
    ("Net income for the quarter was $73 million.", "Net income was $37 million.",
     "claim says 73 million, evidence says 37 million"),
    ("The first patient was a 12-year-old boy.",
     "The first patient to receive an insulin injection was a 14-year-old boy named Leonard Thompson in 1922.",
     "claim says 12, evidence says 14"),
    ("Type 1 diabetes is usually managed first with diet, exercise, and metformin.",
     "Type 2 diabetes is often managed initially with diet, exercise, and oral medications such as metformin.",
     "claim says 1, evidence says 2"),
    ("Support for Python 2 officially ended on January 1, 2018.",
     "Support for Python 2 officially ended on January 1, 2020.", "claim says 2018, evidence says 2020"),
])
def test_number_mismatch_compares_numbers_of_the_same_kind(claim, evidence, expected):
    assert facts.number_mismatch(claim, evidence) == f"number mismatch: {expected}"


@pytest.mark.parametrize("claim, evidence", [
    # the 14 percent is stated in the next sentence; the best one only has the revenue
    ("Corvane Systems brought in $412 million in revenue in the third quarter of fiscal 2026, "
     "a 14 percent increase over the same period last year.",
     "Corvane Systems reported revenue of $412 million for the third quarter of fiscal 2026."),
    # '84 students' and 'two lab sessions' have no same-kind number in a sentence about minutes and times
    ("Each of the 84 students attended two lab sessions a week apart, napping for 20 minutes in one "
     "and resting quietly while awake in the other.",
     "In one session participants took a 20-minute nap at 1:30 p.m., and in the other they rested quietly "
     "while awake for the same period."),
])
def test_number_of_another_kind_is_not_a_mismatch(claim, evidence):
    assert facts.number_mismatch(claim, evidence) is None


@pytest.mark.parametrize("claim, evidence", [
    ("The city has a population of 2 million.", "The city has a population of 2,000,000."),
    ("Revenue reached $2.1 billion.", "Revenue reached $2,100 million."),
    ("The service had 1.2 million users at the end of the year.",
     "The service had 1,200,000 users at the end of the year."),
    ("The company has 25 offices worldwide.", "The company has twenty-five offices."),
    ("The survey included seventy-five hospitals.", "The survey included 75 hospitals."),
    ("The bridge is 1,280.5 meters long.", "The bridge is 1280.5 meters long."),
])
def test_equal_amounts_in_different_notation_agree(nlp, claim, evidence):
    out = facts.conflicts(nlp, claim, facts.extract(nlp, claim), [evidence], evidence.lower())
    assert not any("number mismatch" in r for r in out)


@pytest.mark.parametrize("claim, evidence", [
    ("The city has a population of 3 million.", "The city has a population of 2,000,000."),
    ("Revenue was $1,234.56 million.", "Revenue was $1,234.99 million."),
    ("The lowest temperature was -40 degrees Celsius.", "The lowest temperature was 40 degrees Celsius."),
    ("The study included fifteen hospitals.", "The study included 16 hospitals."),
])
def test_different_amounts_still_disagree(nlp, claim, evidence):
    out = facts.conflicts(nlp, claim, facts.extract(nlp, claim), [evidence], evidence.lower())
    assert any("number mismatch" in r for r in out)


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


@pytest.mark.parametrize("claim, evidence", [
    # a negated copula in a side clause is about another predicate, not 'be'
    ("The Eiffel Tower is 330 meters tall.",
     "The Eiffel Tower, which is not the tallest structure in France, is 330 meters tall."),
    ("The vaccine is approved for adults.",
     "The vaccine is approved for adults, but it is not available in rural clinics."),
    ("Mercury is the closest planet to the Sun.",
     "Mercury is the closest planet to the Sun, although it is not the hottest."),
    # 'never take two tablets' negates a different action than 'take it'
    ("If you forget a dose, take it once you remember on the same day.",
     "If you miss a dose of Dravotine, take it as soon as you remember on the same day, "
     "but never take two tablets to make up for a forgotten one."),
    # the claim only uses 'combine' in a side clause
    ("You should steer clear of potassium supplements, because combining them with Dravotine can push "
     "your blood potassium too high.",
     "Dravotine should not be combined with potassium supplements, because this can raise potassium levels."),
])
def test_negation_elsewhere_in_the_sentence_is_not_a_flip(nlp, claim, evidence):
    assert not facts.polarity_flip(nlp, claim, evidence)


@pytest.mark.parametrize("claim, evidence", [
    ("The drug is safe.", "The drug is not safe."),
    ("The cat is a mammal.", "The cat is not a mammal."),
    ("Dravotine can be combined with potassium supplements.",
     "Dravotine should not be combined with potassium supplements."),
])
def test_negated_main_predicate_is_a_flip(nlp, claim, evidence):
    assert facts.polarity_flip(nlp, claim, evidence)


@pytest.mark.parametrize("claim, source", [
    ("The restaurant was opened by Sean O’Brien in Dublin.", "The restaurant was opened by Sean O'Brien in Dublin."),
    ("The restaurant was opened by Sean O'Brien in Dublin.", "The restaurant was opened by Sean O’Brien in Dublin."),
    ("McDonald’s opened its first restaurant in 1940.", "McDonald's opened its first restaurant in 1940."),
])
def test_typographic_apostrophes_match_plain_ones(nlp, claim, source):
    facts_ = facts.extract(nlp, claim)
    assert facts.conflicts(nlp, claim, facts_, [source], source.lower()) == []
    assert facts.novel(facts_, source.lower()) == []


def test_entity_must_match_whole_words(nlp):
    claim = facts.extract(nlp, "Ian Smith founded the company in 1998.")
    assert "'Ian Smith'" in facts.novel(claim, "brian smithson founded the company in 1998.")
    # a demonym still grounds the place name
    assert facts.mentioned("Europe", "european sales were flat.")


def test_person_counts_without_a_leading_title(nlp):
    source = facts.fold("During the call, Chief Financial Officer Desmond Okafor said margins improved. "
                        "Chief Executive Officer Marisol Atwater spoke too. The study was led by Dr. Halvora Brenquist.")
    assert facts.grounded("CFO Desmond Okafor", "PERSON", source, set())
    assert facts.grounded("Halvora Brenquist", "PERSON", source, set())
    assert facts.grounded("Prof. Halvora Brenquist", "PERSON", source, set())
    # the source gives this person another role, so the claimed CFO does not exist
    assert not facts.grounded("CFO Marisol Atwater", "PERSON", source, set())


def test_title_before_a_name_is_not_fabricated(nlp):
    source = "During the investor call, Chief Financial Officer Desmond Okafor said the gross margin improvement " \
             "came mostly from lower component costs."
    claim = facts.extract(nlp, "Gross margin widened, which CFO Desmond Okafor attributed to cheaper components.")
    assert facts.novel(claim, source.lower()) == []


@pytest.mark.parametrize("claim, evidence", [
    ("The treaty was signed in March 2004.", "The treaty was signed in October 2004."),
    ("Apollo 11 landed on the Moon in August 1969.", "Apollo 11 landed on the Moon on July 20, 1969."),
])
def test_date_with_the_wrong_month_is_a_conflict(nlp, claim, evidence):
    out = facts.conflicts(nlp, claim, facts.extract(nlp, claim), [evidence], evidence.lower())
    assert any(r.startswith("DATE mismatch") for r in out)
    assert any("'" in o for o in facts.novel(facts.extract(nlp, claim), evidence.lower()))


def test_date_with_the_same_month_in_other_words_is_grounded(nlp):
    evidence = "apollo 11 landed on the moon on july 20, 1969."
    for claim in ["Apollo 11 landed on the Moon in July 1969.", "Apollo 11 landed on the Moon on 20 July 1969."]:
        assert facts.novel(facts.extract(nlp, claim), evidence) == []


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


# ---------------------------------------------------------------- end to end, no NLI: the fact checks alone
@pytest.mark.parametrize("claim, source", [
    ("The city has a population of 2 million.", "The city has a population of 2,000,000."),
    ("She finished 3rd in the final.", "She finished third in the final."),
    ("Paris is one of the largest cities in Europe.", "Paris is among the largest cities in Europe."),
    ("The Eiffel Tower is 330 meters tall.",
     "The Eiffel Tower, which is not the tallest structure in France, is 330 meters tall."),
    ("The restaurant was opened by Sean O’Brien in Dublin.", "The restaurant was opened by Sean O'Brien in Dublin."),
])
def test_true_claims_are_supported(detector_no_nli, claim, source):
    assert detector_no_nli.check_claim(claim, source).label == SUPPORTED


@pytest.mark.parametrize("claim, source", [
    ("The city has a population of 3 million.", "The city has a population of 2,000,000."),
    ("The lowest recorded temperature in the town was -40 degrees Celsius.",
     "The lowest recorded temperature in the town was 40 degrees Celsius."),
    ("The treaty was signed in March 2004.", "The treaty was signed in October 2004."),
    ("Ian Smith founded the company in 1998.", "Brian Smithson founded the company in 1998."),
])
def test_false_claims_are_caught(detector_no_nli, claim, source):
    assert detector_no_nli.check_claim(claim, source).label in (CONTRADICTED, FABRICATED)


@pytest.mark.parametrize("claim, evidence, expected", [
    ("Revenue was $2.1 billion.", "Revenue was $2,100 million.", True),
    ("The tower has twenty-one floors.", "The tower has 21 floors.", True),
    ("The bridge is 1280.5 meters long.", "The bridge is 1,280.5 meters long.", True),
    ("Revenue was $2.1 billion.", "Revenue was $2.1 billion.", False),          # written the same way
    ("Revenue was $3.1 billion.", "Revenue was $2.1 billion.", False),          # a different amount
    ("Python was created by Guido van Rossum.", "Python was created by Guido van Rossum.", False),   # no numbers
])
def test_same_values_other_notation(claim, evidence, expected):
    assert facts.same_values_other_notation(claim, evidence) is expected
