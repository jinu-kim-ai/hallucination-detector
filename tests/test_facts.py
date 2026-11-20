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
    # compound number words add up
    ("one hundred twenty people", {120.0}),
    ("one hundred and twenty", {120.0}),
    ("one thousand nine hundred", {1900.0}),
    ("one hundred twenty-five", {125.0}),
    ("two million three hundred thousand", {2.3e6}),
    ("between one hundred and two hundred", {100.0, 200.0}),
    ("twenty one hundred", {2100.0}),
    # 'hundred and fifty thousand': the thousand scales the whole group, British style
    ("Two hundred and fifty thousand people attended.", {250000.0}),
    ("two hundred and fifty thousand", {250000.0}),
    ("one hundred and ten thousand", {110000.0}),
    ("one hundred and twenty-five thousand", {125000.0}),
    ("one million two hundred and fifty thousand", {1250000.0}),
    ("one thousand and fifty", {1050.0}),
    # pinned reading of an ambiguous phrase: one number, not the range 100 to 50,000
    ("between one hundred and fifty thousand", {150000.0}),
])
def test_number_formats(text, expected):
    assert facts.numbers(text) == expected


@pytest.mark.parametrize("text, expected", [
    # an ordinal is a rank or position, not an amount the sources must contain
    ("First, the drug lowers blood pressure.", set()),
    ("Armstrong was the first person on the Moon.", set()),
    ("Busan is the second-largest city.", set()),
    ("Fourth-quarter revenue", set()),
    ("finished 3rd in 2019", {2019.0}),             # digit ordinals alike
    # a day of the month is a date, not a rank
    ("on July 20th, 1969", {20.0, 1969.0}),
    ("the 5th of May", {5.0}),
])
def test_plain_numbers_leave_out_ordinals(text, expected):
    assert facts.plain_numbers(text) == expected


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
    # different nouns for counted people
    ("Today the plant employs 300 workers in total.", "Today the plant employs 1,500 people in total.",
     "claim says 300, evidence says 1500"),
    ("The company has 500 people.", "The company has 450 employees.", "claim says 500, evidence says 450"),
    # the unit is implicit on one side
    ("Sales reached 3.2 million units.", "Sales reached 2.9 million.", "claim says 3.2 million, evidence says 2.9 million"),
    # a comma-less count is no year when a noun follows it
    ("The factory has 1,500 workers.", "The factory has 1800 workers.", "claim says 1500, evidence says 1800"),
    # ...but stays a year after a preposition
    ("By 2005 sales had doubled.", "Sales doubled by 2004 when the plant opened.", "claim says 2005, evidence says 2004"),
    # ordinals compare with ordinals
    ("She ranked third in the race.", "She ranked fifth in the race.", "claim says 3, evidence says 5"),
    # a decline is not a rise of the same size
    ("Operating margin rose 3.2% year over year.", "Operating margin changed by -3.2% year over year.",
     "claim says 3.2, evidence says -3.2"),
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
    # a bare number never stands in for money, a percentage or a year
    ("Corvane ranked 3 among peers with revenue of $412 million.", "Corvane reported revenue of $412 million."),
    ("Adults aged 18 to 24 made up 14 percent of respondents in 2000.",
     "Adults made up 14 percent of respondents in 2000."),
    # a rank is not a count
    ("Armstrong was the first person to walk on the Moon.", "About 300 people watched the launch."),
    ("First, the drug lowers blood pressure in 3 of 4 adults.", "The drug lowers blood pressure in 3 of 4 adults."),
])
def test_number_of_another_kind_is_not_a_mismatch(claim, evidence):
    assert facts.number_mismatch(claim, evidence) is None


@pytest.mark.parametrize("claim, evidence", [
    # 'member' counts members, not the people who founded the club or the states of a union
    ("The Riverside Rowing Club has 5,000 members.", "The Riverside Rowing Club was founded by 12 people in 1898."),
    ("The European Union has 450 million people.", "The European Union has 27 member states."),
    # a year after a month name is a date even when a noun follows
    ("In March 2020 workers at the Detroit plant went on strike.",
     "About 1,500 workers at the Detroit plant went on strike."),
    ("On 4 March 2020 staff walked out of the Leeds office.", "About 900 staff walked out of the Leeds office."),
    # the same amount in British wording
    ("Two hundred and fifty thousand people attended the rally.", "250,000 people attended the rally."),
])
def test_number_of_other_people_or_a_date_is_not_a_mismatch(claim, evidence):
    assert facts.number_mismatch(claim, evidence) is None


def test_year_after_a_month_name_still_compares_with_years():
    assert facts.number_mismatch("In March 2020 workers at the Detroit plant went on strike.",
                                 "In March 2021 workers at the Detroit plant went on strike.") == \
        "number mismatch: claim says 2020, evidence says 2021"


# An article does not make a 4-digit number a year: 'The 1500 employees' is a head count.
ARTICLE_COUNTS = [
    ("The 1500 employees at the Ohio plant build engines.",
     "The company has 1,500 employees in total. The Ohio plant employs 900 employees who build engines.",
     "number mismatch: claim says 1500, evidence says 900"),
    ("The team plays in a 1200-seat arena downtown.",
     "The club sold 1,200 season tickets. The team plays in a 4,000-seat arena downtown.",
     "number mismatch: claim says 1200, evidence says 4000"),
    ("The 1800 students at the school wear uniforms.",
     "The district enrolls 1,800 students. The school has 650 students, all of whom wear uniforms.",
     "number mismatch: claim says 1800, evidence says 650"),
    ("The 1800 residents of the village were evacuated.",
     "All 1,200 residents of the village were evacuated after the flood.",
     "number mismatch: claim says 1800, evidence says 1200"),
]


def test_four_digit_count_after_an_article_keeps_its_noun():
    (q,) = facts.quantities("The 1500 employees at the Ohio plant build engines.")
    assert q.kinds == frozenset({"calendar year", "person"})     # employees count people
    (q,) = facts.quantities("The team plays in a 1200-seat arena downtown.")
    assert "seat" in q.kinds


@pytest.mark.parametrize("claim, source, reason", ARTICLE_COUNTS)
def test_wrong_count_after_an_article_is_contradicted(detector_no_nli, claim, source, reason):
    verdict = detector_no_nli.check_claim(claim, source)
    assert verdict.label == CONTRADICTED
    assert reason in verdict.reasons


def test_count_after_an_article_is_compared_with_the_count_not_only_the_year(detector_no_nli):
    verdict = detector_no_nli.check_claim("The 1800 employees received a bonus in 2021.",
                                          "In 2021, the 1,200 employees received a bonus.")
    assert verdict.label == CONTRADICTED
    assert "number mismatch: claim says 1800, evidence says 1200, 2021" in verdict.reasons


SMITH = ("The Smith Tower, designed by John Smith, has 30 floors.",
         "The Smith Tower was designed by architect John Smith when he was 45. The building has 30 floors.")
BRIGHTMART = ("Brightmart, ranked 15 on the Fortune list, opened 12 new stores in Ohio.",
              "Brightmart, ranked 15 on the Fortune list, opened new stores in Ohio. "
              "In total it opened 12 stores that year.")


def test_lone_partner_is_dropped_when_the_related_evidence_states_the_claim_number_with_its_unit(nlp):
    claim, source = SMITH
    best = "The Smith Tower was designed by architect John Smith when he was 45."
    # an age is the only bare number in the best sentence; the 30 floors are in the next one
    assert facts.number_mismatch(claim, best, source) is None
    out = facts.conflicts(nlp, claim, facts.extract(nlp, claim), [best, "The building has 30 floors."], source.lower())
    assert out == []
    # with nothing else stated, an unlabelled same-sized number is still the one the claim must mean
    assert facts.number_mismatch("Sales reached 3.2 million units.", "Sales reached 2.9 million.",
                                 "Sales reached 2.9 million.") == \
        "number mismatch: claim says 3.2 million, evidence says 2.9 million"


MUSEUM = ("The museum welcomed 1.4 million visitors in 2019.",
          "Attendance at the museum hit 1.1 million in 2019. In 2018 attendance was 1.4 million, a record.")


def test_lone_partner_stays_when_the_claim_number_is_elsewhere_without_its_unit(detector_no_nli):
    claim, source = MUSEUM
    best = "Attendance at the museum hit 1.1 million in 2019."
    # the 1.4 million of 2018 has no 'visitors' after it, so it does not show where the claim's figure came from
    assert facts.number_mismatch(claim, best, source) == \
        "number mismatch: claim says 1.4 million, evidence says 1.1 million"
    verdict = detector_no_nli.check_claim(claim, source)
    assert verdict.label == CONTRADICTED
    assert "number mismatch: claim says 1.4 million, evidence says 1.1 million" in verdict.reasons


def test_lone_partner_is_never_a_number_the_claim_states_itself():
    claim, _ = BRIGHTMART
    # the 15 is the claim's own rank, so it cannot also be the partner of the 12 stores
    assert facts.number_mismatch(claim, "Brightmart, ranked 15 on the Fortune list, opened new stores in Ohio.") is None


@pytest.mark.parametrize("claim, source", [SMITH, BRIGHTMART])
def test_age_or_rank_beside_a_count_is_no_contradiction(detector, detector_no_nli, claim, source):
    assert detector.check_claim(claim, source).label == SUPPORTED
    assert detector_no_nli.check_claim(claim, source).label == SUPPORTED


@pytest.mark.parametrize("claim, evidence", [
    ("Operating margin fell 3.2% year over year.", "Operating margin changed by -3.2% year over year."),
    ("Net income declined 12% in the quarter.", "Net income changed by -12% in the quarter."),
    ("Net income changed by -12% in the quarter.", "Net income declined by 12% in the quarter."),
    ("Sales posted a 5% decline.", "Sales changed by -5%."),
])
def test_a_fall_matches_a_negative_number(nlp, claim, evidence):
    assert facts.number_mismatch(claim, evidence) is None
    assert facts.novel(facts.extract(nlp, claim), evidence.lower()) == []


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


@pytest.mark.parametrize("claim, evidence", [
    # 'X but not Y': the negation sits on the excluded argument, not on the verb
    ("The Silver plan covers dental care and vision care.", "The Silver plan covers dental care but not vision care."),
    ("The Silver plan covers vision care.", "The Silver plan covers dental care but not vision care."),
    ("The store ships to the United States and to Canada.", "The store ships to the United States but not to Canada."),
    ("The store ships to Canada.", "The store ships to the United States but not to Canada."),
    ("The vaccine is approved for adults and for children.", "The vaccine is approved for adults but not for children."),
    ("The vaccine is approved for children.", "The vaccine is approved for adults but not for children."),
])
def test_asserting_an_excluded_argument_is_a_flip(nlp, claim, evidence):
    assert facts.polarity_flip(nlp, claim, evidence)
    assert facts.polarity_flip(nlp, evidence, claim)


@pytest.mark.parametrize("claim, evidence", [
    # the claim keeps to what the evidence includes, or repeats the same exclusion
    ("The Silver plan covers dental care.", "The Silver plan covers dental care but not vision care."),
    ("The Silver plan covers dental care but not vision care.", "The Silver plan covers dental care but not vision care."),
    ("The store ships to the United States.", "The store ships to the United States but not to Canada."),
    ("The vaccine is approved for adults.", "The vaccine is approved for adults but not for children."),
])
def test_keeping_to_the_included_argument_is_not_a_flip(nlp, claim, evidence):
    assert not facts.polarity_flip(nlp, claim, evidence)


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
    # a demonym still grounds the place name, in the plural too
    assert facts.mentioned("Europe", "european sales were flat.")
    assert facts.mentioned("America", "most americans support it.")
    assert facts.mentioned("Europe", "europeans")
    assert facts.mentioned("Africa", "africans")
    assert not facts.mentioned("Ian Smith", "ian smithsons")


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


@pytest.mark.parametrize("claim, source", [
    ("November 4, 2019", "the company completed the merger on 2019-11-04 after regulators approved it."),
    ("July 1, 2021", "the updated remote work policy took effect on 2021-07-01."),
    ("March 15, 2022", "the company announced the deal on 03/15/2022."),
    ("November 2019", "the merger closed on 2019/11/04."),
])
def test_date_written_in_numbers_is_grounded(claim, source):
    assert facts.grounded(claim, "DATE", source, facts.numbers(source))


@pytest.mark.parametrize("claim, source", [
    ("November 4, 2019", "the company completed the merger on 2019-04-11 after regulators approved it."),   # April 11
    ("November 5, 2019", "the merger closed on 2019-11-04."),
    ("November 4, 2020", "the merger closed on 2019-11-04."),
])
def test_date_written_in_numbers_must_match_exactly(claim, source):
    assert not facts.grounded(claim, "DATE", source, facts.numbers(source))


def test_numeric_date_is_no_date_mismatch(nlp):
    claim = "The company completed the merger on November 4, 2019 after regulators approved it."
    evidence = ["The company completed the merger on 2019-11-04 after regulators approved it."]
    assert facts.conflicts(nlp, claim, facts.extract(nlp, claim), evidence, evidence[0].lower()) == []
    swapped = ["The company completed the merger on 2019-04-11 after regulators approved it."]
    out = facts.conflicts(nlp, claim, facts.extract(nlp, claim), swapped, swapped[0].lower())
    assert any(r.startswith("DATE mismatch") for r in out)


@pytest.mark.parametrize("ent, source, expected", [
    # the modal 'may' and the verb 'march' are no months
    ("May 2019", "in march 2019 the agency approved the drug, which may cause dizziness.", False),
    ("March 2019", "in may 2019 the agency approved it, and protesters could march on.", False),
    # a month word next to a number, or introduced as a date, still counts
    ("May 2019", "in may 2019 the agency approved the drug.", True),
    ("May 2019", "on 5 may 2019 the agency approved the drug.", True),
    ("May 2019", "in may, the agency approved the 2019 plan.", True),
    ("March 2019", "the agency approved it on march 3, 2019.", True),
    # the start of a range that 'from' / 'between' opens and a second month name closes
    ("March 2019", "production ran from march to june 2019 at the ohio plant.", True),
    ("March 2019", "production ran between march and june 2019 at the ohio plant.", True),
    ("March 2019", "production ran from march through june 2019.", True),
    ("May 2019", "production ran from june to may 2019.", True),
    # without both, 'march through june' could be the verb ('plan to march through june'): left ungrounded
    ("March 2019", "production ran march through june 2019.", False),
    ("May 2019", "costs from may exceed estimates in 2019.", False),
    # the modal and the verb stay ordinary words next to 'to', 'and', 'this', 'each' and 'every'
    ("May 2019", "in 2019 the agency said this may cause delays.", False),
    ("May 2019", "prices rose in 2019 and may fall.", False),
    ("March 2019", "in 2019 protesters plan to march on the capital.", False),
    ("May 2019", "in 2019 each may apply for a grant.", False),
    ("March 2019", "in 2019 every march drew crowds.", False),
])
def test_ambiguous_month_words_count_only_as_dates(ent, source, expected):
    assert facts.grounded(ent, "DATE", source, facts.numbers(source)) is expected


@pytest.mark.parametrize("claim, source", [
    ("Production began in March 2019.", "Production ran from March to June 2019 at the Ohio plant."),
    ("Production began in March 2019.", "Production ran between March and June 2019 at the Ohio plant."),
    ("Production began in May 2019.", "Production ran from May to June 2019 at the Ohio plant."),
])
def test_month_in_a_source_range_is_not_fabricated(detector, claim, source):
    assert detector.check_claim(claim, source).label != FABRICATED


@pytest.mark.parametrize("claim, source, ent", [
    ("The new terminal opened in May 2023.",
     "Passenger numbers dipped in April and may recover by the summer. The new terminal opened in 2023.",
     "May 2023"),
    ("The company will report earnings in May 2025.",
     "Shares fell sharply in February and may fall further, analysts said. The company will report earnings in 2025.",
     "May 2025"),
    ("The rally took place in March 2019.",
     "Protesters plan to march through June, organisers said. The rally took place in 2019.", "March 2019"),
    ("The exhibit opens in May 2024.",
     "Prices rose in June and may continue to rise in 2024. The exhibit opens in 2024.", "May 2024"),
    ("The vaccine was approved in May 2021.",
     "The choice between may and must matters in law. The vaccine was approved in 2021.", "May 2021"),
    ("Construction began in May 2015.",
     "Construction began in 2015. Officials said costs from may exceed estimates.", "May 2015"),
])
def test_modal_or_verb_beside_a_month_name_grounds_no_month(detector, detector_no_nli, claim, source, ent):
    assert facts.grounded(ent, "DATE", source.lower(), facts.numbers(source.lower())) is False
    for det in (detector, detector_no_nli):
        verdict = det.check_claim(claim, source)
        assert verdict.label == FABRICATED
        assert f"not found in source: '{ent}'" in verdict.reasons


@pytest.mark.parametrize("ent, source, expected", [
    # an ordinal written as a word on one side and in digits on the other
    ("The Nineteenth Amendment", "the 19th amendment to the u.s. constitution, ratified in 1920.", True),
    ("5th Avenue", "the flagship store is located on fifth avenue in new york.", True),
    ("The 3rd Infantry Division", "the third infantry division entered baghdad in april 2003.", True),
    # a different ordinal, or the same number in other words, does not ground the name
    ("The Eighteenth Amendment", "the 19th amendment to the u.s. constitution, ratified in 1920.", False),
    ("6th Avenue", "the flagship store is located on fifth avenue in new york.", False),
    ("The 4th Infantry Division", "the third infantry division entered baghdad in april 2003.", False),
    ("The Nineteenth Amendment", "19 people voted for the amendment.", False),
    ("The Nineteenth Amendment", "the 19th century amendment was popular.", False),
])
def test_ordinal_in_other_notation_grounds_a_name(ent, source, expected):
    assert facts.grounded(ent, "LAW", source, facts.numbers(source)) is expected


ORDINAL_PAIRS = [
    ("The Nineteenth Amendment granted women the right to vote in 1920.",
     "The 19th Amendment to the U.S. Constitution, ratified in 1920, granted women the right to vote."),
    ("The store is located on 5th Avenue in New York.", "The flagship store is located on Fifth Avenue in New York."),
    ("The 3rd Infantry Division entered Baghdad in April 2003.",
     "The Third Infantry Division entered Baghdad in April 2003."),
]


@pytest.mark.parametrize("claim, source", ORDINAL_PAIRS)
def test_ordinal_name_in_other_notation_is_supported(nlp, detector, claim, source):
    assert facts.novel(facts.extract(nlp, claim), source) == []
    assert detector.check_claim(claim, source).label == SUPPORTED


@pytest.mark.parametrize("claim, source", [
    ("The Eighteenth Amendment granted women the right to vote in 1920.", ORDINAL_PAIRS[0][1]),
    ("The store is located on 6th Avenue in New York.", ORDINAL_PAIRS[1][1]),
    ("The 4th Infantry Division entered Baghdad in April 2003.", ORDINAL_PAIRS[2][1]),
])
def test_wrong_ordinal_in_a_name_is_contradicted(detector_no_nli, claim, source):
    assert detector_no_nli.check_claim(claim, source).label == CONTRADICTED


@pytest.mark.parametrize("text", ["The app has 5M users.", "The model has 3B parameters.", "Northwind employs 1.2K people."])
def test_abbreviated_scale_is_read_the_same_in_the_source(nlp, text):
    claim = facts.extract(nlp, text)
    assert facts.novel(claim, text) == []
    assert facts.novel(claim, text.lower()) == []       # the pipeline passes lower-cased source text


def test_lower_case_metre_does_not_ground_millions_in_cased_source(nlp):
    assert facts.novel(facts.extract(nlp, "The app has 5M users."), "The wall is 5m tall.") == ["5 million"]


def test_ordinal_is_not_a_novel_fact(nlp):
    source = "the drug lowers blood pressure in adults, according to a large trial."
    for claim in ["First, the drug lowers blood pressure in adults.", "Third, the drug lowers blood pressure in adults.",
                  "At first, the drug lowered blood pressure in adults."]:
        assert facts.novel(facts.extract(nlp, claim), source) == []
    # 'Fourth-quarter' is no checkable date on its own
    found = facts.extract(nlp, "Fourth-quarter 2023 revenue was $412 million, up 8% from a year earlier.")
    assert facts.novel(found, "q4 2023 revenue was $412 million, up 8% from q4 2022.") == []


@pytest.mark.parametrize("text, expected", [
    ("Revenue was $2,100 million.", "Revenue was $2100000000."),
    ("Revenue was $2.1 billion.", "Revenue was $2100000000."),
    ("The bridge is 1,280.5 meters long.", "The bridge is 1280.5 meters long."),
    ("The tower has twenty-one floors.", "The tower has 21 floors."),
    ("The app has 2.1B users.", "The app has 2100000000 users."),
    ("She finished third.", "She finished 3rd."),
    ("Margin changed by -3.2%.", "Margin changed by -3.2%."),
    ("No numbers here.", "No numbers here."),
])
def test_canonical_numbers(text, expected):
    assert facts.canonical_numbers(text) == expected


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
    # K/M/B suffixes copied verbatim from the source
    ("The messaging app now has 5M users across Europe.", "The messaging app now has 5M users across Europe."),
    ("The model has 3B parameters.", "The model has 3B parameters."),
    ("Northwind employs 1.2K people in its Ohio offices.", "Northwind employs 1.2K people in its Ohio offices."),
    # ordinals are wording, not numbers to verify
    ("First, the drug lowers blood pressure in adults.",
     "The drug lowers blood pressure in adults, according to a large trial."),
    ("Third, the drug lowers blood pressure in adults.",
     "The drug lowers blood pressure in adults, according to a large trial."),
    ("The drug was first approved in 2019.", "The drug was approved in 2019."),
    ("Fourth-quarter 2023 revenue was $412 million, up 8% from a year earlier.",
     "Q4 2023 revenue was $412 million, up 8% from Q4 2022."),
    # a fall matches the source's signed percentage
    ("Operating margin fell 3.2% year over year, according to the annual report.",
     "Operating margin changed by -3.2% year over year, according to the annual report."),
    ("Net income declined 12% in the third quarter.", "Net income changed by -12% in the third quarter."),
    # dates the source writes in numbers
    ("The updated remote work policy took effect on July 1, 2021.",
     "The updated remote work policy took effect on 2021-07-01."),
    ("The company announced the deal on March 15, 2022.", "The company announced the deal on 03/15/2022."),
    ("The company completed the merger on November 4, 2019 after regulators approved it.",
     "The company completed the merger on 2019-11-04 after regulators approved it."),
    # a plural demonym grounds the place
    ("A 2022 survey found that most people in America support paid family leave.",
     "A 2022 survey found that most Americans support paid family leave."),
    # compound number words
    ("The ship carried one hundred twenty passengers.", "The ship carried 120 passengers."),
    ("The ship carried one hundred and twenty passengers.", "The ship carried 120 passengers."),
    ("Two hundred and fifty thousand people attended the rally.", "250,000 people attended the rally."),
    # ordinals written differently in a name
    *ORDINAL_PAIRS,
    # club members are not the founders; a year after a month name is no head count
    ("The Riverside Rowing Club has 5,000 members.",
     "The Riverside Rowing Club was founded by 12 people in 1898. Today the club has 5,000 members."),
    ("In March 2020 workers at the Detroit plant went on strike.",
     "The strike began in March 2020. About 1,500 workers at the Detroit plant went on strike."),
    ("On 4 March 2020 staff walked out of the Leeds office.",
     "The walkout began on 4 March 2020. About 900 staff walked out of the Leeds office."),
    # a month at one end of a source range
    ("Production began in March 2019.", "Production ran from March to June 2019 at the Ohio plant."),
])
def test_true_claims_are_supported(detector_no_nli, claim, source):
    assert detector_no_nli.check_claim(claim, source).label == SUPPORTED


@pytest.mark.parametrize("claim, source", [
    ("The city has a population of 3 million.", "The city has a population of 2,000,000."),
    ("The lowest recorded temperature in the town was -40 degrees Celsius.",
     "The lowest recorded temperature in the town was 40 degrees Celsius."),
    ("The treaty was signed in March 2004.", "The treaty was signed in October 2004."),
    ("Ian Smith founded the company in 1998.", "Brian Smithson founded the company in 1998."),
    # the agency approved it in March; 'which may cause' is no month
    ("The agency approved the drug in May 2019.",
     "In March 2019 the agency approved the drug, which may cause dizziness."),
])
def test_false_claims_are_caught(detector_no_nli, claim, source):
    assert detector_no_nli.check_claim(claim, source).label in (CONTRADICTED, FABRICATED)


@pytest.mark.parametrize("response, source", [
    ("Today the plant employs 300 workers in total.",
     "The plant opened in 1998 with 300 workers. Today the plant employs 1,500 people in total."),
    ("The company has 500 people.", "The company has 450 employees."),
    ("Sales reached 3.2 million units.", "Sales reached 2.9 million."),
    ("The factory has 1,500 workers.", "The factory has 1800 workers."),
    ("She ranked third in the race.", "She ranked fifth in the race."),
    ("Operating margin rose 3.2% year over year, according to the annual report.",
     "Operating margin changed by -3.2% year over year, according to the annual report."),
    ("The company completed the merger on November 4, 2019 after regulators approved it.",
     "The company completed the merger on 2019-04-11 after regulators approved it."),
    ("The Silver plan covers dental care and vision care.", "The Silver plan covers dental care but not vision care."),
    ("The store ships to the United States and to Canada.", "The store ships to the United States but not to Canada."),
    ("The vaccine is approved for adults and for children.", "The vaccine is approved for adults but not for children."),
])
def test_conflicting_claims_are_contradicted_without_nli(detector_no_nli, response, source):
    assert detector_no_nli.detect(response, source).verdicts[0].label == CONTRADICTED


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


@pytest.mark.parametrize("claim, evidence, flipped", [
    ("The company did not cut jobs in 2023.", "In 2023 the company did not raise prices or cut jobs.", False),
    ("The museum can't be photographed on Mondays.", "The museum can't be visited or photographed on Mondays.", False),
    ("Patients should not operate machinery.", "Patients should not drive or operate machinery.", False),
    ("The company cut jobs in 2023.", "In 2023 the company did not raise prices or cut jobs.", True),
    ("The museum can be photographed on Mondays.", "The museum can't be visited or photographed on Mondays.", True),
    # 'and' does not carry the negation to the second verb
    ("People with type 1 diabetes require insulin therapy.",
     "People with type 1 diabetes do not produce enough insulin and require insulin therapy.", False),
])
def test_negation_shared_through_or_covers_every_verb(nlp, claim, evidence, flipped):
    assert facts.polarity_flip(nlp, claim, evidence) is flipped


# ---------------------------------------------------------------- long sources
def test_long_source_is_checked_in_linear_time(nlp):
    # Reading every number's kind from the whole text before it, once per claim, took 13-16 s per claim
    # on a source this size. The source's numbers are now read once, each from a short window.
    import time

    from hallucination_detector import HallucinationDetector

    detector = HallucinationDetector(nlp=nlp, use_nli=False, embedder="hashing")
    source = " ".join(f"In {1900 + i % 120} the plant in district {i} employed {100 + i} workers and "
                      f"earned ${i}.5 million." for i in range(2000))
    response = ("The plant in district 7 employed 107 workers in 1907. "
                "The plant in district 9 earned $9.5 million. The plant in district 11 employed 5,000 workers.")
    start = time.perf_counter()
    report = detector.detect(response, source)
    elapsed = time.perf_counter() - start
    assert len(report.verdicts) == 3
    assert elapsed < 20, f"detect() took {elapsed:.1f} s on a 2,000-sentence source"


def test_number_kinds_do_not_depend_on_distant_text():
    near = facts.quantities("In March 2020 workers at the plant went on strike. The factory has 1800 workers.")
    far = facts.quantities("Filler text without digits. " * 5000
                           + "In March 2020 workers at the plant went on strike. The factory has 1800 workers.")
    assert near == far


@pytest.mark.parametrize("text, expected", [
    ("a $5-million grant", {5e6}),
    ("a $5 million grant", {5e6}),
    ("from 1990-1995", {1990.0, 1995.0}),              # a range, not a scale
    ("COVID-19 cases", {19.0}),
    ("the 2021 billion-dollar deal", {2021.0, 1e9}),   # the year is not multiplied by the compound's scale
    ("a 12-year-old boy", {12.0}),
])
def test_hyphenated_scale_word(text, expected):
    assert facts.numbers(text) == expected
