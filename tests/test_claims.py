import pytest

from hallucination_detector import CONTRADICTED, SUPPORTED, ClaimExtractor


@pytest.fixture(scope="module")
def extract(nlp):
    extractor = ClaimExtractor(nlp)
    return extractor.extract


def texts(claims):
    return [c.text for c in claims]


def test_coordinated_verbs_share_the_subject(extract):
    claims = extract("Marie Curie discovered polonium and won two Nobel Prizes.")
    assert texts(claims) == ["Marie Curie discovered polonium.", "Marie Curie won two Nobel Prizes."]


def test_coordinated_clauses_keep_their_own_subjects(extract):
    claims = extract("Armstrong walked on the Moon first, and Aldrin followed 19 minutes later.")
    assert texts(claims) == ["Armstrong walked on the Moon first.", "Aldrin followed 19 minutes later."]


def test_list_inside_a_noun_phrase_stays_one_claim(extract):
    assert len(extract("The crew was Armstrong, Aldrin and Collins.")) == 1


def test_non_restrictive_relative_clause_becomes_its_own_claim(extract):
    claims = extract("Marie Curie, who was born in Warsaw, won the Nobel Prize in Physics in 1903.")
    assert texts(claims) == ["Marie Curie won the Nobel Prize in Physics in 1903.", "Marie Curie was born in Warsaw."]


def test_restrictive_relative_clause_is_not_split(extract):
    claims = extract("The telescope that NASA launched in 1990 is still operating.")
    assert texts(claims) == ["The telescope that NASA launched in 1990 is still operating."]


def test_questions_pleasantries_and_fragments_are_dropped(extract):
    claims = extract("Python was created by Guido van Rossum. Is that right? Yes, exactly. I hope this helps!")
    assert texts(claims) == ["Python was created by Guido van Rossum."]


def test_empty_response_has_no_claims(extract):
    assert extract("") == []
    assert extract("   \n\n ") == []


@pytest.mark.parametrize("response, pronoun_claim, checked_as", [
    ("Marie Curie was born in Warsaw. She died in 1934.", "She died in 1934.", "Marie Curie died in 1934."),
    ("Northwind Robotics reported revenue of $412 million. It employs 4,200 people.",
     "It employs 4,200 people.", "Northwind Robotics employs 4,200 people."),
    ("The astronauts collected samples. They returned on July 24, 1969.",
     "They returned on July 24, 1969.", "The astronauts returned on July 24, 1969."),
])
def test_leading_pronoun_is_resolved_for_checking(extract, response, pronoun_claim, checked_as):
    claim = extract(response)[-1]
    assert claim.text == pronoun_claim           # shown as written
    assert claim.resolved_text == checked_as     # checked with the referent


def test_pronoun_without_a_known_referent_is_left_alone(extract):
    claim = extract("She died in 1934.")[0]
    assert claim.resolved_text == claim.text


def test_markdown_lists_headings_and_emphasis(extract):
    response = "# Summary\n- Python was created by Guido van Rossum.\n- **Python 3.0** was released in 2008.\n"
    claims = extract(response)
    assert texts(claims) == ["Python was created by Guido van Rossum.", "Python 3.0 was released in 2008."]


def test_duplicate_claims_are_marked(extract):
    claims = extract("The astronauts collected samples. The mission ended in July. The astronauts collected samples.")
    assert [c.duplicate_of for c in claims] == [None, None, 1]


def test_spans_index_the_original_response(extract):
    response = "Intro line.\n\n- Marie Curie discovered polonium and won two Nobel Prizes."
    claims = extract(response)
    assert response[claims[-2].start:claims[-2].end] == "Marie Curie discovered polonium"
    assert response[claims[-1].start:claims[-1].end] == "won two Nobel Prizes"
    assert [c.id for c in claims] == list(range(1, len(claims) + 1))


# ------------------------------------------------------------------ openers and pleasantries
@pytest.mark.parametrize("response, expected", [
    ("In conclusion, the Apollo 11 crew landed on the Moon in 1972.", "The Apollo 11 crew landed on the Moon in 1972."),
    ("Sure, Python was released in 1991 by Guido van Rossum.", "Python was released in 1991 by Guido van Rossum."),
    ("Of course, the Eiffel Tower is 330 meters tall.", "The Eiffel Tower is 330 meters tall."),
    ("Overall, the company's revenue grew 20 percent in 2023.", "The company's revenue grew 20 percent in 2023."),
    ("In summary, Marie Curie won two Nobel Prizes.", "Marie Curie won two Nobel Prizes."),
])
def test_opener_is_cut_but_the_fact_after_it_is_kept(extract, response, expected):
    claims = extract(response)
    assert texts(claims) == [expected]
    assert response[claims[0].start:claims[0].end].lower() == expected.lower()     # the span skips the opener


def test_opener_followed_by_a_pleasantry_is_still_dropped(extract):
    assert extract("Sure, here is the answer. I hope this helps!") == []
    assert texts(extract("Here are the facts: Paris is the capital of France.")) == [
        "Here are the facts: Paris is the capital of France."]


def test_opener_keeps_pronoun_resolution_and_splitting(extract):
    claims = extract("Marie Curie was born in Warsaw. Overall, she discovered polonium and won two Nobel Prizes.")
    assert [c.resolved_text for c in claims[1:]] == ["Marie Curie discovered polonium.", "Marie Curie won two Nobel Prizes."]


@pytest.mark.parametrize("response", ['“Did Curie win two prizes?”', 'Is it true that Curie won two prizes?”',
                                      '"Did Curie win two prizes?"', "(Did Curie win two prizes?)"])
def test_question_ending_in_a_closing_quote_is_dropped(extract, response):
    assert extract(response) == []


def test_no_extra_period_after_a_closing_quote(extract):
    assert texts(extract('The motto of the mission was "We came in peace."')) == [
        'The motto of the mission was "We came in peace."']


# ------------------------------------------------------------------ coordination
def test_three_coordinated_verbs_give_three_claims(extract):
    claims = extract("The company designs chips, builds servers, and sells cloud services.")
    assert texts(claims) == ["The company designs chips.", "The company builds servers.", "The company sells cloud services."]
    response = "Curie isolated radium, studied its properties, and won the Nobel Prize in 1911."
    claims = extract(response)
    assert texts(claims) == ["Curie isolated radium.", "Curie studied its properties.", "Curie won the Nobel Prize in 1911."]
    assert [response[c.start:c.end] for c in claims] == [
        "Curie isolated radium", "studied its properties", "won the Nobel Prize in 1911"]


def test_short_clause_with_its_own_subject_is_kept(extract):
    assert texts(extract("Armstrong walked first, and Aldrin followed.")) == ["Armstrong walked first.", "Aldrin followed."]
    assert extract("Yes, exactly. Thanks, Bob.") == []


@pytest.mark.parametrize("response, expected", [
    ("Approved warranty claims are repaired or replaced free of charge.",
     ["Approved warranty claims are repaired.", "Approved warranty claims are replaced free of charge."]),
    ("Everyone was recruited from one campus and tested in a quiet lab.",
     ["Everyone was recruited from one campus.", "Everyone was tested in a quiet lab."]),
    ("You can begin the return online and get a free label.",
     ["You can begin the return online.", "You can get a free label."]),
    ("The company was founded in 1999 and went public in 2005.",       # no shared auxiliary: different tense
     ["The company was founded in 1999.", "The company went public in 2005."]),
    ("The paper was submitted in March and accepted in July.",         # tagged VBD, but a participle
     ["The paper was submitted in March.", "The paper was accepted in July."]),
    # A past tense that rarely takes an object is not a participle: no "was died".
    ("Albert Einstein was born in Ulm in 1879 and died in Princeton in 1955.",
     ["Albert Einstein was born in Ulm in 1879.", "Albert Einstein died in Princeton in 1955."]),
    ("She was elected in 2008 and retired in 2016.", ["She was elected in 2008.", "She retired in 2016."]),
    ("The rover was launched in 2020 and landed on Mars in 2021.",
     ["The rover was launched in 2020.", "The rover landed on Mars in 2021."]),
    ("The species was discovered in 1902 and survived until 1936.",
     ["The species was discovered in 1902.", "The species survived until 1936."]),
    ("The patient was admitted on Monday and improved over the week.",
     ["The patient was admitted on Monday.", "The patient improved over the week."]),
    ("The company was founded in 1998 and expanded rapidly in Asia.",
     ["The company was founded in 1998.", "The company expanded rapidly in Asia."]),
])
def test_coordinated_verb_shares_the_auxiliary(extract, response, expected):
    assert texts(extract(response)) == expected


@pytest.mark.parametrize("response, expected", [
    ("The company did not raise prices or cut jobs in 2023.",
     ["The company did not raise prices.", "The company did not cut jobs in 2023."]),
    ("Patients should not drive or operate machinery.", ["Patients should not drive.", "Patients should not operate machinery."]),
    ("The bridge will not be repaired or replaced until 2030.",
     ["The bridge will not be repaired.", "The bridge will not be replaced until 2030."]),
    ("The vaccine was never licensed or sold in Europe.",
     ["The vaccine was never licensed.", "The vaccine was never sold in Europe."]),
    ("Sales shouldn't be delayed or cancelled.", ["Sales shouldn't be delayed.", "Sales shouldn't be cancelled."]),
    ("The museum can't be visited or photographed on Mondays.",
     ["The museum can't be visited.", "The museum can't be photographed on Mondays."]),
    ("The plant won't be expanded or sold.", ["The plant won't be expanded.", "The plant won't be sold."]),
    ("The company did not raise prices nor cut jobs.", ["The company did not raise prices.", "The company did not cut jobs."]),
    ("The company never raised prices or cut jobs.", ["The company never raised prices.", "The company never cut jobs."]),
])
def test_negation_shared_through_or_goes_with_every_piece(extract, response, expected):
    # "not A or B" means neither happened, so each piece keeps the negation (and a contraction stays whole).
    assert texts(extract(response)) == expected


@pytest.mark.parametrize("response", [
    "The company did not raise prices and cut jobs in 2023.",       # may mean the jobs were cut
    "People with type 1 diabetes do not produce enough insulin and require insulin therapy.",   # does mean it
    "The company did not raise prices but cut jobs.",
    "The company neither raised prices nor cut jobs.",
])
def test_negated_coordination_with_unclear_scope_stays_whole(extract, response):
    assert texts(extract(response)) == [response]


def test_words_after_the_and_belong_to_the_second_clause(extract):
    claims = extract("The most common side effects are dizziness, headache and a dry cough, "
                     "and in clinical studies about 3 in 100 patients stopped treatment because of the cough.")
    assert texts(claims) == ["The most common side effects are dizziness, headache and a dry cough.",
                             "In clinical studies about 3 in 100 patients stopped treatment because of the cough."]


@pytest.mark.parametrize("response, expected", [
    ("Potassium supplements are fine to use alongside Dravotine, although ibuprofen can make the medicine less effective.",
     ["Potassium supplements are fine to use alongside Dravotine.", "Ibuprofen can make the medicine less effective."]),
    ("Although the trial was small, the results were significant.",
     ["The results were significant.", "The trial was small."]),
    ("The drug is cheap, but its side effects are severe.", ["The drug is cheap.", "Its side effects are severe."]),
])
def test_contrastive_clauses_are_split(extract, response, expected):
    claims = extract(response)
    assert texts(claims) == expected
    for claim in claims:
        assert response[claim.start:claim.end].lower() == claim.text[:-1].lower()


def test_while_clause_stays_with_its_main_clause(extract):
    assert len(extract("Asia-Pacific revenue rose 22 percent while Europe was flat.")) == 1


# ------------------------------------------------------------------ relative clauses
def test_relative_clause_describes_the_noun_right_before_the_comma(extract):
    response = "For pain relief, paracetamol is usually a better choice than ibuprofen, which can make Dravotine less effective."
    claims = extract(response)
    assert texts(claims) == ["For pain relief, paracetamol is usually a better choice than ibuprofen.",
                             "Ibuprofen can make Dravotine less effective."]
    assert response[claims[1].start:claims[1].end] == "which can make Dravotine less effective"


def test_relative_clause_keeps_the_whole_noun_phrase(extract):
    assert texts(extract("Python 3.0, which was released in 2008, broke compatibility.")) == [
        "Python 3.0 broke compatibility.", "Python 3.0 was released in 2008."]
    assert texts(extract("Your blender is inside Tallowmere's 30-day return window, which runs from delivery.")) == [
        "Your blender is inside Tallowmere's 30-day return window.", "Tallowmere's 30-day return window runs from delivery."]


def test_object_relative_clause_is_split(extract):
    claims = extract("Gross margin widened to 46.2 percent from 43.8 percent a year ago, "
                     "which CFO Desmond Okafor attributed mainly to cheaper components.")
    assert texts(claims) == ["Gross margin widened to 46.2 percent from 43.8 percent a year ago.",
                             "CFO Desmond Okafor attributed this mainly to cheaper components."]
    assert texts(extract("He wrote a novel, which critics praised in 1990.")) == [
        "He wrote a novel.", "Critics praised a novel in 1990."]


@pytest.mark.parametrize("response, expected", [
    ("The report, which the agency plans to release in May, covers 2023.",
     ["The report covers 2023.", "The agency plans to release the report in May."]),
    ("The vaccine, which the company hopes to sell in Europe, costs $20.",
     ["The vaccine costs $20.", "The company hopes to sell the vaccine in Europe."]),
    ("The board approved the merger, which the regulators signed off on in March.",
     ["The board approved the merger.", "The regulators signed off on the merger in March."]),
    ("The telescope, which NASA launched in 2021, cost $10 billion.",
     ["The telescope cost $10 billion.", "NASA launched the telescope in 2021."]),
    # "which" is the subject of an inner clause: no grammatical rewrite, so the sentence stays whole.
    ("The drug, which experts believe is safe, was approved in 2019.",
     ["The drug, which experts believe is safe, was approved in 2019."]),
])
def test_object_relative_goes_back_where_it_was_taken_from(extract, response, expected):
    assert texts(extract(response)) == expected


@pytest.mark.parametrize("response, described", [
    ("Paris is the capital of France, which has a population of about 68 million.",
     "France has a population of about 68 million."),
    ("He visited the museum of Paris, which is the capital of France.", "Paris is the capital of France."),
    ("The CEO of Acme, who joined in 2010, resigned in 2020.", "The CEO of Acme joined in 2010."),
    ("The University of Oxford, which was founded in 1096, is in England.",
     "The University of Oxford was founded in 1096."),
    ("He admired the works of Picasso, who was born in Malaga.", "Picasso was born in Malaga."),
    ("The price of oil, which rose 5 percent, hit a record.", "The price of oil rose 5 percent."),
])
def test_relative_clause_after_of_describes_the_right_noun(extract, response, described):
    assert texts(extract(response))[-1] == described


def test_clause_level_which_is_checked_with_the_clause_it_describes(extract):
    response = "Revenue rose 10% in 2023, which beat expectations. Operating costs rose 30% in 2023, which beat expectations."
    claims = extract(response)
    assert texts(claims) == ["Revenue rose 10% in 2023.", "This beat expectations.",
                             "Operating costs rose 30% in 2023.", "This beat expectations."]
    assert [c.resolved_text for c in claims[1::2]] == ["Revenue rose 10% in 2023, which beat expectations.",
                                                       "Operating costs rose 30% in 2023, which beat expectations."]
    assert [c.duplicate_of for c in claims] == [None, None, None, None]     # different facts, not duplicates


def test_false_clause_level_which_is_contradicted(detector):
    report = detector.detect(
        "Revenue rose 10% in 2023, which beat expectations. Operating costs rose 30% in 2023, which beat expectations.",
        "Revenue rose 10% in 2023, which beat analyst expectations. Operating costs rose 30% in 2023, far above the "
        "12% analysts had forecast, which missed expectations.")
    assert [v.label for v in report.verdicts] == [SUPPORTED, SUPPORTED, SUPPORTED, CONTRADICTED]


# ------------------------------------------------------------------ pronouns
@pytest.mark.parametrize("response, checked_as", [
    ("Marie Curie was born in Warsaw, and she died in 1934.", "Marie Curie died in 1934."),
    ("The company was founded in 1999, and it went public in 2005.", "The company went public in 2005."),
    ("Rust is fast and it is memory safe.", "Rust is memory safe."),
    ("Marie Curie was born in Warsaw. In 1903, she won the Nobel Prize.", "In 1903, Marie Curie won the Nobel Prize."),
])
def test_pronoun_is_resolved_by_a_subject_in_the_same_sentence(extract, response, checked_as):
    assert extract(response)[-1].resolved_text == checked_as


@pytest.mark.parametrize("response", [
    "The FDA approved Zestril in 2019, and it became the best-selling blood pressure drug in 2021.",
    "Marie Curie married Pierre Curie in 1895, and he died in 1906.",
    "The FDA approved Zestril in 2019. It became the best-selling blood pressure drug in 2021.",
    "Marie Curie married Pierre Curie in 1895. He died in 1906.",
    "Apple acquired Beats in 2014, and it was founded in 2006.",
])
def test_pronoun_that_could_mean_a_named_object_is_left_alone(extract, response):
    claim = extract(response)[-1]
    assert claim.resolved_text == claim.text      # guessing the subject would check a different fact


def test_named_object_does_not_block_a_pronoun_before_it_or_a_newer_subject(extract):
    assert extract("Marie Curie was born in Warsaw. She married Pierre Curie in 1895.")[-1].resolved_text == \
        "Marie Curie married Pierre Curie in 1895."
    assert extract("The FDA approved Zestril in 2019. Zestril sold well, and it became popular.")[-1].resolved_text == \
        "Zestril became popular."


@pytest.mark.parametrize("second", ["It is estimated that Amazon has 1.5 million employees.",
                                    "It is true that Amazon has 1.5 million employees.",
                                    "It seems that Amazon has 1.5 million employees.",
                                    "It is important to note that Amazon has 1.5 million employees."])
def test_dummy_it_is_not_resolved(extract, second):
    claim = extract("Amazon was founded in 1994. " + second)[-1]
    assert claim.resolved_text == claim.text == second


def test_referential_it_with_a_that_clause_is_resolved(extract):
    assert extract("Amazon was founded in 1994. It said that sales rose.")[-1].resolved_text == "Amazon said that sales rose."


# ------------------------------------------------------------------ markdown and whitespace
def test_unicode_bullets_are_list_markers(extract):
    response = "◦ The Eiffel Tower is 330 meters tall.\n▪ It was completed in 1889.\n– It is in Paris."
    claims = extract(response)
    assert texts(claims) == ["The Eiffel Tower is 330 meters tall.", "It was completed in 1889.", "It is in Paris."]
    assert [c.resolved_text for c in claims[1:]] == ["The Eiffel Tower was completed in 1889.", "The Eiffel Tower is in Paris."]


def test_single_marker_emphasis_is_removed(extract):
    assert texts(extract("**Marie Curie** was born in *Warsaw* in 1867.")) == ["Marie Curie was born in Warsaw in 1867."]
    assert texts(extract("_Marie Curie_ was born in Warsaw.")) == ["Marie Curie was born in Warsaw."]
    assert texts(extract("The snake_case name is 2 * 3 in the docs.")) == ["The snake_case name is 2 * 3 in the docs."]


def test_label_value_bullets_are_claims(extract):
    claims = extract("- Revenue: $412M\r\n- Net income: $38M\r\n- Employees: 4,200\n- Step 1")
    assert texts(claims) == ["Revenue: $412M.", "Net income: $38M.", "Employees: 4,200."]


@pytest.mark.parametrize("response, expected", [
    ("Marie Curie was born in Warsaw.\nShe won two Nobel Prizes.\nShe died in 1934.",
     [("Marie Curie was born in Warsaw.", 0, 31), ("She won two Nobel Prizes.", 32, 57), ("She died in 1934.", 58, 75)]),
    ("Marie Curie was born in Warsaw.\r\nShe died in 1934.",
     [("Marie Curie was born in Warsaw.", 0, 31), ("She died in 1934.", 33, 50)]),
    ("Marie Curie was born\nin Warsaw in 1867.", [("Marie Curie was born in Warsaw in 1867.", 0, 39)]),
])
def test_line_breaks_stay_out_of_claim_text_and_spans(extract, response, expected):
    assert [(c.text, c.start, c.end) for c in extract(response)] == expected


def test_line_break_after_a_coordinated_sentence_stays_out_of_the_span(extract):
    response = "Marie Curie was born in Warsaw.\nShe discovered polonium and won two Nobel Prizes.\nShe died in 1934."
    claim = extract(response)[1]
    assert response[claim.start:claim.end] == "She discovered polonium"


def test_contractions_are_rebuilt_as_written(extract):
    claims = extract("Your doctor might raise the dose if your blood pressure hasn't come down, and the design cannot "
                     "show more.")
    assert texts(claims) == ["Your doctor might raise the dose if your blood pressure hasn't come down.",
                             "The design cannot show more."]


# ------------------------------------------------------------------ duplicates
@pytest.mark.parametrize("response", [
    "Inflation was 1.5 percent in 2023. Inflation was 15 percent in 2023.",
    "The low was -89 degrees. The low was 89 degrees.",
    "The unemployment rate rose to 45% in March. The unemployment rate rose to 4.5% in March.",
    "The company paid $1,200 per share. The company paid $12.00 per share.",
])
def test_claims_differing_only_in_a_number_are_not_duplicates(extract, response):
    assert [c.duplicate_of for c in extract(response)] == [None, None]


def test_claims_differing_only_in_punctuation_are_duplicates(extract):
    assert [c.duplicate_of for c in extract("In 1969, Apollo 11 landed on the Moon. In 1969 Apollo 11 landed on the Moon!")] \
        == [None, 1]
