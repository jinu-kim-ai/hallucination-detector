import pytest

from hallucination_detector import CONTRADICTED, SUPPORTED, UNSUPPORTED, ClaimExtractor


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


@pytest.mark.parametrize("response, expected", [
    ("The bill was passed by the House but never signed by the president.",
     ["The bill was passed by the House.", "The bill was never signed by the president."]),
    ("The patient was admitted on Monday and never discharged.",
     ["The patient was admitted on Monday.", "The patient was never discharged."]),
    ("The bridge was built in 1990 and never repaired.", ["The bridge was built in 1990.", "The bridge was never repaired."]),
    ("The drug was tested in 2019 and never approved in Europe.",
     ["The drug was tested in 2019.", "The drug was never approved in Europe."]),
    ("The patient was admitted on Monday and not discharged until Friday.",
     ["The patient was admitted on Monday.", "The patient was not discharged until Friday."]),
    ("She will attend the meeting and never speak.", ["She will attend the meeting.", "She will never speak."]),
    # The head's own negation and a "do" stay behind: no "was not never", no "did never".
    ("The vaccine was not licensed and never sold in Europe.",
     ["The vaccine was not licensed.", "The vaccine was never sold in Europe."]),
    ("The company did not raise prices and never cut jobs.", ["The company did not raise prices.", "The company never cut jobs."]),
    ("Albert Einstein was born in Ulm and never returned to Germany.",       # still no "was never returned"
     ["Albert Einstein was born in Ulm.", "Albert Einstein never returned to Germany."]),
    # The conjunct's own "never" goes after the first auxiliary.
    ("The site has been excavated since 1990 and never fully explored.",
     ["The site has been excavated since 1990.", "The site has never been fully explored."]),
])
def test_conjunct_with_its_own_negation_shares_the_auxiliary(extract, response, expected):
    assert texts(extract(response)) == expected


@pytest.mark.parametrize("response, expected", [
    # spaCy tags the verb after "never" as a participle; a verb with an object, or one that takes none, is not passive.
    ("The vaccine was given to 5,000 people and never caused serious side effects.",
     ["The vaccine was given to 5,000 people.", "The vaccine never caused serious side effects."]),
    ("The program was launched in 2008 and never succeeded.",
     ["The program was launched in 2008.", "The program never succeeded."]),
    ("The company is based in Ohio and never moved.", ["The company is based in Ohio.", "The company never moved."]),
    ("Revenue was not affected and never declined.", ["Revenue was not affected.", "Revenue never declined."]),
    ("The patient was diagnosed in 2019 and never relapsed.",
     ["The patient was diagnosed in 2019.", "The patient never relapsed."]),
])
def test_active_conjunct_after_never_does_not_take_a_passive_auxiliary(extract, response, expected):
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
    # A place noun is not an empty head: the clause describes the country, not "a state of Germany".
    ("Bavaria is a state of Germany, which has about 83 million people.", "Germany has about 83 million people."),
    ("Lyon is a city of France, which has a population of 68 million.", "France has a population of 68 million."),
    # The verb's number rules out the inner noun: "was" is not "the founders".
    ("One of the founders, who was born in Ulm, left in 1990.", "One of the founders was born in Ulm."),
    ("One of the founders, who were all engineers, was born in Ulm.", "The founders were all engineers."),
    # A quantity head: the clause describes the whole phrase.
    ("The price of oil, which rose 5 percent, hit a record.", "The price of oil rose 5 percent."),
    ("The number of users, which grew to 2 billion in 2023, keeps rising.",
     "The number of users grew to 2 billion in 2023."),
    ("The share of renewables, which reached 30 percent in 2023, keeps rising.",
     "The share of renewables reached 30 percent in 2023."),
    # A name that spaCy labels DATE is still the noun the clause describes.
    ("Malaysia Airlines Flight 370, which disappeared in 2014, was never found.",
     "Malaysia Airlines Flight 370 disappeared in 2014."),
])
def test_relative_clause_after_of_describes_the_right_noun(extract, response, described):
    assert texts(extract(response))[-1] == described


@pytest.mark.parametrize("response, expected", [
    # Kept whole, one claim would carry two facts that the source states with separate numbers.
    ("The rate of inflation, which peaked at 9.1 percent in June 2022, fell to 3 percent by mid-2023.",
     ["The rate of inflation fell to 3 percent by mid-2023.", "The rate of inflation peaked at 9.1 percent in June 2022."]),
    ("Part of the old town, which was destroyed in 1944, was rebuilt in the 1950s.",
     ["Part of the old town was rebuilt in the 1950s.", "Part of the old town was destroyed in 1944."]),
])
def test_relative_clause_after_a_quantity_head_is_split(extract, response, expected):
    assert texts(extract(response)) == expected


@pytest.mark.parametrize("response, source", [
    ("The rate of inflation, which peaked at 9.1 percent in June 2022, fell to 3 percent by mid-2023.",
     "US inflation peaked at 9.1 percent in June 2022. By June 2023 the inflation rate had fallen to 3 percent."),
    ("Part of the old town, which was destroyed in 1944, was rebuilt in the 1950s.",
     "The old town was largely destroyed in 1944. Reconstruction of the old town took place in the 1950s."),
    ("Malaysia Airlines Flight 370, which disappeared in 2014, was never found.",
     "Malaysia Airlines Flight 370 disappeared on 8 March 2014 while flying from Kuala Lumpur to Beijing. The main "
     "wreckage was never found. Air France Flight 447 crashed in 2009; its wreckage was located in 2011."),
])
def test_true_relative_clause_facts_are_supported(detector, response, source):
    assert [v.label for v in detector.detect(response, source).verdicts] == [SUPPORTED, SUPPORTED]


def test_false_year_in_a_relative_clause_after_a_mislabelled_name_is_contradicted(detector):
    report = detector.detect(
        "Malaysia Airlines Flight 370, which disappeared in 2015, was never found.",
        "Malaysia Airlines Flight 370 disappeared on 8 March 2014 while flying from Kuala Lumpur to Beijing. The main "
        "wreckage was never found. Air France Flight 447 crashed in 2009; its wreckage was located in 2011.")
    assert [v.claim.text for v in report.verdicts][-1] == "Malaysia Airlines Flight 370 disappeared in 2015."
    assert [v.label for v in report.verdicts] == [SUPPORTED, CONTRADICTED]


@pytest.mark.parametrize("response", [
    "The company cut 800 jobs in 2023, which analysts attributed to falling sales.",
    "Revenue rose 15% in 2023, which beat expectations.",
    "She won the race in 2019, which surprised fans.",
    # Even when the clause has its own number or name, "This was up from 9% in 2022." could only be checked
    # without knowing what "this" is.
    "Revenue rose 15% in 2023, which was up from 9% in 2022.",
    "Tesla delivered 1.8 million cars in 2023, which was up 50 percent from 2022.",
    "The population of Houston grew to 2.3 million in 2020, which made it the third-largest city in the United States.",
    "Boeing built the 747, which first flew in 1972.",
])
def test_clause_level_which_stays_with_its_clause(extract, response):
    assert texts(extract(response)) == [response]


@pytest.mark.parametrize("response, expected", [
    # "46.2 percent" is not what the CFO credited, and "this" goes right after "credited", not after "lower".
    ("Gross margin improved to 46.2 percent, which CFO Marisol Atwater credited to lower component costs and a richer "
     "product mix.",
     ["Gross margin improved to 46.2 percent.",
      "CFO Marisol Atwater credited this to lower component costs and a richer product mix."]),
    ("Gross margin widened to 46.2 percent from 43.8 percent a year ago, which CFO Desmond Okafor attributed mainly to "
     "cheaper components.",
     ["Gross margin widened to 46.2 percent from 43.8 percent a year ago.",
      "CFO Desmond Okafor attributed this mainly to cheaper components."]),
    ("Revenue reached $394 billion, which Tim Cook called a disappointment.",
     ["Revenue reached $394 billion.", "Tim Cook called this a disappointment."]),
])
def test_attribution_of_a_whole_clause_is_split_but_checked_with_it(extract, response, expected):
    claims = extract(response)
    assert texts(claims) == expected
    assert claims[1].resolved_text == response      # "this" is only checkable together with what it stands for


def test_attributions_from_different_sentences_are_not_duplicates(extract):
    claims = extract("Sales rose, which CFO Dana Lee credited to Asia. Costs fell, which CFO Dana Lee credited to Asia.")
    assert texts(claims)[1] == texts(claims)[3] == "CFO Dana Lee credited this to Asia."
    assert [c.duplicate_of for c in claims] == [None, None, None, None]


@pytest.mark.parametrize("response, source, label", [
    ("Revenue rose 15% in 2023, which was up from 9% in 2022.",
     "Revenue rose 15% in 2023, after growing 12% in 2022. Operating margin was 9% in 2022.", CONTRADICTED),
    ("Tesla delivered 1.8 million cars in 2023, which was up 50 percent from 2022.",
     "Tesla delivered 1.81 million vehicles in 2023, up 38 percent from 2022. Revenue grew 50 percent in 2022.",
     CONTRADICTED),
    ("The population of Houston grew to 2.3 million in 2020, which made it the third-largest city in the United States.",
     "Houston had a population of 2.3 million in 2020, making it the largest city in Texas and the fourth-largest in "
     "the United States. Chicago is the third-largest city in the United States.", CONTRADICTED),
    ("Boeing built the 747, which first flew in 1972.",
     "The Boeing 747 first flew in February 1969. The DC-10 first flew in 1970 and the L-1011 TriStar first flew in "
     "1970. The Airbus A300 first flew in 1972.", CONTRADICTED),
])
def test_false_clause_level_which_is_not_supported(detector, response, source, label):
    assert [v.label for v in detector.detect(response, source).verdicts] == [label]


def test_false_attribution_of_a_whole_clause_is_not_supported(detector):
    report = detector.detect(
        "Revenue reached $394 billion, which Tim Cook called a disappointment.",
        "Apple's revenue reached $394 billion in fiscal 2022. Tim Cook called the result a record. Cook called the "
        "decline in Mac sales a disappointment.")
    assert [v.label for v in report.verdicts] == [SUPPORTED, UNSUPPORTED]


def test_clause_level_which_keeps_a_resolved_subject(extract):
    claim = extract("Serena Williams turned pro in 1995. She won the race in 2019, which surprised fans.")[-1]
    assert claim.resolved_text == "Serena Williams won the race in 2019, which surprised fans."


@pytest.mark.parametrize("response, source", [
    ("The company cut 800 jobs in 2023, which analysts attributed to falling sales.",
     "The company cut 500 jobs in 2023. Analysts attributed the cuts to falling sales."),
    ("Revenue rose 15% in 2023, which beat expectations.", "Revenue rose 10% in 2023, which beat analyst expectations."),
])
def test_wrong_number_before_a_clause_level_which_is_counted_once(detector, response, source):
    assert [v.label for v in detector.detect(response, source).verdicts] == [CONTRADICTED]


def test_false_clause_level_which_is_contradicted(detector):
    report = detector.detect(
        "Revenue rose 10% in 2023, which beat expectations. Operating costs rose 30% in 2023, which beat expectations.",
        "Revenue rose 10% in 2023, which beat analyst expectations. Operating costs rose 30% in 2023, far above the "
        "12% analysts had forecast, which missed expectations.")
    assert [v.label for v in report.verdicts] == [SUPPORTED, CONTRADICTED]


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


@pytest.mark.parametrize("response", [
    "Marie Curie married Pierre Curie in 1895. She died in 1906.",      # either could be "she" as far as we know
    "Elon Musk leads Tesla. He founded SpaceX in 2002.",                # spaCy tags Tesla as a PERSON
    "Barack Obama defeated John McCain in 2008. He was born in Arizona.",
    "Marie Curie was born in Warsaw. She married Pierre Curie in 1895. He died in 1906.",
    # An earlier "he" that meant the subject does not show that a later one does.
    "Barack Obama was born in Hawaii. He defeated John McCain in 2008. He served as a senator from Arizona.",
    "Barack Obama was born in Hawaii. He defeated John McCain in 2008. He was born in the Panama Canal Zone.",
])
def test_person_pronoun_after_a_named_person_object_is_left_alone(extract, response):
    claim = extract(response)[-1]
    assert claim.resolved_text == claim.text


@pytest.mark.parametrize("response, source", [
    ("Barack Obama was born in Hawaii. He defeated John McCain in 2008. He served as a senator from Arizona.",
     "Barack Obama was born in Honolulu, Hawaii, in 1961. Obama defeated Republican nominee John McCain in the 2008 "
     "presidential election. John McCain served as a United States senator from Arizona from 1987 until his death in "
     "2018. Obama served as a U.S. senator from Illinois from 2005 to 2008."),
    ("Barack Obama was born in Hawaii. He defeated John McCain in 2008. He was born in the Panama Canal Zone.",
     "Barack Obama was born in Honolulu, Hawaii, in 1961. Obama defeated Republican nominee John McCain in the 2008 "
     "presidential election. McCain was born in the Panama Canal Zone in 1936."),
])
def test_true_claim_about_a_named_object_is_not_contradicted(detector, response, source):
    first, second, third = detector.detect(response, source).verdicts
    assert [first.label, second.label] == [SUPPORTED, SUPPORTED]
    # "He" may be Obama or McCain. It must never be contradicted by guessing the wrong one; since nothing says
    # which is meant, the pipeline does not let McCain's sentence support it either.
    assert third.claim.ambiguous_pronoun == "he"
    assert third.label == UNSUPPORTED and any("unresolved pronoun" in r for r in third.reasons)


@pytest.mark.parametrize("response, checked_as", [
    # No gender is remembered from an earlier (here wrong) resolution, so it cannot block a later one.
    ("Pierre Curie was born in Paris. His wife Marie was born in Warsaw. She won two Nobel Prizes. He died in 1908.",
     "Pierre Curie died in 1908."),
    ("Marie Curie worked closely with her husband, Pierre. He died in 1906. She won a second Nobel Prize in 1912.",
     "Marie Curie won a second Nobel Prize in 1912."),
])
def test_pronoun_resolution_does_not_depend_on_an_earlier_pronoun(extract, response, checked_as):
    assert extract(response)[-1].resolved_text == checked_as


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
