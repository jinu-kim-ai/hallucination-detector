import pytest

from hallucination_detector import ClaimExtractor


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
