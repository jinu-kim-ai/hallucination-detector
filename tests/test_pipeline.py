"""End-to-end behaviour: the four labels, multiple documents, citations, and degenerate input."""
from dataclasses import replace

import pytest

from hallucination_detector import (
    CONTRADICTED, FABRICATED, SUPPORTED, UNSUPPORTED, HallucinationDetector,
)
from hallucination_detector.config import Config
from hallucination_detector.pipeline import _digit_numbers, _looks_english
from hallucination_detector.schemas import SourceDocument

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


def test_claim_without_related_evidence_cites_nothing(detector, apollo):
    # an unrelated best sentence is not evidence, but its similarity is still reported
    v = detector.detect("The crew felt nervous.", apollo).verdicts[0]
    assert v.label == UNSUPPORTED and v.citations == []
    assert v.reasons[0].startswith("best evidence similarity 0.")
    assert v.to_dict()["citations"] == []


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


@pytest.mark.parametrize("fixture", ["detector", "detector_no_nli"])
def test_number_word_in_another_source_is_not_a_conflict(request, fixture):
    # the blog never gives a year; "two researchers" is not a rival to "1921"
    docs = [{"id": "encyclopedia", "text": "Insulin was first isolated in 1921 by Banting and Best."},
            {"id": "blog", "text": "Insulin was first isolated by Banting and Best, two Canadian researchers."}]
    report = request.getfixturevalue(fixture).detect("Insulin was first isolated in 1921 by Banting and Best.", docs)
    assert report.verdicts[0].label == SUPPORTED
    assert not any("sources conflict" in w for w in report.warnings)


@pytest.mark.parametrize("fixture", ["detector", "detector_no_nli"])
def test_sources_that_disagree_with_each_other_raise_a_warning(request, fixture):
    # both sources contradict the claim, and they also give different years from each other
    docs = [{"id": "encyclopedia", "text": "Insulin was first isolated in 1921 by Banting and Best."},
            {"id": "blog", "text": "Insulin was first isolated in 1925 by Banting and Best."}]
    report = request.getfixturevalue(fixture).detect("Insulin was first isolated in 1923 by Banting and Best.", docs)
    assert report.verdicts[0].label == CONTRADICTED
    assert report.warnings == ["sources conflict on claim 1: 'encyclopedia' and 'blog' disagree with each other"]


@pytest.mark.parametrize("text, expected", [
    ("$4.5 billion", {4.5e9}), ("$4.5B", {4.5e9}), ("1.2 million people", {1.2e6}), ("3.2 thousand", {3200}),
    ("two thousand people", set()), ("3 million and two thousand", {3e6}), ("two Canadian researchers", set()),
])
def test_digit_numbers_keep_their_scale_word(text, expected):
    assert _digit_numbers(text) == expected


SCALE_CLAIM = "Acme reported full-year revenue of $4.5 billion in 2025."


def test_sources_that_differ_in_scale_word_raise_a_warning(detector_no_nli):
    # '$4.5 billion' and '$4.5 million' were both read as 4.5, so the conflict went unnoticed
    docs = [{"id": "press", "text": SCALE_CLAIM}, {"id": "blog", "text": SCALE_CLAIM.replace("billion", "million")}]
    report = detector_no_nli.detect(SCALE_CLAIM, docs)
    assert report.warnings == ["sources conflict on claim 1: 'press' supports it, 'blog' contradicts it"]


@pytest.mark.parametrize("fixture", ["detector", "detector_no_nli"])
@pytest.mark.parametrize("claim, census, atlas", [
    ("The city had 1.2 million people in 2020.", "The city had 1.2 million people in 2020.",
     "The city had 1,200,000 people in 2020."),
    ("Acme reported revenue of $1.62 billion in 2025.", "Acme reported revenue of $1.62 billion in 2025.",
     "Acme reported revenue of $1,620 million in 2025."),
])
def test_sources_agreeing_in_other_notation_are_no_conflict(request, fixture, claim, census, atlas):
    docs = [{"id": "census", "text": census}, {"id": "atlas", "text": atlas}]
    report = request.getfixturevalue(fixture).detect(claim, docs)
    assert not any("sources conflict" in w for w in report.warnings)


FELL = "Acme's revenue fell to $2.1 billion in 2023."
ROSE_OTHER_NOTATION = "Acme's revenue rose to $2,100 million in 2023."
ROSE = "Acme's revenue rose to $2.1 billion in 2023."
DOWN = "The town had a population of 1.2 million in 2020, down from the previous census."
UP_OTHER_NOTATION = "The town had a population of 1,200,000 in 2020, up from the previous census."
UP = "The town had a population of 1.2 million in 2020, up from the previous census."


@pytest.mark.parametrize("claim, first, second", [
    (FELL, ROSE_OTHER_NOTATION, ROSE), (FELL, ROSE, ROSE_OTHER_NOTATION), (DOWN, UP_OTHER_NOTATION, UP),
])
def test_source_contradicting_in_other_notation_is_no_supporter(detector, claim, first, second):
    # both sources contradict the claim ('rose' vs 'fell'); the one in another notation was called a supporter
    report = detector.detect(claim, [{"id": "a", "text": first}, {"id": "b", "text": second}])
    assert report.verdicts[0].label == CONTRADICTED
    assert report.warnings == []


def test_source_agreeing_in_other_notation_still_conflicts_with_a_reversed_one(detector):
    report = detector.detect(ROSE, [{"id": "a", "text": ROSE_OTHER_NOTATION}, {"id": "b", "text": FELL}])
    assert report.warnings == ["sources conflict on claim 1: 'a' supports it, 'b' contradicts it"]


@pytest.mark.parametrize("text, expected", [
    ("a $5-million grant in 2022", {5e6, 2022}), ("a $5 million grant in 2022", {5e6, 2022}),
    ("from 1990-1995", {1990, 1995}),
    # a dash, unlike a hyphen, does not join a year to the scale word of the next phrase
    ("In 2019—million-unit launches were common", {2019}),
    ("Acme raised $2 billion in 2021—billion-dollar rounds were common that year.", {2e9, 2021}),
    ("In 2019–million-unit launches were common", {2019}),
])
def test_digit_numbers_read_a_hyphenated_scale_word(text, expected):
    assert _digit_numbers(text) == expected


def test_hyphenated_scale_word_in_a_source_is_no_conflict(detector_no_nli):
    claim = "The foundation awarded a $5 million grant to the university in 2022."
    docs = [{"id": "globe", "text": "In 2022 the foundation awarded the university a $5-million grant."},
            {"id": "release", "text": "The foundation awarded a $5 million grant to the university in 2022."}]
    report = detector_no_nli.detect(claim, docs)
    assert report.verdicts[0].label == SUPPORTED
    assert report.warnings == []


def test_year_before_an_em_dash_keeps_a_sources_conflict(detector_no_nli):
    # '2021—billion-dollar' was read as 2.021e12, so source a lost its 2021 and the conflict went unreported
    docs = [{"id": "a", "text": "Acme raised $2 billion in 2021—billion-dollar rounds were common that year."},
            {"id": "b", "text": "Acme raised $3 billion in 2021."}]
    report = detector_no_nli.detect("Acme raised $2 billion in 2021.", docs)
    assert report.warnings == ["sources conflict on claim 1: 'a' supports it, 'b' contradicts it"]


def test_fact_spread_over_two_sentences(detector, monkeypatch):
    source = "Marie Curie was born in 1867. She was born in Warsaw, Poland."
    claim = "Marie Curie was born in Warsaw in 1867."
    v = detector.check_claim(claim, source)
    assert v.label == SUPPORTED
    assert {c.sentence_index for c in v.citations} == {0, 1}          # each sentence used is cited
    # with neighbor sentences in the NLI premise, the joined fact is entailed outright
    monkeypatch.setattr(detector, "config", replace(detector.config, nli_context_sentences=1))
    v = detector.check_claim(claim, source)
    assert v.label == SUPPORTED and v.citations[0].nli["entailment"] >= detector.config.nli_threshold


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


@pytest.mark.parametrize("response", [
    "아폴로 11호는 1969년 7월 16일 케네디 우주 센터에서 발사되었습니다.",
    "아폴로 11호는 1969년 발사",                                                      # short: 8 letters
    "Apollo 11 a été lancé depuis le centre spatial Kennedy le 16 juillet 1969.",
    "El Apolo 11 fue lanzado desde el Centro Espacial Kennedy el 16 de julio de 1969. "
    "Los astronautas recogieron 45 kilogramos de muestras.",
    "Die Astronauten sammelten etwa 21,5 Kilogramm Mondgestein.",
    "A Apollo 11 foi lançada do Centro Espacial Kennedy em 16 de julho de 1969.",
    "Gli astronauti hanno raccolto circa 21,5 chilogrammi di campioni lunari.",
    "De astronauten verzamelden ongeveer 21,5 kilogram maanstenen.",
])
def test_non_english_response_is_flagged_not_scored(detector, apollo, response):
    report = detector.detect(response, apollo)
    assert report.reliability_score is None and report.unscorable and report.verdicts == []
    assert any("English" in w for w in report.warnings)


@pytest.mark.parametrize("text", [
    "Apollo 11 launched in 1969.",
    "Is that right?",
    "Pelé scored 1000 goals for Santos FC in Brazil between 1956 and 1974.",
    "Müller won.",
    "The café on Rue de la Paix serves crème brûlée.",
    "Pedro de la Rosa drove for Arrows, Jaguar and McLaren.",
    "Die Hard was released in 1988 starring Bruce Willis.",
    "Real Madrid won La Liga in 2022 under Carlo Ancelotti.",
    "Apple Inc. CEO Tim Cook announced iPhone 15 Pro Max pricing.",
    "Q3 revenue rose 14% to $412M; EPS $1.32 vs $1.10.",
    "Angela Merkel, Emmanuel Macron, Pedro Sánchez and Giorgia Meloni met in Brussels on 12 March 2023.",
    # function words inside titles and names are not the language of the answer
    "Gabriel García Márquez wrote Cien años de soledad and El amor en los tiempos del cólera.",
    "Victor Hugo wrote Les Misérables and Notre-Dame de Paris. Albert Camus wrote L'Étranger and La Peste. "
    "Antoine de Saint-Exupéry wrote Le Petit Prince.",
    "La La Land won six Oscars in 2017.",
    "Antoine de Saint-Exupéry wrote Le Petit Prince.",
    "Simone de Beauvoir wrote Le Deuxième Sexe in 1949.",
    "Leonardo da Vinci painted La Gioconda.",
    "Johann Wolfgang von Goethe wrote Die Leiden des jungen Werthers at the age of 24.",
    "Antoine de Saint-Exupéry wrote Vol de nuit.",
    'Márquez wrote "Cien años de soledad" in 1967.',
    "- Real Madrid won La Liga in 2022\n- The club was coached by Carlo Ancelotti\n- It also won the Champions League",
    "The museum is in Den Haag and opened in 1902.",
    # no English-only function word anywhere, one foreign article per title (round-three review)
    "Antoine de Saint-Exupéry wrote Le Petit Prince.\nVictor Hugo wrote Les Misérables.",
    "- Thomas Mann wrote Der Zauberberg\n- Günter Grass wrote Die Blechtrommel",
    "Albert Camus wrote La Peste. Victor Hugo wrote Les Misérables.",
    "Federico Fellini directed La Dolce Vita.",
    "Édith Piaf recorded La Vie en rose.",
    "Pedro Almodóvar directed Todo sobre mi madre.",
    "Wagner composed Der Ring des Nibelungen.",
    "Fritz Lang directed Das Testament des Dr. Mabuse.",
    "Umberto Eco wrote Il nome della rosa.",
    "Godard directed À bout de souffle.",
    "Mozart composed Le nozze di Figaro.",
    "Johann Wolfgang von Goethe wrote Die Leiden des jungen Werthers.",
    "La La Land won six Oscars.",
])
def test_english_with_names_numbers_or_accents_counts_as_english(text):
    assert _looks_english(text)


@pytest.mark.parametrize("text", [
    "Marie Curie nació en Varsovia en 1867.",
    "La Torre Eiffel fue construida en 1889.",
    "Le Petit Prince est un livre de Saint-Exupéry.",
    "Die Mission startete vom Kennedy Space Center und landete im Meer der Ruhe.",
    # a capitalized word in the first line used to start a 'title' that ran across the following lines
    "- Het museum ligt in Den Haag\n- Het werd gebouwd door een bekende architect\n- Er komen elk jaar veel bezoekers",
    "El Real Madrid ganó La Liga\n- Los jugadores celebraron en la plaza\n- El entrenador renovó su contrato",
])
def test_other_language_that_starts_with_a_name_or_title_is_not_english(text):
    assert not _looks_english(text)


@pytest.mark.xfail(strict=True, reason="documented limit of _in_names: inside one sentence a capitalized "
                                       "foreign word starts a title that runs to the sentence end")
@pytest.mark.parametrize("text", [
    "Am Ende gewann die Mannschaft das Finale mit drei Toren.",
    "Zur Zeit arbeiten dort etwa 300 Menschen.",
    "Zum Teil lag das an den hohen Kosten für Energie.",
    "Zum Beispiel hat die Firma im letzten Jahr rund 5 Millionen Euro verdient.",
    "El Real Madrid ganó La Liga tras una temporada muy complicada.",
    "El Real Madrid ganó La Liga tras una temporada muy complicada, según The Athletic.",
    "Am Ende gewann die Mannschaft das Finale mit drei Toren, berichtete The Guardian.",
    "Zum Beispiel hat die Bank of America im letzten Jahr rund 5 Millionen Euro verdient.",
])
def test_documented_limit_one_foreign_sentence_with_a_capitalized_word_inside(text):
    # Telling 'ganó La Liga tras una ...' from 'wrote El amor en los tiempos ...' needs more than function
    # words. A gate on 'any English function word in the response' left English answers such as 'Umberto Eco
    # wrote Il nome della rosa.' unscored and was still bypassed by an English name ('The Athletic'), so it
    # was removed: scoring these few foreign answers is preferred to leaving English answers unscored.
    assert not _looks_english(text)


@pytest.mark.parametrize("response, source", [
    ("Antoine de Saint-Exupéry wrote Le Petit Prince.\nVictor Hugo wrote Les Misérables.",
     "Antoine de Saint-Exupéry wrote Le Petit Prince in 1943. Victor Hugo wrote Les Misérables in 1862."),
    ("- Thomas Mann wrote Der Zauberberg\n- Günter Grass wrote Die Blechtrommel",
     "Thomas Mann wrote Der Zauberberg in 1924. Günter Grass wrote Die Blechtrommel in 1959."),
])
@pytest.mark.parametrize("fixture", ["detector", "detector_no_nli"])
def test_short_english_answer_naming_foreign_titles_is_scored(request, fixture, response, source):
    report = request.getfixturevalue(fixture).detect(response, [source])
    assert not report.unscorable
    assert [v.label for v in report.verdicts] == [SUPPORTED, SUPPORTED]


def test_english_response_naming_spanish_works_is_scored(detector_no_nli):
    response = "Gabriel García Márquez wrote Cien años de soledad and El amor en los tiempos del cólera."
    source = "Gabriel García Márquez wrote Cien años de soledad (1967) and El amor en los tiempos del cólera (1985)."
    report = detector_no_nli.detect(response, [source])
    assert not report.unscorable and report.verdicts and report.reliability_score == 100.0


def test_language_check_reads_its_limits_from_the_config():
    text = "Marie Curie nació en Varsovia en 1867."                    # 2 Spanish function words, 0 English
    assert not _looks_english(text)
    assert _looks_english(text, Config(min_foreign_function_words=3))
    korean = "아폴로 11호는 1969년 발사"
    assert not _looks_english(korean)
    assert _looks_english(korean, Config(min_non_ascii_letters=20))
    assert _looks_english(korean, Config(min_ascii_letter_ratio=0.0))


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


def test_sentence_cache_is_a_bounded_lru(nlp):
    detector = HallucinationDetector(nlp=nlp, use_nli=False, embedder="hashing", cache_size=2)
    a, b, c = (SourceDocument(i, f"Document {i} says the bridge opened in 19{n}0.") for n, i in enumerate("abc"))
    for doc in (a, b, a, c):                       # reading `a` again makes `b` the least recently used
        detector._split(doc)
    assert [key[0] for key in detector._sentence_cache] == ["a", "c"]
    assert detector.retriever.cache_size == 2


def test_uncited_claim_keeps_its_best_similarity(detector, apollo):
    verdict = detector.detect("The astronauts felt very nervous during the descent.", apollo).verdicts[0]
    assert verdict.citations == [] and 0 < verdict.best_similarity < detector.config.related_threshold
    assert verdict.to_dict()["best_similarity"] == round(verdict.best_similarity, 3)


def test_report_records_its_configuration(detector, detector_no_nli, apollo):
    assert detector.detect("Apollo 11 launched in 1969.", apollo).config["use_nli"] is True
    report = detector_no_nli.detect("Apollo 11 launched in 1969.", apollo)
    assert report.config["use_nli"] is False and report.config["nli_model"] is None
    assert report.verdicts[0].citations[0].nli is None


@pytest.mark.parametrize("source, claim", [
    ("The company reported revenue of $2,100 million for the year.", "The company reported revenue of $2.1 billion for the year."),
    ("The tower has 21 floors.", "The tower has twenty-one floors."),
])
def test_number_written_differently_is_not_an_nli_contradiction(detector, source, claim):
    verdict = detector.check_claim(claim, source)
    assert verdict.label == SUPPORTED
    assert any("written differently" in r for r in verdict.reasons)


def test_different_amount_is_still_contradicted(detector):
    verdict = detector.check_claim("The company reported revenue of $3.1 billion for the year.",
                                   "The company reported revenue of $2.1 billion for the year.")
    assert verdict.label == CONTRADICTED


def test_scale_suffix_in_a_verbatim_claim_is_grounded(detector_no_nli):
    """'5M' (5 million) must not be read differently in the claim and in the cased source text."""
    source = "The app reached 5M users in 2024, up from 2.4M a year earlier."
    verdict = detector_no_nli.check_claim("The app reached 5M users in 2024.", source)
    assert verdict.label == SUPPORTED


CURIE_SOURCE = ("Marie Curie married Pierre Curie in 1895. Pierre Curie died in 1906 in a street accident in Paris. "
                "Marie Curie died in 1934.")


def test_ambiguous_pronoun_is_not_supported_by_evidence_about_someone_else(detector):
    report = detector.detect("Marie Curie married Pierre Curie in 1895. She died in 1906.", CURIE_SOURCE)
    first, second = report.verdicts
    assert first.label == SUPPORTED
    assert second.claim.resolved_text == "She died in 1906." and second.label != SUPPORTED
    assert any("unresolved pronoun" in r for r in second.reasons)


def test_resolved_pronoun_is_still_supported(detector):
    report = detector.detect("Marie Curie was born in Warsaw. She died in 1934.",
                             "Marie Curie was born in Warsaw in 1867. Marie Curie died in 1934.")
    assert [v.label for v in report.verdicts] == [SUPPORTED, SUPPORTED]


def test_ambiguous_pronoun_supported_by_evidence_about_the_subject_stays_supported(detector):
    """The guard only steps in when the supporting sentence is about the rival, not the remembered subject."""
    report = detector.detect("Marie Curie married Pierre Curie in 1895. She won a second Nobel Prize in 1911.",
                             "Marie Curie married Pierre Curie in 1895. Marie Curie won a second Nobel Prize in 1911.")
    assert [v.label for v in report.verdicts] == [SUPPORTED, SUPPORTED]


@pytest.mark.parametrize("source", [
    # the supporting sentence names neither person
    "Steve Jobs hired John Sculley in 1983. John Sculley grew up in New York. He was born in 1939, the son of a "
    "lawyer. Steve Jobs was born in 1955.",
    # the supporting sentence is about the rival and mentions the subject only in passing
    "Steve Jobs hired John Sculley in 1983. John Sculley, whom Steve Jobs recruited from Pepsi, was born in 1939.",
])
def test_ambiguous_he_is_supported_only_by_a_sentence_clearly_about_the_subject(detector, source):
    verdict = detector.detect("Steve Jobs hired John Sculley in 1983. He was born in 1939.", source).verdicts[-1]
    assert verdict.claim.ambiguous_pronoun == "he"
    assert verdict.label == UNSUPPORTED


@pytest.mark.parametrize("response, source", [
    ("The FDA approved Vioxx in 1999. It was withdrawn in 2004.",
     "The FDA approved Vioxx in 1999. Merck withdrew Vioxx from the market in 2004 over heart risks."),
    ("Microsoft bought GitHub in 2018. It hosts more than 100 million developers.",
     "Microsoft bought GitHub in 2018. GitHub hosts more than 100 million developers."),
    # function words such as "the" do not count as part of a name
    ("The Federal Reserve hired Goldman Sachs in 2019. It was founded in 1869.",
     "The Federal Reserve hired Goldman Sachs in 2019. Goldman Sachs was founded in 1869 in the city of New York."),
])
def test_ambiguous_it_about_the_object_stays_supported(detector, response, source):
    assert detector.detect(response, source).verdicts[-1].label == SUPPORTED
