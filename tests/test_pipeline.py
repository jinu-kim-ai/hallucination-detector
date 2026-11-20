"""End-to-end behaviour, including the edge cases listed in PLAN.md section 4."""
from dataclasses import replace

import pytest

from hallucination_detector import (
    CONTRADICTED, FABRICATED, SUPPORTED, UNSUPPORTED, HallucinationDetector,
)
from hallucination_detector.pipeline import _looks_english
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
    # PLAN section 3: an unrelated best sentence is not evidence, but its similarity is still reported
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
])
def test_english_with_names_numbers_or_accents_counts_as_english(text):
    assert _looks_english(text)


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
