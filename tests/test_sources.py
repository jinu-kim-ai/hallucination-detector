import json

import pytest

from hallucination_detector import SourceDocument
from hallucination_detector.sources import (
    MAX_SENTENCE_WORDS, as_documents, load_request, load_text_sources, split_sentences,
)


def test_as_documents_accepts_strings_dicts_and_documents():
    docs = as_documents(["first text", {"id": "b", "text": "second", "title": "B"}, SourceDocument("c", "third")])
    assert [d.id for d in docs] == ["source-1", "b", "c"]
    assert docs[1].title == "B"
    assert [d.id for d in as_documents("just one string")] == ["source-1"]
    assert as_documents(None) == [] and as_documents([]) == []


def test_duplicate_source_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        as_documents([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}])


def test_sentence_spans_point_into_the_document(nlp, apollo):
    doc = SourceDocument("apollo", "  " + apollo + "\n")
    sentences, warnings = split_sentences(doc, nlp)
    assert len(sentences) == 5 and warnings == []
    assert [s.index for s in sentences] == [0, 1, 2, 3, 4]
    for s in sentences:
        assert doc.text[s.start:s.end] == s.text
        assert s.text == s.text.strip() and s.source_id == "apollo"


def test_blank_document_has_no_sentences(nlp):
    assert split_sentences(SourceDocument("empty", "   \n"), nlp) == ([], [])


def test_very_short_document_is_kept_as_one_unit(nlp):
    sentences, _ = split_sentences(SourceDocument("tiny", "Revenue: $412M"), nlp)
    assert [s.text for s in sentences] == ["Revenue: $412M"]


def test_overlong_sentence_is_split_with_a_warning(nlp):
    text = "The report lists " + ", ".join(f"item number {i} of the inventory" for i in range(60)) + "."
    sentences, warnings = split_sentences(SourceDocument("long", text), nlp)
    assert len(sentences) > 1 and warnings
    assert all(len(s.text.split()) <= MAX_SENTENCE_WORDS for s in sentences)
    assert all(text[s.start:s.end] == s.text for s in sentences)


def test_load_request_and_text_sources(tmp_path):
    req = tmp_path / "request.json"
    req.write_text(json.dumps({"response": "R", "sources": [{"id": "a", "text": "T"}]}), encoding="utf-8")
    response, docs = load_request(req)
    assert response == "R" and docs == [SourceDocument("a", "T")]

    req.write_text(json.dumps({"sources": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="response"):
        load_request(req)

    (tmp_path / "one.txt").write_text("first", encoding="utf-8")
    (tmp_path / "two.txt").write_text("second", encoding="utf-8")
    docs = load_text_sources([tmp_path / "one.txt", tmp_path / "two.txt"])
    assert [(d.id, d.text) for d in docs] == [("one.txt", "first"), ("two.txt", "second")]
