import json

import pytest

from hallucination_detector import SourceDocument
from hallucination_detector.sources import (
    MAX_SENTENCE_WORDS, _chunks, as_documents, load_request, load_text_sources, split_sentences,
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


def test_generated_ids_skip_ids_the_caller_chose():
    assert [d.id for d in as_documents([{"id": "source-2", "text": "a"}, {"text": "b"}])] == ["source-2", "source-2#2"]
    assert [d.id for d in as_documents(["a", {"id": "source-1", "text": "b"}])] == ["source-1#2", "source-1"]
    assert [d.id for d in as_documents([{"id": 0, "text": "a"}, {"id": "source-1", "text": "b"}])] == ["0", "source-1"]


@pytest.mark.parametrize("bad", [
    {"id": "s", "text": 1911}, {"id": "s", "text": ["x"]}, {"id": "s", "text": "ok", "title": 5}, 42,
])
def test_non_string_source_fields_are_rejected(bad):
    with pytest.raises(ValueError, match="must be a string"):
        as_documents([bad])


def test_none_source_is_blank_not_the_word_none():
    assert as_documents([None, {"id": "b", "text": None}]) == [SourceDocument("source-1", ""), SourceDocument("b", "")]


def test_source_documents_get_the_same_normalization_as_dicts():
    assert as_documents([SourceDocument("s", None)]) == [SourceDocument("s", "")]
    assert [d.id for d in as_documents([SourceDocument("", "a"), {"text": "b"}])] == ["source-1", "source-2"]
    assert [d.id for d in as_documents([SourceDocument("", "a"), SourceDocument("", "b")])] == ["source-1", "source-2"]
    with pytest.raises(ValueError, match="must be a string"):
        as_documents([SourceDocument("s", "x", 5)])


def test_source_document_with_no_text_does_not_crash_detect(detector_no_nli):
    report = detector_no_nli.detect("Marie Curie was born in Warsaw in 1867.", [SourceDocument("s", None)])
    assert report.sources == [SourceDocument("s", "")]
    assert any("no source text" in w for w in report.warnings)


@pytest.mark.parametrize("payload, message", [
    ("my response text", "JSON object"),
    (["response"], "JSON object"),
    ({"response": None, "sources": []}, "'response' must be a string"),
    ({"response": "x"}, "missing required field 'sources'"),
    ({"response": "x", "sources": {"id": "a", "text": "t"}}, "'sources' must be a list"),
    ({"response": "x", "sources": [{"id": "s", "text": 1911}]}, "'text' must be a string"),
])
def test_load_request_rejects_bad_field_types(tmp_path, payload, message):
    req = tmp_path / "request.json"
    req.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message) as err:
        load_request(req)
    assert str(req) in str(err.value)


def test_load_request_reports_invalid_json_as_value_error(tmp_path):
    req = tmp_path / "request.json"
    req.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_request(req)


def test_files_with_a_bom_load_without_it(tmp_path):
    req = tmp_path / "request.json"
    req.write_bytes(b"\xef\xbb\xbf" + json.dumps({"response": "R", "sources": []}).encode())
    assert load_request(req) == ("R", [])
    (tmp_path / "bom.txt").write_bytes(b"\xef\xbb\xbfMarie Curie won the prize in 1911.")
    assert load_text_sources([tmp_path / "bom.txt"])[0].text == "Marie Curie won the prize in 1911."


def test_crlf_text_files_keep_their_line_endings_so_spans_index_the_file(tmp_path, nlp):
    raw = "Line one is about Paris.\r\nMarie Curie won the Nobel Prize in Chemistry in 1911.\r\n"
    (tmp_path / "crlf.txt").write_bytes(raw.encode())
    doc = load_text_sources([tmp_path / "crlf.txt"])[0]
    assert doc.text == raw
    sentences, _ = split_sentences(doc, nlp)
    assert [raw[s.start:s.end] for s in sentences] == [s.text for s in sentences]
    assert sentences[-1].text == "Marie Curie won the Nobel Prize in Chemistry in 1911."


def test_text_file_ids_are_unique_even_when_names_and_paths_repeat(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "notes.txt").write_text("A doc here.", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("B doc here.", encoding="utf-8")
    docs = load_text_sources(["a/notes.txt", "notes.txt", "notes.txt"])
    assert [d.id for d in docs] == ["notes.txt", "notes.txt#2", "notes.txt#3"]
    assert as_documents(docs) == docs


@pytest.mark.parametrize("text", ["﻿", "​", "...", " ​\n"])
def test_text_without_any_word_character_is_blank(nlp, text):
    assert split_sentences(SourceDocument("blank", text), nlp) == ([], [])


def test_document_of_short_sentences_is_grouped_below_the_word_limit(nlp):
    text = " ".join("Item one." for _ in range(200))
    sentences, warnings = split_sentences(SourceDocument("M", text), nlp)
    assert len(sentences) > 1 and warnings
    assert all(len(s.text.split()) <= MAX_SENTENCE_WORDS for s in sentences)
    assert all(text[s.start:s.end] == s.text for s in sentences)
    assert sum(len(s.text.split()) for s in sentences) == 400


def test_limits_can_be_passed_in(nlp):
    text = "The report lists " + ", ".join(f"item number {i} of the inventory" for i in range(10)) + "."
    sentences, warnings = split_sentences(SourceDocument("long", text), nlp, max_words=20)
    assert len(sentences) > 1 and warnings and all(len(s.text.split()) <= 20 for s in sentences)
    assert split_sentences(SourceDocument("short", "Paris is big. Rome is old and big."), nlp, min_words=5)[0][0].text \
        == "Rome is old and big."


def test_text_over_spacy_max_length_is_parsed_in_chunks(nlp, apollo, monkeypatch):
    doc = SourceDocument("apollo", "\n\n".join([apollo] * 3))
    expected, _ = split_sentences(doc, nlp)
    monkeypatch.setattr(nlp, "max_length", 300)     # far below the document's length
    sentences, _ = split_sentences(doc, nlp)
    assert [(s.start, s.end, s.text) for s in sentences] == [(s.start, s.end, s.text) for s in expected]
    assert all(doc.text[s.start:s.end] == s.text for s in sentences)


def test_chunks_cover_the_text_and_respect_the_limit():
    for text in ["x" * 25, "word " * 40, "One. Two three.\nFour five six.\n\nSeven eight. " * 10]:
        chunks = _chunks(text, 30)
        assert "".join(c for _, c in chunks) == text
        assert all(len(c) <= 30 and text[o:o + len(c)] == c for o, c in chunks)
