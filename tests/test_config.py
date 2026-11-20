import dataclasses

import pytest

from hallucination_detector import Config, HallucinationDetector
from hallucination_detector.config import V1, V1_NLI
from hallucination_detector.sources import MAX_SENTENCE_WORDS, MIN_SOURCE_WORDS


def test_defaults_and_presets_are_valid():
    assert Config().top_k == 3 and V1.use_nli is False and V1_NLI.nli_top_k == 1


@pytest.mark.parametrize("overrides, message", [
    ({"top_k": 0}, "top_k"),
    ({"top_k": -1}, "top_k"),
    ({"nli_top_k": 0}, "nli_top_k"),
    ({"min_claim_words": 0}, "min_claim_words"),
    ({"min_source_words": 0}, "min_source_words"),
    ({"max_sentence_words": 0}, "max_sentence_words"),
    ({"cache_size": 0}, "cache_size"),
    ({"nli_context_sentences": -1}, "nli_context_sentences"),
    ({"top_k": 2.5}, "top_k"),
    ({"support_threshold": 1.5}, "support_threshold"),
    ({"nli_threshold": -0.1}, "nli_threshold"),
    ({"related_threshold": "0.5"}, "related_threshold"),
    ({"related_threshold": 0.8}, "related_threshold .* contradiction_threshold"),
    ({"contradiction_threshold": 0.3}, "related_threshold .* contradiction_threshold"),
    # SUPPORTED by similarity alone, yet below the citation bar: a supported claim with no evidence shown
    ({"support_threshold": 0.4}, "support_threshold .* related_threshold"),
    ({"related_threshold": 0.7, "support_threshold": 0.6}, "support_threshold .* related_threshold"),
    ({"min_non_ascii_letters": 0}, "min_non_ascii_letters"),
    ({"min_ascii_letter_ratio": 1.5}, "min_ascii_letter_ratio"),
    ({"min_foreign_function_words": 0}, "min_foreign_function_words"),
    ({"min_foreign_function_words": 2.5}, "min_foreign_function_words"),
    ({"foreign_to_english_ratio": -1}, "foreign_to_english_ratio"),
    ({"foreign_to_english_ratio": float("nan")}, "foreign_to_english_ratio"),
    ({"foreign_to_english_ratio": "2"}, "foreign_to_english_ratio"),
])
def test_invalid_values_are_rejected_on_construction(overrides, message):
    with pytest.raises(ValueError, match=message):
        Config(**overrides)


def test_replace_validates_too():
    with pytest.raises(ValueError, match="top_k"):
        dataclasses.replace(Config(), top_k=0)


def test_edge_values_are_allowed():
    Config(related_threshold=0.0, support_threshold=1.0, contradiction_threshold=0.45, nli_context_sentences=0)
    Config(related_threshold=0.6, support_threshold=0.6)


def test_language_check_limits_live_in_config():
    c = Config()
    assert (c.min_non_ascii_letters, c.min_ascii_letter_ratio) == (3, 0.70)
    assert (c.min_foreign_function_words, c.foreign_to_english_ratio) == (2, 2.0)


def test_detector_rejects_support_below_related():
    with pytest.raises(ValueError, match="support_threshold"):
        HallucinationDetector(use_nli=False, related_threshold=0.7, support_threshold=0.6)


def test_source_splitting_limits_live_in_config():
    assert Config().max_sentence_words == MAX_SENTENCE_WORDS == 120
    assert Config().min_source_words == MIN_SOURCE_WORDS == 3


def test_public_records_settings_but_not_memory_limits():
    d = Config().public()
    assert "max_sentence_words" in d and "min_source_words" in d and "cache_size" not in d
    assert Config(use_nli=False).public()["nli_model"] is None
