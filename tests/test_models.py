"""Model loading: the local cache is tried first, and failures become one clear error."""
import logging

import pytest

from hallucination_detector import HallucinationDetector
from hallucination_detector.models import ModelLoadError, load_pretrained, load_spacy


class Recorder:
    """Stands in for SentenceTransformer / CrossEncoder: records how it was called."""

    def __init__(self, cached: bool, online: bool = True):
        self.cached, self.online, self.calls = cached, online, []

    def __call__(self, name, local_files_only=False):
        self.calls.append(local_files_only)
        if local_files_only and not self.cached:
            raise OSError("not found in the cached files")
        if not local_files_only and not self.online:
            raise OSError("We couldn't connect to 'https://huggingface.co'.\nCheck your internet connection.")
        return f"model:{name}"


def test_cached_model_is_loaded_without_touching_the_network():
    factory = Recorder(cached=True, online=False)
    assert load_pretrained(factory, "m", "embedding") == "model:m"
    assert factory.calls == [True]


def test_uncached_model_is_downloaded_with_a_notice(caplog):
    factory = Recorder(cached=False)
    with caplog.at_level(logging.INFO, logger="hallucination_detector"):
        assert load_pretrained(factory, "m", "NLI") == "model:m"
    assert factory.calls == [True, False]
    assert "Downloading the NLI model 'm'" in caplog.text


def test_failed_download_is_one_clear_error():
    with pytest.raises(ModelLoadError) as info:
        load_pretrained(Recorder(cached=False, online=False), "some/model", "embedding")
    message = str(info.value)
    assert "embedding model 'some/model'" in message and "couldn't connect" in message
    assert "\n" not in message and "internet connection" in message


def test_request_id_noise_is_dropped():
    def factory(name, local_files_only=False):
        raise OSError("401 Client Error. (Request ID: Root=1-abc;def)\nRepository Not Found")

    with pytest.raises(ModelLoadError, match=r"\(401 Client Error\)"):
        load_pretrained(factory, "m", "NLI")


def test_missing_spacy_model_says_how_to_install_it():
    with pytest.raises(ModelLoadError, match="python -m spacy download en_core_web_nope"):
        load_spacy("en_core_web_nope")
    with pytest.raises(ModelLoadError):
        HallucinationDetector(spacy_model="en_core_web_nope", use_nli=False, embedder="hashing")


def test_model_load_error_is_a_runtime_error():
    assert issubclass(ModelLoadError, RuntimeError)       # callers that caught RuntimeError keep working
