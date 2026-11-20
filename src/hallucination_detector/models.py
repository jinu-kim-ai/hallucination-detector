"""Loading the pretrained models: local cache first, and one clear error when a model cannot be had.

Hugging Face libraries contact the Hub on every load, even when the model is already on disk. A slow
or broken connection then delays or breaks a run that needs no download at all. Loading from the cache
first makes a second run work offline and start without a network round trip.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, TypeVar

T = TypeVar("T")
log = logging.getLogger("hallucination_detector")


class ModelLoadError(RuntimeError):
    """A model is not installed or cached, and could not be downloaded."""


def load_pretrained(factory: Callable[..., T], name: str, kind: str) -> T:
    """factory(name, local_files_only=True) if the model is cached, else factory(name), which downloads it."""
    try:
        return factory(name, local_files_only=True)
    except Exception:                       # not cached yet, or only partly downloaded
        pass
    log.info("Downloading the %s model '%s' (first run only) ...", kind, name)
    try:
        return factory(name)
    except Exception as exc:                # the Hub raises many types: connection, 404, gated repo, disk full
        reason = next((line.strip() for line in str(exc).splitlines() if line.strip()), type(exc).__name__)
        reason = re.sub(r"\s*\(Request ID:[^)]*\)", "", reason)       # noise from the Hub's error text
        raise ModelLoadError(
            f"could not load the {kind} model '{name}': it is not in the local cache and downloading it failed "
            f"({reason.rstrip('.')}). Check the model name and your internet connection.") from exc


def load_spacy(name: str = "en_core_web_sm"):
    import spacy

    try:
        return spacy.load(name)
    except OSError as exc:                  # the model is a pip package that has not been installed
        raise ModelLoadError(f"spaCy model '{name}' is not installed. Run: python -m spacy download {name}") from exc
