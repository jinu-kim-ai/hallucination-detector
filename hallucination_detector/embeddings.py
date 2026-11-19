"""Sentence embedding backends used to find evidence for a claim."""
from __future__ import annotations

import hashlib
import re

import numpy as np


class SentenceTransformerEmbedder:
    """Default backend: a Sentence Transformers bi-encoder (all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


class HashingEmbedder:
    """Dependency-free lexical fallback (hashed word uni/bigrams).

    Much weaker than a neural model; used for fast unit tests and offline runs.
    """

    name = "hashing"

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", text.lower())
            for gram in words + [f"{a}_{b}" for a, b in zip(words, words[1:])]:
                h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def get_embedder(name: str = "all-MiniLM-L6-v2"):
    return HashingEmbedder() if name == "hashing" else SentenceTransformerEmbedder(name)
