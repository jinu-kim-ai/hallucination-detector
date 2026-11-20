"""Evidence retrieval: embed source sentences once, then cosine top-k per claim."""
from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .config import Config
from .models import load_pretrained
from .schemas import Sentence


class SentenceTransformerEmbedder:
    """Default backend: a Sentence Transformers bi-encoder."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = load_pretrained(SentenceTransformer, model_name, "embedding")
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64))


class HashingEmbedder:
    """Dependency-free lexical fallback (hashed word uni/bigrams). Much weaker; for offline use and fast tests."""

    name = "hashing"

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", text.lower())
            for gram in words + [f"{a}_{b}" for a, b in zip(words, words[1:])]:
                out[row, int(hashlib.md5(gram.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def get_embedder(name: str):
    return HashingEmbedder() if name == "hashing" else SentenceTransformerEmbedder(name)


@dataclass
class SourceIndex:
    sentences: list[Sentence]      # every sentence of every document, in order
    matrix: np.ndarray             # one normalized embedding row per sentence
    text_lower: str                # all source text, lower-cased (for "does this name appear anywhere" checks)
    text: str = ""                 # the same text with its case: '5M users' and '5m tall' differ only by case

    def neighbors(self, i: int, radius: int) -> list[Sentence]:
        """Sentence i with up to `radius` sentences on each side, from the same document."""
        me = self.sentences[i]
        lo, hi = max(0, i - radius), min(len(self.sentences), i + radius + 1)
        return [s for s in self.sentences[lo:hi] if s.source_id == me.source_id]


class Retriever:
    def __init__(self, embedder, cache_size: int = Config.cache_size):
        self.embedder = embedder
        self.cache_size = max(1, cache_size)
        # document hash -> sentence embeddings; LRU so a long-lived detector does not grow forever
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def index(self, per_document: list[tuple[str, list[Sentence]]]) -> SourceIndex:
        """per_document: (document text, its sentences) pairs. Embeddings are cached per document."""
        sentences, blocks, texts = [], [], []
        for text, sents in per_document:
            texts.append(text)
            if not sents:
                continue
            sentences.extend(sents)
            blocks.append(self._embeddings(sents))
        matrix = np.vstack(blocks) if blocks else np.zeros((0, 1), dtype=np.float32)
        joined = "\n".join(texts)
        return SourceIndex(sentences, matrix, joined.lower(), joined)

    def _embeddings(self, sents: list[Sentence]) -> np.ndarray:
        texts = [s.text for s in sents]
        # JSON keeps sentence boundaries unambiguous, whatever characters the sentences contain
        key = hashlib.sha1(json.dumps([self.embedder.name, texts]).encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = self.embedder.encode(texts)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return self._cache[key]

    def search(self, index: SourceIndex, query: str, k: int) -> list[tuple[int, float]]:
        """(sentence position, cosine similarity) for the k closest sentences, best first."""
        if k < 1:      # [:0] would cite nothing and [:-1] would drop the worst hit, both silently
            raise ValueError(f"k must be at least 1, got {k}")
        if not index.sentences:
            return []
        sims = index.matrix @ self.embedder.encode([query])[0]
        order = np.argsort(-sims, kind="stable")[:k]
        return [(int(i), float(sims[i])) for i in order]
