import numpy as np
import pytest

from hallucination_detector.retrieval import HashingEmbedder, Retriever, get_embedder
from hallucination_detector.schemas import Sentence


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return super().encode(texts)


def sentences(source_id, texts):
    out, pos = [], 0
    for i, t in enumerate(texts):
        out.append(Sentence(source_id, i, t, pos, pos + len(t)))
        pos += len(t) + 1
    return out


DOC_A = ["The lunar module landed in the Sea of Tranquility.", "The crew collected lunar samples."]
DOC_B = ["Python was created by Guido van Rossum.", "Python uses indentation for code blocks."]


def build(retriever):
    return retriever.index([(" ".join(DOC_A), sentences("a", DOC_A)), (" ".join(DOC_B), sentences("b", DOC_B))])


def test_search_returns_best_first_across_documents():
    retriever = Retriever(HashingEmbedder())
    index = build(retriever)
    hits = retriever.search(index, "Guido van Rossum created Python", k=3)
    assert len(hits) == 3
    sims = [sim for _, sim in hits]
    assert sims == sorted(sims, reverse=True)
    best = index.sentences[hits[0][0]]
    assert (best.source_id, best.index) == ("b", 0)


def test_index_keeps_all_source_text_for_grounding_checks():
    index = build(Retriever(HashingEmbedder()))
    assert "sea of tranquility" in index.text_lower and "guido van rossum" in index.text_lower
    assert index.matrix.shape[0] == len(index.sentences) == 4


def test_neighbors_stay_inside_one_document():
    index = build(Retriever(HashingEmbedder()))
    assert [s.source_id for s in index.neighbors(1, radius=1)] == ["a", "a"]      # last sentence of doc a
    assert [s.index for s in index.neighbors(2, radius=1)] == [0, 1]              # first sentence of doc b
    assert [s.index for s in index.neighbors(2, radius=0)] == [0]


def test_document_embeddings_are_cached():
    embedder = CountingEmbedder()
    retriever = Retriever(embedder)
    build(retriever)
    assert embedder.calls == 2            # one call per document
    build(retriever)
    assert embedder.calls == 2            # second build reuses the cache


def test_empty_index_returns_no_hits():
    retriever = Retriever(HashingEmbedder())
    index = retriever.index([("", [])])
    assert retriever.search(index, "anything", k=3) == []


def test_embedding_cache_is_bounded_lru():
    embedder = CountingEmbedder()
    retriever = Retriever(embedder, cache_size=2)
    docs = [sentences(name, [f"Sentence about {name}."]) for name in ("a", "b", "c")]
    retriever.index([("a", docs[0])])
    retriever.index([("b", docs[1])])
    retriever.index([("a", docs[0])])            # touch a, so b is now the oldest
    retriever.index([("c", docs[2])])            # evicts b
    assert len(retriever._cache) == 2 and embedder.calls == 3
    retriever.index([("a", docs[0])])
    assert embedder.calls == 3                   # a survived
    retriever.index([("b", docs[1])])
    assert embedder.calls == 4                   # b was evicted


def test_cache_key_keeps_sentence_boundaries():
    retriever = Retriever(HashingEmbedder())
    retriever.index([("d1", sentences("d1", ["apple\0banana", "cherry"]))])
    index = retriever.index([("d2", sentences("d2", ["apple", "banana\0cherry"]))])
    assert len(retriever._cache) == 2
    assert np.allclose(index.matrix[0], HashingEmbedder().encode(["apple"])[0])


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_k_below_one(k):
    retriever = Retriever(HashingEmbedder())
    with pytest.raises(ValueError, match="at least 1"):
        retriever.search(build(retriever), "alpha", k=k)


def test_embeddings_are_unit_length():
    vecs = get_embedder("hashing").encode(["one two three", "four five"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
