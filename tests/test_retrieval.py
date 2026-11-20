import numpy as np

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


def test_embeddings_are_unit_length():
    vecs = get_embedder("hashing").encode(["one two three", "four five"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
