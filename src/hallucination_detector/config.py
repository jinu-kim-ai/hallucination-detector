"""Every model name and threshold in one place."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    # models
    spacy_model: str = "en_core_web_sm"
    embedder: str = "all-MiniLM-L6-v2"              # or "hashing" for a dependency-free lexical fallback
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    use_nli: bool = True

    # retrieval
    top_k: int = 3                                  # evidence sentences cited per claim
    related_threshold: float = 0.45                 # below this, a sentence is not evidence at all

    # decision
    support_threshold: float = 0.65                 # similarity that counts as support on its own
    contradiction_threshold: float = 0.70           # only evidence this close can contradict a claim
    nli_threshold: float = 0.70                     # entailment / contradiction probability that is decisive
    nli_top_k: int = 3                              # how many cited sentences the NLI model reads
    nli_context_sentences: int = 0                  # neighbor sentences on each side added to the NLI premise

    # claim extraction
    min_claim_words: int = 3

    def public(self) -> dict:
        """The settings worth recording next to a result."""
        d = asdict(self)
        if not self.use_nli:
            d["nli_model"] = None
        return d


# v1 behavior: NLI off, as shipped in the first release (used to prove the restructure changed nothing)
V1 = Config(use_nli=False)
# v1 with its optional --nli flag: the cross-encoder reads only the single best sentence
V1_NLI = Config(use_nli=True, nli_top_k=1, nli_context_sentences=0)
