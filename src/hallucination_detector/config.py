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

    # source splitting
    min_source_words: int = 3                       # shorter source sentences are dropped (unless all are short)
    max_sentence_words: int = 120                   # the embedder truncates around 256 word pieces; split before that

    # English-language check (the models are English-only; other text is reported, not scored)
    min_non_ascii_letters: int = 3                  # fewer accented letters are normal in English ("Pelé", "Müller")
    min_ascii_letter_ratio: float = 0.70            # with more, below this share of ASCII letters it is another script
    # Latin-script text is another language when it has at least min_foreign_function_words of that
    # language's function words (outside names and titles) and more than foreign_to_english_ratio
    # times as many as English-only function words.
    min_foreign_function_words: int = 2
    foreign_to_english_ratio: float = 2.0

    # memory
    cache_size: int = 256                           # documents kept in the sentence and embedding caches (LRU)

    def __post_init__(self) -> None:
        # Fail when the config is built: a bad value otherwise surfaces later as a wrong label or an IndexError.
        for name in ("top_k", "nli_top_k", "min_claim_words", "min_source_words", "max_sentence_words",
                     "cache_size", "min_non_ascii_letters", "min_foreign_function_words"):
            _require_int(self, name, minimum=1)
        _require_int(self, "nli_context_sentences", minimum=0)
        for name in ("related_threshold", "support_threshold", "contradiction_threshold", "nli_threshold",
                     "min_ascii_letter_ratio"):
            value = getattr(self, name)
            if not _is_number(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a number between 0 and 1, got {value!r}")
        if not _is_number(self.foreign_to_english_ratio) or not self.foreign_to_english_ratio >= 0:   # NaN too
            raise ValueError(f"foreign_to_english_ratio must be a number of at least 0, "
                             f"got {self.foreign_to_english_ratio!r}")
        # conflicts are looked for among related evidence; a higher related bar would leave none to compare
        if self.related_threshold > self.contradiction_threshold:
            raise ValueError(f"related_threshold ({self.related_threshold}) must not be above "
                             f"contradiction_threshold ({self.contradiction_threshold})")
        # only related sentences are cited; a lower support bar would give SUPPORTED claims no evidence
        if self.support_threshold < self.related_threshold:
            raise ValueError(f"support_threshold ({self.support_threshold}) must not be below "
                             f"related_threshold ({self.related_threshold})")

    def public(self) -> dict:
        """The settings worth recording next to a result."""
        d = asdict(self)
        del d["cache_size"]                         # memory only; it never changes a result
        if not self.use_nli:
            d["nli_model"] = None
        return d


def _is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _require_int(cfg: Config, name: str, minimum: int) -> None:
    value = getattr(cfg, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}, got {value!r}")


# v1 behavior: NLI off, as shipped in the first release (used to prove the restructure changed nothing)
V1 = Config(use_nli=False)
# v1 with its optional --nli flag: the cross-encoder reads only the single best sentence
V1_NLI = Config(use_nli=True, nli_top_k=1, nli_context_sentences=0)
