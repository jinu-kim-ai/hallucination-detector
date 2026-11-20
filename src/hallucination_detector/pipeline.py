"""The detector: the only module that knows the order of the stages.

    response --claims.py--> claims
    sources  --sources.py--> sentences --retrieval.py--> top-k evidence per claim
    claim + evidence --facts.py / nli.py--> conflicts, novel facts, entailment
    --scoring.py--> label, confidence --> Report with citations
"""
from __future__ import annotations

import importlib
import re
from collections import OrderedDict
from dataclasses import replace
from functools import lru_cache

from . import facts, scoring
from .claims import ClaimExtractor, load_nlp
from .config import Config
from .retrieval import Retriever, SourceIndex, get_embedder
from .schemas import Citation, Claim, ClaimVerdict, Report, SourceDocument, UNSUPPORTED
from .sources import as_documents, split_sentences

# Latin-script languages most often mistaken for English. spaCy ships their stop lists, so the
# language check needs no extra model or dependency.
OTHER_LATIN_LANGUAGES = ("es", "fr", "de", "pt", "it", "nl")
# A response is in another language when it has at least MIN_FOREIGN_FUNCTION_WORDS of that
# language's function words (ones that are not also English) and more than FOREIGN_TO_ENGLISH_RATIO
# times as many as English-only function words. Measured on the dev set, the examples and the sample
# cases: no English sentence crosses it, not even name-heavy ones like "Pedro de la Rosa drove for
# Arrows" (2 vs 2), while a 6-word French or Spanish sentence does ("Marie Curie nació en Varsovia
# en 1867": 2 vs 0). A sentence with a single such word ("... startete am 16. Juli ...") is missed.
MIN_FOREIGN_FUNCTION_WORDS = 2
FOREIGN_TO_ENGLISH_RATIO = 2


class HallucinationDetector:
    def __init__(self, config: Config | None = None, nlp=None, **overrides):
        """`overrides` are Config fields, e.g. HallucinationDetector(use_nli=False, support_threshold=0.6)."""
        self.config = replace(config or Config(), **overrides)
        self.nlp = nlp or load_nlp(self.config.spacy_model)
        self.extractor = ClaimExtractor(self.nlp, self.config.min_claim_words)
        self.retriever = Retriever(get_embedder(self.config.embedder), self.config.cache_size)
        self._nli = None
        # Split documents kept per detector. A long-running service reuses one detector with ever new
        # sources, so this is a bounded LRU (config.cache_size) rather than a dict that grows forever.
        self._sentence_cache: OrderedDict[tuple, tuple] = OrderedDict()

    @property
    def nli(self):
        if self.config.use_nli and self._nli is None:      # loaded on first use: it is the largest model
            from .nli import NLIScorer

            self._nli = NLIScorer(self.config.nli_model)
        return self._nli if self.config.use_nli else None

    # ------------------------------------------------------------------ public API
    def detect(self, response: str, sources) -> Report:
        """Check every claim in `response` against `sources` (a string, or a list of strings / dicts / documents)."""
        docs = as_documents(sources)
        index, warnings = self._index(docs)
        report = Report(response, [], docs, warnings, self.config.public())

        if not _looks_english(response):
            report.warnings.append("response does not look like English text; these models are English-only")
            report.unscorable = True
            return report

        claims = self.extractor.extract(response)
        if not claims:
            report.warnings.append("no checkable claims found in the response")
            return report

        by_id: dict[int, ClaimVerdict] = {}
        for claim in claims:
            if claim.duplicate_of is not None:              # same fact stated twice: reuse the verdict
                first = by_id[claim.duplicate_of]
                verdict = ClaimVerdict(claim, first.label, first.confidence,
                                       [f"duplicate of claim {first.claim.id}"], first.citations,
                                       first.best_similarity)
            else:
                verdict = self._check(claim, index, report.warnings)
            by_id[claim.id] = verdict
            report.verdicts.append(verdict)
        return report

    def check_claim(self, claim: str, sources) -> ClaimVerdict:
        """Check one already-written claim (used by the evaluation harness)."""
        index, _ = self._index(as_documents(sources))
        return self._check(Claim(1, claim, claim, 0, len(claim)), index, [])

    # ------------------------------------------------------------------ stages
    def _index(self, docs: list[SourceDocument]) -> tuple[SourceIndex, list[str]]:
        warnings: list[str] = []
        per_document = []
        for doc in docs:
            sentences, doc_warnings = self._split(doc)
            warnings += doc_warnings
            per_document.append((doc.text, sentences))
        index = self.retriever.index(per_document)
        if not index.sentences:
            warnings.append("no source text provided; every claim is unsupported")
        return index, warnings

    def _split(self, doc: SourceDocument) -> tuple:
        """split_sentences(doc), cached: the same sources are usually checked against many responses."""
        cfg = self.config
        # the split limits are part of the key: evaluate.sweep_thresholds swaps self.config between calls
        key = (doc.id, doc.text, cfg.min_source_words, cfg.max_sentence_words)
        if key in self._sentence_cache:
            self._sentence_cache.move_to_end(key)
        else:
            self._sentence_cache[key] = split_sentences(doc, self.nlp, min_words=cfg.min_source_words,
                                                        max_words=cfg.max_sentence_words)
            while len(self._sentence_cache) > cfg.cache_size:
                self._sentence_cache.popitem(last=False)          # least recently used
        return self._sentence_cache[key]

    def _check(self, claim: Claim, index: SourceIndex, warnings: list[str]) -> ClaimVerdict:
        cfg, text = self.config, claim.resolved_text
        hits = self.retriever.search(index, text, cfg.top_k)
        if not hits:
            return ClaimVerdict(claim, UNSUPPORTED, 1.0, ["no source text to check against"], [])

        citations = [
            Citation(s.source_id, s.index, (s.start, s.end), s.text, sim)
            for s, sim in ((index.sentences[i], sim) for i, sim in hits)
        ]
        best_sim = citations[0].similarity
        related = [c for c in citations if c.similarity >= cfg.related_threshold]

        claim_facts = facts.extract(self.nlp, text)
        # Only evidence about the *same* fact can contradict; weakly related text cannot.
        conflicts = (facts.conflicts(self.nlp, text, claim_facts, [c.quote for c in related], index.text_lower)
                     if best_sim >= cfg.contradiction_threshold else [])
        novel = facts.novel(claim_facts, index.text_lower)

        if self.nli is not None:
            # v1 read only the best sentence; v2 also reads the other related ones
            to_score = [(pos, c) for (pos, _), c in zip(hits, citations)
                        if c is citations[0] or c.similarity >= cfg.related_threshold][: cfg.nli_top_k]
            premises = [" ".join(s.text for s in index.neighbors(pos, cfg.nli_context_sentences))
                        for pos, _ in to_score]
            for (_, citation), probs in zip(to_score, self.nli.score([(p, text) for p in premises])):
                citation.nli = probs
        nli = scoring.decisive_nli(citations, cfg)

        notation_note = None
        if (nli and nli["contradiction"] >= cfg.nli_threshold and not conflicts
                and facts.same_values_other_notation(text, citations[0].quote)):
            # the numbers agree in value and only their writing differs: NLI alone must not call that a contradiction
            nli = {**nli, "contradiction": 0.0}
            notation_note = "NLI contradiction ignored: the numbers match the evidence, written differently"

        label, confidence, reasons = scoring.decide(best_sim, conflicts, novel, nli, cfg)
        if notation_note:
            reasons.append(notation_note)
        self._warn_if_sources_conflict(claim, citations, warnings)
        # PLAN section 3: cite only sentences that are actually evidence. With none, the list is empty
        # and the best similarity is still given in the reasons.
        return ClaimVerdict(claim, label, round(confidence, 3), reasons, related, best_sim)

    def _warn_if_sources_conflict(self, claim: Claim, citations: list[Citation], warnings: list[str]) -> None:
        """Two close sentences from different documents that disagree about the claim's numbers or its truth."""
        cfg = self.config
        close = [c for c in citations if c.similarity >= cfg.contradiction_threshold]
        if len({c.source_id for c in close}) < 2:
            return
        claim_nums = _digit_numbers(claim.resolved_text)

        def stance(c: Citation) -> str | None:
            if c.nli and c.nli["entailment"] >= cfg.nli_threshold:
                return "supports"
            if c.nli and c.nli["contradiction"] >= cfg.nli_threshold:
                return "contradicts"
            # NLI neutral or off: fall back to the claim's numbers, kept even when NLI is neutral
            # because the NLI model misses some number conflicts
            if _rival_numbers(claim_nums, c.quote):
                return "contradicts"
            quote_nums = _digit_numbers(c.quote)
            if claim_nums and all(facts.has_number(quote_nums, n) for n in claim_nums):
                return "supports"
            return None

        best: dict[str, Citation] = {}
        for c in close:                                                 # best citation per source wins
            best.setdefault(c.source_id, c)
        stances = {s: stance(c) for s, c in best.items()}
        supporters = [s for s, v in stances.items() if v == "supports"]
        opposers = [s for s, v in stances.items() if v == "contradicts"]
        if supporters and opposers:
            warnings.append(f"sources conflict on claim {claim.id}: '{supporters[0]}' supports it, "
                            f"'{opposers[0]}' contradicts it")
            return
        # every close source contradicts the claim, but they also give different values from each other
        rivals = {s: _rival_numbers(claim_nums, best[s].quote) for s in opposers}
        for i, a in enumerate(opposers):
            for b in opposers[i + 1:]:
                if rivals[a] and rivals[b] and rivals[a].isdisjoint(rivals[b]):
                    warnings.append(f"sources conflict on claim {claim.id}: '{a}' and '{b}' disagree with each other")
                    return


def _digit_numbers(text: str) -> set[float]:
    """Numbers written with digits. Number words ('two Canadian researchers') are too often incidental
    to stand in for a claimed value, so they are left out."""
    no_words = re.sub(r"[A-Za-z]+", lambda m: " " if m.group().lower() in facts.WORD_NUMBERS else m.group(), text)
    return facts.numbers(no_words)


def _number_kind(n: float) -> str:
    # a year only competes with a year: '1921' vs '2 researchers' is not a disagreement
    return "year" if n.is_integer() and 1000 <= n <= 2100 else "other"


def _rival_numbers(claim_nums: set[float], quote: str) -> set[float]:
    """Numbers in the quote that take the place of a claim number the quote lacks (same kind, not in the claim)."""
    quote_nums = _digit_numbers(quote)
    missing_kinds = {_number_kind(n) for n in claim_nums if not facts.has_number(quote_nums, n)}
    return {m for m in quote_nums if _number_kind(m) in missing_kinds and not facts.has_number(claim_nums, m)}


def _looks_english(text: str) -> bool:
    return _mostly_latin(text) and not _other_latin_language(text)


def _mostly_latin(text: str) -> bool:
    letters = re.findall(r"[^\W\d_]", text)
    non_ascii = sum(not ch.isascii() for ch in letters)
    # a few accented letters are normal in English ("Pelé", "Müller"); short Korean text is not
    if non_ascii < 3:
        return True
    return (len(letters) - non_ascii) / len(letters) >= 0.7


@lru_cache(maxsize=None)
def _stop_words(lang: str) -> frozenset[str]:
    return frozenset(importlib.import_module(f"spacy.lang.{lang}.stop_words").STOP_WORDS)


def _other_latin_language(text: str) -> bool:
    """French, Spanish, German... text: many of that language's function words, few English-only ones."""
    words = [w.lower() for w in re.findall(r"[^\W\d_]+", text)]
    english = _stop_words("en")
    for lang in OTHER_LATIN_LANGUAGES:
        other = _stop_words(lang)
        foreign_only = sum(w in other and w not in english for w in words)
        english_only = sum(w in english and w not in other for w in words)
        if foreign_only >= MIN_FOREIGN_FUNCTION_WORDS and foreign_only > FOREIGN_TO_ENGLISH_RATIO * english_only:
            return True
    return False
