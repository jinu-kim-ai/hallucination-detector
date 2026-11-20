"""The detector: the only module that knows the order of the stages.

    response --claims.py--> claims
    sources  --sources.py--> sentences --retrieval.py--> top-k evidence per claim
    claim + evidence --facts.py / nli.py--> conflicts, novel facts, entailment
    --scoring.py--> label, confidence --> Report with citations
"""
from __future__ import annotations

import re
from dataclasses import replace

from . import facts, scoring
from .claims import ClaimExtractor, load_nlp
from .config import Config
from .retrieval import Retriever, SourceIndex, get_embedder
from .schemas import Citation, Claim, ClaimVerdict, Report, SourceDocument, UNSUPPORTED
from .sources import as_documents, split_sentences


class HallucinationDetector:
    def __init__(self, config: Config | None = None, nlp=None, **overrides):
        """`overrides` are Config fields, e.g. HallucinationDetector(use_nli=False, support_threshold=0.6)."""
        self.config = replace(config or Config(), **overrides)
        self.nlp = nlp or load_nlp(self.config.spacy_model)
        self.extractor = ClaimExtractor(self.nlp, self.config.min_claim_words)
        self.retriever = Retriever(get_embedder(self.config.embedder))
        self._nli = None
        self._sentence_cache: dict[tuple[str, str], tuple] = {}

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

        if not _mostly_latin(response):
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
                                       [f"duplicate of claim {first.claim.id}"], first.citations)
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
            key = (doc.id, doc.text)
            if key not in self._sentence_cache:
                self._sentence_cache[key] = split_sentences(doc, self.nlp)
            sentences, doc_warnings = self._sentence_cache[key]
            warnings += doc_warnings
            per_document.append((doc.text, sentences))
        index = self.retriever.index(per_document)
        if not index.sentences:
            warnings.append("no source text provided; every claim is unsupported")
        return index, warnings

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

        label, confidence, reasons = scoring.decide(best_sim, conflicts, novel, nli, cfg)
        self._warn_if_sources_conflict(claim, claim_facts, citations, warnings)
        # cite only sentences that are actually evidence; always keep the best one so the verdict is explainable
        cited = [c for c in citations if c.similarity >= cfg.related_threshold] or citations[:1]
        return ClaimVerdict(claim, label, round(confidence, 3), reasons, cited)

    def _warn_if_sources_conflict(self, claim, claim_facts, citations, warnings) -> None:
        """Two close sentences from different documents that disagree about the claim's numbers or its truth."""
        cfg = self.config
        close = [c for c in citations if c.similarity >= cfg.contradiction_threshold]
        if len({c.source_id for c in close}) < 2:
            return

        def stance(c: Citation) -> str | None:
            if c.nli and c.nli["entailment"] >= cfg.nli_threshold:
                return "supports"
            if c.nli and c.nli["contradiction"] >= cfg.nli_threshold:
                return "contradicts"
            nums = facts.numbers(c.quote)
            if claim_facts.numbers and nums:
                agrees = all(facts.has_number(nums, n) for n in claim_facts.numbers)
                return "supports" if agrees else "contradicts"
            return None

        stances = {c.source_id: stance(c) for c in reversed(close)}      # best citation per source wins
        supporters = [s for s, v in stances.items() if v == "supports"]
        opposers = [s for s, v in stances.items() if v == "contradicts"]
        if supporters and opposers:
            warnings.append(f"sources conflict on claim {claim.id}: '{supporters[0]}' supports it, "
                            f"'{opposers[0]}' contradicts it")


def _mostly_latin(text: str) -> bool:
    letters = re.findall(r"[^\W\d_]", text)
    if len(letters) < 10:
        return True
    return sum(ch.isascii() for ch in letters) / len(letters) >= 0.7
