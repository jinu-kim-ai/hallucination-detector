# PLAN: Hallucination Detector v2

A design and implementation plan for a modular hallucination detector. Given an AI-generated response and one or more source documents, it:

1. breaks the response into atomic claims (spaCy),
2. finds the best-matching evidence in the sources (Sentence Transformers),
3. decides whether the evidence entails or contradicts each claim (fact checks + an NLI cross-encoder),
4. returns a factual reliability score with claim-by-claim citations.

## Status: implemented

All 22 checklist steps are done; results are in the README. Six things turned out differently from the plan below:

1. **Neighbor sentences in the NLI premise are off by default.** On the development set they caused 2 extra errors, so `nli_context_sentences` defaults to 0. The option remains in `config.py`.
2. **Only the best sentence can contradict a claim.** The plan let any cited sentence contradict. In practice a sentence about a different fact (the 1903 Physics prize) contradicted a true claim about the 1911 Chemistry prize. Entailment may still come from any cited sentence.
3. **Over-long source sentences are split at commas and semicolons with a warning**, rather than at parsed clause boundaries.
4. **Multi-document claims went into the held-out set, not the development set**, so the development numbers stay comparable with v1. The conflicting-sources case is covered by a test (`tests/test_pipeline.py`) rather than a labeled data row, because its correct label is ambiguous by design. The units row in section 4 is a documented limit without a test.

5. **Hedged sentences are checked, not dropped.** Section 4 said hedges are dropped. A hedge does not make a false statement true: "I think Apollo 11 launched in 1970" still states a wrong year. Only questions and pleasantries ("I hope this helps!") are dropped. A discourse opener ("Overall,", "In summary,", "Sure,") is cut off and the rest of the sentence is checked.
6. **Uncited claims still report their closest match.** Section 3 says a claim with no related evidence has an empty `citations` list, and it does. The similarity of the closest sentence is kept in a new `best_similarity` field, so an `UNSUPPORTED` verdict can still be explained.

Step 5 was verified exactly: after the restructure, every one of the 50 development claims received the same label, confidence and similarity as in v1, with and without NLI. The held-out set (step 20) was scored once and its errors were left unfixed.

### Review and fix rounds

After v2 shipped, the code was audited against this plan by six parallel reviewers. Each reported defect was reproduced, then confirmed by an independent skeptic: 56 defects across every module. Five fixers repaired them in parallel, in isolated worktrees, one module group each. Each review of the merged fixes found fewer new defects introduced by the fixes themselves: 23, then 13, then 10, then 6. Each was fixed in another round, and from the third round on every fixer's work was reviewed and repaired inside its own worktree before merging. Every fix has a regression test (125 tests became 842). The sample cases in `examples/cases/` were written during this audit; they are realistic factual and hallucinated answers with hand-assigned labels, used to find and measure false alarms. A quadratic slowdown on long sources found during the audit was fixed; detection time is now linear in source length.

The rest of this document is the plan as written before implementation.

## 0. Starting point

This repository already contains a working v1 (flat `hallucination_detector/` package, 9 passing tests, published results). The plan below is a restructure and extension of that code, not a rewrite from zero.

| Area | v1 today | v2 target |
|---|---|---|
| Layout | flat package at repo root | `src/` layout, installable with `pip install -e .` |
| Sources | one source document | many documents, each with an id and title |
| Evidence unit | one sentence | one sentence, plus its neighbors as NLI context |
| Citations | evidence sentence text only | source id, sentence index, character span, quote |
| NLI | optional flag, best sentence only | on by default, over the top-k sentences |
| Entry points | `python -m hallucination_detector` | `hallucheck` CLI + Streamlit `app.py` |
| Output | terminal text, HTML | the same, plus a documented JSON schema |
| Modules | `detector.py` holds retrieval, checks and scoring | one module per stage |

v1 results to preserve (50 hand-labeled claims in `data/`): F1 0.931 without NLI, F1 1.000 with NLI. Both are optimistic, because the thresholds were tuned on the same 50 claims (see step 20).

## 1. Folder structure

```
hallucination-detector/
├── PLAN.md
├── README.md
├── requirements.txt
├── pyproject.toml              # package metadata, src layout, `hallucheck` console script
├── app.py                      # Streamlit UI: paste response + sources, see highlighted report
├── src/
│   └── hallucination_detector/
│       ├── __init__.py         # public API: HallucinationDetector, schemas
│       ├── config.py           # model names, thresholds, top-k (one dataclass, no magic numbers elsewhere)
│       ├── schemas.py          # SourceDocument, Sentence, Claim, Citation, ClaimVerdict, Report
│       ├── claims.py           # spaCy claim decomposition
│       ├── sources.py          # load documents, split into sentences with character offsets
│       ├── retrieval.py        # embed sentences, cosine top-k
│       ├── nli.py              # cross-encoder entail / contradict / neutral
│       ├── facts.py            # number, entity and negation checks
│       ├── scoring.py          # label decision, confidence, reliability score
│       ├── pipeline.py         # HallucinationDetector: wires the stages together
│       ├── report.py           # terminal, HTML and JSON rendering
│       ├── evaluate.py         # precision / recall / F1, per domain, confusion matrix, sweep
│       └── cli.py              # `hallucheck detect ...` and `hallucheck evaluate ...`
├── tests/
│   ├── conftest.py             # shared fixtures: one loaded detector per test session
│   ├── test_claims.py
│   ├── test_sources.py
│   ├── test_retrieval.py
│   ├── test_facts.py
│   ├── test_scoring.py
│   ├── test_pipeline.py        # end-to-end, includes every edge case in section 4
│   ├── test_report.py
│   └── test_evaluate.py
├── data/
│   ├── sources.json            # existing: 5 source documents
│   ├── labeled_claims.jsonl    # existing: 50 labeled claims (development set)
│   └── heldout/                # new: claims never used for tuning (step 20)
├── examples/
│   ├── request.json            # sample input in the section 3 schema
│   ├── source.txt, answer.txt  # existing plain-text sample
│   └── report.html             # generated sample report
└── results/                    # saved evaluation runs
```

Module rule: each stage imports only `schemas.py` and `config.py`, never another stage. `pipeline.py` is the only file that knows the order. This keeps every stage testable alone and replaceable (for example, a different embedder).

## 2. Libraries and models

Everything runs on CPU. Versions are the ones v1 was tested with on Python 3.13.

| Library | Tested version | Used for |
|---|---|---|
| `spacy` | 3.8 | sentence splitting, dependency parse, named entities |
| `sentence-transformers` | 6.3 | bi-encoder embeddings and the NLI cross-encoder |
| `torch` | 2.15 (CPU) | installed as a dependency of sentence-transformers |
| `numpy` | 2.5 | cosine similarity |
| `streamlit` | 1.67 | `app.py` only |
| `pytest` | 9.2 | tests |

No vector database: sources are small enough for an in-memory NumPy matrix.

| Model | Role | Size on disk (measured) | Notes |
|---|---|---|---|
| `en_core_web_sm` (spaCy 3.8.0) | claims, sentences, entities | 15 MB | install with `python -m spacy download en_core_web_sm` |
| `sentence-transformers/all-MiniLM-L6-v2` | evidence retrieval | 87 MB | 384-dim embeddings, inputs truncated at 256 word pieces |
| `cross-encoder/nli-deberta-v3-xsmall` | entailment vs contradiction | 281 MB | output order: contradiction, entailment, neutral |

Total first-run download is about 380 MB; later runs load from the local cache. `--no-nli` skips the cross-encoder for a faster, 100 MB setup.

## 3. Data schemas and scoring

### Input

```json
{
  "response": "Apollo 11 landed on July 20, 1969. The crew brought back 45 kg of samples.",
  "sources": [
    {"id": "nasa-summary", "title": "Apollo 11 mission summary", "text": "Apollo 11 launched ..."}
  ]
}
```

- `response`: required string.
- `sources`: required list; may be empty (see section 4). `id` must be unique; `title` is optional.
- The CLI also accepts `--answer FILE --source FILE [--source FILE ...]`, using each file name as the id.

### Output

```json
{
  "reliability_score": 49.9,
  "hallucination_rate": 0.5,
  "counts": {"SUPPORTED": 4, "CONTRADICTED": 2, "FABRICATED": 1, "UNSUPPORTED": 1},
  "warnings": [],
  "claims": [
    {
      "id": 5,
      "text": "The crew brought back about 45 kilograms of lunar samples.",
      "response_span": [212, 269],
      "label": "CONTRADICTED",
      "confidence": 0.93,
      "reasons": ["number mismatch: claim says 45, evidence says 21.5"],
      "citations": [
        {
          "source_id": "nasa-summary",
          "sentence_index": 5,
          "source_span": [318, 381],
          "quote": "The astronauts collected about 21.5 kilograms of lunar samples.",
          "similarity": 0.83,
          "nli": {"entailment": 0.00, "contradiction": 0.99, "neutral": 0.01}
        }
      ]
    }
  ],
  "config": {"embedder": "all-MiniLM-L6-v2", "nli": "nli-deberta-v3-xsmall", "support_threshold": 0.65}
}
```

- `response_span` and `source_span` are character offsets, so a UI can highlight both sides.
- `citations` holds up to `top_k` (default 3) sentences, best first. A claim with no related evidence has an empty list.
- `reliability_score` is `null` when the response has no checkable claims.
- `config` records the settings that produced the result, so any report can be reproduced.

### Labels

| Label | Meaning |
|---|---|
| `SUPPORTED` | evidence closely matches and every number and name in the claim is grounded |
| `CONTRADICTED` | evidence covers the same fact and disagrees |
| `FABRICATED` | the claim adds specific numbers or names that appear in no source |
| `UNSUPPORTED` | no source backs the claim (opinion, speculation, off-topic) |

### Per-claim scoring

Inputs for a claim:

- `s`: cosine similarity of its best evidence sentence.
- `E`, `C`: highest NLI entailment and contradiction probabilities over the cited sentences. The NLI premise is the evidence sentence plus one neighbor on each side; the hypothesis is the claim.
- `conflicts`: numbers, entities or negations that disagree with the best evidence.
- `novel`: numbers or entities found in no source.

Rules, first match wins:

| # | Condition | Label |
|---|---|---|
| 1 | `s ≥ 0.70` and (`conflicts` not empty or `C ≥ 0.70`) | `CONTRADICTED` |
| 2 | `novel` not empty | `FABRICATED` |
| 3 | `s ≥ 0.65`, or (`E ≥ 0.70` and `s ≥ 0.45`) | `SUPPORTED` |
| 4 | otherwise | `UNSUPPORTED` |

Rule 1 requires high similarity on purpose: only evidence about the same fact can contradict a claim. Weakly related text with different numbers means the claim is fabricated, not contradicted.

Confidence (0 to 1), carried over from v1:

| Label | Confidence |
|---|---|
| `CONTRADICTED` | `min(1, 0.55 + 0.15·len(conflicts) + 0.30·(s − 0.45)/0.55)` |
| `FABRICATED` | `min(1, 0.55 + 0.10·len(novel) + 0.35·(1 − s))` |
| `SUPPORTED` | `max(0.5 + 0.5·min(1, (s − 0.45)/0.55), E)` |
| `UNSUPPORTED` | `0.5 + 0.5·(0.65 − s)/0.65` |

These are heuristics, not calibrated probabilities. The README must say so.

### Response-level scoring

- `reliability_score = 100 × (sum of confidence over SUPPORTED claims) / (sum of confidence over all claims)`
- `hallucination_rate = (claims not SUPPORTED) / (all claims)`

A confident contradiction lowers the score more than a borderline one, and a confidently supported claim raises it more than a marginal one.

All thresholds (0.45, 0.65, 0.70, NLI 0.70, `top_k` 3) live in `config.py` and can be overridden from the CLI.

## 4. Edge cases

| Case | Example | Planned behavior |
|---|---|---|
| Compound sentence, shared subject | "Curie discovered polonium and won two Nobel Prizes." | split on coordinated verbs; copy the subject into the second claim (v1 does this) |
| Compound sentence, own subjects | "Armstrong walked first, and Aldrin followed." | split into two claims, each keeping its own subject |
| List inside one noun phrase | "The crew was Armstrong, Aldrin and Collins." | keep as one claim; each name is checked as an entity |
| Relative clause | "Curie, who was born in Warsaw, won in 1903." | v2: split off the clause as its own claim; until then, checked as one claim |
| Negated claim | "Python 3 was fully backward compatible." vs source "was not" | negation attached to a verb both texts share counts as a conflict; NLI is the second check |
| Negation on a different verb | source: "do not produce insulin and require therapy"; claim: "require therapy" | not a conflict: the negated verb is not in the claim (v1 regression test) |
| Pronoun subject | "She died in 1934." | v2: replace a leading pronoun with the most recent named subject before retrieval |
| Empty source list or blank text | `"sources": []` | every claim `UNSUPPORTED`, score 0, warning "no source text provided"; never crash |
| Empty response, or no factual sentences | "I hope this helps!" | zero claims, `reliability_score: null`, warning "no checkable claims" |
| Questions, hedges, pleasantries | "Is that right?" | dropped during claim extraction |
| Very short fragment | "Yes, exactly." | dropped (fewer than 3 words) |
| Number formats | "1,200", "12%", "$1.6 billion", "two" | normalize to numeric values before comparing |
| Same number, different unit | "21.5 pounds" vs "21.5 kilograms" | not caught by the number check; relies on NLI. Listed as a known limit |
| Relative dates | "last year", "the century" | not treated as checkable facts |
| Right entities, wrong relation | "Aldrin was the first to walk on the Moon." | fact checks pass; only NLI catches it, so NLI is on by default |
| Fact spread over two sentences | claim joins a date and a place from adjacent sentences | NLI premise includes neighbor sentences; cite each sentence used |
| Sources disagree with each other | doc A says 1921, doc B says 1925 | cite both; label by the higher-similarity evidence; add warning "sources conflict" |
| Duplicate claims | the same fact stated twice | scored once; the second is marked as a duplicate of the first |
| Long source document | 50 pages | embed in batches; cache embeddings per document hash |
| Long sentence | over 256 word pieces | embedder truncates; split on clause boundaries first and warn |
| Markdown or bullets in the response | "- Revenue: $412M" | strip list markers and headings before parsing |
| Non-English text | Korean response | out of scope for these English-only models; warn and return `null` score |
| Named-entity mistakes | spaCy tags "Moon" as a person | an entity found anywhere in the sources is never a conflict (v1 fix) |

Every row with a planned behavior gets a test in `tests/test_pipeline.py`.

## 5. Implementation checklist

Each step ends with the full test suite passing.

**Restructure, no behavior change**

1. Add `pyproject.toml` with the `src/` layout and a `hallucheck` console script. Confirm `pip install -e .` works.
2. Move the package to `src/hallucination_detector/`. Run the 9 existing tests; all must pass unchanged.
3. Add `schemas.py` and `config.py`. Move every threshold and model name into `config.py`.
4. Split `detector.py` into `retrieval.py`, `facts.py`, `scoring.py` and `pipeline.py`. Move the cross-encoder into `nli.py`.
5. Re-run the evaluation. It must reproduce v1 exactly: F1 0.931 without NLI, 1.000 with NLI. This is the proof that the refactor changed nothing.

**New capability**

6. `sources.py`: accept a list of documents; split each into sentences with character offsets.
7. `retrieval.py`: one embedding matrix across all documents, with a map from row to (source id, sentence index). Cache by document hash.
8. Citations: attach source id, sentence index, character span and quote to each claim's evidence.
9. NLI on by default, run over the top-k sentences, with neighbor sentences as premise context. Add `--no-nli`.
10. `claims.py`: record each claim's character span in the response; mark duplicates.
11. Claim splitting for relative clauses. Coordinated clauses with their own subjects already split in v1; add a test for them.
12. Pronoun replacement for leading subject pronouns.
13. Input cleaning: strip markdown list markers and headings.
14. Warnings: empty sources, no checkable claims, conflicting sources, truncated sentences, non-English text.

**Interfaces**

15. `report.py`: JSON output in the section 3 schema; HTML report with source-side highlighting and source ids.
16. `cli.py`: `hallucheck detect --request request.json` and `--answer FILE --source FILE ...`; `hallucheck evaluate`.
17. `app.py`: Streamlit page with a response box, one or more source boxes, the highlighted response, a per-claim table with citations, and the score.

**Quality**

18. Split tests into one file per module. Add a test for every edge case in section 4.
19. Extend `data/` to multi-document cases, including one pair of conflicting sources.
20. Build a held-out set in `data/heldout/` that is never used for tuning. Report development and held-out scores side by side in the README. This removes the "tuned on the test set" caveat.
21. Measure and record speed: seconds per response on CPU, with and without NLI.
22. Rewrite `README.md`: quick start, schema, scoring rules, results on both sets, limits.

## 6. Decisions to confirm before step 1

1. **Restructure in place, or start fresh?** This plan assumes v1 is restructured in place, keeping its history and results.
2. **NLI on by default?** It costs a 281 MB download and more CPU time per claim; it is what catches relation errors such as the Aldrin example.
3. **Streamlit for `app.py`?** It matches StepProof's dashboard. A plain HTML report is the lighter alternative.
