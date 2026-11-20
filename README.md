# Hallucination Detector

Checks an AI-generated response against source documents, claim by claim. It returns a 0–100 reliability score and, for every claim, a label, a confidence, and citations to the exact source sentences used as evidence.

```
[ 3] SUPPORTED    conf=1.00  Neil Armstrong commanded the mission.
      evidence [crew-and-return #0] (0.76): Neil Armstrong served as mission commander, and Buzz Aldrin was ...
[ 4] SUPPORTED    conf=0.97  He was joined on the surface by Buzz Aldrin about 19 minutes later.
      checked as: Neil Armstrong was joined on the surface by Buzz Aldrin about 19 minutes later.
      evidence [mission-summary #3] (0.72): Armstrong became the first person to walk on the Moon, followed by ...
[ 5] CONTRADICTED conf=1.00  The crew brought back about 45 kilograms of lunar samples.
      evidence [crew-and-return #1] (0.83): The astronauts collected about 21.5 kilograms of lunar samples.
      - number mismatch: claim says 45, evidence says 21.5
      - NLI contradiction p=0.99
[ 7] FABRICATED   conf=0.95  The landing site was chosen by a team of 300 geologists at Stanford University.
      - best evidence similarity 0.42
      - not found in source: 300, 'Stanford University'

Claims: 8  (SUPPORTED=4, CONTRADICTED=1, FABRICATED=2, UNSUPPORTED=1)
Hallucination rate: 50%   Reliability score: 51.5/100
```

That is real output for [`examples/request.json`](examples/request.json), shortened to four of its eight claims: one response checked against two documents. Claim 3 was split out of *"Neil Armstrong, who commanded the mission, became the first person..."*, and claim 4's "He" was resolved to "Neil Armstrong" before checking. Claim 7 has no evidence line because no source sentence is close enough to cite; its closest similarity is shown instead.

## How it works

1. **Claims** (`claims.py`, spaCy). The response is split into atomic claims. Coordinated verbs ("designs chips, builds servers, and sells cloud services") and contrastive or relative clauses become separate claims, and a shared "not" is carried into each piece so a split never reverses a claim. A leading he / she / it / they is resolved to the person or thing it refers to. Questions, pleasantries, headings and markdown markers are dropped, and an opener such as "Overall," is cut off while the fact after it is kept.
2. **Evidence** (`sources.py`, `retrieval.py`, Sentence Transformers). Every source document is split into sentences with character offsets. Each claim retrieves its top 3 sentences by cosine similarity across all documents.
3. **Fact checks** (`facts.py`). Numbers, named entities and negation in the claim are compared with the best evidence sentence and with all source text. Numbers are compared by what they count ("14 percent" is not checked against "$412 million"), and equal amounts in different notation ("$2.1 billion", "$2,100 million") match.
4. **Entailment** (`nli.py`, cross-encoder). An NLI model reads each (evidence, claim) pair and says whether the evidence entails or contradicts the claim. When the only difference is how a number is written, the pair is scored again with both numbers in one notation.
5. **Scoring** (`scoring.py`). Rules turn those signals into a label and confidence per claim, then a reliability score for the response.

| Label | Meaning |
|---|---|
| `SUPPORTED` | evidence closely matches and every number and name in the claim is grounded |
| `CONTRADICTED` | evidence covers the same fact and disagrees |
| `FABRICATED` | the claim adds specific numbers or names that appear in no source |
| `UNSUPPORTED` | no source backs the claim (opinion, speculation, off-topic) |

## Quick start

```bash
pip install -e .
python -m spacy download en_core_web_sm
```

The first run downloads about 380 MB of models (sizes below). Everything runs on CPU.

```bash
# check a response against its sources; write an HTML report with highlighted citations
hallucheck detect --request examples/request.json --html examples/report.html

# plain text files work too (repeat --source for more documents)
hallucheck detect --answer examples/answer.txt --source examples/source.txt

# JSON to stdout, without the NLI model
hallucheck --no-nli detect --request examples/request.json --json -

# run the sample cases: realistic factual and hallucinated answers with true labels
hallucheck cases --verbose

# web UI
pip install -e ".[app]"
streamlit run app.py

# evaluation and tests
hallucheck evaluate --data data
hallucheck evaluate --data data/heldout
pip install -e ".[dev]" && pytest -q
```

Every threshold has a flag (`--support-threshold`, `--related-threshold`, `--contradiction-threshold`, `--nli-threshold`, `--top-k`, `--nli-top-k`, `--nli-context-sentences`, `--min-claim-words`); `hallucheck --help` lists them with their defaults. Bad input (a missing file, invalid JSON, an out-of-range threshold) ends with a one-line error and exit status 2.

Python API:

```python
from hallucination_detector import HallucinationDetector

detector = HallucinationDetector()            # HallucinationDetector(use_nli=False) for the light setup
report = detector.detect(response_text, [{"id": "doc1", "title": "...", "text": "..."}])
print(report.reliability_score)
for v in report.verdicts:
    print(v.label, v.confidence, v.claim.text, [c.source_id for c in v.citations], v.reasons)
```

## Input and output

Input (`--request`):

```json
{
  "response": "The crew brought back about 45 kilograms of lunar samples.",
  "sources": [{"id": "crew-and-return", "title": "Crew roles and return to Earth", "text": "..."}]
}
```

Output (`--json`), one entry per claim:

```json
{
  "reliability_score": 51.5,
  "hallucination_rate": 0.5,
  "counts": {"SUPPORTED": 4, "CONTRADICTED": 1, "FABRICATED": 2, "UNSUPPORTED": 1},
  "warnings": [],
  "claims": [{
    "id": 5,
    "text": "The crew brought back about 45 kilograms of lunar samples.",
    "response_span": [221, 278],
    "label": "CONTRADICTED",
    "confidence": 1.0,
    "best_similarity": 0.832,
    "reasons": ["best evidence similarity 0.83", "number mismatch: claim says 45, evidence says 21.5", "NLI contradiction p=0.99"],
    "citations": [{
      "source_id": "crew-and-return",
      "sentence_index": 1,
      "source_span": [88, 151],
      "quote": "The astronauts collected about 21.5 kilograms of lunar samples.",
      "similarity": 0.832,
      "nli": {"contradiction": 0.995, "entailment": 0.001, "neutral": 0.005}
    }]
  }],
  "config": {"embedder": "all-MiniLM-L6-v2", "use_nli": true, "support_threshold": 0.65}
}
```

The full file is [`examples/report.json`](examples/report.json); `claims` and `config` are shortened here.
- `response_span` and `source_span` are character offsets into the original text, so a UI can highlight both sides.
- `citations` lists only sentences similar enough to count as evidence. It is empty when none is, and `best_similarity` still gives the closest match.
- `checked_as` appears when a pronoun was resolved, and `duplicate_of` when a claim repeats an earlier one.
- `config` records every setting behind the result.
- `reliability_score` is `null` when the response has nothing to check, or is not English.

## Scoring rules

For each claim: `s` is the similarity of its best evidence sentence, and `E` / `C` are the NLI entailment / contradiction probabilities. First match wins:

| # | Condition | Label |
|---|---|---|
| 1 | `s ≥ 0.70` and (a number, entity or negation conflict, or `C ≥ 0.70`) | `CONTRADICTED` |
| 2 | the claim has numbers or names found in no source | `FABRICATED` |
| 3 | `s ≥ 0.65`, or (`E ≥ 0.70` and `s ≥ 0.45`) | `SUPPORTED` |
| 4 | otherwise | `UNSUPPORTED` |

- Only the single best sentence can contradict a claim. A neighbor about a different fact would otherwise contradict true claims: a sentence about the 1903 Physics prize "contradicts" a correct claim about the 1911 Chemistry prize.
- Entailment may come from any of the cited sentences.
- **Ambiguous pronouns.** When "She" could mean either of two people ("Marie Curie married Pierre Curie. She died in 1906."), it is left unresolved rather than guessed. Such a claim is `SUPPORTED` only if the sentence that supports it names the remembered subject and not the other person; otherwise it is `UNSUPPORTED` with the reason. An ambiguous "it" is also left unresolved, but evidence about either candidate may support it, because after "The FDA approved Vioxx. It was withdrawn ..." the object is usually what is meant.
- **Reliability score** = 100 × (confidence of supported claims) / (confidence of all claims). A repeated claim counts once.
- Confidences are heuristics, not calibrated probabilities. The formulas are in [`scoring.py`](src/hallucination_detector/scoring.py). Every retrieval, scoring, sentence-splitting and language-check threshold is a field in [`config.py`](src/hallucination_detector/config.py), validated when the detector is built.

## Results

"Positive" means hallucination: any label other than `SUPPORTED`.

| Set | Claims | Configuration | Precision | Recall | F1 | 4-way label accuracy |
|---|---|---|---|---|---|---|
| Development (`data/`) | 50 | embeddings + fact checks | 0.964 | 0.900 | 0.931 | 0.90 |
| Development (`data/`) | 50 | + NLI (default) | 1.000 | 1.000 | 1.000 | 0.98 |
| **Held-out (`data/heldout/`)** | 40 | embeddings + fact checks | 0.895 | 0.810 | 0.850 | 0.75 |
| **Held-out (`data/heldout/`)** | 40 | + NLI (default) | 0.905 | 0.905 | **0.905** | 0.825 |

**The held-out row is the one to trust, with one caveat.** The thresholds were tuned on the development set, so its scores are inflated. The held-out set (5 new domains, including claims that need two documents) was written after the design was frozen. It was first scored at F1 0.884 (with NLI) / 0.829 (without).

The current, higher score came after the review-and-fix rounds below. Those fixes were driven by the code review and the sample cases, never by the held-out claims. But the earlier held-out errors were listed in this README, so the set is no longer a pure first look. A fresh held-out set is the honest next step. Both sets are small and were written by the same person as the detector, so treat every number here as an estimate.

NLI adds about 5 points of F1 on held-out data, all from recall: it catches contradictions that keep every name and number, such as *"Chlorophyll absorbs mostly green light"*.

### Sample cases

[`examples/cases/`](examples/cases/) holds 8 realistic AI answers: for each of four domains (a medication leaflet, a store's return policy, a quarterly earnings report, a research study), one fully factual answer and one with planted hallucinations. Each case has 2 source documents and the true label of every claim (72 claims in all). All names, companies and products are invented. `hallucheck cases` scores the detector against them:

| | Before the review | Now |
|---|---|---|
| Agrees on hallucination vs supported | 69% | **86%** |
| Exact label | 62% | **78%** |
| Factual answers' reliability scores | 62–77 | 71–94 |
| Hallucinated answers' reliability scores | 32–52 | 50–55 |

The remaining misses are mostly loose paraphrases of long sentences (similarity below the support threshold), opinions that sit close to a factual sentence, and one swapped place name the entity tagger does not recognize.

### What it gets wrong on held-out data

The 7 errors with NLI, left unfixed so the held-out set stays held-out:

| Kind | Claim | Cause |
|---|---|---|
| False alarm | "Mount Everest's summit is 8,849 meters..." | the possessive `'s` stops the name matching the source |
| False alarm | "...the first modern Olympics in Athens in 1896." | "Olympics" vs "Olympic Games" read as different events |
| Missed | "Photosynthesis uses oxygen ... and releases carbon dioxide" | roles swapped but every word is in the source; NLI missed it |
| Missed | "The Vega 3 is the best battery on the market." | an opinion that sits close to a factual sentence |
| Wrong label, still flagged | wrong year in two claims | similarity just under 0.70, so `FABRICATED` instead of `CONTRADICTED` |
| Wrong label, still flagged | "...most important chemical process on Earth." | "Earth" treated as an invented name: `FABRICATED` instead of `UNSUPPORTED` |

### Speed

On one CPU (Intel i9-13900KF):

| Workload | Time |
|---|---|
| Example request (8 claims, 8 source sentences), with NLI | 0.44 s per response (about 12 s for the first call, including model load) |
| Same, without NLI | 0.22 s per response |
| A 247,000-character source, first claim / each later claim | 7.4 s / 0.08 s |
| A 2.85-million-character source, first claim / each later claim | 72 s / 0.5 s |

Time grows linearly with source length. The first call on a large source is mostly spaCy splitting it into sentences; after that, the same sources are cached.

## How it was built and checked

v2 was designed in [`PLAN.md`](PLAN.md), then restructured from v1. After the restructure, all 50 development claims got the same label, confidence and similarity as in v1. The code was then audited and fixed in rounds:

1. Six reviewers audited the code against `PLAN.md` in parallel, each on one module group. Every reported defect had to be reproduced by running code, then confirmed by an independent skeptic: **56 defects**. Five fixers repaired them in parallel, in isolated git worktrees.
2. A review of those merged fixes found **23 new defects** introduced by the fixes themselves; a second round fixed them.
3. The next review found **13**. From then on, each fixer's work was reviewed and repaired inside its own worktree before merging.
4. The next review found 10, mostly small, and a targeted round fixed them.
5. A last review of that round found 6, four of them in the ambiguous-pronoun rule; they were fixed with a stricter, simpler rule.

Each round also re-ran every earlier reproduction. Every fix has a regression test, and the suite grew from 125 to 842 tests. Eight further tests are marked strict expected failures and pin one documented limit (below).

## Models

| Model | Role | Size on disk |
|---|---|---|
| `en_core_web_sm` | claim splitting, sentences, entities | 15 MB |
| `sentence-transformers/all-MiniLM-L6-v2` | evidence retrieval | 87 MB |
| `cross-encoder/nli-deberta-v3-xsmall` | entailment vs contradiction | 281 MB |

## Project layout

```
src/hallucination_detector/
  config.py     every model name and threshold, validated on construction
  schemas.py    SourceDocument, Claim, Citation, ClaimVerdict, Report
  claims.py     claim decomposition (spaCy)
  sources.py    document loading (BOM and CRLF safe), sentence splitting with offsets
  retrieval.py  embeddings, cosine top-k, bounded per-document cache
  nli.py        cross-encoder entailment
  facts.py      number, entity and negation checks
  scoring.py    label rules, confidence, which NLI result counts
  pipeline.py   HallucinationDetector: wires the stages together
  report.py     terminal, JSON and HTML output
  evaluate.py   precision / recall / F1, per domain, confusion matrix, threshold sweep
  cases.py      sample cases: load, run, compare with the true labels
  cli.py        the `hallucheck` command
app.py          Streamlit UI
tests/          842 tests, one file per module plus end-to-end edge cases and the sample cases
data/           development set; data/heldout/ is never used for tuning
examples/       a two-document request, its reports, and examples/cases/
```

Each stage imports only `schemas.py` and `config.py`; `pipeline.py` is the only file that knows the order. [`PLAN.md`](PLAN.md) has the design, the edge-case list, and the deliberate deviations from it.

## Limits

- **English only.** Non-English answers are detected and returned with a warning and no score. Exception: a single short German or Spanish sentence that contains a capitalized word can pass as English (pinned by the expected-failure tests).
- **The error causes listed under the held-out results.**
- **Units are not compared:** "21.5 pounds" against "21.5 kilograms" passes the number check and relies on NLI.
- **Pronouns.** Resolution is conservative and gender-blind. When a second person or thing is mentioned as an object ("married Pierre Curie", "leads Tesla"), a later he / she / it is left unresolved rather than guessed and checked as written. A true he / she claim can therefore come out `UNSUPPORTED` when the source never names the person in the supporting sentence.
- **Clause splitting is heuristic.** "X of Y, which ..." with a quantity word ("the price of oil, which ...") is kept as one claim instead of guessing which noun the clause describes. A clause-level "which" with nothing checkable of its own stays with its sentence. An attribution such as ", which the CFO credited to lower costs" becomes its own claim and is checked with the whole sentence, so a wrong number in the main clause can count against both.
- **People counts.** Numbers of people are compared when both sides count people ("300 workers" vs "1,500 people"), even if the groups differ.
- **Retrieval is sentence-level.** A claim that needs two distant sentences to verify may be scored as weaker than it is.
- **Conflicting sources.** When sources disagree, the claim is labeled by the closest sentence and a "sources conflict" warning is added. The detector does not decide which source is right.
- **Hedges are checked, not dropped.** "I think Apollo 11 launched in 1970" is still checked as a claim, because a hedge does not make a false statement true.
