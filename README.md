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
      - not found in source: 300, 'Stanford University'

Claims: 8  (SUPPORTED=4, CONTRADICTED=1, FABRICATED=2, UNSUPPORTED=1)
Hallucination rate: 50%   Reliability score: 51.5/100
```

That is real output for [`examples/request.json`](examples/request.json): one response checked against two documents. Claim 3 was split out of *"Neil Armstrong, who commanded the mission, became the first person..."*, and claim 4's "He" was resolved to "Neil Armstrong" before checking.

## How it works

1. **Claims** (`claims.py`, spaCy). The response is split into atomic claims: coordinated verbs and non-restrictive relative clauses become separate claims, leading pronouns are resolved, and questions, pleasantries, headings and markdown markers are dropped.
2. **Evidence** (`sources.py`, `retrieval.py`, Sentence Transformers). Every source document is split into sentences with character offsets. Each claim retrieves its top 3 sentences by cosine similarity across all documents.
3. **Fact checks** (`facts.py`). Numbers, named entities and negation in the claim are compared with the best evidence sentence and with all source text.
4. **Entailment** (`nli.py`, cross-encoder). An NLI model reads each (evidence, claim) pair and says whether the evidence entails or contradicts the claim.
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

# web UI
pip install -e ".[app]"
streamlit run app.py

# evaluation and tests
hallucheck evaluate --data data
hallucheck evaluate --data data/heldout
pip install -e ".[dev]" && pytest -q
```

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

The full file is [`examples/report.json`](examples/report.json); `claims` and `config` are shortened here. `response_span` and `source_span` are character offsets, so a UI can highlight both sides. `config` records every setting behind the result. `reliability_score` is `null` when the response has nothing to check.

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
- **Reliability score** = 100 × (confidence of supported claims) / (confidence of all claims). A repeated claim counts once.
- Confidences are heuristics, not calibrated probabilities. The formulas are in [`scoring.py`](src/hallucination_detector/scoring.py), and every threshold is in [`config.py`](src/hallucination_detector/config.py).

## Results

"Positive" means hallucination: any label other than `SUPPORTED`.

| Set | Claims | Configuration | Precision | Recall | F1 | 4-way label accuracy |
|---|---|---|---|---|---|---|
| Development (`data/`) | 50 | embeddings + fact checks | 0.964 | 0.900 | 0.931 | 0.90 |
| Development (`data/`) | 50 | + NLI (default) | 1.000 | 1.000 | 1.000 | 0.98 |
| **Held-out (`data/heldout/`)** | 40 | embeddings + fact checks | 0.850 | 0.810 | 0.829 | 0.75 |
| **Held-out (`data/heldout/`)** | 40 | + NLI (default) | 0.864 | 0.905 | **0.884** | 0.80 |

**The held-out row is the one to trust.** The thresholds were tuned on the development set, so its scores are inflated. The held-out set (5 new domains, including claims that need two documents) was written after the design was frozen and scored once; nothing was changed in response to it. Both sets are small and were written by the same person as the detector, so treat even the held-out numbers as an estimate.

NLI adds about 5 points of F1 on held-out data, all from recall: it catches contradictions that keep every name and number, such as *"Chlorophyll absorbs mostly green light"*.

### What it got wrong on held-out data

All 8 errors (with NLI), left unfixed on purpose so the held-out set stays held-out:

| Kind | Claim | Cause |
|---|---|---|
| False alarm | "Mount Everest's summit is 8,849 meters..." | the possessive `'s` stops the name matching the source |
| False alarm | "...the first modern Olympics in Athens in 1896." | "Olympics" vs "Olympic Games" read as different events |
| False alarm | "The Vega 3 battery stores 85 kilowatt-hours." | the 3 in the product name is treated as a number to verify |
| Missed | "Photosynthesis uses oxygen ... and releases carbon dioxide" | roles swapped but every word is in the source; NLI missed it |
| Missed | "The Vega 3 is the best battery on the market." | an opinion that sits close to a factual sentence |
| Wrong label, still flagged | wrong year in two claims | similarity just under 0.70, so `FABRICATED` instead of `CONTRADICTED` |
| Wrong label, still flagged | "...most important chemical process on Earth." | "Earth" treated as an invented name: `FABRICATED` instead of `UNSUPPORTED` |

### Speed

On one CPU (Intel i9-13900KF), for the example request (8 claims, 8 source sentences):

| Configuration | Per response | First call, including model load |
|---|---|---|
| with NLI | 0.62 s | about 11 s |
| without NLI | 0.25 s | about 1 s (models already on disk) |

## Models

| Model | Role | Size on disk |
|---|---|---|
| `en_core_web_sm` | claim splitting, sentences, entities | 15 MB |
| `sentence-transformers/all-MiniLM-L6-v2` | evidence retrieval | 87 MB |
| `cross-encoder/nli-deberta-v3-xsmall` | entailment vs contradiction | 281 MB |

## Project layout

```
src/hallucination_detector/
  config.py     every model name and threshold
  schemas.py    SourceDocument, Claim, Citation, ClaimVerdict, Report
  claims.py     claim decomposition (spaCy)
  sources.py    document loading, sentence splitting with offsets
  retrieval.py  embeddings, cosine top-k, per-document cache
  nli.py        cross-encoder entailment
  facts.py      number, entity and negation checks
  scoring.py    label rules, confidence, which NLI result counts
  pipeline.py   HallucinationDetector: wires the stages together
  report.py     terminal, JSON and HTML output
  evaluate.py   precision / recall / F1, per domain, confusion matrix, threshold sweep
  cli.py        the `hallucheck` command
app.py          Streamlit UI
tests/          125 tests, one file per module plus end-to-end edge cases
data/           development set; data/heldout/ is never used for tuning
```

Each stage imports only `schemas.py` and `config.py`; `pipeline.py` is the only file that knows the order. [`PLAN.md`](PLAN.md) has the design and the edge-case list.

## Limits

- English only. Other languages are detected and returned with a warning and no score.
- The seven error causes listed above.
- Units are not compared: "21.5 pounds" against "21.5 kilograms" passes the number check and relies on NLI.
- Pronoun resolution handles a leading he / she / it / they by the most recent matching subject. It is not full coreference.
- Relative clauses are split only when set off by commas and introduced by who / which / that as the subject.
- Retrieval is sentence-level. A claim that needs two distant sentences to verify may be scored as weaker than it is.
- When sources disagree, the claim is labeled by the closest sentence and a "sources conflict" warning is added; the detector does not decide which source is right.
