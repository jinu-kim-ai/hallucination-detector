# Hallucination Detector

Claim-level hallucination detection for LLM answers that are supposed to be grounded in a source document.

Given a **source document** and a **model-generated answer**, the detector:

1. **Decomposes** the answer into atomic factual claims (spaCy dependency parse: coordinated clauses such as *"Curie discovered polonium and won two Nobel Prizes"* become two claims; questions and filler like *"I hope this helps!"* are dropped).
2. **Retrieves evidence** for each claim with Sentence Transformers embeddings (`all-MiniLM-L6-v2`, cosine similarity over source sentences).
3. **Checks facts** in the claim against the evidence: numbers, named entities (people, places, organizations, dates), and negation on a shared verb.
4. *(optional, `--nli`)* Scores the claim against its best evidence with a cross-encoder NLI model (`cross-encoder/nli-deberta-v3-xsmall`).
5. **Labels** each claim with a confidence and a human-readable reason:

| Label | Meaning | Example reason |
|---|---|---|
| `SUPPORTED` | Closely matching evidence, every fact grounded | `best evidence similarity 0.95` |
| `CONTRADICTED` | Evidence covers the same fact but disagrees | `number mismatch: claim says 45, evidence says 21.5` |
| `FABRICATED` | Introduces specific facts that appear nowhere in the source | `not found in source: 300, 'Stanford University'` |
| `UNSUPPORTED` | Nothing in the source backs it (opinion, speculation) | `no source sentence reaches support threshold 0.65` |

Results are rendered in the terminal or as an **HTML report**. The report highlights each answer sentence by its most severe label, with highlight strength set by confidence. It also shows a 0–100 **reliability score**: the confidence-weighted share of grounded claims.

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# check one answer against one document
python -m hallucination_detector detect --source examples/source.txt --answer examples/answer.txt --html examples/report.html

# same, plus the NLI cross-encoder
python -m hallucination_detector --nli detect --source examples/source.txt --answer examples/answer.txt

# benchmark on the labeled dataset (precision / recall / F1, per-domain, confusion matrix, threshold sweep)
python -m hallucination_detector evaluate --data data --sweep
python -m hallucination_detector --nli evaluate --data data

pytest -q
```

Python API:

```python
from hallucination_detector import HallucinationDetector

det = HallucinationDetector(use_nli=True)
report = det.detect(answer_text, source_text)
for r in report.results:
    print(r.label, r.confidence, r.claim, r.reasons)
print(report.reliability_score)
```

## Example output

```
[ 5] CONTRADICTED conf=1.00  The crew brought back about 45 kilograms of lunar samples.
      evidence (0.83): The astronauts collected about 21.5 kilograms of lunar samples.
      - number mismatch: claim says 45, evidence says 21.5
      - NLI contradiction p=0.99
[ 7] FABRICATED   conf=0.95  The landing site was chosen by a team of 300 geologists at Stanford University.
      - not found in source: 300, 'Stanford University'
[ 8] UNSUPPORTED  conf=0.88  Many historians consider it the greatest engineering achievement of the century.

Claims: 8  (SUPPORTED=4, CONTRADICTED=2, FABRICATED=1, UNSUPPORTED=1)
Hallucination rate: 50%   Reliability score: 49.9/100
```

## Evaluation

`data/` contains 50 hand-labeled claims across 5 domains (science, history, medicine, technology, finance). There are 20 supported, 15 contradicted, 10 fabricated and 5 unsupported claims. "Positive" means hallucination, i.e. any non-`SUPPORTED` label.

| Configuration | Precision | Recall | F1 | 4-way label accuracy |
|---|---|---|---|---|
| Embeddings + fact checks | 0.964 | 0.900 | 0.931 | 0.90 |
| + NLI cross-encoder (`--nli`) | 1.000 | 1.000 | 1.000 | 0.98 |

What NLI adds: some contradictions keep every entity and number but change the meaning. Examples are *"Nobel Prize in **Chemistry**"* where the source says Physics, *"**Aldrin** was the first to walk on the Moon"*, and *"demand **weakened**"* where the source says it remained strong. Lexical fact checks can't see these; the NLI model catches them.

**Caveat:** the thresholds (`support=0.65`, `contradiction=0.70`) were tuned on this same 50-claim set, so these numbers are optimistic. The next step is a larger held-out benchmark, for example a public hallucination dataset converted to `labeled_claims.jsonl` format.

## Project layout

```
hallucination_detector/
  claims.py      claim decomposition (spaCy)
  embeddings.py  Sentence Transformers backend (+ hashing fallback for offline use)
  detector.py    evidence retrieval, fact checks, labeling, confidence
  report.py      terminal + HTML rendering
  evaluate.py    precision / recall / F1, per-domain metrics, confusion matrix, threshold sweep
data/            sources.json + labeled_claims.jsonl
examples/        sample source / answer / generated HTML report
tests/
```

## Limitations

- Claim splitting handles coordinated verb clauses, not every compound sentence (relative clauses, lists of facts in one noun phrase).
- Entity checks depend on spaCy's small NER model, which sometimes mislabels entities.
- Retrieval is sentence-level. A claim that combines facts from two distant sentences can look weaker than it is.
