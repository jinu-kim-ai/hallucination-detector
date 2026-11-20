"""Natural-language inference: does the evidence entail or contradict the claim?"""
from __future__ import annotations

import numpy as np

from .models import load_pretrained

LABELS = ["contradiction", "entailment", "neutral"]     # output order of the cross-encoder/nli-* models


class NLIScorer:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-xsmall"):
        from sentence_transformers import CrossEncoder

        self.model = load_pretrained(CrossEncoder, model_name, "NLI")
        self.name = model_name

    def score(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """pairs: (premise = evidence, hypothesis = claim). Returns one probability dict per pair."""
        if not pairs:
            return []
        logits = np.asarray(self.model.predict(pairs))
        logits = logits.reshape(len(pairs), -1)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return [dict(zip(LABELS, map(float, row))) for row in probs]
