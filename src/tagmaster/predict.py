"""Single-document inference.

Note what is *not* used here: `document_description`. It names the source domain
in 93% of rows and feeds the tier-2 prototypes during training, but it is
dataset metadata rather than part of the document, so it never appears at
inference. The only input is the document text.
"""

from __future__ import annotations

import numpy as np

from .bundle import ModelBundle
from .config import Config
from .taxonomy import OTHER


class Predictor:
    """Loads a trained bundle and classifies raw document text."""

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    @classmethod
    def load(cls, cfg: Config) -> Predictor:
        return cls(ModelBundle.load(cfg))

    def _score(self, texts: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        arr = np.array(texts, dtype=object)
        b = self.bundle

        embeddings = b.tier1_space.embed(arr) if b.tier1_space.uses_embeddings else None
        X1 = b.tier1_space.combine(b.tier1_space.transform_sparse(arr), embeddings)
        tier1_lp = b.tier1.log_proba(X1)

        X2 = b.tier2_space.combine(b.tier2_space.transform_sparse(arr), embeddings)
        tier2_lp: dict[str, np.ndarray] = {}
        for top, head in b.tier2.items():
            lexical = b.matcher.score_matrix(arr, top).channels()
            channels = head.channel_scores(X2, embeddings, lexical)
            tier2_lp[top] = head.log_proba(head.combine(channels, head.weights))
        return tier1_lp, tier2_lp

    def predict_batch(self, texts: list[str], top_k: int = 3) -> list[dict]:
        tier1_lp, tier2_lp = self._score(texts)
        decoded = self.bundle.decoder.decode(tier1_lp, tier2_lp)
        alternatives = self.bundle.decoder.top_k_pairs(tier1_lp, tier2_lp, k=top_k)

        labels = list(self.bundle.taxonomy.top_labels)
        out = []
        for i in range(len(texts)):
            out.append(
                {
                    "industry": decoded.top[i],
                    "sub_tag": decoded.tag[i],
                    "industry_confidence": round(float(decoded.top_confidence[i]), 4),
                    "sub_tag_confidence": round(float(decoded.tag_confidence[i]), 4)
                    if decoded.top[i] != OTHER
                    else None,
                    "industry_probabilities": {
                        label: round(float(np.exp(tier1_lp[i, j])), 4)
                        for j, label in enumerate(labels)
                    },
                    "alternatives": alternatives[i],
                }
            )
        return out

    def predict(self, text: str, top_k: int = 3) -> dict:
        return self.predict_batch([text], top_k=top_k)[0]
