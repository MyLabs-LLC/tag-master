"""Joint decoding over (industry, sub-tag) pairs.

Gating tier 2 behind tier 1 throws away the sharpest signal in the problem. A
document containing "voter registration form" pins the industry far harder than
its general topical vocabulary does, so tier-2 evidence should be allowed to
correct tier-1 mistakes rather than being conditioned on them.

The decoder scores every valid pair

    score(top, tag) = log P(top | x) + lambda * log P(tag | x, top)

and takes the global argmax. `other` carries no sub-tag and so contributes only
its tier-1 term. lambda = 0 reproduces the plain gated pipeline, which makes the
tuned value directly interpretable as how much tier 2 is trusted to override
tier 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .taxonomy import OTHER


@dataclass
class JointPrediction:
    """Decoded output for a batch of documents."""

    top: np.ndarray
    tag: np.ndarray
    score: np.ndarray
    top_confidence: np.ndarray
    tag_confidence: np.ndarray

    def __len__(self) -> int:
        return len(self.top)


@dataclass
class JointDecoder:
    """Combines calibrated tier-1 and tier-2 log-probabilities."""

    top_labels: list[str]
    industries: list[str]
    tags_by_top: dict[str, list[str]]
    lam: float = 0.3
    other_bias: float = 0.0
    _pairs: list[tuple[str, str | None]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._pairs = []
        for top in self.top_labels:
            if top == OTHER:
                self._pairs.append((OTHER, None))
            else:
                for tag in self.tags_by_top.get(top, []):
                    self._pairs.append((top, tag))

    @property
    def pairs(self) -> list[tuple[str, str | None]]:
        return self._pairs

    def decode(
        self,
        tier1_log_proba: np.ndarray,
        tier2_log_proba: dict[str, np.ndarray],
        lam: float | None = None,
        other_bias: float | None = None,
    ) -> JointPrediction:
        """Decode a batch.

        `tier1_log_proba` is (n, 4) aligned to `top_labels`. `tier2_log_proba`
        maps each industry to an (n, n_tags_for_that_industry) matrix; tier 2 is
        evaluated for every industry on every document, not just the one tier 1
        preferred, which is what lets the correction happen.
        """
        lam = self.lam if lam is None else lam
        bias = self.other_bias if other_bias is None else other_bias
        n = tier1_log_proba.shape[0]
        top_index = {t: i for i, t in enumerate(self.top_labels)}

        best_score = np.full(n, -np.inf)
        best_top = np.zeros(n, dtype=np.int32)
        best_tag = np.full(n, -1, dtype=np.int32)
        best_tag_lp = np.zeros(n)

        for top in self.top_labels:
            t1 = tier1_log_proba[:, top_index[top]]
            if top == OTHER:
                score = t1 + bias
                better = score > best_score
                best_score = np.where(better, score, best_score)
                best_top = np.where(better, top_index[top], best_top)
                best_tag = np.where(better, -1, best_tag)
                best_tag_lp = np.where(better, 0.0, best_tag_lp)
                continue

            t2 = tier2_log_proba.get(top)
            if t2 is None or t2.shape[1] == 0:
                continue
            arg = t2.argmax(axis=1)
            rows = np.arange(n)
            tag_lp = t2[rows, arg]
            score = t1 + lam * tag_lp
            better = score > best_score
            best_score = np.where(better, score, best_score)
            best_top = np.where(better, top_index[top], best_top)
            best_tag = np.where(better, arg, best_tag)
            best_tag_lp = np.where(better, tag_lp, best_tag_lp)

        labels = np.array(self.top_labels, dtype=object)
        top_out = labels[best_top]
        tag_out = np.empty(n, dtype=object)
        for i in range(n):
            top = top_out[i]
            idx = best_tag[i]
            tag_out[i] = self.tags_by_top[top][idx] if (top != OTHER and idx >= 0) else None

        return JointPrediction(
            top=top_out,
            tag=tag_out,
            score=best_score,
            top_confidence=np.exp(tier1_log_proba[np.arange(n), best_top]),
            tag_confidence=np.exp(best_tag_lp),
        )

    def top_k_pairs(
        self,
        tier1_log_proba: np.ndarray,
        tier2_log_proba: dict[str, np.ndarray],
        k: int = 3,
        lam: float | None = None,
    ) -> list[list[dict]]:
        """The k best (industry, sub-tag) pairs per document, for inspection."""
        lam = self.lam if lam is None else lam
        n = tier1_log_proba.shape[0]
        top_index = {t: i for i, t in enumerate(self.top_labels)}

        columns: list[tuple[str, str | None, np.ndarray]] = []
        for top in self.top_labels:
            t1 = tier1_log_proba[:, top_index[top]]
            if top == OTHER:
                columns.append((OTHER, None, t1 + self.other_bias))
                continue
            t2 = tier2_log_proba.get(top)
            if t2 is None or t2.shape[1] == 0:
                continue
            for j, tag in enumerate(self.tags_by_top[top]):
                columns.append((top, tag, t1 + lam * t2[:, j]))

        matrix = np.column_stack([c[2] for c in columns])
        order = np.argsort(-matrix, axis=1)[:, :k]
        out: list[list[dict]] = []
        for i in range(n):
            out.append(
                [
                    {
                        "industry": columns[j][0],
                        "sub_tag": columns[j][1],
                        "score": float(matrix[i, j]),
                    }
                    for j in order[i]
                ]
            )
        return out


def tune_lambda(
    decoder: JointDecoder,
    tier1_log_proba: np.ndarray,
    tier2_log_proba: dict[str, np.ndarray],
    gold_top: np.ndarray,
    gold_tag: np.ndarray,
    grid: list[float],
    objective: str = "end_to_end",
) -> tuple[float, dict[float, dict[str, float]]]:
    """Pick lambda on dev.

    The default objective is end-to-end exact match (both tiers right on the
    same document), since optimising either tier alone would let the decoder
    trade one against the other.
    """
    history: dict[float, dict[str, float]] = {}
    best_lam, best_value = grid[0], -np.inf

    for lam in grid:
        pred = decoder.decode(tier1_log_proba, tier2_log_proba, lam=lam)
        top_ok = pred.top == gold_top
        industry_rows = gold_top != OTHER
        both_ok = top_ok & (pred.tag == gold_tag)
        metrics = {
            "tier1_accuracy": float(top_ok.mean()),
            "end_to_end": float(
                (both_ok | (top_ok & ~industry_rows)).mean()
            ),
            "industry_end_to_end": float(both_ok[industry_rows].mean())
            if industry_rows.any()
            else 0.0,
        }
        history[lam] = metrics
        value = metrics[objective]
        if value > best_value:
            best_value, best_lam = value, lam

    decoder.lam = best_lam
    return best_lam, history
