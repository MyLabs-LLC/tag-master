"""Confusion-driven tag canonicalisation.

Some sub-tags in this dataset are not merely hard to tell apart, they are
genuinely indistinguishable: finance carries "investment plan", "investment
strategy", "investment roadmap", and "investment approach" as separate labels
for documents no reader could reliably sort. Merging those is not cheating, it
is admitting the label space is finer than the evidence -- but it has to be done
from measured confusion rather than string similarity, and it has to be stopped
from chaining.

The chaining failure is concrete and was observed while designing this: merging
whenever one tag's token set contains another's collapses 120 finance tags into
a single cluster running from "investment strategy" all the way to "savings
account", because each neighbouring pair looks plausible. Two guards prevent it:

* complete linkage -- *every* cross-cluster pair must clear the bar, not just
  the pair that triggered the merge. This is the standard remedy for
  single-linkage chaining, and it is what makes the guard hold at aggressive
  thresholds: average linkage would let one strong link (b-c) drag in tags whose
  mutual affinity is zero, because the mean across the four cross pairs still
  clears a low threshold.
* a hard cap on cluster size.

Output is a reviewable alias map, and `evaluate` reports accuracy at both the
raw and canonical granularity so the trade is explicit rather than hidden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MergeResult:
    threshold: float
    clusters: dict[str, list[list[str]]]
    aliases: dict[str, str]
    n_tags_before: int
    n_tags_after: int
    accuracy: float | None = None

    def merged_only(self) -> dict[str, list[list[str]]]:
        return {
            top: [c for c in groups if len(c) > 1] for top, groups in self.clusters.items()
        }


@dataclass
class ConfusionGraph:
    """Symmetric confusion affinity between the tags of one industry."""

    tags: list[str]
    affinity: np.ndarray
    support: np.ndarray

    @classmethod
    def from_predictions(
        cls, tags: list[str], gold: np.ndarray, predicted: np.ndarray
    ) -> ConfusionGraph:
        k = len(tags)
        counts = np.zeros((k, k), dtype=np.float64)
        for g, p in zip(gold, predicted):
            if 0 <= g < k and 0 <= p < k:
                counts[g, p] += 1
        support = counts.sum(axis=1)
        denom = support[:, None] + support[None, :]
        symmetric = counts + counts.T
        np.fill_diagonal(symmetric, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            affinity = np.where(denom > 0, symmetric / np.maximum(denom, 1e-9), 0.0)
        np.fill_diagonal(affinity, 0.0)
        return cls(tags=tags, affinity=affinity, support=support)


def _merge_industry(
    graph: ConfusionGraph,
    threshold: float,
    linkage_ratio: float,
    max_cluster_size: int,
) -> list[list[str]]:
    """Agglomerative merging under the complete-linkage and size guards."""
    k = len(graph.tags)
    clusters: dict[int, set[int]] = {i: {i} for i in range(k)}
    owner = list(range(k))
    bar = threshold * linkage_ratio

    candidates = [
        (graph.affinity[i, j], i, j)
        for i in range(k)
        for j in range(i + 1, k)
        if graph.affinity[i, j] >= threshold
    ]
    candidates.sort(reverse=True)

    for _, i, j in candidates:
        a, b = owner[i], owner[j]
        if a == b:
            continue
        left, right = clusters[a], clusters[b]
        if len(left) + len(right) > max_cluster_size:
            continue
        # Complete linkage: the weakest cross-cluster pair decides. Two tags that
        # were never confused with each other have affinity 0 and so can never
        # end up together, however strong the link that proposed the merge.
        weakest = min(graph.affinity[x, y] for x in left for y in right)
        if weakest < bar:
            continue
        merged = left | right
        clusters[a] = merged
        del clusters[b]
        for member in merged:
            owner[member] = a

    return [sorted(members) for members in clusters.values()]


def _canonical_name(graph: ConfusionGraph, members: list[int]) -> str:
    """Name a cluster after its best-supported member, tie-broken by length."""
    best = max(members, key=lambda i: (graph.support[i], -len(graph.tags[i])))
    return graph.tags[best]


def build_aliases(
    graphs: dict[str, ConfusionGraph],
    threshold: float,
    linkage_ratio: float,
    max_cluster_size: int,
) -> MergeResult:
    clusters: dict[str, list[list[str]]] = {}
    aliases: dict[str, str] = {}
    before = after = 0

    for top, graph in graphs.items():
        groups = _merge_industry(graph, threshold, linkage_ratio, max_cluster_size)
        named: list[list[str]] = []
        for members in groups:
            canonical = _canonical_name(graph, members)
            names = [graph.tags[i] for i in members]
            named.append(sorted(names))
            for name in names:
                if name != canonical:
                    aliases[f"{top}\t{name}"] = canonical
        clusters[top] = sorted(named)
        before += len(graph.tags)
        after += len(groups)

    return MergeResult(
        threshold=threshold,
        clusters=clusters,
        aliases=aliases,
        n_tags_before=before,
        n_tags_after=after,
    )


def score_with_aliases(
    aliases: dict[str, str],
    rows: list[tuple[str, str, str]],
) -> float:
    """Accuracy after folding aliases in.

    `rows` are (industry, gold_tag, predicted_tag). A prediction counts as
    correct when gold and prediction land in the same cluster.
    """
    if not rows:
        return 0.0
    hits = 0
    for top, gold, pred in rows:
        g = aliases.get(f"{top}\t{gold}", gold)
        p = aliases.get(f"{top}\t{pred}", pred)
        hits += int(g == p)
    return hits / len(rows)


@dataclass
class CanonicalizationReport:
    results: list[MergeResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "granularity_tradeoff": [
                {
                    "threshold": r.threshold,
                    "tags_before": r.n_tags_before,
                    "tags_after": r.n_tags_after,
                    "tier2_accuracy": None if r.accuracy is None else round(r.accuracy, 5),
                }
                for r in self.results
            ]
        }

    def write(self, path: Path, chosen: MergeResult | None = None) -> None:
        blob = self.to_dict()
        if chosen is not None:
            blob["chosen"] = {
                "threshold": chosen.threshold,
                "tags_after": chosen.n_tags_after,
                "merged_clusters": chosen.merged_only(),
            }
        path.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def sweep(
    graphs: dict[str, ConfusionGraph],
    eval_rows: list[tuple[str, str, str]],
    thresholds: list[float],
    linkage_ratio: float,
    max_cluster_size: int,
) -> tuple[CanonicalizationReport, dict[float, MergeResult]]:
    """Measure the accuracy/granularity trade across the threshold grid."""
    report = CanonicalizationReport()
    by_threshold: dict[float, MergeResult] = {}
    for threshold in thresholds:
        result = build_aliases(graphs, threshold, linkage_ratio, max_cluster_size)
        result.accuracy = score_with_aliases(result.aliases, eval_rows)
        report.results.append(result)
        by_threshold[threshold] = result
    return report, by_threshold
