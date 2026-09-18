from __future__ import annotations

import numpy as np

from tagmaster.canonicalize import (
    ConfusionGraph,
    build_aliases,
    score_with_aliases,
    sweep,
)


def _graph(tags, pairs, support=40):
    """Build a confusion graph where `pairs` are mutually confused."""
    k = len(tags)
    index = {t: i for i, t in enumerate(tags)}
    gold, pred = [], []
    for t in tags:
        for _ in range(support):
            gold.append(index[t])
            pred.append(index[t])
    for a, b, n in pairs:
        for _ in range(n):
            gold.append(index[a])
            pred.append(index[b])
            gold.append(index[b])
            pred.append(index[a])
    return ConfusionGraph.from_predictions(tags, np.array(gold), np.array(pred))


def test_affinity_is_symmetric_with_zero_diagonal():
    g = _graph(["a", "b", "c"], [("a", "b", 20)])
    assert np.allclose(g.affinity, g.affinity.T)
    assert np.allclose(np.diag(g.affinity), 0.0)
    assert g.affinity[0, 1] > g.affinity[0, 2]


def test_confused_pair_is_merged_and_named_after_the_larger_member():
    tags = ["investment plan", "investment strategy", "passport"]
    gold = np.array([0] * 20 + [1] * 60 + [2] * 40)
    pred = np.array([1] * 20 + [1] * 60 + [2] * 40)
    graph = ConfusionGraph.from_predictions(tags, gold, pred)
    result = build_aliases({"finance": graph}, 0.1, 0.5, 8)
    assert result.aliases["finance\tinvestment plan"] == "investment strategy"
    assert result.n_tags_after == 2


def test_unconfused_tags_are_left_alone():
    g = _graph(["a", "b", "c"], [])
    result = build_aliases({"x": g}, 0.1, 0.5, 8)
    assert result.aliases == {}
    assert result.n_tags_after == result.n_tags_before == 3


def test_max_cluster_size_caps_growth():
    tags = [f"t{i}" for i in range(10)]
    pairs = [(tags[i], tags[j], 30) for i in range(10) for j in range(i + 1, 10)]
    g = _graph(tags, pairs, support=5)
    capped = build_aliases({"x": g}, 0.01, 0.0, 3)
    sizes = [len(c) for c in capped.clusters["x"]]
    assert max(sizes) <= 3


def test_complete_linkage_guard_prevents_chaining():
    # The observed failure mode: a chain of individually plausible links
    # dragging unrelated tags into one cluster. Each neighbouring pair is
    # strongly confused but the ends of the chain are not confused at all.
    # Note the aggressive threshold: this is where average linkage fails, since
    # the mean across cross pairs still clears a low bar even when half of them
    # are zero.
    tags = ["a", "b", "c", "d", "e"]
    chain = [("a", "b", 40), ("b", "c", 40), ("c", "d", 40), ("d", "e", 40)]
    g = _graph(tags, chain, support=10)

    permissive = build_aliases({"x": g}, 0.05, 0.0, 99)
    guarded = build_aliases({"x": g}, 0.05, 0.5, 99)

    assert max(len(c) for c in permissive.clusters["x"]) == 5
    assert max(len(c) for c in guarded.clusters["x"]) == 2
    # With the guard active, the two ends of the chain must not co-cluster.
    for cluster in guarded.clusters["x"]:
        assert not ({"a", "e"} <= set(cluster))


def test_complete_linkage_blocks_never_confused_pairs_at_any_threshold():
    # Two tags with zero mutual confusion must never share a cluster, however
    # aggressive the threshold gets.
    tags = ["a", "b", "c"]
    g = _graph(tags, [("a", "b", 50), ("b", "c", 50)], support=5)
    for threshold in (0.5, 0.25, 0.08, 0.01):
        result = build_aliases({"x": g}, threshold, 0.5, 99)
        for cluster in result.clusters["x"]:
            assert not ({"a", "c"} <= set(cluster))


def test_higher_threshold_merges_less():
    tags = ["a", "b", "c", "d"]
    g = _graph(tags, [("a", "b", 30), ("c", "d", 6)], support=30)
    loose = build_aliases({"x": g}, 0.02, 0.5, 8)
    strict = build_aliases({"x": g}, 0.3, 0.5, 8)
    assert loose.n_tags_after <= strict.n_tags_after


def test_score_with_aliases_credits_within_cluster_predictions():
    aliases = {"finance\tinvestment plan": "investment strategy"}
    rows = [
        ("finance", "investment plan", "investment strategy"),
        ("finance", "investment plan", "passport"),
    ]
    assert score_with_aliases(aliases, rows) == 0.5
    # Without the alias both rows are wrong.
    assert score_with_aliases({}, rows) == 0.0


def test_score_with_aliases_handles_empty_input():
    assert score_with_aliases({}, []) == 0.0


def test_sweep_reports_monotonic_granularity():
    tags = ["a", "b", "c", "d"]
    g = _graph(tags, [("a", "b", 30), ("c", "d", 10)], support=30)
    rows = [("x", "a", "b"), ("x", "c", "d"), ("x", "a", "a")]
    report, by_threshold = sweep({"x": g}, rows, [0.4, 0.2, 0.05], 0.5, 8)
    counts = [r.n_tags_after for r in report.results]
    assert counts == sorted(counts, reverse=True)
    assert all(0.0 <= r.accuracy <= 1.0 for r in report.results)
    assert set(by_threshold) == {0.4, 0.2, 0.05}


def test_merged_only_hides_singletons():
    tags = ["a", "b", "c"]
    g = _graph(tags, [("a", "b", 40)], support=10)
    result = build_aliases({"x": g}, 0.05, 0.5, 8)
    merged = result.merged_only()["x"]
    assert all(len(c) > 1 for c in merged)
