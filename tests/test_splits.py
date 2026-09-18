"""Tests for the leakage invariant, which is what keeps every metric honest."""

from __future__ import annotations

import numpy as np
import pytest

from tagmaster.data import Splits, _stratified_uid_assignment, assert_no_uid_leakage


def _frame(frame_factory, uids, tops):
    return frame_factory(
        [{"uid": u, "top": t, "locale": "us", "document_format": "structured"} for u, t in zip(uids, tops)]
    )


def test_assert_no_uid_leakage_passes_on_disjoint_splits(frame_factory):
    splits = Splits(
        train=_frame(frame_factory, ["a", "a", "b"], ["finance"] * 3),
        dev=_frame(frame_factory, ["c", "c"], ["other"] * 2),
        test=_frame(frame_factory, ["d"], ["healthcare"]),
    )
    assert_no_uid_leakage(splits)


def test_assert_no_uid_leakage_catches_a_shared_uid(frame_factory):
    # This is the exact failure mode a row-level split would produce: the `us`
    # row in train and the `intl` row of the same record in test.
    splits = Splits(
        train=_frame(frame_factory, ["a"], ["finance"]),
        dev=_frame(frame_factory, ["b"], ["other"]),
        test=_frame(frame_factory, ["a"], ["finance"]),
    )
    with pytest.raises(AssertionError, match="appear in both train and test"):
        assert_no_uid_leakage(splits)


def test_stratified_assignment_covers_every_uid_exactly_once():
    uids = np.array([f"u{i}" for i in range(200)], dtype=object)
    strata = np.array(["a" if i % 2 else "b" for i in range(200)], dtype=object)
    assignment = _stratified_uid_assignment(uids, strata, (0.7, 0.15, 0.15), seed=7)
    assert set(assignment) == set(uids.tolist())
    assert set(assignment.values()) <= {"train", "dev", "test"}


def test_stratified_assignment_balances_each_stratum():
    uids = np.array([f"u{i}" for i in range(1000)], dtype=object)
    strata = np.array(["rare" if i < 100 else "common" for i in range(1000)], dtype=object)
    assignment = _stratified_uid_assignment(uids, strata, (0.7, 0.15, 0.15), seed=1)

    for name, size in (("rare", 100), ("common", 900)):
        members = [u for u, s in zip(uids.tolist(), strata.tolist()) if s == name]
        counts = {"train": 0, "dev": 0, "test": 0}
        for u in members:
            counts[assignment[u]] += 1
        assert counts["train"] == pytest.approx(0.7 * size, abs=2)
        assert counts["dev"] == pytest.approx(0.15 * size, abs=2)
        assert counts["test"] == pytest.approx(0.15 * size, abs=2)


def test_tiny_stratum_still_yields_a_training_example():
    # A sub-tag with two records must not end up entirely in test.
    uids = np.array(["x", "y", "z"], dtype=object)
    strata = np.array(["only"] * 3, dtype=object)
    assignment = _stratified_uid_assignment(uids, strata, (0.7, 0.15, 0.15), seed=3)
    assert "train" in assignment.values()


def test_assignment_is_deterministic_for_a_seed():
    uids = np.array([f"u{i}" for i in range(50)], dtype=object)
    strata = np.array(["a"] * 50, dtype=object)
    a = _stratified_uid_assignment(uids, strata, (0.7, 0.15, 0.15), seed=11)
    b = _stratified_uid_assignment(uids, strata, (0.7, 0.15, 0.15), seed=11)
    c = _stratified_uid_assignment(uids, strata, (0.7, 0.15, 0.15), seed=12)
    assert a == b
    assert a != c
