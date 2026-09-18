from __future__ import annotations

import numpy as np

from tagmaster.hierarchy import JointDecoder, tune_lambda
from tagmaster.taxonomy import OTHER

TOPS = ["healthcare", "finance", "government", OTHER]
TAGS = {
    "healthcare": ["discharge summary", "prescription"],
    "finance": ["bank statement", "loan application"],
    "government": ["passport", "voter registration"],
}


def _decoder(lam: float = 0.3) -> JointDecoder:
    return JointDecoder(
        top_labels=list(TOPS), industries=TOPS[:3], tags_by_top=dict(TAGS), lam=lam
    )


def _log_probs(rows: list[list[float]]) -> np.ndarray:
    arr = np.array(rows, dtype=np.float64)
    return arr - np.log(np.exp(arr).sum(axis=1, keepdims=True))


def test_pairs_enumerate_every_valid_combination():
    decoder = _decoder()
    # Six industry sub-tags plus a single bare `other`.
    assert len(decoder.pairs) == 7
    assert (OTHER, None) in decoder.pairs
    assert ("healthcare", "prescription") in decoder.pairs


def test_decode_output_shapes_and_types():
    decoder = _decoder()
    t1 = _log_probs([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]])
    t2 = {top: _log_probs([[1.0, 0.0], [0.0, 1.0]]) for top in TAGS}
    pred = decoder.decode(t1, t2)
    assert len(pred) == 2
    assert pred.top.shape == (2,)
    assert pred.tag.shape == (2,)
    assert pred.top[0] == "healthcare"
    assert pred.tag[0] in TAGS["healthcare"]


def test_other_carries_no_sub_tag():
    decoder = _decoder()
    t1 = _log_probs([[0.0, 0.0, 0.0, 9.0]])
    t2 = {top: _log_probs([[1.0, 0.0]]) for top in TAGS}
    pred = decoder.decode(t1, t2)
    assert pred.top[0] == OTHER
    assert pred.tag[0] is None


def test_lambda_zero_reproduces_the_gated_pipeline():
    # With lambda 0 the sub-tag evidence cannot influence the industry, which is
    # exactly the behaviour a plain tier1-then-tier2 pipeline would have.
    decoder = _decoder(lam=0.0)
    t1 = _log_probs([[1.0, 0.9, 0.0, 0.0]])
    t2 = {
        "healthcare": _log_probs([[0.0, 0.0]]),
        "finance": _log_probs([[9.0, 0.0]]),
        "government": _log_probs([[0.0, 0.0]]),
    }
    assert decoder.decode(t1, t2).top[0] == "healthcare"


def test_sharp_sub_tag_evidence_overrides_a_close_tier1_call():
    # The reason the decoder exists: tier 1 marginally prefers healthcare, but
    # the finance sub-tag head is emphatic, so the pair score should flip it.
    decoder = _decoder(lam=1.0)
    t1 = _log_probs([[1.0, 0.9, 0.0, 0.0]])
    t2 = {
        "healthcare": _log_probs([[0.0, 0.0]]),
        "finance": _log_probs([[9.0, 0.0]]),
        "government": _log_probs([[0.0, 0.0]]),
    }
    pred = decoder.decode(t1, t2)
    assert pred.top[0] == "finance"
    assert pred.tag[0] == "bank statement"


def test_other_bias_shifts_the_boundary():
    decoder = _decoder(lam=0.3)
    t1 = _log_probs([[1.0, 0.0, 0.0, 0.8]])
    t2 = {top: _log_probs([[1.0, 0.0]]) for top in TAGS}
    assert decoder.decode(t1, t2).top[0] == "healthcare"
    assert decoder.decode(t1, t2, other_bias=5.0).top[0] == OTHER


def test_confidences_are_probabilities():
    decoder = _decoder()
    t1 = _log_probs([[2.0, 0.0, 0.0, 0.0]])
    t2 = {top: _log_probs([[1.0, 0.0]]) for top in TAGS}
    pred = decoder.decode(t1, t2)
    assert 0.0 <= pred.top_confidence[0] <= 1.0
    assert 0.0 <= pred.tag_confidence[0] <= 1.0


def test_top_k_pairs_are_ranked_and_sized():
    decoder = _decoder()
    t1 = _log_probs([[2.0, 1.0, 0.0, 0.0]])
    t2 = {top: _log_probs([[1.0, 0.0]]) for top in TAGS}
    alts = decoder.top_k_pairs(t1, t2, k=3)[0]
    assert len(alts) == 3
    assert [a["score"] for a in alts] == sorted((a["score"] for a in alts), reverse=True)
    assert alts[0]["industry"] == "healthcare"


def test_tune_lambda_selects_the_value_that_maximises_end_to_end():
    decoder = _decoder()
    # Tier 1 is wrong about row 0 (gold finance) but tier 2 is confident, so a
    # positive lambda should score better than zero.
    t1 = _log_probs([[1.0, 0.9, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
    t2 = {
        "healthcare": _log_probs([[0.0, 0.0], [0.0, 0.0]]),
        "finance": _log_probs([[9.0, 0.0], [9.0, 0.0]]),
        "government": _log_probs([[0.0, 0.0], [0.0, 0.0]]),
    }
    gold_top = np.array(["finance", "finance"], dtype=object)
    gold_tag = np.array(["bank statement", "bank statement"], dtype=object)
    lam, history = tune_lambda(decoder, t1, t2, gold_top, gold_tag, [0.0, 1.0])
    assert lam == 1.0
    assert history[1.0]["end_to_end"] > history[0.0]["end_to_end"]
    assert decoder.lam == 1.0


def test_tune_lambda_counts_other_rows_as_end_to_end_correct():
    # An `other` document has no sub-tag, so getting tier 1 right is the whole
    # task for that row and it must not be penalised for a missing tag.
    decoder = _decoder()
    t1 = _log_probs([[0.0, 0.0, 0.0, 5.0]])
    t2 = {top: _log_probs([[1.0, 0.0]]) for top in TAGS}
    gold_top = np.array([OTHER], dtype=object)
    gold_tag = np.array([None], dtype=object)
    _, history = tune_lambda(decoder, t1, t2, gold_top, gold_tag, [0.5])
    assert history[0.5]["end_to_end"] == 1.0
