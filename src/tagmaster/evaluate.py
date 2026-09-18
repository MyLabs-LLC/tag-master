"""Metrics and reporting.

Three numbers matter and they are reported separately, never conflated:

* tier-1 accuracy and macro-F1 over the four industries;
* tier-2 accuracy *conditional on the industry being right*, which is the
  "given it is healthcare, which form is it" question;
* end-to-end exact match, both tiers right on the same document.

The third is the honest headline. Two independent tiers at 0.90 multiply to
about 0.81 end-to-end, so quoting a per-tier number as if it were end-to-end
would overstate the model by roughly nine points.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .bundle import ModelBundle
from .canonicalize import ConfusionGraph, sweep
from .config import Config
from .data import Frame, get_splits
from .features import EmbeddingStore
from .hierarchy import JointPrediction
from .taxonomy import OTHER


def _confusion(labels: list[str], gold: np.ndarray, pred: np.ndarray) -> dict:
    index = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    matrix = np.zeros((k, k), dtype=int)
    for g, p in zip(gold, pred):
        if g in index and p in index:
            matrix[index[g], index[p]] += 1
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
        "per_class": {
            label: {
                "support": int(matrix[i].sum()),
                "recall": round(float(matrix[i, i] / max(matrix[i].sum(), 1)), 5),
                "precision": round(float(matrix[i, i] / max(matrix[:, i].sum(), 1)), 5),
                "f1": round(
                    float(2 * matrix[i, i] / max(matrix[i].sum() + matrix[:, i].sum(), 1)), 5
                ),
            }
            for i, label in enumerate(labels)
        },
    }


def _macro_f1(confusion: dict) -> float:
    per = confusion["per_class"].values()
    return float(np.mean([c["f1"] for c in per])) if per else 0.0


def coverage_curve(
    correct: np.ndarray, confidence: np.ndarray, points: int = 19
) -> list[dict]:
    """Accuracy as a function of coverage, sorted by confidence.

    If a tier lands below its target outright, this is what turns it into a
    usable guarantee: "0.90 at N% coverage", with the remainder routed to review.
    """
    order = np.argsort(-confidence)
    ordered = correct[order]
    n = len(ordered)
    if n == 0:
        return []
    out = []
    for frac in np.linspace(1.0 / points, 1.0, points):
        take = max(1, int(round(frac * n)))
        out.append(
            {
                "coverage": round(take / n, 4),
                "accuracy": round(float(ordered[:take].mean()), 5),
                "threshold": round(float(confidence[order][take - 1]), 6),
            }
        )
    return out


def accuracy_at_target(curve: list[dict], target: float) -> dict | None:
    """The widest coverage whose accuracy still clears `target`."""
    best = None
    for point in curve:
        if point["accuracy"] >= target:
            if best is None or point["coverage"] > best["coverage"]:
                best = point
    return best


def _slice_metrics(
    frame: Frame, pred: JointPrediction, gold_tag: np.ndarray, attribute: str
) -> dict:
    """Break the headline numbers down by locale or document format.

    The locale slice is a genuine robustness check that the shipped split cannot
    provide: every record exists in a `us` and an `intl` rendering, so a large
    gap between them would mean the model latched onto surface locale cues.
    """
    values = getattr(frame, attribute).astype(str)
    out: dict[str, dict] = {}
    for value in sorted(set(values.tolist())):
        mask = values == value
        top_ok = pred.top[mask] == frame.top.astype(str)[mask]
        industry = frame.top.astype(str)[mask] != OTHER
        tag_ok = pred.tag[mask] == gold_tag[mask]
        out[value] = {
            "rows": int(mask.sum()),
            "tier1_accuracy": round(float(top_ok.mean()), 5),
            "end_to_end": round(float((top_ok & (tag_ok | ~industry)).mean()), 5),
        }
    return out


def evaluate_predictions(
    frame: Frame, pred: JointPrediction, bundle: ModelBundle
) -> dict:
    taxonomy = bundle.taxonomy
    gold_top = frame.top.astype(str)
    gold_tag = np.array(
        [t if s != OTHER else None for s, t in zip(gold_top, frame.tag.tolist())],
        dtype=object,
    )

    top_ok = pred.top == gold_top
    industry = gold_top != OTHER
    tag_ok = pred.tag == gold_tag
    both = top_ok & (tag_ok | ~industry)

    tier1_conf = _confusion(list(taxonomy.top_labels), gold_top, pred.top)

    # Tier 2 conditional on tier 1 being right: the question "given the industry
    # is correct, is the sub-tag correct".
    conditional = industry & top_ok
    tier2_conditional = float(tag_ok[conditional].mean()) if conditional.any() else 0.0

    per_industry: dict[str, dict] = {}
    for top in taxonomy.industries:
        mask = gold_top == top
        cond = mask & top_ok
        per_industry[top] = {
            "rows": int(mask.sum()),
            "tier1_recall": round(float(top_ok[mask].mean()), 5) if mask.any() else 0.0,
            "tier2_conditional_accuracy": round(float(tag_ok[cond].mean()), 5)
            if cond.any()
            else 0.0,
            "end_to_end": round(float(both[mask].mean()), 5) if mask.any() else 0.0,
            "sub_tags": len(bundle.vocab.tags(top)),
        }

    tier1_curve = coverage_curve(top_ok.astype(float), pred.top_confidence)
    e2e_curve = coverage_curve(
        both.astype(float), pred.top_confidence * np.clip(pred.tag_confidence, 1e-9, None)
    )

    return {
        "headline": {
            "rows": len(frame),
            "tier1_accuracy": round(float(top_ok.mean()), 5),
            "tier1_macro_f1": round(_macro_f1(tier1_conf), 5),
            "tier2_conditional_accuracy": round(tier2_conditional, 5),
            "end_to_end_exact_match": round(float(both.mean()), 5),
            "end_to_end_industry_rows_only": round(float(both[industry].mean()), 5)
            if industry.any()
            else 0.0,
        },
        "tier1_confusion": tier1_conf,
        "per_industry": per_industry,
        "by_locale": _slice_metrics(frame, pred, gold_tag, "locale"),
        "by_format": _slice_metrics(frame, pred, gold_tag, "document_format"),
        "coverage": {
            "tier1": tier1_curve,
            "end_to_end": e2e_curve,
            "tier1_at_0.95": accuracy_at_target(tier1_curve, 0.95),
            "end_to_end_at_0.90": accuracy_at_target(e2e_curve, 0.90),
        },
    }


def predict_frame(
    bundle: ModelBundle, frame: Frame, store: EmbeddingStore, split_name: str
) -> tuple[JointPrediction, np.ndarray, dict[str, np.ndarray]]:
    from .train import tier2_log_probabilities

    embeddings = store.get(split_name, frame.text)
    X1 = bundle.tier1_space.combine(
        bundle.tier1_space.transform_sparse(frame.text), embeddings
    )
    t1_lp = bundle.tier1.log_proba(X1)
    t2_lp = tier2_log_probabilities(
        bundle.tier2, bundle.tier2_space, bundle.matcher, frame, embeddings
    )
    return bundle.decoder.decode(t1_lp, t2_lp), t1_lp, t2_lp


def canonicalization_report(
    cfg: Config, bundle: ModelBundle, frame: Frame, t2_lp: dict[str, np.ndarray]
) -> dict:
    """Learn merges from dev confusion, then measure them on the eval split.

    Clusters come from the *dev* split so the reported coarse accuracy is not
    fitted to the split it is scored on.
    """
    taxonomy = bundle.taxonomy
    splits = get_splits(cfg, taxonomy)
    store = EmbeddingStore(bundle.tier1_space.embedder or bundle.tier2_space.embedder)
    dev_pred, _, dev_t2 = predict_frame(bundle, splits.dev, store, "dev")

    graphs: dict[str, ConfusionGraph] = {}
    for top in taxonomy.industries:
        tags = bundle.vocab.tags(top)
        mask = splits.dev.top.astype(str) == top
        if not mask.any():
            continue
        index = {t: i for i, t in enumerate(tags)}
        gold = np.array([index.get(t, -1) for t in splits.dev.tag[mask].tolist()])
        predicted = dev_t2[top][mask].argmax(axis=1)
        graphs[top] = ConfusionGraph.from_predictions(tags, gold, predicted)

    gold_top = frame.top.astype(str)
    rows: list[tuple[str, str, str]] = []
    for top in taxonomy.industries:
        tags = bundle.vocab.tags(top)
        mask = gold_top == top
        if not mask.any():
            continue
        predicted = t2_lp[top][mask].argmax(axis=1)
        for gold_tag, p in zip(frame.tag[mask].tolist(), predicted):
            rows.append((top, gold_tag, tags[p]))

    conf = cfg["canonicalize"]
    report, by_threshold = sweep(
        graphs,
        rows,
        [float(t) for t in conf["thresholds"]],
        float(conf["linkage_ratio"]),
        int(conf["max_cluster_size"]),
    )

    raw_accuracy = sum(1 for top, g, p in rows if g == p) / max(len(rows), 1)
    blob = report.to_dict()
    blob["raw_granularity"] = {
        "tags": bundle.vocab.total,
        "tier2_accuracy": round(raw_accuracy, 5),
    }

    # Surface the coarsest option that buys at least a point of accuracy, so the
    # trade is a choice rather than a default.
    worthwhile = [
        r for r in report.results if r.accuracy and r.accuracy >= raw_accuracy + 0.01
    ]
    chosen = max(worthwhile, key=lambda r: r.accuracy) if worthwhile else None
    if chosen is not None:
        blob["recommended"] = {
            "threshold": chosen.threshold,
            "tags_after": chosen.n_tags_after,
            "tier2_accuracy": round(chosen.accuracy, 5),
            "merged_clusters": chosen.merged_only(),
        }
        (cfg.paths.models / "tag_aliases.json").write_text(
            json.dumps(
                {"threshold": chosen.threshold, "aliases": chosen.aliases},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return blob


def run_evaluate(cfg: Config, split: str = "test", with_canonicalization: bool = True) -> dict:
    cfg.paths.ensure()
    bundle = ModelBundle.load(cfg)
    splits = get_splits(cfg, bundle.taxonomy)
    frame = {"train": splits.train, "dev": splits.dev, "test": splits.test}[split]

    embedder = bundle.tier1_space.embedder or bundle.tier2_space.embedder
    store = EmbeddingStore(embedder)
    pred, _, t2_lp = predict_frame(bundle, frame, store, split)

    report = evaluate_predictions(frame, pred, bundle)
    report["split"] = split
    report["model"] = {
        "lambda": bundle.decoder.lam,
        "tier2_weights": {k: v.weights for k, v in bundle.tier2.items()},
        "total_sub_tags": bundle.vocab.total,
    }
    if with_canonicalization:
        report["canonicalization"] = canonicalization_report(cfg, bundle, frame, t2_lp)

    dest = cfg.paths.reports / f"evaluation_{split}.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def error_samples(
    bundle: ModelBundle, frame: Frame, pred: JointPrediction, limit: int = 25
) -> list[dict]:
    """A sample of misclassifications, for inspecting failure modes by hand."""
    gold_top = frame.top.astype(str)
    wrong = np.where((pred.top != gold_top))[0][:limit]
    out = []
    for i in wrong:
        out.append(
            {
                "gold_industry": gold_top[i],
                "gold_domain": frame.domain[i],
                "gold_sub_tag": frame.tag[i],
                "predicted_industry": pred.top[i],
                "predicted_sub_tag": pred.tag[i],
                "confidence": round(float(pred.top_confidence[i]), 4),
                "text": frame.text[i][:240],
            }
        )
    return out


def group_errors_by_domain(frame: Frame, pred: JointPrediction) -> dict:
    """Which source domains drive tier-1 errors.

    The expectation from the config's `hard_negative_domains` is that adjacent
    `other` domains dominate; this checks whether that held.
    """
    gold_top = frame.top.astype(str)
    wrong = pred.top != gold_top
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for domain, predicted in zip(frame.domain[wrong].tolist(), pred.top[wrong].tolist()):
        counts[domain][predicted] += 1
    totals = {d: sum(v.values()) for d, v in counts.items()}
    return {
        d: {"errors": totals[d], "predicted_as": dict(counts[d])}
        for d in sorted(totals, key=lambda x: -totals[x])[:20]
    }
