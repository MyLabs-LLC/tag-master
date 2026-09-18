"""Training orchestration for both tiers and the joint decoder.

The split discipline here is what keeps the reported numbers honest: `train`
fits every model, `dev` selects all hyperparameters (regularisation strength,
ensemble weights, calibration, and the joint lambda), and `test` is touched only
by `evaluate`.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np

from .bundle import ModelBundle
from .config import Config
from .data import Frame, Splits, get_splits
from .embed import build_embedder
from .features import EmbeddingStore, FeatureProfile, FeatureSpace
from .headings import HeadingMatcher
from .hierarchy import JointDecoder, tune_lambda
from .models import PrototypeScorer, Tier1Head, Tier2Head
from .prepare import build_vocabulary
from .taxonomy import OTHER, TagVocabulary, Taxonomy, build_taxonomy

TIER2_CHANNELS = ("linear", "rocchio", "prototype", "heading", "verbatim")


def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def tier1_sample_weights(frame: Frame, taxonomy: Taxonomy, hard_negative_weight: float) -> np.ndarray:
    """Balanced class weights, with adjacent `other` domains up-weighted.

    Plain balanced weighting treats a Chemicals safety sheet and a Legal
    confidentiality agreement as equally informative negatives, but only the
    second one sits on a decision boundary. The measured failure was government
    precision of 0.623, driven almost entirely by these adjacent domains, so
    they get extra weight.
    """
    labels = frame.top.astype(str)
    weights = np.ones(len(labels), dtype=np.float64)
    n = len(labels)
    k = len(taxonomy.top_labels)
    for top in taxonomy.top_labels:
        mask = labels == top
        count = int(mask.sum())
        if count:
            weights[mask] = n / (k * count)
    hard = np.array([taxonomy.is_hard_negative(d) for d in frame.domain.tolist()])
    weights[hard & (labels == OTHER)] *= hard_negative_weight
    return weights


def _encode_top(frame: Frame, taxonomy: Taxonomy) -> np.ndarray:
    index = {t: i for i, t in enumerate(taxonomy.top_labels)}
    return np.array([index[t] for t in frame.top.astype(str)], dtype=np.int32)


def _encode_tags(frame: Frame, tags: list[str]) -> np.ndarray:
    index = {t: i for i, t in enumerate(tags)}
    return np.array([index.get(t, -1) for t in frame.tag.tolist()], dtype=np.int32)


def train_tier1(
    cfg: Config,
    taxonomy: Taxonomy,
    splits: Splits,
    store: EmbeddingStore,
) -> tuple[Tier1Head, FeatureSpace, dict]:
    profile = FeatureProfile.from_config(cfg, "tier1")
    _log(f"fitting tier-1 feature space ({profile.word_max_features} max word features)")
    t0 = time.time()
    space = FeatureSpace.fit(profile, splits.train.text, embedder=store.embedder)
    _log(f"  {space.describe()} in {time.time() - t0:.0f}s")

    Xtr = space.combine(
        space.transform_sparse(splits.train.text), store.get("train", splits.train.text)
    )
    Xdv = space.combine(
        space.transform_sparse(splits.dev.text), store.get("dev", splits.dev.text)
    )
    ytr = _encode_top(splits.train, taxonomy)
    ydv = _encode_top(splits.dev, taxonomy)
    weights = tier1_sample_weights(
        splits.train, taxonomy, float(cfg["tier1"]["hard_negative_weight"])
    )

    best: tuple[float, Tier1Head] | None = None
    history: dict[str, float] = {}
    for C in cfg["tier1"]["C_grid"]:
        t0 = time.time()
        head = Tier1Head(labels=list(taxonomy.top_labels)).fit(Xtr, ytr, weights, C=float(C))
        acc = float((head.margins(Xdv).argmax(axis=1) == ydv).mean())
        history[str(C)] = round(acc, 5)
        _log(f"  C={C}: dev accuracy {acc:.4f} ({time.time() - t0:.0f}s)")
        if best is None or acc > best[0]:
            best = (acc, head)

    head = best[1]
    head.calibrate(Xdv, ydv)
    _log(f"  selected C={head.C}, dev accuracy {best[0]:.4f}")
    return head, space, {"C_grid": history, "selected_C": head.C, "dev_accuracy": best[0]}


def descriptions_by_tag(frame: Frame) -> dict[str, list[str]]:
    """Group the dataset's own document descriptions by normalised tag.

    Training-time supervision only: `document_description` is dataset metadata,
    not part of the document, and never reaches inference.
    """
    out: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for tag, desc in zip(frame.tag.tolist(), frame.description.tolist()):
        key = (tag, desc[:80])
        if key in seen:
            continue
        seen.add(key)
        out[tag].append(desc)
    return out


def _coordinate_ascent(
    channels: dict[str, np.ndarray],
    gold: np.ndarray,
    head: Tier2Head,
    grid: list[float],
    passes: int = 5,
) -> tuple[dict[str, float], float]:
    """Tune ensemble weights by multi-start coordinate ascent on dev accuracy.

    A full grid over five channels would be 16,807 points; coordinate ascent
    needs about a hundred evaluations per start, and each evaluation is only a
    weighted sum of pre-computed matrices. Several starting points are tried
    because plain single-start ascent gets trapped: starting from
    `linear` alone it would leave the Rocchio centroid at zero weight even
    though it scores 0.80-0.90 on its own, simply because no single step from
    that corner improves things.
    """
    names = list(channels)

    def accuracy(w: dict[str, float]) -> float:
        return float((head.combine(channels, w).argmax(axis=1) == gold).mean())

    starts: list[dict[str, float]] = [
        {n: (1.0 if n == "linear" else 0.0) for n in names},
        {n: (1.0 if n == "linear" else 0.5) for n in names},
        {n: (1.0 if n == "linear" else 0.25) for n in names},
        {n: 1.0 for n in names},
    ]

    best_weights: dict[str, float] = starts[0]
    best_score = -1.0

    for start in starts:
        weights = dict(start)
        score = accuracy(weights)
        for _ in range(passes):
            improved = False
            for name in names:
                current = weights[name]
                for value in grid:
                    if value == current:
                        continue
                    trial = dict(weights)
                    trial[name] = value
                    trial_score = accuracy(trial)
                    if trial_score > score + 1e-6:
                        score, weights, current, improved = trial_score, trial, value, True
            if not improved:
                break
        if score > best_score:
            best_score, best_weights = score, weights

    if all(v == 0.0 for v in best_weights.values()):
        best_weights["linear"] = 1.0
        best_score = accuracy(best_weights)
    return best_weights, best_score


def train_tier2(
    cfg: Config,
    taxonomy: Taxonomy,
    splits: Splits,
    vocab: TagVocabulary,
    store: EmbeddingStore,
    matcher: HeadingMatcher,
) -> tuple[dict[str, Tier2Head], FeatureSpace, dict]:
    profile = FeatureProfile.from_config(cfg, "tier2")
    industry_mask = splits.train.top.astype(str) != OTHER
    industry_train = splits.train.select(industry_mask)

    _log(f"fitting tier-2 feature space on {len(industry_train)} industry rows")
    t0 = time.time()
    space = FeatureSpace.fit(profile, industry_train.text, embedder=store.embedder)
    _log(f"  {space.describe()} in {time.time() - t0:.0f}s")

    train_emb = store.get("train", splits.train.text)
    dev_emb = store.get("dev", splits.dev.text)

    heads: dict[str, Tier2Head] = {}
    report: dict[str, dict] = {}
    grid = [float(x) for x in cfg["tier2"]["ensemble_weight_grid"]]

    for top in taxonomy.industries:
        tags = vocab.tags(top)
        tr_mask = splits.train.top.astype(str) == top
        dv_mask = splits.dev.top.astype(str) == top
        tr = splits.train.select(tr_mask)
        dv = splits.dev.select(dv_mask)
        _log(f"{top}: {len(tags)} tags, {len(tr)} train rows, {len(dv)} dev rows")

        Xtr = space.combine(
            space.transform_sparse(tr.text),
            train_emb[tr_mask] if train_emb is not None else None,
        )
        Xdv = space.combine(
            space.transform_sparse(dv.text),
            dev_emb[dv_mask] if dev_emb is not None else None,
        )
        ytr = _encode_tags(tr, tags)
        ydv = _encode_tags(dv, tags)
        keep = ytr >= 0
        if not keep.all():
            Xtr, ytr = Xtr[keep], ytr[keep]

        head = Tier2Head(top=top, tags=tags)
        head.prototypes = PrototypeScorer.build(
            tags, descriptions_by_tag(tr), store.embedder
        )

        best: tuple[float, Tier2Head] | None = None
        c_history: dict[str, float] = {}
        for C in cfg["tier2"]["C_grid"]:
            candidate = Tier2Head(top=top, tags=tags, prototypes=head.prototypes)
            candidate.fit(Xtr, ytr, C=float(C))
            acc = float(
                (candidate.channel_scores(Xdv, None, None)["linear"].argmax(axis=1) == ydv).mean()
            )
            c_history[str(C)] = round(acc, 5)
            if best is None or acc > best[0]:
                best = (acc, candidate)
        head = best[1]
        _log(f"  linear head alone: dev {best[0]:.4f} (C={head.C})")

        lexical = matcher.score_matrix(dv.text, top).channels()
        channels = head.channel_scores(
            Xdv, dev_emb[dv_mask] if dev_emb is not None else None, lexical
        )
        single = {
            name: float((mat.argmax(axis=1) == ydv).mean()) for name, mat in channels.items()
        }
        weights, ensemble_acc = _coordinate_ascent(channels, ydv, head, grid)
        head.weights = weights
        head.calibrate(head.combine(channels, weights), np.maximum(ydv, 0))
        _log(f"  ensemble: dev {ensemble_acc:.4f}  weights={weights}")

        heads[top] = head
        report[top] = {
            "tags": len(tags),
            "train_rows": int(keep.sum()),
            "dev_rows": len(dv),
            "C_grid": c_history,
            "selected_C": head.C,
            "channel_only_dev_accuracy": {k: round(v, 5) for k, v in single.items()},
            "ensemble_weights": weights,
            "ensemble_dev_accuracy": round(ensemble_acc, 5),
            "temperature": round(head.calibrator.temperature, 4),
        }

    return heads, space, report


def tier2_log_probabilities(
    bundle_tier2: dict[str, Tier2Head],
    space: FeatureSpace,
    matcher: HeadingMatcher,
    frame: Frame,
    embeddings: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Score every industry's tier-2 head on every row.

    Deliberately not restricted to the industry tier 1 preferred: the joint
    decoder needs all of them to be able to override tier 1.
    """
    sparse = space.transform_sparse(frame.text)
    X = space.combine(sparse, embeddings)
    out: dict[str, np.ndarray] = {}
    for top, head in bundle_tier2.items():
        lexical = matcher.score_matrix(frame.text, top).channels()
        channels = head.channel_scores(X, embeddings, lexical)
        out[top] = head.log_proba(head.combine(channels, head.weights))
    return out


def run_train(cfg: Config, use_embeddings: bool = True) -> dict:
    cfg.paths.ensure()
    taxonomy = build_taxonomy(cfg)
    splits = get_splits(cfg, taxonomy)
    vocab = build_vocabulary(splits, taxonomy)

    embedder = build_embedder(cfg) if use_embeddings else None
    store = EmbeddingStore(embedder)
    if embedder is not None:
        _log("loading document embeddings (cached after the first run)")
        store.get("train", splits.train.text)
        store.get("dev", splits.dev.text)

    tier1, tier1_space, tier1_report = train_tier1(cfg, taxonomy, splits, store)

    matcher = HeadingMatcher.fit(
        splits.train.text, vocab, heading_window=int(cfg["features"]["heading_chars"])
    )
    tier2, tier2_space, tier2_report = train_tier2(
        cfg, taxonomy, splits, vocab, store, matcher
    )

    _log("tuning the joint decoder on dev")
    decoder = JointDecoder(
        top_labels=list(taxonomy.top_labels),
        industries=list(taxonomy.industries),
        tags_by_top=vocab.tags_by_top,
    )
    Xdv = tier1_space.combine(
        tier1_space.transform_sparse(splits.dev.text), store.get("dev", splits.dev.text)
    )
    t1_lp = tier1.log_proba(Xdv)
    t2_lp = tier2_log_probabilities(
        tier2, tier2_space, matcher, splits.dev, store.get("dev", splits.dev.text)
    )
    gold_top = splits.dev.top.astype(str)
    gold_tag = np.array(
        [t if s != OTHER else None for s, t in zip(gold_top, splits.dev.tag.tolist())],
        dtype=object,
    )
    lam, lam_history = tune_lambda(
        decoder, t1_lp, t2_lp, gold_top, gold_tag, [float(x) for x in cfg["hierarchy"]["lambda_grid"]]
    )
    _log(f"  lambda={lam}  dev {lam_history[lam]}")

    summary = {
        "tier1": tier1_report,
        "tier2": tier2_report,
        "hierarchy": {
            "lambda": lam,
            "grid": {str(k): v for k, v in lam_history.items()},
        },
        "feature_spaces": {
            "tier1": tier1_space.describe(),
            "tier2": tier2_space.describe(),
        },
        "label_space": {"total_tags": vocab.total},
    }

    bundle = ModelBundle(
        taxonomy=taxonomy,
        vocab=vocab,
        tier1_space=tier1_space,
        tier2_space=tier2_space,
        tier1=tier1,
        tier2=tier2,
        matcher=matcher,
        decoder=decoder,
        metadata=summary,
    )
    bundle.save(cfg)
    (cfg.paths.reports / "training.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
