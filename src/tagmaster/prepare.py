"""The `prepare` stage: build splits, derive the tier-2 label space, and report
what the data actually contains."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import Config
from .data import Frame, Splits, describe, get_splits
from .headings import HeadingMatcher
from .taxonomy import OTHER, TagVocabulary, Taxonomy, build_taxonomy


def build_vocabulary(splits: Splits, taxonomy: Taxonomy) -> TagVocabulary:
    """Derive the tier-2 label space from the training split only.

    Restricting to train is what makes the label space legitimate: a tag first
    seen at test time is genuinely unpredictable, and counting it in the
    vocabulary would quietly inflate the denominator instead of showing up as an
    error. `unseen_tag_rate` in the report quantifies how often that happens.
    """
    tags_by_top: dict[str, list[str]] = {}
    for top in taxonomy.industries:
        frame = splits.train.where_top(top)
        tags_by_top[top] = sorted({t for t in frame.tag.tolist() if t})
    return TagVocabulary(tags_by_top=tags_by_top, aliases={})


def unseen_tag_rate(frame: Frame, vocab: TagVocabulary, taxonomy: Taxonomy) -> dict:
    """How many evaluation rows carry a tag the training split never showed."""
    out: dict[str, dict] = {}
    for top in taxonomy.industries:
        sub = frame.where_top(top)
        known = set(vocab.tags(top))
        missing = int(sum(1 for t in sub.tag.tolist() if t not in known))
        out[top] = {
            "rows": len(sub),
            "unseen_rows": missing,
            "unseen_rate": round(missing / max(len(sub), 1), 4),
        }
    return out


def tag_report(splits: Splits, taxonomy: Taxonomy, vocab: TagVocabulary) -> dict:
    """Per-industry tier-2 label-space statistics, including support thinness."""
    report: dict = {"per_industry": {}, "total_tags": vocab.total}
    for top in taxonomy.industries:
        frame = splits.train.where_top(top)
        per_tag_uids: dict[str, set[str]] = defaultdict(set)
        for tag, uid in zip(frame.tag.tolist(), frame.uid.tolist()):
            per_tag_uids[tag].add(uid)
        support = sorted(len(v) for v in per_tag_uids.values())
        report["per_industry"][top] = {
            "tags": len(vocab.tags(top)),
            "train_rows": len(frame),
            "train_uids": len(set(frame.uid.tolist())),
            "support_uids": {
                "min": support[0] if support else 0,
                "median": int(np.median(support)) if support else 0,
                "max": support[-1] if support else 0,
                "under_10": int(sum(1 for s in support if s < 10)),
            },
        }
    return report


def domain_report(splits: Splits, taxonomy: Taxonomy) -> dict:
    """Row counts per source domain and how each maps to tier 1."""
    counts: Counter[str] = Counter()
    for _, frame in splits.items():
        counts.update(frame.domain.tolist())
    return {
        dom: {
            "rows": n,
            "top": taxonomy.top_for_domain(dom),
            "hard_negative": taxonomy.is_hard_negative(dom),
        }
        for dom, n in sorted(counts.items(), key=lambda kv: -kv[1])
    }


def signal_report(splits: Splits, taxonomy: Taxonomy, vocab: TagVocabulary, sample: int = 4000) -> dict:
    """Validate the two measurements the architecture is built on.

    `verbatim_rate` should land near 0.74 and `heading_hit_rate` shows how much
    of that the extractor actually recovers.
    """
    rng = np.random.default_rng(0)
    frame = splits.dev
    industry_mask = np.isin(frame.top.astype(str), list(taxonomy.industries))
    sub = frame.select(industry_mask)
    if len(sub) > sample:
        sub = sub.select(rng.choice(len(sub), size=sample, replace=False))

    matcher = HeadingMatcher.fit(splits.train.text, vocab)
    per_industry = {}
    for top in taxonomy.industries:
        part = sub.where_top(top)
        if len(part):
            per_industry[top] = {
                k: round(v, 4) for k, v in matcher.top1_accuracy(part.text, part.tag, top).items()
            }
    return {
        "sample_rows": len(sub),
        "tag_verbatim_in_text": round(matcher.verbatim_rate(sub.text, sub.tag), 4),
        "tag_recovered_from_heading": round(matcher.heading_hit_rate(sub.text, sub.tag), 4),
        "domain_named_in_description": round(
            float(
                np.mean(
                    [
                        d.lower().find(dom.lower()) >= 0
                        for d, dom in zip(sub.description.tolist(), sub.domain.tolist())
                    ]
                )
            ),
            4,
        ),
        "lexical_only_tier2_top1": per_industry,
    }


def run_prepare(cfg: Config, rebuild: bool = False) -> dict:
    cfg.paths.ensure()
    taxonomy = build_taxonomy(cfg)
    splits = get_splits(cfg, taxonomy, rebuild=rebuild)
    vocab = build_vocabulary(splits, taxonomy)
    vocab.to_json(cfg.paths.data / "vocabulary.json")

    report = {
        "dataset": cfg["dataset"]["repo_id"],
        "mapping_variant": cfg.get("mapping_variant", "core"),
        "top_labels": list(taxonomy.top_labels),
        "counts": describe(splits, taxonomy),
        "tier2_label_space": tag_report(splits, taxonomy, vocab),
        "unseen_tags": {
            "dev": unseen_tag_rate(splits.dev, vocab, taxonomy),
            "test": unseen_tag_rate(splits.test, vocab, taxonomy),
        },
        "signals": signal_report(splits, taxonomy, vocab),
        "domains": domain_report(splits, taxonomy),
    }
    dest: Path = cfg.paths.reports / "prepare.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
