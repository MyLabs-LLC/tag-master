"""Serialising a trained model: the artifacts that `evaluate` and `predict` need."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from .config import Config
from .embed import build_embedder
from .features import FeatureSpace
from .headings import HeadingMatcher
from .hierarchy import JointDecoder
from .models import Tier1Head, Tier2Head
from .taxonomy import TagVocabulary, Taxonomy, build_taxonomy


@dataclass
class ModelBundle:
    taxonomy: Taxonomy
    vocab: TagVocabulary
    tier1_space: FeatureSpace
    tier2_space: FeatureSpace
    tier1: Tier1Head
    tier2: dict[str, Tier2Head]
    matcher: HeadingMatcher
    decoder: JointDecoder
    metadata: dict

    def save(self, cfg: Config) -> Path:
        out = cfg.paths.models
        out.mkdir(parents=True, exist_ok=True)
        self.tier1_space.save(out / "tier1_features.joblib")
        self.tier2_space.save(out / "tier2_features.joblib")
        joblib.dump(self.tier1, out / "tier1_head.joblib", compress=3)
        joblib.dump(self.tier2, out / "tier2_heads.joblib", compress=3)
        joblib.dump(self.matcher, out / "heading_matcher.joblib", compress=3)
        self.vocab.to_json(out / "vocabulary.json")
        (out / "decoder.json").write_text(
            json.dumps(
                {
                    "top_labels": list(self.decoder.top_labels),
                    "industries": list(self.decoder.industries),
                    "lam": self.decoder.lam,
                    "other_bias": self.decoder.other_bias,
                    "tier2_weights": {k: v.weights for k, v in self.tier2.items()},
                    "metadata": self.metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, cfg: Config) -> ModelBundle:
        out = cfg.paths.models
        if not (out / "decoder.json").exists():
            raise FileNotFoundError(
                f"no trained model in {out}; run `tagmaster train` first"
            )
        embedder = build_embedder(cfg)
        taxonomy = build_taxonomy(cfg)
        vocab = TagVocabulary.from_json(out / "vocabulary.json")
        blob = json.loads((out / "decoder.json").read_text(encoding="utf-8"))
        decoder = JointDecoder(
            top_labels=blob["top_labels"],
            industries=blob["industries"],
            tags_by_top=vocab.tags_by_top,
            lam=float(blob["lam"]),
            other_bias=float(blob["other_bias"]),
        )
        return cls(
            taxonomy=taxonomy,
            vocab=vocab,
            tier1_space=FeatureSpace.load(out / "tier1_features.joblib", embedder),
            tier2_space=FeatureSpace.load(out / "tier2_features.joblib", embedder),
            tier1=joblib.load(out / "tier1_head.joblib"),
            tier2=joblib.load(out / "tier2_heads.joblib"),
            matcher=joblib.load(out / "heading_matcher.joblib"),
            decoder=decoder,
            metadata=blob.get("metadata", {}),
        )
