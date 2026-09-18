"""Feature spaces for the two tiers.

Tier 1 and tier 2 solve different problems and get different feature profiles.
Tier 1 is a topical four-class decision over all 140k training rows, where word
n-grams suffice and character n-grams would cost roughly 700M nonzeros. Tier 2
discriminates between near-duplicate form names inside a single industry (at
most 15k rows), where sub-word shape carries real signal -- the corpus is full of
form-field labels and identifiers like `Medical Record Number:` and `IRS 1040`
-- and the matrices stay small enough to afford it.

CPU sentence embeddings are shared between the profiles through one on-disk
cache, so they are computed at most once per split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Config
from .embed import OnnxEmbedder


@dataclass(frozen=True)
class FeatureProfile:
    name: str
    word_chars: int = 2500
    word_ngram_max: int = 2
    word_min_df: int = 3
    word_max_features: int = 300000
    char_ngram: tuple[int, int] | None = None
    char_chars: int = 1200
    char_min_df: int = 3
    char_max_features: int = 300000
    use_embeddings: bool = True
    embedding_scale: float = 0.6

    @classmethod
    def from_config(cls, cfg: Config, name: str) -> FeatureProfile:
        raw = cfg["features"][name]
        char = raw.get("char_ngram")
        emb = cfg["features"].get("embedding", {}) or {}
        return cls(
            name=name,
            word_chars=int(raw.get("word_chars", 2500)),
            word_ngram_max=int(raw.get("word_ngram_max", 2)),
            word_min_df=int(raw.get("word_min_df", 3)),
            word_max_features=int(raw.get("word_max_features", 300000)),
            char_ngram=tuple(char) if char else None,
            char_chars=int(raw.get("char_chars", 1200)),
            char_min_df=int(raw.get("char_min_df", 3)),
            char_max_features=int(raw.get("char_max_features", 300000)),
            use_embeddings=bool(raw.get("use_embeddings", True))
            and bool(emb.get("enabled", True)),
            embedding_scale=float(emb.get("scale", 0.6)),
        )


def _truncate(texts, limit: int) -> list[str]:
    return [(t or "")[:limit] for t in texts]


class FeatureSpace:
    """Fitted vectorizers for one profile, plus the transform they imply."""

    def __init__(
        self,
        profile: FeatureProfile,
        word: TfidfVectorizer,
        char: TfidfVectorizer | None,
        embedder: OnnxEmbedder | None,
    ):
        self.profile = profile
        self.word = word
        self.char = char
        self.embedder = embedder

    @classmethod
    def fit(
        cls,
        profile: FeatureProfile,
        texts: np.ndarray,
        embedder: OnnxEmbedder | None = None,
    ) -> FeatureSpace:
        word = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, profile.word_ngram_max),
            min_df=profile.word_min_df,
            max_features=profile.word_max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        word.fit(_truncate(texts, profile.word_chars))

        char = None
        if profile.char_ngram is not None:
            char = TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=profile.char_ngram,
                min_df=profile.char_min_df,
                max_features=profile.char_max_features,
                sublinear_tf=True,
                dtype=np.float32,
            )
            char.fit(_truncate(texts, profile.char_chars))

        return cls(
            profile=profile,
            word=word,
            char=char,
            embedder=embedder if profile.use_embeddings else None,
        )

    @property
    def uses_embeddings(self) -> bool:
        return self.embedder is not None

    def transform_sparse(self, texts) -> sp.csr_matrix:
        blocks = [self.word.transform(_truncate(texts, self.profile.word_chars))]
        if self.char is not None:
            blocks.append(self.char.transform(_truncate(texts, self.profile.char_chars)))
        if len(blocks) == 1:
            return blocks[0].tocsr()
        return sp.hstack(blocks, format="csr", dtype=np.float32)

    def embed(self, texts, cache_tag: str | None = None) -> np.ndarray | None:
        if self.embedder is None:
            return None
        docs = _truncate(texts, 2000)
        if cache_tag:
            return self.embedder.encode_cached(docs, cache_tag)
        return self.embedder.encode(docs, progress=len(docs) > 2000)

    def combine(self, sparse: sp.csr_matrix, dense: np.ndarray | None) -> sp.csr_matrix:
        if dense is None:
            return sparse
        scaled = sp.csr_matrix(dense.astype(np.float32) * self.profile.embedding_scale)
        return sp.hstack([sparse, scaled], format="csr", dtype=np.float32)

    def transform(self, texts, cache_tag: str | None = None) -> sp.csr_matrix:
        return self.combine(self.transform_sparse(texts), self.embed(texts, cache_tag))

    @property
    def dimension(self) -> int:
        n = len(self.word.vocabulary_)
        if self.char is not None:
            n += len(self.char.vocabulary_)
        if self.embedder is not None:
            n += self.embedder.dim
        return n

    def describe(self) -> dict:
        return {
            "profile": self.profile.name,
            "word_features": len(self.word.vocabulary_),
            "char_features": len(self.char.vocabulary_) if self.char is not None else 0,
            "embedding_dim": self.embedder.dim if self.embedder is not None else 0,
            "dimension": self.dimension,
        }

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump(
            {"profile": self.profile, "word": self.word, "char": self.char},
            path,
            compress=3,
        )

    @classmethod
    def load(cls, path: Path, embedder: OnnxEmbedder | None) -> FeatureSpace:
        import joblib

        blob = joblib.load(path)
        profile: FeatureProfile = blob["profile"]
        return cls(
            profile=profile,
            word=blob["word"],
            char=blob["char"],
            embedder=embedder if profile.use_embeddings else None,
        )


class EmbeddingStore:
    """Per-split embedding lookup, so each split is embedded at most once."""

    def __init__(self, embedder: OnnxEmbedder | None):
        self.embedder = embedder
        self._cache: dict[str, np.ndarray | None] = {}

    def get(self, split: str, texts) -> np.ndarray | None:
        if self.embedder is None:
            return None
        if split not in self._cache:
            self._cache[split] = self.embedder.encode_cached(
                _truncate(texts, 2000), split, progress=True
            )
        return self._cache[split]
