"""The tier-1 and tier-2 heads, their calibration, and the tier-2 ensemble members.

Tier 1 is a single calibrated four-class linear model. Tier 2 is an ensemble,
because no single scorer covers the whole label space: a discriminative head is
strongest on well-supported tags, a Rocchio centroid degrades far more gracefully
as support thins, the lexical channels handle tags whose name is simply printed
in the document, and the prototype scorer gives even a two-example tag a usable
representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


def log_softmax(scores: np.ndarray) -> np.ndarray:
    m = scores.max(axis=1, keepdims=True)
    shifted = scores - m
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax(scores: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(scores))


def _margins(model, X) -> np.ndarray:
    """Decision values as a (n, n_classes) matrix, including the binary case."""
    d = model.decision_function(X)
    if d.ndim == 1:
        d = np.column_stack([-d, d])
    return d.astype(np.float64)


@dataclass
class MarginCalibrator:
    """Multinomial Platt scaling on top of a linear model's decision values.

    A LinearSVC gives the best accuracy per unit of CPU on TF-IDF of this shape,
    but its margins are not probabilities, and the joint decoder needs
    comparable log-probabilities from both tiers. Fitting a small multinomial
    logistic regression on held-out margins turns them into calibrated
    probabilities without retraining the expensive model.
    """

    model: LogisticRegression | None = None
    n_classes: int = 0

    def fit(self, margins: np.ndarray, y: np.ndarray, n_classes: int) -> MarginCalibrator:
        self.n_classes = n_classes
        if len(np.unique(y)) < 2:
            self.model = None
            return self
        self.model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        self.model.fit(margins, y)
        return self

    def log_proba(self, margins: np.ndarray) -> np.ndarray:
        if self.model is None:
            return log_softmax(margins)
        lp = self.model.predict_log_proba(margins)
        if lp.shape[1] == self.n_classes:
            return lp
        # Classes absent from the calibration sample get floored rather than dropped.
        full = np.full((lp.shape[0], self.n_classes), -1e4)
        full[:, self.model.classes_] = lp
        return full


@dataclass
class TemperatureCalibrator:
    """Single-parameter (temperature) calibration for wide label spaces.

    Tier 2 has up to 315 classes, so the multinomial Platt scaling used for
    tier 1 would fit ~99k parameters against ~3.3k dev rows and overfit
    immediately. One scalar temperature preserves the ranking, costs nothing,
    and is all the joint decoder needs to make tier-2 log-probabilities
    comparable to tier-1's.
    """

    temperature: float = 1.0

    def fit(self, scores: np.ndarray, y: np.ndarray) -> TemperatureCalibrator:
        if len(scores) == 0:
            return self
        rows = np.arange(len(y))
        best_t, best_nll = 1.0, float("inf")
        for t in np.geomspace(0.05, 20.0, 60):
            lp = log_softmax(scores / t)
            nll = -float(lp[rows, y].mean())
            if nll < best_nll:
                best_nll, best_t = nll, float(t)
        self.temperature = best_t
        return self

    def log_proba(self, scores: np.ndarray) -> np.ndarray:
        return log_softmax(scores / self.temperature)


@dataclass
class Tier1Head:
    """Four-class industry classifier with calibrated probabilities."""

    labels: list[str]
    model: LinearSVC | None = None
    calibrator: MarginCalibrator = field(default_factory=MarginCalibrator)
    C: float = 1.0

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    def fit(self, X, y: np.ndarray, sample_weight: np.ndarray | None = None, C: float = 1.0):
        self.C = C
        self.model = LinearSVC(C=C, loss="squared_hinge", dual=True, max_iter=3000, tol=1e-4)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def margins(self, X) -> np.ndarray:
        return _margins(self.model, X)

    def calibrate(self, X, y: np.ndarray) -> None:
        self.calibrator.fit(self.margins(X), y, self.n_classes)

    def log_proba(self, X) -> np.ndarray:
        return self.calibrator.log_proba(self.margins(X))

    def predict(self, X) -> np.ndarray:
        return self.log_proba(X).argmax(axis=1)


@dataclass
class RocchioCentroid:
    """Nearest-centroid scorer over L2-normalised features.

    Included because it was the strongest single baseline measured on this
    dataset (0.853 at full 635-way granularity, against 0.50 for Naive Bayes).
    Averaging within a class rather than fitting a boundary makes it far more
    robust than a discriminative head when a tag has only a handful of examples.
    """

    centroids: np.ndarray | None = None

    def fit(self, X, y: np.ndarray, n_classes: int) -> RocchioCentroid:
        Xn = normalize(X)
        dim = Xn.shape[1]
        cent = np.zeros((n_classes, dim), dtype=np.float32)
        for c in range(n_classes):
            mask = y == c
            if not mask.any():
                continue
            rows = Xn[mask]
            mean = np.asarray(rows.mean(axis=0)).ravel() if sp.issparse(rows) else rows.mean(axis=0)
            cent[c] = mean
        norms = np.linalg.norm(cent, axis=1, keepdims=True)
        self.centroids = cent / np.clip(norms, 1e-9, None)
        return self

    def scores(self, X) -> np.ndarray:
        return np.asarray(normalize(X) @ self.centroids.T, dtype=np.float32)


@dataclass
class PrototypeScorer:
    """Label-as-text scorer built from the dataset's own tag descriptions.

    `document_description` names the source domain in 93% of rows and describes
    the document kind in prose. It is metadata, not part of the document, so it
    is never an inference input -- but at training time it lets every tag get a
    dense prototype from its *name and description* rather than only from its
    examples. That is what makes a two-example tag representable at all.
    """

    prototypes: np.ndarray | None = None
    sources: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        tags: list[str],
        descriptions_by_tag: dict[str, list[str]],
        embedder,
        max_descriptions: int = 6,
    ) -> PrototypeScorer:
        if embedder is None or not tags:
            return cls(prototypes=None, sources=[])
        texts: list[str] = []
        for tag in tags:
            parts = [tag]
            for d in descriptions_by_tag.get(tag, [])[:max_descriptions]:
                parts.append(d)
            texts.append(" ".join(parts)[:2000])
        vecs = embedder.encode(texts, progress=False)
        return cls(prototypes=vecs, sources=texts)

    def scores(self, doc_embeddings: np.ndarray | None, n_rows: int, n_tags: int) -> np.ndarray:
        if self.prototypes is None or doc_embeddings is None:
            return np.zeros((n_rows, n_tags), dtype=np.float32)
        return np.asarray(doc_embeddings @ self.prototypes.T, dtype=np.float32)


@dataclass
class Tier2Head:
    """Per-industry sub-tag classifier: a linear head plus ensemble members.

    Channel scores are z-normalised per document before mixing, because the
    members are on incompatible scales (SVM margins, cosine similarities, and
    bounded lexical scores). Weights are tuned on dev by `train.py`.
    """

    top: str
    tags: list[str]
    model: LinearSVC | None = None
    calibrator: TemperatureCalibrator = field(default_factory=TemperatureCalibrator)
    rocchio: RocchioCentroid = field(default_factory=RocchioCentroid)
    prototypes: PrototypeScorer = field(default_factory=PrototypeScorer)
    weights: dict[str, float] = field(default_factory=dict)
    C: float = 1.0

    @property
    def n_classes(self) -> int:
        return len(self.tags)

    def fit(self, X, y: np.ndarray, C: float = 1.0) -> Tier2Head:
        self.C = C
        self.model = LinearSVC(C=C, loss="squared_hinge", dual=True, max_iter=3000, tol=1e-4)
        self.model.fit(X, y)
        self.rocchio.fit(X, y, self.n_classes)
        return self

    def channel_scores(
        self,
        X,
        doc_embeddings: np.ndarray | None,
        lexical: dict[str, np.ndarray] | None,
    ) -> dict[str, np.ndarray]:
        """Raw, uncombined score matrices for every ensemble member."""
        n = X.shape[0]
        out: dict[str, np.ndarray] = {}

        margins = _margins(self.model, X)
        if margins.shape[1] != self.n_classes:
            full = np.full((n, self.n_classes), margins.min() - 1.0)
            full[:, self.model.classes_] = margins
            margins = full
        out["linear"] = margins.astype(np.float32)
        out["rocchio"] = self.rocchio.scores(X)
        out["prototype"] = self.prototypes.scores(doc_embeddings, n, self.n_classes)
        for name in ("heading", "verbatim"):
            out[name] = (
                lexical[name]
                if lexical and name in lexical
                else np.zeros((n, self.n_classes), dtype=np.float32)
            )
        return out

    @staticmethod
    def _zscore(mat: np.ndarray) -> np.ndarray:
        mu = mat.mean(axis=1, keepdims=True)
        sd = mat.std(axis=1, keepdims=True)
        return (mat - mu) / np.clip(sd, 1e-6, None)

    def combine(self, channels: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> np.ndarray:
        w = weights if weights is not None else self.weights
        total = None
        for name, mat in channels.items():
            weight = float(w.get(name, 0.0))
            if weight == 0.0:
                continue
            term = self._zscore(mat) * weight
            total = term if total is None else total + term
        if total is None:
            total = np.zeros_like(next(iter(channels.values())))
        return total

    def calibrate(self, combined: np.ndarray, y: np.ndarray) -> None:
        self.calibrator.fit(combined.astype(np.float64), y)

    def log_proba(self, combined: np.ndarray) -> np.ndarray:
        return self.calibrator.log_proba(combined.astype(np.float64))
