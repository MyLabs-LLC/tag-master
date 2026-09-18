"""Heading extraction and lexical tag matching.

The `document_type` label appears verbatim in the document text ~72% of the
time, because the structured half of the corpus opens with its own title
(`**Blood Donor Registration Form**`). Recovering that turns much of tier 2 into
a lookup, and unlike a discriminative head it still works for the thin tail --
the rarest government tag has two training records, which no classifier can
learn but a title match handles for free.

Exact heading equality alone only recovers ~25% of labels, because unstructured
documents state the tag mid-sentence ("This Brokerage and Investment Agreement
is entered into...") and structured titles pick up domain prefixes
("Brokerage and Confidentiality Agreement" for `confidentiality agreement`). So
matching is split into two channels that the tier-2 ensemble weights
independently:

* `heading`  -- structural evidence from an extracted title.
* `verbatim` -- the tag occurring anywhere in the body, weighted by position.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .taxonomy import TagVocabulary, normalize_tag, tag_tokens

_WORD = re.compile(r"[a-z][a-z0-9]+")

# Markdown-ish title forms seen in the corpus, most explicit first.
_HEADING_PATTERNS = (
    re.compile(r"^\s*#{1,6}\s*(?P<h>[^\n#]{3,120})", re.MULTILINE),
    re.compile(r"^\s*\*\*(?P<h>[^*\n]{3,120})\*\*\s*:?\s*$", re.MULTILINE),
    re.compile(r"^\s*__(?P<h>[^_\n]{3,120})__\s*:?\s*$", re.MULTILINE),
)

# Leading boilerplate that precedes the real title in unstructured documents.
_SALUTATION = re.compile(
    r"^\s*(dear\b|to whom it may concern|hello\b|hi\b|greetings\b|subject\s*:|re\s*:)",
    re.IGNORECASE,
)

# Table rows, code fences, and rule lines are never titles.
_NON_TITLE = re.compile(r"^(```|~~~|\||[-=_*\s]+$)")

_TITLE_STOP = re.compile(r"[.!?]\s")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def extract_headings(text: str, window: int = 400) -> list[str]:
    """Candidate titles for a document, best first.

    Several candidates are returned rather than one because the corpus mixes
    markdown headings, bold titles, `Subject:` lines, and plain first lines; the
    matcher scores all of them and keeps the best.
    """
    if not text:
        return []
    head = text[: max(window, 64)]
    candidates: list[str] = []

    for pat in _HEADING_PATTERNS:
        for m in pat.finditer(head):
            h = m.group("h").strip(" :*_#-")
            if h and not _NON_TITLE.match(h):
                candidates.append(h)

    for line in head.splitlines():
        raw = line.strip()
        if _NON_TITLE.match(raw):
            continue
        line = raw.strip(" :*_#-\t")
        if not line or len(line) < 3 or _SALUTATION.match(line):
            continue
        # A title is a short fragment, not a sentence.
        first = _TITLE_STOP.split(line)[0].strip()
        if 3 <= len(first) <= 120:
            candidates.append(first)
        if len(candidates) >= 14:
            break

    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


@dataclass
class LexicalScores:
    """Two (n_docs, n_tags) score channels, aligned to `vocab.tags(top)`."""

    heading: np.ndarray
    verbatim: np.ndarray

    def channels(self) -> dict[str, np.ndarray]:
        return {"heading": self.heading, "verbatim": self.verbatim}


@dataclass
class HeadingMatcher:
    """Scores documents against every tag in an industry's label space.

    The heading channel is IDF-weighted coverage of the *tag name* by the
    document, boosted when tag tokens land inside an extracted title and again
    when the title's tokens are a superset of the tag's. Coverage is measured
    over the tag rather than the document so long documents do not dominate, and
    it is normalised by total tag IDF so a five-token tag is not systematically
    outscored by a two-token one.
    """

    idf: dict[str, float]
    default_idf: float
    tags_by_top: dict[str, list[str]]
    heading_window: int = 400
    body_chars: int = 2500

    @classmethod
    def fit(
        cls,
        texts: np.ndarray,
        vocab: TagVocabulary,
        heading_window: int = 400,
        sample: int = 40000,
        seed: int = 0,
    ) -> HeadingMatcher:
        rng = np.random.default_rng(seed)
        if len(texts) > sample:
            texts = texts[rng.choice(len(texts), size=sample, replace=False)]
        df: dict[str, int] = {}
        for t in texts:
            for w in set(tokenize(t[:2000])):
                df[w] = df.get(w, 0) + 1
        n = max(len(texts), 1)
        return cls(
            idf={w: math.log(n / (1 + d)) for w, d in df.items()},
            default_idf=math.log(n),
            tags_by_top=dict(vocab.tags_by_top),
            heading_window=heading_window,
        )

    def _idf(self, token: str) -> float:
        return self.idf.get(token, self.default_idf)

    def _tag_cache(self, top: str) -> list[tuple[frozenset[str], float]]:
        tags = self.tags_by_top.get(top, [])
        out = []
        for tag in tags:
            toks = tag_tokens(tag)
            out.append((toks, sum(self._idf(t) for t in toks)))
        return out

    def score_document(self, text: str, top: str) -> LexicalScores:
        tags = self.tags_by_top.get(top, [])
        heading = np.zeros(len(tags), dtype=np.float32)
        verbatim = np.zeros(len(tags), dtype=np.float32)
        if not tags or not text:
            return LexicalScores(heading, verbatim)

        headings = extract_headings(text, self.heading_window)
        heading_token_sets = [frozenset(tokenize(h)) for h in headings[:5]]
        heading_tokens: set[str] = set().union(*heading_token_sets) if heading_token_sets else set()
        heading_norms = {normalize_tag(h) for h in headings[:6]}

        body = text[: self.body_chars]
        body_norm = normalize_tag(body)
        body_tokens = set(body_norm.split(" "))
        body_len = max(len(body_norm), 1)

        for i, (toks, total) in enumerate(self._tag_cache(top)):
            if not toks or total <= 0:
                continue

            got = 0.0
            for t in toks:
                w = self._idf(t)
                if t in heading_tokens:
                    got += 2.0 * w
                elif t in body_tokens:
                    got += w
            score = got / (2.0 * total)

            tag = tags[i]
            if tag in heading_norms:
                score += 1.0
            elif any(toks <= hs for hs in heading_token_sets if hs):
                # "Brokerage and Confidentiality Agreement" contains
                # `confidentiality agreement`: strong, but weaker than equality.
                score += 0.5
            heading[i] = score

            # Verbatim channel: only worth a substring scan when every token of
            # the tag is present, which prunes the vast majority of candidates.
            if toks <= body_tokens:
                pos = body_norm.find(tag)
                if pos >= 0:
                    verbatim[i] = 1.0 - 0.5 * (pos / body_len)

        return LexicalScores(heading, verbatim)

    def score_matrix(self, texts: np.ndarray, top: str, progress: bool = False) -> LexicalScores:
        tags = self.tags_by_top.get(top, [])
        heading = np.zeros((len(texts), len(tags)), dtype=np.float32)
        verbatim = np.zeros((len(texts), len(tags)), dtype=np.float32)
        iterator = enumerate(texts)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(texts), desc=f"lexical[{top}]", unit="doc")
        for i, t in iterator:
            s = self.score_document(t, top)
            heading[i] = s.heading
            verbatim[i] = s.verbatim
        return LexicalScores(heading, verbatim)

    def verbatim_rate(self, texts: np.ndarray, tags: np.ndarray) -> float:
        """Fraction of documents whose gold tag appears verbatim in the text.

        A data sanity check: should land near the 0.72 measured directly against
        the parquet files.
        """
        hits = sum(
            1 for text, tag in zip(texts, tags) if tag and tag in normalize_tag(text[:3000])
        )
        return hits / max(len(texts), 1)

    def heading_hit_rate(self, texts: np.ndarray, tags: np.ndarray) -> float:
        """Fraction whose gold tag is recovered exactly from an extracted heading."""
        hits = 0
        for text, tag in zip(texts, tags):
            norms = {normalize_tag(h) for h in extract_headings(text, self.heading_window)}
            if tag in norms:
                hits += 1
        return hits / max(len(texts), 1)

    def top1_accuracy(self, texts: np.ndarray, tags: np.ndarray, top: str) -> dict[str, float]:
        """Standalone top-1 accuracy of each channel, for the prepare report."""
        tag_list = self.tags_by_top.get(top, [])
        if not tag_list:
            return {}
        index = {t: i for i, t in enumerate(tag_list)}
        gold = np.array([index.get(t, -1) for t in tags])
        scores = self.score_matrix(texts, top)
        out = {}
        for name, mat in scores.channels().items():
            out[name] = float(np.mean(mat.argmax(axis=1) == gold))
        combined = scores.heading + 0.5 * scores.verbatim
        out["combined"] = float(np.mean(combined.argmax(axis=1) == gold))
        return out
