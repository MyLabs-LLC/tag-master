"""Mapping the dataset's 58 free-text domains onto four tier-1 classes, and
normalising the free-text `document_type` into a stable tier-2 label space."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config

OTHER = "other"

_PUNCT = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")

# Tokens that survive normalisation but carry no discriminative weight when
# comparing two tag names for near-duplicate detection.
_NOISE_TOKENS = frozenset({"a", "an", "the", "of", "for", "and", "to", "in", "on"})

# Word endings that only look like plurals. Without these, "diagnosis" becomes
# "diagnosi" and "census" becomes "censu", which is harmless for matching but
# makes every report and prediction look mangled.
_FALSE_PLURAL_ENDINGS = ("ss", "is", "us", "as", "ys")


def _singularize(token: str) -> str:
    if len(token) <= 3 or not token.endswith("s"):
        return token
    if token.endswith(_FALSE_PLURAL_ENDINGS):
        return token
    # supplies -> supply, policies -> policy. Stripping the bare "s" would give
    # "supplie", which also fails to match the singular form "supply".
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    return token[:-1]


class TaxonomyError(RuntimeError):
    """Raised when the configured mapping and the observed data disagree."""


def normalize_tag(raw: str) -> str:
    """Canonical surface form for a `document_type` value.

    The column is free text, so the same document kind appears as
    "Admission Checklist" and "admission checklist", and singular/plural
    variants coexist ("medical record" / "medical records"). Casing, punctuation,
    whitespace, and trailing plurals are folded; nothing semantic is touched.
    """
    s = _PUNCT.sub(" ", raw.strip().lower())
    s = _WS.sub(" ", s).strip()
    if not s:
        return ""
    return " ".join(_singularize(tok) for tok in s.split(" "))


def tag_tokens(tag: str) -> frozenset[str]:
    """Content tokens of a normalised tag, for near-duplicate comparison."""
    return frozenset(t for t in tag.split(" ") if t and t not in _NOISE_TOKENS)


@dataclass(frozen=True)
class Taxonomy:
    """The tier-1 label space plus the domain mapping that produces it."""

    industries: tuple[str, ...]
    domain_to_top: dict[str, str]
    known_domains: frozenset[str]
    hard_negatives: dict[str, frozenset[str]]

    @property
    def top_labels(self) -> tuple[str, ...]:
        """Tier-1 classes in a fixed order, with `other` last."""
        return (*self.industries, OTHER)

    def top_for_domain(self, domain: str) -> str:
        return self.domain_to_top.get(domain, OTHER)

    def is_hard_negative(self, domain: str) -> bool:
        """True for `other` domains that sit adjacent to one of the industries.

        These generate most of the tier-1 boundary errors (a Legal
        confidentiality agreement against a Government one), so the trainer
        up-weights them rather than treating all `other` domains alike.
        """
        return any(domain in doms for doms in self.hard_negatives.values())

    def validate_domains(self, observed: set[str]) -> None:
        unknown = sorted(observed - self.known_domains)
        if unknown:
            raise TaxonomyError(
                "dataset contains domains missing from configs/taxonomy.yaml "
                f"known_domains: {unknown}. Add them explicitly so they are not "
                "silently absorbed into the 'other' class."
            )


def build_taxonomy(cfg: Config) -> Taxonomy:
    industries_cfg: dict[str, list[str]] = cfg["industries"]
    industries = tuple(industries_cfg.keys())
    if OTHER in industries:
        raise TaxonomyError(f"'{OTHER}' is the implicit fourth class and must not be an industry key")

    domain_to_top: dict[str, str] = {}
    for top, domains in industries_cfg.items():
        for dom in domains:
            if dom in domain_to_top:
                raise TaxonomyError(
                    f"domain {dom!r} is claimed by both {domain_to_top[dom]!r} and {top!r}"
                )
            domain_to_top[dom] = top

    known = frozenset(cfg["known_domains"])
    missing = sorted(set(domain_to_top) - known)
    if missing:
        raise TaxonomyError(f"industry domains absent from known_domains: {missing}")

    hard_raw: dict[str, list[str]] = cfg.get("hard_negative_domains") or {}
    hard: dict[str, frozenset[str]] = {}
    for top, domains in hard_raw.items():
        overlap = sorted(set(domains) & set(domain_to_top))
        if overlap:
            raise TaxonomyError(
                f"hard negatives for {top!r} include domains mapped to an industry: {overlap}"
            )
        hard[top] = frozenset(domains)

    return Taxonomy(
        industries=industries,
        domain_to_top=domain_to_top,
        known_domains=known,
        hard_negatives=hard,
    )


@dataclass
class TagVocabulary:
    """The tier-2 label space: normalised tags, grouped by industry.

    Tier 2 is only ever asked "which sub-tag, given this industry", so the label
    space is stored per industry rather than as one flat list. `aliases` is the
    optional canonicalisation layer produced by `canonicalize.py`.
    """

    tags_by_top: dict[str, list[str]]
    aliases: dict[str, str]

    def __post_init__(self) -> None:
        self.tags_by_top = {k: sorted(set(v)) for k, v in self.tags_by_top.items()}
        self._index = {
            top: {tag: i for i, tag in enumerate(tags)} for top, tags in self.tags_by_top.items()
        }

    def tags(self, top: str) -> list[str]:
        return self.tags_by_top.get(top, [])

    def index(self, top: str, tag: str) -> int | None:
        return self._index.get(top, {}).get(tag)

    def canonical(self, top: str, tag: str) -> str:
        """Apply the learned alias map, keyed by industry to avoid cross-industry merges."""
        return self.aliases.get(f"{top}\t{tag}", tag)

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.tags_by_top.values())

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"tags_by_top": self.tags_by_top, "aliases": self.aliases},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> TagVocabulary:
        blob = json.loads(path.read_text(encoding="utf-8"))
        return cls(tags_by_top=blob["tags_by_top"], aliases=blob.get("aliases", {}))


def apply_aliases(vocab: TagVocabulary, aliases: dict[str, str]) -> TagVocabulary:
    """Return a coarser vocabulary with `aliases` folded in."""
    merged: dict[str, list[str]] = {}
    for top, tags in vocab.tags_by_top.items():
        merged[top] = sorted({aliases.get(f"{top}\t{t}", t) for t in tags})
    return TagVocabulary(tags_by_top=merged, aliases=aliases)
