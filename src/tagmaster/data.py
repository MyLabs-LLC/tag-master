"""Fetching nvidia/Nemotron-PII and splitting it without leakage.

Two properties of the source dataset drive everything here:

1. The shipped train/test splits use near-disjoint `domain` vocabularies -- only
   `Insurance` appears in both, and `Healthcare`/`Government`/`Finance` occur
   only in train. Training on one and evaluating on the other is impossible for
   this label set, so both splits are pooled and re-split.
2. Every logical record appears twice, once per `locale` (`us` and `intl`),
   sharing a `uid`. A row-level split therefore puts a near-duplicate of most
   test documents into train. All splitting is on `uid`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

from .config import Config
from .taxonomy import OTHER, Taxonomy, normalize_tag

PARQUET_URL = "https://huggingface.co/api/datasets/{repo}/parquet/default/{split}/0.parquet"
SOURCE_SPLITS = ("train", "test")

COLUMNS = (
    "uid",
    "domain",
    "document_type",
    "document_description",
    "document_format",
    "locale",
    "text",
)


@dataclass
class Frame:
    """A column-oriented view of one split. Parallel arrays, one entry per row."""

    uid: np.ndarray
    domain: np.ndarray
    top: np.ndarray
    tag: np.ndarray
    description: np.ndarray
    document_format: np.ndarray
    locale: np.ndarray
    text: np.ndarray

    def __len__(self) -> int:
        return len(self.uid)

    def select(self, mask: np.ndarray) -> Frame:
        return Frame(**{k: v[mask] for k, v in self.__dict__.items()})

    def where_top(self, top: str) -> Frame:
        return self.select(self.top == top)


@dataclass
class Splits:
    train: Frame
    dev: Frame
    test: Frame

    def items(self) -> list[tuple[str, Frame]]:
        return [("train", self.train), ("dev", self.dev), ("test", self.test)]


def download(cfg: Config, force: bool = False) -> dict[str, Path]:
    """Fetch the raw parquet files. ~307 MB total, cached on disk."""
    cfg.paths.ensure()
    repo = cfg["dataset"]["repo_id"]
    out: dict[str, Path] = {}
    for split in SOURCE_SPLITS:
        dest = cfg.paths.raw / f"{split}.parquet"
        if force or not dest.exists():
            url = PARQUET_URL.format(repo=repo, split=split)
            con = duckdb.connect()
            con.execute("INSTALL httpfs; LOAD httpfs;")
            con.execute(
                f"COPY (SELECT {', '.join(COLUMNS)} FROM read_parquet(?)) TO ? (FORMAT PARQUET)",
                [url, str(dest)],
            )
            con.close()
        out[split] = dest
    return out


def _load_pooled(cfg: Config, taxonomy: Taxonomy) -> Frame:
    files = download(cfg)
    con = duckdb.connect()
    rel = con.execute(
        f"SELECT {', '.join(COLUMNS)} FROM read_parquet(?)",
        [[str(p) for p in files.values()]],
    )
    rows = rel.fetchall()
    con.close()

    cols = list(zip(*rows))
    by_name = dict(zip(COLUMNS, cols))
    domain = np.array(by_name["domain"], dtype=object)
    taxonomy.validate_domains(set(domain.tolist()))

    top = np.array([taxonomy.top_for_domain(d) for d in domain], dtype=object)
    tag = np.array([normalize_tag(t) for t in by_name["document_type"]], dtype=object)

    return Frame(
        uid=np.array(by_name["uid"], dtype=object),
        domain=domain,
        top=top,
        tag=tag,
        description=np.array(by_name["document_description"], dtype=object),
        document_format=np.array(by_name["document_format"], dtype=object),
        locale=np.array(by_name["locale"], dtype=object),
        text=np.array(by_name["text"], dtype=object),
    )


def _stratified_uid_assignment(
    uids: np.ndarray,
    strata: np.ndarray,
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """Assign each uid to train/dev/test, balancing every stratum independently.

    Locale needs no explicit handling: every uid carries exactly one `us` and one
    `intl` row, so locale balance follows from splitting on uid.
    """
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    f_train, f_dev, _ = fractions

    for stratum in np.unique(strata):
        members = uids[strata == stratum]
        order = rng.permutation(len(members))
        members = members[order]
        n = len(members)
        n_train = int(round(f_train * n))
        n_dev = int(round(f_dev * n))
        # Tiny strata must still yield a usable train partition.
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_dev = max(1, min(n_dev, n - n_train - 1))
        for i, uid in enumerate(members):
            if i < n_train:
                assignment[uid] = "train"
            elif i < n_train + n_dev:
                assignment[uid] = "dev"
            else:
                assignment[uid] = "test"
    return assignment


def build_splits(cfg: Config, taxonomy: Taxonomy) -> Splits:
    frame = _load_pooled(cfg, taxonomy)

    # Collapse to uid level for splitting. Format and top are constant within a
    # uid, so taking the first occurrence is exact rather than approximate.
    uids, first_idx = np.unique(frame.uid, return_index=True)
    strata = np.array(
        [f"{frame.top[i]}|{frame.document_format[i]}" for i in first_idx],
        dtype=object,
    )
    sp = cfg["split"]
    assignment = _stratified_uid_assignment(
        uids,
        strata,
        (float(sp["train"]), float(sp["dev"]), float(sp["test"])),
        int(sp["seed"]),
    )

    row_split = np.array([assignment[u] for u in frame.uid], dtype=object)
    splits = Splits(
        train=frame.select(row_split == "train"),
        dev=frame.select(row_split == "dev"),
        test=frame.select(row_split == "test"),
    )
    assert_no_uid_leakage(splits)
    return splits


def assert_no_uid_leakage(splits: Splits) -> None:
    """The invariant that keeps every reported metric honest."""
    sets = {name: set(frame.uid.tolist()) for name, frame in splits.items()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        shared = sets[a] & sets[b]
        if shared:
            raise AssertionError(
                f"{len(shared)} uids appear in both {a} and {b}; the us/intl pair of a "
                "test document would be visible during training"
            )


def cache_splits(cfg: Config, splits: Splits) -> Path:
    cfg.paths.ensure()
    dest = cfg.paths.data / "splits.npz"
    payload: dict[str, np.ndarray] = {}
    for name, frame in splits.items():
        for field_name, arr in frame.__dict__.items():
            payload[f"{name}__{field_name}"] = arr
    np.savez_compressed(dest, **payload)
    return dest


def load_cached_splits(cfg: Config) -> Splits | None:
    src = cfg.paths.data / "splits.npz"
    if not src.exists():
        return None
    blob = np.load(src, allow_pickle=True)
    frames: dict[str, Frame] = {}
    for name in ("train", "dev", "test"):
        frames[name] = Frame(
            **{f: blob[f"{name}__{f}"] for f in Frame.__dataclass_fields__}
        )
    return Splits(**frames)


def get_splits(cfg: Config, taxonomy: Taxonomy, rebuild: bool = False) -> Splits:
    if not rebuild:
        cached = load_cached_splits(cfg)
        if cached is not None:
            return cached
    splits = build_splits(cfg, taxonomy)
    cache_splits(cfg, splits)
    return splits


def split_fingerprint(splits: Splits) -> str:
    """Stable id for a split, so feature caches cannot be reused across splits."""
    h = hashlib.sha256()
    for name, frame in splits.items():
        h.update(name.encode())
        h.update(str(len(frame)).encode())
        h.update(",".join(sorted(set(frame.uid.tolist()))[:64]).encode())
    return h.hexdigest()[:16]


def describe(splits: Splits, taxonomy: Taxonomy) -> dict:
    """Row/uid counts per split and tier-1 class, for the prepare report."""
    out: dict = {"splits": {}, "totals": {}}
    for name, frame in splits.items():
        per_top = {}
        for top in taxonomy.top_labels:
            mask = frame.top == top
            per_top[top] = {
                "rows": int(mask.sum()),
                "uids": int(len(set(frame.uid[mask].tolist()))),
            }
        out["splits"][name] = {
            "rows": len(frame),
            "uids": len(set(frame.uid.tolist())),
            "per_top": per_top,
            "per_locale": {
                loc: int((frame.locale == loc).sum())
                for loc in sorted(set(frame.locale.tolist()))
            },
        }
    industry_rows = sum(
        v["per_top"][t]["rows"] for v in out["splits"].values() for t in taxonomy.industries
    )
    other_rows = sum(v["per_top"][OTHER]["rows"] for v in out["splits"].values())
    out["totals"] = {"industry_rows": industry_rows, "other_rows": other_rows}
    return out
