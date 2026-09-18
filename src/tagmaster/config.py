"""Configuration loading and the on-disk layout of generated artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "taxonomy.yaml"


def _artifact_root() -> Path:
    env = os.environ.get("TAGMASTER_ARTIFACTS")
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "artifacts"


@dataclass(frozen=True)
class Paths:
    """Where generated data, caches, models, and reports live.

    Everything here is regenerable and git-ignored; nothing under these paths is
    required to check out and read the code.
    """

    root: Path = field(default_factory=_artifact_root)

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def reports(self) -> Path:
        return REPO_ROOT / "reports"

    def ensure(self) -> None:
        for p in (self.raw, self.data, self.cache, self.models, self.reports):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path
    paths: Paths = field(default_factory=Paths)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def seed(self) -> int:
        return int(self.raw["split"]["seed"])


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, path=cfg_path)
