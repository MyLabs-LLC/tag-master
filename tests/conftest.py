from __future__ import annotations

import numpy as np
import pytest

from tagmaster.config import load_config
from tagmaster.data import Frame
from tagmaster.taxonomy import build_taxonomy


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def taxonomy(cfg):
    return build_taxonomy(cfg)


def make_frame(rows: list[dict]) -> Frame:
    """Build a Frame from dicts, so tests need no dataset download."""
    fields = list(Frame.__dataclass_fields__)
    return Frame(
        **{
            name: np.array([r.get(name, "") for r in rows], dtype=object)
            for name in fields
        }
    )


@pytest.fixture
def frame_factory():
    return make_frame
