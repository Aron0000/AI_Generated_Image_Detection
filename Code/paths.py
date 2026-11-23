"""Single source of truth for project paths.

The repository root is resolved from this file's location, so the project can
be cloned or copied anywhere.  Environment variables let a machine keep the
feature cache, model weights or dataset on another drive without editing code:

    EE5438_DATA     dataset root          (default: <root>/Dataset)
    EE5438_CACHE    feature cache          (default: ~/ee5438_cache)
    EE5438_DINO     DINOv2 weights         (default: ~/dinov2-large)
"""

from __future__ import annotations

import os
from pathlib import Path


def _path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default


ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "Code"

DATA = _path("EE5438_DATA", ROOT / "Dataset")
CACHE = _path("EE5438_CACHE", Path.home() / "ee5438_cache")
DINO_WEIGHTS = _path("EE5438_DINO", Path.home() / "dinov2-large")

RESULTS = ROOT / "Results"
RESULTS_V3 = RESULTS / "final_v3"
REPORT = ROOT / "Report"
