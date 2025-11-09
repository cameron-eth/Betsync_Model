"""
Helpers for locating and validating model bundle artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class BundleMetadata:
    version: str
    trained_through: str
    feature_hash: str
    extra: Dict[str, Any]

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "BundleMetadata":
        return cls(
            version=data.get("version", "0.0.0"),
            trained_through=data.get("trained_through", ""),
            feature_hash=data.get("feature_hash", ""),
            extra={k: v for k, v in data.items() if k not in {"version", "trained_through", "feature_hash"}},
        )


def metadata_path() -> Path:
    return ARTIFACT_DIR / "metadata.json"


def load_metadata() -> BundleMetadata:
    """Load bundle metadata if present, otherwise return defaults."""
    path = metadata_path()
    if not path.exists():
        return BundleMetadata(version="0.0.0", trained_through="", feature_hash="", extra={})

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return BundleMetadata.from_json(data)


def require_artifact(name: str) -> Path:
    """Return the path to an artifact, raising if it is missing."""
    path = ARTIFACT_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Expected artifact '{name}' not found in {ARTIFACT_DIR}. "
            "Add the trained model files before building the package."
        )
    return path

