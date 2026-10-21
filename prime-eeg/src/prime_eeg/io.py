"""Portable file handling; all dataset locations are supplied by the caller."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

RESOURCE_ROOT = Path(__file__).parent / "resources"
CACHE_BASE: Path | None = None


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def finite(value):
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    # An all-missed bootstrap sample has undefined conditional latency. Serialize
    # that missing quantity as null, not invalid JSON NaN, without altering scores.
    path.write_text(json.dumps(finite(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resource(name: str):
    return read_json(RESOURCE_ROOT / name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_cache_file(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        if CACHE_BASE is None:
            raise RuntimeError("Set the dataset cache root before loading feature files")
        path = CACHE_BASE / path
        if not path.resolve().is_relative_to(CACHE_BASE.resolve()):
            raise ValueError("Cache path escapes the configured dataset directory")
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {path}")
    return path


def load_features(path: str) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    with np.load(resolve_cache_file(path), allow_pickle=False) as archive:
        extras = {key: np.asarray(archive[key]) for key in ("labels", "seconds") if key in archive.files}
        return (
            np.asarray(archive["features"], dtype=np.float32),
            [str(value) for value in archive["names"]],
            extras,
        )


def portable_manifest(path: Path, root: Path) -> None:
    """Normalize only newly generated manifests; feature arrays are untouched."""
    payload = read_json(path)

    def visit(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "cache_file":
                    obj[key] = Path(value).resolve().relative_to(root.resolve()).as_posix()
                else:
                    visit(value)
        elif isinstance(obj, list):
            for value in obj:
                visit(value)

    visit(payload)
    write_json(path, payload)


def verify_adapter() -> dict:
    """Verify release configuration, not the hashes of a different source layout."""
    path = RESOURCE_ROOT / "siena_adapter.json"
    checksums = resource("checksums.json")
    if sha256(path) != checksums["siena_adapter.json"]:
        raise RuntimeError("Release Siena adapter configuration changed")
    return {**read_json(path), "sha256": checksums["siena_adapter.json"]}


def verify_resources() -> int:
    entries = resource("checksums.json")
    for name, expected in entries.items():
        if sha256(RESOURCE_ROOT / name) != expected:
            raise AssertionError(f"Frozen resource changed: {name}")
    return len(entries)
