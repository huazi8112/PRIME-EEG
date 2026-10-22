"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from .io import load_features
DATASETS = {}


def robust_parameters(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1e-4, scale, np.maximum(fallback, 1e-4))
    return center.astype(np.float32), scale.astype(np.float32)


def align(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Robust patient-background coordinate alignment; parameters use fit data only."""
    return np.clip((values - center) / scale, -20.0, 20.0).astype(np.float32)


@dataclass
class SubjectSource:
    dataset: str
    subject: str
    root: Path
    manifest: dict
    names: list[str]
    center: np.ndarray
    scale: np.ndarray
    fit_raw: list[np.ndarray]
    calibration_raw: list[np.ndarray]
    calibration_aligned: np.ndarray
    positive_aligned: np.ndarray


def load_source(dataset: str, subject: str) -> SubjectSource:
    root = DATASETS[dataset] / subject
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    normal_rows = {row["record"]: row for row in manifest["normals"]}
    roles = manifest["normal_records"]
    names: list[str] | None = None
    fit_raw = []
    calibration_raw = []
    for role, destination in (("fit", fit_raw), ("calibration", calibration_raw)):
        for record in roles[role]:
            values, current_names, _ = load_features(normal_rows[record]["cache_file"])
            names = current_names if names is None else names
            if names != current_names:
                raise AssertionError(f"schema mismatch {dataset}:{subject}:{record}")
            destination.append(values)
    assert names is not None
    center, scale = robust_parameters(np.vstack(fit_raw))
    calibration_aligned = np.vstack(
        [align(values, center, scale) for values in calibration_raw]
    )
    positives = []
    for event in manifest["events"]:
        values, current_names, extras = load_features(event["cache_file"])
        if current_names != names:
            raise AssertionError(f"event schema mismatch {dataset}:{subject}")
        labels = np.asarray(extras["labels"], dtype=bool)
        positives.append(align(values[labels], center, scale))
    return SubjectSource(
        dataset=dataset,
        subject=subject,
        root=root,
        manifest=manifest,
        names=names,
        center=center,
        scale=scale,
        fit_raw=fit_raw,
        calibration_raw=calibration_raw,
        calibration_aligned=calibration_aligned,
        positive_aligned=np.vstack(positives),
    )


def capped_rows(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return values[indices]


@dataclass
class PrototypeModel:
    keep: np.ndarray
    weights: np.ndarray
    prototypes: np.ndarray
    signs: np.ndarray
    excess_reference: np.ndarray

    def scores(self, values: np.ndarray) -> dict[str, np.ndarray]:
        selected = values[:, self.keep]
        projection = selected @ self.weights
        value_norm = np.linalg.norm(selected, axis=1, keepdims=True)
        proto_norm = np.linalg.norm(self.prototypes, axis=1, keepdims=True).T
        cosine = (selected @ self.prototypes.T) / np.maximum(
            value_norm * proto_norm, 1e-6
        )
        directed = selected * self.signs
        signed_excess = np.mean(
            np.maximum(directed - self.excess_reference, 0.0), axis=1
        )
        return {
            "projection": projection.astype(np.float32),
            "prototype": np.max(cosine, axis=1).astype(np.float32),
            "signed_excess": signed_excess.astype(np.float32),
        }


def fit_prototype(
    fit_normal: np.ndarray,
    first_positive: np.ndarray,
    available: np.ndarray,
    top_k: int,
) -> PrototypeModel:
    normal = fit_normal[:, available]
    positive = first_positive[:, available]
    normal_center = np.median(normal, axis=0)
    positive_center = np.median(positive, axis=0)
    effect = positive_center - normal_center
    positive_spread = 1.4826 * np.median(
        np.abs(positive - positive_center), axis=0
    )
    reliability = np.abs(effect) / np.maximum(0.5 + positive_spread, 0.5)
    local = np.argsort(reliability)[-min(top_k, len(available)) :]
    keep = available[local]
    kept_effect = effect[local]
    weights = kept_effect / max(float(np.linalg.norm(kept_effect)), 1e-6)
    prototypes = first_positive[:, keep]
    if len(prototypes) > 24:
        positions = np.linspace(0, len(prototypes) - 1, 24).round().astype(int)
        prototypes = prototypes[positions]
    signs = np.sign(kept_effect)
    signs[signs == 0] = 1.0
    excess_reference = np.quantile(fit_normal[:, keep] * signs, 0.95, axis=0)
    return PrototypeModel(
        keep=keep,
        weights=weights.astype(np.float32),
        prototypes=prototypes.astype(np.float32),
        signs=signs.astype(np.float32),
        excess_reference=excess_reference.astype(np.float32),
    )
