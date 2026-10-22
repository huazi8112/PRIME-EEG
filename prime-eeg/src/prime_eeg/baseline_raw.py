"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
import itertools
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
TARGET = {'event_sensitivity_min': 0.75, 'far_per_hour_max': 0.7, 'event_precision_min': 0.25}


RAW_FEATURES = (
    "maximum", "p99", "p95", "mean", "top1pct_mean", "fraction_05",
    "second_maximum", "second_p95", "second_mean",
)


@dataclass(frozen=True)
class Rule:
    representation: str
    threshold_mode: str
    threshold_value: float
    persistence: int


def load_score(path: Path) -> dict:
    with np.load(path) as archive:
        output = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key != "metadata_json"
        }
        output.update(json.loads(str(archive["metadata_json"])))
    return output


def feature_matrix(record: dict) -> np.ndarray:
    return np.column_stack([np.asarray(record[name], dtype=float) for name in RAW_FEATURES])


def role_values(subject_root: Path, specifications: list[str]) -> np.ndarray:
    parts = []
    for specification in specifications:
        fields = specification.split(":")
        record = load_score(subject_root / "records" / f"{fields[0]}.npz")
        mask = np.ones(len(record[RAW_FEATURES[0]]), dtype=bool)
        if len(fields) == 3:
            start, stop = float(fields[1]), float(fields[2])
            mask &= np.asarray(record["interval_starts"], dtype=float) >= start
            mask &= np.asarray(record["interval_ends"], dtype=float) <= stop
        parts.append(feature_matrix(record)[mask])
    if not parts or not any(len(part) for part in parts):
        raise ValueError(f"empty normal role under {subject_root}")
    return np.vstack(parts)


def robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    fallback = (q75 - q25) / 1.349
    scale = np.maximum(mad, np.maximum(fallback, 1e-5))
    return center, scale


def unit_direction(normalized_positive: np.ndarray) -> np.ndarray:
    direction = np.median(normalized_positive, axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.zeros(normalized_positive.shape[1], dtype=float)
        direction[0] = 1.0
        return direction
    return direction / norm


def max_z(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.max((values - center) / scale, axis=1)


def persistent(mask: np.ndarray, count: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if count <= 1:
        return mask
    cumulative = np.cumsum(mask.astype(np.int32))
    totals = cumulative.copy()
    totals[count:] -= cumulative[:-count]
    return mask & (totals >= count)


def threshold_array(payload: dict, record: dict, rule: Rule) -> np.ndarray:
    if rule.threshold_mode == "absolute":
        return np.full(len(record["interval_ends"]), rule.threshold_value, dtype=float)
    if rule.representation == "projection_sequential":
        references = record["projection_sequential_calibration"]
        return np.quantile(references, rule.threshold_value, axis=1)
    reference = payload["calibration"][rule.representation]
    return np.full(
        len(record["interval_ends"]),
        float(np.quantile(reference, rule.threshold_value)),
        dtype=float,
    )


def representation_values(record: dict, representation: str) -> np.ndarray:
    if representation in RAW_FEATURES:
        return record["raw"][representation]
    return np.asarray(record[representation], dtype=float)


def configuration_key(rule: Rule) -> tuple:
    return (
        rule.representation,
        rule.threshold_mode,
        float(rule.threshold_value),
        int(rule.persistence),
    )


def selection_key(result: dict, rule: Rule) -> tuple:
    far_violation = max(
        0.0,
        (result["far_per_hour"] - TARGET["far_per_hour_max"])
        / TARGET["far_per_hour_max"],
    )
    precision_violation = max(
        0.0,
        (TARGET["event_precision_min"] - result["event_precision"])
        / TARGET["event_precision_min"],
    )
    feasible = far_violation == 0.0 and precision_violation == 0.0
    if feasible:
        return (
            0, -result["event_sensitivity"], -result["event_f1"],
            result["far_per_hour"], configuration_key(rule),
        )
    return (
        1, max(far_violation, precision_violation),
        far_violation + precision_violation, -result["event_sensitivity"],
        -result["event_f1"], configuration_key(rule),
    )


def rules() -> list[Rule]:
    output = []
    quantiles = (0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999, 0.9995)
    representations = RAW_FEATURES + (
        "max_z", "projection_static", "projection_sequential",
    )
    for representation, quantile, persistence_count in itertools.product(
        representations, quantiles, (1, 2, 3, 4)
    ):
        output.append(
            Rule(representation, "patient_normal_quantile", quantile, persistence_count)
        )
    # The checkpoint's published sigmoid output uses an absolute probability.
    # Include common operating thresholds without granting target-patient tuning.
    for representation, threshold, persistence_count in itertools.product(
        RAW_FEATURES, (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), (1, 2, 3, 4)
    ):
        output.append(Rule(representation, "absolute", threshold, persistence_count))
    return sorted(set(output), key=configuration_key)
