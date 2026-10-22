"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
TARGET = {'event_sensitivity_min': 0.75, 'far_per_hour_max': 0.7, 'event_precision_min': 0.25}


def aggregate(rows: list[dict]) -> dict:
    detected = sum(int(row["detected"]) for row in rows)
    total = sum(int(row["total"]) for row in rows)
    matched = sum(int(row["matched_predictions"]) for row in rows)
    false = sum(int(row["false_alarms"]) for row in rows)
    hours = sum(float(row["monitoring_hours"]) for row in rows)
    sensitivity = detected / max(total, 1)
    precision = matched / max(matched + false, 1)
    patient_sensitivity = [float(row["sensitivity"]) for row in rows if row["total"]]
    latencies = [float(value) for row in rows for value in row["latencies"]]
    return {
        "subjects": len(rows),
        "detected": detected,
        "total": total,
        "event_sensitivity": sensitivity,
        "macro_patient_sensitivity": float(np.mean(patient_sensitivity)),
        "median_patient_sensitivity": float(np.median(patient_sensitivity)),
        "patients_with_any_detection": sum(int(row["detected"] > 0) for row in rows),
        "matched_predictions": matched,
        "false_alarms": false,
        "monitoring_hours": hours,
        "far_per_hour": false / max(hours, 1e-12),
        "event_precision": precision,
        "event_f1": 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12),
        "median_latency_seconds": float(np.median(latencies)) if latencies else None,
    }


def configuration_key(configuration: dict) -> tuple:
    return (
        float(configuration["high_quantile"]),
        int(configuration["persistence"]),
        float(configuration["low_quantile"]),
        int(configuration["reset_seconds"]),
    )


def selection_key(result: dict, configuration: dict) -> tuple:
    """Sensitivity is primary; FAR and precision are hard inner guardrails."""
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
    tie_break = configuration_key(configuration)
    if feasible:
        return (
            0,
            -result["event_sensitivity"],
            -result["event_f1"],
            result["far_per_hour"],
            tie_break,
        )
    return (
        1,
        max(far_violation, precision_violation),
        far_violation + precision_violation,
        -result["event_sensitivity"],
        -result["event_f1"],
        tie_break,
    )


def passes_targets(result: dict) -> bool:
    return bool(
        result["event_sensitivity"] >= TARGET["event_sensitivity_min"]
        and result["far_per_hour"] <= TARGET["far_per_hour_max"]
        and result["event_precision"] >= TARGET["event_precision_min"]
    )
