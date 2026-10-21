"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
import itertools
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
CACHE = None
TARGET = {'event_sensitivity_min': 0.75, 'far_per_hour_max': 0.7, 'event_precision_min': 0.25}


@dataclass(frozen=True)
class Rule:
    threshold: float
    minimum_event_seconds: float


def load_record(path: Path) -> dict:
    with np.load(path) as archive:
        output = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key != "metadata_json"
        }
        output.update(json.loads(str(archive["metadata_json"])))
    return output


def threshold_key(threshold: float) -> str:
    return f"t{int(round(100 * threshold)):03d}"


def make_payload(subject: str, monitoring_hours: float) -> dict:
    root = CACHE / subject
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["complete_monitoring_set"]:
        raise ValueError(f"{subject}: incomplete official cache")
    target_ids = {event["event_id"] for event in manifest["events"][1:]}
    records = []
    for row in manifest["monitoring_records"]:
        records.append(
            {
                "record": row["record"],
                "events": row["events"],
                "score": load_record(root / "records" / f"{row['record']}.npz"),
            }
        )
    return {
        "subject": subject,
        "target_events": sorted(target_ids),
        "records": records,
        "monitoring_hours": monitoring_hours,
    }


def selection_key(result: dict, rule: Rule) -> tuple:
    far_violation = max(
        0.0, (result["far_per_hour"] - TARGET["far_per_hour_max"])
        / TARGET["far_per_hour_max"],
    )
    precision_violation = max(
        0.0, (TARGET["event_precision_min"] - result["event_precision"])
        / TARGET["event_precision_min"],
    )
    feasible = far_violation == 0.0 and precision_violation == 0.0
    key = (rule.threshold, rule.minimum_event_seconds)
    if feasible:
        return (
            0, -result["event_sensitivity"], -result["event_f1"],
            result["far_per_hour"], key,
        )
    return (
        1, max(far_violation, precision_violation), far_violation + precision_violation,
        -result["event_sensitivity"], -result["event_f1"], key,
    )
