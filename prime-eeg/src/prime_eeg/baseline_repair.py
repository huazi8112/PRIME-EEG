"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
import itertools
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from . import baseline_raw as raw
from .io import read_json
RAW_CACHE = None


def positive_support(record: dict, event: dict) -> np.ndarray:
    starts = np.asarray(record["interval_starts"], dtype=float)
    ends = np.asarray(record["interval_ends"], dtype=float)
    return (ends > float(event["start_seconds"])) & (
        starts < float(event["end_seconds"])
    )


def make_repaired_payload(
    subject: str,
    monitoring_hours: float,
    perturb_event: str | None = None,
) -> dict:
    subject_root = RAW_CACHE / subject
    manifest = read_json(subject_root / "manifest.json")
    if not manifest["complete_monitoring_set"]:
        raise ValueError(f"{subject}: incomplete raw-score cache")

    fit = raw.role_values(subject_root, manifest["roles"]["fit"])
    calibration = raw.role_values(subject_root, manifest["roles"]["calibration"])
    center, scale = raw.robust_location_scale(fit)
    calibration_z = (calibration - center) / scale

    first_event = manifest["events"][0]
    first_record = raw.load_score(
        subject_root / "records" / f"{first_event['record']}.npz"
    )
    first_mask = positive_support(first_record, first_event)
    first_positive = (raw.feature_matrix(first_record)[first_mask] - center) / scale
    if not len(first_positive):
        raise ValueError(f"{subject}: no positive initialization score block")
    confirmed_positive_z = [first_positive]
    initial_direction = raw.unit_direction(np.vstack(confirmed_positive_z))
    first_release = max(
        float(first_event["end_seconds"]),
        float(np.max(np.asarray(first_record["interval_ends"])[first_mask])),
    )

    records = []
    update_ledger = []
    target_ids = {event["event_id"] for event in manifest["events"][1:]}
    for row in manifest["monitoring_records"]:
        record = raw.load_score(subject_root / "records" / f"{row['record']}.npz")
        values = raw.feature_matrix(record)
        normalized = (values - center) / scale
        ends = np.asarray(record["interval_ends"], dtype=float)
        static_projection = normalized @ initial_direction
        sequential_projection = np.zeros(len(values), dtype=float)
        sequential_reference = np.zeros(
            (len(values), len(calibration)), dtype=np.float32
        )
        updates = []
        for event in row["events"]:
            if event.get("consumed_initialization_event"):
                continue
            mask = positive_support(record, event)
            positive = normalized[mask]
            if not len(positive):
                update_ledger.append(
                    {
                        "subject": subject,
                        "record": row["record"],
                        "event_id": event["event_id"],
                        "positive_blocks": 0,
                        "release_seconds": float(event["end_seconds"]),
                        "updated": False,
                    }
                )
                continue
            if event["event_id"] == perturb_event:
                positive = positive.copy()
                positive[-1] += np.linspace(11.0, 19.0, positive.shape[1])
            selected_ends = ends[mask]
            release = max(float(event["end_seconds"]), float(np.max(selected_ends)))
            updates.append((release, event, positive, selected_ends))
        updates.sort(key=lambda item: (item[0], item[1]["event_id"]))

        start_index = 0
        for release, event, positive, selected_ends in updates:
            # Include an output exactly at release in the old-model segment.
            stop_index = int(np.searchsorted(ends, release, side="right"))
            direction = raw.unit_direction(np.vstack(confirmed_positive_z))
            sequential_projection[start_index:stop_index] = (
                normalized[start_index:stop_index] @ direction
            )
            sequential_reference[start_index:stop_index] = (
                calibration_z @ direction
            )[None, :]
            confirmed_positive_z.append(positive)
            update_ledger.append(
                {
                    "subject": subject,
                    "record": row["record"],
                    "event_id": event["event_id"],
                    "annotation_start_seconds": float(event["start_seconds"]),
                    "annotation_end_seconds": float(event["end_seconds"]),
                    "positive_support_start_min_seconds": float(
                        np.min(np.asarray(record["interval_starts"])[positive_support(record, event)])
                    ),
                    "positive_support_end_max_seconds": float(np.max(selected_ends)),
                    "positive_blocks": int(len(positive)),
                    "release_seconds": release,
                    "outputs_scored_with_old_direction_through_seconds": (
                        float(ends[stop_index - 1]) if stop_index else None
                    ),
                    "first_output_using_updated_direction_seconds": (
                        float(ends[stop_index]) if stop_index < len(ends) else None
                    ),
                    "updated": True,
                }
            )
            start_index = stop_index
        direction = raw.unit_direction(np.vstack(confirmed_positive_z))
        sequential_projection[start_index:] = normalized[start_index:] @ direction
        sequential_reference[start_index:] = (calibration_z @ direction)[None, :]
        records.append(
            {
                "record": row["record"],
                "events": row["events"],
                "interval_starts": np.asarray(record["interval_starts"], dtype=float),
                "interval_ends": ends,
                "raw": {
                    name: np.asarray(record[name], dtype=float)
                    for name in raw.RAW_FEATURES
                },
                "max_z": raw.max_z(values, center, scale),
                "projection_static": static_projection,
                "projection_sequential": sequential_projection,
                "projection_sequential_calibration": sequential_reference,
            }
        )

    # The first-event adapter must be ready before the first archived eligible
    # raw-score output (which starts at least 60 s after event offset).
    first_monitoring_record = next(
        row for row in records if row["record"] == first_event["record"]
    )
    first_valid_ends = np.asarray(first_monitoring_record["interval_ends"])[
        np.asarray(first_monitoring_record["interval_starts"])
        >= float(first_event["end_seconds"])
    ]
    if len(first_valid_ends) and not first_release <= float(first_valid_ends[0]) + 1e-9:
        raise AssertionError((subject, first_release, first_valid_ends[0]))

    return {
        "dataset": "chbmit",
        "subject": subject,
        "target_events": sorted(target_ids),
        "first_event": first_event,
        "records": records,
        "monitoring_hours": float(monitoring_hours),
        "calibration": {
            **{
                name: np.asarray(calibration[:, index], dtype=float)
                for index, name in enumerate(raw.RAW_FEATURES)
            },
            "max_z": raw.max_z(calibration, center, scale),
            "projection_static": calibration_z @ initial_direction,
        },
        "repair_audit": {
            "first_event_id": first_event["event_id"],
            "first_event_positive_blocks": int(first_mask.sum()),
            "first_event_release_seconds": first_release,
            "updates": update_ledger,
        },
    }
