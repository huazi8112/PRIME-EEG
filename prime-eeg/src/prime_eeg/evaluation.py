"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from . import memory as incremental
from .metrics import aggregate
PREICTAL_TOLERANCE_SECONDS = 30.0
POSTICTAL_TOLERANCE_SECONDS = 60.0
MERGE_GAP_SECONDS = 90.0
MAXIMUM_EVENT_SECONDS = 300.0


def independent_subject(subject: str) -> str:
    """CHB-MIT documents chb01 and chb21 as recordings of one person."""
    return "chb01+chb21" if subject in {"chb01", "chb21"} else subject


def causal_alarm_events(indices: np.ndarray, seconds: int) -> list[dict]:
    """Cluster alarms using their first physically available time (window end)."""
    events: list[dict] = []
    for index in np.asarray(indices, dtype=int):
        available = float((index + 1) * seconds)
        if (
            events
            and available - events[-1]["last_available"] <= MERGE_GAP_SECONDS
            and available - events[-1]["first_available"] <= MAXIMUM_EVENT_SECONDS
        ):
            events[-1]["last_available"] = available
            events[-1]["alarm_count"] += 1
            events[-1]["available_times"].append(available)
        else:
            events.append(
                {
                    "first_available": available,
                    "last_available": available,
                    "alarm_count": 1,
                    "available_times": [available],
                }
            )
    return events


def overlaps_available_alarm(prediction: dict, reference: dict) -> bool:
    allowed_start = float(reference["start_seconds"]) - PREICTAL_TOLERANCE_SECONDS
    allowed_end = float(reference["end_seconds"]) + POSTICTAL_TOLERANCE_SECONDS
    return any(
        allowed_start <= available <= allowed_end
        for available in prediction["available_times"]
    )


def evaluate_timing_corrected(payload: dict, rule: incremental.IncrementalRule) -> dict:
    """Match actual window-end alarm points with strict initialization masks.

    Signal causality is established by the separate strict-cache builder, not by
    this event scorer. Conditional latency is calculated only for detected events.
    """
    seconds = int(payload["window_seconds"])
    target_ids = set(payload["target_events"])
    detected = {event_id: False for event_id in target_ids}
    latencies = {event_id: None for event_id in target_ids}
    false = 0
    matched = 0
    monitoring_windows = 0
    negative_delay_count = 0
    for row in payload["records"]:
        score = np.asarray(row["score"])
        window_starts = np.arange(len(score), dtype=float) * seconds
        valid = np.ones(len(score), dtype=bool)
        events = sorted(row["events"], key=lambda event: event["start_seconds"])
        for event in events:
            if event.get("consumed_initialization_event"):
                # A scored window must lie completely after initialization.
                valid &= window_starts >= float(event["end_seconds"])
        indices = incremental.variable_hysteresis(
            score,
            row["thresholds"][rule.high_quantile],
            row["thresholds"][rule.low_quantile],
            rule.persistence,
            max(1, math.ceil(rule.reset_seconds / seconds)),
            valid,
        )
        monitoring_windows += int(np.sum(valid))
        references = [event for event in events if event["event_id"] in target_ids]
        unmatched = {event["event_id"] for event in references}
        for prediction in causal_alarm_events(indices, seconds):
            matches = [
                event
                for event in references
                if event["event_id"] in unmatched
                and overlaps_available_alarm(prediction, event)
            ]
            if not matches:
                false += 1
                continue
            event = min(matches, key=lambda value: value["start_seconds"])
            event_id = event["event_id"]
            unmatched.remove(event_id)
            detected[event_id] = True
            allowed_start = (
                float(event["start_seconds"]) - PREICTAL_TOLERANCE_SECONDS
            )
            allowed_end = float(event["end_seconds"]) + POSTICTAL_TOLERANCE_SECONDS
            matched_available = min(
                available
                for available in prediction["available_times"]
                if allowed_start <= available <= allowed_end
            )
            latency = matched_available - float(event["start_seconds"])
            latencies[event_id] = latency
            negative_delay_count += int(latency < 0)
            matched += 1
    count = sum(int(value) for value in detected.values())
    total = len(detected)
    hours = monitoring_windows * seconds / 3600.0
    return {
        "dataset": payload["dataset"],
        "subject": payload["subject"],
        "independent_subject": independent_subject(payload["subject"]),
        "detected": count,
        "total": total,
        "sensitivity": count / max(total, 1),
        "matched_predictions": matched,
        "false_alarms": false,
        "monitoring_hours": hours,
        "far_per_hour": false / max(hours, 1e-12),
        "precision": matched / max(matched + false, 1),
        "latencies": [value for value in latencies.values() if value is not None],
        "negative_latency_detections": negative_delay_count,
        "detected_event_ids": sorted(key for key, value in detected.items() if value),
        "target_event_ids": sorted(detected),
    }


def aggregate_independent(rows: list[dict]) -> dict:
    result = aggregate(rows)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(independent_subject(row["subject"]), []).append(row)
    group_sensitivities = []
    for group_rows in groups.values():
        group_total = sum(int(row["total"]) for row in group_rows)
        if group_total:
            group_sensitivities.append(
                sum(int(row["detected"]) for row in group_rows) / group_total
            )
    result["cases"] = len(rows)
    result["independent_subjects"] = len(groups)
    result["subjects"] = len(groups)
    result["macro_independent_subject_sensitivity"] = float(
        np.mean(group_sensitivities)
    )
    result["negative_latency_detections"] = sum(
        int(row["negative_latency_detections"]) for row in rows
    )
    return result


def rule(configuration: dict) -> incremental.IncrementalRule:
    return incremental.IncrementalRule(
        high_quantile=float(configuration["high_quantile"]),
        persistence=int(configuration["persistence"]),
        low_quantile=float(configuration["low_quantile"]),
        reset_seconds=int(configuration["reset_seconds"]),
    )
