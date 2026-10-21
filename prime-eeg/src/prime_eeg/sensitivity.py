"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from . import representation as components, memory as incremental
from .evaluation import aggregate_independent, causal_alarm_events, independent_subject
QUANTILES = (0.97, 0.98, 0.99, 0.995, 0.997, 0.998, 0.999, 0.9992)


def predicted_interval(prediction: dict, seconds: int) -> tuple[float, float]:
    """Causal displayed interval: it starts when the first alarm is available."""
    return (
        float(prediction["first_available"]),
        float(prediction["last_available"] + seconds),
    )


def union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]


def total_duration(intervals: list[tuple[float, float]]) -> float:
    return float(sum(end - start for start, end in union(intervals)))


def intersection_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    left, right = union(left), union(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return float(total)


def match_prediction(
    prediction: dict,
    reference: dict,
    seconds: int,
    mode: str,
    preictal_seconds: float,
    postictal_seconds: float,
) -> bool:
    start = float(reference["start_seconds"])
    end = float(reference["end_seconds"])
    if mode == "alarm_timestamp":
        return any(
            start - preictal_seconds <= value <= end + postictal_seconds
            for value in prediction["available_times"]
        )
    if mode == "strict_interval_overlap":
        pred_start, pred_end = predicted_interval(prediction, seconds)
        return pred_start < end and pred_end > start
    raise ValueError(mode)


def evaluate(
    payload: dict,
    current_rule,
    *,
    high_gap_delta: float = 0.0,
    preictal_seconds: float = 30.0,
    postictal_seconds: float = 60.0,
    mode: str = "alarm_timestamp",
) -> dict:
    seconds = int(payload["window_seconds"])
    target_ids = set(payload["target_events"])
    detected = {event_id: False for event_id in target_ids}
    latencies = {event_id: None for event_id in target_ids}
    false = matched = monitoring_windows = negative = 0
    predicted_seconds = reference_seconds = overlap_seconds = 0.0

    for row in payload["records"]:
        score = np.asarray(row["score"])
        starts = np.arange(len(score), dtype=float) * seconds
        valid = np.asarray(row.get("valid", np.ones(len(score), dtype=bool)), dtype=bool).copy()
        events = sorted(row["events"], key=lambda value: value["start_seconds"])
        for event in events:
            if event.get("consumed_initialization_event"):
                valid &= starts >= float(event["end_seconds"])
        high = np.asarray(row["thresholds"][current_rule.high_quantile])
        low = np.asarray(row["thresholds"][current_rule.low_quantile])
        adjusted_high = high + high_gap_delta * (high - low)
        indices = incremental.variable_hysteresis(
            score,
            adjusted_high,
            low,
            current_rule.persistence,
            max(1, math.ceil(current_rule.reset_seconds / seconds)),
            valid,
        )
        monitoring_windows += int(np.sum(valid))
        predictions = causal_alarm_events(indices, seconds)
        references = [event for event in events if event["event_id"] in target_ids]

        prediction_intervals = [predicted_interval(value, seconds) for value in predictions]
        reference_intervals = [
            (float(value["start_seconds"]), float(value["end_seconds"]))
            for value in references
        ]
        predicted_seconds += total_duration(prediction_intervals)
        reference_seconds += total_duration(reference_intervals)
        overlap_seconds += intersection_duration(prediction_intervals, reference_intervals)

        unmatched = {event["event_id"] for event in references}
        for prediction in predictions:
            matches = [
                event
                for event in references
                if event["event_id"] in unmatched
                and match_prediction(
                    prediction,
                    event,
                    seconds,
                    mode,
                    preictal_seconds,
                    postictal_seconds,
                )
            ]
            if not matches:
                false += 1
                continue
            event = min(matches, key=lambda value: value["start_seconds"])
            event_id = event["event_id"]
            unmatched.remove(event_id)
            detected[event_id] = True
            if mode == "alarm_timestamp":
                lower = float(event["start_seconds"]) - preictal_seconds
                upper = float(event["end_seconds"]) + postictal_seconds
                available = min(
                    value
                    for value in prediction["available_times"]
                    if lower <= value <= upper
                )
            else:
                available = float(prediction["first_available"])
            latency = available - float(event["start_seconds"])
            latencies[event_id] = latency
            negative += int(latency < 0)
            matched += 1

    count = sum(int(value) for value in detected.values())
    hours = monitoring_windows * seconds / 3600.0
    return {
        "dataset": payload["dataset"],
        "subject": payload["subject"],
        "independent_subject": independent_subject(payload["subject"]),
        "detected": count,
        "total": len(detected),
        "sensitivity": count / max(len(detected), 1),
        "matched_predictions": matched,
        "false_alarms": false,
        "monitoring_hours": hours,
        "far_per_hour": false / max(hours, 1e-12),
        "precision": matched / max(matched + false, 1),
        "latencies": [value for value in latencies.values() if value is not None],
        "negative_latency_detections": negative,
        "detected_event_ids": sorted(key for key, value in detected.items() if value),
        "target_event_ids": sorted(detected),
        "predicted_event_seconds": predicted_seconds,
        "reference_event_seconds": reference_seconds,
        "overlap_event_seconds": overlap_seconds,
    }


def aggregate(rows: list[dict]) -> dict:
    result = aggregate_independent(rows)
    predicted = sum(float(row["predicted_event_seconds"]) for row in rows)
    reference = sum(float(row["reference_event_seconds"]) for row in rows)
    overlap = sum(float(row["overlap_event_seconds"]) for row in rows)
    result.update(
        {
            "false_alarms_per_24h": 24.0 * result["far_per_hour"],
            "predicted_event_seconds": predicted,
            "reference_event_seconds": reference,
            "overlap_event_seconds": overlap,
            "time_based_ppv": overlap / max(predicted, 1e-12),
            "time_based_recall": overlap / max(reference, 1e-12),
        }
    )
    return result


def prepare_source(source: components.SubjectSource) -> dict:
    fit = np.vstack(
        [components.align(values, source.center, source.scale) for values in source.fit_raw]
    )
    positives = incremental.load_event_positives(source)
    records = []
    base = 0.0
    for row in source.manifest["monitoring_records"]:
        raw, names, _ = components.load_features(row["cache_file"])
        if names != source.names:
            raise AssertionError("strict cache schema mismatch")
        values = components.align(raw, source.center, source.scale)
        records.append({"row": row, "values": values, "base": base})
        base += len(values) * int(source.manifest["window_seconds"])
    event_global = {}
    for record in records:
        for event in record["row"]["events"]:
            event_global[event["event_id"]] = {
                "start": record["base"] + float(event["start_seconds"]),
                "end": record["base"] + float(event["end_seconds"]),
            }
    return {
        "source": source,
        "fit": fit,
        "calibration": source.calibration_aligned,
        "positives": positives,
        "records": records,
        "event_global": event_global,
    }
