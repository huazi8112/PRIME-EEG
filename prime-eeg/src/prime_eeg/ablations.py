"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from . import representation as components, memory as incremental, persistence as scoring
from .evaluation import causal_alarm_events, overlaps_available_alarm, independent_subject


def available_indices(source: components.SubjectSource, view: str) -> np.ndarray:
    if view == "all":
        return np.arange(len(source.names))
    if view == "physiology":
        return np.flatnonzero(
            [
                name.startswith("spectral_") or name.startswith("temporal_")
                for name in source.names
            ]
        )
    if view == "geometry":
        return np.flatnonzero(
            [name.startswith("geometry_") for name in source.names]
        )
    raise ValueError(view)


def build_payload(
    source: components.SubjectSource,
    view: str = "all",
    update_enabled: bool = True,
) -> dict:
    """Rebuild a v48 trace while removing exactly one frozen component."""
    fit = np.vstack(
        [components.align(values, source.center, source.scale) for values in source.fit_raw]
    )
    calibration = source.calibration_aligned
    available = available_indices(source, view)
    positives = incremental.load_event_positives(source)
    event_order = source.manifest["events"]
    first_id = event_order[0]["event_id"]
    confirmed = [first_id]
    model_cache: dict[tuple[str, ...], tuple] = {}

    def model_and_thresholds(ids: tuple[str, ...]):
        if ids not in model_cache:
            positive = np.vstack([positives[event_id] for event_id in ids])
            model = components.fit_prototype(fit, positive, available, 50)
            normal_score = model.scores(calibration)["prototype"]
            model_cache[ids] = (model, normal_score)
        return model_cache[ids]

    records = []
    quantiles = (
        0.97,
        0.98,
        0.99,
        0.995,
        0.997,
        0.998,
        0.999,
        0.9992,
    )
    for row in source.manifest["monitoring_records"]:
        raw, names, _ = components.load_features(row["cache_file"])
        if names != source.names:
            raise AssertionError("strict-cache record schema mismatch")
        values = components.align(raw, source.center, source.scale)
        score = np.zeros(len(values), dtype=np.float32)
        thresholds = {
            quantile: np.zeros(len(values), dtype=np.float32)
            for quantile in quantiles
        }
        seconds = int(source.manifest["window_seconds"])
        start = 0
        for event in sorted(row["events"], key=lambda value: value["end_seconds"]):
            if event["event_id"] == first_id:
                continue
            stop = min(len(values), math.ceil(float(event["end_seconds"]) / seconds))
            model, normal_score = model_and_thresholds(tuple(confirmed))
            score[start:stop] = model.scores(values[start:stop])["prototype"]
            for quantile in quantiles:
                thresholds[quantile][start:stop] = float(
                    np.quantile(normal_score, quantile)
                )
            if update_enabled:
                confirmed.append(event["event_id"])
            start = stop
        model, normal_score = model_and_thresholds(tuple(confirmed))
        score[start:] = model.scores(values[start:])["prototype"]
        for quantile in quantiles:
            thresholds[quantile][start:] = float(
                np.quantile(normal_score, quantile)
            )
        records.append(
            {
                "record": row["record"],
                "events": row["events"],
                "score": score,
                "thresholds": thresholds,
            }
        )
    return {
        "dataset": source.dataset,
        "subject": source.subject,
        "window_seconds": int(source.manifest["window_seconds"]),
        "target_events": [event["event_id"] for event in event_order[1:]],
        "records": records,
        "feature_view": view,
        "feature_count": int(len(available)),
        "update_enabled": update_enabled,
    }


def no_hysteresis_evaluate(payload: dict, current_rule: incremental.IncrementalRule) -> dict:
    """Remove low-threshold re-arming but retain the locked high persistence."""
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
        starts = np.arange(len(score), dtype=float) * seconds
        valid = np.ones(len(score), dtype=bool)
        events = sorted(row["events"], key=lambda event: event["start_seconds"])
        for event in events:
            if event.get("consumed_initialization_event"):
                valid &= starts >= float(event["end_seconds"])
        candidate = scoring.persistent(
            (score >= row["thresholds"][current_rule.high_quantile]) & valid,
            current_rule.persistence,
        )
        monitoring_windows += int(np.sum(valid))
        references = [event for event in events if event["event_id"] in target_ids]
        unmatched = {event["event_id"] for event in references}
        for prediction in causal_alarm_events(np.flatnonzero(candidate), seconds):
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
            allowed_start = float(event["start_seconds"]) - 30.0
            allowed_end = float(event["end_seconds"]) + 60.0
            available = min(
                value
                for value in prediction["available_times"]
                if allowed_start <= value <= allowed_end
            )
            latency = available - float(event["start_seconds"])
            latencies[event_id] = latency
            negative_delay_count += int(latency < 0)
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
        "negative_latency_detections": negative_delay_count,
        "detected_event_ids": sorted(
            event_id for event_id, value in detected.items() if value
        ),
        "target_event_ids": sorted(detected),
    }
