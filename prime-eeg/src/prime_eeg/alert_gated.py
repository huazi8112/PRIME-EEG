"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from . import representation as components, memory as incremental
from .evaluation import independent_subject, causal_alarm_events, overlaps_available_alarm


def alert_gated_subject(
    source: components.SubjectSource,
    current_rule: incremental.IncrementalRule,
) -> dict:
    fit = np.vstack(
        [components.align(values, source.center, source.scale) for values in source.fit_raw]
    )
    calibration = source.calibration_aligned
    available = np.arange(len(source.names))
    positives = incremental.load_event_positives(source)
    ordered_events = source.manifest["events"]
    first_id = ordered_events[0]["event_id"]
    target_ids = {event["event_id"] for event in ordered_events[1:]}
    confirmed = [first_id]
    model_cache: dict[tuple[str, ...], tuple] = {}

    def model_and_thresholds(ids: tuple[str, ...]):
        if ids not in model_cache:
            positive = np.vstack([positives[event_id] for event_id in ids])
            model = components.fit_prototype(fit, positive, available, 50)
            normal_score = model.scores(calibration)["prototype"]
            model_cache[ids] = (
                model,
                float(np.quantile(normal_score, current_rule.high_quantile)),
                float(np.quantile(normal_score, current_rule.low_quantile)),
            )
        return model_cache[ids]

    detected = {event_id: False for event_id in target_ids}
    latencies = {event_id: None for event_id in target_ids}
    false = 0
    matched = 0
    monitoring_windows = 0
    negative_delay_count = 0
    update_log = []
    seconds = int(source.manifest["window_seconds"])
    reset_count = max(1, math.ceil(current_rule.reset_seconds / seconds))

    for row in source.manifest["monitoring_records"]:
        raw, names, _ = components.load_features(row["cache_file"])
        if names != source.names:
            raise AssertionError("strict-cache record schema mismatch")
        values = components.align(raw, source.center, source.scale)
        starts = np.arange(len(values), dtype=float) * seconds
        valid = np.ones(len(values), dtype=bool)
        events = sorted(row["events"], key=lambda event: event["start_seconds"])
        for event in events:
            if event.get("consumed_initialization_event"):
                valid &= starts >= float(event["end_seconds"])
        monitoring_windows += int(np.sum(valid))

        armed = False
        high_run = 0
        low_run = 0
        alarms: list[int] = []
        unused_for_update: set[int] = set()
        start = 0

        def process(stop: int) -> None:
            nonlocal armed, high_run, low_run, start
            stop = min(stop, len(values))
            if stop <= start:
                return
            model, high, low = model_and_thresholds(tuple(confirmed))
            scores = model.scores(values[start:stop])["prototype"]
            for offset, value in enumerate(scores):
                index = start + offset
                if not valid[index]:
                    armed = False
                    high_run = 0
                    low_run = 0
                    continue
                if armed:
                    high_run = high_run + 1 if value >= high else 0
                    if high_run >= current_rule.persistence:
                        alarms.append(index)
                        unused_for_update.add(index)
                        armed = False
                        high_run = 0
                        low_run = 0
                else:
                    low_run = low_run + 1 if value <= low else 0
                    if low_run >= reset_count:
                        armed = True
                        high_run = 0
                        low_run = 0
            start = stop

        for event in sorted(row["events"], key=lambda event: event["end_seconds"]):
            if event["event_id"] == first_id:
                continue
            # This includes only windows physically available by the first window
            # end at or after clinical event end. The event prototype itself still
            # contains ictal windows only.
            stop = math.ceil(float(event["end_seconds"]) / seconds)
            process(stop)
            allowed_start = float(event["start_seconds"]) - 30.0
            allowed_end = float(event["end_seconds"]) + 60.0
            eligible = [
                index
                for index in sorted(unused_for_update)
                if allowed_start <= float((index + 1) * seconds) <= allowed_end
            ]
            if eligible:
                used = eligible[0]
                unused_for_update.remove(used)
                confirmed.append(event["event_id"])
                update_log.append(
                    {
                        "event_id": event["event_id"],
                        "updated": True,
                        "alarm_available_seconds": float((used + 1) * seconds),
                        "update_available_after_seconds": max(
                            float(event["end_seconds"]), float((used + 1) * seconds)
                        ),
                    }
                )
            else:
                update_log.append(
                    {
                        "event_id": event["event_id"],
                        "updated": False,
                        "reason": "no causally available detector alarm matched this clinically confirmed event",
                    }
                )
        process(len(values))

        references = [event for event in events if event["event_id"] in target_ids]
        unmatched = {event["event_id"] for event in references}
        for prediction in causal_alarm_events(np.asarray(alarms, dtype=int), seconds):
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
            available_time = min(
                value
                for value in prediction["available_times"]
                if allowed_start <= value <= allowed_end
            )
            latency = available_time - float(event["start_seconds"])
            latencies[event_id] = latency
            negative_delay_count += int(latency < 0)
            matched += 1

    count = sum(int(value) for value in detected.values())
    hours = monitoring_windows * seconds / 3600.0
    return {
        "dataset": "chbmit",
        "subject": source.subject,
        "independent_subject": independent_subject(source.subject),
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
        "confirmed_updates": sum(int(value["updated"]) for value in update_log),
        "update_log": update_log,
    }
