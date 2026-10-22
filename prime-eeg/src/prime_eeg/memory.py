"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from . import representation as components


@dataclass(frozen=True)
class IncrementalRule:
    high_quantile: float
    persistence: int
    low_quantile: float
    reset_seconds: int


def load_event_positives(source: components.SubjectSource) -> dict[str, np.ndarray]:
    output = {}
    for event in source.manifest["events"]:
        values, names, extras = components.load_features(event["cache_file"])
        if names != source.names:
            raise AssertionError("event schema mismatch")
        labels = np.asarray(extras["labels"], dtype=bool)
        output[event["event_id"]] = components.align(
            values[labels], source.center, source.scale
        )
    return output


def build_incremental_payload(
    source: components.SubjectSource, update_enabled: bool = True
) -> dict:
    fit = np.vstack([
        components.align(values, source.center, source.scale)
        for values in source.fit_raw
    ])
    calibration = source.calibration_aligned
    all_indices = np.arange(len(source.names))
    positives = load_event_positives(source)
    event_order = source.manifest["events"]
    first_id = event_order[0]["event_id"]
    confirmed = [first_id]
    model_cache = {}

    def model_and_thresholds(ids: tuple[str, ...]):
        if ids not in model_cache:
            positive = np.vstack([positives[event_id] for event_id in ids])
            model = components.fit_prototype(fit, positive, all_indices, 50)
            calibration_score = model.scores(calibration)["prototype"]
            model_cache[ids] = (model, calibration_score)
        return model_cache[ids]

    records = []
    for row in source.manifest["monitoring_records"]:
        raw, names, _ = components.load_features(row["cache_file"])
        if names != source.names:
            raise AssertionError("record schema mismatch")
        values = components.align(raw, source.center, source.scale)
        score = np.zeros(len(values), dtype=np.float32)
        calibration_by_window = {
            q: np.zeros(len(values), dtype=np.float32)
            for q in (
                0.5, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995,
                0.997, 0.998, 0.999, 0.9992, 0.9995,
            )
        }
        seconds = int(source.manifest["window_seconds"])
        record_events = sorted(row["events"], key=lambda event: event["end_seconds"])
        start = 0
        for event in record_events:
            if event["event_id"] == first_id:
                continue
            stop = min(len(values), math.ceil(float(event["end_seconds"]) / seconds))
            ids = tuple(confirmed)
            model, normal_score = model_and_thresholds(ids)
            score[start:stop] = model.scores(values[start:stop])["prototype"]
            for quantile in calibration_by_window:
                calibration_by_window[quantile][start:stop] = float(
                    np.quantile(normal_score, quantile)
                )
            # Database annotations simulate independent confirmation. The current
            # complete window is scored before this history update is absorbed.
            if update_enabled:
                confirmed.append(event["event_id"])
            start = stop
        ids = tuple(confirmed)
        model, normal_score = model_and_thresholds(ids)
        score[start:] = model.scores(values[start:])["prototype"]
        for quantile in calibration_by_window:
            calibration_by_window[quantile][start:] = float(
                np.quantile(normal_score, quantile)
            )
        records.append({
            "record": row["record"],
            "events": row["events"],
            "score": score,
            "thresholds": calibration_by_window,
        })
    return {
        "dataset": source.dataset,
        "subject": source.subject,
        "window_seconds": int(source.manifest["window_seconds"]),
        "target_events": [event["event_id"] for event in event_order[1:]],
        "records": records,
    }


def variable_hysteresis(
    score: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    high_count: int,
    reset_count: int,
    valid: np.ndarray,
) -> np.ndarray:
    armed = False
    high_run = 0
    low_run = 0
    output = []
    for index, value in enumerate(score):
        if not valid[index]:
            armed = False
            high_run = 0
            low_run = 0
            continue
        if armed:
            high_run = high_run + 1 if value >= high[index] else 0
            if high_run >= high_count:
                output.append(index)
                armed = False
                high_run = 0
                low_run = 0
        else:
            low_run = low_run + 1 if value <= low[index] else 0
            if low_run >= reset_count:
                armed = True
                high_run = 0
                low_run = 0
    return np.asarray(output, dtype=int)
