"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from . import preprocessing as cache_builder, sensitivity as historical, representation as components, grouping as grouped


def record_clock(source, prepared: dict) -> tuple[list[dict], dict, float]:
    seconds = int(source.manifest["window_seconds"])
    rows = []
    events = {}
    for prepared_record in prepared["records"]:
        row = prepared_record["row"]
        start, duration, _ = cache_builder.edf_metadata(
            cache_builder.DATA_ROOT / source.subject / f"{row['record']}.edf"
        )
        cache_origin = start.timestamp() + float(row["time_offset_seconds"])
        count = len(prepared_record["values"])
        clock_row = {
            "record": row["record"],
            "edf_start_iso": start.isoformat(),
            "edf_duration_seconds": duration,
            "cache_origin_epoch_seconds": cache_origin,
            "first_window_start_epoch_seconds": cache_origin,
            "last_window_end_epoch_seconds": cache_origin + count * seconds,
            "count": count,
            "prepared": prepared_record,
        }
        rows.append(clock_row)
        for event in row["events"]:
            events[event["event_id"]] = {
                "record": row["record"],
                "annotation_start_epoch_seconds": cache_origin
                + float(event["start_seconds"]),
                "annotation_end_epoch_seconds": cache_origin
                + float(event["end_seconds"]),
                "consumed_initialization_event": bool(
                    event.get("consumed_initialization_event")
                ),
            }
    first_event = next(value for value in events.values() if value["consumed_initialization_event"])
    first_row = next(row for row in rows if row["record"] == first_event["record"])
    relative_end = first_event["annotation_end_epoch_seconds"] - first_row[
        "cache_origin_epoch_seconds"
    ]
    first_eligible_index = int(math.ceil((relative_end - 1e-9) / seconds))
    task_origin = (
        first_row["cache_origin_epoch_seconds"] + first_eligible_index * seconds
    )
    return rows, events, task_origin


def release_schedule(
    source, events: dict, task_origin: float, mode: str, amount: float
) -> list[dict]:
    output = []
    first_id = source.manifest["events"][0]["event_id"]
    for event in source.manifest["events"][1:]:
        current = events[event["event_id"]]
        end = current["annotation_end_epoch_seconds"]
        if mode == "fixed":
            release = end + amount
        elif mode == "batch":
            cycles = math.ceil((end - task_origin + 1e-9) / amount)
            release = task_origin + max(1, cycles) * amount
        else:
            raise ValueError(mode)
        output.append(
            {
                "event_id": event["event_id"],
                "record": current["record"],
                "annotation_end_epoch_seconds": end,
                "release_epoch_seconds": release,
                "scheduled_delay_seconds": release - end,
            }
        )
    assert first_id not in {row["event_id"] for row in output}
    return sorted(output, key=lambda row: (row["release_epoch_seconds"], row["event_id"]))


def elapsed_payload(prepared: dict, clock_rows: list[dict], schedule: list[dict]) -> tuple[dict, list[dict]]:
    source = prepared["source"]
    seconds = int(source.manifest["window_seconds"])
    first_id = source.manifest["events"][0]["event_id"]
    confirmed = [first_id]
    pointer = 0
    model_cache = {}
    ledger = [dict(row) for row in schedule]

    def model_and_thresholds(ids: tuple[str, ...]):
        if ids not in model_cache:
            positive = np.vstack([prepared["positives"][event_id] for event_id in ids])
            model = components.fit_prototype(
                prepared["fit"], positive, np.arange(len(source.names)), 50
            )
            normal_score = model.scores(prepared["calibration"])["prototype"]
            model_cache[ids] = (model, normal_score)
        return model_cache[ids]

    outputs = []
    first_use = {}
    for clock_row in clock_rows:
        record = clock_row["prepared"]
        values = record["values"]
        count = len(values)
        origin = float(clock_row["cache_origin_epoch_seconds"])
        after_last_start = origin + count * seconds

        while pointer < len(schedule) and schedule[pointer]["release_epoch_seconds"] <= origin + 1e-9:
            event_id = schedule[pointer]["event_id"]
            confirmed.append(event_id)
            first_use.setdefault(event_id, {"record": record["row"]["record"], "window_start_epoch_seconds": origin})
            pointer += 1

        boundaries = {0, count}
        for item in schedule[pointer:]:
            release = float(item["release_epoch_seconds"])
            if release > after_last_start + 1e-9:
                break
            if release > origin + 1e-9:
                index = min(
                    count,
                    max(0, int(math.ceil((release - origin - 1e-9) / seconds))),
                )
                boundaries.add(index)
        ordered = sorted(boundaries)
        scores = np.zeros(count, dtype=np.float32)
        thresholds = {
            quantile: np.zeros(count, dtype=np.float32)
            for quantile in historical.QUANTILES
        }
        for start, stop in zip(ordered[:-1], ordered[1:]):
            window_start = origin + start * seconds
            while pointer < len(schedule) and schedule[pointer]["release_epoch_seconds"] <= window_start + 1e-9:
                event_id = schedule[pointer]["event_id"]
                confirmed.append(event_id)
                first_use.setdefault(
                    event_id,
                    {
                        "record": record["row"]["record"],
                        "window_start_epoch_seconds": window_start,
                    },
                )
                pointer += 1
            model, normal_score = model_and_thresholds(tuple(confirmed))
            scores[start:stop] = model.scores(values[start:stop])["prototype"]
            for quantile in historical.QUANTILES:
                thresholds[quantile][start:stop] = float(
                    np.quantile(normal_score, quantile)
                )
        outputs.append(
            {
                "record": record["row"]["record"],
                "events": record["row"]["events"],
                "score": scores,
                "thresholds": thresholds,
            }
        )

    for row in ledger:
        use = first_use.get(row["event_id"])
        row["used_within_observed_stream"] = use is not None
        row["first_model_use_record"] = None if use is None else use["record"]
        row["first_model_use_window_start_epoch_seconds"] = (
            None if use is None else use["window_start_epoch_seconds"]
        )
        row["quantization_wait_seconds"] = (
            None
            if use is None
            else use["window_start_epoch_seconds"] - row["release_epoch_seconds"]
        )

    return (
        {
            "dataset": source.dataset,
            "subject": source.subject,
            "window_seconds": seconds,
            "target_events": [
                event["event_id"] for event in source.manifest["events"][1:]
            ],
            "records": outputs,
        },
        ledger,
    )


def cluster_bootstrap(rows: list[dict], seed: int, replicates: int = 10000) -> dict:
    groups = grouped.group_rows(rows)
    ordered = [groups[key] for key in sorted(groups)]
    counts = np.asarray(
        [
            [
                sum(row["detected"] for row in values),
                sum(row["total"] for row in values),
                sum(row["false_alarms"] for row in values),
                sum(row["monitoring_hours"] for row in values),
            ]
            for values in ordered
        ],
        dtype=float,
    )
    indices = np.random.default_rng(seed).integers(
        0, len(ordered), size=(replicates, len(ordered))
    )
    pooled = counts[indices].sum(axis=1)
    tp, total, false, hours = pooled.T
    arrays = {
        "event_sensitivity": tp / total,
        "event_precision": np.divide(
            tp, tp + false, out=np.zeros(len(tp)), where=(tp + false) > 0
        ),
        "event_f1": 2 * tp / (tp + total + false),
        "far_per_hour": false / hours,
        "macro_independent_subject_sensitivity": np.mean(
            (counts[:, 0] / counts[:, 1])[indices], axis=1
        ),
    }
    latency = [
        np.asarray([value for row in values for value in row["latencies"]], dtype=float)
        for values in ordered
    ]
    medians = []
    for sample in indices:
        values = np.concatenate([latency[index] for index in sample])
        medians.append(float(np.median(values)) if len(values) else np.nan)
    arrays["median_latency_seconds"] = np.asarray(medians)
    return {
        "replicates": replicates,
        "seed": seed,
        "unit": "independent subject with repeated-cluster multiplicity",
        "ci95": {
            key: [float(value) for value in np.nanquantile(array, [0.025, 0.975])]
            for key, array in arrays.items()
        },
    }
