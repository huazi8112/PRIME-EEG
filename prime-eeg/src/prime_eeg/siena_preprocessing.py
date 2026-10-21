"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import time
from datetime import datetime, timedelta
import pyedflib
from scipy import signal
from .features import extract_fast_features
from .siena_common import COMMON_CHANNELS, MINIMUM_PREFIX_SECONDS, PREICTAL_EXCLUSION_SECONDS, annotation_path, branch_seconds_v35, edf_order, initialization_plan, parse_all_events, parse_first_event_only, read_bipolar_interval, subject_directories
from .io import write_json, verify_adapter as verify_freeze
DATA_ROOT = None
CACHE_ROOT = None
TARGET_FREQUENCY = 128
EXPECTED_SOURCE_FREQUENCY = 512
DECIMATION = 4
RESET_WARMUP_SECONDS = 30.0
CHUNK_SECONDS = 900.0
CONTIGUITY_TOLERANCE_SECONDS = 1.0 / EXPECTED_SOURCE_FREQUENCY


def filter_design(source_frequency: float) -> np.ndarray:
    return signal.butter(
        4,
        [0.5, 45.0],
        btype="bandpass",
        fs=source_frequency,
        output="sos",
    )


def causal_filter_downsample(
    raw: np.ndarray,
    sos: np.ndarray,
    state: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=float)
    if state is None:
        state = np.zeros((len(sos), raw.shape[0], 2), dtype=float)
    filtered, final_state = signal.sosfilt(sos, raw, axis=1, zi=state)
    return filtered[:, ::DECIMATION].astype(np.float32), final_state


def edf_metadata(path: Path) -> tuple[object, float, float]:
    reader = pyedflib.EdfReader(str(path))
    try:
        frequencies = sorted({float(value) for value in reader.getSampleFrequencies()})
        if len(frequencies) != 1:
            raise ValueError(f"mixed sample rates in {path}: {frequencies}")
        return reader.getStartdatetime(), float(reader.getFileDuration()), frequencies[0]
    finally:
        reader.close()


def is_contiguous(previous_end: object | None, current_start: object) -> bool:
    return previous_end is not None and abs(
        (current_start - previous_end).total_seconds()
    ) <= CONTIGUITY_TOLERANCE_SECONDS


def future_perturbation_test(raw: np.ndarray, window_seconds: int) -> dict:
    count = min(raw.shape[1], 5 * 60 * EXPECTED_SOURCE_FREQUENCY)
    sample = np.asarray(raw[:, :count], dtype=float)
    boundary = count // 2
    perturbed = sample.copy()
    rng = np.random.default_rng(4851 + window_seconds)
    amplitude = np.maximum(np.std(sample, axis=1, keepdims=True), 1.0)
    perturbed[:, boundary:] += rng.normal(
        0.0, 20.0, size=perturbed[:, boundary:].shape
    ) * amplitude
    sos = filter_design(EXPECTED_SOURCE_FREQUENCY)
    left, _ = causal_filter_downsample(sample, sos, None)
    right, _ = causal_filter_downsample(perturbed, sos, None)
    filtered_prefix = boundary // DECIMATION
    filter_equal = np.array_equal(left[:, :filtered_prefix], right[:, :filtered_prefix])
    left_features, left_names = extract_fast_features(
        left, TARGET_FREQUENCY, window_seconds, COMMON_CHANNELS
    )
    right_features, right_names = extract_fast_features(
        right, TARGET_FREQUENCY, window_seconds, COMMON_CHANNELS
    )
    complete = int((boundary / EXPECTED_SOURCE_FREQUENCY) // window_seconds)
    feature_equal = np.array_equal(left_features[:complete], right_features[:complete])
    passed = filter_equal and feature_equal and left_names == right_names
    return {
        "status": "PASS" if passed else "FAIL",
        "future_perturbation_sample": boundary,
        "unchanged_filtered_prefix_samples": filtered_prefix,
        "unchanged_complete_feature_windows": complete,
        "filtered_prefix_exactly_equal": bool(filter_equal),
        "feature_prefix_exactly_equal": bool(feature_equal),
    }


def extract_strict_record(
    path: Path,
    window_seconds: int,
    state: np.ndarray | None,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    _, duration, source_frequency = edf_metadata(path)
    if not math.isclose(source_frequency, EXPECTED_SOURCE_FREQUENCY):
        raise ValueError(
            f"{path} has {source_frequency} Hz, expected {EXPECTED_SOURCE_FREQUENCY} Hz"
        )
    sos = filter_design(source_frequency)
    blocks = []
    names: list[str] | None = None
    position = 0.0
    audit_raw = None
    while position < duration - 1e-9:
        stop = min(duration, position + CHUNK_SECONDS)
        raw, actual_frequency = read_bipolar_interval(path, position, stop)
        if not math.isclose(actual_frequency, source_frequency):
            raise AssertionError("sample rate changed inside one Siena EDF")
        if audit_raw is None:
            audit_raw = raw[:, : min(raw.shape[1], 5 * 60 * EXPECTED_SOURCE_FREQUENCY)]
        filtered, state = causal_filter_downsample(raw, sos, state)
        samples_per_window = window_seconds * TARGET_FREQUENCY
        usable = (filtered.shape[1] // samples_per_window) * samples_per_window
        if usable:
            current, current_names = extract_fast_features(
                filtered[:, :usable], TARGET_FREQUENCY, window_seconds, COMMON_CHANNELS
            )
            names = current_names if names is None else names
            if current_names != names:
                raise AssertionError(f"feature schema changed inside {path.name}")
            blocks.append(current)
        position = stop
    if not blocks or names is None or audit_raw is None or state is None:
        raise ValueError(f"no complete strict-causal windows in {path}")
    return np.vstack(blocks).astype(np.float32), names, state, audit_raw


def save_record(
    path: Path,
    features: np.ndarray,
    names: list[str],
    time_offset_seconds: float,
    final_state: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(features, dtype=np.float32),
        names=np.asarray(names, dtype=str),
        time_offset_seconds=np.asarray(time_offset_seconds, dtype=np.float64),
        filter_state=np.asarray(final_state, dtype=np.float64),
    )


def build_subject(subject: str, force: bool = False) -> dict:
    started = time.perf_counter()
    output = CACHE_ROOT / subject
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "subject": subject,
            "eligible": True,
            "cached": True,
            "window_seconds": manifest["window_seconds"],
            "events": len(manifest["events"]),
            "post_first_targets": len(manifest["events"]) - 1,
            "monitoring_hours": sum(row["hours"] for row in manifest["monitoring_records"]),
            "future_perturbation_invariance": manifest["future_perturbation_invariance"]["status"],
        }
    subject_dir = DATA_ROOT / subject
    annotation = annotation_path(subject_dir)
    first_inspected = parse_first_event_only(annotation)
    plan = initialization_plan(subject_dir, first_inspected)
    if not plan["eligible_normal_initialization"]:
        raise ValueError(
            "insufficient patient-normal initialization: "
            f"{plan['usable_prefix_seconds']:.0f}s; minimum {MINIMUM_PREFIX_SECONDS:.0f}s"
        )
    events = parse_all_events(annotation)
    if len(events) < 2:
        raise ValueError("fewer than two annotated seizures")
    first = events[0]
    if (
        first.file_name != first_inspected.file_name
        or first.start_seconds != first_inspected.start_seconds
        or first.end_seconds != first_inspected.end_seconds
    ):
        raise AssertionError("full parser changed the pre-freeze first event")
    order = edf_order(subject_dir)
    window_seconds = branch_seconds_v35(first.end_seconds - first.start_seconds)
    record_dir = output / "records"
    event_dir = output / "events"
    normal_dir = output / "normals"
    for directory in (record_dir, event_dir, normal_dir):
        directory.mkdir(parents=True, exist_ok=True)

    previous_end = None
    state = None
    schema = None
    record_rows = []
    record_lookup = {}
    invariance = None
    state_resets = 0
    state_carries = 0
    for position, file_name in enumerate(order):
        path = subject_dir / file_name
        start_datetime, duration, frequency = edf_metadata(path)
        if not math.isclose(frequency, EXPECTED_SOURCE_FREQUENCY):
            raise ValueError(f"unexpected Siena sample rate: {frequency}")
        contiguous = is_contiguous(previous_end, start_datetime)
        if contiguous:
            state_carries += 1
        else:
            state = None
            state_resets += 1
        print(
            f"v48 Siena causal {subject} {position + 1}/{len(order)} {Path(file_name).stem} "
            f"{'carry' if contiguous else 'reset'}",
            flush=True,
        )
        features, names, state, audit_raw = extract_strict_record(
            path, window_seconds, state
        )
        drop_windows = 0 if contiguous else math.ceil(RESET_WARMUP_SECONDS / window_seconds)
        time_offset = float(drop_windows * window_seconds)
        features = features[drop_windows:]
        cache_path = record_dir / f"{Path(file_name).stem}.npz"
        save_record(cache_path, features, names, time_offset, state)
        if invariance is None:
            invariance = future_perturbation_test(audit_raw, window_seconds)
        schema = names if schema is None else schema
        if names != schema:
            raise AssertionError(f"feature schema changed in {subject}")
        row = {
            "record": Path(file_name).stem,
            "file_name": file_name,
            "record_position": position,
            "windows": int(len(features)),
            "hours": len(features) * window_seconds / 3600.0,
            "time_offset_seconds": time_offset,
            "filter_state_action": "carry" if contiguous else "reset_then_discard_warmup",
            "cache_file": str(cache_path.resolve()),
        }
        record_rows.append(row)
        record_lookup[file_name] = row
        previous_end = start_datetime + timedelta(seconds=duration)
    assert schema is not None and invariance is not None

    events = sorted(events, key=lambda value: (order.index(value.file_name), value.start_seconds))
    event_rows = []
    events_by_file: dict[str, list[dict]] = {}
    for event in events:
        record_row = record_lookup[event.file_name]
        with np.load(record_row["cache_file"]) as archive:
            features = np.asarray(archive["features"], dtype=np.float32)
            names = [str(value) for value in archive["names"]]
        offset = float(record_row["time_offset_seconds"])
        centers = offset + (np.arange(len(features)) + 0.5) * window_seconds
        labels = (centers >= event.start_seconds) & (centers < event.end_seconds)
        if not np.any(labels):
            labels[int(np.argmin(np.abs(centers - 0.5 * (event.start_seconds + event.end_seconds))))] = True
        event_path = event_dir / f"{event.event_id}.npz"
        np.savez_compressed(
            event_path,
            features=features[labels],
            names=np.asarray(names, dtype=str),
            labels=np.ones(int(np.sum(labels)), dtype=np.int8),
            seconds=(centers[labels] - event.start_seconds).astype(np.float64),
        )
        event_row = {
            "event_id": event.event_id,
            "sequence": event.sequence,
            "file_name": event.file_name,
            "record": record_row["record"],
            "record_position": record_row["record_position"],
            "start_seconds": float(event.start_seconds),
            "end_seconds": float(event.end_seconds),
            "duration_seconds": float(event.end_seconds - event.start_seconds),
            "windows": int(np.sum(labels)),
            "ictal_windows": int(np.sum(labels)),
            "cache_file": str(event_path.resolve()),
        }
        event_rows.append(event_row)
        events_by_file.setdefault(event.file_name, []).append(event_row)

    normal_rows = []
    if plan["mode"] == "all_earlier_seizure_free_edfs":
        for role in ("fit", "calibration"):
            for role_index, file_name in enumerate(plan[role]):
                row = record_lookup[file_name]
                normal_rows.append(
                    {
                        "record": row["record"],
                        "role": role,
                        "role_index": role_index,
                        "record_position": row["record_position"],
                        "windows": row["windows"],
                        "hours": row["hours"],
                        "cache_file": row["cache_file"],
                    }
                )
    else:
        first_row = record_lookup[first.file_name]
        with np.load(first_row["cache_file"]) as archive:
            features = np.asarray(archive["features"], dtype=np.float32)
            names = [str(value) for value in archive["names"]]
        offset = float(first_row["time_offset_seconds"])
        ends = offset + (np.arange(len(features)) + 1) * window_seconds
        prefix = features[ends <= first.start_seconds - PREICTAL_EXCLUSION_SECONDS]
        split = (2 * len(prefix)) // 3
        if split < 1 or len(prefix) - split < 1:
            raise ValueError("strict Siena prefix cannot be divided")
        for role, values in (("fit", prefix[:split]), ("calibration", prefix[split:])):
            cache_path = normal_dir / f"{subject}_prefix_{role}.npz"
            np.savez_compressed(
                cache_path,
                features=values,
                names=np.asarray(names, dtype=str),
            )
            normal_rows.append(
                {
                    "record": f"{subject}_prefix_{role}",
                    "role": role,
                    "role_index": 0,
                    "record_position": first_row["record_position"],
                    "windows": int(len(values)),
                    "hours": len(values) * window_seconds / 3600.0,
                    "cache_file": str(cache_path.resolve()),
                }
            )

    first_position = record_lookup[first.file_name]["record_position"]
    event_files = {event.file_name for event in events}
    for row in record_rows[first_position + 1 :]:
        if row["file_name"] in event_files:
            continue
        normal_rows.append(
            {
                "record": row["record"],
                "role": "evaluation",
                "role_index": sum(value["role"] == "evaluation" for value in normal_rows),
                "record_position": row["record_position"],
                "windows": row["windows"],
                "hours": row["hours"],
                "cache_file": row["cache_file"],
            }
        )

    monitoring_rows = []
    for row in record_rows[first_position:]:
        offset = float(row["time_offset_seconds"])
        record_events = [
            {
                "event_id": event["event_id"],
                "start_seconds": event["start_seconds"] - offset,
                "end_seconds": event["end_seconds"] - offset,
                "raw_start_seconds": event["start_seconds"],
                "raw_end_seconds": event["end_seconds"],
                "consumed_initialization_event": event["event_id"] == event_rows[0]["event_id"],
            }
            for event in events_by_file.get(row["file_name"], [])
        ]
        monitoring_rows.append(
            {
                "record": row["record"],
                "record_position": row["record_position"],
                "windows": row["windows"],
                "hours": row["hours"],
                "time_offset_seconds": offset,
                "cache_file": row["cache_file"],
                "events": record_events,
            }
        )

    manifest = {
        "status": "V48_SIENA_STRICT_CAUSAL_CACHE",
        "dataset_role": "post-freeze cross-database stress test only",
        "subject": subject,
        "window_seconds": window_seconds,
        "first_seizure_duration_seconds": first.end_seconds - first.start_seconds,
        "normal_initialization_mode": plan["mode"],
        "normal_records": {
            role: [row["record"] for row in normal_rows if row["role"] == role]
            for role in ("fit", "calibration", "evaluation")
        },
        "events": event_rows,
        "normals": normal_rows,
        "monitoring_records": monitoring_rows,
        "causal_preprocessing": {
            "channels": COMMON_CHANNELS,
            "filter": "one-way stateful fourth-order Butterworth SOS bandpass 0.5-45 Hz",
            "source_frequency_hz": EXPECTED_SOURCE_FREQUENCY,
            "target_frequency_hz": TARGET_FREQUENCY,
            "decimation": "causal 4:1 after 45-Hz antialias bandpass",
            "state_carry": "only across timestamp-contiguous EDFs",
            "state_resets": state_resets,
            "state_carries": state_carries,
            "warmup_discard_seconds_after_reset": RESET_WARMUP_SECONDS,
        },
        "future_perturbation_invariance": invariance,
        "adapter_freeze_sha256": verify_freeze()["sha256"],
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(manifest_path, manifest)
    return {
        "subject": subject,
        "eligible": True,
        "cached": False,
        "window_seconds": window_seconds,
        "events": len(event_rows),
        "post_first_targets": len(event_rows) - 1,
        "monitoring_hours": sum(row["hours"] for row in monitoring_rows),
        "future_perturbation_invariance": invariance["status"],
        "runtime_seconds": manifest["runtime_seconds"],
    }
