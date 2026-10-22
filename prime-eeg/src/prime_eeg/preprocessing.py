"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import time
from datetime import timedelta
import pyedflib
from scipy import signal
from .features import COMMON_CHANNELS, extract_fast_features, fast_edf_duration, parse_summary, read_file_interval, summary_file_order
from .io import write_json
DATA_ROOT = None
CACHE_ROOT = None
SOURCE_ROOT = Path(__file__).parent / 'resources' / 'cohort'
TARGET_FREQUENCY = 128
SOURCE_FREQUENCY = 256
RESET_WARMUP_SECONDS = 30.0
PREICTAL_EXCLUSION_SECONDS = 300.0
CONTIGUITY_TOLERANCE_SECONDS = 1.0 / SOURCE_FREQUENCY


def filter_design() -> np.ndarray:
    # The 45-Hz high cutoff is also the anti-alias limit before 2:1 decimation;
    # target Nyquist is 64 Hz.  A one-way SOS realization is stable and streamable.
    return signal.butter(
        4,
        [0.5, 45.0],
        btype="bandpass",
        fs=SOURCE_FREQUENCY,
        output="sos",
    )


def causal_filter_downsample(
    raw: np.ndarray, sos: np.ndarray, state: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=float)
    if state is None:
        state = np.zeros((len(sos), raw.shape[0], 2), dtype=float)
    filtered, final_state = signal.sosfilt(sos, raw, axis=1, zi=state)
    # Every output uses only the current and earlier source samples.  Because the
    # causal bandpass has already removed content above 45 Hz, this 2:1 decimator
    # does not require a zero-phase or future-looking antialiasing stage.
    downsampled = filtered[:, ::2]
    return downsampled.astype(np.float32), final_state


def edf_metadata(path: Path) -> tuple[object, float, tuple[float, ...]]:
    reader = pyedflib.EdfReader(str(path))
    try:
        return (
            reader.getStartdatetime(),
            float(reader.getFileDuration()),
            tuple(float(value) for value in reader.getSampleFrequencies()),
        )
    finally:
        reader.close()


def is_contiguous(previous_end: object | None, current_start: object) -> bool:
    if previous_end is None:
        return False
    return abs((current_start - previous_end).total_seconds()) <= CONTIGUITY_TOLERANCE_SECONDS


def load_record(path: Path, window_seconds: int, state: np.ndarray | None):
    duration = fast_edf_duration(path)
    raw, frequency = read_file_interval(path, 0.0, duration)
    if not math.isclose(frequency, SOURCE_FREQUENCY):
        raise ValueError(f"{path} has {frequency} Hz, expected {SOURCE_FREQUENCY} Hz")
    filtered, final_state = causal_filter_downsample(raw, filter_design(), state)
    features, names = extract_fast_features(
        filtered, TARGET_FREQUENCY, window_seconds, COMMON_CHANNELS
    )
    return features, names, final_state, raw


def save_feature_file(
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


def load_cached_record(path: Path):
    with np.load(path) as archive:
        return (
            np.asarray(archive["features"], dtype=np.float32),
            [str(value) for value in archive["names"]],
            float(archive["time_offset_seconds"]),
            np.asarray(archive["filter_state"], dtype=float),
        )


def future_perturbation_test(raw: np.ndarray, window_seconds: int) -> dict:
    # Five minutes are sufficient to test the exact invariance property while
    # keeping this mandatory audit cheap for every subject.
    count = min(raw.shape[1], 5 * 60 * SOURCE_FREQUENCY)
    sample = np.asarray(raw[:, :count], dtype=float)
    boundary = count // 2
    perturbed = sample.copy()
    rng = np.random.default_rng(4800 + window_seconds)
    amplitude = np.maximum(np.std(sample, axis=1, keepdims=True), 1.0)
    perturbed[:, boundary:] += rng.normal(
        0.0, 20.0, size=perturbed[:, boundary:].shape
    ) * amplitude
    left, _ = causal_filter_downsample(sample, filter_design())
    right, _ = causal_filter_downsample(perturbed, filter_design())
    filtered_prefix = boundary // 2
    filter_equal = np.array_equal(left[:, :filtered_prefix], right[:, :filtered_prefix])
    left_features, left_names = extract_fast_features(
        left, TARGET_FREQUENCY, window_seconds, COMMON_CHANNELS
    )
    right_features, right_names = extract_fast_features(
        right, TARGET_FREQUENCY, window_seconds, COMMON_CHANNELS
    )
    complete_prefix_windows = int((boundary / SOURCE_FREQUENCY) // window_seconds)
    feature_equal = np.array_equal(
        left_features[:complete_prefix_windows],
        right_features[:complete_prefix_windows],
    )
    return {
        "status": "PASS" if filter_equal and feature_equal and left_names == right_names else "FAIL",
        "future_perturbation_sample": boundary,
        "unchanged_filtered_prefix_samples": filtered_prefix,
        "unchanged_complete_feature_windows": complete_prefix_windows,
        "filtered_prefix_exactly_equal": bool(filter_equal),
        "feature_prefix_exactly_equal": bool(feature_equal),
    }


def build_subject(subject: str, force: bool = False) -> dict:
    started = time.perf_counter()
    source_manifest = json.loads(
        (SOURCE_ROOT / subject / "manifest.json").read_text(encoding="utf-8")
    )
    subject_dir = DATA_ROOT / subject
    summary_path = subject_dir / f"{subject}-summary.txt"
    order = summary_file_order(summary_path)
    events = parse_summary(summary_path)
    window_seconds = int(source_manifest["window_seconds"])
    output = CACHE_ROOT / subject
    record_dir = output / "records"
    event_dir = output / "events"
    normal_dir = output / "normals"
    record_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    previous_end = None
    state = None
    record_rows = []
    record_lookup = {}
    schema = None
    invariance = None
    continuity_count = 0
    reset_count = 0
    for number, file_name in enumerate(order, 1):
        path = subject_dir / file_name
        record = Path(file_name).stem
        start_datetime, duration, frequencies = edf_metadata(path)
        if set(frequencies) != {float(SOURCE_FREQUENCY)}:
            raise ValueError(f"mixed or unexpected frequencies in {path}: {set(frequencies)}")
        contiguous = is_contiguous(previous_end, start_datetime)
        if not contiguous:
            state = None
            reset_count += 1
        else:
            continuity_count += 1
        cache_path = record_dir / f"{record}.npz"
        print(
            f"v48 causal {subject} {number}/{len(order)} {record} "
            f"{'carry' if contiguous else 'reset'}",
            flush=True,
        )
        if cache_path.exists() and not force:
            features, names, time_offset, final_state = load_cached_record(cache_path)
            # A cached carry-state record is safe only because the exact previous
            # state was used when it was created and the subject build is immutable.
            state = final_state
        else:
            features, names, final_state, raw = load_record(
                path, window_seconds, state
            )
            drop_windows = (
                0
                if contiguous
                else int(math.ceil(RESET_WARMUP_SECONDS / window_seconds))
            )
            time_offset = float(drop_windows * window_seconds)
            features = features[drop_windows:]
            save_feature_file(
                cache_path, features, names, time_offset, final_state
            )
            state = final_state
            if invariance is None:
                invariance = future_perturbation_test(raw, window_seconds)
        if schema is None:
            schema = names
        elif names != schema:
            raise AssertionError(f"feature schema changed in {subject}:{record}")
        row = {
            "record": record,
            "file_name": file_name,
            "record_position": number - 1,
            "windows": int(len(features)),
            "hours": len(features) * window_seconds / 3600.0,
            "time_offset_seconds": time_offset,
            "filter_state_action": "carry" if contiguous else "reset_then_discard_warmup",
            "cache_file": str(cache_path.resolve()),
        }
        record_rows.append(row)
        record_lookup[record] = row
        previous_end = start_datetime + timedelta(seconds=duration)

    if invariance is None:
        # Existing immutable caches may make raw recomputation unnecessary.  The
        # build-level test file from the first run remains authoritative.
        old_manifest = output / "manifest.json"
        if old_manifest.exists():
            invariance = json.loads(old_manifest.read_text(encoding="utf-8"))[
                "future_perturbation_invariance"
            ]
        else:
            raise AssertionError("future perturbation test was not executed")

    event_rows = []
    events_by_record: dict[str, list[dict]] = {}
    for event in events:
        record = Path(event.file_name).stem
        record_row = record_lookup[record]
        with np.load(record_row["cache_file"]) as archive:
            features = np.asarray(archive["features"], dtype=np.float32)
            names = [str(value) for value in archive["names"]]
        offset = float(record_row["time_offset_seconds"])
        centers = offset + (np.arange(len(features)) + 0.5) * window_seconds
        labels = (centers >= event.start_seconds) & (centers < event.end_seconds)
        if not np.any(labels):
            nearest = int(np.argmin(np.abs(centers - 0.5 * (event.start_seconds + event.end_seconds))))
            labels[nearest] = True
        positive = features[labels]
        relative_seconds = centers[labels] - float(event.start_seconds)
        event_path = event_dir / f"{event.event_id}.npz"
        np.savez_compressed(
            event_path,
            features=positive,
            names=np.asarray(names, dtype=str),
            labels=np.ones(len(positive), dtype=np.int8),
            seconds=relative_seconds.astype(np.float64),
        )
        event_row = {
            "event_id": event.event_id,
            "record": record,
            "record_position": record_row["record_position"],
            "start_seconds": event.start_seconds,
            "end_seconds": event.end_seconds,
            "duration_seconds": event.end_seconds - event.start_seconds,
            "windows": int(len(positive)),
            "ictal_windows": int(len(positive)),
            "cache_file": str(event_path.resolve()),
        }
        event_rows.append(event_row)
        events_by_record.setdefault(record, []).append(event_row)

    source_normal_rows = {row["record"]: row for row in source_manifest["normals"]}
    normal_rows = []
    if source_manifest["normal_initialization_mode"] == "first_record_prefix_with_last_300s_excluded":
        first_event = event_rows[0]
        first_record = record_lookup[first_event["record"]]
        with np.load(first_record["cache_file"]) as archive:
            prefix_features = np.asarray(archive["features"], dtype=np.float32)
            names = [str(value) for value in archive["names"]]
        offset = float(first_record["time_offset_seconds"])
        ends = offset + (np.arange(len(prefix_features)) + 1) * window_seconds
        prefix_features = prefix_features[ends <= first_event["start_seconds"] - PREICTAL_EXCLUSION_SECONDS]
        split = int((2 * len(prefix_features)) // 3)
        split -= split % 1
        if split < 1 or len(prefix_features) - split < 1:
            raise ValueError(f"insufficient strict causal prefix for {subject}")
        for role, values in (
            ("fit", prefix_features[:split]),
            ("calibration", prefix_features[split:]),
        ):
            name = f"{subject}_prefix_{role}"
            cache_path = normal_dir / f"{name}.npz"
            np.savez_compressed(
                cache_path,
                features=values,
                names=np.asarray(names, dtype=str),
            )
            normal_rows.append(
                {
                    "record": name,
                    "role": role,
                    "role_index": 0,
                    "record_position": first_record["record_position"],
                    "windows": int(len(values)),
                    "hours": len(values) * window_seconds / 3600.0,
                    "cache_file": str(cache_path.resolve()),
                }
            )
    else:
        for role in ("fit", "calibration", "evaluation"):
            for role_index, record in enumerate(source_manifest["normal_records"][role]):
                strict = record_lookup[record]
                normal_rows.append(
                    {
                        "record": record,
                        "role": role,
                        "role_index": role_index,
                        "record_position": strict["record_position"],
                        "windows": strict["windows"],
                        "hours": strict["hours"],
                        "cache_file": strict["cache_file"],
                    }
                )
    # Prefix-mode evaluation records are ordinary EDFs and were not added above.
    known_normal_records = {row["record"] for row in normal_rows}
    for role_index, record in enumerate(source_manifest["normal_records"]["evaluation"]):
        if record in known_normal_records:
            continue
        strict = record_lookup[record]
        normal_rows.append(
            {
                "record": record,
                "role": "evaluation",
                "role_index": role_index,
                "record_position": strict["record_position"],
                "windows": strict["windows"],
                "hours": strict["hours"],
                "cache_file": strict["cache_file"],
            }
        )

    first_position = event_rows[0]["record_position"]
    monitoring_rows = []
    for record_row in record_rows[first_position:]:
        offset = float(record_row["time_offset_seconds"])
        record_events = []
        for event in events_by_record.get(record_row["record"], []):
            record_events.append(
                {
                    "event_id": event["event_id"],
                    "start_seconds": float(event["start_seconds"]) - offset,
                    "end_seconds": float(event["end_seconds"]) - offset,
                    "raw_start_seconds": event["start_seconds"],
                    "raw_end_seconds": event["end_seconds"],
                    "consumed_initialization_event": event["event_id"] == event_rows[0]["event_id"],
                }
            )
        monitoring_rows.append(
            {
                "record": record_row["record"],
                "record_position": record_row["record_position"],
                "windows": record_row["windows"],
                "hours": record_row["hours"],
                "time_offset_seconds": offset,
                "cache_file": record_row["cache_file"],
                "events": record_events,
            }
        )

    manifest = {
        "status": "V48_STRICT_CAUSAL_CHBMIT_CACHE",
        "subject": subject,
        "window_seconds": window_seconds,
        "first_seizure_duration_seconds": source_manifest["first_seizure_duration_seconds"],
        "normal_initialization_mode": source_manifest["normal_initialization_mode"],
        "normal_records": source_manifest["normal_records"],
        "events": event_rows,
        "normals": normal_rows,
        "monitoring_records": monitoring_rows,
        "causal_preprocessing": {
            "filter": "one-way stateful 4th-order Butterworth SOS bandpass 0.5-45 Hz",
            "resampling": "causal 2:1 decimation after the 45-Hz antialiasing bandpass",
            "source_frequency_hz": SOURCE_FREQUENCY,
            "target_frequency_hz": TARGET_FREQUENCY,
            "state_carry": "only across sample-contiguous EDFs",
            "state_reset": "on every acquisition gap",
            "warmup_discard_seconds_after_reset": RESET_WARMUP_SECONDS,
            "contiguous_transitions": continuity_count,
            "state_resets": reset_count,
        },
        "future_perturbation_invariance": invariance,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output / "manifest.json", manifest)
    return {
        "subject": subject,
        "eligible": True,
        "window_seconds": window_seconds,
        "events": len(event_rows),
        "records": len(monitoring_rows),
        "monitoring_hours": sum(row["hours"] for row in monitoring_rows),
        "future_perturbation_invariance": invariance["status"],
        "runtime_seconds": manifest["runtime_seconds"],
    }
