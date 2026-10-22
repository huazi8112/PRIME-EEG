"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import re
from dataclasses import dataclass, asdict
from .features import COMMON_CHANNELS
DATA_ROOT = None
ELECTRODE_ALIASES = {'T3': 'T7', 'T5': 'P7', 'T4': 'T8', 'T6': 'P8'}
PREICTAL_EXCLUSION_SECONDS = 300.0
MINIMUM_PREFIX_SECONDS = 1200.0
CHUNK_SECONDS = 900.0


@dataclass(frozen=True)
class SienaEvent:
    event_id: str
    sequence: int
    file_name: str
    start_seconds: float
    end_seconds: float


def branch_seconds_v35(first_duration: float) -> int:
    if first_duration < 20.0:
        return 2
    if first_duration <= 75.0:
        return 4
    return 10


def edf_module():
    import pyedflib

    return pyedflib


def subject_directories() -> list[Path]:
    return sorted(path for path in DATA_ROOT.glob("PN*") if path.is_dir())


def natural_record_key(path: Path | str) -> tuple[int, ...]:
    stem = Path(path).stem
    suffix = stem.split("-", 1)[1]
    values = tuple(int(value) for value in re.findall(r"\d+", suffix))
    if not values:
        raise ValueError(f"cannot derive Siena record order from {path}")
    return values


def edf_order(subject_dir: Path) -> list[str]:
    return [path.name for path in sorted(subject_dir.glob("*.edf"), key=natural_record_key)]


def canonical_electrode(label: str) -> str:
    value = re.sub(r"^EEG\s+", "", label.strip(), flags=re.IGNORECASE).upper()
    value = re.sub(r"\s+", "", value)
    return ELECTRODE_ALIASES.get(value, value)


def electrode_indices(reader: pyedflib.EdfReader) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, label in enumerate(reader.getSignalLabels()):
        canonical = canonical_electrode(label)
        if canonical not in indices:
            indices[canonical] = index
    required = sorted({part for channel in COMMON_CHANNELS for part in channel.split("-")})
    missing = [channel for channel in required if channel not in indices]
    if missing:
        raise ValueError(f"missing canonical Siena electrodes: {missing}")
    return indices


def edf_duration(path: Path) -> float:
    reader = edf_module().EdfReader(str(path))
    try:
        return float(reader.file_duration)
    finally:
        reader.close()


def read_bipolar_interval(
    path: Path, start_seconds: float, stop_seconds: float
) -> tuple[np.ndarray, float]:
    reader = edf_module().EdfReader(str(path))
    try:
        indices = electrode_indices(reader)
        required = sorted({part for channel in COMMON_CHANNELS for part in channel.split("-")})
        frequencies = np.asarray(
            [reader.getSampleFrequency(indices[channel]) for channel in required],
            dtype=float,
        )
        if not np.allclose(frequencies, frequencies[0]):
            raise ValueError(f"mixed EEG sample rates in {path}")
        frequency = float(frequencies[0])
        duration = float(reader.file_duration)
        start_seconds = max(0.0, float(start_seconds))
        stop_seconds = min(duration, float(stop_seconds))
        if stop_seconds <= start_seconds:
            raise ValueError(f"empty interval {path.name}: {start_seconds}:{stop_seconds}")
        start = int(round(start_seconds * frequency))
        stop = int(round(stop_seconds * frequency))
        signals = {
            channel: reader.readSignal(indices[channel], start=start, n=stop - start)
            for channel in required
        }
        bipolar = np.vstack(
            [signals[first] - signals[second] for first, second in (item.split("-") for item in COMMON_CHANNELS)]
        )
        return bipolar, frequency
    finally:
        reader.close()


def clock_seconds(value: str) -> float:
    normalized = re.sub(r"\s+", "", value)
    match = re.search(r"(\d{1,2})[.:](\d{2})[.:](\d{2})", normalized)
    if match is None:
        raise ValueError(f"invalid Siena clock value: {value!r}")
    hour, minute, second = (int(part) for part in match.groups())
    if hour > 23 or minute > 59 or second > 59:
        raise ValueError(f"out-of-range Siena clock value: {value!r}")
    return float(hour * 3600 + minute * 60 + second)


def relative_clock(value: str, registration_start: str) -> float:
    relative = clock_seconds(value) - clock_seconds(registration_start)
    if relative < 0:
        relative += 24 * 3600
    return relative


def resolve_annotation_file(subject_dir: Path, file_name: str) -> str:
    file_name = file_name.strip()
    if (subject_dir / file_name).exists():
        return file_name
    if "-" in file_name:
        subject_corrected = f"{subject_dir.name}-{file_name.split('-', 1)[1]}"
        if (subject_dir / subject_corrected).exists():
            return subject_corrected
    stem = Path(file_name).stem
    candidates = sorted(subject_dir.glob(f"{stem}-*.edf"), key=natural_record_key)
    if len(candidates) == 1:
        return candidates[0].name
    all_edfs = list(subject_dir.glob("*.edf"))
    if "-" in file_name:
        annotated_numbers = tuple(
            int(value) for value in re.findall(r"\d+", file_name.split("-", 1)[1])
        )
        if annotated_numbers:
            range_candidates = [
                path
                for path in all_edfs
                if natural_record_key(path)[: len(annotated_numbers)]
                == annotated_numbers
            ]
            if len(range_candidates) == 1:
                return range_candidates[0].name
    if len(all_edfs) == 1:
        return all_edfs[0].name
    raise ValueError(f"cannot resolve annotation file {file_name!r} in {subject_dir}")


def event_from_block(
    block: dict[str, str], file_occurrence: int, subject_dir: Path
) -> SienaEvent:
    file_name = resolve_annotation_file(subject_dir, block["file name"])
    start_value = block.get("seizure start time", block.get("start time"))
    end_value = block.get("seizure end time", block.get("end time"))
    if start_value is None or end_value is None:
        raise ValueError(f"incomplete Siena seizure time block: {block}")
    start = relative_clock(start_value, block["registration start time"])
    end = relative_clock(end_value, block["registration start time"])
    if end <= start:
        end += 24 * 3600
    sequence = int(block["sequence"])
    return SienaEvent(
        event_id=f"{Path(file_name).stem}_s{file_occurrence}",
        sequence=sequence,
        file_name=file_name,
        start_seconds=start,
        end_seconds=end,
    )


def annotation_path(subject_dir: Path) -> Path:
    return subject_dir / f"Seizures-list-{subject_dir.name}.txt"


def parse_first_event_only(path: Path) -> SienaEvent:
    context: dict[str, str] = {}
    block: dict[str, str] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = re.match(r"Seizure\s+n\s+(\d+)", line, flags=re.IGNORECASE)
            if match:
                if block is not None:
                    raise ValueError(f"incomplete first seizure block in {path}")
                block = {**context, "sequence": match.group(1)}
                continue
            pair = re.match(r"([^:]+):\s*(.*?)\s*$", line)
            if pair:
                key = pair.group(1).strip().lower()
                value = pair.group(2).strip()
                if block is None:
                    context[key] = value
                else:
                    block[key] = value
            if block is not None and (
                "seizure end time" in block or "end time" in block
            ):
                return event_from_block(block, 1, path.parent)
    raise ValueError(f"no complete first seizure in {path}")


def parse_all_events(path: Path) -> list[SienaEvent]:
    blocks: list[dict[str, str]] = []
    context: dict[str, str] = {}
    current: dict[str, str] | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        match = re.match(r"Seizure\s+n\s+(\d+)", line, flags=re.IGNORECASE)
        if match:
            if current is not None:
                raise ValueError(f"incomplete seizure block before {line!r} in {path}")
            current = {**context, "sequence": match.group(1)}
            continue
        pair = re.match(r"([^:]+):\s*(.*?)\s*$", line)
        if pair:
            key = pair.group(1).strip().lower()
            value = pair.group(2).strip()
            if current is None:
                context[key] = value
            else:
                current[key] = value
        if current is not None and (
            "seizure end time" in current or "end time" in current
        ):
            blocks.append(current)
            current = None
    if current is not None:
        raise ValueError(f"incomplete final seizure block in {path}")
    occurrences: dict[str, int] = {}
    events = []
    for block in blocks:
        file_name = resolve_annotation_file(path.parent, block["file name"])
        occurrences[file_name] = occurrences.get(file_name, 0) + 1
        events.append(event_from_block(block, occurrences[file_name], path.parent))
    return events


def initialization_plan(subject_dir: Path, first: SienaEvent) -> dict:
    order = edf_order(subject_dir)
    if first.file_name not in order:
        raise ValueError(f"first seizure file missing: {first.file_name}")
    first_position = order.index(first.file_name)
    historical = order[:first_position]
    if len(historical) >= 2:
        calibration_count = min(2, max(1, len(historical) // 3))
        return {
            "eligible_normal_initialization": True,
            "mode": "all_earlier_seizure_free_edfs",
            "fit": historical[:-calibration_count],
            "calibration": historical[-calibration_count:],
            "usable_prefix_seconds": first.start_seconds - PREICTAL_EXCLUSION_SECONDS,
        }
    usable_prefix = first.start_seconds - PREICTAL_EXCLUSION_SECONDS
    if usable_prefix < MINIMUM_PREFIX_SECONDS:
        return {
            "eligible_normal_initialization": False,
            "mode": "insufficient_normal_initialization",
            "fit": [],
            "calibration": [],
            "usable_prefix_seconds": usable_prefix,
        }
    return {
        "eligible_normal_initialization": True,
        "mode": "first_record_prefix_with_last_300s_excluded",
        "fit": [f"{subject_dir.name}_prefix_fit"],
        "calibration": [f"{subject_dir.name}_prefix_calibration"],
        "usable_prefix_seconds": usable_prefix,
    }
