"""Numerical routines curated from the frozen research implementation."""


from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import re
from dataclasses import dataclass
import pyedflib
from scipy import signal


COMMON_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class Seizure:
    event_id: str
    file_name: str
    start_seconds: int
    end_seconds: int


def parse_summary(path: Path) -> list[Seizure]:
    current_file = ""
    starts = []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        file_match = re.match(r"File Name:\s+(.+\.edf)", line)
        if file_match:
            current_file = file_match.group(1).strip()
        start_match = re.match(r"Seizure(?: \d+)? Start Time:\s+(\d+) seconds", line)
        if start_match:
            starts.append((current_file, int(start_match.group(1))))
        end_match = re.match(r"Seizure(?: \d+)? End Time:\s+(\d+) seconds", line)
        if end_match:
            file_name, start = starts.pop(0)
            index = 1 + sum(event.file_name == file_name for event in events)
            events.append(
                Seizure(
                    event_id=f"{Path(file_name).stem}_s{index}",
                    file_name=file_name,
                    start_seconds=start,
                    end_seconds=int(end_match.group(1)),
                )
            )
    return events


def summary_file_order(path: Path) -> list[str]:
    output = []
    seen = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"File Name:\s+(.+\.edf)", line)
        if match and match.group(1).strip() not in seen:
            name = match.group(1).strip()
            output.append(name)
            seen.add(name)
    if not output:
        raise ValueError(f"no EDF order in {path}")
    return output


def fast_edf_duration(path: Path) -> float:
    with path.open("rb") as handle:
        header = handle.read(256)
    if len(header) != 256:
        raise ValueError(f"short EDF header: {path}")
    data_records = int(header[236:244].decode("ascii").strip())
    record_seconds = float(header[244:252].decode("ascii").strip())
    return data_records * record_seconds


def read_file_interval(
    path: Path, start_seconds: float, stop_seconds: float
) -> tuple[np.ndarray, float]:
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = [label.strip() for label in reader.getSignalLabels()]
        frequencies = np.asarray(reader.getSampleFrequencies(), dtype=float)
        if not np.allclose(frequencies, frequencies[0]):
            raise ValueError(f"mixed sample rates in {path}")
        frequency = float(frequencies[0])
        start = int(round(start_seconds * frequency))
        stop = int(round(stop_seconds * frequency))
        rows = []
        missing = []
        for channel in COMMON_CHANNELS:
            if channel in labels:
                rows.append(
                    reader.readSignal(labels.index(channel), start=start, n=stop - start)
                )
                continue
            # A few CHB12 EDFs switch from the longitudinal bipolar montage to
            # referential "electrode-CS2" channels. Reconstruct the exact same
            # bipolar derivation algebraically without using outcome information.
            first, second = channel.split("-", 1)
            first_reference = f"{first}-CS2"
            second_reference = f"{second}-CS2"
            if first_reference in labels and second_reference in labels:
                rows.append(
                    reader.readSignal(
                        labels.index(first_reference), start=start, n=stop - start
                    )
                    - reader.readSignal(
                        labels.index(second_reference), start=start, n=stop - start
                    )
                )
            else:
                # Other CHB12 records omit the common-reference suffix entirely.
                # O1 is mistyped as "01" in that block, so normalize only that
                # documented label before taking the same algebraic difference.
                first_plain = "01" if first == "O1" and "01" in labels else first
                second_plain = "01" if second == "O1" and "01" in labels else second
                if first_plain in labels and second_plain in labels:
                    rows.append(
                        reader.readSignal(
                            labels.index(first_plain), start=start, n=stop - start
                        )
                        - reader.readSignal(
                            labels.index(second_plain), start=start, n=stop - start
                        )
                    )
                else:
                    missing.append(channel)
        if missing:
            raise ValueError(f"{path.name} missing channels: {missing}")
        return np.vstack(rows), frequency
    finally:
        reader.close()


def extract_fast_features(
    filtered: np.ndarray,
    frequency: int,
    window_seconds: int,
    channels: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return 180 channel-wise features and 153 Fisher-z pair correlations.

    Historical ``geometry_tangent_corr`` names are retained for cache compatibility;
    these are scalar Fisher transforms, not SPD matrix logarithms or graph metrics.
    """
    window = int(window_seconds * frequency)
    count = filtered.shape[1] // window
    if count == 0:
        raise ValueError("no complete windows")
    # (window, channel, sample) permits one batched Welch transform and batched
    # temporal/correlation calculations instead of hundreds of Python loops.
    values = filtered[:, : count * window].reshape(len(channels), count, window)
    values = np.transpose(values, (1, 0, 2)).astype(float)
    values -= np.mean(values, axis=2, keepdims=True)
    frequencies, psd = signal.welch(
        values,
        fs=frequency,
        nperseg=min(256, window),
        axis=2,
        scaling="density",
    )
    trapezoid = getattr(np, "trapezoid", np.trapz)
    total_mask = (frequencies >= 0.5) & (frequencies <= 45.0)
    total = np.maximum(
        trapezoid(psd[:, :, total_mask], frequencies[total_mask], axis=2), 1e-12
    )
    blocks = []
    names = []
    for band, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        relative = trapezoid(psd[:, :, mask], frequencies[mask], axis=2) / total
        blocks.append(np.log(relative + 1e-8))
        names.extend(f"spectral_{band}_{channel}" for channel in channels)

    variance = np.var(values, axis=2)
    first = np.diff(values, axis=2)
    second = np.diff(first, axis=2)
    mobility = np.sqrt(np.var(first, axis=2) / np.maximum(variance, 1e-12))
    second_mobility = np.sqrt(
        np.var(second, axis=2) / np.maximum(np.var(first, axis=2), 1e-12)
    )
    complexity = second_mobility / np.maximum(mobility, 1e-12)
    line_length = np.mean(np.abs(first), axis=2)
    probability = psd[:, :, total_mask] / np.maximum(
        np.sum(psd[:, :, total_mask], axis=2, keepdims=True), 1e-12
    )
    entropy = -np.sum(probability * np.log(probability + 1e-12), axis=2) / np.log(
        probability.shape[2]
    )
    for metric, metric_values in (
        ("log_variance", np.log(variance + 1e-12)),
        ("mobility", mobility),
        ("complexity", complexity),
        ("log_line_length", np.log(line_length + 1e-12)),
        ("spectral_entropy", entropy),
    ):
        blocks.append(metric_values)
        names.extend(f"temporal_{metric}_{channel}" for channel in channels)

    scale = np.sqrt(np.mean(values**2, axis=2, keepdims=True))
    standardized = values / np.maximum(scale, 1e-12)
    correlation = np.einsum(
        "nci,ndi->ncd", standardized, standardized, optimize=True
    ) / window
    correlation = np.clip(np.nan_to_num(correlation, nan=0.0), -0.999, 0.999)
    upper = np.triu_indices(len(channels), k=1)
    blocks.append(np.arctanh(correlation[:, upper[0], upper[1]]))
    names.extend(
        f"geometry_tangent_corr_{channels[row]}__{channels[column]}"
        for row, column in zip(upper[0], upper[1])
    )
    return np.hstack(blocks).astype(np.float32), names
