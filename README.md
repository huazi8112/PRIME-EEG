# PRIME-EEG

Patient-referenced, confirmation-gated incremental exemplar memory for continuous scalp-EEG seizure detection.

This repository curates the **frozen implementation** used for the PRIME-EEG study. Package version `0.1.0` describes the software layout, not a new detector. The underlying method remains `v48-strict-causal-frozen`; no threshold tuning or new experiments are introduced by this release.

[中文使用说明](docs/QUICKSTART_zh.md) · [Reproduction protocol](docs/REPRODUCIBILITY.md) · [Baseline comparison scope](docs/BASELINES.md)

## Scope

The task starts **after the first independently confirmed seizure** in each recording stage. Pre-first-seizure non-seizure data establish a fixed patient reference. Later independent confirmations update reliable features, exemplar memory and calibration thresholds. Detector alarms never become training labels by themselves.

This is not first-seizure discovery, seizure prediction, a prospective bedside trial, or an untouched external validation. CHB-MIT comprises 22 recording stages from 21 independent subjects (`chb01` and `chb21` are one person). Fold rules were selected on a historical timing-corrected, noncausal-preprocessing cache, then frozen and transferred to strict-causal features. The release does **not** describe the entire development/selection pipeline as end-to-end causal. Siena is a historical cross-database stress test, not successful independent external validation.

## What is included

- Strict-causal EDF preprocessing and the 333-dimensional feature dictionary.
- Patient median/robust-scale alignment, top-50 reliability selection, at most 24 active exemplars, maximum cosine matching and dual-threshold hysteresis.
- The 21 previously frozen fold rules and patient-reference split metadata.
- Main evaluation; four component ablations; alarm-gated updating; strict-ictal matching; five real-elapsed-time confirmation schedules; fixed-configuration Siena stress reproduction.
- Cached neural-baseline replay using locked comparison rules, repaired sequential updates and the harmonized scorer; no new neural inference.
- Small, derived reference result ledgers, provenance hashes, automated tests and CI configuration.

Raw EEG, derived feature/neural-score caches, model weights, manuscript drafts and local environments are **not** included. Cached reference numbers are not a substitute for raw-data recomputation. Neural-baseline inference is not bundled; its provenance, cached-score replay and comparison outputs are separated from the detector (see [baseline documentation](docs/BASELINES.md)).

## Installation

Tested numerical environment: CPython **3.12.14**, NumPy **2.3.5**, SciPy **1.16.3**, pyEDFlib **0.1.42**. Python 3.12 is required for this pinned release. Commands below are run from the repository root.

```text
python -m venv .venv
```

Activate the environment: Windows PowerShell `.\.venv\Scripts\Activate.ps1`; Linux/macOS `source .venv/bin/activate`. Then:

```text
python -m pip install -e .
python -m unittest discover -s tests -v
python -m prime_eeg verify-results
```

The last command checks frozen configuration integrity and arithmetic from the supplied derived ledgers. It does **not** run EEG processing or neural inference. It needs the repository's `reference/` directory; an installed package alone does not contain these ledgers.

## Main result reproduction

Obtain the CHB-MIT Scalp EEG Database 1.0.0 independently from its official PhysioNet distribution. Set `--data-root` to the directory directly containing `chb01/`, `chb02/`, etc., with each stage's EDF files and summary text. No workstation-specific dataset path is assumed.

```text
python -m prime_eeg build-cache --dataset chbmit --data-root "data/chbmit" --output "cache/chbmit"
python -m prime_eeg evaluate --cache "cache/chbmit" --output "runs/main" --bootstrap 10000
python -m prime_eeg verify-results --recomputed "runs/main/result.json"
```

All output directories must be **new**. The commands refuse to overwrite an existing directory. If a build is interrupted, keep it for inspection and choose a new directory for a rerun; there is deliberately no force-overwrite or retuning flag.

Expected full-cohort point estimates:

| Quantity | Frozen result |
|---|---:|
| Target seizures detected | 118 / 153 |
| Event sensitivity | 77.12% |
| False alarm events | 289 |
| Eligible monitored exposure | 629.522778 h |
| False alarms per hour | 0.459078 |
| Event precision | 28.99% |
| Event F1 | 42.14% |
| Macro independent-subject sensitivity | 80.59% |
| Conditional median latency | 14.5 s |

Latency is conditional on detection, retains allowed negative values and is not an all-event time-to-detection estimate. Approximately 11 false alarms per 24 monitored hours is not a claim of unattended clinical suitability.

For a small functional run, add `--subjects chb19` to both build and evaluation. A subset is explicitly marked as a subset and cannot be compared with the full-cohort totals.

## Other analyses

Each command below needs a new output directory:

```text
python -m prime_eeg evaluate --cache "cache/chbmit" --variant no-update --output "runs/no-update"
python -m prime_eeg evaluate --cache "cache/chbmit" --variant physiology-only --output "runs/physiology"
python -m prime_eeg evaluate --cache "cache/chbmit" --variant connectivity-only --output "runs/connectivity"
python -m prime_eeg evaluate --cache "cache/chbmit" --variant no-hysteresis --output "runs/no-hysteresis"
python -m prime_eeg evaluate --cache "cache/chbmit" --variant alarm-gated --output "runs/alarm-gated"
python -m prime_eeg evaluate --cache "cache/chbmit" --strict-ictal --output "runs/strict-ictal"
python -m prime_eeg delay --cache "cache/chbmit" --data-root "data/chbmit" --output "runs/elapsed-delay" --bootstrap 10000
```

`delay` reads EDF header times: acquisition gaps count toward elapsed confirmation delays but not monitored exposure. Its schedules are immediate, 5, 30 and 60 minutes, and the next 8-hour review anchored at each stage's task origin. They are simulated information-release schedules, not measured clinician response times.

Siena uses its independently downloaded 1.0.0 EDF/annotation distribution; `--data-root` directly contains `PN*/` directories:

```text
python -m prime_eeg build-cache --dataset siena --data-root "data/siena" --output "cache/siena"
python -m prime_eeg evaluate --dataset siena --cache "cache/siena" --output "runs/siena"
```

The locked Siena cohort has 9 eligible subjects and 25 targets. Expected stress-test outcome: 15 detected events, 70 false alarms, 77.149444 h, sensitivity 60.00%, FAR 0.907330/h. It must not be presented as successful external validation.

## Repository layout

```text
src/prime_eeg/        Detector, preprocessing, scoring and command-line interface
  resources/         Frozen fold rules, adapter, cohort splits and checksums
reference/           Small derived result ledgers; no EEG or feature arrays
tests/               Synthetic numerical and temporal invariance tests
docs/                Scientific scope, data setup and reproduction instructions
provenance/          Source ancestry and artifact hashes
.github/workflows/   Lightweight tests; no patient data needed
```

## Provenance and interpretation

Active numerical function/class bodies were extracted from the frozen research implementation; imports, file handling and entry points were reorganized. Source-file and syntax hashes are listed in `provenance/source_manifest.json`. A few stale documentation strings were corrected without changing numerical expressions (see `CHANGELOG.md`). Original cache arrays and frozen experimental files are not rewritten.

Archived primary macro-sensitivity confidence intervals had a repeated-cluster weighting defect. `reference/confidence_intervals.json` contains the existing post-freeze correction, while retaining original pooled-metric seeds and point estimates. This is a statistical reporting correction, not a changed detector.

## Data, citation and rights

Use each dataset under its own access and reuse terms; this repository does not redistribute EEG recordings. Reference ledgers contain dataset pseudonyms, event identifiers and derived metrics, not subject names or raw signals. See [data description](docs/DATA.md) and [citation guidance](CITATION.md).

No software license is granted in this release. The absence of a `LICENSE` file is intentional; public visibility alone does not grant permission to reuse or redistribute the software. Third-party code and checkpoints retain their own terms.

Research software only. Not a medical device or clinical decision system.
