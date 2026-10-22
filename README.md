# PRIME-EEG

**Patient-Referenced Incremental Memory of Exemplars** for continuous scalp-EEG seizure detection.

PRIME-EEG detects subsequent seizures in the same patient after a first seizure has been independently confirmed. It combines a fixed non-seizure reference, reliable feature selection, a small active exemplar memory, maximum cosine matching and dual-threshold hysteresis. Independent confirmations update the model; detector alarms do not supply their own labels.

This repository provides a curated reproduction package for **`v48-strict-causal-frozen`**. Software package version `0.1.0` identifies the packaging, not a new detector. The frozen numerical method and reported results are not retuned by this release.

[中文使用说明](prime-eeg/docs/QUICKSTART_zh.md) · [Reproduction protocol](prime-eeg/docs/REPRODUCIBILITY.md) · [Data setup](prime-eeg/docs/DATA.md) · [Neural-baseline scope](prime-eeg/docs/BASELINES.md)

## Scope and evidence

- **Task:** continuous detection after the first independently confirmed seizure in each recording stage. This is not first-seizure discovery or seizure prediction.
- **Main evaluation:** 22 CHB-MIT recording stages from 21 independent subjects, with 153 subsequent target seizures. `chb01` and `chb21` have separate stage-specific initialization but are jointly held out and resampled as one person.
- **Selection:** 21 outer-fold operating rules were selected on a historical timing-corrected cache with noncausal preprocessing, then frozen before transfer to strict-causal features. The deployed signal processing and sequential use of information are causal; the entire historical development and selection pipeline is not described as end-to-end causal.
- **Confirmation:** database annotations simulate an independent confirmation stream. No actual clinician response times were measured. At a shared scoring/update boundary, score first; the updated state affects later windows only.
- **Generalization:** CHB-MIT is internal evaluation with frozen transfer, not an untouched external cohort. Siena is a historical post-freeze cross-database stress test, not successful independent external validation.

The primary event scorer allows actual alarm points from 30 s before seizure onset to 60 s after offset. Therefore, the reported sensitivity does not mean every detection preceded independent confirmation. A separate ictal-only analysis is supplied. Latency is conditional on detection and retains permitted negative values.

## What is included

- EDF processing with one-way stateful filtering, causal downsampling and complete-window availability.
- A 333-dimensional dictionary: 180 physiological features and 153 elementwise Fisher-z correlation features.
- Fixed patient median/robust-scale alignment, top-50 reliability selection, at most 24 active matching exemplars and dual-threshold alarms.
- Frozen cohort/reference metadata and the 21 previously selected fold rules.
- Main evaluation, four component ablations, alarm-gated updating, ictal-only scoring, elapsed-time confirmation-delay analysis and the fixed-configuration Siena stress test.
- Replay of already selected neural-baseline comparison rules from separately prepared score caches, including the repaired sequential adapter and harmonized event scorer.
- Small derived reference ledgers, source/artifact provenance, checksums and synthetic tests.

Raw EEG, feature caches, neural-score caches, model checkpoints, manuscript drafts and local environments are **not included**. The active exemplar cap does not bound the full confirmed history or total program memory.

## Installation and quick check

Use **Python 3.12**. The reference numerical environment is CPython **3.12.14**, NumPy **2.3.5**, SciPy **1.16.3** and pyEDFlib **0.1.42**.

The installable package is inside **`prime-eeg/`**, not at the GitHub repository root:

```text
git clone https://github.com/huazi8112/PRIME-EEG.git
cd PRIME-EEG/prime-eeg
python -m venv .venv
```

If using GitHub's **Download ZIP**, extract it and open a terminal inside the inner `prime-eeg` folder containing `pyproject.toml`. Do not run the installation from the outer repository folder.

Activate the environment:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

Then run:

```text
python -m pip install -e .
python -m unittest discover -s tests -v
python -m prime_eeg verify-results
```

**All following commands and example relative paths assume this package directory as the working directory.** If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python`; changing the system execution policy is not required.

`verify-results` checks frozen resources and reconstructs aggregate arithmetic from the supplied derived ledgers. It does **not** process raw EEG, rerun neural inference or establish full end-to-end reproduction. Keep `reference/` alongside the source checkout: these ledgers are not bundled into the installed Python package.

## Reproduce the primary detector result

Download [CHB-MIT 1.0.0 from PhysioNet](https://physionet.org/content/chbmit/1.0.0/) independently under its applicable terms. The data root must directly contain `chb01/`, `chb02/`, etc.; preserve original EDF files, summary annotations and header timestamps. Dataset paths below are examples, not required workstation locations.

```text
python -m prime_eeg build-cache --dataset chbmit --data-root "data/chbmit" --output "cache/chbmit"
python -m prime_eeg evaluate --cache "cache/chbmit" --output "runs/main" --bootstrap 10000
python -m prime_eeg verify-results --recomputed "runs/main/result.json"
```

Use a **new output directory** for every build or analysis. Existing directories are not overwritten; interrupted runs should be retained or inspected, and a different output directory used for a rerun. No rule search or tuning is performed on these strict-causal outcomes.

For a smaller functional run, add `--subjects chb19` to **both** cache construction and evaluation. Do not compare subset totals with the full-cohort results or pass them to the full-cohort `--recomputed` check.

### Frozen full-cohort results

| Metric | Result |
|---|---:|
| Target seizures detected | 118 / 153 |
| Event sensitivity | 77.12% |
| False alarm events | 289 |
| Eligible monitored exposure | 629.522778 h |
| False-alarm rate | 0.459078 events/h |
| Event precision | 28.99% |
| Event F1 | 42.14% |
| Macro independent-subject sensitivity | 80.59% |
| Conditional median latency | 14.5 s |

Approximately 11 false alarms per 24 monitored hours is not evidence of suitability for unattended clinical use. Subject-level heterogeneity and uncertainty remain important; detailed definitions and bootstrap conventions are in the [reproduction protocol](prime-eeg/docs/REPRODUCIBILITY.md).

## Additional analyses

After building the CHB-MIT cache, use separate new output directories:

```text
python -m prime_eeg evaluate --cache "cache/chbmit" --variant no-update --output "runs/no-update"
python -m prime_eeg evaluate --cache "cache/chbmit" --variant physiology-only --output "runs/physiology"
python -m prime_eeg evaluate --cache "cache/chbmit" --variant connectivity-only --output "runs/connectivity"
python -m prime_eeg evaluate --cache "cache/chbmit" --variant no-hysteresis --output "runs/no-hysteresis"
python -m prime_eeg evaluate --cache "cache/chbmit" --variant alarm-gated --output "runs/alarm-gated"
python -m prime_eeg evaluate --cache "cache/chbmit" --strict-ictal --output "runs/strict-ictal"
python -m prime_eeg delay --cache "cache/chbmit" --data-root "data/chbmit" --output "runs/elapsed-delay" --bootstrap 10000
```

The delay analysis uses EDF header elapsed times, including acquisition gaps in the confirmation schedule but not in monitored exposure. It holds first-seizure initialization fixed and delays later confirmations: immediate, 5, 30 or 60 minutes, or the next task-anchored 8-hour review. These are simulated release schedules, not observed clinical response times. Use a fixed-offset timezone without daylight saving for the archived relative schedules; the reference environment used `Asia/Shanghai`.

### Siena stress test

Download [Siena Scalp EEG 1.0.0 from PhysioNet](https://physionet.org/content/siena-scalp-eeg/1.0.0/) separately. The data root must directly contain the original `PN*/` directories and annotation files.

```text
python -m prime_eeg build-cache --dataset siena --data-root "data/siena" --output "cache/siena"
python -m prime_eeg evaluate --dataset siena --cache "cache/siena" --output "runs/siena"
```

The locked stress-test cohort has 9 eligible subjects and 25 targets. Archived results are 15/25 detections, 70 false alarms over 77.149444 h, sensitivity 60.00% and FAR 0.907330 events/h. These results demonstrate a transfer limitation, not successful external validation.

## SeizureTransformer comparison: reproduction limits

The package can replay **already selected** comparison rules using separately prepared neural-score caches. It does **not** include neural inference, third-party network weights/code, or an entry point to repeat the complete nested baseline candidate sweep. Installing this detector package alone does not regenerate the full neural comparison. Required upstream provenance, cache schema and replay commands are documented in [BASELINES.md](prime-eeg/docs/BASELINES.md).

Under the locked post-first-seizure task, PRIME-EEG has higher sensitivity, precision and F1 point estimates than the reported conservative public-default/nested comparison rules, but its FAR is higher. Precision superiority is not established by the paired uncertainty intervals. The fixed-0.5 high-false-alarm reference is separate. Whole-record standardized public workflows are offline transfer references; differences in supervision, montage and input support prevent claims of universal or architecture-level superiority.

## Repository layout

```text
README.md                         GitHub homepage and repository entry point
.github/workflows/tests.yml       Repository-level synthetic test workflow
prime-eeg/                        Installable reproduction package
  pyproject.toml                  Package metadata and pinned dependencies
  README.md                       Package-directory instructions
  src/prime_eeg/                  Processing, detector, scoring and CLI
    resources/                    Frozen rules, cohort metadata and checksums
  reference/                      Small derived result ledgers; no raw EEG
  tests/                          Synthetic numerical and temporal tests
  docs/                           Data, protocol and baseline documentation
  provenance/                     Original source ancestry and artifact hashes
```

## Provenance and verification

The numerical implementation was extracted from the frozen research code; imports, paths and entry points were reorganized. Original source ancestry is documented in [source_manifest.json](prime-eeg/provenance/source_manifest.json). Original artifact hashes and the packaged-file manifest are separate: `provenance/release_files.sha256` covers the files inside the package directory, excluding the manifest itself, and is regenerated when package documentation changes. It does not replace the original scientific freeze hashes.

[release_validation.json](prime-eeg/provenance/release_validation.json) records validation performed for the original package export, including cache replay, synthetic tests and representative raw-EDF rebuilds. It does not claim a full raw rebuild of every subject or rerun of neural inference. A workflow file defines automated checks; its presence is not evidence that a remote GitHub Actions run has passed.

The existing post-freeze correction to the macro-sensitivity confidence interval is retained in `reference/confidence_intervals.json`; frozen pooled point estimates and numerical detector behavior are unchanged. See [CHANGELOG.md](prime-eeg/CHANGELOG.md).

## Data, citation and license status

Obtain and use each dataset under the terms shown on its official distribution page. This repository does not redistribute EEG recordings. The small reference ledgers contain public dataset pseudonyms, event identifiers and derived outcomes, not raw signals. Dataset access/reuse terms are separate from software and checkpoint terms.

See [CITATION.md](prime-eeg/CITATION.md). When reporting reproduction, identify the frozen method, software version, exact repository commit, numerical environment and dataset versions. Cite the associated PRIME-EEG manuscript using its final bibliographic metadata when available, and the original datasets and baseline method as applicable; no paper DOI is invented here.

**No software license has been assigned to this release; no `LICENSE` file is included.** Public access to the repository does not replace applicable permissions. Third-party software, weights and data retain their own terms.

Research software only. Not a medical device or autonomous clinical decision system.
