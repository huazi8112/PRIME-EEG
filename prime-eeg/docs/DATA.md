# Data contract

Datasets must be obtained independently from their official distributions. The release contains no EEG, sample excerpts, converted montage signals, feature caches or neural checkpoints.

## CHB-MIT 1.0.0

`--data-root` must contain the original `chbXX` directories and their `chbXX-summary.txt` files. Keep original EDF names, labels and header timestamps. Do not concatenate across acquisition gaps or rename records before building the cache.

The fixed 22-stage cohort is in `src/prime_eeg/resources/cohort.json`. `chb03` and `chb24` do not meet the prespecified patient-reference availability requirements. `chb01` and `chb21` remain separate recording-stage models, but are always a single held-out/resampled independent subject.

Each stage's window length and pre-first-seizure fit/calibration roles are provided as metadata in `resources/cohort/`. No legacy noncausal feature cache is needed to interpret those split definitions. Fit data alone establish location/scale; calibration data alone establish score quantiles.

## Siena 1.0.0

`--data-root` must directly contain the original `PNXX` directories, EDF files and their seizure annotation text files. The frozen adapter reconstructs the same 18 bipolar derivations from canonical referential electrodes, including T3/T4/T5/T6 aliases. Missing required electrodes cause failure, not imputation. Only the frozen 9 eligible subjects are processed by default.

## Portable cache schema

Generated subject manifests contain `cache_file` paths relative to the dataset cache root. Copying a newly built cache directory to another machine does not require replacing author-specific paths. Loading existing absolute-path manifests is supported for comparison with the original archive; public resources do not contain such paths.

Each `.npz` record contains `features` (float32), `names`, `time_offset_seconds` and final filter state. Event files contain positive features, labels and within-event seconds. Manifests preserve warmup offsets, initialization-event masks, annotations and record order. Loading `.npz` never enables pickle.

Store caches outside version control. Even derived arrays may retain sensitive information and remain subject to dataset terms. The small `reference/` ledgers contain public dataset pseudonyms, event IDs, counts and latencies for reproducibility.
