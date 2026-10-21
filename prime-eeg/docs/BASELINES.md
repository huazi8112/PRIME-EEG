# SeizureTransformer comparison: provenance and limits

The comparison does not establish architecture superiority or a general state-of-the-art claim. The methods differ in supervision, montage reconstruction, input support, whole-record normalization and eligibility/exposure. PRIME-EEG uses independently confirmed patient-specific history; the public neural weights are not retrained in the reported comparison. Conservative neural working points have lower FAR than PRIME-EEG.

## Frozen neural inputs

- Official repository: `keruiwu/SeizureTransformer` on GitHub.
- Pinned source commit: `cf83f5906a8aea88b60b56e4f962c5d6657c28f7`.
- Converted checkpoint provenance: `eugenehp/seizuretransformer` on Hugging Face.
- Checkpoint SHA-256: `2cdc841001a0fbcdf1dfcbb02b3a26fa7af14002e01ebf9815fa09c82be06f61`.
- Model parameters in the archived comparison: 41,001,281.
- Fixed, label-free pseudoinverse reconstruction maps 18 bipolar derivations to the network's 19-channel reference layout, followed by average reference. This is a montage adaptation, not recovered ground-truth referential signals.

The public whole-record standardized workflow is an **offline transfer reference**, not equivalent causal-online evidence. The separate raw-score path uses 60 s complete inputs with a 15 s stride. Its sequential adapter was corrected post-freeze: each update waits until all positive summary inputs used by that update are fully available; a same-timestamp output is scored before the update. The original 636-rule baseline grid and 21 independent-subject folds were retained.

## Cached-score replay

`reference/baseline_comparison.json` contains harmonized archived public-default, nested postprocessing and fixed-0.5 results, plus the **repaired** nested raw-score results and selections. The superseded unrepaired raw result is not the current baseline. Native and common-exposure results remain separately labeled. Derived rows allow independent aggregate checking without downloading neural checkpoints.

The numerical raw-score adapter, repaired sequential update, official postprocessing adapter, actual-point scorer, original rule-grid generator and fixed selected comparison rules are retained. Given separately prepared caches in the archived schema:

```text
python -m prime_eeg baseline-replay --raw-cache "cache/neural-raw" --official-cache "cache/neural-official" --domain native --output "runs/baselines-native"
python -m prime_eeg baseline-replay --raw-cache "cache/neural-raw" --official-cache "cache/neural-official" --domain common --output "runs/baselines-common"
```

This replays the **already selected** comparison rules rather than repeating the candidate search. The shipped exposure registry is derived from the frozen original data and supports; it is not valid for a changed dataset/preprocessing/window definition. Raw-cache `.npz` files hold the nine summary arrays in `baseline_raw.RAW_FEATURES`, input interval starts/ends and `metadata_json`; manifests define normal roles, events and monitoring records. Official-cache `.npz` files hold thresholded segment starts/ends (`t030` through `t095`, according to the original threshold grid), with manifest and metadata. Arrays/weights are not bundled.

The package **does not include neural inference, a command to rerun the full nested baseline candidate sweep, or third-party network code**. Those steps are not silently represented by cached reference numbers. Exact end-to-end neural comparison regeneration additionally requires the pinned upstream implementation/checkpoint and the archived 18-to-19 montage adapter/preprocessing conventions to regenerate the complete score caches. The checkpoint and upstream code must be obtained under their respective terms; installing this detector package alone does not recreate them.

This is an explicit coverage limit of the curated detector release. Do not describe `verify-results` as a rerun of neural inference or as verification of every baseline preprocessing operation. The frozen detector can be reproduced independently of neural packages, XGBoost or scikit-learn.
