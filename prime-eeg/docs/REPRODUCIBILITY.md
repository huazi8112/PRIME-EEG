# Frozen reproduction protocol

## Method versus packaging

The primary detector identifier is `v48-strict-causal-frozen`. Reorganization into this Python package changes dependencies, path configuration, CLI and documentation, not the numerical detector. Its original protocol digest, 21 fold rules, cohort metadata and source ancestry are retained. Original source hashes refer to the original files, not to the reorganized module bytes.

## Data flow

1. CHB-MIT: 18 fixed bipolar channels, one-way fourth-order Butterworth SOS bandpass 0.5–45 Hz at 256 Hz, then causal 2:1 sample selection to 128 Hz. Siena: corresponding filter at 512 Hz and 4:1 decimation. This is finite-order filtering, not an ideal brick-wall anti-alias filter.
2. Filter state is inherited only across sample-contiguous EDFs. A reset discards `ceil(30 / L)` complete windows. Alarm state resets at every EDF boundary, including boundaries where filter state carries. Trailing incomplete windows and acquisition gaps do not add exposure.
3. Fixed window length is selected once from the first seizure's duration: 2 s for <20 s, 4 s for 20–75 s, and 10 s for >75 s. CHB-MIT values are exported in cohort metadata. No later event changes the window length.
4. The 333-dimensional dictionary contains 180 channel-wise physiology features and 153 Fisher-z pair correlations. Historical `geometry_tangent_corr` schema labels do not denote an SPD matrix logarithm, a tangent-space model or graph topology metrics.
5. Median and 1.4826 × raw MAD are fit only to the patient's non-seizure fit data. If scale is not >1e-4, use population standard deviation from the same fit data with a 1e-4 floor. Aligned coordinates are clipped to [-20,20]. The reference remains fixed.
6. All released positive windows determine feature reliability: absolute median background separation divided by `max(0.5 + 1.4826 × positive raw MAD, 0.5)`. Select the highest 50 dimensions with the original NumPy sort. Keep all positive history for reliability; retain at most 24 active matching exemplars by deterministic rounded, equally spaced row positions. The 24-row limit is not a bound on stored feature history or total RAM.
7. Score by maximum cosine similarity, with a 1e-6 norm-product floor. Recompute high/low quantiles on the fixed non-seizure calibration set after each update. The score is similarity, not a calibrated seizure probability. Extra projection/excess fields remain in the original class for numerical provenance but are not used by the primary detector.
8. The hysteresis state starts unarmed; consecutive low-score windows re-arm it, and consecutive high-score windows trigger an alarm at complete-window end, then reset it. Invalid windows and EDF boundaries clear the state. No model output supplies its own positive label.

## Confirmation availability and event scoring

Annotations simulate independent confirmations. Event exemplars are annotation-positive, center-selected complete windows (nearest-center fallback if necessary). They can include samples outside the annotated interval; input support ends before an exemplar is used. The immediate primary replay scores the complete window through an event boundary with the old model before updating. This does not certify a real clinician confirmation service.

The real-elapsed-time delay replay reads EDF header starts, preserves acquisition gaps and releases later-event histories after annotation end plus the fixed delay, or at the next task-anchored 8-hour review. A window can use released history only if its start is at or after release. The first seizure's initialization is held fixed. The release ledger records first actual use. Absolute epoch labels can differ with the machine's local timezone; a fixed-offset timezone without daylight saving is needed for reproducing the original relative schedules across dated records. The tested environment used Asia/Shanghai.

Alarm clustering: adjacent available points differ by at most 90 s, and first-to-last span is at most 300 s. Clusters and references are matched chronologically one-to-one, requiring an actual point in the inclusive [onset−30 s, offset+60 s] interval. Matching is not inferred solely from a cluster envelope. Sensitivity uses all post-first targets; precision uses matched alarm events/(matched+false alarm events); FAR uses each method's eligible monitored exposure. Latency is the earliest qualifying point minus onset, conditional on detection. Strict-ictal sensitivity uses the same detector but actual points in [onset,offset].

## Selection and uncertainty

The original 128-rule grid comprised high quantiles {0.997,0.998,0.999,0.9992}, persistence {2,3}, low quantiles {0.97,0.98,0.99,0.995} and reset seconds {10,20,30,45}. Inner feasibility required FAR ≤0.7/h and precision ≥0.25; sensitivity, F1, FAR and deterministic rule order resolved selection, with a relative-violation fallback.

Primary rules were selected before strict-preprocessing outer outcomes, on the historical timing-corrected cache. They are supplied explicitly. Running the release does not search the grid or reselect a more favorable rule on the strict cache. CHB-MIT is internal evaluation/frozen transfer, not pristine external validation. The 0.7/h value is an inner-selection constraint, not an enforced maximum FAR for every held-out subject.

Bootstrap resamples 21 independent-subject clusters, jointly including chb01/chb21, with 10,000 repetitions by default when requested. Pooled counts/exposure are reaggregated and repeated cluster multiplicity is retained for the macro mean. It conditions on saved out-of-fold predictions, without rerunning selection within each resample. Seeds for different archived analyses differ; matching a point estimate does not imply identical Monte Carlo confidence limits. `reference/confidence_intervals.json` uses the primary seed 20261310 and the existing corrected macro interval. Delay seeds begin at 20260909 and increment in schedule order.

For compatibility, some aggregate JSON fields retain historical names: `macro_patient_sensitivity`, `median_patient_sensitivity` and `patients_with_any_detection` are recording-stage summaries (22 CHB-MIT stages), not 21-person summaries. The manuscript's macro subject result uses `macro_independent_subject_sensitivity`. Do not reinterpret the stage count as the number of independent subjects with a detection. Undefined conditional latency or bootstrap endpoints serialize as JSON `null`, never as a zero latency.

## Coverage and checks

- `tests/` uses synthetic values, not patient data, to test numerical shapes, fallbacks, filter state, future invariance, confirmation ordering, alarm rules, portable I/O and resampling.
- `verify-results` reconstructs published aggregate counts and metrics from supplied ledgers. It does not process raw data.
- `evaluate` reconstructs detector traces from strict-causal feature caches. `verify-results --recomputed ...` additionally checks per-stage event IDs, latencies, false alarm counts and exposure against the frozen primary ledger.
- `build-cache` builds from EDFs without depending on a legacy feature cache. Check arrays when comparing environments; ZIP container hashes can vary even if their stored arrays are identical. Summation/BLAS and sort behavior can also vary near decision ties; the pinned environment is the reference.
- See `provenance/release_validation.json` for checks actually performed for this export. Configured CI is not a claim that remote CI already passed.

No workflow modifies original experimental artifacts, learns from held-out strict outcomes, or overwrites an existing output directory. New scientific changes require a separately named method version and protocol, not edits to these frozen resources.
