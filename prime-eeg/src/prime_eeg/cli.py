"""Explicit, non-overwriting reproduction commands for the frozen detector."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import METHOD_IDENTIFIER, io
from .evaluation import aggregate_independent, evaluate_timing_corrected, rule

VARIANTS = ("main", "no-update", "physiology-only", "connectivity-only", "no-hysteresis", "alarm-gated")
DELAY_SPECS = (
    ("immediate", "fixed", 0.0), ("five_minutes", "fixed", 300.0),
    ("thirty_minutes", "fixed", 1800.0), ("sixty_minutes", "fixed", 3600.0),
    ("next_task_anchored_8h_review", "batch", 28800.0),
)


def new_output(path: str) -> Path:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"Use a new output directory (nothing was overwritten): {target}")
    target.mkdir(parents=True)
    return target


def chosen_subjects(dataset: str, requested, available=None) -> list[str]:
    cohort = io.resource("cohort.json")[dataset]
    selected = list(requested) if requested else list(available or cohort)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Specify at least one subject, without duplicates")
    if set(selected) - set(cohort):
        raise ValueError(f"Subjects outside the frozen {dataset} cohort: {sorted(set(selected) - set(cohort))}")
    if available is not None and set(selected) - set(available):
        raise ValueError("Requested subjects are absent from the cache index")
    return selected


def attach_cache(args):
    from . import representation
    root = Path(args.cache).resolve()
    index = io.read_json(root / "index.json")
    if not index.get("all_future_perturbation_tests_pass", False):
        raise ValueError("Cache has no passing future-perturbation certificate")
    subjects = chosen_subjects(args.dataset, args.subjects, index["eligible_subjects"])
    for subject in subjects:
        manifest = io.read_json(root / subject / "manifest.json")
        expected = {"chbmit": "V48_STRICT_CAUSAL_CHBMIT_CACHE", "siena": "V48_SIENA_STRICT_CAUSAL_CACHE"}
        if manifest["status"] != expected[args.dataset]:
            raise ValueError("Only strict-causal feature caches are accepted")
        if manifest["future_perturbation_invariance"]["status"] != "PASS":
            raise ValueError(f"Causality check failed for {subject}")
    io.CACHE_BASE = root
    representation.DATASETS[args.dataset] = root
    return subjects


def frozen_rule(dataset, subject):
    if dataset == "siena":
        return rule(io.resource("siena_adapter.json")["locked_alarm_configuration"])
    matches = [fold for fold in io.resource("frozen_rules.json")["folds"] if subject in fold["held_out_cases"]]
    if len(matches) != 1:
        raise ValueError(f"No unique frozen fold rule for {subject}")
    return rule(matches[0]["selected_configuration"])


def build_cache(args):
    from . import preprocessing, siena_common, siena_preprocessing
    data_root = Path(args.data_root).resolve(strict=True)
    subjects = chosen_subjects(args.dataset, args.subjects)
    for subject in subjects:
        if not (data_root / subject).is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {data_root / subject}")
    output = new_output(args.output)
    builder = preprocessing if args.dataset == "chbmit" else siena_preprocessing
    builder.DATA_ROOT, builder.CACHE_ROOT = data_root, output
    siena_common.DATA_ROOT = data_root
    rows = []
    for subject in subjects:
        row = builder.build_subject(subject)
        io.portable_manifest(output / subject / "manifest.json", output)
        rows.append(row)
        print(json.dumps(row), flush=True)
    io.write_json(output / "index.json", {
        "status": "V48_STRICT_CAUSAL_CACHE_COMPLETE", "dataset": args.dataset,
        "eligible_subjects": subjects, "subjects": rows,
        "full_frozen_cohort": set(subjects) == set(io.resource("cohort.json")[args.dataset]),
        "all_future_perturbation_tests_pass": all(row["future_perturbation_invariance"] == "PASS" for row in rows),
        "method_identifier": METHOD_IDENTIFIER,
    })


def evaluate(args):
    from . import representation, memory, ablations, alert_gated, sensitivity
    from .elapsed import cluster_bootstrap
    subjects = attach_cache(args)
    output = new_output(args.output)
    rows = []
    for index, subject in enumerate(subjects, 1):
        print(f"{index}/{len(subjects)} {subject}: {args.variant}", flush=True)
        source = representation.load_source(args.dataset, subject)
        current_rule = frozen_rule(args.dataset, subject)
        if args.variant == "alarm-gated":
            row = alert_gated.alert_gated_subject(source, current_rule)
            # The historical update log records annotation/alarm time rather than
            # full-window availability; it is not exported as a release-time ledger.
            row.pop("update_log", None)
        else:
            if args.variant == "main":
                payload = memory.build_incremental_payload(source)
            else:
                view = {"physiology-only": "physiology", "connectivity-only": "geometry"}.get(args.variant, "all")
                payload = ablations.build_payload(source, view=view, update_enabled=args.variant != "no-update")
            if args.variant == "no-hysteresis":
                row = ablations.no_hysteresis_evaluate(payload, current_rule)
            elif args.strict_ictal:
                row = sensitivity.evaluate(payload, current_rule, preictal_seconds=0, postictal_seconds=0)
            else:
                row = evaluate_timing_corrected(payload, current_rule)
        rows.append(row)
    result = aggregate_independent(rows)
    payload = {
        "method_identifier": METHOD_IDENTIFIER, "dataset": args.dataset,
        "variant": args.variant, "strict_ictal": args.strict_ictal,
        "full_frozen_cohort": set(subjects) == set(io.resource("cohort.json")[args.dataset]),
        "rule_selection": "Previously frozen rules; no selection on these outcomes",
        "result": result, "patients": rows,
    }
    if args.bootstrap:
        payload["bootstrap"] = cluster_bootstrap(rows, args.seed, args.bootstrap)
    io.write_json(output / "result.json", payload)
    print(json.dumps(result, indent=2))


def delay(args):
    from . import representation, preprocessing, sensitivity
    from .elapsed import record_clock, release_schedule, elapsed_payload, cluster_bootstrap
    subjects = attach_cache(args)
    preprocessing.DATA_ROOT = Path(args.data_root).resolve(strict=True)
    output = new_output(args.output)
    analyses = {name: {"mode": mode, "nominal_seconds": amount, "patients": []} for name, mode, amount in DELAY_SPECS}
    ledgers = {}
    for index, subject in enumerate(subjects, 1):
        print(f"{index}/{len(subjects)} {subject}: EDF elapsed-time delay", flush=True)
        source = representation.load_source("chbmit", subject)
        prepared = sensitivity.prepare_source(source)
        clocks, events, origin = record_clock(source, prepared)
        for name, mode, amount in DELAY_SPECS:
            schedule = release_schedule(source, events, origin, mode, amount)
            payload, ledger = elapsed_payload(prepared, clocks, schedule)
            analyses[name]["patients"].append(sensitivity.evaluate(payload, frozen_rule("chbmit", subject)))
            ledgers[f"{subject}/{name}"] = ledger
    for index, (name, _, _) in enumerate(DELAY_SPECS):
        rows = analyses[name]["patients"]
        analyses[name]["result"] = aggregate_independent(rows)
        if args.bootstrap:
            analyses[name]["bootstrap"] = cluster_bootstrap(rows, args.seed + index, args.bootstrap)
    io.write_json(output / "result.json", {"method_identifier": METHOD_IDENTIFIER, "analyses": analyses, "full_frozen_cohort": set(subjects) == set(io.resource("cohort.json")["chbmit"])})
    io.write_json(output / "release_ledger.json", ledgers)
    print(json.dumps({k: v["result"] for k, v in analyses.items()}, indent=2))


def check_metrics(actual, expected, label):
    keys = ("detected", "total", "false_alarms", "matched_predictions", "monitoring_hours", "event_sensitivity", "far_per_hour", "event_precision", "event_f1", "macro_independent_subject_sensitivity", "median_latency_seconds")
    for key in keys:
        if key not in expected:
            continue
        left, right = actual[key], expected[key]
        if left is None or right is None:
            if left is not right:
                raise AssertionError(f"{label}: {key} differs")
        elif not math.isclose(left, right, rel_tol=0, abs_tol=1e-10):
            raise AssertionError(f"{label}: {key}: {left} != {right}")


def verify_results(args):
    references = Path(args.reference).resolve()
    count = io.verify_resources()
    main = io.read_json(references / "main.json")
    check_metrics(aggregate_independent(main["patients"]), main["result"], "main")
    for filename in ("ablations.json", "elapsed_delay.json"):
        for name, entry in io.read_json(references / filename).items():
            check_metrics(aggregate_independent(entry["patients"]), entry["result"], f"{filename}:{name}")
    for filename in ("siena.json", "alarm_gated.json"):
        entry = io.read_json(references / filename)
        check_metrics(aggregate_independent(entry["patients"]), entry["result"], filename)
    for name, entry in io.read_json(references / "tolerance_sensitivity.json").items():
        check_metrics(aggregate_independent(entry["patients"]), entry["result"], f"tolerance:{name}")
    baselines = io.read_json(references / "baseline_comparison.json")
    for name, entry in baselines["archived_harmonized_methods"].items():
        rows = [{**row, "negative_latency_detections": sum(value < 0 for value in row["latencies"])} for row in entry["rows"]]
        check_metrics(aggregate_independent(rows), entry["result"], f"baseline:{name}")
    for name, entry in baselines["postfreeze_sequential_raw_repair"].items():
        rows = [{**row, "negative_latency_detections": sum(value < 0 for value in row["latencies"])} for row in entry["patients"]]
        check_metrics(aggregate_independent(rows), entry["result"], f"raw-repaired:{name}")
    if args.recomputed:
        entry = io.read_json(Path(args.recomputed))
        if entry.get("variant") != "main" or entry.get("dataset") != "chbmit" or entry.get("strict_ictal"):
            raise ValueError("--recomputed must be the full, main CHB-MIT evaluation")
        check_metrics(entry["result"], main["result"], "recomputed main")
        archived = {row["subject"]: row for row in main["patients"]}
        if set(archived) != {row["subject"] for row in entry["patients"]}:
            raise AssertionError("Full-cohort subject set differs")
        for row in entry["patients"]:
            ref = archived[row["subject"]]
            for key in ("detected_event_ids", "target_event_ids", "false_alarms", "monitoring_hours"):
                if row[key] != ref[key]:
                    raise AssertionError(f"{row['subject']}: {key} differs")
            # Original lists were built from sets: order is process-dependent and
            # they are not event-ID-to-latency maps. Compare multisets instead.
            if sorted(row["latencies"]) != sorted(ref["latencies"]):
                raise AssertionError(f"{row['subject']}: latency multiset differs")
    print(f"PASS: {count} frozen resources; main, ablations, confirmation schedules, tolerance, Siena and neural-baseline ledger aggregates.")
    print("This validates archived result arithmetic; raw-data recomputation requires build-cache/evaluate.")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="prime-eeg", description="Reproduce the frozen PRIME-EEG detector without retuning")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-cache", help="Build a new strict-causal feature cache from EDFs")
    build.add_argument("--dataset", choices=("chbmit", "siena"), default="chbmit")
    build.add_argument("--data-root", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--subjects", nargs="+")
    build.set_defaults(action=build_cache)
    evaluate_parser = sub.add_parser("evaluate", help="Apply previously frozen rules; no parameter selection")
    evaluate_parser.add_argument("--dataset", choices=("chbmit", "siena"), default="chbmit")
    evaluate_parser.add_argument("--cache", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--subjects", nargs="+")
    evaluate_parser.add_argument("--variant", choices=VARIANTS, default="main")
    evaluate_parser.add_argument("--strict-ictal", action="store_true", help="Require an actual alarm within the annotated ictal interval")
    evaluate_parser.add_argument("--bootstrap", type=int, default=0, metavar="N")
    evaluate_parser.add_argument("--seed", type=int, default=20261310)
    evaluate_parser.set_defaults(action=evaluate)
    delays = sub.add_parser("delay", help="Confirmation-release sensitivity on the EDF elapsed-time clock")
    delays.add_argument("--dataset", choices=("chbmit",), default="chbmit")
    delays.add_argument("--cache", required=True)
    delays.add_argument("--data-root", required=True)
    delays.add_argument("--output", required=True)
    delays.add_argument("--subjects", nargs="+")
    delays.add_argument("--bootstrap", type=int, default=0, metavar="N")
    delays.add_argument("--seed", type=int, default=20260909)
    delays.set_defaults(action=delay)
    verify = sub.add_parser("verify-results", help="Validate reference result arithmetic and frozen resources")
    verify.add_argument("--reference", default="reference")
    verify.add_argument("--recomputed", help="Also compare a full recomputed main result, including event IDs")
    verify.set_defaults(action=verify_results)
    from .baseline_cli import replay
    baseline = sub.add_parser("baseline-replay", help="Replay locked comparison rules from existing neural-score caches; no inference")
    baseline.add_argument("--raw-cache", required=True)
    baseline.add_argument("--official-cache", required=True)
    baseline.add_argument("--domain", choices=("native", "common"), default="native")
    baseline.add_argument("--subjects", nargs="+")
    baseline.add_argument("--output", required=True)
    baseline.set_defaults(action=replay)
    args = parser.parse_args(argv)
    if getattr(args, "bootstrap", 0) < 0:
        parser.error("--bootstrap must be nonnegative")
    if getattr(args, "strict_ictal", False) and args.variant in ("no-hysteresis", "alarm-gated"):
        parser.error("--strict-ictal is not implemented for this ablation")
    if getattr(args, "dataset", "chbmit") == "siena" and getattr(args, "variant", "main") != "main":
        parser.error("Siena stress reproduction uses the fixed main configuration only")
    io.verify_resources()
    args.action(args)
