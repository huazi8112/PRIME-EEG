"""Replay locked neural comparison rules from separately prepared score caches."""
from pathlib import Path

import numpy as np

from . import io
from .evaluation import aggregate_independent

METHODS = ("public-default", "nested-postprocessing", "nested-raw-repaired", "fixed-0.5")


def replay(args):
    from . import baseline_raw as raw, baseline_official as official
    from . import baseline_repair as repair, baseline_scoring as scoring
    from .cli import chosen_subjects, new_output
    subjects = chosen_subjects("chbmit", args.subjects)
    raw_root, official_root = Path(args.raw_cache).resolve(strict=True), Path(args.official_cache).resolve(strict=True)
    for root in (raw_root, official_root):
        for subject in subjects:
            if not (root / subject / "manifest.json").is_file():
                raise FileNotFoundError(f"Missing score-cache manifest: {root / subject}")
    output = new_output(args.output)
    repair.RAW_CACHE, official.CACHE = raw_root, official_root
    rules = io.resource("baseline_rules.json")
    registry = io.resource("baseline_exposure.json")
    raw_selected = {case: fold["selected_configuration"] for fold in rules["raw_repaired"][args.domain] for case in fold["held_out_cases"]}
    official_selected = {case: fold["selected_configuration"] for fold in rules["official"][f"official_nested_{args.domain}_reselected"] for case in fold["held_out_cases"]}
    results = {name: {"patients": []} for name in METHODS}
    for index, subject in enumerate(subjects, 1):
        print(f"{index}/{len(subjects)} {subject}: locked neural baseline replay", flush=True)
        native_raw = {record: entry["raw"] for record, entry in registry[subject].items()}
        raw_hours = sum(scoring.duration(intervals) for intervals in native_raw.values()) / 3600
        raw_payload = repair.make_repaired_payload(subject, raw_hours)
        official_payload = official.make_payload(subject, 0.)
        configurations = {
            "public-default": {"threshold": .8, "minimum_event_seconds": 2.},
            "nested-postprocessing": official_selected[subject],
            "nested-raw-repaired": raw_selected[subject],
            "fixed-0.5": {"representation": "maximum", "threshold_mode": "absolute", "threshold_value": .5, "persistence": 1},
        }
        for name, configuration in configurations.items():
            family = "official" if name in METHODS[:2] else "raw"
            payload = official_payload if family == "official" else raw_payload
            points = scoring.official_points(payload, configuration) if family == "official" else scoring.raw_points(payload, configuration, {})
            exposure_key = family if args.domain == "native" else "common"
            exposures = {record: entry[exposure_key] for record, entry in registry[subject].items()}
            hours = sum(scoring.duration(intervals) for intervals in exposures.values()) / 3600
            records = scoring.case_records(subject, family, payload)
            row = scoring.score(subject, records, points, hours, args.domain, exposures=exposures, details=True)
            row["negative_latency_detections"] = sum(value < 0 for value in row["latencies"])
            results[name]["patients"].append(row)
    for entry in results.values():
        entry["result"] = aggregate_independent(entry["patients"])
    io.write_json(output / "result.json", {
        "scope": "Cached neural outputs, locked selected rules, actual-point harmonized scorer; no neural inference or reselection",
        "domain": args.domain, "full_frozen_cohort": set(subjects) == set(io.resource("cohort.json")["chbmit"]),
        "methods": results,
    })
    import json
    print(json.dumps({key: entry["result"] for key, entry in results.items()}, indent=2))
