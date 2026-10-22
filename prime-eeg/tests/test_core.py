from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from prime_eeg import io, preprocessing, representation, memory
from prime_eeg.cli import frozen_rule, new_output, verify_results
from prime_eeg.elapsed import cluster_bootstrap, release_schedule
from prime_eeg.evaluation import causal_alarm_events, overlaps_available_alarm, independent_subject
from prime_eeg.features import COMMON_CHANNELS, extract_fast_features


class CoreTests(unittest.TestCase):
    def test_frozen_resources_and_folds(self):
        self.assertGreater(io.verify_resources(), 20)
        folds = io.resource("frozen_rules.json")["folds"]
        self.assertEqual(len(folds), 21)
        cases = [case for fold in folds for case in fold["held_out_cases"]]
        self.assertEqual(len(set(cases)), 22)
        self.assertEqual(frozen_rule("chbmit", "chb01"), frozen_rule("chbmit", "chb21"))
        self.assertEqual(independent_subject("chb21"), "chb01+chb21")

    def test_robust_fallback_and_clipping(self):
        x = np.array([[0., 1., 0.], [0., 1., 0.], [9., 1., 0.]], dtype=np.float32)
        center, scale = representation.robust_parameters(x)
        np.testing.assert_array_equal(center, [0., 1., 0.])
        self.assertAlmostEqual(float(scale[0]), float(np.std(x[:, 0])), places=6)
        self.assertAlmostEqual(float(scale[1]), 1e-4, places=9)
        self.assertEqual(representation.align(np.array([[1e6, 1., 0.]]), center, scale)[0, 0], 20.)

    def test_feature_schema(self):
        raw = np.random.default_rng(7).normal(size=(18, 128 * 4 * 3))
        values, names = extract_fast_features(raw, 128, 4, COMMON_CHANNELS)
        self.assertEqual(values.shape, (3, 333))
        self.assertEqual(sum(n.startswith(("spectral_", "temporal_")) for n in names), 180)
        self.assertEqual(sum(n.startswith("geometry_") for n in names), 153)
        self.assertTrue(np.isfinite(values).all())

    def test_top50_and_max24(self):
        rng = np.random.default_rng(8)
        model = representation.fit_prototype(rng.normal(size=(100, 333)), rng.normal(size=(55, 333)), np.arange(333), 50)
        self.assertEqual(len(model.keep), 50)
        self.assertEqual(model.prototypes.shape, (24, 50))
        self.assertEqual(model.scores(np.zeros((2, 333)))["prototype"].tolist(), [0., 0.])
        self.assertAlmostEqual(float(model.scores(np.eye(333)[[0]])["prototype"][0]), float(np.max((np.eye(333)[[0]][:, model.keep] @ model.prototypes.T) / np.maximum(np.linalg.norm(np.eye(333)[[0]][:, model.keep], axis=1, keepdims=True) * np.linalg.norm(model.prototypes, axis=1)[None, :], 1e-6))), places=6)

    def test_hysteresis_initially_unarmed_and_invalid_resets(self):
        score = np.array([1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 1.])
        high, low = np.full(len(score), .8), np.full(len(score), .2)
        np.testing.assert_array_equal(memory.variable_hysteresis(score, high, low, 2, 2, np.ones(len(score), bool)), [5, 10])
        valid = np.ones(len(score), bool)
        valid[4] = False
        np.testing.assert_array_equal(memory.variable_hysteresis(score, high, low, 2, 2, valid), [10])

    def test_filter_is_causal_and_stateful(self):
        raw = np.random.default_rng(9).normal(size=(18, 256 * 16))
        changed = raw.copy()
        changed[:, 256 * 8:] += 500
        sos = preprocessing.filter_design()
        left, _ = preprocessing.causal_filter_downsample(raw, sos)
        right, _ = preprocessing.causal_filter_downsample(changed, sos)
        np.testing.assert_array_equal(left[:, :128 * 8], right[:, :128 * 8])
        first, state = preprocessing.causal_filter_downsample(raw[:, :256 * 8], sos)
        second, _ = preprocessing.causal_filter_downsample(raw[:, 256 * 8:], sos, state)
        np.testing.assert_array_equal(left, np.concatenate([first, second], axis=1))

    def test_contiguity_and_warmup(self):
        now = datetime(2020, 1, 1)
        self.assertFalse(preprocessing.is_contiguous(None, now))
        self.assertTrue(preprocessing.is_contiguous(now, now + timedelta(seconds=1 / 512)))
        self.assertFalse(preprocessing.is_contiguous(now, now + timedelta(seconds=1)))

    def test_actual_point_matching_not_cluster_envelope(self):
        # Test the point-matching predicate independently of clustering. A wide
        # envelope alone must not supply a fictitious intermediate alarm point.
        prediction = {"available_times": [10., 200.]}
        self.assertFalse(overlaps_available_alarm(prediction, {"start_seconds": 80, "end_seconds": 90}))
        self.assertTrue(overlaps_available_alarm(prediction, {"start_seconds": 40, "end_seconds": 50}))

    def test_cluster_boundaries(self):
        points = causal_alarm_events(np.array([0, 9, 18, 27, 30, 31, 50]), 10)
        self.assertEqual([x["alarm_count"] for x in points], [5, 1, 1])
        self.assertEqual(points[0]["first_available"], 10.)

    def test_paths_portable_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root / "x.npz", features=np.zeros((1, 333)), names=np.arange(333).astype(str))
            with patch.object(io, "CACHE_BASE", root):
                self.assertEqual(io.load_features("x.npz")[0].shape, (1, 333))
                with self.assertRaises(ValueError):
                    io.resolve_cache_file("../escape.npz")
            with self.assertRaises(FileExistsError):
                new_output(str(root))

    def test_bootstrap_keeps_repeated_subject_weights(self):
        rows = [{"subject": "chb02", "detected": 0, "total": 1, "false_alarms": 1, "monitoring_hours": 1., "latencies": []}, {"subject": "chb04", "detected": 1, "total": 1, "false_alarms": 1, "monitoring_hours": 1., "latencies": [3.]}]
        ci = cluster_bootstrap(rows, 42, 200)["ci95"]["macro_independent_subject_sensitivity"]
        self.assertEqual(ci, [0., 1.])

    def test_release_schedule_anchored_and_gaps_count(self):
        from types import SimpleNamespace
        source = SimpleNamespace(manifest={"events": [{"event_id": "S1"}, {"event_id": "S2"}]})
        events = {"S2": {"record": "r", "annotation_end_epoch_seconds": 1100.}}
        fixed = release_schedule(source, events, 100., "fixed", 300.)
        batch = release_schedule(source, events, 100., "batch", 28800.)
        self.assertEqual(fixed[0]["release_epoch_seconds"], 1400.)
        self.assertEqual(batch[0]["release_epoch_seconds"], 28900.)

    def test_confirmation_cannot_change_earlier_scores(self):
        from types import SimpleNamespace
        rng = np.random.default_rng(10)
        arrays = {"first": rng.normal(size=(4, 60)), "later": rng.normal(size=(4, 60)), "record": rng.normal(size=(20, 60))}
        names = [f"f{i}" for i in range(60)]
        source = SimpleNamespace(dataset="synthetic", subject="synthetic", names=names,
            center=np.zeros(60), scale=np.ones(60), fit_raw=[rng.normal(size=(20, 60))], calibration_aligned=rng.normal(size=(12, 60)),
            manifest={"window_seconds": 2, "events": [{"event_id": "S1", "cache_file": "first"}, {"event_id": "S2", "cache_file": "later"}], "monitoring_records": [{"record": "r", "cache_file": "record", "events": [{"event_id": "S2", "start_seconds": 12., "end_seconds": 20.}]}]})
        def loader(path):
            return arrays[path], names, {"labels": np.ones(len(arrays[path]), bool)}
        with patch.object(representation, "load_features", loader):
            before = memory.build_incremental_payload(source)
            arrays["later"] = arrays["later"] + 100
            after = memory.build_incremental_payload(source)
        np.testing.assert_array_equal(before["records"][0]["score"][:10], after["records"][0]["score"][:10])
        for q in before["records"][0]["thresholds"]:
            np.testing.assert_array_equal(before["records"][0]["thresholds"][q][:10], after["records"][0]["thresholds"][q][:10])


if __name__ == "__main__":
    unittest.main()
