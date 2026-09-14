import unittest
import json

import numpy as np
import pandas as pd

from driftlab.benchmark import binomial_interval, run_benchmark
from driftlab.simulator import (DRIFT_MODES, FEATURES, DriftConfig, alert_flags, binary_metrics,
                               generate_frame, ks_test_approx, population_stability_index,
                               predict_proba, roc_auc, run_experiment, serializable_result)


class DriftLabTests(unittest.TestCase):
    def test_run_experiment_returns_metrics_and_drift_table(self):
        result = run_experiment(DriftConfig(rows=1200, drift_strength=1.1, drift_mode="mixed", seed=7))

        self.assertEqual(len(result["reference"]), 1200)
        self.assertEqual(len(result["current"]), 1200)
        self.assertEqual(len(result["drift_table"]), 4)
        self.assertIn("auc", result["reference_metrics"])
        self.assertIn("auc", result["current_metrics"])
        self.assertGreaterEqual(result["reference_metrics"]["auc"], 0.5)
        self.assertTrue(result["alerts"])

    def test_stronger_drift_increases_top_psi(self):
        low = run_experiment(DriftConfig(rows=1500, drift_strength=0.2, drift_mode="mean_shift", seed=11))
        high = run_experiment(DriftConfig(rows=1500, drift_strength=1.8, drift_mode="mean_shift", seed=11))

        self.assertGreater(high["drift_table"]["psi"].max(), low["drift_table"]["psi"].max())

    def test_label_noise_keeps_binary_labels(self):
        frame = generate_frame(rows=500, rng=__import__("numpy").random.default_rng(3), label_noise=0.25)

        self.assertEqual(set(frame["label"].unique()).issubset({0, 1}), True)

    def test_training_and_reference_are_independent_and_scaling_uses_training_only(self):
        result = run_experiment(DriftConfig(rows=200, train_rows=300, seed=21))
        manifest = result["split_manifest"]
        self.assertEqual([manifest[name]["rows"] for name in ("training", "reference", "current")], [300, 200, 200])
        self.assertEqual(len({entry["sha256_csv"] for entry in manifest.values()}), 3)
        self.assertEqual([entry["used_for_fitting"] for entry in manifest.values()], [True, False, False])
        model = result["model"]
        np.testing.assert_allclose(model["feature_means"], result["training"][FEATURES].mean().to_numpy())
        np.testing.assert_allclose(model["feature_stds"], result["training"][FEATURES].std(ddof=0).to_numpy())
        probabilities = predict_proba(result["reference"][FEATURES], np.array(model["weights"]),
                                      np.array(model["feature_means"]), np.array(model["feature_stds"]))
        self.assertEqual(result["reference_metrics"], binary_metrics(result["reference"]["label"], probabilities))

    def test_current_scenario_cannot_change_training_or_reference(self):
        low = run_experiment(DriftConfig(rows=200, drift_mode="none", seed=10))
        high = run_experiment(DriftConfig(rows=200, drift_mode="mixed", drift_strength=2.0, seed=10))
        pd.testing.assert_frame_equal(low["training"], high["training"])
        pd.testing.assert_frame_equal(low["reference"], high["reference"])
        self.assertEqual(low["model"], high["model"])
        self.assertEqual(low["reference_metrics"], high["reference_metrics"])

    def test_zero_strength_has_identical_label_rule_in_every_scenario(self):
        reference = generate_frame(150, np.random.default_rng(71), drift_mode="none", label_noise=0.08)
        for mode in DRIFT_MODES:
            with self.subTest(mode=mode):
                frame = generate_frame(150, np.random.default_rng(71), drift_mode=mode, drift_strength=0, label_noise=0.08)
                pd.testing.assert_frame_equal(reference, frame)

    def test_seed_reproduces_full_serialized_result(self):
        config = DriftConfig(rows=100, seed=4)
        self.assertEqual(serializable_result(run_experiment(config)), serializable_result(run_experiment(config)))

    def test_single_class_auc_is_unavailable_and_serializes_to_null(self):
        metrics = binary_metrics(pd.Series([1, 1, 1]), np.array([0.2, 0.5, 0.8]))
        self.assertIsNone(metrics["auc"])
        self.assertFalse(metrics["auc_defined"])
        self.assertIn('"auc": null', json.dumps(metrics, allow_nan=False))
        drift = pd.DataFrame([{"feature": "signal", "psi": 0.0}])
        flags = alert_flags(metrics, metrics, drift, DriftConfig())
        self.assertFalse(flags["model_degradation"])
        self.assertFalse(any(flags.values()))

    def test_auc_ties_and_perfect_ranking(self):
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(roc_auc(labels, np.ones(4)), 0.5)
        self.assertEqual(roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(roc_auc(labels, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)

    def test_constant_reference_detects_change(self):
        ref = pd.Series([0.0] * 20)
        self.assertEqual(population_stability_index(ref, ref), 0.0)
        self.assertGreater(population_stability_index(ref, pd.Series([1.0] * 20)), 0.2)

    def test_invalid_samples_are_rejected_instead_of_producing_nan(self):
        for labels, scores in (([], []), ([0], [0.1, 0.2]), ([2], [0.2]), ([0], [np.nan]), ([0], [1.5])):
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                binary_metrics(pd.Series(labels), np.array(scores))
        for func in (population_stability_index, ks_test_approx):
            with self.assertRaises(ValueError):
                func(pd.Series([], dtype=float), pd.Series([1.0]))

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({"rows": 0}, {"rows": 2.5}, {"train_rows": 0}, {"seed": -1},
                       {"label_noise": 1.1}, {"drift_mode": "unknown"}, {"drift_strength": -1},
                       {"psi_alert_threshold": float("nan")}, {"auc_drop_threshold": -0.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DriftConfig(**kwargs)

    def test_small_windows_do_not_fabricate_auc(self):
        result = serializable_result(run_experiment(DriftConfig(rows=2, seed=2)))
        json.dumps(result, allow_nan=False)
        for name in ("reference_metrics", "current_metrics"):
            metrics = result[name]
            self.assertEqual(metrics["auc_defined"], metrics["positive_count"] > 0 and metrics["negative_count"] > 0)

    def test_no_alert_message_does_not_count_as_an_alert(self):
        result = run_experiment(DriftConfig(rows=1000, drift_mode="none", seed=20))
        self.assertEqual(result["alert_count"], sum(result["alert_flags"].values()))
        if result["alerts"][0].startswith("No major"):
            self.assertEqual(result["alert_count"], 0)

    def test_benchmark_contains_null_baseline_and_independent_seed_counts(self):
        result = run_benchmark(DriftConfig(rows=80), [5, 6])
        self.assertEqual(result["scenario_summary"][0]["scenario"], "none")
        self.assertEqual(len(result["trials"]), 2 * len(DRIFT_MODES))
        for summary in result["scenario_summary"]:
            selected = [r for r in result["trials"] if r["scenario"] == summary["scenario"]]
            rate = summary["alert_rates"]["any_alert"]
            self.assertEqual(rate["count"], sum(r["any_alert"] for r in selected))
            self.assertEqual(rate["trials"], 2)
        json.dumps(result, allow_nan=False)

    def test_zero_observed_false_alerts_still_have_positive_upper_bound(self):
        lower, upper = binomial_interval(0, 40)
        self.assertAlmostEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)
        self.assertLess(upper, 0.10)

    def test_duplicate_benchmark_seeds_are_rejected(self):
        with self.assertRaises(ValueError):
            run_benchmark(DriftConfig(rows=100), [1, 1])


if __name__ == "__main__":
    unittest.main()
