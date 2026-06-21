import unittest

from driftlab.simulator import DriftConfig, generate_frame, run_experiment


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


if __name__ == "__main__":
    unittest.main()
