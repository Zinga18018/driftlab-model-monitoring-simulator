import unittest

from driftlab.benchmark import binomial_interval, run_benchmark
from driftlab.simulator import DriftConfig
from scripts_no_drift import summarize, trial


class NoDriftExtensionTests(unittest.TestCase):
    def test_trial_matches_existing_benchmark_record(self):
        baseline = run_benchmark(DriftConfig(rows=2000, drift_strength=1.15), [1000, 1001])
        self.assertEqual(trial(1000), baseline['trials'][0])
        self.assertEqual(trial(1001), baseline['trials'][1])

    def test_summary_counts_every_alert_type_including_nonzero(self):
        records = [dict(any_alert=True, auc_drop=None, alert_flags=dict(
            distribution_drift=True, model_degradation=False, positive_rate_shift=False)),
            dict(any_alert=True, auc_drop=.1, alert_flags=dict(
            distribution_drift=False, model_degradation=True, positive_rate_shift=True)),
            dict(any_alert=False, auc_drop=0, alert_flags=dict(
            distribution_drift=False, model_degradation=False, positive_rate_shift=False))]
        result = summarize(records)
        self.assertEqual(result['unavailable_auc_comparisons'], 1)
        self.assertEqual(result['alert_rates']['any_alert']['count'], 2)
        for name in ('distribution_drift', 'model_degradation', 'positive_rate_shift'):
            self.assertEqual(result['alert_rates'][name]['count'], 1)
        self.assertEqual(result['alert_rates']['any_alert']['wilson_95_interval'], binomial_interval(2, 3))

    def test_zero_of_ten_thousand_interval_is_not_zero(self):
        lo, hi = binomial_interval(0, 10000)
        self.assertAlmostEqual(lo, 0)
        self.assertAlmostEqual(hi, 0.00038399837067659573)
