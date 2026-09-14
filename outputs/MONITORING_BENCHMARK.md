# Repeated synthetic monitoring benchmark

40 independent seeds per scenario; 2,000 rows in each training, held-out reference, and current window.
PSI threshold 0.20; AUC-drop threshold 0.08; absolute positive-rate-change threshold > 0.08. Thresholds were fixed before this run, with no tuning.

| Scenario | Mean max PSI | Mean AUC drop | Distribution alerts | Degradation alerts | Any alert |
|---|---:|---:|---:|---:|---:|
| none | 0.0140 | -0.0003 | 0/40 | 0/40 | 0/40 |
| mean_shift | 0.9054 | 0.0032 | 40/40 | 0/40 | 40/40 |
| variance_shift | 0.2869 | -0.0525 | 40/40 | 0/40 | 40/40 |
| class_prior_shift | 0.5401 | 0.0134 | 40/40 | 0/40 | 40/40 |
| mixed | 0.5392 | 0.4341 | 40/40 | 40/40 | 40/40 |

No-drift false-alert rate (any alert): **0/40 = 0.0%**, Wilson 95% interval **[0.0%, 8.8%]**.

A positive AUC drop means performance fell. Feature drift can occur while AUC stays stable or improves; a distribution alert is not proof of model degradation.

These estimates describe independent synthetic windows at the stated sample size. They do not estimate production false alarms per day, repeated testing over overlapping windows, or delayed-label behavior. KS p-values are approximate diagnostics and do not drive alerts. Single-class AUC is unavailable and excluded from AUC means; counts are recorded in the JSON.

See [full trials, split fingerprints, class counts, seeds, thresholds, and source hashes](monitoring_benchmark.json). These runs are development evidence, not a locked benchmark or a claim of untouched evaluation data.
