# No-drift extension: 10,000 fresh seeds

Run completed 2026-10-28T21:45:17Z against source commit `1a33b24cd9a1597cebf3378197e42d9a84e24e00`.

**Observed any-alert false alerts: 0/10,000 (0.000000%). Two-sided Wilson 95% interval: [0.000000%, 0.038400%].** The original 0/40 upper endpoint was 8.7622%; the new upper endpoint is 0.038400%, about 228 times smaller. These are separate samples, not 10,040 trials silently combined.

## Fixed protocol

- 10,000 fresh, distinct pseudorandom seeds **1040–11039 inclusive**; no overlap with original seeds 1000–1039.
- 2,000 rows in **each** training, held-out reference, and current window; fresh fitted model per seed. Three independent `SeedSequence` child streams within each experiment.
- `drift_mode="none"`, strength 0, label noise 0.08. The simulator and benchmark modules are unchanged from the source commit. No tuning, filtering, or removal of alerts.
- Distribution alert: any feature PSI >=0.20. Degradation alert: reference AUC minus current AUC >=0.08. Positive-rate alert: absolute current-minus-reference label-positive rate >0.08.
- **Any alert** is the union of these three flags. KS is diagnostic only. A run with several flags counts once in the union.
- Four processes, one BLAS thread each; original experiment function called directly for every seed. Progress/data flushed every100 records.
- `metadata.json`'s `config` is a **template**: its default seed42 is overridden by each actual trial seed. `start_seed`, `end_seed_inclusive`, and every JSONL record identify actual seeds; this is not 10,000 repetitions of seed42.

| Event | Count / trials | Observed rate | Wilson95% interval |
|---|---:|---:|---:|
| distribution_drift | 0/10000 | 0.000000% | [0.000000%, 0.038400%] |
| model_degradation | 0/10000 | 0.000000% | [0.000000%, 0.038400%] |
| positive_rate_shift | 0/10000 | 0.000000% | [0.000000%, 0.038400%] |
| any_alert | 0/10000 | 0.000000% | [0.000000%, 0.038400%] |

No unavailable AUC comparisons: 0. Runtime recorded by runner: 139.37s.

## Reproduce and verify

Python **3.12.14**, NumPy **2.3.5**, pandas **2.2.3**. Existing environment used; no packages installed. Relevant dependency versions, source SHA256 hashes, and trials SHA256 are alongside this report.

```bash
python -m pip install numpy==2.3.5 pandas==2.2.3
python scripts_no_drift.py --runs 10000 --start-seed 1040 --workers 4 --output-dir outputs/reproduced_no_drift_10000
python -m unittest discover -s tests -v
```

The simulator-only runner needs NumPy and pandas; full tests additionally need Streamlit (1.66.1 in this verification environment). Use a fresh output directory: existing trial evidence is not overwritten. `trials.jsonl` contains one record per seed, with config, split fingerprints/class counts, metrics, all flags, and any-alert status. `summary.json` contains precise counts and intervals. The interval uses the unchanged `binomial_interval` implementation (z=1.959963984540054).

**21/21 tests passed**, including the original Streamlit smoke test, a new runner-vs-original-benchmark equality test, mixed/nonzero alert counting, and zero-event interval precision. Additional checks: serial and two-worker trial files identical for seeds1040–1041; existing-output refusal preserved the file; `git diff --check` passed. Core source files were verified unchanged.

The original **200-trial** benchmark was rerun: protocol and all five scenario summaries match exactly; all trial metrics, alert flags and counts match, including **0/40 no-drift alerts**. 195 trial records match fully. The only five differences are the training CSV fingerprint for seed1017 reused across the five scenarios. The current environment differs from the archive; the cause of that fingerprint difference was not established. We do **not** claim byte-identical baseline reproduction. See `baseline_verification.json`.

## Interpretation and limits

More independent synthetic trials substantially narrowed the binomial interval under this fixed generating mechanism. Zero observed alerts does not mean zero risk. This interval is across independently seeded model/window experiments, not repeated monitoring of one fixed production model. It does not establish production false alarms per day, account for overlapping windows or temporal dependence, or cover delayed labels, seasonality, missing data, and distribution changes absent from this simulator. These are synthetic development experiments, not a locked holdout or a production guarantee. The decision to extend the prior run followed observing the prior result; the fresh 10,000 sample is reported separately.

## Publication status

Prepared for `Zinga18018/driftlab-model-monitoring-simulator`, but **not published**: the active GitHub connection identifies `CharishmaReddyYerra` and reports `push=false` for the target repository. No remote write or CI run for this change occurred. A connection with write access to the target repository is required; local test success is not remote CI success.
