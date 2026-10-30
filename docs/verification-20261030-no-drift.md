# No-drift follow-up verification, 2026-10-30

This check prepares the archived 2026-10-28 experiment for publication. It audits the saved 10,000 records and verifies the runner on Windows; it **does not rerun all 10,000 simulations**. [Machine-readable results](verification-20261030-no-drift.json).

## Archive and protocol audit

- The prepared ZIP matched SHA256 `ac511199a3697976b1ac63b49c6ca85f3e818aaeb11f55255132edd68ed3c111`, size 2,822,166 bytes, and passed ZIP CRC checks before extraction.
- `trials.jsonl` contains 10,000 records, 15,039,116 bytes, and SHA256 `5c184fc397c4d9d6037ca7f3fe96233b7e59a96d00655031f771adc2fc991679`, matching the recorded metadata.
- Seeds 1040-11039 are distinct, ordered, and separate from the original seeds 1000-1039. The manifests contain 30,000 distinct window fingerprints, three 2,000-row windows per trial, expected stream keys, and training-only fitting flags. Recorded class and confusion counts are internally consistent.
- Every record retains the fixed no-drift configuration and thresholds. Recomputing distribution, AUC-degradation, and positive-rate flags from the saved metrics gives **0/10,000** for each flag and their union. No AUC comparison is unavailable.
- An independent Wilson calculation with `z = 1.959963984540054` gives **0%-0.03839983706765957%**, agreeing with the saved summaries. It describes the synthetic seeded experiment, not production false alarms per day.
- Simulator, benchmark, and new-runner hashes match the archive metadata. Existing simulator code, the historical 200-trial evidence, app, and CI workflow remain unchanged.

This recomputes alert decisions from archived metrics, not the underlying raw-data AUC or PSI. Distinct recorded fingerprints and matching source hashes support the documented protocol; they do not independently prove statistical independence.

## Fresh local checks

Environment: Windows 11; Python 3.12.14, NumPy 2.5.0, pandas 3.0.3, Streamlit 1.58.0. This differs from the archived Linux run, which used NumPy 2.3.5 and pandas 2.2.3.

- `python -B -m unittest discover -s tests -v`: **21/21 passed**, including the Streamlit default/no-drift render check and three new runner/counting/interval tests.
- Fresh seeds 1040-1041 run with one worker and with two workers produced byte-identical trial files and zero alerts. These two-seed files were written outside the checked-in evidence directory.
- Reusing an existing output directory was rejected with exit code 2, and the existing trial-file hash remained unchanged.
- Compared with the archived records for those two seeds, all recorded metrics and alert flags match. Seed 1040 has a different training CSV fingerprint; other fields match, and seed 1041 matches fully. The cause was not established. Full byte-identical reproduction is not claimed.
- `git diff --check` passed.

The [original follow-up report](../outputs/no_drift_10000/REPORT.md), metadata, trial evidence, and recorded test/audit files are preserved. Its publication-status section describes the original preparation on 2026-10-28. This local verification does not establish remote CI success or the state of the hosted Streamlit app.
