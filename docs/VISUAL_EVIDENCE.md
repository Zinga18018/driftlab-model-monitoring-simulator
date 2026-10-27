# README plot provenance

## What the plot shows

[`images/drift-vs-degradation.svg`](images/drift-vs-degradation.svg) plots **all 200 stored synthetic trials** in `outputs/monitoring_benchmark.json`: 40 seeds (1000–1039) for each of five scenarios. Each marker uses its trial's recorded maximum feature PSI and AUC drop (reference AUC minus current AUC), without jitter, sampling, smoothing, or aggregation. Marker shape and color both identify the scenario. An opaque light background remains legible in light and dark GitHub themes; click the README image to inspect it at full size.

The dashed lines show the recorded fixed alert rules: PSI ≥ 0.20 and AUC drop ≥ 0.08. Negative AUC drop means AUC improved. The plot does not show positive-rate alerts, KS diagnostics, or confidence intervals; those remain in the linked benchmark report. The positive-rate scenario changes features and the conditional label rule, so it is **not pure label shift**.

This is a visualization of archived synthetic development evidence, **not a new model run, screenshot, locked evaluation, production monitoring result, or measured business impact**. Zero observed no-drift alerts does not establish zero risk. The existing report gives its Wilson 95% interval of 0.0%–8.8% across independent seeds, not production false alarms per day.

## Pinned input

- Repository: [`Zinga18018/driftlab-model-monitoring-simulator`](https://github.com/Zinga18018/driftlab-model-monitoring-simulator).
- Source commit verified on 2026-10-27: [`b25bbd7d0f0d589360350e6df8bf5f4391013c7f`](https://github.com/Zinga18018/driftlab-model-monitoring-simulator/commit/b25bbd7d0f0d589360350e6df8bf5f4391013c7f).
- Input: [`outputs/monitoring_benchmark.json`](../outputs/monitoring_benchmark.json), 502,544 bytes.
- Input SHA-256: `bdfffa3a451fefe6cf8ec0b4a39ea8f36db4ddf75c1e7c85621d574dab037c7a`.
- Git blob SHA-1 at that commit: `e6fea8d5c28f599d5a52fe351c09e0ec6a44fb13`; local input bytes were matched to this public Git tree entry before rendering.
- Data and the existing benchmark evidence were not modified.

The archived JSON contains an older simulator-source hash. The existing [2026-10-26 verification](verification-20261026.md) explains how current source reproduced the saved trials. This visualization does not replace or expand that verification claim.

## Reproduce and verify

No extra dependency, network access, API key, or model execution is needed. With Python 3.10+ from the repository root:

```sh
python scripts/render_monitoring_evidence.py
python scripts/render_monitoring_evidence.py --check
```

The script checks the pinned input hash before rendering. It independently checks all 200 points' AUC differences and alert flags; all five scenarios' means, sample standard deviations, alert counts, rates, and Wilson intervals; 40 unique seeds per scenario; 2,000 rows in each training/reference/current window; train-only fitting flags; and matching training/reference fingerprints across paired scenarios. It then emits deterministic SVG bytes. `--check` compares the checked-in SVG with a fresh in-memory rendering and fails if it is stale. The fixed axis bounds are checked against every point, so none are clipped.

If the benchmark changes intentionally, review the new evidence before updating the pinned hash and provenance. The current script deliberately rejects changed input instead of silently presenting different results as the same figure.

## Visual QA

The SVG was rendered to PNG locally using Inkscape and inspected at native size and a 390-pixel-wide mobile preview. XML parsing, point counts, exact regeneration, and README image/source links were checked. The SVG includes a title and descriptive text for assistive technology, contains no scripts, external assets, or `foreignObject`, and keeps all labels as selectable text. This is asset QA, not an end-to-end Streamlit or hosted-app check.
