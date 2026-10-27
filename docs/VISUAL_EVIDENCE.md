# Plot source and reproduction

The [scatter plot](images/drift-vs-degradation.svg) uses all 200 archived synthetic trials: 40 seeds (1000–1039) in each of five scenarios. Each marker shows its recorded maximum feature PSI and reference-minus-current AUC, without jitter, sampling, or smoothing. Dashed lines mark the fixed PSI ≥ 0.20 and AUC-drop ≥ 0.08 rules. Negative AUC drop means AUC improved.

## Input

- File: [`outputs/monitoring_benchmark.json`](../outputs/monitoring_benchmark.json), 502,544 bytes; unchanged.
- Source commit: [`b25bbd7d0f0d589360350e6df8bf5f4391013c7f`](https://github.com/Zinga18018/driftlab-model-monitoring-simulator/commit/b25bbd7d0f0d589360350e6df8bf5f4391013c7f).
- SHA-256: `bdfffa3a451fefe6cf8ec0b4a39ea8f36db4ddf75c1e7c85621d574dab037c7a`.
- Git blob SHA-1: `e6fea8d5c28f599d5a52fe351c09e0ec6a44fb13`, matched to the public repository tree on 2026-10-27.

## Reproduce

The figure is generated with Matplotlib's non-interactive Agg and SVG backends, not hand-drawn plot marks. Tested with Python 3.12.14. Install the optional plotting dependencies separately from the app runtime (installation requires package-index access):

```sh
python -m pip install -r requirements-visuals.txt
python scripts/render_monitoring_evidence.py --png /tmp/driftlab-plot.png
python scripts/render_monitoring_evidence.py --check
```

Rendering makes no network calls and does not rerun the model. The script verifies the pinned input hash, all 200 AUC differences and alert flags, scenario summaries and Wilson intervals, seed counts, window sizes, fitting flags, and paired training/reference fingerprints. It also verifies the actual Matplotlib scatter coordinates against all 200 input pairs. Changed input is rejected until its provenance is reviewed and updated.

`--check` requires exact SVG regeneration in the recorded plotting environment: Matplotlib 3.10.8, NumPy 2.3.5 and Pillow 12.3.0. SVG fonts remain text, the SVG identifier salt is fixed, and variable date metadata is omitted. Exact bytes are not promised across other dependency versions or font/rendering environments. `--check --png PATH` can also compare a local PNG preview; no PNG preview is required by the app.

## Limits

These are synthetic development results, not a locked benchmark, production risk estimate, screenshot, or new model run. The positive-rate scenario changes features and the conditional label rule; it is not pure label shift. Positive-rate alerts, KS diagnostics, and uncertainty intervals are not plotted. The [benchmark report](../outputs/MONITORING_BENCHMARK.md) gives the no-drift result of 0/40 alerts and its Wilson 95% interval of 0.0%–8.8%, not a guarantee of zero risk.

The archived JSON records an older simulator-file hash; [the existing verification](verification-20261026.md) explains how current source reproduced it. This rendering changes neither that evidence nor its scope.

The SVG has descriptive alternative text, no external assets or scripts, and an opaque white background. Native and 390-pixel-wide renders were inspected; point counts, coordinates, links, and stale-file rejection were checked. This is asset QA, not a Streamlit or hosted-app test.
