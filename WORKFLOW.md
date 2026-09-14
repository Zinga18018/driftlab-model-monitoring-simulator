# DriftLab Monitoring Simulator: workflow

Create controlled synthetic distribution changes and inspect how drift signals and model performance react.

**Relevant roles:** data science, ML engineering.

## Flowchart

```mermaid
flowchart TD
    A["Seed, drift scenario and strength"] --> B["Generate independent training, reference and current windows"]
    B --> C["Fit feature scaling and logistic model on training only"]
    C --> D["Score held-out reference and current windows"]
    B --> E["Compute drift using reference quantile bins"]
    D --> F["Compare model metrics and label prevalence"]
    E --> G["Apply fixed alert thresholds"]
    F --> G
    G --> H["Incident summary and Streamlit views"]
    A --> I["Repeat no-drift baseline across independent seeds"]
    I --> J["Measure false-alert rate and Wilson interval"]
    J --> K["Compare drift scenarios at unchanged thresholds"]
    K --> L["Save trials, class counts, split fingerprints and source hashes"]
```

## Explain it in an interview

“I train on one synthetic window and measure baseline performance on an independent reference window. I first check how often unchanged data triggers an alert across repeated seeds. Then I introduce a known change and inspect which signals respond. This distinguishes feature drift from performance degradation and makes the false-alert uncertainty visible.”

## What this diagram does and does not establish

The model and preprocessing do not fit the reference or current windows. These are newly generated synthetic development runs, not a locked benchmark or evidence that a previous holdout was untouched. AUC requires both label classes; unavailable comparisons are explicit. The app does not operate a production alerting or retraining service.

## Follow the code and evidence

- [Simulation, independent windows, scoring and alert logic](driftlab/simulator.py)
- [Repeated no-drift and scenario benchmark](driftlab/benchmark.py)
- [Reproducible benchmark command](scripts_benchmark.py)
- [Generated report and limitations](outputs/MONITORING_BENCHMARK.md)
- [Interactive controls and results](app.py)
- [Core verification](tests/test_simulator.py) and [Streamlit checks](tests/test_app.py)
