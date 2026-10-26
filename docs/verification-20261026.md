# Local verification: 2026-10-26

This review changed documentation and added an architecture diagram. It did not change simulator code, thresholds, metrics, or checked-in benchmark outputs.

## Benchmark reproduction

The archived benchmark stores a simulator-file SHA-256 that differs from the current file. The hashes for `driftlab/benchmark.py` and `scripts_benchmark.py` still match. Rather than assuming that mismatch was harmless, the complete benchmark was rerun locally against the current code, with output written to a separate directory.

```powershell
python scripts_benchmark.py --runs 40 --rows 2000 --start-seed 1000 --output-dir outputs/reproduced
```

All **200 trial records**, all **five scenario summaries**, and the **protocol** matched the checked-in JSON exactly. Trial comparison includes reference/current metrics, alert flags, class counts, seeds, and split fingerprints. The rerun used Python **3.14.7**, NumPy **2.5.3**, and pandas **2.3.3**, rather than the original Python 3.12 environment. The recorded comparison and source hashes are in [verification-20261026.json](verification-20261026.json).

The original benchmark remains historical development evidence. Reproducing it does not make it an untouched test set, a production evaluation, or a guarantee about future alerts.

## Tests and documentation

- `python -m unittest discover -s tests -p test_simulator.py -v`: **17 tests passed** using an existing local environment with NumPy and pandas. No dependencies were installed for this review.
- The full test command using the original Streamlit environment could not start: Windows Application Control blocked that environment's Python executable. The Streamlit app smoke test was **not rerun** in this review.
- All README relative links resolved, the SVG parsed as XML, and `git diff --check` passed.
- `docs/monitoring-flow.svg` was rasterized with the existing Sharp library and visually inspected at 1200 × 960. Text, arrows, and fit/evaluation boundaries were readable and unclipped.

## Preservation and naming

The existing uncommitted README in the original checkout was read but left unchanged. Its monitoring and repeated-benchmark flow informed the expanded diagram. These edits are in a separate local review branch.

Keep the repository name `driftlab-model-monitoring-simulator`: it already states the project and scope. No remote settings, links, deployments, or public repository names were changed.
