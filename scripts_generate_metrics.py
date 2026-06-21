"""Generate a small verified metrics snapshot for the README/resume."""

from __future__ import annotations

import json
from pathlib import Path

from driftlab.simulator import DriftConfig, run_experiment, serializable_result


def main() -> None:
    config = DriftConfig(rows=4000, drift_strength=1.15, drift_mode="mixed", label_noise=0.08, seed=42)
    result = serializable_result(run_experiment(config))
    output = Path("outputs/demo_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    print(f"reference_auc={result['reference_metrics']['auc']}")
    print(f"current_auc={result['current_metrics']['auc']}")
    print(f"max_psi={max(row['psi'] for row in result['drift_table'])}")
    print(f"alerts={len(result['alerts'])}")


if __name__ == "__main__":
    main()
