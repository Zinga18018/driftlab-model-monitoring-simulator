"""Checkpointed no-drift extension; calls the unchanged simulator for every seed."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import time

# Bound BLAS threads before importing NumPy in parent and spawned workers.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import pandas as pd
from driftlab.benchmark import binomial_interval
from driftlab.simulator import DriftConfig, run_experiment, serializable_result


def trial(seed: int) -> dict:
    result = serializable_result(run_experiment(DriftConfig(
        rows=2000, seed=seed, drift_mode='none', drift_strength=0.0, label_noise=0.08)))
    ref, cur = result['reference_metrics']['auc'], result['current_metrics']['auc']
    return dict(seed=seed, scenario='none', config=result['config'],
                split_manifest=result['split_manifest'], reference_metrics=result['reference_metrics'],
                current_metrics=result['current_metrics'],
                auc_drop=None if ref is None or cur is None else ref-cur,
                max_psi=max(r['psi'] for r in result['drift_table']),
                alert_flags=result['alert_flags'], any_alert=any(result['alert_flags'].values()))


def summarize(records: list[dict]) -> dict:
    rates = {}
    for flag in ('distribution_drift', 'model_degradation', 'positive_rate_shift', 'any_alert'):
        count = sum(r['any_alert'] if flag == 'any_alert' else r['alert_flags'][flag] for r in records)
        rates[flag] = dict(count=count, trials=len(records), rate=count/len(records),
                           wilson_95_interval=binomial_interval(count, len(records)))
    return dict(completed_trials=len(records), alert_rates=rates,
                unavailable_auc_comparisons=sum(r['auc_drop'] is None for r in records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-seed', type=int, default=1040)
    parser.add_argument('--runs', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/no_drift_10000'))
    args = parser.parse_args()
    if args.start_seed < 0 or args.runs < 1 or not 1 <= args.workers <= 8:
        parser.error('require nonnegative seed, positive runs, and 1–8 workers')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / 'trials.jsonl'
    if destination.exists():
        parser.error('output already contains trials.jsonl; use a fresh directory (never overwrite evidence)')
    root = Path(__file__).resolve().parent
    source_names = ('driftlab/simulator.py', 'driftlab/benchmark.py', 'scripts_no_drift.py')
    metadata = dict(schema_version=1, source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        source_sha256={p: sha256((root/p).read_bytes()).hexdigest() for p in source_names},
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        platform=platform.platform(), workers=args.workers, blas_threads=1,
        start_seed=args.start_seed, end_seed_inclusive=args.start_seed+args.runs-1,
        requested_trials=args.runs, config=asdict(DriftConfig(rows=2000, drift_mode='none', drift_strength=0)),
        thresholds={'psi_greater_equal': .2, 'auc_drop_greater_equal': .08, 'absolute_positive_rate_change_greater': .08},
        protocol='Fresh model per seed; three isolated SeedSequence child streams; unchanged run_experiment; no tuning',
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    started = time.monotonic()
    records = []
    with destination.open('x') as out, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(trial, range(args.start_seed, args.start_seed+args.runs), chunksize=10):
            out.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
            records.append(record)
            if len(records) % 100 == 0 or len(records) == args.runs:
                out.flush()
                checkpoint = summarize(records)
                checkpoint['elapsed_seconds'] = time.monotonic()-started
                temporary = args.output_dir/'summary.tmp'
                temporary.write_text(json.dumps(checkpoint, indent=2)+'\n')
                temporary.replace(args.output_dir/'summary.json')
                print(f"{len(records)}/{args.runs}: {checkpoint['alert_rates']['any_alert']['count']} alerts; {checkpoint['elapsed_seconds']:.1f}s", flush=True)
    metadata.update(completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    trials_sha256=sha256(destination.read_bytes()).hexdigest())
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
