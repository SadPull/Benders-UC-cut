# Exact-block control experiment

Release: v0.13.0 (2026-10-01). Original 10-unit, 24-hour, three-scenario model.

- [Findings and limitations](findings.md): Chinese interpretation of the control.
- [Generated report](report.md) and [summary/checkpoints](summary.json).
- [Validation audit](audit.json) and [run manifest](manifest.json).
- [Full reproducibility archive](reproducibility.zip), checked by [SHA256SUMS](SHA256SUMS).

All 45 offline snapshots and three 10-round closed loops finished within their
budgets. All 924 original-block MIQPs were certified within the configured
numerical tolerances; all 33 evaluated full schedules passed validation. No
QUBO, SA, LLM or quantum hardware was called by this control.

The archive retains the complete experiment directory, including raw block
certificates, multiplier trajectories, full candidate dispatches and the exact
solver source snapshot. Validation tests are also included under `validation/`.
The original source hashes are preserved; release documentation changes do not
rewrite experimental provenance. SHA256SUMS covers every published artifact
except itself. Verify it before extracting the archive under `results/`, without
overwriting existing results.

Source SA trajectories and the independent Gurobi reference remain in the
[v0.12.0 study](../paper-transfer-20260930-v2/README.md). Restore its archive to
`results/paper-transfer-20260930-v2/` before a new control run:

```powershell
python -m suc_benders.exact_experiment --output-dir results/exact-block-control-new
```

Use the same arguments with `--resume` to continue. Fingerprints must match.
Interrupted wall time since the last active checkpoint, including downtime,
is charged to the remaining budget rather than reset. Completed stages are
not rerun. Recovering the original study requires its archived source and
recorded dependency versions if the current checkout has changed.

Lower-bound gap and feasible-cost difference are distinct. Every best closed-loop
bound was obtained at the initial zero prices; these runs did not establish
dual convergence. Regression tests overlapped part of the experiment, so wall
times are not controlled speed benchmarks.
