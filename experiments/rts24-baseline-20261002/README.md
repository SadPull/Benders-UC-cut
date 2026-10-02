# RTS-24 derived baseline validation (v0.16.0)

[Results and protocol](report.md) · [Acceptance audit](audit.json) ·
[Source/configuration manifest](manifest.json) · [Regression log](regression.log)

Both profiles passed acceptance. Each SA run completed three rounds and three
sampling calls, with no LLM, independent decommitment or random candidates.
Gurobi reached the requested 1e-4 MIP gap in both independent reference runs.
The SA costs were 46.892743% (deterministic) and 32.702773% (stochastic) above
their respective feasible references. This validates the input/solver pipeline,
not a quality or speed advantage. No repeat runs were added to improve results.

All 153 regression tests passed before optimization started. Independent checks
validated 17 fully evaluated SA schedules and both reference schedules; maximum
physical residual was 1.683e-11 MW. Re-solving continuous dispatch for the fixed
reference commitments changed total costs by less than 4.877e-7.

## Evidence and reproduction

The source commit is `65205d0c070215f7cb1ec3a3dd9683866856e227` and remains in
Git history. The manifest records hashes for solver, tests and benchmark inputs,
plus Python/package versions and the exact protocol. Public source files,
conversion script and per-field provenance are in `data/rts24/` at the repository
root. This directory contains all four runs, including full dispatches, raw SA
events, solver logs, process states and reference fixed-dispatch checks.

The deterministic and stochastic profiles are different optimization problems;
do not compare their absolute costs as an algorithm ranking. Every reported
cost gap uses the matching profile's independent reference. The certified
error bound uses Gurobi's lower bound and numerical tolerances.

At this release, audit saved runs without optimization:

```powershell
python data/rts24/convert.py --check
python -m suc_benders.rts24_experiment --audit-only --output-dir experiments/rts24-baseline-20261002
```

To intentionally execute another validation, use a new output directory on a
clean committed checkout:

```powershell
python -m suc_benders.rts24_experiment --output-dir results/rts24-new-validation
```

The runner refuses any existing output directory. Audit verifies current source
hashes; use the release checkout if future solver changes cause a mismatch.
`checksums.json` covers the archived files except itself. File bytes are preserved
by the local Git attributes. No API credentials or external model outputs are
part of this study.
