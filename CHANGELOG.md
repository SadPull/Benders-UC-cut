# Changelog

## v0.12.0 - 2026-10-01

### Paper transfer

- Add opt-in compact and dynamic total-cost QUBO encodings and temporal
  SA-Benders decomposition with bounded Lagrangian cutting-plane coordination.
- Preserve global time-boundary constraints, rebuild objective cuts at changed
  multipliers, reuse valid feasibility certificates, and validate repaired
  full-horizon schedules with continuous dispatch only.
- Add reproducible six-strategy, five-seed experiments, 300-second runtime and
  40-complete-evaluation budgets, source fingerprints, and independent reference
  solves after the research runs.
- Complete all 30 runs with physically valid schedules. Dynamic whole-horizon
  and one-hour temporal methods reduce mean cost by approximately 0.63% and
  0.64% against the legacy baseline; temporal decomposition reduces maximum
  QUBO size but does not guarantee the 500-bit hardware reference limit.
- Pass 109 tests and the nine-test paper module recheck. Preserve browsable
  reports and a checksummed full reproducibility archive under
  `experiments/paper-transfer-20260930-v2/`.

The following attribution and correctness changes were also pending since
v0.11.0 and are included in this release.

### Added

- Add a reproducible 150-run attribution suite with factorial Q/LLM/decommitment
  ablations, raw-annealing and random-repair controls, per-run candidate budgets,
  sanitized provenance and API traces, phase timing, physical validation, and
  resume checks against source, data, configuration, and dependency fingerprints.
- Add optional Chat Completions support while retaining Messages defaults;
  distinguish API failures, truncation, malformed output, and duplicate advice.
- Add independent post-run reference solves, fixed-budget comparisons, paired
  bootstrap effects, CSV exports, scientific plots, and a Chinese experiment report.
- Add regression tests for protocol compatibility, retry limits and redaction,
  budget enforcement, cache attribution, diagnostic isolation, randomized
  controls, resume/reference ordering, and statistical calculations.

### Fixed

- Apply startup and shutdown capacity bounds separately, including one-hour
  on-blocks. Align candidate checks, repair, LLM reserve context, both continuous
  subproblem formulations, and the independent full model; remove the redundant
  reference capacity inequality that used the previous combined reduction.
- Encode reserve slack on the exact capacity lattice. Quantize QUBO copies of
  Benders cuts conservatively with bounded error and explicit integer slack;
  preserve original cuts and handle constant/zero-slack inequalities.
- Scale costs and cut residuals before squaring and calibrate annealing
  temperatures to spin energy changes and objective resolution. Add the optional
  `energy_unit` keyword to `solve_qubo` without changing research result fields.
- Compute per-unit expected fuel cost over scenarios and include shutdown costs
  in hourly and per-unit LLM cost summaries.
- Replace the full/subproblem self-comparison test with comparison against the
  independently reoptimized full model with fixed commitments. Add regressions
  for one-hour capacity, cut validity, encoding feasibility and ordering,
  conservative quantization, cost conservation, and temperature scaling.

### Validation

- All 99 tests passed after adding the attribution suite (332.166 seconds).
  Four targeted design/resume tests also passed after including BLAS/OpenMP
  thread settings in the resume fingerprint.

- The 77-test full suite and the additional real-scenario QUBO cut regression
  passed (78 tests total) with Python 3.10, Kaiwu 1.4.1, and Gurobi 13.0.1.
- Two-iteration seed-7 runs completed with and without annealing. The second
  annealing matrix's largest coefficient fell from 2.78e18 to 3.59e8; raw
  candidates still required repair. These are diagnostics, not a convergence
  certificate or a general performance claim.

## v0.11.0 - 2026-09-30

### Fixed

- Restrict research commitment generation to Kaiwu simulated annealing,
  decommitment, and optional LLM advice, with deterministic initialization and repair.
- Remove Gurobi master candidates, bound feedback, and convergence decisions
  from the research Benders loop. Gurobi only solves continuous subproblems there.

### Changed

- Remove the exact master module and independent local perturbation, including
  their configuration/CLI options and benchmark strategies.
- Remove research bound/gap, exact-master, and perturbation result fields;
  callers must migrate to the reduced research API.
- Add `FullResult.lower_bound` from Gurobi `ObjBound`. The independent full-model
  CLI reports bound, objective, status, achieved MIP gap, and target tolerance.
- Run the full reference after all benchmark research strategies. Report its
  objective, lower bound, status, MIP gap, tolerance, and runtime separately.
- Compute `relative_error` and `reference_gap` only for post-run comparisons;
  unavailable values serialize as null. Reference runtime is excluded from research runtime.

### Validation

- All 66 tests passed with Python 3.10, Kaiwu SDK 1.4.1, and Gurobi 13.0.1.
- A one-iteration benchmark with seed 7 completed using default annealing and
  decommitment, followed by the independent full-model reference.
- Add regression tests guarding Gurobi optimization calls: research models must
  contain only continuous variables and never call the full reference solver.
- Cover external-reference independence, removed interfaces, comparison metrics,
  LLM recovery, duplicate candidates, and scenario infeasibility feedback.

## v0.10.0 - 2026-09-29

### Fixed

- Count no-load cost savings only once when ranking decommitment moves, while
  preserving the full move's startup and shutdown cost changes.
- Preserve SDK-generated slack variables in the QUBO matrix instead of dropping
  their constraint contributions.

### Changed

- Upgrade the Kaiwu dependency from 1.3.1 to 1.4.1 and use the documented
  `kaiwu.core.QuboModel` and matrix-conversion APIs and current variable names.
- Remove the legacy license-check override; use the SDK's normal activation flow.
- Document Python 3.10 and Kaiwu SDK 1.4.1 installation requirements.
- `QuantumMaster.matrix()` now includes auxiliary bits in `qubo_variables()`
  order. `decode()` expects that full vector; `variable_count()` continues to
  count decision bits only.

### Validation

- All 72 tests passed with Python 3.10 and Kaiwu SDK 1.4.1.
- Regression coverage includes decommitment ranking, full QUBO energy
  equivalence, auxiliary-bit decoding, and local simulated annealing.
