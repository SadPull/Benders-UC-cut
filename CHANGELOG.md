# Changelog

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
