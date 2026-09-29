# Changelog

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
