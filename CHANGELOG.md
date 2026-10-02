# Changelog

## Unreleased — RTS-24 cleanup

- Keep the RTS-24 dynamic whole-horizon SA/LLM workflow, independent Gurobi reference,
  physical validation, and the three RTS-24 experiment archives.
- Make `python -m suc_benders` use the current advisor runner and RTS-24 data by default.
- Remove the 10×24 case, historical attribution/efficiency/subagent experiment drivers,
  legacy/static master modes, temporal coordination, independent decommitment search,
  exact-block controls and their results.
- Retain current model and candidate-validation regression tests; migrate fixtures
  to RTS-24 or small synthetic cases and remove tests of retired features.
- Preserve source snapshots and checksums in retained RTS-24 archives. Historical
  versions remain available in Git history.

## Current RTS-24 features

- Dynamic QUBO with exact capacity lattices and conservative cut quantization.
- Pure SA, fixed/stagnation/burst SA+LLM, and LLM-only controls.
- Codex Astra file exchange or explicit HTTP Chat Completions, with automatic
  proposal feedback and request artifacts saved before dispatch.
- Independently validated reference targets and recoverable run checkpoints.
