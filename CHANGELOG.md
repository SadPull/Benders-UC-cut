# Changelog

## v0.19.0 — 2026-10-05

- Default to 3-hour temporal decomposition (1/2/3 supported, including a short tail),
  continuous-relaxation initialization, inner Benders/SA and outer Lagrangian cutting planes.
- Certify native block master lower bounds from ObjBound through a bound-only interface;
  never use the integer certificate solution as a candidate, cut source or SA start.
- Use relative tolerance 1e-4, absolute tolerance 1e-5 and finite 20/20 outer/inner limits.
  Distinguish dual convergence, inner nonconvergence, outer limit, failures and resource stops.
- Rebuild price-dependent cuts and caches after multiplier changes. Preserve raw local u/v/w;
  recover and physically validate full-day candidates separately using original costs.
- Compute QP supporting cuts from a feasible dual objective; count Lagrangian constants once.
- Keep measured full-day advisor feedback and global call numbering across blocks and iterations.
- Retire target-cost, no-new-candidate and reference-gap stopping; update runners and reports.
- Preserve prior experiment archives and frozen source bytes; add temporal acceptance evidence.
- Pass all 138 regression tests. Both seed-7 RTS-24 SA acceptance runs stop at the
  inner iteration limit with validated full-day incumbents, without claiming convergence.

## v0.18.0 — 2026-10-04

- Add the frozen five-arm RTS-24 comparison: fixed-seed SA, changing-seed SA,
  LLM-only, SA+LLM, and SA+rule controls, each with seeds 7/17/27 and a 1800 s budget.
- Record candidate repair, duplicate detection, contribution sources, elapsed
  time, and independent physical/cost validation in experiment reports.
- Add independent Astra/max recovery supervision with fresh sessions, verified
  completion receipts, deadline enforcement, account preflight, and cumulative
  session limits. Preserve earlier authentication failures and interrupted runs.
- Complete the four prespecified reruns C/17, D/7, C/27, and D/17 using live
  monitoring. Their reference differences are 0.0815%, 0.6057%, 0.0808%, and
  0.4919%; accepted response counts are 2, 1, 2, and 2. Numerical audits pass,
  but three-response replications and evidence for bidirectional synergy remain incomplete.
- Retain all 25 search attempts and the separate Gurobi reference, original
  archives, recovery bundle, checksums, and audit records. Recovery sessions total 17/18.
- Archive the project reading textbook against its explicitly recorded
  `bae3200` source snapshot.
- Preserve experiment bytes across Git checkouts and include their audited logs;
  declare the recovery controller's psutil dependency.
- Pass all 111 current regression tests before publishing the tag; retain the
  test output in `docs/validation-v0.18.0.txt`.

### RTS-24 cleanup

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
