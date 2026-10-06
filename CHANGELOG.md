# Changelog

## v0.20.0 — 2026-10-06 (prerelease)

- Add optional native candidate fallback and the audited two-hour RTS-24 gap-closure
  protocol; retain default behavior and provenance throughout dispatch and recovery.
- Add four pure-SA experiment variants, inexact outer rounds, same-price analytic
  floors, exact binary propagation, active-cut encoding and diverse candidate screening.
- Isolate pure-SA search from online integer masters. Certify only saved cuts in a
  separate bound-only process, including deadline snapshots and report reconstruction.
- Preserve both executed source commits and publish immutable full experiment ZIPs
  as Release assets; keep curated reports, inputs, references and schedules in Git.
- Pass 184 regression tests. Pure-SA search integer-master calls are zero. All four
  32-unit screening configurations fail numerical audits, so all 11 subsequent
  ablation/comparison/confirmation slots are skipped. The <=1% goal is not achieved.
- Valid independent classical diagnostics reach 4.495790% (3h) and 4.810266% (5h).
  Numerical failures and timeouts do not establish an unreachable decomposition ceiling.
- Record 3410.28s formal execution/archiving and 3746.37s including supplementary review,
  below the 14400s total budget; review extends the 300s audit allowance by 100.23s.

## v0.19.1 — 2026-10-05

- Add the fixed SA scale/seed diagnostic: 24 frozen QUBOs, 144 sampling trials and
  18 full runs across RTS-derived systems, with isolated references and retained failures.
- Retry transient Windows state-file read/write conflicts without losing deadlines;
  archive the complete confirmation run and preserve the earlier interrupted evidence.
- Support explicit 5-hour blocks with a short final block; retain the 3-hour production default.
- Add the fixed three-unit, 24-hour comparison of 1/3/5-hour blocks using three varying
  seeds, 90-second run budgets and 60/20 inner/outer limits. Save per-block bounds,
  separate dual/cost results, convergence plots and a report-only reconstruction command.
- Complete all nine new runs in 124.13 seconds. Dual convergence counts are 3/3, 2/3
  and 2/3; reference-optimal full-day cost counts are 3/3, 2/3 and 3/3. The 5-hour
  seed-200007 run reaches optimal cost but stops at the last block's inner iteration limit.
- Validate 154 regression tests and five report tests. All nine returned schedules pass
  physical/cost checks; preserve 105 raw archive files with decompressed-byte SHA256 validation.
- Larger 10/32-unit cases still make no improvement over initialization. Three seeds
  support screening only; dual convergence and cost gaps remain distinct metrics.

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
