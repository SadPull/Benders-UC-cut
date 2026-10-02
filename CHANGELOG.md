# Changelog

## Unreleased

- Archive the user-stopped RTS-24 DeepSeek Flash comparison: 14 SA calls,
  10 completed LLM calls and one interrupted request; best reference difference
  2.527027%, above the amended inclusive 1% target. Calls 8–10 exhausted the
  provider generation limit. Preserve unknown usage for the interrupted request.
- Save HTTP prompt/request artifacts before sending future requests, so local
  interruption retains inputs. The completed study keeps its original source
  snapshot and honestly records the missing in-flight request body.

- Allow the same burst/target runner to use an explicit Chat Completions model
  with an environment-only API key, optional token/temperature overrides, and
  saved per-call requests, responses, usage, repair diagnostics and feedback.

- Add `stagnation_burst`: start with SA, switch after three consecutive rounds
  below 0.05% improvement, validate and refresh context between three sequential
  LLM calls, then reset stagnation and return to SA. Preserve existing strategies.
- Check the strict reference target after complete physical/cost validation,
  including calls without an event sink; stop within an unfinished LLM batch.
- Pass 173 regression tests and archive the RTS-24 three-scenario seed-7 run:
  5 SA calls and 2 Astra calls reach cost 889097.785912, 0.455455% above the
  independent reference. Stop after the first validated candidate in call 2;
  preserve the three unvalidated proposals without treating them as infeasible.

## v0.17.0 - 2026-10-02

- Add a native `codex_subagent` advisor path with atomic file exchange,
  exclusive dispatch claims, identity/hash checks, cancellation and single-use
  receipts. Codex dispatches fresh gpt-6-astra/max agents; Python makes no LLM
  HTTP requests and does not require an API key for this path.
- Generate proposal-level repair, duplicate, validation and measured-cost
  feedback automatically. Keep existing HTTP interfaces and historical runs.
- Add standalone fixed, stagnation-triggered, SA-only and advisor-only runs,
  plus an opt-in strict reference-cost stopping condition independent of rounds.
  Recompute and physically validate the independent Gurobi reference dispatch.
- Pass the two-call integration acceptance and archive its exact inputs,
  outputs, feedback and source snapshot. Archive the subsequent seed-7 study
  with 16 SA calls and 16 fresh Astra calls, stopped after full validation of
  call 16 at the user's explicit request. The original 0.1% target was NOT met.
- Best validated cost: 572242.837016, 0.215193% above the independently checked
  Gurobi reference 571014.051896. The final proposal required no repair.
  Matched pure SA costs 614177.849905 after 16 calls and 612502.196470 after 50.
  These are not equal total-compute budgets; no speed advantage is claimed.
- Preserve every proposal, invalid/duplicate outcomes, physical audits,
  user-stop amendment, timing interruption, excluded attempts, source/data
  snapshots and checksummed archives. A new three-call LLM burst strategy has
  not been started in this release.
- Pass all 168 regression tests on the release workspace; retain the test log
  and source/test fingerprints with the experiment archive.

## v0.16.0 - 2026-10-02

- Add a 32-generator, 24-hour IEEE RTS-24 derived copperplate UC/SUC benchmark
  with pinned public source bytes, per-field provenance, an offline deterministic
  converter and explicit single/three-scenario profiles. Preserve old defaults.
- Support validated optional `initial_output` in JSON; historical inputs retain
  their initial-at-Pmin behavior.
- Add a one-shot serial baseline runner with hard SA deadlines, independent
  Gurobi references, complete schedules/events, fixed-dispatch cost checks and
  source-fingerprinted offline audits. No LLM or QPU calls.
- Pass 153 regression tests and both baseline audits. Each SA run completed
  three rounds; its feasible cost exceeded the corresponding Gurobi reference
  by 46.892743% (deterministic) and 32.702773% (stochastic). No quality or speed
  advantage is claimed and no optimization runs were repeated.
- Archive all four runs and checksummed evidence under
  `experiments/rts24-baseline-20261002/`. Both references reached the requested
  1e-4 MIP gap; 19 schedules passed independent physical/cost validation.

## v0.15.0 - 2026-10-02

- Add optional Responses API input/output support and reject incomplete or
  truncated outputs while preserving existing Chat Completions defaults.
- Add a file-based Codex subagent experiment driver with per-call input/output
  hashes, single-use receipts, the original candidate repair, and no HTTP calls.
- Preserve the stopped Farluna 50-round study: one complete SA run, two
  interrupted SA+LLM traces, and no complete 50-round pairs.
- Archive the single Responses trial (65536 reasoning tokens, no candidate)
  and same-input Astra trial (four repaired feasible candidates; best
  571753.316901, 0.129465% above the feasible reference).
- Archive the subsequent Astra/max seed-7 experiment: seven subagent calls,
  six complete rounds, and a validated unrepaired candidate costing
  571504.279645 in round 7, 0.085852% above the feasible reference.
- Record the user's mid-study amendment to at most 10 rounds and early stop
  below 0.1%; stop the worker during round 7 and cancel the other seeds.
  Preserve this post-hoc rule without claiming a completed paired study.
- Save the reports, exact source snapshots, raw inputs/outputs and checksummed
  archives under `experiments/llm-advisor-evaluation-20261002/`.
- Verify current solver/test files match the snapshot that passed all 145
  regression tests, and retain physical/cost audits and secret-exclusion checks.

### Fixed-round protocol

- Add an opt-in 50-round paired rerun without overall or client request timeouts.
- Keep duplicate-candidate rounds running to the requested round count and
  distinguish the final equal-round comparison from timed checkpoints.
- Preserve the default v0.14.1 timed protocol and all historical results.

## v0.14.1 - 2026-10-01

- Switch the active LLM endpoint to Farluna and retain max reasoning / 65536 tokens.
- Honor the configured request timeout inside dynamic Benders: the rerun uses
  300 seconds per request, capped by the remaining 600-second run budget.
- Preserve v0.14.0 results and complete a separate five-seed paired study.
- Record one normal LLM response, four request timeouts and five requests
  unfinished at deadline. Two repaired suggestions passed full validation;
  one improved the incumbent locally, but all five final costs were worse.
- Report a mean cost increase of 8236.76 and no SA-call savings at jointly
  reached targets; retain all unreached thresholds without dropping seeds.
- Pass 134 regression tests, audit 185 complete candidates, and verify that
  completed-state resume leaves all event hashes unchanged.
- Publish the Chinese findings and checksummed source/raw-data archive.

## v0.14.0 - 2026-10-01

- Add dynamic whole-horizon LLM candidates sharing continuous Benders feedback,
  cache, repair and full physical validation; keep temporal LLM unsupported.
- Configure the requested DeepSeek endpoint, max reasoning and 65536-token
  output budget; retain secrets outside version control.
- Add paired 600-second process-supervised experiments, durable event traces,
  fixed quality thresholds and input/source-checked resume without rerunning
  interrupted trials. Independent decommitment stays disabled in this study.
- Complete eight 600-second runs (four seed pairs); preserve both interrupted
  seed-7 traces without rerunning them. The configured LLM produced no usable
  candidates: all 16 returned responses in complete runs truncated at 32000
  reported reasoning tokens. Four requests remained unfinished at deadline.
- Find zero first-passage SA-call savings at jointly reached targets and a mean
  final-cost increase of 6503.98 for SA+LLM. This evaluates the deployed
  configuration, not the potential benefit of valid LLM advice.
- Pass 132 regression tests, the expanded 10-test efficiency module, and physical
  validation of all 175 complete candidates; verify resume preserves raw traces.

## v0.13.0 - 2026-10-01

- Add an independent original-block MIQP control without changing the default
  SA backend: offline same-price diagnostics and bounded exact-block closed loops.
- Preserve 30-second block limits, an 1800-second offline budget, and 300-second /
  10-round closed loops, with fingerprinted resume and conservative interrupted
  time accounting. Keep solver bounds separate from time-limited incumbents.
- Complete 45 offline snapshots and three 10-round closed loops: all 924 block
  solves certified within numerical tolerances; 33 complete schedules validated.
- Obtain lower-bound gaps of 2.8605%, 1.5049% and 1.1225% for 1/2/3-hour blocks.
  All best bounds occur at zero initial prices; later rounds do not improve them.
  Feasible-cost differences remain 6.3348%, 5.0938% and 4.7453%, respectively.
- Preserve reports, audit and checksummed reproducibility archive under
  `experiments/exact-block-control-20261001/`. Timing overlaps with regression
  tests are disclosed; these results do not establish a speedup.
- Pass the 121-test full suite and the subsequently expanded 15-test control
  module, including timeout, budget recovery, partial-round and source checks.

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
