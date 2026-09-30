# SUC + Benders (10×24)

Current release: [v0.12.0](CHANGELOG.md).

This repository contains a small, reproducible two-stage stochastic unit commitment model. The research Benders algorithm generates binary commitment candidates using Kaiwu local simulated annealing, decommitment search, and an optional LLM. Gurobi solves only the continuous dispatch subproblems within that algorithm. A separate full deterministic equivalent Gurobi model supplies an external reference solution and proven lower bound for post-run comparisons.

## Scope

- 10 generators, 24 hourly periods
- three load scenarios:
  - `low`: 0.90 × base load, probability 0.25
  - `base`: 1.00 × base load, probability 0.50
  - `high`: 1.05 × base load, probability 0.25
- 5% spinning reserve
- exact 10×24 base load and generator parameters from the original MATLAB data
- simulated annealing and decommitment for the binary master search
- optional LLM candidate advice over HTTP; no QPU calls

> The `high` scenario is capped at 1.05 because 1.10 × peak load plus 5% reserve exceeds the installed capacity of the 10-generator system. If you want a different high scenario, provide your own `scenario_file`.

## Install

```powershell
python -m pip install -r requirements.txt
```

Use Python 3.10 with Kaiwu SDK 1.4.1, as specified by the
[SDK installation guide](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/getting_started/sdk_installation_instructions.html).
If the SDK requests activation, follow its console instructions using your
own user ID and SDK authorization code.

The Kaiwu integration uses the documented `kaiwu.core.QuboModel` and
`kaiwu.core.qubo_matrix_to_ising_matrix` APIs. The annealer receives the full
QUBO matrix, including inequality slack variables; decoding extracts the
commitment and theta bits and discards the auxiliary bits.

## Capacity and QUBO encoding

`pstart` and `pshut` are separate startup and shutdown output limits. Dispatch
and reserve must satisfy both limits; a unit that starts in hour t and shuts
down in hour t+1 has capacity `min(pstart, pshut)`. Candidate checks, repair,
context, scenario subproblems, and the full reference use this same rule.
The QUBO uses an auxiliary AND bit for this case when minimum up-time is one
hour. Slack and AND bits are excluded from `variable_count()` and included in
`qubo_variables()` and `matrix()`.

Reserve inequalities use the exact decimal capacity lattice and explicitly
bounded integer slack. Demand is rounded up to the next point on that lattice,
which preserves the binary feasible set. `reserve_scale` affects conditioning,
not the lattice or its feasible set. Candidate capacity checks use a numerical
tolerance of 1e-6 MW rather than allowing a 0.5 MW shortage.

Only copies of Benders cuts entering the QUBO are quantized. Constants and
commitment coefficients are rounded downward in the form `affine <= 0`, so
quantization can only relax a cut. The worst-case relaxation is at most
`theta_step` for an optimality cut and 1e-3 in total-shortfall units for a
feasibility cut. Theta coefficients remain exact on the chosen grid. Original
cuts, Gurobi dispatch, and reported costs retain their original precision.
Zero-range and constant inequalities do not request invalid SDK slack ranges.

The objective and optimality cuts use `max(1, theta_cap)` as their common cost
scale; reserve and feasibility cuts use `reserve_scale`. Scaling occurs before
squaring constraint residuals. `QuantumConfig.initial_temperature` is a multiplier
of the Ising single-spin energy-change bound; `cutoff_temperature` is a multiplier
of the smallest positive objective coefficient. Both are expressed in the same
units as the normalized matrix. `solve_qubo(matrix, config, *, energy_unit=1.0)`
remains callable with two arguments; the research loop supplies the master's
objective resolution explicitly. Cooling to this resolution can take more
temperature steps than the former fixed normalized cutoff. These changes do not
guarantee feasible annealing samples or global optimality.

## Architecture

```text
Kaiwu SA + repair + decommit + optional LLM -> commitment candidates
Gurobi continuous subproblems              -> dispatch, costs, duals, Benders cuts
Benders cuts                              -> next Kaiwu QUBO

Independent full Gurobi model              -> reference solution and lower bound
Completed research results + reference    -> post-run comparison only
```

- **Candidate generation** uses simulated annealing and decommitment by default;
  LLM advice is optional. Deterministic initialization, repair, deduplication, and
  incumbent reuse support those searches without another optimization solver.
- **Subproblems** receive fixed `u/v/w`; all Gurobi decision variables inside the
  research loop are continuous. Their costs, duals, and cuts provide normal
  Benders feedback and decommitment/LLM context.
- **The full Gurobi reference is independent.** Its commitment, objective, bound,
  and status never enter candidate generation, cuts, LLM context, or stopping
  decisions. The benchmark solves it after all research strategies finish.
- The research algorithm stops at `MAX_ITERATIONS` or `NO_NEW_CANDIDATES`.
  Temporary LLM failures can retry within the iteration budget. If no evaluated
  candidate is feasible in every scenario, it returns `NO_FEASIBLE_SOLUTION`;
  this does not prove the original model infeasible.
- The heuristic master does not supply a certified global lower bound or a
  Benders convergence certificate. QUBO energies and sampled objectives must
  not be treated as proven lower bounds.

## Changes in v0.11.0

- Remove the exact Gurobi master and its influence on candidates and termination.
- Remove independent local perturbation; keep decommitment moves and their LLM
  context (`unit_hour_decommit_gradient`, `top_decommit_opportunities`,
  `hourly_commitment_profile`, `peaker_usage`, and `decommit_ladder`).
- Report the full model's `ObjBound`, feasible objective, status, MIP gap, and
  runtime separately as external reference results.
- Remove the exact-master and perturbation settings from `BendersConfig` and
  the CLI, including `--no-exact-master`, `--master-time-limit`,
  `--master-mip-gap`, and `--no-perturb`. Old callers must remove these arguments.
- Remove research result/history fields `lower_bound`, `gap`, `master_status`,
  `master_runtime`, `exact_candidates`, and `perturbed_candidates`.

## LLM context

When `--llm` is enabled, the advisor receives a compact P0/P1 context containing:

- incumbent cost decomposition and per-scenario fuel cost
- base candidate costs and Hamming distance to the incumbent
- scenario balance and reserve dual prices
- Benders cut coverage and tightness summary
- recent local move cost changes
- candidate comparison table
- scenario dispatch summary and reserve margins
- tight constraints and infeasibility diagnostics
- recent iteration history
- unit marginal costs and unit cost decomposition

All context values are rounded to three decimals and lists are capped to keep the prompt compact. LLM output remains advisory only and is always validated locally.

Per-unit fuel cost is the probability-weighted cost of scenario outputs,
`E[a*p^2 + b*p]`, rather than the cost of the expected output. Unit decomposition
includes `shutdown_cost`; hourly fixed costs include shutdowns even when the
unit is already off in that hour.

## Run the benchmark

```powershell
python -m suc_benders.benchmark --max-iterations 5
```

The default strategy rows are `quantum-only`, `quantum+repair`, and
`quantum+repair+decommit`. Every row retains the deterministic initial candidate.
Select an individual strategy with:

```powershell
python -m suc_benders.benchmark --max-iterations 5 --strategies quantum+repair+decommit
```

To include `quantum+repair+llm` and `quantum+repair+decommit+llm`:

```powershell
python -m suc_benders.benchmark --max-iterations 5 --llm
```

`LLM_API_KEY` must be set before using `--llm`. Without it, the LLM row safely returns no candidates.

Benchmark output keeps `exact_objective` as the independent reference's best
feasible objective and adds `exact_lower_bound`, `exact_status`, `exact_mip_gap`,
`exact_mip_gap_tolerance`, and `exact_runtime`. The benchmark uses a reference
MIP gap tolerance of `1e-4`. `OPTIMAL` means optimal within the declared tolerance;
with `TIME_LIMIT`, the objective is only the best feasible reference solution.
The proven lower bound always comes from `ObjBound`, never the feasible objective.

The following metrics are computed only after the research runs finish:

```text
relative_error = (research objective - reference objective) / reference objective
reference_gap  = (research objective - reference lower bound) / max(1, abs(research objective))
```

Unavailable comparisons serialize as `null`, including a zero reference objective
for `relative_error`. `reference_gap` is an external comparison, not a research
Benders stopping criterion. Each strategy's `runtime` excludes the full model.

## Run

Benders with simulated annealing and decommitment:

```powershell
python -m suc_benders.run --max-iterations 5
```

Add optional LLM advice:

```powershell
python -m suc_benders.run --max-iterations 5 --llm
```

The `--no-quantum` and `--no-decommit` switches support ablations. Disabling
annealing skips Kaiwu/QUBO construction while retaining the deterministic seed.

Independent full Gurobi reference:

```powershell
python -m suc_benders.full
```

This command reports both `objective` and `lower_bound`, along with `status`,
`mip_gap`, and `mip_gap_tolerance`. It accepts `--time-limit` and `--mip-gap`.

Optional research arguments:

```text
--data data/suc_10x24.json
--scenario-file data/scenarios.json
--quiet
--no-quantum
--no-decommit
--llm
--output run.json
```

To use a custom scenario set, create a JSON file such as:

```json
{
  "reserve_fraction": 0.05,
  "scenarios": [
    {"name": "low",  "probability": 0.25, "load_factor": 0.9},
    {"name": "base", "probability": 0.50, "load_factor": 1.0},
    {"name": "high", "probability": 0.25, "load": [700, 750, 850, 950, 1000, 1100, 1150, 1200, 1300, 1400, 1450, 1500, 1400, 1300, 1200, 1050, 1000, 1100, 1200, 1400, 1300, 1100, 900, 800]}
  ]
}
```

## Attribution experiment

The fixed `attribution` suite compares all eight on/off combinations of Kaiwu
local simulated annealing (Q), LLM advice (L), and decommitment (D), with the same
repair and initial candidate. Two controls add raw annealing without repair and
16 independent Bernoulli(0.5) commitment proposals followed by the same repair.
The latter is an experimental control, not a quantum backend.

The suite runs the canonical three-scenario 10x24 model, a 90% load variant, and
a two-hour cyclic load shift. Seeds are 7, 17, 27, 37, and 47: 150 research runs.
Each run is limited to 10 iterations and 40 unique complete candidate evaluations,
including initialization. Cache hits do not consume this budget. The additional
`EVALUATION_LIMIT` status only applies when this optional limit is configured.
The eight main groups retain the existing candidate evaluation order.

Set `LLM_API_KEY` in the process environment before running; never save the key
in the experiment directory. The selected service uses OpenAI-compatible Chat
Completions. The original Messages protocol remains the ordinary CLI default.
Use `--llm-api-format chat_completions`, `--llm-base-url`, and `--llm-model` with
`suc_benders.run` to select that protocol for an individual run.

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m suc_benders.benchmark --suite attribution --llm-base-url http://47.101.31.14:8300/v1 --llm-model deepseek-v4.1-flash --pilot
python -m suc_benders.benchmark --suite attribution --llm-base-url http://47.101.31.14:8300/v1 --llm-model deepseek-v4.1-flash --resume
```

`--pilot` completes the first 10-strategy block without solving the full
reference. `--resume` retains completed runs only when the solver fingerprint,
inputs, dependencies, and settings match. `--output-dir` defaults to
`results/attribution-20260930`. Existing benchmark arguments such as
`--max-iterations` and `--seed` belong to the legacy suite; attribution uses its
fixed, recorded matrix. The suite executes serially with one Gurobi thread.

LLM settings are temperature 0.2, at most four candidates, 8192 completion tokens,
180-second timeout, and one retry after 30 seconds for network errors, HTTP 429,
or HTTP 5xx. There are at most 600 logical calls and 1200 HTTP requests. Auth
errors, invalid JSON, empty output, truncation, and duplicate proposals are
reported distinctly. Live call progress is written to `progress.log`.

Each run records provenance before and after repair, candidate hashes, cache
hits, dispatch feasibility, actual incumbent improvement, timing by phase,
QUBO size, raw annealing diversity, and sanitized LLM requests/responses with
reported token usage. A run is checked for physical residuals and cost
conservation before its completion record is saved. A process interruption
does not turn an incomplete run into a completed statistical sample.

After all research runs, three independent full models supply reference
objectives and ObjBound values, with a 300-second time limit and MIPGap 1e-4.
Reports compare budgets of 10/20/40 evaluations and 30/60/120/300 seconds,
preserving observed stopping points. Paired bootstrap intervals use 10,000
resamples. For each fixed D, interaction is C10+C01-C00-C11; a positive value
means superadditive cost improvement. Deterministic objective repeats are not
independent algorithm samples; remote LLM outputs are not guaranteed to be
reproducible from the local seed.

Artifacts include the source snapshot and manifest, per-run JSON and JSONL
traces, `runs.csv`, `summary.csv`, `checkpoints.csv`, `effects.csv`, `controls.csv`, static plots,
and a Chinese `report.md`. Report generation is independent of research state.

The completed 2026-09-30 study is in
[`results/attribution-20260930/report.md`](results/attribution-20260930/report.md).
Its recovery ledger includes interrupted-call overhead in the total API budget.

## Paper transfer: temporal SA-Benders

The opt-in paper transfer implements bounded Lagrangian cutting-plane
coordination of 1/2/3-hour blocks, with local continuous convex dispatch,
SA-only integer masters, and full-horizon repair and validation. It preserves
the original global time indices: initial conditions are not repeated at block
boundaries, and cross-block ramps, minimum up/down constraints, transitions,
and next-hour shutdown capacity limits enter the coupling rows.

```powershell
python -m suc_benders.run --master-encoding dynamic --decomposition temporal --block-hours 2 --no-decommit --max-iterations 10 --time-limit-seconds 300 --max-candidate-evaluations 40
```

`--master-encoding compact` selects a static total-cost QUBO;
`--master-encoding dynamic` enables continuous master relaxation and local
objective bounds. Omit `--decomposition temporal` for the whole-horizon
versions. These experimental modes require repair and reject LLM, decommitment,
random-candidate, or disabled-SA configurations. The default remains legacy.
`theta` continues to mean recourse cost; the new internal total-cost variable
can be negative under Lagrangian prices. Fixed variables, bounded objective
bits, exact capacity lattices and one-sided cut quantization are retained.

Local optimality cuts and objective upper bounds are rebuilt at every price
change; local feasibility cuts survive. The sum of local sampled costs is not
a certified global lower bound. Only the continuous relaxation bounds at the
same prices may be summed, including the coupling constant. Kelley planes are
affine upper bounds, not necessarily tangent planes with an approximate SA
oracle. The multiplier box is fixed at +/-10 for normalized equalities and
[0,10] for inequalities; no global convergence claim is made.

Run the fixed six-strategy, five-seed comparison (original 10x24 case only):

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m suc_benders.paper_experiment --pilot
python -m suc_benders.paper_experiment --resume
```

Each run allows 300 seconds and 40 unique complete candidate evaluations;
block evaluations are counted separately, with all costs included in runtime.
SA calls cannot be interrupted, so overshoots are reported and time-checkpoint
comparisons exclude results completed after the checkpoint. The independent
full reference runs after all 30 research runs. The manifest fingerprints
sources, inputs, dependencies and settings; resume refuses mismatched solvers.
An implementation or validation exception produces an error file, not a
completed sample. Reports and CSVs are in `results/paper-transfer-20260930-v2`.

The completed 30-run study is preserved in Git under
[`experiments/paper-transfer-20260930-v2/`](experiments/paper-transfer-20260930-v2/README.md),
including the [Chinese report](experiments/paper-transfer-20260930-v2/report.md),
CSVs, plots, validation audit, and a checksummed archive with the complete run
data and exact experiment source snapshot. The archive predates release-only
documentation changes; use its source snapshot to reproduce its fingerprint.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Limitations

- Kaiwu local simulated annealing is the only implemented quantum backend; cloud/QPU is not configured.
- Candidate search is heuristic and does not prove global optimality or infeasibility.
- `solve_full()` is an external validation reference, not part of the research algorithm.
- The feasibility cut is a generic total-shortfall cut derived from Gurobi duals; it is not a specialized strong cut.
- Attribution results concern three related 10x24 cases, not general scaling
  behavior or quantum hardware advantage.
