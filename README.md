# SUC + Benders (10×24)

Current release: [v0.11.0](CHANGELOG.md).

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
- optional LLM candidate advice, no cloud/QPU calls

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

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Limitations

- Kaiwu local simulated annealing is the only implemented quantum backend; cloud/QPU is not configured.
- Candidate search is heuristic and does not prove global optimality or infeasibility.
- `solve_full()` is an external validation reference, not part of the research algorithm.
- The feasibility cut is a generic normalized-shortfall cut derived from Gurobi duals; it is not a specialized strong cut.
- The project is model scaffolding, not a research benchmark.
