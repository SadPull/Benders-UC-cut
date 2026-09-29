# SUC + Benders (10×24)

This repository contains a small, reproducible two-stage stochastic unit commitment model with a hybrid Benders architecture. The first stage is a binary commitment problem: an exact Gurobi Benders master supplies the certified lower bound and an exact master candidate, while Kaiwu local simulated annealing and an optional LLM only generate additional candidates. The second stage is a continuous economic-dispatch subproblem solved by Gurobi. A full deterministic equivalent Gurobi model is included as an external validation baseline.

## Scope

- 10 generators, 24 hourly periods
- three load scenarios:
  - `low`: 0.90 × base load, probability 0.25
  - `base`: 1.00 × base load, probability 0.50
  - `high`: 1.05 × base load, probability 0.25
- 5% spinning reserve
- exact 10×24 base load and generator parameters from the original MATLAB data
- hybrid exact Gurobi Benders master for a certified lower bound and Benders gap
- optional LLM candidate advice, no cloud/QPU calls

> The `high` scenario is capped at 1.05 because 1.10 × peak load plus 5% reserve exceeds the installed capacity of the 10-generator system. If you want a different high scenario, provide your own `scenario_file`.

## Install

```powershell
python -m pip install -r requirements.txt
```

Python 3.10+ is expected.

## Architecture

```text
Gurobi exact Benders master       ->  certified lower bound, exact candidate
Gurobi subproblems (per scenario) ->  dispatch, cuts
Kaiwu SA + repair + decommit + optional LLM -> additional candidates (optional)
```

- The **exact Gurobi Benders master** is the convergence authority. It solves the
  cut relaxation with a continuous `theta`, adds a first-hour ramp-reaching cover,
  and reports Gurobi's proven `ObjBound`. The reported gap is therefore a valid
  Benders gap for the current cut relaxation, not merely a heuristic distance.
- **Kaiwu local simulated annealing** is a candidate generator, not a certified
  master. Its QUBO output is repaired, validated, and then evaluated by Gurobi.
- **LLM advice is optional and heuristic only.** It never enters the lower bound.

The Benders gap is:

```text
gap = (UB - LB) / max(1, abs(UB))
```

## What this version adds

- **Hybrid exact Benders master** (`suc_benders/exact_master.py`) with certified
  lower bound, a first-hour ramp-reaching cover, and a `CONVERGED` stopping rule
- **Independent exact-only mode** that skips Kaiwu/QUBO entirely and seeds the
  first master with an evaluated deterministic feasible commitment
- Feasibility feedback from every failed scenario, not only the first failure
- **Decommitment-first candidate generation**: from the incumbent, greedily turn
  unit-hours OFF, scored by a per-(unit, hour) marginal-cost gradient
- **Decommitment gradient context for the LLM**: `unit_hour_decommit_gradient`,
  `top_decommit_opportunities`, `hourly_commitment_profile`, `peaker_usage`,
  `decommit_ladder`
- Deterministic candidate repair for infeasible Kaiwu schedules
- Local perturbation around repaired candidates and the incumbent
- Reserve constraints scaled to 100 MW in the QUBO energy
- Optional LLM candidate advisor, disabled by default
- A benchmark comparing:
  - quantum-only
  - quantum + repair
  - quantum + repair + local
  - quantum + repair + local + LLM (when `--llm` is passed)

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

Exact-only arm with the certified bound and no Kaiwu/QUBO dependency:

```powershell
python -m suc_benders.benchmark --max-iterations 12 --strategies exact-only
```

Hybrid arm with the certified bound:

```powershell
python -m suc_benders.benchmark --max-iterations 5 --strategies hybrid-exact
```

To include the optional LLM row:

```powershell
python -m suc_benders.benchmark --max-iterations 5 --llm
```

`LLM_API_KEY` must be set before using `--llm`. Without it, the LLM row safely returns no candidates.

## Run

Benders with the hybrid master (exact Gurobi bound plus Kaiwu candidates):

```powershell
python -m suc_benders.run --max-iterations 5
```

Independent exact Benders path without Kaiwu/QUBO:

```powershell
python -m suc_benders.run --max-iterations 12 --no-quantum --no-decommit --no-perturb
```

Kaiwu-only path without the exact master:

```powershell
python -m suc_benders.run --max-iterations 5 --no-exact-master
```

Exact full Gurobi benchmark:

```powershell
python -m suc_benders.full
```

Optional arguments:

```text
--data data/suc_10x24.json
--scenario-file data/scenarios.json
--quiet
--no-exact-master
--no-quantum
--master-time-limit 60
--master-mip-gap 1e-6
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
- The certified gap is valid for the current cut relaxation from the first master
  solve onward, provided the cuts and bound are valid. A small gap is not by itself a
  claim of quantum advantage.
- `solve_full()` remains the external reference optimum. `exact_master` is a MILP
  master relaxation, not the full MIQP.
- The feasibility cut is a generic normalized-shortfall cut derived from Gurobi duals; it is not a specialized strong cut.
- If the master hits its time limit, the reported bound is Gurobi's proven `ObjBound`,
  never the incumbent objective.
- The project is model scaffolding, not a research benchmark.
