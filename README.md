# SUC + Benders (10×24)

This repository contains a small, reproducible two-stage stochastic unit commitment model with optional local and LLM candidate advice. The first stage is a binary commitment problem solved as a QUBO by Kaiwu local simulated annealing. The second stage is a continuous economic-dispatch subproblem solved by Gurobi. A full deterministic equivalent Gurobi model is included as a validation baseline.

## Scope

- 10 generators, 24 hourly periods
- three load scenarios:
  - `low`: 0.90 × base load, probability 0.25
  - `base`: 1.00 × base load, probability 0.50
  - `high`: 1.05 × base load, probability 0.25
- 5% spinning reserve
- exact 10×24 base load and generator parameters from the original MATLAB data
- optional LLM candidate advice, no cloud/QPU calls, and no strict Benders lower-bound/gap claim

> The `high` scenario is capped at 1.05 because 1.10 × peak load plus 5% reserve exceeds the installed capacity of the 10-generator system. If you want a different high scenario, provide your own `scenario_file`.

## Install

```powershell
python -m pip install -r requirements.txt
```

Python 3.10+ is expected.

## What this version adds

- Deterministic candidate repair for infeasible Kaiwu schedules
- Local perturbation around repaired candidates and the incumbent
- Reserve constraints scaled to 100 MW in the QUBO energy
- Optional LLM candidate advisor, disabled by default
- A benchmark comparing:
  - quantum-only
  - quantum + repair
  - quantum + repair + local
  - quantum + repair + local + LLM (when `--llm` is passed)

The local-enhanced path on the 10x24 instance reaches roughly `7.5%` relative error to the exact Gurobi optimum after 5 iterations.

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

To include the optional LLM row:

```powershell
python -m suc_benders.benchmark --max-iterations 5 --llm
```

`LLM_API_KEY` must be set before using `--llm`. Without it, the LLM row safely returns no candidates.

## Run

Benders main problem with Kaiwu local simulated annealing:

```powershell
python -m suc_benders.run --max-iterations 5
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
- The Benders loop reports feasible incumbent objective values, cut counts, and iteration history, but does not claim a certified optimality gap.
- The feasibility cut is a generic normalized-shortfall cut derived from Gurobi duals; it is not a specialized strong cut.
- The project is model scaffolding, not a research benchmark.
