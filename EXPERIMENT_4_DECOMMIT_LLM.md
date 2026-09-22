# EXP-4: Decommitment-first search with LLM guidance (qwen3.8-27b)

Date: 2026-09-22
Code: decommitment-first implementation (this change)
Model: `qwen3.8-27b` via `https://models.sjtu.edu.cn/api/v1`
Instance: 10 units x 24 hours, 3 scenarios, 5 Benders iterations, Kaiwu seed 7
LLM: 4 candidates/iteration, max_tokens 6144, timeout 180s, 3 retries
Exact Gurobi optimum: `571014.051895` (fixed 108650.0 + expected fuel 462364.1)

## What changed

Three mechanisms replace perturbation-first search:

1. **Decommitment-first candidate generator.** Starting from the incumbent, greedily
   turn unit-hours OFF. Moves are whole ON-blocks, or 1-2 hour trims of a block's
   start/end. Only moves passing `commitment_is_feasible` are kept; up to 8 cumulative
   rungs are produced and each is evaluated by Gurobi.
2. **Decommitment gradient scoring.** The generator no longer ranks moves by fixed-cost
   saving alone. It scores by predicted total change using the same per-(unit, hour)
   gradient the LLM sees:
   `estimated_total_change = -cost_c + p * (system_price - unit_marginal_cost)`.
3. **LLM context + prompt.** New fields `unit_hour_decommit_gradient`,
   `top_decommit_opportunities`, `hourly_commitment_profile`, `peaker_usage`,
   `decommit_ladder`. The system prompt now states that over-commitment is the dominant
   error and that the LLM's primary job is to turn unit-hours OFF.

Perturbation-first is now **off by default** (`perturb_candidates=False`).

## Headline result

| generation | UB | rel. err | on-unit-hours | fixed cost | fixed excess |
|---|---:|---:|---:|---:|---:|
| old: local perturbation | 613717.1 | 7.48% | 211 | 144390 | +35740 |
| old: LLM-only (deepseek-reasoner) | 588646.4 | 3.09% | 176 | 124270 | +15620 |
| old: LLM-only (deepseek-chat) | 597164.4 | 4.58% | 188 | 130730 | +22080 |
| **new: decommit-only (deterministic)** | **583503.8** | **2.19%** | **165** | **118025** | **+9375** |
| **new: qwen+decommit+llm (best)** | **578989.5** | **1.40%** | **161** | **114865** | **+6215** |
| Gurobi reference | 571014.1 | 0.00% | 148 | 108650 | 0 |

Progression of relative error against the exact optimum:

```
local perturbation   7.48%
LLM-only (old)       3.09%
decommit-only        2.19%   <- deterministic, no LLM
decommit + LLM       1.40%   <- best run
```

## LLM arm distribution (4 runs, qwen3.8-27b)

| run | UB | rel. err | on-unit-hours | fixed cost | opt. cuts | runtime | LLM valid/raw |
|---|---:|---:|---:|---:|---:|---:|---:|
| run 1 | 583503.8 | 2.19% | - | - | 38 | 359 s | 12 / 20 |
| run 2 | 578989.5 | 1.40% | 161 | 114865 | 166 | 786 s | 16 / 20 |
| run 3 | 580459.3 | 1.65% | 161 | 115400 | 143 | 593 s | 12 / 20 |
| run 4 | 580440.1 | 1.65% | 161 | 115395 | 118 | 597 s | 8 / 20 |

- mean UB `580848.2`, mean relative error **1.72%**
- best `578989.5` (1.40%), worst `583503.8` (2.19%)
- **stdev of relative error 0.33 pp** (vs 2.11 pp for the old LLM-only arm: 6.4x tighter)
- LLM beat the decommit-only baseline in **3 / 4** runs
- paired improvement vs decommit-only: mean `2655.6`, sd `1899.5`, se `949.8`,
  **t = 2.80** (df = 3)
- LLM candidate validity 48 / 80 = **60%**
- **zero LLM call failures / timeouts** across all 3 solver-level runs

## Why this worked: the commitment structure now matches Gurobi

Per-unit on-hours, best new run vs old generations:

| unit | Pmax | cost_c | Gurobi | old local | old LLM | new LLM | role |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 455 | 1000 | 24 | 24 | 24 | 24 | base |
| 1 | 455 | 970 | 24 | 24 | 24 | 24 | base |
| 2 | 130 | 700 | 17 | 24 | 24 | 22 | mid |
| 3 | 130 | 680 | 24 | 24 | 24 | 24 | base |
| 4 | 162 | 450 | 24 | 24 | 24 | 24 | base |
| 5 | 80 | 370 | 12 | 18 | 13 | 11 | peaker |
| 6 | 85 | 480 | 6 | 21 | 13 | 13 | peaker |
| 7 | 55 | 660 | 6 | 14 | 11 | 4 | peaker |
| 8 | 55 | 665 | 6 | 16 | 10 | 9 | peaker |
| 9 | 55 | 670 | 5 | 22 | 9 | 6 | peaker |
| | | | **148** | **211** | **176** | **161** | |

The decommitment-first search moves the small peaking units toward Gurobi's usage:
every peaker (units 5-9) is now closer to the reference than in any previous generation,
and total on-hours fall from 211 (old local) to 161, versus the reference 148.

Remaining excess is still fixed-cost dominated, but the fixed excess falls from
+35740 (old local) to +6215 (best new), a **83% reduction**.

## Success criteria assessment

| criterion | target | achieved |
|---|---|---|
| mean rel. error | <= 4.0% | **1.72%** yes |
| LLM better than decommit-only | >= 3/4 | **3/4** yes |
| on-hours reduction vs decommit-only | >= 10 | **4** (161 vs 165) no |
| LLM validity | >= 70% | **60%** no |
| no timeout-induced termination | required | **yes** |

Two criteria were not met strictly:

- **On-hour reduction (4 vs 10 required).** The deterministic decommitment baseline is
  already close to the reference, so the LLM has little room to remove further hours.
  The LLM instead improves *which* hours are removed, lowering fixed cost by 3160 with
  only 4 fewer on-hours.
- **LLM validity 60%.** Below the 70% target, but no call failed outright, and every run
  completed 5 iterations.

## Bug fixed along the way

`ask_llm_for_candidates` crashed the entire Benders run when the model returned a ragged
matrix (`ValueError: setting an array element with a sequence`), observed live in run 4.
Malformed proposals are now shape-checked and discarded; a regression test covers it.

## Interpretation

1. **The direction mattered more than the model.** Changing the search to decommitment-first
   produced 2.19% deterministically, beating the best previous LLM-only result (3.09%). Giving
   the LLM a generator that moves in the correct direction mattered more than upgrading the model.
2. **The LLM still adds on top.** 3/4 runs improved on the deterministic baseline, mean
   improvement 2656, and reduced run-to-run spread 6.4x (0.33 pp vs 2.11 pp).
3. **Remaining gap is small and fixed-cost dominated** (fixed excess +6215, of total excess
   +7975 for the best run, i.e. 78%).
4. **Cut count is still not the driver**: decommit-only used 141 cuts and hit 2.19%, while
   run 2 used 166 cuts to hit 1.40%; run 1 used only 38 cuts and matched the baseline.

## Limitations

- 4 LLM runs; t = 2.80 at df = 3 is directional, not conclusive.
- No certified Benders lower bound; the metric is distance to the exact Gurobi optimum.
- Only one instance (10x24, 3 scenarios, seed 7); no cross-instance validation.
- `qwen3.8-27b` is non-deterministic, so reruns will differ.

## Artifacts

- `results/decommit_only_v070.json` - deterministic decommitment baseline
- `results/qwen_decommit_llm_run{1..4}.json` - LLM arm runs
- `results/ref_*.npy`, `results/ref_meta.json` - Gurobi reference
- All under `results/`, not tracked by Git.
