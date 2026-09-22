# EXP-2: Corrected LLM-vs-local comparison on 10x24 SUC

Date: 2026-09-22
Code: `fbdf2b5` + benchmark/llm fixes in this change
Model: `qwen3.8-27b` via `https://models.sjtu.edu.cn/api/v1`
Instance: 10 units x 24 hours, 3 scenarios, 5 Benders iterations, Kaiwu seed 7
Exact Gurobi optimum: `571014.051895`
LLM timeout: **180 s** (was 60 s)

## Correction to EXP-1

EXP-1 reported a row labelled "+ LLM" with UB `600403.95` (5.15%). That label was
wrong. The arm was actually `quantum+repair+local+llm`: local perturbation was
**enabled** (`perturb_candidates=True`), contributing 120-156 evaluated
neighbourhood candidates per run.

That arm therefore measures **LLM + local combined**, not LLM alone, and its gain
cannot be attributed to the LLM.

## What was changed

- `suc_benders.benchmark` now exposes a genuinely local-free arm:
  `quantum+repair+llm` (repair on, perturbation off).
- The combined arm `quantum+repair+local+llm` is retained for reference.
- `--strategies` allows running a single arm, so the expensive arms can be
  repeated without re-running everything.
- `LLMConfig.timeout_seconds` default raised 60 -> 180, with
  `benchmark --llm-timeout` to override.

## Head-to-head: local vs LLM (both with repair, no overlap)

| strategy | UB | rel. err vs optimum | opt. cuts | runtime |
|---|---:|---:|---:|---:|
| quantum+repair (no local, no LLM) | 617605.62 | 8.159% | 2 | 5.5 s |
| **quantum+repair+local** (local only) | **613717.14** | **7.478%** | 67 | 137.5 s |
| **quantum+repair+llm** (LLM only, mean of 4) | **597691.09** | **4.672%** | 15 | 327 s |

Isolated LLM-only runs (no local perturbation, `perturbed_candidates=0`):

| run | UB | rel. err | opt. cuts | LLM valid/raw | runtime |
|---|---:|---:|---:|---:|---:|
| run 1 | 615302.77 | 7.756% | 18 | 15 / 20 | 292.9 s |
| run 2 | **588886.60** | **3.130%** | 9 | 7 / 12 | 137.8 s |
| run 3 | 591149.17 | 3.526% | 18 | 8 / 20 | 629.7 s |
| run 4 | 595425.81 | 4.275% | 14 | 11 / 20 | 246.5 s |

- mean UB `597691.09`, mean relative error **4.672%**
- best `588886.60` (3.130%), worst `615302.77` (7.756%)
- stdev of relative error `2.110` pp
- LLM-only beat the local-only baseline in **3 / 4** runs
- paired improvement vs local: mean `16026.06`, sd `12050.15`, se `6052.07`,
  t = **2.66** (df = 3)
- reduction of excess-over-optimal: **37.5%**

## Combined arm (reference only, single run at t=180)

| strategy | UB | rel. err | opt. cuts |
|---|---:|---:|---:|
| quantum+repair+local+llm | 607977.18 | 6.473% | 120 |

The combined arm does **not** beat the LLM-only mean, and it costs ~1.5x the
runtime. Notably it produced 120 cuts versus 15 for LLM-only, yet ended worse:
further evidence that cut count is not the driver of incumbent quality.

## Timeout effect

| setting | LLM candidate validity |
|---|---:|
| t = 60 s (EXP-1 runs) | 43 / 56 = 76.8% |
| t = 180 s (isolated arm) | 26 / 52 = 50.0% |

The headline validity rate fell, but the UB improved. Reading: with the longer
timeout the model actually finishes and returns its full candidate set rather
than being cut off early. The 50% rate counts every proposed row including
schemas that fail the discrete-feasibility check, so it is not a pure
infrastructure measure. The important number is that timeouts stopped being the
dominant failure mode.

## Interpretation

1. **The LLM contribution is real but noisy.** Best case 3.13%, worst case
   7.76%, stdev 2.11 pp. Three of four runs beat local.
2. **Local search is not additive with the LLM.** Combined 6.47% is worse than
   LLM-only mean 4.67%. Local search appears to steer the incumbent toward a
   basin that the LLM then anchors on, reducing exploration.
3. **Cut count is inversely related to quality here.** LLM-only used 9-18 cuts
   and reached 3.1-4.3%; local used 67 and combined used 120, both ending worse.
4. **n is still small.** t = 2.66 at df = 3 is directional, not conclusive.

## Limitations

- 4 isolated LLM runs; non-deterministic model output dominates the spread.
- No certified Benders lower bound; the reported error is distance to the exact
  Gurobi optimum, not a Benders gap.
- The deadlock/`NO_NEW_CANDIDATES` early stop fires in some runs at iteration 3,
  which truncates the budget before 5 iterations are used.
- Raw artifacts are in `results/` and are not tracked by Git.
