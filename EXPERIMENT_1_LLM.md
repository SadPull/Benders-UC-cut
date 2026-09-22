# EXP-1: Real-LLM benchmark on 10x24 SUC (v0.4.0)

Date: 2026-09-22
Code: commit `fbdf2b5` (tag `v0.4.0`)
Model: `qwen3.8-27b` via `https://models.sjtu.edu.cn/api/v1`
Config: 10 units x 24 hours, 3 scenarios, 5 Benders iterations, Kaiwu seed 7
Exact Gurobi optimum: `571014.051895`

## Purpose

Test whether the P0/P1 context injected in v0.4.0 makes the LLM candidate
advisor actually improve the incumbent, after v0.3.0 showed no gain.

## Results

| Strategy | UB | rel. err vs optimum | opt. cuts | runtime |
|---|---:|---:|---:|---:|
| quantum-only | 631821.24 | 10.65% | 1 | 3.9 s |
| quantum + repair | 617605.62 | 8.16% | 2 | 5.8 s |
| quantum + repair + local | 613717.14 | 7.48% | 67 | 134.3 s |
| **quantum + repair + local + LLM** (mean of 4 runs) | **600403.95** | **5.15%** | 86 | 393 s |

LLM run distribution (4 independent runs, same seed and protocol):

| Run | UB | rel. err | opt. cuts | LLM raw/valid | LLM timeouts |
|---|---:|---:|---:|---:|---:|
| benchmark | 599323.87 | 4.96% | 84 | 16 / 11 | 0 |
| repeat 1 | 592115.31 | 3.70% | 85 | 16 / 12 | 1 |
| repeat 2 | 609367.46 | 6.72% | 87 | 8 / 7 | 3 |
| repeat 3 | 600809.16 | 5.22% | 88 | 16 / 13 | 1 |

Aggregate:

- mean UB `600403.95` (5.147%), best `592115.31` (3.695%), worst `609367.46` (6.717%)
- sample stdev of UB `7079.94`; stdev of relative error `1.240` pp
- paired improvement vs local baseline: mean `13313.19`, sd `7079.94`,
  se `3539.97`, t = 3.76 (df = 3)
- LLM beat the local baseline in **4 / 4** runs
- relative reduction of excess-over-optimal: **31.2%**
- LLM candidate validity overall: 43 / 56 = **77%**

## Attribution

The local-only control reproduces exactly and gives a natural experiment:

| Run | LLM call at iteration 1 | iteration-1 incumbent |
|---|---|---:|
| local-only control | not used | 616773.760 |
| repeat 1 | ok (3/4 valid) | 599229.545 |
| repeat 3 | ok (4/4 valid) | 603001.066 |
| repeat 2 | **timeout** | 616773.760 |

When the iteration-1 LLM call timed out, the iteration-1 incumbent equals the
local-only value exactly, and the run still ends better than local because the
LLM succeeded in later iterations. When the call succeeded, the incumbent drops
immediately by 13-17k. This isolates the gain to the LLM candidates rather than
to any other change in the pipeline.

## Comparison with v0.3.0

v0.3.0 (LLM without P0/P1 context) returned UB `613736.15` (7.482%) and was
slightly worse than the local baseline `613717.14` (7.478%). Adding the context
moved the LLM from no-gain to a repeatable `~5.1%` mean.

## Known limitations

- LLM calls are non-deterministic; 4 runs is enough for a directional reading,
  not for a tight confidence interval. The t interval at df=3 is wide.
- Timeouts (60 s default) are the dominant failure mode and clearly cost
  quality: repeat 2 lost 3 of 5 calls and finished worst. Raising
  `LLMConfig.timeout_seconds` is the obvious next lever.
- No certified Benders lower bound is reported, so the headline gap is the
  distance to the exact Gurobi optimum, not a Benders gap.
- Raw per-run artifacts live in `results/` and are not tracked by Git.
