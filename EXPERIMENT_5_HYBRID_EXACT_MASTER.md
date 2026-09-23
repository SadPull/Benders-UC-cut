# EXP-5: Hybrid exact Benders master (certified bound)

Date: 2026-09-23
Instance: 10 units x 24 hours, 3 scenarios, seed 7
Exact Gurobi optimum: `571014.051895`

## What changed

`v0.8.0` adds `suc_benders/exact_master.py`. The Benders master is now solved
exactly by Gurobi each iteration; Kaiwu SA and the optional LLM are demoted to
candidate generators. This makes the reported gap a certified Benders gap over
the accumulated cut set rather than a distance to an external optimum.

## Certified bound is valid but the gap is UB-side

Running the hybrid arm for 12 iterations:

| iteration | LB | UB | Benders gap |
|---|---:|---:|---:|
| 1 | 105620.0 | 615086.6 | 82.8% |
| 2 | 569038.9 | 603033.8 | 5.6% |
| 3 | 569050.6 | 592621.7 | 4.0% |
| 12 | 569197.0 | 592621.7 | 3.95% |

The bound is legitimate: `LB < true optimum` throughout, and it sits only
**0.32%** below the true optimum (`569197.0` vs `571014.05`). The remaining
3.95% gap is almost entirely the **incumbent's fault (UB side)**, not a weak
master relaxation.

## Why the bound alone cannot close the gap

Two independent checks:

1. **A cut from the true optimum does not raise the bound much.**
   Building the aggregated optimality cut at Gurobi's own optimal commitment and
   solving the exact master with only that cut gives `LB = 566414.87`, i.e. an
   `0.81%` gap. Even a perfect commitment's own cut is not a tight bound,
   because the affine subgradient underestimates convex `q_s(x)` away from `x0`.

2. **Heuristic cuts already beat it.** The 33-39 cuts accumulated by the hybrid
   loop reach `LB = 569197.0`, which is *tighter* than the single reference cut.
   Adding more cuts is still useful, but with sharply diminishing returns.

So the residual error is not a missing-cut problem. It is the same
commitment-search problem identified in `ANALYSIS_WHY_GAP_PERSISTS.md`: the
incumbent over-commits peaking units, and no amount of master strengthening can
lower the UB.

## Practical consequence

Closing the gap further requires improving the **incumbent** (UB), for example:

- exact MIP polish in a Hamming neighbourhood of the incumbent;
- stronger structured decommitment moves around the peak-to-shoulder transition;
- letting the exact master candidate seed an exact local search.

Increasing `max_iterations` with the current generator mix converges the bound
slowly (569039 -> 569197 over 10 more iterations) while the UB plateaus.

## Verified properties

`tests/test_exact_master.py` and `tests/test_solver_exact_master.py` assert:

- the master candidate is discrete-feasible;
- an aggregated optimality cut never exceeds the true expected subproblem cost;
- adding cuts never decreases the bound;
- `LB <= true optimum` and `LB <= UB`;
- a truncated master reports `ObjBound`, never the incumbent objective;
- `--no-exact-master` restores the previous path.
