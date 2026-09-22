# EXP-3: Why the gap does not close -- structural analysis of Gurobi vs LLM/QUBO solutions

Date: 2026-09-22
Instance: 10 units x 24 hours, 3 scenarios, seed 7
Exact Gurobi optimum: `571014.051895` (fixed 108650.0 + expected fuel 462364.1)

Solutions compared:

| source | UB | rel. err vs optimum |
|---|---:|---:|
| Gurobi reference | 571014.1 | 0.00% |
| deepseek-reasoner (best run, r3) | 588646.4 | 3.09% |
| deepseek-chat | 597164.4 | 4.58% |
| local-only (Kaiwu + repair + perturbation) | 613717.1 | 7.48% |

## Headline finding

The gap is **not** a fuel/dispatch problem and **not** a missing-Benders-cut problem.
It is a **first-stage commitment-count problem**, and it is dominated by **fixed cost**:

| method | excess over optimum | from fixed cost | from fuel cost |
|---|---:|---:|---:|
| reasoner r3 | 17632.3 | 15620.0 (88.6%) | 2012.3 (11.4%) |
| chat | 26150.3 | 22080.0 (84.4%) | 4070.3 (15.6%) |
| local-only | 42703.1 | 35740.0 (83.7%) | 6963.1 (16.3%) |

Every heuristic solution **over-commits units**. On-unit-hours:

| method | on-unit-hours | vs reference |
|---|---:|---:|
| Gurobi reference | 148 | 0 |
| reasoner r3 | 176 | +28 |
| chat | 188 | +40 |
| local-only | 211 | +63 |

The fuel cost is nearly right in all cases (within 0.4-1.5% of the optimum). What the
heuristics get wrong is **keeping expensive peaking units online longer than necessary**.

## The structural difference, concretely

Gurobi treats the five small units (5-9) as **peakers**: it switches them on only for the
demand peak, and switches them off again immediately.

```
unit c_c  012345678901234567890123
   5  370 ref #......########...###...
   6  480 ref ........######..........
   7  660 ref .........####......##...
   8  665 ref .........####......##...
   9  670 ref ..........###......##...
```

The heuristic solution keeps them on across a wider window:

```
   5  370 rea .......#######....######
   6  480 rea .......#######....######
   7  660 rea ........######.....#####
   8  665 rea .........#####.....#####
   9  670 rea ..........####.....#####
```

The excess is concentrated in the shoulder hours t=21,22,23 (i.e. the *end* of the day):

| t | ref ON | heuristic ON | extra | high-scenario load |
|---|---:|---:|---:|---:|
| 21 | 5 | 10 | +5 | 1155 |
| 22 | 4 | 10 | +6 | 945 |
| 23 | 4 | 10 | +6 | 840 |

At t=22-23 the load is only 945/840 MW (high scenario), the reference runs 4 units, and the
heuristic runs all 10. That is pure wasted fixed cost with no reserve justification
(reserve slack at t=23: reference 320 MW, reasoner 780 MW, local 715 MW).

## Why the search cannot fix this

Three independent reasons, all verified:

### 1. The feasible region is genuinely rugged

The reference differs from the best heuristic solution by **32 unit-hour flips**. Testing each
single flip individually: only **14/32** remain feasible. So no sequence of single flips
reaches the reference, and simple neighbourhood search cannot walk there.

The blocked flips are the ones that matter most (turning unit 2 off at t=0..4), which violate
min-up/min-down.

### 2. Local search moves far too slowly

The local-only trajectory gains ~750/hour-step per iteration:

```
it1 616773.8
it2 615963.8   -810
it3 615216.8   -747
it4 614494.8   -722
it5 613717.1   -778
```

Total improvement 3057 over 5 iterations, while the target is 42703 away. At this rate the
run would need ~55+ iterations to reach the optimum, and it is still in the wrong basin.

### 3. The QUBO's own objective does prefer the reference

I rebuilt the master with optimality cuts generated from BOTH the reference and the heuristic
solution, and evaluated the master objective `fixed + max_k cut_k(x)`:

| point | fixed | theta (cut max) | master objective |
|---|---:|---:|---:|
| reference | 108650.0 | 462364.1 | **571014.1** |
| reasoner r3 | 124270.0 | 464376.4 | 588646.4 |

So when the cut set contains the reference's own cut, the master objective correctly ranks the
reference as better by 17632. **The model is not wrong. The solver cannot find the optimum
that the model already prefers.** This rules out cut deficiency as the primary cause and
confirms it is a search/representation problem.

## Why the over-commitment is attractive to the search

Two mechanisms push every heuristic toward over-commitment:

1. **Reserve constraint is nearly binding everywhere.** At the high-scenario peak,
   total Pmax (1662) exceeds the reserve requirement (1653.75) by only **8.25 MW**. Any unit
   off at t=11 is infeasible. Penalty-based QUBO formulations satisfy this by keeping units on
   rather than by carefully scheduling, because "leave it on" is always feasible.

2. **Over-committing is the safe direction.** Turning a unit on costs `cost_c` per hour; turning
   it off risks an infeasible subproblem and a feasibility cut. Under a penalty formulation,
   the expected cost of an infeasibility penalty (1e6) dwarfs the fixed cost of keeping a small
   unit on (55-85 MW units cost only 370-670/hour). So the energy landscape tilts decisively
   toward over-commitment.

Measured reserve slack (high scenario, capacity minus requirement):

| method | mean slack |
|---|---:|
| Gurobi reference | 116.9 MW |
| reasoner r3 | 225.2 MW |
| chat | 262.1 MW |
| local-only | 316.5 MW |

The heuristics buy reliability with commitment.

## Why the LLM helps but does not solve it

The LLM produces **large jumps** that local search cannot:

```
it1 613772.8
it2 592469.9   -21303   <- LLM-driven jump
it3 588646.4    -3824
it4 588646.4        0   <- stalls
it5 588646.4        0
```

It gets into a much better basin immediately, then **stalls at the same structural
limitation**: the candidate it proposes still over-commits at t=21-23, and its remaining error
(17632) is 88.6% fixed cost from exactly those extra on-hours.

The LLM is not being told the marginal cost of an extra on-hour, nor which hours are
over-committed relative to a tight solution, so it optimizes load-following (which it can see)
rather than commitment economy (which it cannot).

## Conclusion

The gap persists because:

1. **It is a fixed-cost problem, 84-89% of the excess**, driven by keeping 5 small peakers
   online ~28-63 unit-hours too long, concentrated in the t=21-23 shoulder.
2. **The model is not the bottleneck** - with the right cuts, the master objective already
   ranks the optimum correctly by 17632.
3. **The search is the bottleneck** - the reference is 32 flips away with only 14/32 feasible
   single flips, and local search moves ~750/hour-step against a 42700 target.
4. **The QUBO energy tilts toward over-commitment** because the reserve constraint is nearly
   binding at the peak (8.25 MW slack) and infeasibility penalties dominate fixed costs.
5. **More LLM candidates alone will not fix it** - the LLM stalls at the same structural
   limitation after 3 iterations.

## Actionable implications (not yet implemented)

- Give the LLM explicit **per-hour marginal cost of an on-unit-hour** and the **decommitment
  opportunity** at each hour, rather than only the incumbent matrix.
- Add a **commitment-count gradient** signal: for each (unit, hour), the change in fixed cost
  and expected fuel cost of turning it off, i.e. a local search table the LLM can read.
- Consider a **decommitment-first** candidate generator (start from the incumbent and greedily
  remove on-hours subject to feasibility) instead of perturbation-first.
- The t=21-23 shoulder is where every method fails; a targeted repair that trims the tail of
  the horizon would recover most of the fixed-cost excess.

## Artifacts

- `results/ref_u.npy`, `ref_v.npy`, `ref_w.npy`, `ref_p.npy`, `ref_meta.json` - Gurobi reference
- `results/reasoner_iso_run3.json` - best reasoner run (3.09%)
- `results/chat_iso_run1.json` - deepseek-chat run (4.58%)
- `results/local_only_v0.4.0.json` - local-only run (7.48%)
- All under `results/`, not tracked by Git.
