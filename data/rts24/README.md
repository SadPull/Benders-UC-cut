# IEEE RTS-24 derived copperplate UC/SUC

This is a **32-generator, 24-hour derived benchmark**, not the original IEEE
network or a reproduction of the UC+ACOPF formulation. All physical input
parameters are traceable to the pinned public files in `source/`.

MATPOWER `case24_ieee_rts.m` supplies generator limits and quadratic production
costs, fixed startup costs and shutdown costs. Anya Castillo's public
`ucacopf/RTS79_ucacopf.dat` supplies hourly bus demands, initial output and
signed on/off durations, minimum up/down times, ramp rates and startup/shutdown
output limits. Exact URLs, commits, hashes, mappings and transformations are
in `provenance.json`. Source bytes are retained unmodified. Cite both repositories
and the IEEE RTS 1979 / RTS-96 references listed in the MATPOWER source.

Rebuild offline with `python data/rts24/convert.py`; use `--check` to verify
that the checked-in inputs exactly match the converter. The converter checks
source SHA-256 values before parsing and uses decimal arithmetic for aggregation.
No MATLAB execution, Pyomo dependency or network access is needed to rebuild.

## Explicit choices and corrections

- Generator IDs follow MATPOWER's original row numbering. Row 15 is a
  synchronous condenser and is excluded. UC capacity and bus assignments are
  checked against MATPOWER. Total installed capacity is 3405 MW.
- The UC `TimePeriod` declaration omits 15, although `NumTimePeriods=24` and
  every bus has all 24 demand values. The converter uses those complete tables;
  peak aggregate demand is 2850 MW. It does not interpolate a missing load.
- UC power and hourly ramp values are multiplied by `baseMVA=100`. MATPOWER
  quadratic costs are already evaluated in MW and are not rescaled.
- Startup/shutdown output limits are capped at Pmax, so a large source limit
  cannot permit output above the unit rating. Raw MW limits remain in provenance.
- Fixed startup costs come from MATPOWER (1500 per unit per start), not the
  UC data's hot/cold startup schedule. Shutdown costs are zero from MATPOWER.
- The six U50 hydro units have dispatchable output bounds but no water/energy
  budget, matching the chosen UC input representation; this is not reservoir
  scheduling. Transmission, losses, reactive power and contingencies are omitted.
- The project retains unconditional hourly output-difference ramp limits,
  including the transition from the published initial output. Minimum up/down
  constraints truncate at hour 24; shutdown at hour 24 is disallowed. No cyclic
  or end-of-day return-to-initial-state constraint is introduced.

## Two profiles

`deterministic.json` uses the published aggregate curve, probability 1 and zero
reserve (the UC source's reserve requirement). `stochastic.json` uses project
assumptions: 0.90/1.00/1.05 load multipliers, probabilities 0.25/0.50/0.25,
and 5% reserve. These scenarios and probabilities are **not IEEE source data**.
Maximum stochastic demand plus reserve is 3142.125 MW.

The default input is `data/rts24/units.json` with the stochastic profile.
Use `--scenario-file data/rts24/deterministic.json` for the deterministic case;
`--scenario-file data/rts24/stochastic.json` selects the stochastic profile explicitly.
