# Pure-SA gap v3 protocol

Approved experiment budget: 14400 seconds, serial, single thread. Development and
regression tests are outside this budget. Old experiments are immutable.

SA variants forbid native integer candidates and online integer certification.
Gurobi continuous preprocessing, dispatch, and coordination LPs remain enabled.
Integer certification runs in a separate process after search and never feeds
candidate generation, multipliers, or stopping conditions. Reference costs and
schedules are not inputs to search workers.

The fixed matrix and configuration live in `pure_sa_gap_experiment.py`. Seeds are
7, 100007, 200007. Final search slots last 1200 seconds, with a separate 600-second
checkpoint and at most 600 seconds of offline certification per search. Report
search time and certification time separately; posthoc certificates are not
online achievements at their source snapshot timestamps.

`sa_strict` preserves the old inner gate with continuous bounds only. `sa_flow`
visits every block with eight inner calls, permits explicitly marked inexact
coordination from validated feasible points, and recovers at most two schedules
per completed outer round. It initializes every block with a valid analytic dual
floor using initial local dual prices and current cross-block prices. Cuts are
rebuilt and retained candidates repriced after every multiplier change.

`sa_compact` adds exact interval propagation, safe dynamic encoding, and sixteen
active cuts. All original cuts remain in the continuous and offline certificate
models. `sa_full` additionally selects up to four novel, diverse candidates from
sixteen SA samples. SA parameters, penalty, and theta precision remain unchanged.

Select block length by valid certified lower bound, excluding only numerical
failures and ceilings proved by valid classical coordination upper bounds below
0.99R. Ties within 1e-5 select shorter blocks. Select among flow/compact/full by
certified lower bound; ties select that order. Freeze both selections before
comparison and confirmation. Timeout is not a proof of an unreachable target.

Primary acceptance is all three final seeds with a valid feasible schedule,
verified source/input hashes, zero online integer-master calls, and certified
(R-L)/R <= 0.01. Challenge threshold is 0.005. Report dual convergence separately.
