# Seed 7: fixed SA vs SA+LLM

Batch status: COMPLETED

Each search starts afresh with the same 1800 s budget, initialization and round-robin stopping rules.
Subagent calls have no global count limit; three stagnant complete rounds trigger one request, followed by SA again.
Reference differences use the independent feasible reference cost; solver gaps use each search's own lower bound.

| Run | Status | Cost | Reference difference | Lower bound | Solver gap | Seconds | SA calls | LLM attempts/valid responses/accepted candidates |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D-7 | TIME_LIMIT | 885963.836139 | 0.101363% | 841376.524843 | 5.032633% | 1800.00 | 96 | 4/3/10 |
| A-7 | MAX_ITERATIONS | 925317.577290 | 4.547778% | 841376.524843 | 9.071594% | 1646.23 | 3200 | 0/0/0 |

Initialization and direct improvement contributions:

- D-7: initial cost 933522.2153656807; subsequent improvements {'llm': 47558.3792266181}; physical/fixed-dispatch audit PASSED.
- A-7: initial cost 933522.2153656807; subsequent improvements {'sa': 8204.638075834257}; physical/fixed-dispatch audit PASSED.

Block coverage (visits include local initialization; SA calls are actual sampling):

| Run | Block | Visits | SA calls | First visit (s) | Last local gap |
|---|---:|---:|---:|---:|---:|
| D-7 | 1 | 12 | 12 | 0.638323399998626 | 0.06207397703670093 |
| D-7 | 2 | 12 | 12 | 1.1448695999988558 | 0.17559879912453535 |
| D-7 | 3 | 12 | 12 | 1.5718789999991714 | 4.730446906700354 |
| D-7 | 4 | 12 | 12 | 1.9818393999994441 | 0.03543826026773039 |
| D-7 | 5 | 12 | 12 | 2.3633153999999195 | 0.055139070311932846 |
| D-7 | 6 | 12 | 12 | 2.7054129999996803 | 0.07845825266590495 |
| D-7 | 7 | 12 | 12 | 3.084223199999542 | 0.12580960307833441 |
| D-7 | 8 | 12 | 12 | 3.467343099999198 | 0.10588387455301515 |
| A-7 | 1 | 400 | 400 | 0.6562971999992442 | 0.5843383141234568 |
| A-7 | 2 | 400 | 400 | 1.216293699999369 | 0.48435543354890787 |
| A-7 | 3 | 400 | 400 | 1.6182607999999163 | 0.17146851661376333 |
| A-7 | 4 | 400 | 400 | 2.024278899998535 | 0.05368668837765342 |
| A-7 | 5 | 400 | 400 | 2.436436500000127 | 0.02271524044098687 |
| A-7 | 6 | 400 | 400 | 2.780225900000005 | 0.04800122532818517 |
| A-7 | 7 | 400 | 400 | 3.1493694000000687 | 0.017940251706408377 |
| A-7 | 8 | 400 | 400 | 3.569881199999145 | 0.10815739640774621 |

D-7: coverage 8/8 (PASSED), complete rounds 12, coordination solves 0.


A-7: coverage 8/8 (PASSED), complete rounds 400, coordination solves 20.


Coverage acceptance: PASSED.

SA+LLM cost reduction relative to SA: 4.252998%.

Reference: {'objective': 885066.7044303418, 'lower_bound': 885066.7044303424, 'status': 'OPTIMAL', 'elapsed_seconds': 92.5252713000009}.

The LLM transport is fresh Codex Astra/max subagents, with no HTTP requests. A valid response need not yield a feasible or improving candidate. Calls are not forced when the trigger is not reached.
Only candidates evaluated and validated before the search deadline enter the result. Fixed-dispatch audits run separately after each search.
SA directly changes one block per batch; the subagent can propose a whole-day schedule. Repair may change hours outside a block. This compares complete methods under equal wall-clock budgets, not equal-scope candidate generators.
The historical 4.3112% reduction came from a different protocol that visited only the first block; it is not evidence for the repaired comparison.
This is one seed and one paired observation, not evidence of statistical significance, general speedup or bidirectional synergy.
