"""Deterministic ON-block removal and end trimming using the LLM's context."""
import time

import numpy as np

from .candidates import commitment_hash, make_candidate, repair_commitment
from .common import commitment_is_feasible
from .llm import LLMAdviceResult


def ranked_moves(data, incumbent, context):
    # Use exactly the rounded/truncated gradient table shown to the LLM.
    # Unlisted hours have zero estimated fuel change, retaining known fixed cost.
    gradient = {(r["unit"], r["period"]): r["estimated_total_change"]
                for r in context["unit_hour_decommit_gradient"]}
    moves = []
    for i, row in enumerate(incumbent.u):
        edges = np.diff(np.r_[0, row, 0])
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            intervals = [(start, stop)]
            intervals += [(start, t) for t in range(start+1, stop)]
            intervals += [(t, stop) for t in range(start+1, stop)]
            for left, right in intervals:
                trial = row.copy()
                trial[left:right] = 0
                prev = np.r_[data.initial_on[i], trial[:-1]]
                starts = np.maximum(trial-prev, 0).sum()-incumbent.v[i].sum()
                stops = np.maximum(prev-trial, 0).sum()-incumbent.w[i].sum()
                score = sum(gradient.get((i, t), -float(data.cost_c[i])) for t in range(left, right))
                score += starts*data.startup_cost[i]+stops*data.shutdown_cost[i]
                moves.append((float(score), i, int(left), int(right)))
    return sorted(moves)


def rule_candidates(data, scenarios, incumbent, context, prior_ids, call_index, limit, deadline):
    diagnostics = dict(call_id=f"rule:{call_index}", proposals=[], repair_seconds=0., usage=None)
    candidates, seen = [], set(prior_ids)
    for score, i, left, right in ranked_moves(data, incumbent, context):
        if time.perf_counter() >= deadline or len(candidates) >= limit:
            break
        raw = incumbent.u.copy()
        raw[i, left:right] = 0
        raw_candidate = make_candidate(data, scenarios, raw)
        phase = time.perf_counter()
        repaired = repair_commitment(data, scenarios, raw)
        diagnostics["repair_seconds"] += time.perf_counter()-phase
        proposal = dict(proposal_id=f"rule:{call_index}:{len(diagnostics['proposals'])}",
            origin="rule", raw_u=raw.tolist(), estimated_change=score,
            unit=i, start=left, stop=right, repair_attempted=True,
            raw_discrete_feasible=commitment_is_feasible(data, scenarios,
                raw_candidate.u, raw_candidate.v, raw_candidate.w),
            outcome="repair_failed", repaired_discrete_feasible=repaired is not None)
        diagnostics["proposals"].append(proposal)
        if repaired is None:
            continue
        identity = commitment_hash(repaired)
        proposal.update(commitment_id=identity, repaired_u=repaired.u.tolist(),
                        hamming_u=int(np.count_nonzero(raw != repaired.u)))
        proposal["outcome"] = "duplicate" if identity in seen else "accepted"
        if identity in seen:
            continue
        seen.add(identity)
        repaired.source = "rule"
        candidates.append(repaired)
    diagnostics.update(raw_count=len(diagnostics["proposals"]), valid_count=len(candidates))
    return LLMAdviceResult(candidates, diagnostics["raw_count"], len(candidates),
                          message="ok", diagnostics=diagnostics)
