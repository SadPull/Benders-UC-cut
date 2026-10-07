"""Synchronous Responses transport for the archived whole-day advisor policy."""
from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any, Sequence
from urllib.request import Request, urlopen

import numpy as np


def request_body(d, scenarios, best, o, context, base_candidates, iteration):
    system, user = _prompt(d, scenarios, iteration, best, base_candidates, 4, context)
    return {"model": o.llm_model, "instructions": system, "input": user, "stream": False, "store": False,
            "reasoning": {"effort": "max"}}


def parse_response(response, d, iteration=1):
    info = {"status": "FAILED", "message": "", "accepted": 0, "format_valid": 0, "proposals": []}
    candidates = []
    try:
        if response.get("status") != "completed":
            raise ValueError("Response is not completed")
        text = "".join(part["text"] for item in response["output"] if item.get("type") == "message"
                       for part in item.get("content", []) if part.get("type") == "output_text")
        text = re.sub(r"^\x60\x60\x60(?:json)?\s*|\s*\x60\x60\x60$", "", text.strip())
        parsed = json.loads(text)
        rows = parsed if isinstance(parsed, list) else parsed.get("candidates", [])
        if not isinstance(rows, list):
            raise ValueError("Candidates must be a list")
        for index, row in enumerate(rows[:4]):
            proposal = {"proposal_id": f"llm:{iteration}:{index}", "origin": "llm"}
            info["proposals"].append(proposal)
            try:
                raw = np.asarray(row.get("u"), dtype=float)
                if raw.shape != (d.n_units, d.horizon):
                    raise ValueError("Invalid shape")
                proposal["raw_u"] = raw.tolist()
                info["format_valid"] += 1
                candidates.append((raw, proposal))
            except (TypeError, ValueError, AttributeError):
                proposal["outcome"] = "invalid_matrix"
        info.update(status="OK", accepted=len(candidates))
    except (KeyError, TypeError, ValueError, AttributeError):
        info["message"] = "INVALID_RESPONSE"
    return candidates, info


def request(d, scenarios, best, o, seconds, context, base_candidates, iteration):
    started = perf_counter()
    info = {"status": "FAILED", "message": "", "accepted": 0, "format_valid": 0, "proposals": []}
    try:
        body = request_body(d, scenarios, best, o, context, base_candidates, iteration)
        remaining = seconds - (perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError
        http = Request(o.llm_url, data=json.dumps(body).encode("utf-8"), method="POST",
                       headers={"Content-Type": "application/json", "Authorization": f"Bearer {o.llm_api_key}"})
        with urlopen(http, timeout=min(remaining, o.llm_timeout)) as response:
            candidates, info = parse_response(json.load(response), d, iteration)
        if perf_counter() - started >= seconds:
            candidates = []; info.update(status="TIME_LIMIT", accepted=0)
    except Exception as error:
        candidates = []
        info["message"] = type(error).__name__
    info["seconds"] = perf_counter() - started
    return candidates, info


def _prompt(
    data: Any,
    scenarios: Sequence[Any],
    iteration: int,
    incumbent: Any | None,
    base_candidates: list[Any],
    max_candidates: int,
    context: dict | None = None,
) -> tuple[str, str]:
    system = (
        "You are a stochastic unit-commitment heuristic advisor. You propose binary "
        "on/off schedules only; you are not the optimizer. Never invent dispatch, "
        "theta, dual cuts, load, costs, or constraints. Return JSON only with a "
        "candidates array. Every proposal will be independently validated. "
        "Use context.unit_hour_decommit_gradient and context.top_decommit_opportunities "
        "to identify possible savings. These gradients are local estimates, NOT measured "
        "savings or feasibility certificates. Consider removing entire ON blocks or "
        "trimming their ends, accounting for startup/shutdown costs, minimum up/down "
        "times, initial conditions, ramping and all scenarios. Turning units ON or "
        "substituting units is allowed when beneficial. No measured decommitment ladder "
        "is required. Minimize total expected cost, not the number of ON hours."
    )
    body = {
        "task": "Suggest diverse feasible-looking first-stage commitment schedules.",
        "iteration": iteration,
        "horizon": data.horizon,
        "units": data.n_units,
        "reserve_fraction": data.reserve_fraction,
        "scenarios": [
            {
                "name": scenario.name,
                "probability": scenario.probability,
                "load": scenario.load.tolist(),
            }
            for scenario in scenarios
        ],
        "pmin": data.pmin.tolist(),
        "pmax": data.pmax.tolist(),
        "pstart": data.pstart.tolist(),
        "pshut": data.pshut.tolist(),
        "ramp_up": data.ramp_up.tolist(),
        "ramp_down": data.ramp_down.tolist(),
        "initial_output": data.initial_output.tolist(),
        "min_up": data.min_up.tolist(),
        "min_down": data.min_down.tolist(),
        "initial_on": data.initial_on.astype(int).tolist(),
        "initial_duration": data.initial_duration.tolist(),
        "startup_cost": data.startup_cost.tolist(),
        "shutdown_cost": data.shutdown_cost.tolist(),
        "cost_a": data.cost_a.tolist(),
        "cost_b": data.cost_b.tolist(),
        "cost_c": data.cost_c.tolist(),
        "incumbent_u": incumbent.u.tolist() if incumbent is not None else None,
        "base_u": [candidate.u.tolist() for candidate in base_candidates[:max_candidates]],
        "output_schema": {
            "candidates": [
                {
                    "name": "short_label",
                    "reason": "brief",
                    "u": "units x horizon binary matrix",
                }
            ]
        },
        "rules": [
            "Do not repeat base schedules",
            "Use only 0/1 values",
            "Consider turning OFF redundant unit-hours or substituting cheaper units",
            "Target shoulder hours and the low-load tail of the horizon, where "
            "peaking units are commonly over-committed",
            "Trim the ends of an ON block before removing its middle",
            "Never turn a unit OFF if that would violate reserve or minimum up/down",
            "Return at most the requested number of candidates",
            "Do not include markdown fences",
        ],
        "requested_candidates": max_candidates,
    }
    if context is not None:
        body["context"] = context
    return system, json.dumps(body, separators=(",", ":"), allow_nan=False)
