"""Optional synchronous Responses advisor; proposals require numerical checks."""
import json
import re
from time import perf_counter
from urllib.request import Request, urlopen

import numpy as np


def request_body(b, best, lower, o, feedback):
    d = b.d
    context = {"hours": b.times + 1, "unit_count": d.n_units, "current_lower_bound": lower,
               "current_cost": best.cost, "u": best.point[b.u], "v": best.point[b.v], "w": best.point[b.w],
               "priced_binary_cost": b.obj[b.binary], "original_binary_cost": b.original_obj[b.binary],
               "lagrangian_binary_price": b.obj[b.binary] - b.original_obj[b.binary],
               "priced_continuous_cost": b.obj[b.continuous], "quadratic_cost": b.q[b.continuous],
               "block_constant": b.constant, "reserve_fraction": d.reserve_fraction,
               "current_point": best.point, "recent_candidates": feedback.get("recent", []),
               "cut_definition": "optimality: recourse >= alpha + beta*binary; feasibility: alpha + beta*binary <= 0"}
    for name in ("pmin", "pmax", "pstart", "pshut", "min_up", "min_down", "ramp_up", "ramp_down",
                 "initial_on", "initial_duration", "initial_output", "cost_a", "cost_b", "cost_c", "startup_cost", "shutdown_cost"):
        context[name] = getattr(d, name)
    context["scenario_loads"] = np.asarray([s.load[b.times] for s in b.scenarios])
    context["scenario_reserve"] = d.reserve_fraction * context["scenario_loads"]
    context["scenario_probability"] = [s.probability for s in b.scenarios]
    coo = b.A.tocoo()
    context["local_constraints"] = {"triplets": np.column_stack([coo.row, coo.col, coo.data]),
        "rhs": b.rhs, "equality": b.eq, "lb": b.lb, "ub": b.ub, "binary": b.binary,
        "continuous": b.continuous, "u": b.u, "v": b.v, "w": b.w,
        "global_columns": b.cols, "one_based_indices": False}
    context["cuts"] = [{"alpha": c.alpha, "beta": c.beta, "optimality": c.optimality}
                       for c in feedback.get("cuts", [])]
    instructions = (
        'Propose up to 3 alternative binary schedules for this stalled SA search. '
        'Return ONLY JSON: {"candidates":[{"u":[...],"v":[...],"w":[...]}]}. '
        'Each matrix has unit_count rows and one column per listed hour, with entries 0 or 1. '
        'u is commitment, v startup and w shutdown. Minimize the priced block objective. '
        'Block gains need not improve full-day cost. Cut slopes are sensitivities and lower bounds, '
        'not guaranteed savings. Triplets encode A*x <= rhs except equality rows. '
        'All vector/matrix indices are ZERO BASED; hours are calendar hours starting at 1. '
        'Only listed local constraints are hard; cross-block constraints are priced. '
        'Do not reset initial conditions at block boundaries. All proposals will be checked '
        'against physical constraints and dispatch QP.')
    return {"model": o.llm_model, "instructions": instructions,
            "input": json.dumps(context, default=_json, allow_nan=False), "stream": False, "store": False}


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def parse_response(response, b):
    info = {"status": "FAILED", "message": "", "accepted": 0, "format_valid": 0}
    candidates = []
    try:
        if response.get("status") != "completed":
            raise ValueError("Response is not completed")
        text = "".join(part["text"] for item in response["output"] if item.get("type") == "message"
                       for part in item.get("content", []) if part.get("type") == "output_text")
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        rows = json.loads(text)["candidates"]
        if not isinstance(rows, list):
            raise ValueError("Candidates must be a list")
        seen = set()
        mapping = np.full(len(b.obj), -1); mapping[b.binary] = np.arange(len(b.binary))
        for row in rows[:3]:
            try:
                x = np.zeros(len(b.binary), dtype=np.uint8)
                for name in ("u", "v", "w"):
                    value = np.asarray(row[name])
                    indices = getattr(b, name)
                    if value.shape != indices.shape or value.dtype.kind not in "iuf" or not np.isin(value, [0, 1]).all():
                        raise ValueError("Invalid schedule matrix")
                    x[mapping[indices]] = value
                info["format_valid"] += 1
                key = x.tobytes()
                if key not in seen:
                    seen.add(key); candidates.append(x)
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        info.update(status="OK", accepted=len(candidates))
    except (KeyError, TypeError, ValueError, AttributeError):
        info["message"] = "INVALID_RESPONSE"
    return candidates, info


def request(b, best, lower, o, seconds, feedback):
    started = perf_counter()
    info = {"status": "FAILED", "message": "", "accepted": 0, "format_valid": 0}
    try:
        body = request_body(b, best, lower, o, feedback)
        remaining = seconds - (perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError
        http = Request(o.llm_url, data=json.dumps(body).encode("utf-8"), method="POST",
                       headers={"Content-Type": "application/json", "Authorization": f"Bearer {o.llm_api_key}"})
        with urlopen(http, timeout=min(remaining, o.llm_timeout)) as response:
            candidates, info = parse_response(json.load(response), b)
        if perf_counter() - started >= seconds:
            candidates = []; info.update(status="TIME_LIMIT", accepted=0)
    except Exception as error:
        candidates = []
        # Never persist raw transport errors, URLs or headers containing credentials.
        info["message"] = type(error).__name__
    info["seconds"] = perf_counter() - started
    return candidates, info
