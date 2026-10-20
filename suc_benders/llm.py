"""Upper-advice prompts, Responses transport and candidate parsing."""
# 模块职责：构造全天启停提示词、发送 Responses 请求并解析候选。
# LLM 只提议改动机组的全天开机区间，程序以提示中的 incumbent_u 还原为 u 矩阵；
# 调度、费用、割和界由数值程序生成及验证。
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Sequence
from urllib.request import Request, urlopen

import numpy as np


# 将系统约束与用户上下文封装为流式、禁止存储的 Responses 请求体，推理档位为 max。
# text.format 用 JSON schema 约束输出结构；返回的请求体不含 Authorization，密钥仅由发送函数放入 HTTP 头。
# continuation=True 用于续接会话：静态规则和机组数据已在首条消息中，只发送动态快照。
def request_body(d, scenarios, best, o, context, base_candidates, iteration, continuation=False):
    system, user = _prompt(d, scenarios, iteration, best, base_candidates,
                           4, context, continuation)
    return {"model": o.llm_model, "instructions": system, "input": user, "stream": True, "store": False,
            "reasoning": {"effort": "max"},
            "text": {"format": {"type": "json_schema", "name": "uc_candidates", "strict": True,
                                "schema": output_schema(d.n_units, d.horizon)}}}


# 候选输出的 JSON schema，Codex --output-schema 与 HTTP text.format 共用。
# 每个候选只列出改动机组及其全天开机区间（1 起始、两端包含），不再让模型逐位抄写整张矩阵。
def output_schema(n_units, horizon):
    hour = dict(type="integer", minimum=1, maximum=horizon)
    interval = dict(type="object", additionalProperties=False, required=["start", "end"],
                    properties=dict(start=hour, end=hour))
    change = dict(type="object", additionalProperties=False, required=["unit", "on_intervals"],
                  properties=dict(unit=dict(type="integer", minimum=1, maximum=n_units),
                                  on_intervals=dict(type="array", items=interval)))
    candidate = dict(type="object", additionalProperties=False, required=["name", "reason", "changes"],
                     properties=dict(name=dict(type="string"), reason=dict(type="string"),
                                     changes=dict(type="array", minItems=1, items=change)))
    return dict(type="object", additionalProperties=False, required=["candidates"],
                properties=dict(candidates=dict(type="array", maxItems=4, items=candidate)))


# 从请求体取出模型看到的 incumbent_u，作为增量候选的还原基准；缺失或无法解析时返回 None。
def prompt_base(body):
    try:
        base = json.loads(body["input"]).get("incumbent_u")
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return None if base is None else np.asarray(base, dtype=float).tolist()


# 以 base 为底，把每个列出机组的整行替换为其开机区间，返回完整 u 矩阵。
# 不合格时抛 ValueError，消息即记录到回执中的 invalid_reason。
def _apply_changes(base, changes, n_units, horizon):
    if base is None:
        raise ValueError("missing base schedule")
    u = np.array(base, dtype=float)
    if u.shape != (n_units, horizon):
        raise ValueError(f"base schedule shape {u.shape} != {(n_units, horizon)}")
    if not isinstance(changes, list) or not changes:
        raise ValueError("changes must be a non-empty list")
    whole = lambda value: isinstance(value, int) and not isinstance(value, bool)
    seen = set()
    for change in changes:
        unit, intervals = change["unit"], change["on_intervals"]
        if not whole(unit) or not 1 <= unit <= n_units:
            raise ValueError(f"unit {unit!r} out of range 1..{n_units}")
        if unit in seen:
            raise ValueError(f"duplicate unit {unit}")
        seen.add(unit)
        if not isinstance(intervals, list):
            raise ValueError(f"on_intervals of unit {unit} is not a list")
        row = np.zeros(horizon)
        for interval in intervals:
            start, end = interval["start"], interval["end"]
            if not (whole(start) and whole(end) and 1 <= start <= end <= horizon):
                raise ValueError(f"interval {start!r}-{end!r} of unit {unit} invalid for hours 1..{horizon}")
            if row[start-1:end].any():
                raise ValueError(f"overlapping intervals for unit {unit}")
            row[start-1:end] = 1
        u[unit-1] = row
    return u


# 提取已完成响应中的 output_text，逐个还原并检查至多四个候选，返回 (原始矩阵及提议记录, 回执)。
# 不合格候选单独剔除并记录 invalid_reason，其余照常保留；全部不合格才算失败。
# 时序与调度仍由数值程序验证。
def parse_response(response, d, iteration=1, base=None):
    info = {"status": "FAILED", "message": "", "accepted": 0, "format_valid": 0, "invalid": 0, "proposals": []}
    try:
        if response.get("status") != "completed":
            raise ValueError("Response is not completed")
        text = "".join(part["text"] for item in response["output"] if item.get("type") == "message"
                       for part in item.get("content", []) if part.get("type") == "output_text")
        parsed = json.loads(text)
        rows = parsed.get("candidates") if isinstance(parsed, dict) else None
        if not isinstance(rows, list) or len(rows) > 4:
            raise ValueError("Expected at most four candidates")
    except (KeyError, TypeError, ValueError, AttributeError):
        info["message"] = "INVALID_RESPONSE"
        return [], info
    candidates = []
    for index, row in enumerate(rows):
        proposal = {"proposal_id": f"llm:{iteration}:{index}", "origin": "llm"}
        info["proposals"].append(proposal)
        try:
            proposal["raw_changes"] = row.get("changes")
            raw = _apply_changes(base, row["changes"], d.n_units, d.horizon)
        except ValueError as error:
            reason = str(error)
        except (KeyError, TypeError, AttributeError):
            reason = "malformed candidate"
        else:
            proposal["raw_u"] = raw.tolist()
            info["format_valid"] += 1
            candidates.append((raw, proposal))
            continue
        proposal.update(outcome="invalid_candidate", invalid_reason=reason)
        info["invalid"] += 1
    if candidates or not rows:
        info.update(status="OK", accepted=len(candidates),
                    message=f"{info['invalid']} of {len(rows)} candidates invalid" if info["invalid"] else "")
    else:
        info["message"] = "NO_VALID_CANDIDATE"
    return candidates, info


# 用指定 URL、密钥、请求体和超时秒数执行一次 HTTP POST，按 SSE 读取事件流，
# 返回终态事件携带的完整响应（与非流式响应同构）。长推理期间中转会切断无数据的
# 非流式连接（约 300 s 返回 503），流式让连接持续有数据。
# 此层不重试也不吞异常，调用方负责预算、脱敏和失败回执；总截止时间由调用方进程控制。
def post_response(url, key, body, seconds):
    http = Request(url, data=json.dumps(dict(body, stream=True)).encode("utf-8"), method="POST",
                   headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                            "Authorization": f"Bearer {key}"})
    with urlopen(http, timeout=seconds) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if event.get("type") in ("response.completed", "response.failed", "response.incomplete"):
                return event["response"]
            if event.get("type") == "error":
                raise RuntimeError(f"Stream error: {event.get('message') or event.get('code')}")
    raise RuntimeError("Stream ended without a final response")


# 续接提示只保留会随迭代变化的字段；其余字段在会话首条消息中已给出。
DYNAMIC_KEYS = ("task", "iteration", "horizon", "units", "incumbent_u", "base_u",
                "requested_candidates", "context")


# 返回系统提示和紧凑 JSON 用户输入，明确原物理规则、现有方案及候选输出形状。
# 冷热启动历史不在块边界重置。
def _prompt(
    data: Any,
    scenarios: Sequence[Any],
    iteration: int,
    incumbent: Any | None,
    base_candidates: list[Any],
    max_candidates: int,
    context: dict | None = None,
    continuation: bool = False,
) -> tuple[str, str]:
    system = (
        "You are a stochastic unit-commitment heuristic advisor. You propose binary "
        "on/off schedules only; you are not the optimizer. Never invent dispatch, "
        "theta, dual cuts, load, costs, or constraints. Return JSON only with a "
        "candidates array. Every proposal will be independently validated. "
        "Use context.unit_hour_decommit_gradient (sorted by estimated saving, best first) "
        "to identify possible savings. These gradients are local estimates, NOT measured "
        "savings or feasibility certificates. Consider removing entire ON blocks or "
        "trimming their ends, accounting for startup/shutdown costs, minimum up/down "
        "times, initial conditions, ramping and all scenarios. Turning units ON or "
        "substituting units is allowed when beneficial. No measured decommitment ladder "
        "is required. Minimize total expected cost, not the number of ON hours. "
        "Physical rules and fixed states override every cost gradient. Plan coordinated continuous "
        "ON/OFF intervals across a small set of movable units, including replacement capacity before "
        "a shutdown. Check each complete unit timeline against initial history, forbidden transitions "
        "and terminal rules before returning it. Do not rely on downstream repair to make your "
        "proposal feasible: repair may undo the entire intended saving. Return fewer than four "
        "candidates if necessary; never claim measured feasibility or savings without evaluation. "
        "Describe each candidate only by the units whose schedule differs from incumbent_u: for every "
        "listed unit give its complete set of ON intervals for the whole day (1-based hours, both ends "
        "inclusive; hours outside the intervals are OFF; an empty list means OFF all day). Units you do "
        "not list keep their incumbent_u rows. Use 1-based unit/hour numbers in name, reason and changes."
    )
    body = {
        "task": "Suggest diverse feasible-looking first-stage commitment schedules.",
        "iteration": iteration,
        "horizon": data.horizon,
        "units": data.n_units,
        "reserve_fraction": data.reserve_fraction,
        "indexing": {
            "matrix": "u[0][0] is unit 1, hour 1; rows=units, columns=hours",
            "context_unit_period": 0,
            "reachability_and_feedback": 1,
            "candidate_changes": 1,
            "explanation": "Context unit, period and unit_ids are zero-based. Reachability unit lists, "
                           "previous_call_feedback unit/start_hour/end_hour and candidate changes "
                           "(unit, on_intervals start/end, inclusive) are one-based. "
                           "All matrices retain their ordinary row/column order; do not shift them.",
        },
        "physical_rules": {
            "ramping": "For EVERY hour, including startup and shutdown: p_t-p_previous <= ramp_up "
                       "and p_previous-p_t <= ramp_down. First-hour p_previous=initial_output; "
                       "there is NO startup/shutdown exception and no state between OFF and pmin.",
            "output": "pmin*u <= p; p+r <= pmax*u; p,r >= 0. At a startup p+r <= pstart. "
                      "Before a next-hour shutdown p+r <= pshut. pstart/pshut are additional capacity "
                      "limits, NOT overrides of ramp_up/ramp_down.",
            "reachability": "Startup from OFF requires pmin <= min(pstart,ramp_up). A shutdown "
                            "after the first hour requires pmin <= min(pshut,ramp_down). First-hour "
                            "shutdown instead requires initial_output <= ramp_down. These are only "
                            "necessary conditions; a movable unit is not freely switchable at every hour.",
            "timing": "v=max(u-u_previous,0), w=max(u_previous-u,0), with u_previous=initial_on "
                      "at hour 1. Before changing state, its consecutive duration (including "
                      "initial_duration) must satisfy min_up or min_down. Do not reset at block boundaries. "
                      "Minimum-time windows after a transition are clipped at the horizon end.",
            "terminal": "Final-hour shutdown is forbidden: w[:, -1] = 0.",
            "scenarios": "The same u applies to all scenarios. In each scenario/hour, sum(p)=load "
                         "and sum(r)>=reserve_fraction*load. Check sum(pmin*u)<=minimum scenario load "
                         "and startup/shutdown-adjusted available capacity >= maximum load plus reserve. "
                         "Passing these aggregate checks does not prove ramp-feasible dispatch.",
        },
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
                    "changes": [
                        {
                            "unit": "1-based unit number whose day differs from incumbent_u",
                            "on_intervals": [{"start": "first ON hour, 1-based",
                                              "end": "last ON hour, inclusive"}],
                        }
                    ],
                }
            ]
        },
        "rules": [
            "Obey context.reachability when present: fixed_u entries other than -1 cannot change; "
            "startup_allowed and shutdown_allowed forbid the indicated transitions. Its explicit unit lists "
            "are 1-based; other field conventions are specified in indexing. These necessary "
            "conditions do not replace full physical feasibility checks.",
            "Do not repeat base schedules",
            "List only changed units; each listed unit's on_intervals define its entire day",
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
    if hasattr(data, "cold_startup_cost"):
        body.update(cold_startup_cost=data.cold_startup_cost.tolist(),
                    cold_start_hours=data.cold_start_hours.tolist(),
                    startup_rule="startup_cost is the hot fee; use cold_startup_cost only when consecutive "
                                 "offline hours > min_down + cold_start_hours. Include initial offline history; "
                                 "do not reset it at block boundaries. Charge startup fees once across scenarios.")
    if context is not None:
        context = deepcopy(context)
        reachability = context.get('reachability')
        if reachability is not None:
            fixed = np.asarray(reachability['fixed_u'])
            reachability['fixed_on_units'] = (np.flatnonzero((fixed == 1).all(axis=1))+1).tolist()
            reachability['fixed_off_units'] = (np.flatnonzero((fixed == 0).all(axis=1))+1).tolist()
            rows = context.get('unit_hour_decommit_gradient', [])
            context['unit_hour_decommit_gradient'] = [r for r in rows if fixed[r['unit'], r['period']] != 1]
            context['decommit_filter'] = dict(removed=len(rows)-len(context['unit_hour_decommit_gradient']),
                rule='Exclude known fixed-ON unit-hours. Remaining gradients are NOT feasibility certificates.')
        if continuation:
            context.pop('reachability', None)
        body["context"] = context
    if continuation:
        system = ""
        body = {key: body[key] for key in DYNAMIC_KEYS if key in body}
        body["static_inputs"] = ("unchanged since the first message of this conversation "
                                 "(rules, unit data, scenarios, reachability)")
    return system, json.dumps(body, separators=(",", ":"), allow_nan=False)
