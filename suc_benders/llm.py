"""Synchronous Responses transport for the archived whole-day advisor policy."""
# 模块职责：构造全天启停提示词、发送 Responses 请求并解析候选；同步/后台发送共用同一传输。
# LLM 只提议机组 × 小时的 u 矩阵；调度、费用、割和界由数值程序生成及验证。
from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any, Sequence
from urllib.request import Request, urlopen

import numpy as np


# 将系统约束与用户上下文封装为非流式、禁止存储的 Responses 请求体，推理档位为 max。
# 返回的请求体不含 Authorization；密钥仅由发送函数放入 HTTP 头。
def request_body(d, scenarios, best, o, context, base_candidates, iteration):
    system, user = _prompt(d, scenarios, iteration, best, base_candidates,
                           2 if o.rts24_improvements else 4, context)
    return {"model": o.llm_model, "instructions": system, "input": user, "stream": False, "store": False,
            "reasoning": {"effort": "max"}}


# 提取已完成响应中的 output_text，解析至多四个候选，返回 (原始矩阵及提议记录, 回执)。
# 这里只检查 JSON 与矩阵形状；0/1、时序及调度可行性留给上/下界数值评估路径处理。
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


# 用指定 URL、密钥、请求体和超时秒数执行一次 HTTP POST，返回原始响应 JSON。
# 此层不重试也不吞异常，调用方负责预算、脱敏和失败回执。
def post_response(url, key, body, seconds):
    http = Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                   headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urlopen(http, timeout=seconds) as response:
        return json.load(response)


# 同步建议入口：把提示词构建时间计入总预算，发送请求并返回候选及脱敏耗时回执。
# 完整流程耗时达到 seconds 后丢弃迟到候选，避免在全局截止时间之后更新方案。
def request(d, scenarios, best, o, seconds, context, base_candidates, iteration):
    started = perf_counter()
    info = {"status": "FAILED", "message": "", "accepted": 0, "format_valid": 0, "proposals": []}
    try:
        body = request_body(d, scenarios, best, o, context, base_candidates, iteration)
        remaining = seconds - (perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError
        raw = post_response(o.llm_url, o.llm_api_key, body, min(remaining, o.llm_timeout))
        candidates, info = parse_response(raw, d, iteration)
        if perf_counter() - started >= seconds:
            candidates = []; info.update(status="TIME_LIMIT", accepted=0)
    except Exception as error:
        candidates = []
        info["message"] = type(error).__name__
    info["seconds"] = perf_counter() - started
    return candidates, info


# 返回系统提示和紧凑 JSON 用户输入，明确原物理规则、现有方案及候选输出形状。
# 按标准 UC/SMS 分支选择末小时、备用和软平衡规则，冷热启动历史不在块边界重置。
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
    if hasattr(data, "cold_startup_cost"):
        body.update(cold_startup_cost=data.cold_startup_cost.tolist(),
                    cold_start_hours=data.cold_start_hours.tolist(),
                    startup_rule="startup_cost is the hot fee; use cold_startup_cost only when consecutive "
                                 "offline hours > min_down + cold_start_hours. Include initial offline history; "
                                 "do not reset it at block boundaries. Charge startup fees once across scenarios.")
    if context is not None:
        body["context"] = context
    if getattr(data, "model_profile", "") == "azema_sms":
        body.update(model_profile="azema_sms", shedding_cost=data.shedding_cost,
            curtailment_cost=data.curtailment_cost,
            balance_rule="Generation + shedding - curtailment = net load; imbalances are penalized, not forbidden.",
            terminal_rule="Shutdown in the last hour is allowed; minimum times truncate at the horizon.",
            ramp_rule="Startup output and pre-shutdown output equal Pmin; on-to-on ramp limits are ramp_up/down.")
        body["rules"] = [r for r in body["rules"] if "violate reserve" not in r]
        body["rules"].append("Respect minimum times and initial power; minimize fuel + fixed + imbalance penalties.")
        system += " This SMS case permits penalized shedding and curtailment and has no reserve constraint."
    return system, json.dumps(body, separators=(",", ":"), allow_nan=False)
