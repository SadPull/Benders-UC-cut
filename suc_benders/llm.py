from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np

from .candidates import repair_commitment
from .common import Scenario
from .data import UCData
from .master import MasterCandidate


@dataclass
class LLMConfig:
    enabled: bool = False
    max_candidates: int = 4
    timeout_seconds: float = 180.0
    temperature: float = 0.2
    # Reasoning models (e.g. deepseek-reasoner) spend part of this budget on
    # hidden reasoning tokens before emitting any text, so the default is sized
    # for those models rather than for a plain chat completion.
    max_tokens: int = 32000
    max_retries: int = 3
    retry_backoff_seconds: float = 30.0


@dataclass
class LLMAdviceResult:
    candidates: list[MasterCandidate]
    raw_count: int = 0
    valid_count: int = 0
    failed: bool = False
    message: str = ""


def _finite(value: Any) -> float | None:
    return float(value) if np.isfinite(value) else None


def _prompt(
    data: UCData,
    scenarios: Sequence[Scenario],
    iteration: int,
    incumbent: MasterCandidate | None,
    base_candidates: list[MasterCandidate],
    max_candidates: int,
    context: dict | None = None,
) -> tuple[str, str]:
    system = (
        "You are a stochastic unit-commitment heuristic advisor. You propose binary "
        "on/off schedules only; you are not the optimizer. Never invent dispatch, "
        "theta, dual cuts, load, costs, or constraints. Return JSON only with a "
        "candidates array. Every proposal will be independently validated. "
        "The dominant source of cost error is OVER-COMMITMENT: expensive small "
        "units are kept online longer than needed. Your primary job is to find "
        "unit-hours that can be turned OFF. Use context.unit_hour_decommit_gradient "
        "and context.top_decommit_opportunities (negative estimated_total_change "
        "means switching that unit-hour off reduces total cost) and "
        "context.decommit_ladder (actual measured cost of progressively turning "
        "hours off). Only keep a unit-hour ON when reserve or minimum up/down "
        "constraints require it."
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
            "Prefer turning unit-hours OFF over turning them ON",
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


def ask_llm_for_candidates(
    data: UCData,
    scenarios: Sequence[Scenario],
    iteration: int,
    base_candidates: list[MasterCandidate],
    incumbent: MasterCandidate | None,
    config: LLMConfig,
    context: dict | None = None,
) -> LLMAdviceResult:
    api_key = (os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("LLM_API_KEY") or "").strip()
    if not api_key:
        return LLMAdviceResult([], message="LLM_API_KEY is not set")

    base_url = (
        os.getenv("ANTHROPIC_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or "https://models.sjtu.edu.cn/api/v1"
    ).rstrip("/")
    model = (os.getenv("ANTHROPIC_MODEL") or os.getenv("LLM_MODEL") or "qwen3.8-27b").strip()
    try:
        system, user = _prompt(
            data,
            scenarios,
            iteration,
            incumbent,
            base_candidates,
            int(config.max_candidates),
            context,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        return LLMAdviceResult([], failed=True, message=type(exc).__name__)

    payload = {
        "model": model,
        "max_tokens": int(config.max_tokens),
        "temperature": float(config.temperature),
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    request = Request(
        base_url + "/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    direct_opener = build_opener(ProxyHandler({}))
    attempts = max(1, int(config.max_retries))
    backoff = float(config.retry_backoff_seconds)
    last_error = "unknown"
    raw = None
    for attempt in range(attempts):
        try:
            with direct_opener.open(
                request, timeout=float(config.timeout_seconds)
            ) as response:
                raw = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code == 429 and attempt + 1 < attempts:
                time.sleep(backoff * (attempt + 1))
                continue
            return LLMAdviceResult([], failed=True, message=last_error)
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = type(exc).__name__
            if attempt + 1 < attempts:
                time.sleep(backoff * (attempt + 1))
                continue
            return LLMAdviceResult([], failed=True, message=last_error)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return LLMAdviceResult([], failed=True, message=type(exc).__name__)

    if raw is None:
        return LLMAdviceResult([], failed=True, message=last_error)

    try:
        blocks = raw.get("content", []) if isinstance(raw, dict) else []
        content = "".join(
            block.get("text", "") for block in blocks if block.get("type") == "text"
        )
        if not content:
            # Reasoning models can consume the whole budget on hidden
            # reasoning and return no text; report that distinctly.
            stop = raw.get("stop_reason") if isinstance(raw, dict) else None
            usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
            raise ValueError(f"empty content (stop_reason={stop}, usage={usage})")
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(content)
        if isinstance(parsed, list):
            entries = parsed
        elif isinstance(parsed, dict):
            entries = parsed.get("candidates", [])
        else:
            raise ValueError("response root must be an object or list")
        if not isinstance(entries, list):
            raise ValueError("candidates is not a list")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return LLMAdviceResult([], failed=True, message=str(exc)[:200])

    output: list[MasterCandidate] = []
    seen = {
        tuple(np.r_[candidate.u.reshape(-1), candidate.v.reshape(-1), candidate.w.reshape(-1)].astype(int).tolist())
        for candidate in base_candidates
    }
    for entry in entries[: int(config.max_candidates)]:
        if not isinstance(entry, dict):
            continue
        try:
            raw_u = np.asarray(entry.get("u"), dtype=float)
        except (TypeError, ValueError):
            # Ragged or non-numeric matrices are malformed proposals, not a
            # reason to abort the whole Benders run.
            continue
        if raw_u.shape != (data.n_units, data.horizon):
            continue
        repaired = repair_commitment(data, scenarios, raw_u)
        if repaired is None:
            continue
        repaired.source = "llm"
        key = tuple(
            np.r_[
                repaired.u.reshape(-1), repaired.v.reshape(-1), repaired.w.reshape(-1)
            ].astype(int).tolist()
        )
        if key not in seen:
            seen.add(key)
            output.append(repaired)
    return LLMAdviceResult(output, len(entries), len(output), message="ok")
