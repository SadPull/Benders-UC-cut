from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import time
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np

from .candidates import commitment_hash, commitment_key, make_candidate, repair_commitment
from .common import Scenario, commitment_is_feasible
from .data import UCData
from .master import MasterCandidate


LOGGER = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    enabled: bool = False
    max_candidates: int = 4
    timeout_seconds: float | None = 180.0
    temperature: float = 0.2
    # Reasoning models (e.g. deepseek-reasoner) spend part of this budget on
    # hidden reasoning tokens before emitting any text, so the default is sized
    # for those models rather than for a plain chat completion.
    max_tokens: int = 65536
    reasoning_effort: str | None = "max"
    max_retries: int = 3
    retry_backoff_seconds: float = 30.0
    api_format: str = "chat_completions"
    base_url: str | None = None
    model: str | None = None
    collect_diagnostics: bool = False


@dataclass
class LLMAdviceResult:
    candidates: list[MasterCandidate]
    raw_count: int = 0
    valid_count: int = 0
    failed: bool = False
    message: str = ""
    diagnostics: dict = field(default_factory=dict)


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


def llm_settings(config: LLMConfig) -> tuple[str, str]:
    """Resolve non-secret settings for both the client and experiment manifest."""
    if config.api_format == "chat_completions":
        base = os.getenv("LLM_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL")
        model = os.getenv("LLM_MODEL") or os.getenv("ANTHROPIC_MODEL")
    else:
        base = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("LLM_BASE_URL")
        model = os.getenv("ANTHROPIC_MODEL") or os.getenv("LLM_MODEL")
    return (
        (config.base_url or base or "https://api.farluna.tech/v1").rstrip("/"),
        (config.model or model or "farluna/deepseek-v4.1-flash").strip(),
    )


def ask_llm_for_candidates(
    data: UCData,
    scenarios: Sequence[Scenario],
    iteration: int,
    base_candidates: list[MasterCandidate],
    incumbent: MasterCandidate | None,
    config: LLMConfig,
    context: dict | None = None,
) -> LLMAdviceResult:
    started = time.perf_counter()
    if config.api_format == "chat_completions":
        api_key = (os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()
    else:
        api_key = (os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("LLM_API_KEY") or "").strip()
    record = {
        "iteration": iteration, "api_format": config.api_format,
        "attempts": [], "proposals": [], "repair_seconds": 0.0,
    }

    def scrub(value):
        if isinstance(value, str):
            return value.replace(api_key, "[REDACTED]") if api_key else value
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    def finish(candidates=None, raw_count=0, *, failed=False, message="ok"):
        candidates = candidates or []
        record.update(
            failed=failed, message=message, raw_count=raw_count,
            valid_count=len(candidates), elapsed_seconds=time.perf_counter() - started,
        )
        LOGGER.info("LLM iteration=%s attempts=%s valid=%s failed=%s elapsed=%.2fs message=%s",
                    iteration, len(record["attempts"]), len(candidates), failed,
                    record["elapsed_seconds"], scrub(message))
        return LLMAdviceResult(
            candidates, raw_count, len(candidates), failed, scrub(message),
            scrub(record) if config.collect_diagnostics else {},
        )

    if not api_key:
        return finish(message="LLM_API_KEY is not set")
    if config.api_format not in {"messages", "chat_completions"}:
        return finish(failed=True, message="unsupported api_format")

    base_url, model = llm_settings(config)
    try:
        system, user = _prompt(
            data, scenarios, iteration, incumbent, base_candidates,
            int(config.max_candidates), context,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        return finish(failed=True, message=type(exc).__name__)

    if config.api_format == "chat_completions":
        endpoint = base_url + "/chat/completions"
        payload = {
            "model": model, "max_tokens": int(config.max_tokens),
            "temperature": float(config.temperature), "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": "Bearer " + api_key}
        if config.reasoning_effort is not None:
            payload["reasoning_effort"] = config.reasoning_effort
    else:
        endpoint = base_url + "/messages"
        payload = {
            "model": model, "max_tokens": int(config.max_tokens),
            "temperature": float(config.temperature), "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    headers.update({"Content-Type": "application/json", "Accept": "application/json"})
    request = Request(
        endpoint, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST",
    )
    if config.collect_diagnostics:
        record.update(endpoint=endpoint, requested_model=model, request=payload)
    direct_opener = build_opener(ProxyHandler({}))
    attempts = max(1, int(config.max_retries))
    raw = None
    for attempt in range(attempts):
        LOGGER.info("LLM request iteration=%s attempt=%s model=%s", iteration, attempt + 1, model)
        attempt_started = time.perf_counter()
        attempt_record = {"attempt": attempt + 1}
        record["attempts"].append(attempt_record)
        try:
            timeout = None if config.timeout_seconds is None else float(config.timeout_seconds)
            with direct_opener.open(request, timeout=timeout) as response:
                attempt_record["status_code"] = getattr(response, "status", 200)
                raw = json.loads(response.read().decode("utf-8"))
            attempt_record["elapsed_seconds"] = time.perf_counter() - attempt_started
            break
        except HTTPError as exc:
            attempt_record.update(
                status_code=exc.code, elapsed_seconds=time.perf_counter() - attempt_started,
            )
            if config.collect_diagnostics:
                try:
                    attempt_record["error_body"] = exc.read(4096).decode("utf-8", errors="replace")
                except OSError:
                    pass
            if (exc.code == 429 or 500 <= exc.code < 600) and attempt + 1 < attempts:
                time.sleep(float(config.retry_backoff_seconds) * (attempt + 1))
                continue
            return finish(failed=True, message=f"HTTP {exc.code}")
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            attempt_record.update(
                error_type=type(exc).__name__,
                elapsed_seconds=time.perf_counter() - attempt_started,
            )
            if attempt + 1 < attempts:
                time.sleep(float(config.retry_backoff_seconds) * (attempt + 1))
                continue
            return finish(failed=True, message=type(exc).__name__)
        except (KeyError, TypeError, ValueError) as exc:
            attempt_record.update(
                error_type=type(exc).__name__,
                elapsed_seconds=time.perf_counter() - attempt_started,
            )
            return finish(failed=True, message="invalid_response_json")

    if config.collect_diagnostics:
        record["response"] = raw
    try:
        if not isinstance(raw, dict):
            raise ValueError("response root must be an object")
        record["usage"] = raw.get("usage")
        record["returned_model"] = raw.get("model")
        if config.api_format == "chat_completions":
            choice = raw["choices"][0]
            content = choice["message"].get("content") or ""
            stop = choice.get("finish_reason")
        else:
            content = "".join(
                block.get("text", "") for block in raw.get("content", [])
                if isinstance(block, dict) and block.get("type") == "text"
            )
            stop = raw.get("stop_reason")
        record["stop_reason"] = stop
        if stop in {"length", "max_tokens"}:
            return finish(failed=True, message="truncated")
        if not isinstance(content, str) or not content.strip():
            return finish(failed=True, message="empty_content")
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
    except json.JSONDecodeError:
        return finish(failed=True, message="invalid_candidate_json")
    except (KeyError, IndexError, AttributeError, TypeError, ValueError) as exc:
        return finish(failed=True, message=str(exc)[:200])

    output: list[MasterCandidate] = []
    seen = {commitment_key(candidate.u, candidate.v, candidate.w) for candidate in base_candidates}
    for index, entry in enumerate(entries[: int(config.max_candidates)]):
        proposal = {"proposal_id": f"llm:{iteration}:{index}", "origin": "llm"}
        if config.collect_diagnostics:
            record["proposals"].append(proposal)
        if not isinstance(entry, dict):
            proposal["outcome"] = "invalid_entry"
            continue
        try:
            raw_u = np.asarray(entry.get("u"), dtype=float)
        except (TypeError, ValueError):
            proposal["outcome"] = "invalid_matrix"
            continue
        if raw_u.shape != (data.n_units, data.horizon):
            proposal["outcome"] = "invalid_shape"
            continue
        if config.collect_diagnostics:
            proposal["raw_u"] = raw_u.tolist()
            binary = bool(np.all(np.isfinite(raw_u)) and np.all((raw_u == 0) | (raw_u == 1)))
            proposal["raw_binary"] = binary
            raw_candidate = make_candidate(data, scenarios, raw_u) if binary else None
            proposal["raw_discrete_feasible"] = bool(
                raw_candidate is not None and commitment_is_feasible(
                    data, scenarios, raw_candidate.u, raw_candidate.v, raw_candidate.w,
                )
            )
        repair_started = time.perf_counter()
        repaired = repair_commitment(data, scenarios, raw_u)
        record["repair_seconds"] += time.perf_counter() - repair_started
        proposal["repair_attempted"] = True
        if repaired is None:
            proposal["outcome"] = "repair_failed"
            continue
        repaired.source = "llm"
        repaired.provenance = {
            "origin": "llm", "proposal_id": proposal["proposal_id"],
            "repair_attempted": True,
            "repair_changed": bool(np.any(repaired.u != raw_u)),
        }
        if config.collect_diagnostics:
            proposal.update(
                commitment_id=commitment_hash(repaired),
                repaired_u=repaired.u.tolist(), repaired_discrete_feasible=True,
                hamming_u=int(np.count_nonzero(repaired.u != raw_u)),
                repair_changed=repaired.provenance["repair_changed"],
            )
        key = commitment_key(repaired.u, repaired.v, repaired.w)
        if key in seen:
            proposal["outcome"] = "duplicate"
            continue
        seen.add(key)
        output.append(repaired)
        proposal["outcome"] = "accepted"
    return finish(output, len(entries))
