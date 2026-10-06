"""Run with fresh Codex Astra subagents or an explicit Chat Completions endpoint."""
from dataclasses import asdict
import argparse
import json
import math
import os
from pathlib import Path
import time

from .data import load_suc_data
from .llm import LLMConfig
from .master import QuantumConfig
from .solver import BendersConfig, solve_benders, result_to_dict
from .subagent import initialize, publish, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/rts24/units.json")
    parser.add_argument("--scenario-file")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=20, help="Maximum outer iterations")
    parser.add_argument("--block-hours", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--max-inner-iterations", type=int, default=20)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-limit-seconds", type=float)
    parser.add_argument("--max-llm-calls", type=int)
    parser.add_argument("--original-feedback", action="store_true")
    parser.add_argument("--http-base-url", help="Use Chat Completions; requires --http-model and LLM_API_KEY")
    parser.add_argument("--http-model")
    parser.add_argument("--strategy", choices=("sa", "fixed", "stagnation", "stagnation_burst", "llm"), default="fixed")
    parser.add_argument("--target-reference-gap", type=float,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if bool(args.http_base_url) != bool(args.http_model):
        parser.error("HTTP base URL and model must be supplied together")
    if args.http_base_url and not os.getenv("LLM_API_KEY", "").strip():
        parser.error("HTTP runs require LLM_API_KEY in the environment")
    if args.rounds < 1 or (args.time_limit_seconds is not None and args.time_limit_seconds <= 0):
        parser.error("rounds and time limit must be positive")
    if args.max_llm_calls is not None and args.max_llm_calls < 0:
        parser.error("max LLM calls must be nonnegative")
    if args.target_reference_gap is not None:
        parser.error("--target-reference-gap is retired; use --convergence-tolerance and --max-inner-iterations")
    if args.max_inner_iterations < 1 or not math.isfinite(args.convergence_tolerance) or args.convergence_tolerance <= 0:
        parser.error("inner iterations and convergence tolerance must be positive and finite")
    data, scenarios = load_suc_data(args.data, args.scenario_file)
    root = Path(args.output_dir).resolve()
    llm = LLMConfig(enabled=args.strategy != "sa", api_format="codex_subagent", model="gpt-6-astra",
                    reasoning_effort="max", max_tokens=None, timeout_seconds=None,
                    max_retries=1, collect_diagnostics=True, exchange_dir=str(root),
                    enhanced_feedback=not args.original_feedback, max_calls=args.max_llm_calls,
                    schedule=args.strategy if args.strategy in {"stagnation", "stagnation_burst"} else "fixed")
    if args.http_base_url:
        llm.api_format = "chat_completions"
        llm.base_url, llm.model = args.http_base_url, args.http_model
        llm.temperature = None
    config = BendersConfig(max_iterations=args.rounds, block_hours=args.block_hours,
        max_inner_iterations=args.max_inner_iterations, convergence_tolerance=args.convergence_tolerance,
        advisor_only=args.strategy == "llm", quantum_candidates=args.strategy != "llm",
        time_limit_seconds=args.time_limit_seconds,
        quantum=QuantumConfig(kaiwu_seed=args.seed, candidate_pool_size=16), llm=llm)
    initialize(root, dict(arguments=vars(args), configuration=asdict(config),
        inputs={args.data: sha(args.data), **({args.scenario_file: sha(args.scenario_file)} if args.scenario_file else {})},
        sources={str(p): sha(p) for p in Path(__file__).parent.glob("*.py")}))
    import shutil
    source = root / "source"
    source.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source / path.name)
    started = time.perf_counter()
    publish(root / "state.json", dict(status="running"))
    http_call = None
    with (root / "events.jsonl").open("a", encoding="utf-8", buffering=1) as stream:
        def event(row):
            nonlocal http_call
            serialized = result_to_dict(row)
            if args.http_base_url:
                if row["kind"] == "llm_start":
                    call_number = row['call_index']
                    http_call = root / "calls" / f"{call_number:03d}"
                    http_call.mkdir(parents=True, exist_ok=False)
                    publish(http_call / "started.json", serialized)
                    print(f"HTTP_LLM_START call={row['call_index']} round={row['iteration']}", flush=True)
                elif row["kind"] == "llm_end":
                    diagnostics = serialized["diagnostics"]
                    publish(http_call / "diagnostics.json", diagnostics)
                    if "request" in diagnostics:
                        publish(http_call / "request.json", diagnostics["request"])
                        for message in diagnostics["request"]["messages"]:
                            name = "system.txt" if message["role"] == "system" else "user.txt"
                            (http_call / name).write_bytes(message["content"].encode("utf-8"))
                    if "response" in diagnostics:
                        publish(http_call / "response.json", diagnostics["response"])
                elif row["kind"] == "llm_feedback":
                    publish(http_call / "feedback.json", serialized["feedback"])
            stream.write(json.dumps(serialized, ensure_ascii=False, allow_nan=False)+"\n")
            if row["kind"] == "evaluation" and row["improved"]:
                publish(root / "best-checkpoint.json", serialized)
            if row["kind"] == "iteration":
                print(f"ROUND {row['iteration']} objective={row['incumbent']}", flush=True)
        config.event_sink = event
        try:
            result = solve_benders(data, scenarios, config)
            publish(root / "result.json", result_to_dict(result))
            publish(root / "convergence.json", result_to_dict(dict(
                status=result.status, dual_converged=result.status == "DUAL_CONVERGED",
                feasible_cost=result.objective, certified_lower_bound=result.diagnostics.get("reliable_lower"),
                primal_gap=result.diagnostics.get("primal_gap"),
                coordination_upper=result.diagnostics.get("coordination_upper"),
                termination_reason=result.diagnostics.get("termination_reason"))))
            publish(root / "state.json", dict(status=result.status, elapsed_seconds=time.perf_counter()-started))
        except BaseException as exc:
            publish(root / "state.json", dict(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    error_type=type(exc).__name__, elapsed_seconds=time.perf_counter()-started))
            raise


if __name__ == "__main__":
    main()
