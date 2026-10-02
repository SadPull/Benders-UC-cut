from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_suc_data
from .solver import BendersConfig, solve_benders, result_to_dict
from .master import QuantumConfig
from .llm import LLMConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve 10x24 SUC with simulated annealing, decommitment and optional LLM candidates")
    parser.add_argument("--data", default="data/suc_10x24.json")
    parser.add_argument("--scenario-file", default=None)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--master-encoding", choices=("legacy", "compact", "dynamic"), default="legacy")
    parser.add_argument("--decomposition", choices=("none", "temporal"), default="none")
    parser.add_argument("--block-hours", type=int, default=2)
    parser.add_argument("--block-inner-iterations", type=int, default=3)
    parser.add_argument("--time-limit-seconds", type=float, default=None)
    parser.add_argument("--max-candidate-evaluations", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--llm-api-format", choices=("messages", "chat_completions", "responses"), default="chat_completions")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--no-quantum", action="store_true", help="Skip Kaiwu QUBO construction and simulated annealing")
    parser.add_argument("--no-decommit", action="store_true", help="Disable decommitment-first search")
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--llm-max-tokens", type=int, default=65536)
    parser.add_argument("--llm-reasoning-effort", default="max")
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    units, scenarios = load_suc_data(args.data, args.scenario_file)
    config = BendersConfig(
        master_encoding=args.master_encoding,
        decomposition=args.decomposition,
        block_hours=args.block_hours,
        block_inner_iterations=args.block_inner_iterations,
        time_limit_seconds=args.time_limit_seconds,
        max_candidate_evaluations=args.max_candidate_evaluations,
        max_iterations=args.max_iterations,
        quantum_candidates=not args.no_quantum,
        decommit_candidates=not args.no_decommit,
        quantum=QuantumConfig(kaiwu_seed=args.seed),
        llm=LLMConfig(
            enabled=args.llm,
            api_format=args.llm_api_format,
            base_url=args.llm_base_url,
            model=args.llm_model,
            timeout_seconds=args.llm_timeout,
            max_tokens=args.llm_max_tokens,
            reasoning_effort=args.llm_reasoning_effort,
            max_retries=args.llm_max_retries,
        ),
    )
    result = solve_benders(units, scenarios, config)
    payload = result_to_dict(result)
    print(f"status={result.status}")
    print(f"objective={result.objective:.6f}")
    print(f"iterations={len(result.history)}")
    print(f"optimality_cuts={result.optimality_cuts}")
    print(f"feasibility_cuts={result.feasibility_cuts}")
    print(f"runtime_seconds={result.runtime:.3f}")
    print(f"message={result.message}")
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
