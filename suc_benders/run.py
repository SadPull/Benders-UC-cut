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
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--no-quantum", action="store_true", help="Skip Kaiwu QUBO construction and simulated annealing")
    parser.add_argument("--no-decommit", action="store_true", help="Disable decommitment-first search")
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--llm-max-tokens", type=int, default=6144)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    units, scenarios = load_suc_data(args.data, args.scenario_file)
    config = BendersConfig(
        max_iterations=args.max_iterations,
        quantum_candidates=not args.no_quantum,
        decommit_candidates=not args.no_decommit,
        quantum=QuantumConfig(kaiwu_seed=args.seed),
        llm=LLMConfig(
            enabled=args.llm,
            timeout_seconds=args.llm_timeout,
            max_tokens=args.llm_max_tokens,
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
