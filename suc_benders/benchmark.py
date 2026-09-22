from __future__ import annotations

import argparse
import json
import time

from .data import load_suc_data
from .full_model import FullConfig, solve_full
from .llm import LLMConfig
from .solver import BendersConfig, solve_benders


def _config(max_iterations: int, seed: int, *, repair: bool, perturb: bool, llm: bool = False) -> BendersConfig:
    return BendersConfig(
        max_iterations=max_iterations,
        repair_candidates=repair,
        perturb_candidates=perturb,
        llm=LLMConfig(enabled=llm),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SUC Benders candidate strategies on the 10x24 instance")
    parser.add_argument("--data", default="data/suc_10x24.json")
    parser.add_argument("--scenario-file", default=None)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    units, scenarios = load_suc_data(args.data, args.scenario_file)
    exact = solve_full(units, scenarios, FullConfig(output_flag=0, mip_gap=1e-4))

    strategies: list[tuple[str, bool, bool, bool]] = [
        ("quantum-only", False, False, False),
        ("quantum+repair", True, False, False),
        ("quantum+repair+local", True, True, False),
    ]
    if args.llm:
        strategies.append(("quantum+repair+local+llm", True, True, True))

    rows = []
    for name, repair, perturb, llm in strategies:
        started = time.perf_counter()
        result = solve_benders(
            units,
            scenarios,
            _config(
                args.max_iterations,
                args.seed,
                repair=repair,
                perturb=perturb,
                llm=llm,
            ),
        )
        rows.append(
            {
                "strategy": name,
                "ub": result.objective,
                "relative_error": (result.objective - exact.objective) / exact.objective,
                "iterations": len(result.history),
                "optimality_cuts": result.optimality_cuts,
                "feasibility_cuts": result.feasibility_cuts,
                "raw_candidates": sum(row.raw_candidates for row in result.history),
                "repaired_candidates": sum(row.repaired_candidates for row in result.history),
                "perturbed_candidates": sum(row.perturbed_candidates for row in result.history),
                "llm_candidates": sum(row.llm_candidates for row in result.history),
                "llm_valid_candidates": sum(row.llm_valid_candidates for row in result.history),
                "runtime": time.perf_counter() - started,
                "status": result.status,
                "message": result.message,
            }
        )
    payload = {
        "exact_objective": exact.objective,
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
