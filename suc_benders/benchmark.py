from __future__ import annotations

import argparse
import json
import time

from .data import load_suc_data
from .exact_master import ExactMasterConfig
from .full_model import FullConfig, solve_full
from .llm import LLMConfig
from .master import QuantumConfig
from .solver import BendersConfig, result_to_dict, solve_benders


def _config(
    max_iterations: int,
    seed: int,
    *,
    repair: bool,
    perturb: bool,
    llm: bool = False,
    llm_timeout: float = 180.0,
    llm_max_tokens: int = 6144,
    decommit: bool = False,
    exact_master: bool = False,
    master_time_limit: float = 60.0,
    quantum_candidates: bool = True,
) -> BendersConfig:
    return BendersConfig(
        max_iterations=max_iterations,
        quantum_candidates=quantum_candidates,
        quantum=QuantumConfig(kaiwu_seed=seed),
        repair_candidates=repair,
        perturb_candidates=perturb,
        decommit_candidates=decommit,
        llm=LLMConfig(
            enabled=llm,
            timeout_seconds=llm_timeout,
            max_tokens=llm_max_tokens,
        ),
        exact_master=exact_master,
        exact_master_config=ExactMasterConfig(time_limit=master_time_limit),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SUC Benders candidate strategies on the 10x24 instance")
    parser.add_argument("--data", default="data/suc_10x24.json")
    parser.add_argument("--scenario-file", default=None)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--llm-max-tokens", type=int, default=6144)
    parser.add_argument(
        "--master-time-limit",
        type=float,
        default=60.0,
        help="Time limit in seconds for each exact Benders master solve.",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated subset of strategy names to run (default: all).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    units, scenarios = load_suc_data(args.data, args.scenario_file)
    exact = solve_full(units, scenarios, FullConfig(output_flag=0, mip_gap=1e-4))

    # (name, repair, perturb/local, llm, decommit, exact_master)
    strategies: list[tuple[str, bool, bool, bool, bool, bool]] = [
        ("exact-only", False, False, False, False, True),
        ("quantum-only", False, False, False, False, False),
        ("quantum+repair", True, False, False, False, False),
        ("quantum+repair+decommit", True, False, False, True, False),
        # Hybrid arms: exact Gurobi Benders master for the certified bound,
        # Kaiwu plus optional LLM only as candidate generators.
        ("hybrid-exact", True, False, False, True, True),
        # Legacy local-search arm, kept for reference only.
        ("quantum+repair+local", True, True, False, False, False),
    ]
    if args.llm:
        # Primary arm: decommitment-first with LLM guidance.
        strategies.append(("quantum+repair+decommit+llm", True, False, True, True, False))
        strategies.append(("hybrid-exact+llm", True, False, True, True, True))
        # Legacy isolated LLM arm (no decommitment), for reference.
        strategies.append(("quantum+repair+llm", True, False, True, False, False))
        strategies.append(("quantum+repair+local+llm", True, True, True, False, False))

    if args.strategies:
        wanted = [item.strip() for item in args.strategies.split(",") if item.strip()]
        available = {item[0] for item in strategies}
        unknown = [item for item in wanted if item not in available]
        if unknown:
            raise SystemExit(
                f"unknown strategy name(s): {unknown}; available: {sorted(available)}"
            )
        strategies = [item for item in strategies if item[0] in wanted]

    rows = []
    for name, repair, perturb, llm, decommit, exact_master in strategies:
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
                llm_timeout=args.llm_timeout,
                llm_max_tokens=args.llm_max_tokens,
                decommit=decommit,
                exact_master=exact_master,
                master_time_limit=args.master_time_limit,
                quantum_candidates=name != "exact-only",
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
                "decommit_candidates": sum(row.decommit_candidates for row in result.history),
                "runtime": time.perf_counter() - started,
                "status": result.status,
                "lower_bound": result.lower_bound,
                "benders_gap": result.gap,
                "master_status": result.master_status,
                "master_runtime": result.master_runtime,
                "exact_candidates": sum(row.exact_candidates for row in result.history),
                "message": result.message,
                "history": result_to_dict(result)["history"],
            }
        )
    payload = {
        "exact_objective": exact.objective,
        "llm_timeout_seconds": args.llm_timeout,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "rows": rows,
    }
    print(json.dumps(result_to_dict(payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
