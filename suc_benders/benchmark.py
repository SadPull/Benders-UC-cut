from __future__ import annotations

import argparse
import json
import math

from .data import load_suc_data
from .full_model import FullConfig, solve_full
from .llm import LLMConfig
from .master import QuantumConfig
from .solver import BendersConfig, result_to_dict, solve_benders


def _config(
    max_iterations: int,
    seed: int,
    *,
    repair: bool,
    llm: bool = False,
    llm_timeout: float = 180.0,
    llm_max_tokens: int = 65536,
    decommit: bool = False,
) -> BendersConfig:
    return BendersConfig(
        max_iterations=max_iterations,
        quantum=QuantumConfig(kaiwu_seed=seed),
        repair_candidates=repair,
        decommit_candidates=decommit,
        llm=LLMConfig(
            enabled=llm,
            timeout_seconds=llm_timeout,
            max_tokens=llm_max_tokens,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SUC Benders candidate strategies on the 10x24 instance")
    parser.add_argument("--suite", choices=("legacy", "attribution"), default="legacy")
    parser.add_argument("--output-dir", default="results/attribution-20260930")
    parser.add_argument("--resume", action="store_true", help="Resume completed attribution runs with matching code and settings")
    parser.add_argument("--pilot", action="store_true", help="Run only the first 10-strategy attribution block")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--data", default="data/suc_10x24.json")
    parser.add_argument("--scenario-file", default=None)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--llm-max-tokens", type=int, default=65536)
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated subset of strategy names to run (default: all).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.suite == "attribution":
        if args.scenario_file or args.strategies:
            parser.error("the attribution suite uses its fixed scenario and strategy matrix")
        from .attribution import run_attribution
        run_attribution(args.data, args.output_dir, resume=args.resume, pilot=args.pilot,
                        base_url=args.llm_base_url, model=args.llm_model)
        return

    units, scenarios = load_suc_data(args.data, args.scenario_file)

    # (name, repair, llm, decommit)
    strategies: list[tuple[str, bool, bool, bool]] = [
        ("quantum-only", False, False, False),
        ("quantum+repair", True, False, False),
        ("quantum+repair+decommit", True, False, True),
    ]
    if args.llm:
        strategies.append(("quantum+repair+decommit+llm", True, True, True))
        strategies.append(("quantum+repair+llm", True, True, False))

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
    for name, repair, llm, decommit in strategies:
        result = solve_benders(
            units,
            scenarios,
            _config(
                args.max_iterations,
                args.seed,
                repair=repair,
                llm=llm,
                llm_timeout=args.llm_timeout,
                llm_max_tokens=args.llm_max_tokens,
                decommit=decommit,
            ),
        )
        rows.append(
            {
                "strategy": name,
                "ub": result.objective,
                "iterations": len(result.history),
                "optimality_cuts": result.optimality_cuts,
                "feasibility_cuts": result.feasibility_cuts,
                "raw_candidates": sum(row.raw_candidates for row in result.history),
                "repaired_candidates": sum(row.repaired_candidates for row in result.history),
                "llm_candidates": sum(row.llm_candidates for row in result.history),
                "llm_valid_candidates": sum(row.llm_valid_candidates for row in result.history),
                "decommit_candidates": sum(row.decommit_candidates for row in result.history),
                "runtime": result.runtime,
                "status": result.status,
                "message": result.message,
                "history": result_to_dict(result)["history"],
            }
        )
    # Solve the independent reference only after all research runs have ended.
    reference_config = FullConfig(output_flag=0, mip_gap=1e-4)
    exact = solve_full(units, scenarios, reference_config)
    for row in rows:
        objective = row["ub"]
        row["relative_error"] = (
            (objective - exact.objective) / exact.objective
            if math.isfinite(objective) and math.isfinite(exact.objective) and exact.objective != 0
            else None
        )
        row["reference_gap"] = (
            (objective - exact.lower_bound) / max(1.0, abs(objective))
            if math.isfinite(objective) and math.isfinite(exact.lower_bound)
            else None
        )
    payload = {
        "exact_objective": exact.objective,
        "exact_lower_bound": exact.lower_bound,
        "exact_status": exact.status,
        "exact_mip_gap": exact.mip_gap,
        "exact_mip_gap_tolerance": reference_config.mip_gap,
        "exact_runtime": exact.runtime,
        "llm_timeout_seconds": args.llm_timeout,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "rows": rows,
    }
    print(json.dumps(result_to_dict(payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
