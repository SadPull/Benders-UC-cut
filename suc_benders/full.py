from __future__ import annotations

import argparse

from .data import load_suc_data
from .full_model import FullConfig, solve_full


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the RTS-24 UC/SUC reference with Gurobi")
    parser.add_argument("--data", default="data/rts24/units.json")
    parser.add_argument("--scenario-file", default=None)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    units, scenarios = load_suc_data(args.data, args.scenario_file)
    result = solve_full(
        units,
        scenarios,
        FullConfig(time_limit=args.time_limit, mip_gap=args.mip_gap, output_flag=0 if args.quiet else 1),
    )
    print(f"status={result.status}")
    print(f"objective={result.objective:.6f}")
    print(f"lower_bound={result.lower_bound:.6f}")
    print(f"fixed_cost={result.fixed_cost:.6f}")
    print(f"expected_fuel_cost={result.expected_fuel_cost:.6f}")
    print(f"mip_gap={result.mip_gap:.6g}")
    print(f"mip_gap_tolerance={args.mip_gap:.6g}")
    print(f"runtime_seconds={result.runtime:.3f}")


if __name__ == "__main__":
    main()
