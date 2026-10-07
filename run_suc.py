"""Run the core solver from Python or the command line."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from suc_benders.anneal import initialize
from suc_benders.config import load_options
from suc_benders.data import load_data
from suc_benders.solver import solve


def run_suc(options=None):
    o = load_options(options)
    d, scenarios = load_data(o.data_file, o.scenario_file)
    state = np.random.get_state()
    try:
        initialize(o.kaiwu_user_id, o.kaiwu_sdk_code)
        return solve(d, scenarios, o)
    finally:
        np.random.set_state(state)


def _serializable(value):
    if isinstance(value, np.ndarray):
        return _serializable(value.tolist())
    if isinstance(value, np.generic):
        return _serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if hasattr(value, "__dict__"):
        return _serializable(vars(value))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, kind in (("scenario-file", str), ("block-hours", int), ("max-outer", int),
                       ("max-inner", int), ("time-limit", float), ("seed", int)):
        parser.add_argument(f"--{name}", type=kind)
    parser.add_argument("--output", type=Path, help="Optional JSON result file")
    args = vars(parser.parse_args())
    output = args.pop("output")
    result = run_suc({key: value for key, value in args.items() if value is not None})
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_serializable(asdict(result)), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 1 if result.status.endswith("FAILED") or result.validation is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
