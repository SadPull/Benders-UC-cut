"""Run from Codex, which services calls/ requests using fresh Astra subagents."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import time

from .data import load_suc_data
from .llm import LLMConfig
from .master import QuantumConfig
from .solver import BendersConfig, solve_benders, result_to_dict
from .subagent import initialize, publish, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/suc_10x24.json")
    parser.add_argument("--scenario-file")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-limit-seconds", type=float)
    parser.add_argument("--max-llm-calls", type=int)
    parser.add_argument("--original-feedback", action="store_true")
    parser.add_argument("--strategy", choices=("sa", "fixed", "stagnation", "stagnation_burst", "llm"), default="fixed")
    parser.add_argument("--target-reference-gap", type=float,
                        help="Run until strictly below this relative Gurobi reference difference; ignores rounds")
    args = parser.parse_args()
    if args.rounds < 1 or (args.time_limit_seconds is not None and args.time_limit_seconds <= 0):
        parser.error("rounds and time limit must be positive")
    if args.max_llm_calls is not None and args.max_llm_calls < 0:
        parser.error("max LLM calls must be nonnegative")
    if args.target_reference_gap is not None and (not 0 < args.target_reference_gap < 1 or args.time_limit_seconds is not None or args.max_llm_calls is not None):
        parser.error("target gap must be in (0,1) and cannot be combined with time/call limits")
    data, scenarios = load_suc_data(args.data, args.scenario_file)
    root = Path(args.output_dir).resolve()
    llm = LLMConfig(enabled=args.strategy != "sa", api_format="codex_subagent", model="gpt-6-astra",
                    reasoning_effort="max", max_tokens=None, timeout_seconds=None,
                    max_retries=1, collect_diagnostics=True, exchange_dir=str(root),
                    enhanced_feedback=not args.original_feedback, max_calls=args.max_llm_calls,
                    schedule=args.strategy if args.strategy in {"stagnation", "stagnation_burst"} else "fixed")
    config = BendersConfig(max_iterations=args.rounds, master_encoding="dynamic",
        decomposition="none", decommit_candidates=False, random_candidates=False,
        advisor_only=args.strategy == "llm", quantum_candidates=args.strategy != "llm",
        stop_on_no_new_candidates=False, time_limit_seconds=args.time_limit_seconds,
        collect_diagnostics=True, quantum=QuantumConfig(kaiwu_seed=args.seed, candidate_pool_size=16), llm=llm)
    initialize(root, dict(arguments=vars(args), configuration=asdict(config),
        inputs={args.data: sha(args.data), **({args.scenario_file: sha(args.scenario_file)} if args.scenario_file else {})},
        sources={str(p): sha(p) for p in Path(__file__).parent.glob("*.py")}))
    import shutil
    source = root / "source"
    source.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source / path.name)
    reference = None
    if args.target_reference_gap is not None:
        import gurobipy as gp
        from .full_model import solve_full, FullConfig
        from .attribution import validate_schedule
        from .rts24_experiment import schedule_object
        gp.setParam("Threads", 1)
        gp.setParam("Seed", args.seed)
        full = solve_full(data, scenarios, FullConfig(time_limit=600, mip_gap=1e-6, output_flag=0))
        reference = result_to_dict({k: v for k, v in vars(full).items() if k != "model"})
        publish(root / "reference-raw.json", reference)
        full.model.dispose()
        # Re-solve dispatch with the rounded binary decisions fixed, using the
        # same continuous models and physical tolerances as candidate evaluation.
        from .subproblem import solve_scenario
        checked = [solve_scenario(data, s, full.u, full.v, full.w,
                    deadline=time.perf_counter()+60, threads=1) for s in scenarios]
        if not all(r.feasible for r in checked):
            raise ValueError("reference fixed commitment is not feasible")
        reference["raw_objective"] = reference["objective"]
        reference["expected_fuel_cost"] = sum(s.probability*r.objective for s, r in zip(scenarios, checked))
        reference["objective"] = reference["fixed_cost"] + reference["expected_fuel_cost"]
        reference["scenarios"] = result_to_dict([dict(name=s.name, probability=s.probability,
            p=r.p, r=r.r, fuel_cost=r.objective) for s, r in zip(scenarios, checked)])
        validation = validate_schedule(data, scenarios, schedule_object(reference))
        publish(root / "reference.json", reference)
        publish(root / "reference-validation.json", validation)
        config.target_cost = reference["objective"] * (1 + args.target_reference_gap)
        manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
        manifest["configuration"] = asdict(config)
        publish(root / "run.json", manifest)
        print(f"REFERENCE cost={reference['objective']} lower={reference['lower_bound']} target_strictly_below={config.target_cost}", flush=True)
    started = time.perf_counter()
    publish(root / "state.json", dict(status="running"))
    with (root / "events.jsonl").open("a", encoding="utf-8", buffering=1) as stream:
        def event(row):
            serialized = result_to_dict(row)
            if reference is not None and row["kind"] == "evaluation" and row["feasible"]:
                serialized["reference_difference"] = row["objective"] / reference["objective"] - 1
                print(f"EVALUATED round={row['iteration']} cost={row['objective']:.6f} reference_difference={100*serialized['reference_difference']:.6f}%", flush=True)
            stream.write(json.dumps(serialized, ensure_ascii=False, allow_nan=False)+"\n")
            if row["kind"] == "evaluation" and row["improved"]:
                publish(root / "best-checkpoint.json", serialized)
            if row["kind"] == "iteration":
                print(f"ROUND {row['iteration']} objective={row['incumbent']}", flush=True)
        config.event_sink = event
        try:
            result = solve_benders(data, scenarios, config)
            publish(root / "result.json", result_to_dict(result))
            if reference is not None:
                publish(root / "comparison.json", dict(reference_cost=reference["objective"],
                    reference_lower_bound=reference["lower_bound"], objective=result.objective,
                    reference_difference=result.objective/reference["objective"]-1,
                    requested_difference=args.target_reference_gap,
                    target_reached=result.objective < config.target_cost,
                    reference_runtime=reference["runtime"], solver_runtime=result.runtime))
            publish(root / "state.json", dict(status=result.status, elapsed_seconds=time.perf_counter()-started))
        except BaseException as exc:
            publish(root / "state.json", dict(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    error_type=type(exc).__name__, elapsed_seconds=time.perf_counter()-started))
            raise


if __name__ == "__main__":
    main()
