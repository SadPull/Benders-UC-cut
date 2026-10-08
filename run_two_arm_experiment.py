"""Serial seed-7 comparison of round-robin SA+LLM and SA under equal budgets."""
import argparse
import contextlib
import csv
from dataclasses import asdict, fields
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from run_suc import _serializable, run_suc
import subagent_advisor
from suc_benders import llm
from suc_benders.anneal import anneal, initialize
from suc_benders.benders import dispatch
from suc_benders.config import ROOT, load_options
from suc_benders.data import load_data
from suc_benders.model import model, partition
from suc_benders.optimization import optimize
from suc_benders.validation import unpack, validate


ORDER = ("D-7", "A-7")
BUDGET = 1800


def public_options(o):
    return {f.name: getattr(o, f.name) for f in fields(o) if f.repr}


def write_json(path, value):
    path.write_text(json.dumps(_serializable(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def freeze(directory, o):
    paths = [ROOT / "run_suc.py", Path(__file__).resolve(), ROOT / "subagent_advisor.py", ROOT / "requirements.txt",
             ROOT / "pytest.ini", *sorted((ROOT / "suc_benders").glob("*.py")),
             *sorted((ROOT / "tests").rglob("*.py")), *sorted((ROOT / "data").glob("*.json"))]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        target = directory / "snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes()
        target.write_bytes(content)
        hashes[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), order=ORDER,
                    options=public_options(o), source_sha256=hashes,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                    python=platform.python_version(), platform=platform.platform(),
                    thread_environment={name: os.environ.get(name) for name in
                                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                    packages={name: importlib.metadata.version(name) for name in
                              ("numpy", "scipy", "gurobipy", "kaiwu", "matplotlib")},
                    reference={"time_limit": 600, "mip_gap": 1e-6, "threads": 1, "seed": 7},
                    protocol={"serial": True, "reference_not_search_input": True, "llm_transport": "codex_subagent",
                              "llm_model": "gpt-6-astra", "reasoning_effort": "max", "fresh_agent_per_call": True,
                              "llm_call_limit": o.llm_max_calls, "calls_per_trigger": 1,
                              "scheduler": "round_robin", "stagnation_unit": "complete_round",
                              "sa_candidate_scope": "current_block", "llm_candidate_scope": "whole_day",
                              "late_candidates_excluded": True, "original_stopping_rules": False,
                              "initialization_in_budget": True, "audit_outside_search_budget": True})
    write_json(directory / "manifest.json", manifest)
    return hashes


def verify_sources(hashes):
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
           for name, digest in hashes.items()):
        raise RuntimeError("Source or input changed after freezing")


def check_kaiwu(o):
    from scipy import sparse
    from types import SimpleNamespace as NS
    started = perf_counter()
    initialize(o.kaiwu_user_id, o.kaiwu_sdk_code)
    q = NS(Q=sparse.csr_matrix(np.diag([-1., 2.])), offset=0., fixed=np.full(2, np.nan),
           free=np.arange(2), nx=2, energy_unit=1.)
    before = np.random.get_state()
    try:
        a, _ = anneal(q, o, 30, 7)
        b, _ = anneal(q, o, 30, 7)
    finally:
        np.random.set_state(before)
    if not np.array_equal(a, b) or min(float(x @ q.Q @ x) for x in a) != -1:
        raise RuntimeError("Kaiwu repeatability or optimum check failed")
    return {"status": "PASSED", "elapsed_seconds": perf_counter() - started}


def audit(m, full, result, directory):
    started = perf_counter()
    original = validate(m.d, m.scenarios, result)
    x = np.concatenate([getattr(result, name).ravel(order="F") for name in ("u", "v", "w")])
    checked = dispatch(full, x, 60)
    if not checked.feasible:
        raise RuntimeError("Fixed dispatch audit failed")
    verified = unpack(m, checked.point)
    physical = validate(m.d, m.scenarios, verified)
    difference = abs(verified.objective - result.objective)
    if difference > 1e-3 + 1e-7 * abs(result.objective):
        raise RuntimeError("Fixed dispatch objective mismatch")
    record = dict(status="PASSED", original=original, fixed_dispatch=physical,
                  objective_difference=difference, verified_objective=verified.objective,
                  elapsed_seconds=perf_counter() - started)
    write_json(directory / "fixed_dispatch.json", verified)
    write_json(directory / "audit.json", record)
    return record


def reference(directory, m, full):
    started = perf_counter()
    raw = optimize(m, 600, integral=True)
    write_json(directory / "raw.json", raw)
    if raw.x is None:
        raise RuntimeError("Reference has no incumbent")
    point = raw.x.copy()
    point[m.binary] = np.rint(point[m.binary])
    result = unpack(m, point)
    checked = audit(m, full, result, directory)
    result.lower_bound = raw.lower
    result.status = raw.status
    result.elapsed_seconds = perf_counter() - started
    result.solver_objective = raw.objval
    result.audit = checked
    write_json(directory / "result.json", result)
    return result


def block_coverage(result, expected_blocks):
    rows = []
    for index in range(1, expected_blocks + 1):
        visits = [row for row in result.diagnostics.get("blocks", []) if row["block"] == index]
        first = [row["first_visit_seconds"] for row in visits if row["first_visit_seconds"] is not None]
        rows.append(dict(block=index, visits=sum(row["visits"] for row in visits),
                         sa_calls=sum(row["sa_calls"] for row in visits),
                         first_visit_seconds=min(first, default=None),
                         converged=visits[-1]["converged"] if visits else False,
                         relative_gap=visits[-1]["relative_gap"] if visits else None))
    visited = sum(row["visits"] > 0 for row in rows)
    passed = bool(rows) and visited == expected_blocks and all(row["sa_calls"] or row["converged"] for row in rows)
    return dict(status="PASSED" if passed else "FAILED", visited=visited, expected=expected_blocks, blocks=rows)


def summarize(result, name, reference_cost, checked, expected_blocks=8):
    evaluations = result.diagnostics["evaluations"]
    initial = [row["objective"] for row in evaluations if row["source"] == "initial" and row["feasible"]]
    contributions = {}
    for row in evaluations:
        if row["improved"] and row["source"] != "initial" and np.isfinite(row["incumbent_before"]):
            contributions[row["source"]] = contributions.get(row["source"], 0.) + row["incumbent_before"] - row["objective"]
    return dict(run=name, status=result.status, objective=result.objective,
                reference_difference=(result.objective-reference_cost)/abs(reference_cost),
                lower_bound=result.lower_bound, relative_gap=result.relative_gap,
                elapsed_seconds=result.elapsed_seconds, initial_objective=min(initial, default=None),
                sa_calls=result.diagnostics["sa_calls"], llm_calls=result.llm_calls,
                llm_completed=sum(row["status"] == "OK" for row in result.llm_history),
                llm_accepted=sum(row["accepted"] for row in result.llm_history),
                improvement_by_source=contributions, timings=result.diagnostics["timings"], audit=checked,
                block_coverage=block_coverage(result, expected_blocks),
                complete_rounds=result.diagnostics.get("complete_rounds", 0),
                coordination_count=result.diagnostics.get("coordination_count", 0))


def report(directory, summary, curves):
    write_json(directory / "summary.json", summary)
    with (directory / "curves.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "elapsed_seconds", "source", "objective", "incumbent", "improved", "sa_calls"])
        writer.writeheader(); writer.writerows(curves)
    lines = ["# Seed 7: fixed SA vs SA+LLM", "", f"Batch status: {summary['status']}", "",
             "Each search starts afresh with the same 1800 s budget, initialization and round-robin stopping rules.",
             "Subagent calls have no global count limit; three stagnant complete rounds trigger one request, followed by SA again.",
             "Reference differences use the independent feasible reference cost; solver gaps use each search's own lower bound.",
             "", "| Run | Status | Cost | Reference difference | Lower bound | Solver gap | Seconds | SA calls | LLM attempts/valid responses/accepted candidates |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["runs"]:
        lines.append(f"| {row['run']} | {row['status']} | {row['objective']:.6f} | {row['reference_difference']:.6%} | {row['lower_bound']:.6f} | {row['relative_gap']:.6%} | {row['elapsed_seconds']:.2f} | {row['sa_calls']} | {row['llm_calls']}/{row['llm_completed']}/{row['llm_accepted']} |")
    lines.extend(["", "Initialization and direct improvement contributions:", ""])
    for row in summary["runs"]:
        lines.append(f"- {row['run']}: initial cost {row['initial_objective']}; subsequent improvements {row['improvement_by_source']}; physical/fixed-dispatch audit {row['audit']['status']}.")
    lines.extend(["", "Block coverage (visits include local initialization; SA calls are actual sampling):", "",
                  "| Run | Block | Visits | SA calls | First visit (s) | Last local gap |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in summary["runs"]:
        coverage = row["block_coverage"]
        for block in coverage["blocks"]:
            first = block["first_visit_seconds"]
            gap = block["relative_gap"]
            lines.append(f"| {row['run']} | {block['block']} | {block['visits']} | {block['sa_calls']} | "
                         f"{first if first is not None else 'unvisited'} | {gap if gap is not None else 'unknown'} |")
    for row in summary["runs"]:
        coverage = row["block_coverage"]
        lines.append(f"\n{row['run']}: coverage {coverage['visited']}/{coverage['expected']} ({coverage['status']}), "
                     f"complete rounds {row['complete_rounds']}, coordination solves {row['coordination_count']}.\n")
    if "coverage_acceptance" in summary:
        lines.extend(["", f"Coverage acceptance: {summary['coverage_acceptance']}."])
    if "paired_improvement" in summary:
        lines.extend(["", f"SA+LLM cost reduction relative to SA: {summary['paired_improvement']:.6%}."])
    if "reference" in summary:
        lines.extend(["", f"Reference: {summary['reference']}."])
    if "preflight" in summary:
        lines.extend(["", f"LLM preflight: {summary['preflight']}."])
    if "failed_stage" in summary:
        lines.extend(["", f"Failed stage: {summary['failed_stage']}; error type: {summary['error_type']}."])
    lines.extend(["", "The LLM transport is fresh Codex Astra/max subagents, with no HTTP requests. A valid response need not yield a feasible or improving candidate. Calls are not forced when the trigger is not reached.",
                  "Only candidates evaluated and validated before the search deadline enter the result. Fixed-dispatch audits run separately after each search.",
                  "SA directly changes one block per batch; the subagent can propose a whole-day schedule. Repair may change hours outside a block. This compares complete methods under equal wall-clock budgets, not equal-scope candidate generators.",
                  "The historical 4.3112% reduction came from a different protocol that visited only the first block; it is not evidence for the repaired comparison.",
                  "This is one seed and one paired observation, not evidence of statistical significance, general speedup or bidirectional synergy."])
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for row in summary["runs"]:
            points = [point for point in curves if point["run"] == row["run"] and np.isfinite(point["incumbent"])]
            if points:
                ax.step([point["elapsed_seconds"] for point in points] + [row["elapsed_seconds"]],
                        [point["incumbent"] for point in points] + [row["objective"]], where="post", label=row["run"])
        ax.set(xlabel="Elapsed search time (s)", ylabel="Validated whole-day cost", xlim=(0, BUDGET))
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(directory / "cost_curve.png", dpi=150); plt.close(fig)


def run_batch(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    summary, curves = {"status": "RUNNING", "runs": []}, []
    stage = "CONFIGURATION"
    try:
        o = load_options({"llm_enabled": True, "seed": 7, "time_limit": BUDGET,
                          "llm_patience": 3, "llm_max_calls": None, "llm_model": "gpt-6-astra",
                          "llm_url": "codex-subagent", "llm_api_key": "not-used"})
        stage = "LLM_PREFLIGHT"
        print(stage, flush=True)
        check = subagent_advisor.preflight(o, directory)
        write_json(directory / "preflight.json", check)
        if check["status"] != "PASSED":
            summary.update(status="STOPPED_LLM_PREFLIGHT", preflight=check)
            write_json(directory / "state.json", {"status": summary["status"], "stage": stage})
            report(directory, summary, curves)
            return 1
        hashes = freeze(directory, o)
        stage = "KAIWU_PREFLIGHT"
        print(stage, flush=True)
        write_json(directory / "kaiwu_check.json", check_kaiwu(o))
        d, scenarios = load_data(o.data_file, o.scenario_file)
        m = model(d, scenarios)
        full = partition(m, d.horizon)[0][0]
        stage = "REFERENCE"
        print(stage, flush=True)
        ref_dir = directory / "reference"; ref_dir.mkdir()
        with (ref_dir / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            ref = reference(ref_dir, m, full)
        summary["reference"] = dict(objective=ref.objective, lower_bound=ref.lower_bound,
                                    status=ref.status, elapsed_seconds=ref.elapsed_seconds)
        report(directory, summary, curves)
        for name in ORDER:
            stage = name
            verify_sources(hashes)
            run_dir = directory / name; run_dir.mkdir()
            o.llm_enabled = name.startswith("D")
            write_json(run_dir / "options.json", public_options(o))
            write_json(directory / "state.json", {"status": "RUNNING", "stage": stage})
            print(f"{name} started", flush=True)
            original_request = llm.request
            llm.request = partial(subagent_advisor.request, directory=run_dir / "calls")
            try:
                with (run_dir / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    result = run_suc(asdict(o))
            finally:
                llm.request = original_request
            write_json(run_dir / "result.json", result)
            for index, info in enumerate(result.llm_history, 1):
                call_dir = run_dir / "calls" / f"{index:03d}"
                if call_dir.exists():
                    write_json(call_dir / "feedback.json", info)
            if any(row["elapsed_seconds"] > BUDGET for row in result.diagnostics.get("evaluations", [])):
                raise RuntimeError("Evaluation after deadline")
            checked = audit(m, full, result, run_dir)
            summary["runs"].append(summarize(result, name, ref.objective, checked,
                                             (d.horizon + o.block_hours - 1) // o.block_hours))
            for row in result.diagnostics["evaluations"]:
                curves.append(dict(run=name, **{key: row[key] for key in
                              ("elapsed_seconds", "source", "objective", "incumbent", "improved", "sa_calls")}))
            print(f"{name}: {result.status}, cost={result.objective:.6f}, SA={result.diagnostics['sa_calls']}, LLM={result.llm_calls}", flush=True)
            report(directory, summary, curves)
        by_name = {row["run"]: row for row in summary["runs"]}
        passed = all(row["block_coverage"]["status"] == "PASSED" for row in summary["runs"])
        summary["coverage_acceptance"] = "PASSED" if passed else "FAILED"
        search_failed = any(row["status"].endswith("FAILED") for row in summary["runs"])
        if passed and not search_failed:
            summary["paired_improvement"] = (by_name["A-7"]["objective"]-by_name["D-7"]["objective"])/abs(by_name["A-7"]["objective"])
        summary["status"] = "FAILED" if search_failed else ("COMPLETED" if passed else "COVERAGE_FAILED")
    except Exception as error:
        summary.update(status="FAILED", failed_stage=stage, error_type=type(error).__name__)
    write_json(directory / "state.json", {"status": summary["status"], "stage": stage})
    report(directory, summary, curves)
    print(summary["status"], flush=True)
    return 0 if summary["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    raise SystemExit(run_batch(parser.parse_args().output_dir))
