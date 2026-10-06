"""Prespecified pure-SA search and separate integer certification protocol."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from .gap_closure_experiment import freeze as legacy_freeze, reference_record, reference_worker, validate_historical, write_csv
from .master import QuantumConfig
from .pure_sa_certification import certify_views, committed_events, snapshot
from .sa_scale_experiment import SEEDS, append, checksum, load_case, read, save
from .solver import BendersConfig, result_to_dict, solve_benders
from .subagent import sha
from .validation import schedule_object, validate_schedule


HOURS = (1, 2, 3, 5)
VARIANTS = ("sa_strict", "sa_flow", "sa_compact", "sa_full")
PHASE_SECONDS = dict(reference=720, sanity=360, diagnostic=2400, screening=1680,
                     ablation=1680, comparison=1860, confirmation=5400, audit=300)
TOTAL_SECONDS = sum(PHASE_SECONDS.values())
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "pure-sa-gap-v3-offline-certification"


def definitions():
    runs = []
    def add(stage, case, hours, method, seed, search, audit):
        runs.append(dict(id=f"{stage}-{case}-h{hours or 'selected'}-{method}-{seed}",
                         stage=stage, case=case, hours=hours, method=method, seed=seed,
                         seconds=search, audit_seconds=audit))
    for seed in SEEDS:
        add("sanity", "n3-det", 1, "sa_full", seed, 60, 20)
    add("sanity", "n10-det", 1, "sa_full", 7, 90, 30)
    for hours in HOURS:
        add("diagnostic", "n32-stoch", hours, "native_diagnostic", 7, 600, 0)
    for hours in HOURS:
        add("screening", "n32-stoch", hours, "sa_full", 7, 300, 120)
    for variant in VARIANTS:
        add("ablation", "n32-stoch", None, variant, 7, 300, 120)
    for seed in SEEDS:
        add("comparison", "n32-stoch", None, "selected", seed, 300, 120)
    add("comparison", "n32-stoch", None, "native_diagnostic", 7, 600, 0)
    for seed in SEEDS:
        add("confirmation", "n32-stoch", None, "selected", seed, 1200, 600)
    return runs


def configuration(definition, seconds=None):
    method = definition["method"]
    if method == "selected":
        raise ValueError("resolve selected variant before starting a search worker")
    native = method == "native_diagnostic"
    return BendersConfig(max_iterations=40, max_inner_iterations=200 if method == "sa_strict" else 8,
        block_hours=definition["hours"], convergence_tolerance=1e-4,
        quantum_candidates=not native, native_fallback_patience=1 if native else None,
        defer_full_recovery=True, gap_experiment_variant=method, sa_seed_per_call=True,
        quantum=QuantumConfig(kaiwu_seed=definition["seed"]),
        time_limit_seconds=definition["seconds"] if seconds is None else seconds)


def search_worker(directory, output, definition, deadline):
    """Reads only the run definition and UC inputs, never references or certificates."""
    data, scenarios = load_case(directory, definition["case"])
    cfg = configuration(definition, max(.001, deadline-time.perf_counter()-2.))
    save(output / "config.json", asdict(cfg))
    checkpoint, sequence = {}, 0
    with (output / "events.jsonl").open("w", encoding="utf-8") as stream:
        def sink(row):
            nonlocal sequence
            sequence += 1
            row = result_to_dict({**row, "sequence": sequence})
            kind = row["kind"]
            if kind == "evaluation" and row.get("feasible"):
                save(output / "schedules" / f"{sequence:08d}.json", row)
            elif kind in ("initial_dual_prices", "multiplier_epoch", "local_cut", "block_bound"):
                outer = row.get("outer", 0)
                name = f"cut-{sequence:08d}" if kind == "local_cut" else f"{kind}-{row.get('block', 'all')}"
                save(output / "search-state" / f"epoch-{outer:04d}" / f"{name}.json", row)
            append(stream, row)
            ready = time.perf_counter()
            append(stream, dict(kind="event_ready", sequence=sequence, clock=ready))
            if ready <= deadline:
                committed = {**row, "clock": ready}
                if kind == "evaluation" and row.get("feasible") and row.get("improved"):
                    checkpoint["best"] = committed
                elif kind == "bound_update" and committed.get("reliable_lower") is not None:
                    checkpoint["bound"] = committed
                else:
                    return
                save(output / "checkpoint.json", checkpoint)
        cfg.event_sink = sink
        result = solve_benders(data, scenarios, cfg)
    save(output / "result.json", result)
    if definition["method"].startswith("sa_") and result.diagnostics["online_integer_master_calls"]:
        raise AssertionError("pure-SA search invoked an integer master")


def search_hashes(output):
    files = [p for p in output.rglob("*") if p.is_file() and
             "offline" not in p.relative_to(output).parts and
             "snapshots" not in p.relative_to(output).parts and p.name != "search-hashes.json"]
    return {p.relative_to(output).as_posix(): sha(p) for p in files}


def check_hashes(base, expected):
    return [name for name, digest in expected.items()
            if not (base / name).is_file() or sha(base / name) != digest]


def certification_worker(directory, output, definition, deadline):
    expected = read(output / "search-hashes.json")
    errors = check_hashes(output, expected)
    if not expected or errors:
        save(output / "offline/result.json", dict(checkpoints={}, errors=["search_hash_mismatch", *errors], records=[]))
        return
    rows = committed_events(output / "events.jsonl")
    state = read(output / "state.json")
    targets = {"final": state["deadline"]}
    if definition["stage"] == "confirmation":
        targets = {"600": min(state["started_clock"]+600., state["deadline"]), **targets}
    views = {label: snapshot(rows, cutoff) for label, cutoff in targets.items()}
    for label, view in views.items():
        save(output / "snapshots" / f"{label}.json", view)
    data, scenarios = load_case(directory, definition["case"])
    def publish(value):
        save(output / "offline/result.json", {**value, "clock": time.perf_counter(),
             "snapshot_sha256": {label: sha(output / "snapshots" / f"{label}.json") for label in views}})
    result = certify_views(data, scenarios, definition, views, deadline-2., publish)
    result["errors"].extend(check_hashes(output, expected))
    publish(result)


def supervise(directory, output, role, definition, seconds, phase_deadline):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = min(phase_deadline, started+seconds)
    if deadline <= started:
        save(output / "state.json", dict(status="phase_budget_exhausted", planned_seconds=seconds))
        return
    save(output / "state.json", dict(status="running", started_clock=started, deadline=deadline,
                                      planned_seconds=seconds, allocated_seconds=deadline-started))
    env = os.environ.copy()
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    env.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-m", "suc_benders.pure_sa_gap_experiment", "--output-dir", str(directory),
               "--worker", role, "--identifier", definition.get("id", "reference"), "--deadline", str(deadline)]
    with (output / "console.txt").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=directory / "source", env=env,
            stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        reason = "completed"
        while process.poll() is None:
            if time.perf_counter() >= deadline:
                reason = "time_limit"
                process.terminate()
                break
            time.sleep(.05)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if reason == "completed" and process.returncode:
            reason = "failed"
    save(output / "state.json", dict(status=reason, exitcode=process.returncode, started_clock=started,
        deadline=deadline, planned_seconds=seconds, allocated_seconds=deadline-started,
        wall_seconds=time.perf_counter()-started))
    print(f"{role} {definition.get('id', 'reference')}: {reason}, {time.perf_counter()-started:.2f}s", flush=True)


def freeze(directory, previous):
    manifest = legacy_freeze(directory, previous)
    protocol_path = ROOT / "docs/pure-sa-gap-v3-protocol.md"
    target = directory / "source/docs" / protocol_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(protocol_path, target)
    manifest["source_sha256"][protocol_path.relative_to(ROOT).as_posix()] = sha(protocol_path)
    for name in ("baseline-tests", "isolation-tests", "flow-tests", "compact-tests", "diversity-tests",
                 "protocol-tests", "final-tests", "ipc-smoke", "ipc-fixture-failure", "ipc-fixture-sign-failure"):
        log = Path(tempfile.gettempdir()) / f"pure-sa-gap-v3-{name}.log"
        if log.exists():
            target = directory / "regression" / log.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(log, target)
    manifest.update(protocol=PROTOCOL, runs=definitions(), total_seconds=TOTAL_SECONDS,
        available_seconds=TOTAL_SECONDS, phase_seconds=PHASE_SECONDS, previous_directory=str(previous),
        base_commit="1c16bdd01213c56f8925255753794a08d3fa10ee", selected_variant_pending=True,
        configs={r["id"]: asdict(configuration(r)) if r["method"] != "selected" else None for r in definitions()},
        source_tag=subprocess.check_output(["git", "tag", "--points-at", "HEAD"], cwd=ROOT, text=True).splitlines())
    manifest["cached_reference_eligible"] = False
    try:
        prior = read(previous / "manifest.json")
        for filename in ("units.json", "scenarios.json"):
            if read(previous / "inputs/n32-stoch" / filename) != read(directory / "inputs/n32-stoch" / filename):
                raise ValueError("cached stochastic input mismatch")
        if prior["source_sha256"]["suc_benders/full_model.py"] != sha(ROOT / "suc_benders/full_model.py"):
            raise ValueError("cached stochastic full model mismatch")
        if not reference_record(previous, "n32-stoch")["certified"]:
            raise ValueError("cached stochastic reference uncertified")
        for filename in ("result.json", "state.json"):
            target = directory / "references/n32-stoch" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(previous / "references/n32-stoch" / filename, target)
        row = read(directory / "references/n32-stoch/result.json")
        data, scenarios = load_case(directory, "n32-stoch")
        row["validation"] = validate_schedule(data, scenarios, schedule_object(row))
        save(directory / "references/n32-stoch/result.json", row)
        manifest.update(cached_reference_eligible=True, cached_reference_source_sha256=sha(
            previous / "references/n32-stoch/result.json"))
    except (ValueError, KeyError, OSError, AssertionError) as exc:
        manifest["cached_reference_rejection"] = str(exc)
    save(directory / "manifest.json", manifest)
    return manifest


def select_hours(screening, diagnostics, reference):
    eligible, excluded = [], []
    for run in screening:
        diagnostic = next((r for r in diagnostics if r["hours"] == run["hours"]), None)
        if not run["numerical_audit_passed"]:
            excluded.append(dict(hours=run["hours"], reason="numerical_audit"))
        elif diagnostic and diagnostic["numerical_audit_passed"] and diagnostic["coordination_upper"] is not None and (
                diagnostic["coordination_upper"] < .99*reference-1e-5):
            excluded.append(dict(hours=run["hours"], reason="certified_dual_ceiling"))
        else:
            eligible.append(run)
    if not eligible:
        return dict(chosen_hours=None, excluded=excluded, reason="no_eligible_block_length")
    maximum = max(r["reliable_lower"] for r in eligible)
    return dict(chosen_hours=min(r["hours"] for r in eligible if maximum-r["reliable_lower"] <= 1e-5),
                excluded=excluded, rule="largest_certified_SA_lower_then_shortest_block")


def select_variant(runs):
    order = ("sa_flow", "sa_compact", "sa_full")
    eligible = [r for r in runs if r["method"] in order and r["numerical_audit_passed"]]
    if not eligible:
        return dict(chosen_variant=None, reason="no_valid_improved_variant")
    maximum = max(r["reliable_lower"] for r in eligible)
    return dict(chosen_variant=min((r["method"] for r in eligible if maximum-r["reliable_lower"] <= 1e-5), key=order.index),
                rule="largest_certified_lower_then_flow_compact_full")


def statistics(directory, registered):
    output = directory / "runs" / registered["id"]
    definition = read(output / "definition.json") or registered
    state, result = read(output / "state.json"), read(output / "result.json")
    errors = []
    try:
        raw = committed_events(output / "events.jsonl")
    except (ValueError, KeyError) as exc:
        raw = []
        errors.append(str(exc))
    deadline = state.get("deadline", -math.inf)
    rows = [r for r in raw if r["clock"] <= deadline]
    data, scenarios = load_case(directory, definition["case"])
    checked = []
    for row in rows:
        if row["kind"] == "evaluation" and row.get("feasible"):
            try:
                check = validate_schedule(data, scenarios, schedule_object(row))
                if not check["feasible_solution"]:
                    raise ValueError("full schedule physical validation failed")
                checked.append((row, check))
            except (AssertionError, ValueError, KeyError, TypeError) as exc:
                errors.append(str(exc))
    best = min((r for r, _ in checked), key=lambda r:r["objective"], default=None)
    initial = min((r["objective"] for r, _ in checked if r["iteration"] == 0), default=None)
    online = max((r["reliable_lower"] for r in rows if r["kind"] == "bound_update" and
                  r.get("reliable_lower") is not None and math.isfinite(r["reliable_lower"])), default=None)
    uppers = [r.get("coordination_upper", r.get("upper") if r["kind"] == "approximate_outer" else None)
              for r in rows if r["kind"] in ("bound_update", "iteration", "approximate_outer")]
    upper = min((u for u in uppers if u is not None and math.isfinite(u)), default=None)
    reference = reference_record(directory, definition["case"])
    ref = reference.get("objective") if reference["certified"] else None
    audit, audit_state = read(output / "offline/result.json"), read(output / "offline/state.json")
    if audit and audit.get("clock", math.inf) > audit_state.get("deadline", -math.inf):
        errors.append("late_offline_result")
        audit = {}
    checkpoints = audit.get("checkpoints", {})
    pure = definition["method"].startswith("sa_")
    native_calls = sum(r["kind"] == "native_start" for r in rows)
    integer_calls = native_calls+sum(r["kind"] == "integer_certification" for r in rows)
    integer_calls = max(integer_calls, result.get("diagnostics", {}).get("online_integer_master_calls", 0))
    lower = checkpoints.get("final", {}).get("lower") if pure else online
    errors.extend(check_hashes(output, read(output / "search-hashes.json")))
    if pure:
        errors.extend(audit.get("errors", []))
        if lower is None and state.get("status") not in ("skipped", "phase_budget_exhausted", None):
            errors.append("missing_final_offline_certificate")
        if integer_calls:
            errors.append("pure_search_called_integer_master")
        if audit_state.get("status") == "failed":
            errors.append("offline_worker_failed")
        for label, digest in audit.get("snapshot_sha256", {}).items():
            path = output / "snapshots" / f"{label}.json"
            if not path.exists() or sha(path) != digest:
                errors.append(f"snapshot_hash_mismatch:{label}")
    if state.get("status") == "failed":
        errors.append("search_worker_failed")
    if result.get("diagnostics", {}).get("failure_stage"):
        errors.append(result["diagnostics"]["failure_stage"])
    cost = best["objective"] if best else None
    for limit in (cost, ref, upper):
        if lower is not None and limit is not None and lower > limit+1e-5:
            errors.append("certified_lower_exceeds_feasible_or_dual_upper")
    valid = bool(lower is not None and math.isfinite(lower) and best and not errors and
                 state.get("status") in ("completed", "time_limit"))
    milestone = checkpoints.get("600", {})
    lower600 = milestone.get("lower")
    best600 = min((r["objective"] for r, _ in checked if r["clock"] <= state.get("started_clock", -math.inf)+600.), default=None)
    if lower600 is not None and any(v is not None and lower600 > v+1e-5 for v in (ref, best600)):
        errors.append("600_second_lower_exceeds_feasible_or_reference")
        valid = False
    samples = [r for r in rows if r["kind"] == "sampling"]
    cuts = [r for r in rows if r["kind"] == "local_cut"]
    sa_cuts = [r for r in cuts if r["origin"] == "sa"]
    epochs = [r for r in rows if r["kind"] == "multiplier_epoch"]
    native = [r for r in rows if r["kind"] == "native_candidate"]
    timings = result.get("diagnostics", {}).get("timings", {})
    return dict(**definition, state=state, solver_status=result.get("status", "TIME_LIMIT" if state.get("status") == "time_limit" else None),
        initial_cost=initial, final_cost=cost, best_source=best.get("source") if best else None,
        best_origins=best.get("candidate_origins", []) if best else [],
        online_lower=online, reliable_lower=lower, reference_cost=ref, reference_certified=reference["certified"],
        paper_gap=(ref-lower)/ref if valid and ref is not None else None,
        online_paper_gap=(ref-online)/ref if ref is not None and online is not None else None,
        reference_cost_difference=(cost-ref)/ref if cost is not None and ref is not None else None,
        primal_gap=(cost-lower)/cost if valid else None, coordination_upper=upper,
        posthoc_dual_converged=bool(valid and upper is not None and
            upper-lower <= 1e-5+1e-4*max(1., abs(lower))),
        numerical_audit_passed=valid, errors=errors, validated_schedules=len(checked),
        max_residual_mw=max((c["max_residual_mw"] for _, c in checked), default=None),
        search_seconds=state.get("wall_seconds"), search_allocated_seconds=state.get("allocated_seconds"),
        offline_seconds=audit_state.get("wall_seconds", 0.), offline_allocated_seconds=audit_state.get("allocated_seconds"),
        lower600=lower600, paper_gap600=(ref-lower600)/ref if lower600 is not None and ref is not None and not errors else None,
        best_cost600=best600, certification_complete=checkpoints.get("final", {}).get("complete", not pure),
        certified_models=checkpoints.get("final", {}).get("certified_models", 0),
        total_models=checkpoints.get("final", {}).get("total_models", 0),
        online_integer_master_calls=integer_calls, native_calls=native_calls,
        native_seconds=sum(r.get("seconds", 0.) for r in native),
        attempted_outer_rounds=len(epochs), completed_outer_rounds=len({r.get("outer", r.get("iteration")) for r in rows if
            r["kind"] in ("iteration", "approximate_outer")}),
        approximate_outer_rounds=sum(r["kind"] == "approximate_outer" for r in rows),
        multiplier_updates=sum(r["kind"] == "multiplier_update" for r in rows),
        attempted_blocks=sum(r["kind"] == "block_start" for r in rows),
        converged_blocks=len({(r["outer"], r["block"]) for r in rows if r["kind"] == "inner_iteration" and r["converged"]}),
        sa_calls=sum(r["kind"] == "sa_start" for r in rows), raw_samples=sum(r["raw_candidates"] for r in samples),
        raw_discrete_feasible=sum(r["raw_discrete_feasible"] for r in samples),
        sa_new_points=len(sa_cuts), sa_feasible_points=sum(r["feasible"] for r in sa_cuts),
        sa_optimality_cuts=sum(r["cut"]["feasible"] for r in sa_cuts),
        sa_feasibility_cuts=sum(not r["cut"]["feasible"] for r in sa_cuts),
        sa_derived_cuts=sum("sa" in r["sources"] for r in cuts),
        qubo_bits_first=samples[0]["bits"] if samples else None,
        qubo_bits_last=samples[-1]["bits"] if samples else None,
        qubo_bits_max=max((r["bits"] for r in samples), default=None),
        qubo_seconds=sum(r["build_seconds"] for r in samples),
        annealing_seconds=sum(r["sampling_seconds"] for r in samples),
        full_dispatch_seconds=timings.get("subproblems"), late_events=len(raw)-len(rows),
        contribution={source: sum(max(0., r["incumbent_before"]-r["objective"])
                      if r.get("incumbent_before") is not None else 0. for r, _ in checked
                      if r.get("improved") and r["source"] == source)
                      for source in {r["source"] for r, _ in checked}})


def plot(directory, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    final = [r for r in read(directory / "manifest.json")["runs"] if r["stage"] == "confirmation"]
    for metric, ylabel, filename in (("paper_gap", "Gap (%)", "paper-gap.png"),
                                      ("reliable_lower", "Certified lower bound", "certified-lower.png")):
        fig, ax = plt.subplots(figsize=(9, 5))
        for run in final:
            points = sorted((r for r in curves if r["run"] == run["id"] and r[metric] is not None and r["valid"]),
                            key=lambda r:(r["seconds"], r["availability"] == "posthoc"))
            for availability, linestyle in (("online", "--"), ("posthoc", "-")):
                selected = [r for r in points if r["availability"] == availability]
                if selected:
                    best = -math.inf
                    x, y = [], []
                    for row in selected:
                        best = max(best, row["reliable_lower"])
                        x.append(row["seconds"])
                        y.append((row["reference_cost"]-best)/row["reference_cost"]*100 if metric == "paper_gap" else best)
                    ax.step(x, y, where="post", linestyle=linestyle, label=f"seed {run['seed']} {availability}")
        if metric == "paper_gap":
            ax.axhline(1., color="gray", linestyle=":", label="1% goal")
            ax.axhline(.5, color="silver", linestyle=":", label="0.5% challenge")
        ax.set(xlabel="Search seconds (posthoc curves use source snapshot times)", ylabel=ylabel, xlim=(0, 1200))
        ax.grid(alpha=.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)


def report(directory):
    manifest = read(directory / "manifest.json")
    hash_errors = []
    for field, prefix in (("source_sha256", "source"), ("input_sha256", ""),
                          ("historical_sha256", ""), ("reference_sha256", "")):
        hash_errors.extend(check_hashes(directory / prefix, manifest.get(field, {})))
    rows = [statistics(directory, run) for run in manifest["runs"]]
    curves, blocks = [], []
    for row in rows:
        output = directory / "runs" / row["id"]
        state = row["state"]
        raw = committed_events(output / "events.jsonl")
        for event in raw:
            if event["clock"] > state.get("deadline", -math.inf):
                continue
            if event["kind"] == "bound_update":
                lower, ref = event["reliable_lower"], row["reference_cost"]
                curves.append(dict(run=row["id"], stage=row["stage"], method=row["method"], seed=row["seed"],
                    hours=row["hours"], seconds=event["clock"]-state["started_clock"], outer=event["outer"],
                    reliable_lower=lower, reference_cost=ref, paper_gap=(ref-lower)/ref if ref else None,
                    availability="online", audit_seconds=None, valid=row["numerical_audit_passed"] and not hash_errors))
            elif event["kind"] == "inner_iteration":
                blocks.append(dict(run=row["id"], availability="online", **event))
        audit = read(output / "offline/result.json")
        initial_lower = max((r["reliable_lower"] for r in raw if
            r["kind"] == "bound_update" and r["outer"] == 0), default=-math.inf)
        for record in audit.get("records", []):
            ref = row["reference_cost"]
            lower = max(record["epoch_lower"], initial_lower)
            curves.append(dict(run=row["id"], stage=row["stage"], method=row["method"], seed=row["seed"],
                hours=row["hours"], seconds=record["source_clock"]-state["started_clock"], outer=record["outer"],
                reliable_lower=lower, reference_cost=ref, paper_gap=(ref-lower)/ref if ref else None,
                availability="posthoc", audit_seconds=record["audit_clock"]-state["started_clock"],
                checkpoint=record["checkpoint"], valid=row["numerical_audit_passed"] and not hash_errors))
            blocks.append(dict(run=row["id"], availability="posthoc", **record))
    controller = read(directory / "state.json")
    final = [r for r in rows if r["stage"] == "confirmation"]
    primary = bool(len(final) == 3 and {r["seed"] for r in final} == set(SEEDS) and not hash_errors and
        controller.get("within_total_budget", False) and all(r["numerical_audit_passed"] and r["reference_certified"] and
        r["paper_gap"] is not None and r["paper_gap"] <= .01 and r["online_integer_master_calls"] == 0 and
        (r["search_allocated_seconds"] or 0.) >= 1200.-1e-5 and
        (r["offline_allocated_seconds"] or 0.) >= 600.-1e-5 for r in final))
    challenge = primary and all(r["paper_gap"] <= .005 for r in final)
    milestone = bool(len(final) == 3 and not hash_errors and all(r["numerical_audit_passed"] and
        r["paper_gap600"] is not None and r["paper_gap600"] <= .01 and
        (r["search_allocated_seconds"] or 0.) >= 600.-1e-5 for r in final))
    summary = dict(protocol=manifest["protocol"], execution_commit=manifest["commit"], source_tag=manifest.get("source_tag"),
        source_hashes_valid=not hash_errors, hash_errors=hash_errors, selection=read(directory / "selection.json"),
        reference={c: {k:reference_record(directory, c).get(k) for k in
                      ("objective", "lower_bound", "mip_gap", "certified")} for c in ("n3-det", "n10-det", "n32-stoch")},
        controller=controller, runs=rows, primary_passed=primary, challenge_passed=challenge,
        milestone600_passed=milestone, certification_policy="post_search_integer_bounds_only")
    save(directory / "summary.json", summary)
    save(directory / "audit.json", dict(passed=not hash_errors and not any(r["errors"] for r in rows),
        hash_errors=hash_errors, primary_passed=primary, challenge_passed=challenge,
        milestone600_passed=milestone, within_total_budget=controller.get("within_total_budget", False),
        all_slots_accounted_for=all(r["state"].get("status") in
            ("completed", "time_limit", "failed", "skipped", "phase_budget_exhausted") for r in rows)))
    columns = ["id", "stage", "method", "seed", "hours", "solver_status", "search_seconds", "offline_seconds",
        "final_cost", "online_lower", "reliable_lower", "paper_gap", "paper_gap600", "reference_cost_difference",
        "primal_gap", "coordination_upper", "posthoc_dual_converged", "numerical_audit_passed", "completed_outer_rounds",
        "multiplier_updates", "online_integer_master_calls", "sa_calls", "sa_new_points", "sa_feasible_points",
        "sa_optimality_cuts", "sa_feasibility_cuts", "qubo_bits_max", "qubo_seconds", "annealing_seconds", "best_source"]
    write_csv(directory / "runs.csv", rows, columns)
    write_csv(directory / "blocks.csv", blocks, ["run", "availability", "checkpoint", "outer", "block", "inner",
        "source_clock", "audit_clock", "lower", "upper", "cuts", "converged", "bound_source", "status", "seconds", "reason"])
    write_csv(directory / "curves.csv", curves, ["run", "stage", "method", "seed", "hours", "seconds", "outer",
        "reliable_lower", "paper_gap", "availability", "audit_seconds", "checkpoint", "valid"])
    plot(directory, curves)
    def number(value):
        return "NA" if value is None else f"{value:.4f}"
    def pct(value):
        return "NA" if value is None else f"{100*value:.4f}%"
    lines = ["# 纯 SA 主问题 Benders：Gap v3 实验", "",
        f"三种子 1200 秒搜索主目标 ≤1%：{primary}；挑战 ≤0.5%：{challenge}；600 秒快照 ≤1%：{milestone}。",
        f"固定分块：{summary['selection'].get('chosen_hours')}h；固定版本：{summary['selection'].get('chosen_variant')}。", "",
        "整数主问题候选由 SA 产生；Gurobi 求连续子问题、连续松弛及协调 LP。整数认证在搜索结束后独立进行。",
        "下界及 Gap 的实线为事后认证，横轴回溯到搜索快照；虚线为搜索期间可用的连续下界。认证结果不反馈搜索。", "",
        "| 阶段 | 方法/种子 | h | 状态 | 搜索秒 | 认证秒 | 可行成本 | 认证下界 | Gap | 600秒Gap | 完整外层 | 在线整数调用 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['stage']} | {r['method']}/{r['seed']} | {r['hours']} | {r['solver_status'] or r['state'].get('status')} | "
            f"{number(r['search_seconds'])} | {number(r['offline_seconds'])} | {number(r['final_cost'])} | "
            f"{number(r['reliable_lower'])} | {pct(r['paper_gap'])} | {pct(r['paper_gap600'])} | "
            f"{r['completed_outer_rounds']} | {r['online_integer_master_calls']} |")
    lines += ["", "Gap=(独立参考成本−认证下界)/独立参考成本。可行成本相对参考、原问题 Gap 及对偶收敛分别列于 runs.csv。",
        "近似协调只使用经过验证的局部可行点形成上支撑平面；内层配额耗尽不记为内层收敛。",
        "独立经典对照用于诊断分解上限；超时和未收敛表示尚未判定。候选来源改进金额是轨迹归类，不表示独立因果贡献。", "",
        "![论文口径 Gap](paper-gap.png)", "", "![认证下界](certified-lower.png)", "",
        "所有失败、中断、迟到事件及未达标运行保留。完整认证覆盖、局部点、割、来源与快照见各运行目录。",
        "源码、输入和参考哈希见 manifest.json；全部交付哈希见 SHA256SUMS.txt。--report-only 可重建数值报告。"]
    (directory / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    for r in final:
        candidates = [row for row in committed_events(directory / "runs" / r["id"] / "events.jsonl") if
            row["kind"] == "evaluation" and row.get("feasible") and row["clock"] <= r["state"].get("deadline", -math.inf)]
        if candidates:
            save(directory / "best-schedules" / f"seed-{r['seed']}.json", min(candidates, key=lambda row:row["objective"]))
    return summary


def archive(directory):
    checksum(directory)
    target = directory.with_suffix(".zip")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(directory).as_posix())
    with zipfile.ZipFile(target) as bundle:
        bad = bundle.testzip()
        if bad:
            raise ValueError(f"archive CRC failure: {bad}")
    digest = sha(target)
    target.with_suffix(".zip.sha256").write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    return dict(path=str(target), sha256=digest, bytes=target.stat().st_size)


def run(directory, previous):
    started = float(os.environ.get("PURE_SA_GAP_STARTED_CLOCK", time.perf_counter()))
    global_deadline = started+TOTAL_SECONDS
    manifest = freeze(directory, previous)
    controller = dict(status="running", started_clock=started, deadline=global_deadline,
                      phase="reference", within_total_budget=True)
    save(directory / "state.json", controller)
    stop_reason = None
    selection = {}
    save(directory / "selection.json", selection)
    try:
        validate_historical(directory)
        if not manifest["cached_reference_eligible"]:
            supervise(directory, directory / "references/n32-stoch", "reference", {}, 600, started+720)
        if not reference_record(directory, "n32-stoch")["certified"]:
            stop_reason = "stochastic_reference_uncertified"
    except (ValueError, AssertionError, KeyError, OSError) as exc:
        stop_reason = "reference_validation_failed:"+str(exc)
    manifest["reference_sha256"] = {p.relative_to(directory).as_posix(): sha(p)
        for p in (directory / "references").rglob("*.json")}
    save(directory / "manifest.json", manifest)
    cumulative = PHASE_SECONDS["reference"]
    for stage, phase_seconds in PHASE_SECONDS.items():
        if stage in ("reference", "audit"):
            continue
        cumulative += phase_seconds
        phase_deadline = min(global_deadline-300., started+cumulative)
        controller.update(phase=stage)
        print(f"phase {stage}, elapsed {time.perf_counter()-started:.2f}s", flush=True)
        for registered in (r for r in manifest["runs"] if r["stage"] == stage):
            output = directory / "runs" / registered["id"]
            if stop_reason:
                save(output / "state.json", dict(status="skipped", reason=stop_reason))
                continue
            definition = {**registered}
            if definition["hours"] is None:
                definition["hours"] = selection.get("chosen_hours")
            if definition["method"] == "selected":
                definition["method"] = selection.get("chosen_variant")
            if definition["hours"] is None or definition["method"] is None:
                stop_reason = "selection_missing"
                save(output / "state.json", dict(status="skipped", reason=stop_reason))
                continue
            save(output / "definition.json", definition)
            controller.update(run_id=registered["id"], seconds=time.perf_counter()-started)
            save(directory / "state.json", controller)
            supervise(directory, output, "search", definition, definition["seconds"], phase_deadline)
            save(output / "search-hashes.json", search_hashes(output))
            if definition["audit_seconds"]:
                supervise(directory, output / "offline", "certify", definition, definition["audit_seconds"], phase_deadline)
            score = statistics(directory, registered)
            save(output / "score.json", score)
            print(f"score {registered['id']}: lower={score['reliable_lower']}, gap={score['paper_gap']}, "
                  f"outer={score['completed_outer_rounds']}, valid={score['numerical_audit_passed']}", flush=True)
        scores = [statistics(directory, r) for r in manifest["runs"] if r["stage"] == stage]
        if not stop_reason and stage == "sanity":
            if not all(r["numerical_audit_passed"] and r["reference_certified"] for r in scores) or any(
                r["case"] == "n3-det" and abs(r["final_cost"]-r["reference_cost"]) > 1e-4 for r in scores):
                stop_reason = "sanity_failed"
        elif not stop_reason and stage == "screening":
            diagnostics = [statistics(directory, r) for r in manifest["runs"] if r["stage"] == "diagnostic"]
            selection.update(select_hours(scores, diagnostics, reference_record(directory, "n32-stoch")["objective"]))
            save(directory / "selection.json", selection)
            if selection["chosen_hours"] is None:
                stop_reason = selection["reason"]
        elif not stop_reason and stage == "ablation":
            selection.update(select_variant(scores))
            save(directory / "selection.json", selection)
            if selection["chosen_variant"] is None:
                stop_reason = selection["reason"]
    audit_started = time.perf_counter()
    # Charge the entire final audit allowance; the receipt records actual wall time.
    charged = audit_started-started+300.
    controller.update(status="completed", phase="audit", stop_reason=stop_reason,
        solver_and_controller_seconds=audit_started-started, final_audit_allowance_seconds=300.,
        accounted_seconds=charged, within_total_budget=charged <= TOTAL_SECONDS+1e-5)
    save(directory / "state.json", controller)
    summary = report(directory)
    archive_record = archive(directory)
    finished = time.perf_counter()
    receipt = dict(protocol=PROTOCOL, started_clock=started, finished_clock=finished,
        total_wall_seconds=finished-started, final_audit_seconds=finished-audit_started,
        budget_seconds=TOTAL_SECONDS, accounted_seconds=charged,
        within_total_budget=finished <= global_deadline and finished-audit_started <= 300.,
        execution_commit=manifest["commit"], archive=archive_record,
        primary_passed=summary["primary_passed"], challenge_passed=summary["challenge_passed"],
        milestone600_passed=summary["milestone600_passed"])
    save(directory.with_suffix(".execution-audit.json"), receipt)
    print(f"completed: {receipt}", flush=True)
    return summary


def worker(args):
    import gurobipy as gp
    gp.setParam("Threads", 1)
    gp.setParam("Seed", 7)
    if args.worker == "reference":
        reference_worker(args.output_dir, args.output_dir / "references/n32-stoch", args.deadline)
    else:
        output = args.output_dir / "runs" / args.identifier
        definition = read(output / "definition.json")
        if args.worker == "search":
            search_worker(args.output_dir, output, definition, args.deadline)
        elif args.worker == "certify":
            certification_worker(args.output_dir, output, definition, args.deadline)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-dir", type=Path)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--worker", choices=("search", "certify", "reference"), help=argparse.SUPPRESS)
    parser.add_argument("--identifier", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.worker:
        worker(args)
    elif args.report_only:
        report(args.output_dir)
        checksum(args.output_dir)
    else:
        if args.previous_dir is None:
            first = subprocess.check_output(["git", "worktree", "list", "--porcelain"], cwd=ROOT, text=True).splitlines()[0]
            args.previous_dir = Path(first.removeprefix("worktree ")) / "experiments/rts24-gap-closure-20261006/revised"
        run(args.output_dir, args.previous_dir.resolve())


if __name__ == "__main__":
    main()
