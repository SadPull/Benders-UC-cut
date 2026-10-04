"""Audit all attempts and assemble the user-authorized complete-run analysis."""
import csv
import json
from pathlib import Path
import sys
import time

workspace = Path(r"C:\Users\12433\Desktop\LLM-benders-SUC")
sys.path.insert(0, str(workspace))
from suc_benders.data import load_suc_data
from suc_benders.minimal_report import summarize
from suc_benders.rts24_experiment import read_events
from suc_benders.subagent import publish, sha
from suc_benders.validation import schedule_object, validate_schedule

root = workspace / "experiments/rts24-minimal-20261003"
def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

started = time.perf_counter()
manifest = read(root / "manifest.json")
amendments = [read(root / name) for name in ("protocol-amendment.json", "protocol-amendment-2.json")]
if read(root / "state.json").get("status") != "replacements_completed":
    raise ValueError("finish both authorized replacements first")
reference = read(root / "reference/result.json")
data, scenarios = load_suc_data(root / "snapshot/data/rts24/units.json", root / "snapshot/data/rts24/stochastic.json")
attempts = []
source_ok = all(sha(root / "snapshot" / p) == digest for p, digest in manifest["source_sha256"].items())
paths = [(f"{g}-{s}", g, s) for g, s in manifest["order"]] + [
    (a["replacement"]["directory"], a["replacement"]["group"], a["replacement"]["seed"]) for a in amendments]
replaced = {f"{a['replacement']['group']}-{a['replacement']['seed']}" for a in amendments}
for name, group, seed in paths:
    run = root / name
    events = read_events(run / "events.jsonl")
    result, state = read(run / "result.json"), read(run / "state.json")
    summary = summarize(events, result, 1800, reference["objective"])
    first_llm = summary["first_llm_improvement_seconds"]
    after_llm = [e for e in events if first_llm is not None and first_llm < e["elapsed_seconds"] <= 1800]
    summary["post_llm_sa"] = dict(
        calls=sum(e["kind"] == "sa_start" for e in after_llm),
        complete_evaluations=sum(e["kind"] == "evaluation" and e["source"] == "sa" for e in after_llm),
        feasible_evaluations=sum(e["kind"] == "evaluation" and e["source"] == "sa" and e["feasible"] for e in after_llm))
    errors, checked = [], 0
    for e in events:
        if e["kind"] == "evaluation" and e["feasible"] and e["elapsed_seconds"] <= 1800:
            try:
                validate_schedule(data, scenarios, schedule_object(e))
                checked += 1
            except (AssertionError, ValueError) as exc:
                errors.append(str(exc))
    seeds = summary["sa_seeds"]
    protocol_ok = (seeds == [seed+i if group == "B" else seed for i in range(len(seeds))]
        and summary["llm_started"] <= (3 if group in {"C", "D"} else 0)
        and (group != "C" or not seeds)
        and (group != "E" or summary["rule_started"] <= 3)
        and all(e["diagnostics"].get("raw_count", 0) <= 4 for e in events if e["kind"] == "llm_end"))
    complete = state.get("status") in {"TIME_LIMIT", "LLM_CALL_LIMIT"} and bool(result)
    initial = next((e for e in events if e["kind"] == "evaluation"), {})
    audit = dict(physical_checked=checked, errors=errors, protocol_ok=protocol_ok,
        result_objective_consistent=bool(result and result.get("objective") == summary["objective"]),
        initial_hash=initial.get("commitment_hash"), complete=complete)
    summary.update(group=group, seed=seed, run_directory=name, state=state, audit=audit,
        selected_for_paired_analysis=name not in replaced, replacement=name.startswith("replacements/"),
        llm_responses_consumed=sum(e["kind"] == "llm_end" and
            e["diagnostics"].get("status") == "completed" for e in events),
        llm_failed=sum(e["kind"] == "llm_end" and
            e["diagnostics"].get("failed", False) for e in events))
    summary["llm_transport_status"] = ("not_applicable" if group not in {"C", "D"} else
        "no_response" if not summary["llm_responses_consumed"] else
        "all_three_received" if summary["llm_responses_consumed"] == 3 else "partial")
    publish(run / "final-audit.json", audit)
    if name.startswith("replacements/"):
        publish(run / "summary.json", summary)
    attempts.append(summary)
selected = [r for r in attempts if r["selected_for_paired_analysis"]]
llm_runs = [r for r in selected if r["group"] in {"C", "D"}]
paired_response_seeds = [seed for seed in (7, 17, 27)
    if all(any(r["group"] == group and r["seed"] == seed and r["llm_responses_consumed"] > 0
               for r in llm_runs) for group in ("C", "D"))]
initial_hashes = {r["audit"]["initial_hash"] for r in selected}
llm_started = sum(r["llm_started"] for r in attempts)
handoffs = []
call_integrity_errors = []
for request_path in sorted(root.rglob("calls/*/request.json")):
    call = request_path.parent
    request, dispatch, receipt = read(request_path), read(call / "dispatch.json"), read(call / "receipt.json")
    output = call / "candidates.json"
    label = str(call.relative_to(root))
    if (request.get("model"), request.get("reasoning_effort"), request.get("fork_turns")) != ("gpt-6-astra", "max", "none"):
        call_integrity_errors.append(f"{label}: model/session configuration")
    for name, digest in request["input_sha256"].items():
        if sha(call / name) != digest:
            call_integrity_errors.append(f"{label}: input {name}")
    if receipt:
        for key in ("run_id", "call_id", "model", "reasoning_effort", "input_sha256"):
            if receipt.get(key) != request.get(key):
                call_integrity_errors.append(f"{label}: receipt {key}")
        if receipt.get("status") == "completed" and (not output.exists() or sha(output) != receipt.get("output_sha256")):
            call_integrity_errors.append(f"{label}: output hash")
    handoffs.append(dict(call=str(call.relative_to(root)),
        output_exists=output.exists(), receipt_status=receipt.get("status"),
        dispatch_delay_seconds=dispatch.get("claimed_at", request["created_at"])-request["created_at"] if dispatch else None,
        file_ready_after_claim_seconds=output.stat().st_mtime-dispatch["claimed_at"] if output.exists() and dispatch else None,
        receipt_delay_after_file_seconds=receipt["completed_at"]-output.stat().st_mtime if receipt and output.exists() else None,
        consumer_status=read(call / "consumed.json").get("status")))
passed = bool(source_ok and not call_integrity_errors and len(selected) == 15 and len(initial_hashes) == 1 and None not in initial_hashes
    and llm_started <= amendments[-1]["amended_max_llm_calls"]
    and read(root / "reference/audit.json").get("feasible_solution")
    and all(r["objective"] is not None and r["audit"]["physical_checked"] > 0
            and r["audit"]["complete"] and r["audit"]["protocol_ok"] and not r["audit"]["errors"]
            and r["audit"]["result_objective_consistent"] for r in selected))
payload = dict(passed=passed, complete=all(r["audit"]["complete"] for r in selected),
    numerical_audit_passed=passed,
    selected_llm_runs_with_response=sum(r["llm_responses_consumed"] > 0 for r in llm_runs),
    selected_llm_runs_all_three_received=sum(r["llm_responses_consumed"] == 3 for r in llm_runs),
    llm_response_replications_complete=all(r["llm_responses_consumed"] > 0 for r in llm_runs),
    paired_seeds_with_llm_response=paired_response_seeds,
    normal_completed_runs=sum(r["audit"]["complete"] for r in attempts),
    attempted_runs=len(attempts), source_hashes_match=source_ok, common_initialization=len(initial_hashes) == 1,
    reference_cost=reference["objective"], reference_lower_bound=reference["lower_bound"],
    llm_started=llm_started, amended_call_limit=amendments[-1]["amended_max_llm_calls"], runs=selected,
    attempts=attempts, interruption=read(root / "D-17/interruption.json"),
    transport_failure=read(root / "C-17/transport-failure.json"),
    protocol_amendments=amendments, handoff_timings=handoffs,
    call_integrity_errors=call_integrity_errors,
    llm_output_files=sum(h["output_exists"] for h in handoffs),
    llm_completed_receipts=sum(h["receipt_status"] == "completed" for h in handoffs),
    llm_consumed=sum(h["consumer_status"] == "completed" for h in handoffs),
    selected_runs_without_llm_response=[r["run_directory"] for r in selected
        if r["llm_transport_status"] == "no_response"],
    posthoc_audit_seconds=time.perf_counter()-started)
publish(root / "final-summary.json", payload)
with (root / "analysis-curves.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["group", "seed", "run_directory", "seconds", "objective", "source", "gain"])
    writer.writeheader()
    for r in selected:
        writer.writerows(dict(group=r["group"], seed=r["seed"], run_directory=r["run_directory"], **p) for p in r["curve"])
print(json.dumps({k: payload[k] for k in ("passed", "normal_completed_runs", "attempted_runs", "llm_started")}))
if not passed:
    raise SystemExit("final audit did not pass; preserve all evidence")
