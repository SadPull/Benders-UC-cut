"""Rebuild the archived search, retry call 2, then finish the paired experiment."""
import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter, sleep, time

import numpy as np

import run_two_arm_experiment as experiment
from run_suc import run_suc
import subagent_advisor
from suc_benders import llm, solver
from suc_benders.config import ROOT, load_options
from suc_benders.data import load_data
from suc_benders.model import model, partition
from suc_benders.validation import commitment_hash


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def same(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, rtol=1e-10, atol=1e-4))
    return left == right


class ScoreClock:
    def __init__(self):
        self.origin = perf_counter()
        self.elapsed = 0.
        self.live_at = None

    def __call__(self):
        return self.origin + self.elapsed + (0. if self.live_at is None else perf_counter()-self.live_at)

    def resume(self, elapsed):
        if elapsed < self.elapsed:
            raise ValueError("Restored clock would move backwards")
        self.elapsed, self.live_at = elapsed, perf_counter()


class RestoreChecked(BaseException):
    pass


class Advisor:
    def __init__(self, previous, output, clock, check_only=False):
        self.previous, self.output, self.clock, self.check_only = previous, output, clock, check_only
        self.used = read(previous / "summary.json")["paused_run"]["last_snapshot_elapsed_seconds"]
        self.retry = previous / "retries/002-01"
        self.checked = False

    def __call__(self, d, scenarios, best, o, seconds, context, bases, iteration):
        body = llm.request_body(d, scenarios, best, o, context, bases, iteration)
        if iteration <= 4:
            old = self.previous / "D-7/calls" / f"{iteration:03d}"
            if not same(json.loads(body["input"]), read(old / "user.json")):
                raise RuntimeError(f"Archived context mismatch at call {iteration}")
        if iteration <= 3:
            consumed = read(old / "consumed.json")
            self.clock.elapsed += consumed["elapsed_seconds"]
            if consumed["status"] != "ACCEPTED":
                return [], dict(status="FAILED", message=consumed["error_type"], accepted=0,
                                format_valid=0, proposals=[], seconds=consumed["elapsed_seconds"],
                                transport="codex_subagent", replayed=True)
            receipt, request = read(old / "receipt.json"), read(old / "request.json")
            if (receipt["request_id"] != request["request_id"] or receipt["response_sha256"] !=
                    hashlib.sha256((old / "response.json").read_bytes()).hexdigest()):
                raise RuntimeError("Archived response integrity mismatch")
            candidates, info = parse(read(old / "response.json"), d, iteration)
            info.update(seconds=consumed["elapsed_seconds"], agent_id=receipt["agent_id"],
                        transport="codex_subagent", replayed=True)
            return candidates, info
        if iteration == 4:
            checkpoint = read(self.previous / "D-7/checkpoint.json")
            if commitment_hash(best) != checkpoint["commitment_id"] or abs(best.objective-checkpoint["objective"]) > 1e-3:
                raise RuntimeError("Pause checkpoint mismatch")
            self.checked = True
            experiment.write_json(self.output / "restoration.json", dict(status="PASSED", objective=best.objective,
                                  commitment_id=commitment_hash(best), archive_contexts_matched=True,
                                  historical_active_seconds=self.used,
                                  historical_curve_times="Reconstructed from archived request waits; not original evaluation timestamps"))
            if self.check_only:
                raise RestoreChecked
            retry_request = read(self.retry / "request.json")
            retry_active = (read(self.retry / "consumed.json")["elapsed_seconds"] if (self.retry / "consumed.json").exists()
                            else time()-retry_request["created_at"])
            self.clock.resume(self.used + retry_active)
            while not (self.retry / "consumed.json").exists() and solver.perf_counter() < self.clock.origin+o.time_limit:
                sleep(.1)
            consumed = read(self.retry / "consumed.json") if (self.retry / "consumed.json").exists() else None
            directory = self.output / "D-7/calls/004"
            directory.mkdir()
            for name in ("system.txt", "user.json", "request.json", "response.json", "receipt.json", "consumed.json", "dispatch.json"):
                if (self.retry / name).exists():
                    shutil.copy2(self.retry / name, directory / name)
            (directory / "evaluation_context.json").write_text(body["input"], encoding="utf-8")
            if consumed is None or consumed["status"] != "ACCEPTED" or solver.perf_counter() >= self.clock.origin+o.time_limit:
                return [], dict(status="TIME_LIMIT" if solver.perf_counter() >= self.clock.origin+o.time_limit else "FAILED",
                                message="Retry not accepted before search deadline", accepted=0, format_valid=0,
                                proposals=[], seconds=consumed["elapsed_seconds"] if consumed else 0.,
                                transport="codex_subagent", retry_of=2)
            candidates, info = parse(read(self.retry / "response.json"), d, iteration)
            info.update(seconds=consumed["elapsed_seconds"], agent_id=read(self.retry / "receipt.json")["agent_id"],
                        transport="codex_subagent", retry_of=2, prompt_context="Archived call 2; evaluated at restored incumbent")
            return candidates, info
        return subagent_advisor.request(d, scenarios, best, o, seconds, context, bases, iteration,
                                        directory=self.output / "D-7/calls")


def parse(payload, d, iteration):
    return llm.parse_response({"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps(payload)}]}]}, d, iteration)


def run(previous, output, resume_seconds, check_only=False):
    previous, output = Path(previous), Path(output)
    old_manifest = read(previous / "manifest.json")
    experiment.verify_sources(old_manifest["source_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    used = read(previous / "summary.json")["paused_run"]["last_snapshot_elapsed_seconds"]
    o = load_options(dict(old_manifest["options"], time_limit=used+resume_seconds,
                          llm_url="codex-subagent", llm_api_key="not-used"))
    clock, original_clock, original_request = ScoreClock(), solver.perf_counter, llm.request
    advisor = Advisor(previous, output, clock, check_only)
    summary, curves = dict(status="RUNNING", runs=[], reference=read(previous / "summary.json")["reference"],
                           budget_seconds=o.time_limit, resume_seconds=resume_seconds,
                           historical_active_seconds=used, previous_directory=str(previous.resolve())), []
    try:
        if not check_only:
            hashes = experiment.freeze(output, o)
            source = Path(__file__).resolve()
            shutil.copy2(source, output / "snapshot" / source.name)
            hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = read(output / "manifest.json")
            manifest["source_sha256"] = hashes
            manifest["resumption"] = dict(previous=str(previous.resolve()), replay_archived_calls=[1, 2, 3],
                  retry_of_call=2, retry_admitted_as_call=4, resumed_budget_seconds=resume_seconds,
                  retry_wait_counts_toward_resumed_budget=True,
                  paused_and_post_retry_approval_wait_excluded=True,
                  reconstruction_not_charged_separately=True)
            experiment.write_json(output / "manifest.json", manifest)
            experiment.write_json(output / "kaiwu_check.json", experiment.check_kaiwu(o))
            shutil.copytree(previous / "reference", output / "reference")
            (output / "D-7/calls").mkdir(parents=True)
            for i in (1, 2, 3):
                shutil.copytree(previous / "D-7/calls" / f"{i:03d}", output / "D-7/calls" / f"{i:03d}")
        d, scenarios = load_data(o.data_file, o.scenario_file)
        m = model(d, scenarios); full = partition(m, d.horizon)[0][0]
        for name in experiment.ORDER:
            if check_only and name == "A-7":
                break
            run_dir = output / name; run_dir.mkdir(exist_ok=True)
            o.llm_enabled = name.startswith("D")
            experiment.write_json(run_dir / "options.json", experiment.public_options(o))
            experiment.write_json(output / "state.json", dict(status="RUNNING", stage=name))
            solver.perf_counter = clock if o.llm_enabled else original_clock
            llm.request = advisor if o.llm_enabled else original_request
            print(f"{name} started", flush=True)
            with (run_dir / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = run_suc(asdict(o))
            if name == "D-7" and not advisor.checked:
                raise RuntimeError("Restoration did not reach the archived pause point")
            experiment.write_json(run_dir / "result.json", result)
            for i, info in enumerate(result.llm_history, 1):
                experiment.write_json(run_dir / "calls" / f"{i:03d}" / "feedback.json", info)
            if any(row["elapsed_seconds"] > o.time_limit for row in result.diagnostics["evaluations"]):
                raise RuntimeError("Evaluation after search deadline")
            checked = experiment.audit(m, full, result, run_dir)
            summary["runs"].append(experiment.summarize(result, name, summary["reference"]["objective"], checked))
            curves.extend(dict(run=name, **{key: row[key] for key in
                ("elapsed_seconds", "source", "objective", "incumbent", "improved", "sa_calls")})
                for row in result.diagnostics["evaluations"])
            print(f"{name}: {result.status}, cost={result.objective:.6f}, SA={result.diagnostics['sa_calls']}, LLM={result.llm_calls}", flush=True)
            experiment.report(output, summary, curves)
        if not check_only:
            by_name = {row["run"]: row for row in summary["runs"]}
            summary.update(status="COMPLETED", paired_improvement=(by_name["A-7"]["objective"]-by_name["D-7"]["objective"])/by_name["A-7"]["objective"])
    except RestoreChecked:
        summary["status"] = "RESTORATION_VERIFIED"
    except Exception as error:
        summary.update(status="FAILED", error_type=type(error).__name__)
        if isinstance(error, RuntimeError):
            summary["error"] = str(error)
    finally:
        solver.perf_counter, llm.request = original_clock, original_request
    experiment.write_json(output / "state.json", dict(status=summary["status"]))
    experiment.report(output, summary, curves)
    path = output / "report.md"
    path.write_text(path.read_text(encoding="utf-8").replace("a 1800 s budget", f"a {o.time_limit:.3f} s cumulative active budget"), encoding="utf-8")
    print(summary["status"], flush=True)
    return 0 if summary["status"] in ("COMPLETED", "RESTORATION_VERIFIED") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-seconds", type=float, default=1800)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.previous_dir, args.output_dir, args.resume_seconds, args.check_only))
