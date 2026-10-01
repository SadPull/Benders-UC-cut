"""Paired, deadline-enforced dynamic SA / SA+LLM efficiency experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, ProxyHandler, build_opener
from urllib.error import HTTPError

from .attribution import write_json
from .data import load_suc_data
from .llm import LLMConfig
from .master import QuantumConfig
from .solver import BendersConfig, result_to_dict, solve_benders

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 17, 27, 37, 47)
BUDGET = 600.
THRESHOLDS = (.10, .08, .06, .05, .04, .03, .02, .01)
CHECKPOINTS = (60, 120, 300, 600)


def load_credentials():
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("LLM_API_KEY="):
                os.environ.setdefault("LLM_API_KEY", line.split("=", 1)[1].strip())
    if not os.getenv("LLM_API_KEY"):
        raise ValueError("LLM_API_KEY is required")


def llm_config(enabled=True):
    return LLMConfig(enabled=enabled, api_format="chat_completions",
        base_url="http://47.101.31.14:8300/v1", model="deepseek-v4.1-flash",
        reasoning_effort="max", max_tokens=65536, timeout_seconds=180.,
        max_retries=1, collect_diagnostics=True)


def preflight():
    """Fail closed on incompatible parameters; HTTP success is not reasoning proof."""
    load_credentials()
    config = llm_config()
    payload = {"model": config.model, "reasoning_effort": config.reasoning_effort,
               "max_tokens": config.max_tokens, "temperature": config.temperature, "stream": False,
               "messages": [{"role": "user", "content": 'Return exactly {"candidates":[]} without markdown.'}]}
    token = os.environ["LLM_API_KEY"]
    result = {"request": payload, "accepted": False, "reasoning_verified": False}
    started = time.perf_counter()
    try:
        request = Request(config.base_url + "/chat/completions", data=json.dumps(payload).encode(),
                          headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        with build_opener(ProxyHandler({})).open(request, timeout=180) as response:
            result.update(status=response.status, response=json.load(response))
        choice = result["response"]["choices"][0]
        result["accepted"] = choice.get("finish_reason") != "length" and json.loads(choice["message"]["content"]) == {"candidates": []}
    except HTTPError as exc:
        result.update(status=exc.code, error=exc.read(4096).decode(errors="replace"))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        result["error"] = type(exc).__name__
    result["seconds"] = time.perf_counter()-started
    return json.loads(json.dumps(result).replace(token, "[REDACTED]"))


def configuration(seed, strategy, budget=BUDGET):
    return BendersConfig(master_encoding="dynamic", decomposition="none",
        decommit_candidates=False, random_candidates=False, run_to_deadline=True,
        time_limit_seconds=budget, max_candidate_evaluations=None, collect_diagnostics=True,
        quantum=QuantumConfig(kaiwu_seed=seed, candidate_pool_size=16),
        llm=llm_config(strategy == "sa_llm"))


def worker(directory, seed, strategy, budget, connection):
    """The supervisor owns the hard deadline, including uninterruptible SDK calls."""
    path = Path(directory)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import gurobipy as gp
    gp.setParam("Threads", 1)
    data, scenarios = load_suc_data(ROOT / "data/suc_10x24.json")
    with (path / "events.jsonl").open("a", encoding="utf-8", buffering=1) as stream:
        def event(row):
            if row["kind"] == "started":
                connection.send(row["deadline_clock"])
            stream.write(json.dumps(result_to_dict(row), ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
        config = configuration(seed, strategy, budget)
        config.event_sink = event
        try:
            result = solve_benders(data, scenarios, config)
            write_json(path / "result.json", result)
        except Exception as exc:
            # Exception text may contain provider secrets; retain only the type.
            write_json(path / "error.json", {"type": type(exc).__name__})
            raise


def read_events(path):
    rows = []
    if not path.exists():
        return rows
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1:
                raise
    return rows


def supervise(path, seed, strategy, budget=BUDGET):
    path.mkdir(parents=True)
    write_json(path / "state.json", {"status": "running", "seed": seed, "strategy": strategy, "budget": budget})
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=worker, args=(str(path), seed, strategy, budget, child))
    process.start()
    child.close()
    status = "failed"
    try:
        # Imports and license initialization precede optimization; guard a hung startup.
        if not parent.poll(60):
            status = "startup_failed"
        else:
            deadline = parent.recv()
            while process.is_alive() and time.perf_counter() < deadline:
                process.join(min(1., max(0., deadline-time.perf_counter())))
            status = "budget_complete" if time.perf_counter() >= deadline else "failed"
            if not process.is_alive() and process.exitcode == 0:
                status = "completed"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
        write_json(path / "state.json", {"status": status, "seed": seed, "strategy": strategy,
                                         "budget": budget, "exitcode": process.exitcode})


def prepare(directory, resume):
    data, scenarios = load_suc_data(ROOT / "data/suc_10x24.json")
    paths = sorted((ROOT / "suc_benders").glob("*.py"))
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    identity = {"sources": sources, "data": result_to_dict(data), "scenarios": result_to_dict(scenarios),
                "seeds": SEEDS, "configurations": {s: asdict(configuration(7, s)) for s in ("sa", "sa_llm")},
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "gurobipy", "kaiwu")},
                "python": sys.version, "thresholds": THRESHOLDS, "checkpoints": CHECKPOINTS}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = directory / "manifest.json"
    if manifest.exists():
        if not resume or json.loads(manifest.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("resume requires identical input, configuration, software and source fingerprint")
        return
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("output directory must be new")
    directory.mkdir(parents=True, exist_ok=True)
    for path in paths + sorted((ROOT / "tests").glob("*.py")) + [ROOT / "data/suc_10x24.json"]:
        target = directory / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    # Check input identity before using the old reference, without exposing it to workers.
    reference_dir = ROOT / "results/paper-transfer-20260930-v2"
    old = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8"))
    if old["data"] != identity["data"] or old["scenarios"] != identity["scenarios"]:
        raise ValueError("reference input mismatch")
    shutil.copy2(reference_dir / "reference.json", directory / "reference.json")
    reference_hash = hashlib.sha256((directory / "reference.json").read_bytes()).hexdigest()
    write_json(manifest, {"fingerprint": fingerprint, **identity, "reference_sha256": reference_hash})


def summarize(events, budget, reference):
    evaluations = [e for e in events if e["kind"] == "evaluation" and e["elapsed_seconds"] <= budget
                   and e["feasible"] and e["objective"] is not None]
    def best_at(seconds):
        rows = [e for e in evaluations if e["elapsed_seconds"] <= seconds]
        return min(rows, key=lambda e:e["objective"]) if rows else None
    best = best_at(budget)
    calls = [e for e in events if e["kind"] == "llm_end"]
    proposals = [p for e in calls for p in e["diagnostics"].get("proposals", [])]
    iterations = [e for e in events if e["kind"] == "iteration" and e["elapsed_seconds"] <= budget]
    timings = iterations[-1]["timings"] if iterations else {}
    tokens = [e["diagnostics"].get("usage") for e in calls]
    return {"objective": best["objective"] if best else None,
            "relative_difference": best["objective"]/reference-1 if best else None,
            "sa_calls": sum(e["kind"] == "sa_start" for e in events),
            "sa_completed": sum(e["kind"] == "sa_end" and e["status"] == "completed" for e in events),
            "llm_calls": sum(e["kind"] == "llm_start" for e in events),
            "llm_completed": sum(e["kind"] == "llm_end" for e in events),
            "dispatch_calls": sum(e["kind"] == "dispatch_start" for e in events),
            "llm_failed": sum(e["diagnostics"].get("failed", False) for e in calls),
            "llm_seconds_completed": sum(e["diagnostics"].get("elapsed_seconds", 0.) for e in calls),
            "llm_proposals": len(proposals),
            "llm_duplicate": sum(p.get("outcome") == "duplicate" for p in proposals),
            "llm_accepted": sum(p.get("outcome") == "accepted" for p in proposals),
            "llm_usage": tokens,
            "last_complete_iteration_timings": timings,
            "reliable_lower": iterations[-1]["reliable_lower"] if iterations else None,
            "validated_candidates": len(evaluations),
            "checkpoints": {str(t): best_at(t)["objective"] if best_at(t) else None for t in CHECKPOINTS},
            "targets": {str(g): next(({"seconds":e["elapsed_seconds"], "sa_calls":e["sa_calls"]}
                for e in evaluations if e["objective"] <= reference*(1+g)), None) for g in THRESHOLDS}}


def report(directory):
    from statistics import mean, median
    reference = json.loads((directory / "reference.json").read_text(encoding="utf-8"))["objective"]
    rows = []
    for path in sorted((directory / "runs").glob("*")):
        state = json.loads((path / "state.json").read_text(encoding="utf-8"))
        if state["status"] == "running":
            state["status"] = "interrupted"
            write_json(path / "state.json", state)
        events = read_events(path / "events.jsonl")
        row = {**state, **summarize(events, state["budget"], reference)}
        rows.append(row)
        write_json(path / "summary.json", row)
    pairs = []
    for seed in SEEDS:
        pair = {r["strategy"]:r for r in rows if r["seed"] == seed}
        if len(pair) != 2 or any(r["status"] not in ("completed", "budget_complete") for r in pair.values()):
            continue
        a, b = pair["sa"], pair["sa_llm"]
        delta = b["objective"]-a["objective"] if a["objective"] is not None and b["objective"] is not None else None
        savings = {}
        for g in THRESHOLDS:
            x, y = a["targets"][str(g)], b["targets"][str(g)]
            savings[str(g)] = (1-y["sa_calls"]/x["sa_calls"]) if x and y and x["sa_calls"] else None
        pairs.append({"seed":seed, "cost_delta_llm_minus_sa":delta, "sa_savings":savings})
    differences = [p["cost_delta_llm_minus_sa"] for p in pairs if p["cost_delta_llm_minus_sa"] is not None]
    stats = {"mean_delta":mean(differences) if differences else None, "median_delta":median(differences) if differences else None,
             "wins":sum(d < -1e-4 for d in differences), "ties":sum(abs(d) <= 1e-4 for d in differences), "losses":sum(d > 1e-4 for d in differences)}
    write_json(directory / "summary.json", {"reference": reference, "runs":rows, "pairs":pairs, "statistics":stats})
    lines = ["# LLM 辅助 SA 效率实验", "", f"参考目标：{reference:.6f}。每次600秒，包含模型构建、SA、LLM网络与推理、修复和验证。",
             "LLM请求max与65536 tokens；接口接受请求，但未能独立确认后端实际推理强度。",
             "仅采用截止前验证完成的调度；预算终止的未完成SA/LLM计入调用数。独立减开机关闭。", "",
             "|种子|组别|状态|成本|相对参考差距|SA调用|LLM调用|验证候选|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        cost = f"{r['objective']:.4f}" if r['objective'] is not None else "未获得"
        gap = f"{r['relative_difference']:.4%}" if r['relative_difference'] is not None else "—"
        lines.append(f"|{r['seed']}|{r['strategy']}|{r['status']}|{cost}|{gap}|{r['sa_calls']}|{r['llm_calls']}|{r['validated_candidates']}|")
    lines += ["", f"完整配对成本差（LLM−SA）：均值 {stats['mean_delta']}，中位数 {stats['median_delta']}；胜/平/负 {stats['wins']}/{stats['ties']}/{stats['losses']}。", "",
              "## 时间检查点", "", "|种子/组别|60秒|120秒|300秒|600秒|", "|---|---:|---:|---:|---:|"]
    for r in rows:
        values = [str(r['checkpoints'][str(t)]) if r['checkpoints'][str(t)] is not None else "未获得" for t in CHECKPOINTS]
        lines.append(f"|{r['seed']}/{r['strategy']}|"+"|".join(values)+"|")
    lines += ["", "## 同质量阈值", "", "未达到保留为未达到；零SA基线时节省比例未定义。阈值在运行前固定，不能事后挑选。", "",
              "|阈值|SA达到数/计划5次|LLM达到数/计划5次|配对节省比例（按种子）|", "|---|---|---|---|"]
    for g in THRESHOLDS:
        counts = [sum(r['targets'][str(g)] is not None for r in rows if r['strategy'] == s and r['status'] in ('completed','budget_complete')) for s in ('sa','sa_llm')]
        values = ', '.join(f"{p['seed']}: {p['sa_savings'][str(g)]:.2%}" if p['sa_savings'][str(g)] is not None else f"{p['seed']}: 未定义/未共同达到" for p in pairs)
        lines.append(f"|{g:.0%}|{counts[0]}/5|{counts[1]}/5|{values}|")
    lines += ["", "|种子/组别|阈值|首次达到秒数|累计SA调用|", "|---|---:|---:|---:|"]
    for r in rows:
        for g in THRESHOLDS:
            target = r["targets"][str(g)]
            lines.append(f"|{r['seed']}/{r['strategy']}|{g:.0%}|"+
                         (f"{target['seconds']:.3f}|{target['sa_calls']}|" if target else "未达到|未达到|"))
    lines += ["", "## 开销和有效性", "", "|种子/组别|调度调用|LLM完成/发起|失败|完成请求耗时|建议/接受/重复|", "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"|{r['seed']}/{r['strategy']}|{r['dispatch_calls']}|{r['llm_completed']}/{r['llm_calls']}|{r['llm_failed']}|{r['llm_seconds_completed']:.3f}|{r['llm_proposals']}/{r['llm_accepted']}/{r['llm_duplicate']}|")
    lines += ["", "接受指修复和去重后候选；最终有效性以完整三场景验证为准。完成请求耗时不包含截止时尚未返回的请求，后者仍消耗总预算。",
              "各阶段累计耗时保存至最后一个完整轮次；中断轮次未完成阶段不归零。完整原始事件保留以审计中断调用。"]
    lines += ["", "逐种子的首次达到时间、累计SA次数、连续调度次数见summary.json；脱敏请求响应及完整调度见各运行events.jsonl。",
              "SA调用减少只表示后端调用次数减少，不代表总计算量减少；模型会改变后续割与QUBO，本实验是整体策略比较。",
              "不完整运行不参与完整配对统计，不自动补跑。历史参考解未反馈到求解流程。"]
    (directory / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    directory = Path(args.output_dir or f"results/llm-sa-efficiency-{datetime.now():%Y%m%d-%H%M%S}").resolve()
    if args.report_only:
        report(directory)
        return
    load_credentials()
    prepare(directory, args.resume)
    preflight_path = directory / "preflight.json"
    if not preflight_path.exists():
        write_json(preflight_path, preflight())
    if not json.loads(preflight_path.read_text(encoding="utf-8"))["accepted"]:
        raise RuntimeError("preflight failed; formal runs not started")
    for index, seed in enumerate(SEEDS):
        for strategy in (("sa", "sa_llm") if index % 2 == 0 else ("sa_llm", "sa")):
            path = directory / "runs" / f"{strategy}__s{seed}"
            if path.exists():
                print(f"SKIP existing {path.name}", flush=True)
                continue
            print(f"START {path.name}", flush=True)
            supervise(path, seed, strategy)
            print(f"DONE {path.name}", flush=True)
            report(directory)
    report(directory)


if __name__ == "__main__":
    main()
