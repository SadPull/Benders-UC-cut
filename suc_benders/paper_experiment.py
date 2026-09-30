"""Reproducible six-strategy, five-seed paper transfer experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback

from .attribution import now, write_json, validate_result
from .data import load_suc_data
from .full_model import FullConfig, solve_full
from .master import QuantumConfig
from .solver import BendersConfig, solve_benders, result_to_dict

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 17, 27, 37, 47)
STRATEGIES = (
    ("A_legacy", "legacy", "none", 24),
    ("B_compact", "compact", "none", 24),
    ("C_dynamic", "dynamic", "none", 24),
    ("D_block1", "dynamic", "temporal", 1),
    ("E_block2", "dynamic", "temporal", 2),
    ("F_block3", "dynamic", "temporal", 3),
)


def run_config(strategy, seed):
    _, encoding, decomposition, hours = strategy
    return BendersConfig(master_encoding=encoding, decomposition=decomposition, block_hours=hours,
                         block_inner_iterations=3, max_iterations=10, time_limit_seconds=300.,
                         max_candidate_evaluations=40, decommit_candidates=False,
                         collect_diagnostics=True, quantum=QuantumConfig(kaiwu_seed=seed, candidate_pool_size=16))


def prepare(directory, data, scenarios, resume):
    paths = sorted((ROOT/"suc_benders").glob("*.py"))
    sources = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    packages = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "kaiwu", "gurobipy", "matplotlib")}
    settings = {s[0]: asdict(run_config(s, 7)) for s in STRATEGIES}
    identity = {"sources": {k: v for k, v in sources.items() if not k.endswith("paper_report.py")},
                "data": result_to_dict(data), "scenarios": result_to_dict(scenarios), "settings": settings,
                "seeds": SEEDS, "packages": packages,
                "threads": {k: os.getenv(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if (directory/"manifest.json").exists():
        old = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        if not resume or old["fingerprint"] != fingerprint:
            raise ValueError("existing experiment requires --resume and identical solver/input/settings fingerprint")
        return
    directory.mkdir(parents=True, exist_ok=True)
    for path in paths+sorted((ROOT/"tests").glob("*.py"))+[ROOT/"README.md", ROOT/"requirements.txt"]:
        target = directory/"source"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for command, filename in ((["git", "status", "--porcelain=v1"], "git-status.txt"),
                              (["git", "diff", "--binary", "HEAD"], "working-tree.patch"),
                              (["git", "rev-parse", "HEAD"], "commit.txt")):
        (directory/filename).write_bytes(subprocess.check_output(command, cwd=ROOT))
    write_json(directory/"manifest.json", {"created_at": now(), "fingerprint": fingerprint,
                **identity, "sources": sources, "python": sys.version, "reference": {"time_limit": 300, "mip_gap": 1e-4},
                "budget_semantics": "300 seconds checked at phase/call boundaries; SA is not interruptible"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/paper-transfer-20260930-v2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    directory = Path(args.output_dir)
    data, scenarios = load_suc_data(ROOT/"data/suc_10x24.json")
    prepare(directory, data, scenarios, args.resume)
    import gurobipy as gp
    gp.setParam("Threads", 1)
    def progress(message):
        line = f"{now()} {message}"
        print(line, flush=True)
        with (directory/"progress.log").open("a", encoding="utf-8") as stream:
            stream.write(line+"\n")
    for seed in SEEDS[:1] if args.pilot else SEEDS:
        strategies = list(STRATEGIES)
        random.Random(f"paper-transfer:{seed}").shuffle(strategies)
        for strategy in strategies:
            run_id = f"{strategy[0]}__s{seed}"
            path = directory/"runs"/f"{run_id}.json"
            if path.exists():
                continue
            progress(f"START {run_id}")
            try:
                result = solve_benders(data, scenarios, run_config(strategy, seed))
                validation = validate_result(data, scenarios, result)
                payload = {"run_id": run_id, "strategy": strategy[0], "seed": seed,
                           "completed_at": now(), "validation": validation, "result": result_to_dict(result)}
                write_json(path, payload)
                sizes = [r["bits"] for r in result.diagnostics["sampling"]]
                progress(f"DONE {run_id} cost={result.objective:.3f} seconds={result.runtime:.2f} max_bits={max(sizes, default=0)} status={result.status}")
            except Exception:
                # Fail visibly. Do not treat implementation/validation failures as samples.
                (directory/f"{run_id}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                progress(f"ERROR {run_id}; no completion record")
                raise
    if not args.pilot:
        if not (directory/"reference.json").exists():
            progress("START independent full reference")
            reference = solve_full(data, scenarios, FullConfig(time_limit=300, mip_gap=1e-4, output_flag=0))
            payload = {k: v for k, v in vars(reference).items() if k != "model"}
            write_json(directory/"reference.json", payload)
            progress("DONE independent full reference")
    from .paper_report import report
    report(directory)
    progress("REPORT complete")


if __name__ == "__main__":
    main()
