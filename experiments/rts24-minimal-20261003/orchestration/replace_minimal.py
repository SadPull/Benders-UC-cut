"""Run the two user-authorized replacements after the original batch."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

workspace = Path(r"C:\Users\12433\Desktop\LLM-benders-SUC")
root = workspace / "experiments/rts24-minimal-20261003"
snapshot = root / "snapshot"
sys.path.insert(0, str(snapshot))
from suc_benders.minimal_experiment import configuration
from suc_benders.subagent import initialize, publish, sha

sys.stdout = (workspace / "results/minimal-replacement.stdout.log").open("a", encoding="utf-8", buffering=1)
sys.stderr = (workspace / "results/minimal-replacement.stderr.log").open("a", encoding="utf-8", buffering=1)
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
amendments = [json.loads((root / name).read_text(encoding="utf-8"))
              for name in ("protocol-amendment.json", "protocol-amendment-2.json")]
old_state = json.loads((root / "state.json").read_text(encoding="utf-8"))
if old_state["status"] != "completed" or not all(a["authorized_by_user"] for a in amendments):
    raise ValueError("finish original batch and obtain both authorizations first")
if any(p.info["cmdline"] and any("resume_minimal.py" in arg for arg in p.info["cmdline"])
       for p in psutil.process_iter(["cmdline"])):
    raise ValueError("original controller still running")
for path, digest in manifest["source_sha256"].items():
    if sha(snapshot / path) != digest:
        raise ValueError(f"source/input changed: {path}")
publish(root / "original-batch-state.json", old_state)
env = os.environ.copy()
env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
env["PYTHONUNBUFFERED"] = "1"

for amendment in amendments:
    replacement = amendment["replacement"]
    group, seed = replacement["group"], replacement["seed"]
    run = root / replacement["directory"]
    config = configuration(group, seed, run, manifest["rule_max_calls"])
    initialize(run, dict(group=group, seed=seed, configuration=asdict(config),
        replacement_of=f"{group}-{seed}", parent_manifest_sha256=sha(root / "manifest.json")))
    started = time.perf_counter()
    with (run / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "suc_benders.minimal_experiment",
            "--worker", group, "--seed", str(seed), "--output-dir", str(run),
            "--rule-max-calls", str(manifest["rule_max_calls"])],
            cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT)
        publish(root / "state.json", dict(status="replacement_running", current=replacement["directory"],
            controller_pid=os.getpid(), worker_pid=process.pid, started_at=time.time()))
        try:
            process.wait(timeout=2100)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=20)
            publish(run / "state.json", dict(status="worker_timeout"))
    state_path = run / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else dict(status="failed")
    if state.get("status") == "running":
        state["status"] = "failed"
    state.update(exitcode=process.returncode, process_wall_seconds=time.perf_counter()-started)
    publish(state_path, state)
    print(dict(run=replacement["directory"], **state), flush=True)
publish(root / "state.json", dict(status="replacements_completed", completed_at=time.time()))
