"""Finish the still-open third call without restarting the live C/17 worker or its budget."""
import json
from pathlib import Path
import sys
import time

workspace = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace))
from suc_benders.recovery_experiment import serve_call, worker_environment, supervise

root = Path(__file__).resolve().parents[1]
run = root / "C-17"
deadline = json.loads((run / "ready.json").read_text(encoding="utf-8"))["deadline_clock"]
codex = Path(json.loads((root / "manifest.json").read_text(encoding="utf-8"))["codex_executable"])
serve_call(run, run / "calls/003", codex, deadline, worker_environment())
while not (run / "result.json").exists() and time.perf_counter() < deadline+300:
    time.sleep(.2)
supervise(root, codex, resume=True)
