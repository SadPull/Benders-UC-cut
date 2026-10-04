import json
from pathlib import Path
import sys
import time

root = Path(r"C:\Users\12433\Desktop\LLM-benders-SUC\experiments\rts24-minimal-20261003")
def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
def status():
    state = read(root / "state.json")
    current = root / state.get("current", "reference")
    best = read(current / "best-checkpoint.json")
    progress = read(current / "progress.json")
    pending = [str(p.parent) for p in root.rglob("calls/*/request.json")
               if not any((p.parent / name).exists() for name in ("dispatch.json", "consumed.json", "receipt.json"))]
    claimed_open = [str(p.parent) for p in root.rglob("calls/*/dispatch.json")
                    if not any((p.parent / name).exists() for name in ("consumed.json", "receipt.json"))]
    return dict(status=state.get("status"), current=state.get("current"),
        progress={k: progress.get(k) for k in ("kind", "elapsed_seconds", "sa_calls", "iteration")},
        best={k: best.get(k) for k in ("objective", "elapsed_seconds", "source")},
        run_status=read(current / "state.json").get("status"), pending=pending, claimed_open=claimed_open,
        finished=read(root / "summary.json").get("completed_searches", 0))
previous_path = root.parent.parent / "results/minimal-last-status.json"
previous = read(previous_path)
deadline = time.monotonic()+(float(sys.argv[1]) if len(sys.argv) > 1 else 0)
while True:
    value = status()
    changed = (any(value.get(k) != previous.get(k) for k in ("status", "current", "pending", "claimed_open"))
               if "--handoff-only" in sys.argv else value != previous)
    if changed or time.monotonic() >= deadline:
        if "--claim" in sys.argv and value["pending"]:
            sys.path.insert(0, str(root.parent.parent))
            from suc_benders.subagent import claim
            call = Path(value["pending"][0])
            value["dispatch_instruction"] = claim(call.parent.parent, int(call.name))
        previous_path.write_text(json.dumps(value), encoding="utf-8")
        print(json.dumps(value), flush=True)
        break
    time.sleep(1)
