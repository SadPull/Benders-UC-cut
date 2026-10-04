"""Read-only, concise status for the independent recovery batch."""
import json
from pathlib import Path
import time

root = Path(__file__).resolve().parents[1]
def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

state = read(root / "state.json")
manifest = read(root / "manifest.json")
run = root / state.get("current", "C-17")
ready = read(run / "ready.json")
best = read(run / "best-checkpoint.json")
reference = read(Path(manifest["original_directory"]) / "reference/result.json").get("objective")
calls = []
for request in sorted((run / "calls").glob("*/request.json")):
    call = request.parent
    calls.append(dict(call=call.name, age_seconds=round(time.time()-read(request)["created_at"],1),
        consumed=read(call / "consumed.json").get("status"),
        completed=read(call / "agent-completion.json").get("actual_turn_completed"),
        output_exists=(call / "candidates.json").exists()))
completed = []
for group, seed in manifest["order"]:
    path = root / f"{group}-{seed}"
    row = read(path / "summary.json")
    if row:
        completed.append(dict(run=path.name, cost=row["objective"], received=row["llm_responses_consumed"],
            gap_percent=100*row["reference_difference"] if row["reference_difference"] is not None else None))
print(json.dumps(dict(state=state, remaining_seconds=round(ready.get("deadline_clock",time.perf_counter())-time.perf_counter(),1),
    best_cost=best.get("objective"), gap_percent=100*(best["objective"]/reference-1) if best and reference else None,
    progress=read(run / "progress.json"), calls=calls, completed=completed), ensure_ascii=False))
