"""Read-only process/deadline observation; never launches or acknowledges models."""
import json
import os
import time
import batch


def main():
    maximum_gap = 0.
    previous = time.time()
    with (batch.ROOT / "monitor.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        while True:
            now = time.time()
            maximum_gap = max(maximum_gap, now-previous)
            previous = now
            try:
                state = batch.status()
                row = dict(observed_at=now, **state)
            except (PermissionError, json.JSONDecodeError) as error:
                row = dict(observed_at=now, read_error=type(error).__name__)
            log.write(json.dumps(row, ensure_ascii=False)+"\n")
            batch.write(batch.ROOT / "monitor-state.json", dict(pid=os.getpid(), last_observed_at=now,
                max_sampling_gap_seconds=maximum_gap, stopped=row.get("status") in {"completed","failed"}))
            if row.get("status") in {"completed","failed"}:
                return
            time.sleep(2)


if __name__ == "__main__":
    main()
