"""Offline audit of Astra exchanges and measured candidate feedback."""
import argparse
import json
from pathlib import Path

from .subagent import sha, publish


def audit(root):
    root = Path(root)
    events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = []
    previous = None
    for call in sorted((root / "calls").glob("*")):
        request = json.loads((call / "request.json").read_text(encoding="utf-8"))
        receipt = json.loads((call / "receipt.json").read_text(encoding="utf-8"))
        consumed = json.loads((call / "consumed.json").read_text(encoding="utf-8"))
        feedback = json.loads((call / "feedback.json").read_text(encoding="utf-8"))
        user = json.loads((call / "user.txt").read_text(encoding="utf-8"))
        assert request["call_id"] == receipt["call_id"] == consumed["call_id"] == feedback["call_id"]
        assert request["model"] == receipt["model"] == "gpt-6-astra"
        assert request["reasoning_effort"] == receipt["reasoning_effort"] == "max"
        assert receipt["agent_id"] and consumed["http_requests"] == 0
        assert all(sha(call / name) == value for name, value in request["input_sha256"].items())
        assert receipt["input_sha256"] == request["input_sha256"]
        if receipt["status"] == "completed":
            assert sha(call / "candidates.json") == receipt["output_sha256"]
        if "previous_call_feedback" in user["context"]:
            assert user["context"]["previous_call_feedback"] == previous
        proposals = feedback["proposals"]
        rows.append(dict(call_id=request["call_id"], agent_id=receipt["agent_id"],
            wait_seconds=consumed["agent_wait_seconds"], proposed=len(proposals),
            completed=sum(p["validation_completed"] for p in proposals),
            feasible=sum(p["feasible"] is True for p in proposals),
            duplicates=sum(p["duplicate"] for p in proposals),
            improvements=sum(p["refreshed_incumbent"] for p in proposals),
            repair_bits=[p["repair_bits"] for p in proposals]))
        previous = feedback
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    summary = dict(audit_passed=True, calls=rows, objective=result["objective"], status=result["status"],
        sa_calls=sum(e["kind"] == "sa_start" for e in events),
        completed_rounds=sum(e["kind"] == "iteration" for e in events),
        continuous_dispatch_calls=sum(e["kind"] == "dispatch_start" for e in events),
        elapsed_seconds=result["runtime"], http_requests=0)
    publish(root / "audit.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
