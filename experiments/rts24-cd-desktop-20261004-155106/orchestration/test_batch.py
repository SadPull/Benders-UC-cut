import json
from pathlib import Path
import tempfile
import unittest

import batch
from audit_support import audit_run
from suc_benders.minimal_report import summarize


class BatchTests(unittest.TestCase):
    def test_six_expected_runs_not_fifteen(self):
        rows = [dict(group=g,seed=s,llm_started=3,llm_responses_consumed=3,
            audit=dict(complete=True,protocol_ok=True,errors=[],result_objective_consistent=True,physical_checked=1))
            for g,s in batch.ORDER]
        self.assertTrue(batch.flags(rows)["numerical_audit_passed"])
        self.assertFalse(batch.flags(rows[:-1])["complete"])
        rows[0]["llm_responses_consumed"] = 2
        self.assertTrue(batch.flags(rows)["numerical_audit_passed"])
        self.assertFalse(batch.flags(rows)["llm_response_replications_complete"])

    def test_failed_response_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "events.jsonl").write_text("",encoding="utf-8")
            call = run / "calls/001"
            call.mkdir(parents=True)
            request = dict(run_id="r",call_id="r:1",model="gpt-6-astra",reasoning_effort="max",fork_turns="none",input_sha256={},created_at=0)
            batch.write(call / "request.json",request)
            batch.write(call / "receipt.json",dict(request,status="failed",agent_id="a",completed_at=1))
            batch.write(call / "consumed.json",dict(status="completed"))
            row = audit_run(run,"C",7,None,None,100)
            self.assertEqual(row["llm_responses_consumed"],0)

    def test_late_candidate_excluded(self):
        events = [dict(kind="evaluation",elapsed_seconds=t,source="llm",objective=cost,
            feasible=True,improved=True,commitment_hash=str(t)) for t,cost in [(10,120),(1801,90)]]
        row = summarize(events,{},1800,100)
        self.assertEqual(row["objective"],120)
        self.assertEqual(row["late_evaluations_excluded"],1)
        self.assertIsNone(row["threshold_seconds"]["0.01"])


if __name__ == "__main__":
    unittest.main()
