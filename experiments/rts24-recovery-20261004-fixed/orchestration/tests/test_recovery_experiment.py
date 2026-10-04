import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from suc_benders.recovery_experiment import call_open, cli_command, completed_agent, agent_events, worker_environment


class RecoveryExperimentTests(unittest.TestCase):
    def test_candidate_file_alone_never_authorizes_a_receipt(self):
        started = {"type": "thread.started", "thread_id": "actual-agent-id"}
        ended = {"type": "turn.completed", "usage": {"output_tokens": 10}}
        self.assertEqual(completed_agent([started], 0, True), ("actual-agent-id", False))
        self.assertEqual(completed_agent([started, ended], 0, True), ("actual-agent-id", True))
        self.assertTrue(completed_agent([started, {"type": "error", "message": "Reconnecting..."}, ended], 0, True)[1])
        for events, code, exists in (([ended], 0, True), ([started, ended], 1, True),
                                     ([started, ended], 0, False),
                                     ([started, ended, {"type": "turn.failed"}], 0, True)):
            self.assertFalse(completed_agent(events, code, exists)[1])

    def test_closed_or_late_calls_cannot_be_delivered(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            call = run / "calls/001"
            call.mkdir(parents=True)
            self.assertTrue(call_open(run, call, time.perf_counter()+60))
            self.assertFalse(call_open(run, call, time.perf_counter()-1))
            for path in (run / "cancel.json", call / "consumed.json", call / "receipt.json"):
                path.write_text("{}", encoding="utf-8")
                self.assertFalse(call_open(run, call, time.perf_counter()+60))
                path.unlink()

    def test_event_log_keeps_actual_identity_and_ignores_partial_last_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_text(json.dumps({"type": "thread.started", "thread_id": "real"})+
                            '\n{"type":"turn.com', encoding="utf-8")
            self.assertEqual(agent_events(path), [{"type": "thread.started", "thread_id": "real"}])

    def test_cli_preserves_model_effort_and_fresh_sessions_without_other_agents(self):
        command = cli_command(Path("codex.exe"), Path("call"))
        self.assertIn("--ephemeral", command)
        self.assertNotIn("--ignore-user-config", command)
        self.assertEqual(command[command.index("--model")+1], "gpt-6-astra")
        self.assertIn('model_reasoning_effort="max"', command)
        self.assertIn('model_provider="openai"', command)
        self.assertIn("multi_agent", command)
        self.assertNotIn("resume", command)
        self.assertNotIn("--max-tokens", command)

    def test_windows_proxy_is_exported_without_overriding_explicit_proxy(self):
        with patch.dict('os.environ', {'HTTPS_PROXY':'http://explicit:1234'}, clear=True), patch(
                'urllib.request.getproxies', return_value={'http':'http://system:7890','https':'http://system:7890'}):
            env = worker_environment()
        self.assertEqual(env['HTTP_PROXY'], 'http://system:7890')
        self.assertEqual(env['HTTPS_PROXY'], 'http://explicit:1234')


if __name__ == "__main__":
    unittest.main()
