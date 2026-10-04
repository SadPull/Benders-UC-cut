import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from suc_benders.recovery_experiment import call_open, cli_command, completed_agent, agent_events, worker_environment, check_session_budget, serve_call


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
        with patch.dict('os.environ', {'HTTPS_PROXY':'http://explicit:1234', 'CODEX_API_KEY':'relay',
                                      'OPENAI_API_KEY':'other'}, clear=True), patch(
                'urllib.request.getproxies', return_value={'http':'http://system:7890','https':'http://system:7890'}):
            env = worker_environment()
        self.assertEqual(env['HTTP_PROXY'], 'http://system:7890')
        self.assertEqual(env['HTTPS_PROXY'], 'http://explicit:1234')
        self.assertNotIn('CODEX_API_KEY', env)
        self.assertNotIn('OPENAI_API_KEY', env)

    def test_prior_failed_and_successful_sessions_both_count_towards_authorized_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = dict(failed_startup_calls=4, interrupted_recovery_calls=2,
                            authorized_recovery_call_limit=16, max_llm_calls_batch=12)
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            for i in range(9):
                call = root / 'D-7/calls' / f'{i+1:03d}'
                call.mkdir(parents=True)
                (call / 'agent-launch.json').write_text('{}', encoding='utf-8')
            check_session_budget(root)
            (root / 'D-7/calls/010').mkdir()
            (root / 'D-7/calls/010/agent-launch.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'quota exhausted'):
                check_session_budget(root)
            manifest['authorized_recovery_call_limit'] = 18
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            check_session_budget(root)

    def test_unavailable_allowance_or_expired_preflight_never_launches_a_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'manifest.json').write_text(json.dumps(dict(max_llm_calls_batch=12)), encoding='utf-8')
            run = root / 'C-17'
            call = run / 'calls/001'
            call.mkdir(parents=True)
            with patch('suc_benders.recovery_experiment.claim') as claimed, patch(
                    'suc_benders.recovery_experiment.subprocess.Popen') as launched:
                with patch('suc_benders.recovery_experiment.model_availability', return_value={'available':False}):
                    with self.assertRaisesRegex(RuntimeError, 'allowance unavailable'):
                        serve_call(run, call, Path('codex.exe'), time.perf_counter()+60, {})
                with patch('suc_benders.recovery_experiment.model_availability', return_value={'available':True}):
                    response = serve_call(run, call, Path('codex.exe'), time.perf_counter()-1, {})
                    self.assertFalse(response['receipt_published'])
                claimed.assert_not_called()
                launched.assert_not_called()


if __name__ == "__main__":
    unittest.main()
