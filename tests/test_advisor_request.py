import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import perf_counter, sleep
from types import SimpleNamespace as NS

import pytest

from suc_benders import advisor_request


@pytest.fixture
def local_responses():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if "http_status" in received[-1]:
                self.send_error(received[-1]["http_status"], "offline-secret")
                return
            body = {"id": "offline-response", "model": "offline", "status": "completed", "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(
                    {"candidates": [{"u": [[1, 1, 1]]}]})}]}]}
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/responses", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_process_returns_receipt_and_excludes_credentials(tiny, options, local_responses):
    d, _ = tiny(1)
    options.llm_url = local_responses[0]
    options.llm_api_key = "offline-secret"
    task = advisor_request.AdvisorRequest(d, options, {"model": "offline", "input": "probe"}, 1, 8.)
    try:
        reply = None
        deadline = perf_counter()+8
        while reply is None and perf_counter() < deadline:
            reply = task.poll()
            if reply is None:
                sleep(.01)
        assert reply is not None
        candidates, info = reply
        assert info["status"] == "OK" and info["response_id"] == "offline-response"
        assert len(candidates) == 1 and local_responses[1]
        assert "offline-secret" not in json.dumps(info)
        assert info["request_sha256"] and not task.process.is_alive()
    finally:
        if task.process.is_alive():
            task.close()


def test_request_deadline_terminates_worker(tiny, options, local_responses):
    d, _ = tiny(1)
    options.llm_url, options.llm_api_key = local_responses[0], "offline-secret"
    task = advisor_request.AdvisorRequest(d, options, {}, 1, .00001)
    candidates, info = task.poll()
    assert not candidates and info["status"] == "TIME_LIMIT"
    assert not task.process.is_alive()


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_failure_receipt_omits_secrets_and_does_not_retry(tiny, local_responses, status):
    d, _ = tiny(1)
    replies, closed = [], []
    connection = NS(send=replies.append, close=lambda: closed.append(True))
    advisor_request._request(connection, local_responses[0], "offline-secret",
                           {"http_status": status}, d, 1, 5.)
    candidates, info = replies[0]
    assert not candidates and info["status"] == "FAILED"
    assert info["message"] == "HTTPError" and info["http_status"] == status
    assert info["seconds"] >= 0 and info["finished_at"] > 0
    assert len(local_responses[1]) == 1 and closed == [True]
    assert "offline-secret" not in json.dumps(info)
