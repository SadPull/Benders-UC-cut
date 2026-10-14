"""One cancellable Responses request; optimization stays in the parent process."""
import hashlib
import json
import multiprocessing as mp
from time import perf_counter
from types import SimpleNamespace as NS

from . import llm


def request_body(d, scenarios, best, o, context, iteration):
    body = llm.request_body(d, scenarios, best, o, {"bound_search": context}, [], iteration)
    body["instructions"] = (
        "You advise a certified integer Benders master for stochastic unit commitment. "
        "Propose binary u schedules to probe underestimated dispatch costs or missing "
        "dispatch feasibility restrictions, not only to improve the incumbent. "
        "Use the current master schedule, its predicted and evaluated costs, and cut feedback. "
        "Every probe is evaluated numerically; never invent costs, bounds, dispatch or cut coefficients. "
        "Respect commitment transitions, initial conditions, minimum up/down times and terminal rules. "
        "Dispatch-infeasible probes can be useful and will generate feasibility cuts. "
        "No heuristic repair is applied. Return JSON with at most four candidates, each with "
        "name, reason and an exact units-by-horizon 0/1 u matrix."
    )
    prompt = json.loads(body["input"])
    if o.rts24_improvements:
        body['instructions'] = body['instructions'].replace('at most four candidates', 'at most two candidates')
    prompt["task"] = "Suggest diverse probes that strengthen the certified master lower bound."
    prompt["terminal_rule"] = ("Shutdown in the last hour is allowed; minimum times truncate at the horizon."
        if getattr(d, "model_profile", "") == "azema_sms" else
        "Shutdown in the last hour is forbidden; minimum times truncate at the horizon.")
    prompt["rules"] = ["Return exact binary u matrices and JSON only", "Do not repeat previous probes",
                       "Prioritize underestimated master solutions and informative nearby alternatives",
                       "Respect discrete commitment rules; dispatch feasibility is checked by the solver"]
    body["input"] = json.dumps(prompt, separators=(",", ":"), allow_nan=False)
    return body


def _request(connection, url, key, body, shape, iteration, seconds):
    started = perf_counter()
    candidates = []
    info = dict(status="FAILED", message="", accepted=0, format_valid=0, proposals=[])
    try:
        raw = llm.post_response(url, key, body, seconds)
        candidates, info = llm.parse_response(raw, shape, iteration)
        info.update(response_id=raw.get("id"), response_model=raw.get("model"), usage=raw.get("usage"),
                    response_output=[part["text"] for item in raw.get("output", [])
                                     if item.get("type") == "message" for part in item.get("content", [])
                                     if part.get("type") == "output_text"])
    except Exception as error:
        info["message"] = type(error).__name__
        if isinstance(getattr(error, "code", None), int):
            info["http_status"] = error.code
    finished = perf_counter()
    info.update(seconds=finished-started, finished_at=finished)
    try:
        connection.send((candidates, info))
    finally:
        connection.close()


class BoundAdvisor:
    def __init__(self, d, o, body, iteration, seconds):
        self.agent = None
        if o.llm_agent_directory:
            from .subagent_advisor import SubagentAdvisor
            self.agent = SubagentAdvisor(d, o, body, iteration, seconds)
            self.started, self.deadline, self.receipt = self.agent.started, self.agent.deadline, self.agent.receipt
            return
        self.started = perf_counter()
        self.deadline = self.started + seconds
        context = mp.get_context("spawn")
        self.connection, sender = context.Pipe(duplex=False)
        self.process = context.Process(target=_request,
            args=(sender, o.llm_url, o.llm_api_key, body,
                  NS(n_units=d.n_units, horizon=d.horizon), iteration, seconds), daemon=True)
        self.receipt = dict(request=body, request_sha256=hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        try:
            self.process.start()
        except Exception:
            self.connection.close()
            raise
        finally:
            sender.close()

    def poll(self):
        if self.agent is not None:
            return self.agent.poll()
        now = perf_counter()
        if self.connection.poll():
            try:
                candidates, info = self.connection.recv()
            except EOFError:
                candidates, info = [], dict(status="FAILED", message="WorkerExited", accepted=0, proposals=[])
            if info.get("finished_at", now) > self.deadline:
                candidates = []
                info.update(status="TIME_LIMIT", accepted=0)
            info["background_seconds"] = max(0., info.get("finished_at", now)-self.started)
            info.update(self.receipt)
            self.close()
            return candidates, info
        if now >= self.deadline or not self.process.is_alive():
            info = dict(status="TIME_LIMIT" if now >= self.deadline else "FAILED",
                        message="RequestDeadline" if now >= self.deadline else "WorkerExited",
                        seconds=now-self.started, background_seconds=now-self.started,
                        accepted=0, proposals=[], **self.receipt)
            self.close()
            return [], info
        return None

    def close(self):
        if self.agent is not None:
            self.agent.close()
            return
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=1.)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.)
        self.connection.close()
