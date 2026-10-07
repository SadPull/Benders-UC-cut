from time import perf_counter
from types import SimpleNamespace as NS

import numpy as np

from suc_benders import solver
from suc_benders.model import model, partition
from suc_benders.validation import pack


def test_late_dispatch_not_accepted(tiny, options, monkeypatch):
    d, s = tiny(1); b = partition(model(d, s), 3)[0][0]
    now = [0.]
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])

    def late(*args):
        now[0] = 11.
        return NS(feasible=True, cost=1.)

    monkeypatch.setattr(solver, "dispatch", late)
    result = solver.search_block(b, [pack(d, np.ones((1, 3)))], options, 10, 0, 1, 1, np.random.default_rng(1))
    assert result.best is None and not result.planes and result.upper == np.inf


def test_late_certification_not_accepted(tiny, options, monkeypatch):
    d, s = tiny(1); b = partition(model(d, s), 3)[0][0]
    now = [0.]
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])

    def late(*args):
        now[0] = 11.
        return None, 225.

    monkeypatch.setattr(solver, "master", late)
    initial_bound = float(np.minimum(b.obj * b.lb, b.obj * b.ub).sum() + b.constant)
    result = solver.search_block(b, [pack(d, np.ones((1, 3)))], options, 10, 0, 1, 1, np.random.default_rng(1))
    assert result.lower == initial_bound and result.best is not None


def test_llm_failure_continues_sa_and_respects_call_limit(tiny, options, monkeypatch):
    d, s = tiny(1); b = partition(model(d, s), 3)[0][0]
    options.llm_enabled = True; options.llm_patience = 1; options.llm_max_calls = 1; options.max_inner = 3
    original = solver.master
    monkeypatch.setattr(solver, "master", lambda b, cuts, seconds: (original(b, cuts)[0], 0.))
    samples = []

    def sample(*args):
        samples.append(1)
        return [], {"bits": 1, "candidates": 0, "seconds": 0, "best_energy": 0}

    monkeypatch.setattr(solver, "anneal", sample)
    monkeypatch.setattr(solver.llm, "request", lambda *args: ([], {"status": "FAILED"}))
    result = solver.search_block(b, [pack(d, np.ones((1, 3)))], options, perf_counter() + 30, 0, 1, 1, np.random.default_rng(1))
    assert len(samples) == 3 and result.llm_calls == 1 and len(result.history) == 3
    assert result.upper == 225


def test_late_sa_candidates_not_dispatched(tiny, options, monkeypatch):
    d, s = tiny(1); b = partition(model(d, s), 3)[0][0]
    now = [0.]; calls = []
    monkeypatch.setattr(solver, "perf_counter", lambda: now[0])
    original_master, original_dispatch = solver.master, solver.dispatch
    monkeypatch.setattr(solver, "master", lambda b, cuts, seconds: (original_master(b, cuts)[0], 0.))

    def dispatch(*args):
        calls.append(1)
        return original_dispatch(*args)

    def late(*args):
        now[0] = 11.
        return [np.zeros(len(b.binary))], {"bits": 1}

    monkeypatch.setattr(solver, "dispatch", dispatch)
    monkeypatch.setattr(solver, "anneal", late)
    result = solver.search_block(b, [pack(d, np.ones((1, 3)))], options, 10, 0, 1, 1, np.random.default_rng(1))
    assert len(calls) == 1 and result.upper == 225 and not result.history
