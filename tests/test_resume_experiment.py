import json

import pytest

import resume_two_arm_experiment as resume


def test_resume_clock_excludes_pause_and_counts_live_work(monkeypatch):
    now = [100.]
    monkeypatch.setattr(resume, "perf_counter", lambda: now[0])
    clock = resume.ScoreClock()
    clock.elapsed = 1600.
    now[0] = 10000.
    assert clock() == 1700.
    clock.resume(1643.)
    now[0] += 10.
    assert clock() == 1753.
    with pytest.raises(ValueError):
        clock.resume(1600.)


def test_archive_comparison_checks_schedule_and_context():
    archived = dict(incumbent_u=[[1, 0]], context=dict(cost=887405.248, feedback="TimeoutError"))
    rebuilt = json.loads(json.dumps(archived))
    rebuilt["context"]["cost"] += 1e-6
    assert resume.same(archived, rebuilt)
    rebuilt["incumbent_u"][0][1] = 1
    assert not resume.same(archived, rebuilt)
    rebuilt["incumbent_u"][0][1] = 0
    rebuilt["context"]["feedback"] = "OK"
    assert not resume.same(archived, rebuilt)
