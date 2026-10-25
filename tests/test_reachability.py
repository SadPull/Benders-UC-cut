from types import SimpleNamespace as NS

import numpy as np

from suc_benders import search
from suc_benders.validation import pack, reachable_capacity, reachable_shortfall, repair
from conftest import make_whole


def units(n, **values):
    base = dict(pmax=100., pstart=100., pshut=100., ramp_up=30., ramp_down=30., initial_output=0., initial_on=0)
    base.update(values)
    return NS(n_units=n, horizon=None, **{k: np.full(n, float(v)) if k != 'initial_on' else np.full(n, int(v))
                                            for k, v in base.items()})


def test_reachable_capacity_chains_startup_and_shutdown_ramps():
    d = units(1); d.horizon = 6
    # 首小时受 ramp_up=30 限制（pstart=pmax 也不能更高），停机前受 ramp_down=30 的后向链限制。
    assert reachable_capacity(d, np.array([[0, 1, 1, 1, 1, 0]])).tolist() == [[0., 30., 60., 60., 30., 0.]]
    d = units(1, initial_on=1, initial_output=50.); d.horizon = 3
    # 初始开机从 initial_output 前向爬坡；时域末不受停机约束。
    assert reachable_capacity(d, np.array([[1, 1, 1]])).tolist() == [[80., 100., 100.]]
    # IEEE118 机组 11 的情形：初始出力 100、爬坡 175、pmax 350，第 2 小时停机；第 1 小时上界 min(275, 175)=175。
    d = units(1, pmax=350., pstart=350., pshut=350., ramp_up=175., ramp_down=175., initial_on=1, initial_output=100.)
    d.horizon = 4
    assert reachable_capacity(d, np.array([[1, 0, 0, 0]]))[0, 0] == 175.


def test_shortfall_uses_the_backward_shutdown_chain():
    d = units(1); d.horizon = 4
    scenarios = [NS(load=np.array([0., 50., 50., 0.]))]
    # 前向可达 30、60；后向：第 3 小时之后关机，第 3 小时 <= 30，故第 2 小时 <= 60、第 3 小时缺 20。
    assert reachable_shortfall(d, scenarios, np.array([[0, 1, 1, 0]])) == [
        dict(hour=2, reachable_mw=30., load_mw=50., shortfall_mw=20.),
        dict(hour=3, reachable_mw=30., load_mw=50., shortfall_mw=20.)]


def test_repair_adds_units_until_ramp_limited_output_reaches_load(tiny):
    d, scenarios = tiny(2, 4)
    d.pstart[:] = d.pmax; d.ramp_up[:] = 20.
    # capacity() 认为单台 55 MW 足够 30 MW 负荷，但首小时只能爬到 20 MW。
    raw = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    x = repair(d, scenarios, raw)
    u = x[:8].reshape((2, 4), order='F')
    assert u[1, 0] == 1 and not reachable_shortfall(d, scenarios, u)
    scenarios[0].load[0] = 45.
    assert repair(d, scenarios, raw) is None


def test_prepare_skips_dispatch_for_reachable_shortfall(tiny, options, monkeypatch):
    d, scenarios = tiny(2, 4)
    d.pstart[:] = d.pmax; d.ramp_up[:] = 20.
    whole = make_whole(d, scenarios, options)
    dispatched = []
    original = search.dispatch
    monkeypatch.setattr(search, 'dispatch', lambda *a, **k: dispatched.append(True) or original(*a, **k))
    monkeypatch.setattr(search, 'repair', lambda d, s, u: pack(d, np.asarray(u)))
    record = {}
    raw = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    assert whole.prepare(raw, 'sa', record) is None
    assert record['outcome'] == 'reachable_infeasible' and not dispatched
