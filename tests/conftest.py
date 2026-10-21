from types import SimpleNamespace as NS

from time import perf_counter
from suc_benders import solver, search as search_core
from suc_benders.model import model, partition
from suc_benders.validation import pack

import numpy as np
import pytest

from suc_benders.config import Options


@pytest.fixture
def tiny():
    def make(n=2, T=3):
        def units(value):
            return np.full(n, value, dtype=float)
        d = NS(n_units=n, horizon=T, pmax=units(55), pmin=units(10), pstart=units(45),
               pshut=units(35), min_up=units(1), min_down=units(1), ramp_up=units(135),
               ramp_down=units(135), initial_on=units(0), initial_duration=units(1),
               initial_output=units(0), base_load=np.full(T, 30.), cost_a=units(.01),
               cost_b=units(2), cost_c=units(5), startup_cost=units(3), shutdown_cost=units(4), reserve_fraction=0.)
        return d, [NS(name="tiny", probability=1., load=d.base_load.copy())]
    return make


@pytest.fixture
def options():
    return Options(max_rounds=50, lagrangian_max_iterations=5, lagrangian_time_limit=10, time_limit=30, verbose=False,
                   sa_reads=4, sa_sweeps=2, sa_temperature=10, sa_cooling=.8, sa_cutoff=.01)


# 为手写算例建立 WholeDay，并以 u（默认全开）作为已校验的初始 incumbent。
def make_whole(d, scenarios, options, u=None):
    m = model(d, scenarios)
    full = partition(m, d.horizon)[0][0]
    started = perf_counter()
    whole = search_core.WholeDay(m, full, options, started+30, solver.Result(), started)
    assert whole.accept(pack(d, np.ones((d.n_units, d.horizon)) if u is None else u), "initial")
    return whole


def whole_day(tiny, options, n=2):
    return make_whole(*tiny(n), options)
