from types import SimpleNamespace as NS

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
    return Options(max_outer=10, max_inner=5, time_limit=30, verbose=False,
                   sa_reads=4, sa_sweeps=2, sa_temperature=10, sa_cooling=.8, sa_cutoff=.01)
