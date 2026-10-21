"""Python interface used by the experiment entry point."""
from suc_benders.config import load_options
from suc_benders.data import load_data
from suc_benders.solver import solve


def run_suc(options=None, *, progress=None, started_at=None):
    o = load_options(options)
    d, scenarios = load_data(o.data_file, o.scenario_file)
    return solve(d, scenarios, o, progress=progress, started_at=started_at)
