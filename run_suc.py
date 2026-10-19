"""Python interface used by the experiment entry point."""
import numpy as np

from suc_benders.anneal import initialize
from suc_benders.config import load_options
from suc_benders.data import load_data
from suc_benders.solver import solve


def run_suc(options=None, *, resume_state=None, progress=None, started_at=None):
    o = load_options(options)
    d, scenarios = load_data(o.data_file, o.scenario_file)
    state = np.random.get_state()
    try:
        if o.sa_enabled:
            initialize(o.kaiwu_user_id, o.kaiwu_sdk_code)
        return solve(d, scenarios, o, resume_state=resume_state, progress=progress, started_at=started_at)
    finally:
        np.random.set_state(state)


def _serializable(value):
    if isinstance(value, np.ndarray):
        return _serializable(value.tolist())
    if isinstance(value, np.generic):
        return _serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if hasattr(value, "__dict__"):
        return _serializable(vars(value))
    return value
