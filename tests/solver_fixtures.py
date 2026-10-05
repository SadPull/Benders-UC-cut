"""Keep scheduler/feedback tests independent of algorithmic convergence."""
from unittest.mock import patch

from suc_benders.block_master_model import continuous_relaxation


def all_on_relaxation(*args):
    result = continuous_relaxation(*args)
    # Only control the rounded initialization in advisor attribution tests.
    result["point"] = {k: 1. if k[0] == "u" else v for k,v in result["point"].items()}
    return result


def keep_inner_running(test, *, all_on=True):
    targets = [("suc_benders.paper_solver.converged", dict(return_value=False))]
    if all_on:
        targets.append(("suc_benders.paper_solver.continuous_relaxation", dict(side_effect=all_on_relaxation)))
    for target, kwargs in targets:
        mock = patch(target, **kwargs)
        mock.start()
        test.addCleanup(mock.stop)
