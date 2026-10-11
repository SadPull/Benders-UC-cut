"""Fingerprint the Python environment actually used by the current C experiment."""
import importlib.metadata
import sys

import gurobipy as gp

from . import GUROBI_VERSION


def environment():
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'scipy', 'gurobipy', 'kaiwu', 'psutil', 'pytest')}
    native = '.'.join(map(str, gp.gurobi.version()))
    if native != GUROBI_VERSION or versions['gurobipy'] != GUROBI_VERSION:
        raise ValueError('Run paper C in its Gurobi 12.0.2 environment')
    return dict(python=sys.version, executable=sys.executable, packages=versions, gurobi_native=native)
