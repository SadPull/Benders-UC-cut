import hashlib
import json

import numpy as np
import pytest

from suc_benders.benders import dispatch
from suc_benders.config import ROOT
from suc_benders.data import load_data
from suc_benders.model import model, partition
from suc_benders.optimization import optimize
from suc_benders.startup import startup_costs
from suc_benders.validation import pack, unpack, validate


def test_default_data_matches_pinned_source_and_hashes():
    folder = ROOT / "data"
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    source = provenance["sources"]["GA10.uc"]
    text = (folder / source["local_file"]).read_text(encoding="utf-8")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == source["local_sha256"] == source["sha256"]
    assert source["commit"] == "8175b0e9dc884e7af5fc1cb1802a9dbe5af38806"
    for name, digest in provenance["input_sha256"].items():
        normalized = (folder / name).read_text(encoding="utf-8").encode("utf-8")
        assert hashlib.sha256(normalized).hexdigest() == digest
    lines = text.split("<units>")[1].split("</units>")[0].strip().splitlines()
    rows = [dict(zip(lines[0].split(";"), line.split(";"))) for line in lines[1:]]
    d, scenarios = load_data(folder / "units.json", folder / "stochastic.json")
    for key, column in {"pmin": "pMin", "pmax": "pMax", "cost_a": "c", "cost_b": "b", "cost_c": "a",
                        "min_up": "MinUp", "min_down": "MinDown"}.items():
        np.testing.assert_array_equal(getattr(d, key), [float(row[column]) for row in rows])
    for key, index in (("startup_cost", 0), ("cold_startup_cost", 1)):
        np.testing.assert_array_equal(getattr(d, key), [float(row["SCV"].split(":")[index]) for row in rows])
    np.testing.assert_array_equal(d.cold_start_hours,
        [int(row["SCI"].split(":")[1])-int(row["MinDown"]) for row in rows])
    np.testing.assert_array_equal((2*d.initial_on-1)*d.initial_duration, [8, 8, -5, -5, -6, -3, -3, -1, -1, -1])
    np.testing.assert_array_equal(d.initial_output, d.initial_on*d.pmin)
    for key in ("ramp_up", "ramp_down", "pstart", "pshut"):
        np.testing.assert_array_equal(getattr(d, key), d.pmax)
    assert not d.shutdown_cost.any()
    demand = text.split("<demands>")[1].split("</demands>")[0].strip().splitlines()[1]
    np.testing.assert_array_equal(d.base_load, [float(v) for v in demand.split("[")[1].split("]")[0].split(":")])
    assert d.n_units == 10 and d.horizon == 24 and d.pmax.sum() == 1662 and d.base_load.max() == 1500
    assert d.reserve_fraction == .05
    for scenario, factor, probability in zip(scenarios, [.9, 1., 1.05], [.25, .5, .25]):
        np.testing.assert_array_equal(scenario.load, factor*d.base_load)
        assert scenario.probability == probability
    assert len(scenarios) == 3


@pytest.mark.parametrize("name,reserve,count", [("deterministic", .1, 1), ("stochastic", .05, 3)])
def test_classic_case_dispatch_and_relaxation(name, reserve, count):
    d, scenarios = load_data(ROOT / "data/units.json", ROOT / f"data/{name}.json")
    assert d.reserve_fraction == reserve and len(scenarios) == count
    m = model(d, scenarios)
    full = partition(m, d.horizon)[0][0]
    u = np.ones((10, 24))
    fixed = dispatch(full, pack(d, u), 30, certified=True)
    assert fixed.feasible
    result = unpack(m, fixed.point)
    assert validate(d, scenarios, result)["feasible"]
    # Every initially offline unit starts hot at hour 1, with no later starts.
    expected_fixed = 24*d.cost_c.sum()+d.startup_cost[2:].sum()
    fuel = sum(s.probability*r.objective for s, r in zip(scenarios, result.scenarios))
    assert result.objective == pytest.approx(expected_fixed+fuel, abs=1e-5)
    relaxation = optimize(m, 30)
    assert relaxation.status == "OPTIMAL"
    assert relaxation.lower <= relaxation.objval+1e-5 <= result.objective+1e-5


@pytest.mark.parametrize("key,value", [
    ("cold_startup_cost", []), ("cold_start_hours", [0]),
    ("cold_startup_cost", [float("nan")]*10), ("cold_start_hours", [float("inf")]*10),
    ("cold_startup_cost", [-1]*10), ("cold_start_hours", [-1]*10),
    ("cold_startup_cost", [0]*10), ("cold_start_hours", [.5]*10),
    ("cold_start_hours", [True]*10),
])
def test_invalid_cold_start_vectors(tmp_path, key, value):
    raw = json.loads((ROOT / "data/units.json").read_text(encoding="utf-8"))
    raw[key] = value
    units = tmp_path / "units.json"
    units.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_data(units, ROOT / "data/deterministic.json")


@pytest.mark.parametrize("missing", ["cold_startup_cost", "cold_start_hours"])
def test_cold_start_vectors_must_be_paired(tmp_path, missing):
    raw = json.loads((ROOT / "data/units.json").read_text(encoding="utf-8"))
    del raw[missing]
    units = tmp_path / "units.json"
    units.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="both"):
        load_data(units, ROOT / "data/deterministic.json")


@pytest.mark.parametrize("equal_cost", [False, True])
def test_fixed_startup_compatibility(tiny, equal_cost):
    d, scenarios = tiny(1)
    d.initial_duration[:] = 100
    if equal_cost:
        d.cold_startup_cost = d.startup_cost.copy()
        d.cold_start_hours = np.zeros(1)
    m = model(d, scenarios)
    assert not hasattr(m.ix, "cold_start")
    assert m.A.shape[1] == 5*d.n_units*d.horizon
    u = np.ones((1, 3))
    np.testing.assert_array_equal(startup_costs(d, u), [[3., 0., 0.]])
    r = dispatch(partition(m, 3)[0][0], pack(d, u), 5)
    assert r.feasible and r.cost == pytest.approx(225.)
    assert validate(d, scenarios, unpack(m, r.point))["feasible"]
