import hashlib
import json

import numpy as np
import pytest

from suc_benders.benders import dispatch
from suc_benders.config import ROOT
from suc_benders.data import load_data
from suc_benders.model import model, partition
from suc_benders.optimization import optimize
from suc_benders.validation import startup_costs
from suc_benders.validation import pack, unpack, validate


CLASSIC = ROOT / "data/classic10"


def test_rcuc20_data_matches_original_parameters_and_hashes():
    folder = ROOT / "data"
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    source = provenance["sources"]["20_0_1_w.mod"]
    raw = (folder / source["local_file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source["sha256"] == source["local_sha256"]
    assert source["sha256"] == "412a5377dbfce0211acd550560246b7cc56ecb68f836a1abe7c69c1677131ec1"
    for name, digest in provenance["input_sha256"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    for name, digest in provenance["preserved_classic10"]["sha256"].items():
        assert hashlib.sha256((CLASSIC / name).read_bytes()).hexdigest() == digest
    text = raw.decode("utf-8")
    lines = text.split("ThermalSection")[1].split("HydroSection")[0].strip().splitlines()
    units = np.array([[float(value) for value in line.split()] for line in lines[::2]])
    ramps = np.array([[float(value) for value in line.split()[1:]] for line in lines[1::2]])
    assert units.shape == (20, 16) and ramps.shape == (20, 2)
    np.testing.assert_array_equal(units[:, 0], np.arange(20))
    d, scenarios = load_data(folder / "units.json", folder / "stochastic.json")
    for key, column in {"cost_a": 1, "cost_b": 2, "cost_c": 3, "pmin": 4, "pmax": 5,
                        "min_up": 7, "min_down": 8, "startup_cost": 13, "initial_output": 15}.items():
        np.testing.assert_array_equal(getattr(d, key), units[:, column])
    np.testing.assert_array_equal((2*d.initial_on-1)*d.initial_duration, units[:, 6])
    np.testing.assert_array_equal(d.ramp_up, ramps[:, 0])
    np.testing.assert_array_equal(d.ramp_down, ramps[:, 1])
    np.testing.assert_array_equal(d.pstart, d.pmax)
    np.testing.assert_array_equal(d.pshut, d.pmax)
    assert not units[:, 9:11].any() and not d.shutdown_cost.any()
    assert not hasattr(d, "cold_startup_cost")
    assert d.n_units == 20 and d.horizon == 24 and d.pmax.sum() == 3414
    assert d.base_load.max() == 2389.360312
    assert np.count_nonzero(d.initial_output) == 11
    assert not np.array_equal(d.initial_output, d.initial_on*d.pmin)
    load = [float(value) for value in text.split("Loads\t1\t24")[1].split("SpinningReserve")[0].split()]
    np.testing.assert_array_equal(d.base_load, load)
    for name in ("deterministic", "stochastic"):
        d, scenarios = load_data(folder / "units.json", folder / f"{name}.json")
        assert d.reserve_fraction == 0 and len(scenarios) == 1
        assert scenarios[0].name == "base" and scenarios[0].probability == 1
        np.testing.assert_array_equal(scenarios[0].load, d.base_load)


def test_classic_data_matches_pinned_source_and_hashes():
    folder = CLASSIC
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
    d, scenarios = load_data(CLASSIC / "units.json", CLASSIC / f"{name}.json")
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
    raw = json.loads((CLASSIC / "units.json").read_text(encoding="utf-8"))
    raw[key] = value
    units = tmp_path / "units.json"
    units.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_data(units, CLASSIC / "deterministic.json")


@pytest.mark.parametrize("missing", ["cold_startup_cost", "cold_start_hours"])
def test_cold_start_vectors_must_be_paired(tmp_path, missing):
    raw = json.loads((CLASSIC / "units.json").read_text(encoding="utf-8"))
    del raw[missing]
    units = tmp_path / "units.json"
    units.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="both"):
        load_data(units, CLASSIC / "deterministic.json")


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


@pytest.mark.parametrize('profile', ['azema_sms', 'unknown'])
def test_removed_model_profile_is_rejected(tmp_path, profile):
    raw = json.loads((ROOT/'data/units.json').read_text())
    raw['model_profile'] = profile
    path = tmp_path/'units.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='model_profile'):
        load_data(path, ROOT/'data/stochastic.json')
    d, scenarios = load_data(ROOT/'data/units.json', ROOT/'data/stochastic.json')
    d.model_profile = profile
    with pytest.raises(ValueError, match='model_profile'):
        model(d, scenarios)
