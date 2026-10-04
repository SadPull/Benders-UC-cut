from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


class DataError(ValueError):
    """Raised when UC or scenario input is internally inconsistent."""


@dataclass(frozen=True)
class UCData:
    n_units: int
    horizon: int
    pmax: np.ndarray
    pmin: np.ndarray
    pstart: np.ndarray
    pshut: np.ndarray
    min_up: np.ndarray
    min_down: np.ndarray
    ramp_up: np.ndarray
    ramp_down: np.ndarray
    initial_on: np.ndarray
    initial_duration: np.ndarray
    initial_output: np.ndarray
    base_load: np.ndarray
    cost_a: np.ndarray
    cost_b: np.ndarray
    cost_c: np.ndarray
    startup_cost: np.ndarray
    shutdown_cost: np.ndarray
    reserve_fraction: float = 0.05


@dataclass(frozen=True)
class Scenario:
    name: str
    probability: float
    load: np.ndarray

def _vector(obj: dict, key: str, n: int) -> np.ndarray:
    value = obj.get(key)
    if not isinstance(value, list) or len(value) != n:
        raise DataError(f"{key} must be a list of length {n}")
    values = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(values)):
        raise DataError(f"{key} contains a non-finite value")
    return values


def _load_units(data_path: Path) -> UCData:
    try:
        obj = json.loads(data_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read UC data file {data_path}: {exc}") from exc

    try:
        n = int(obj["n_units"])
        horizon = int(obj["horizon"])
        base_load = np.asarray(obj["base_load"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError("n_units, horizon, and base_load are required") from exc
    if n <= 0 or horizon <= 0:
        raise DataError("n_units and horizon must be positive")
    if base_load.shape != (horizon,):
        raise DataError(f"base_load must have {horizon} entries")

    pmax = _vector(obj, "pmax", n)
    pmin = _vector(obj, "pmin", n)
    pstart = _vector(obj, "pstart", n)
    pshut = _vector(obj, "pshut", n)
    ramp_up = _vector(obj, "ramp_up", n)
    ramp_down = _vector(obj, "ramp_down", n)
    cost_a = _vector(obj, "cost_a", n)
    cost_b = _vector(obj, "cost_b", n)
    cost_c = _vector(obj, "cost_c", n)
    startup_cost = _vector(obj, "startup_cost", n)
    shutdown_cost = _vector(obj, "shutdown_cost", n)
    min_up = np.asarray(obj["min_up"], dtype=int)
    min_down = np.asarray(obj["min_down"], dtype=int)
    initial_duration = _vector(obj, "initial_duration", n)

    if np.any(pmin < 0) or np.any(pmax < pmin):
        raise DataError("each Pmin must be non-negative and no greater than Pmax")
    if np.any(base_load < 0):
        raise DataError("base_load must be non-negative")
    if np.any(ramp_up < 0) or np.any(ramp_down < 0):
        raise DataError("ramp rates must be non-negative")
    if np.any(min_up < 1) or np.any(min_down < 1):
        raise DataError("min_up and min_down must be positive")
    if np.any(cost_a < 0) or np.any(cost_b < 0) or np.any(cost_c < 0):
        raise DataError("cost coefficients must be non-negative")

    initial_on = initial_duration > 0
    if "initial_output" in obj:
        initial_output = _vector(obj, "initial_output", n)
        if np.any(initial_output[~initial_on] != 0):
            raise DataError("offline units must have zero initial_output")
        if np.any(initial_output[initial_on] < pmin[initial_on]) or np.any(initial_output[initial_on] > pmax[initial_on]):
            raise DataError("online initial_output must lie between Pmin and Pmax")
    else:
        initial_output = np.where(initial_on, pmin, 0.0)
    return UCData(
        n_units=n,
        horizon=horizon,
        pmax=pmax,
        pmin=pmin,
        pstart=pstart,
        pshut=pshut,
        min_up=min_up,
        min_down=min_down,
        ramp_up=ramp_up,
        ramp_down=ramp_down,
        initial_on=initial_on,
        initial_duration=np.abs(initial_duration),
        initial_output=initial_output,
        base_load=base_load,
        cost_a=cost_a,
        cost_b=cost_b,
        cost_c=cost_c,
        startup_cost=startup_cost,
        shutdown_cost=shutdown_cost,
        reserve_fraction=float(obj.get("reserve_fraction", 0.05)),
    )


def _read_scenarios(raw: object, base_load: np.ndarray, horizon: int) -> tuple[list[Scenario], float]:
    if not isinstance(raw, dict):
        raise DataError("scenario JSON root must be an object")
    reserve_fraction = float(raw.get("reserve_fraction", 0.05))
    if reserve_fraction < 0:
        raise DataError("reserve_fraction must be non-negative")
    entries = raw.get("scenarios")
    if not isinstance(entries, list) or not entries:
        raise DataError("scenarios must be a non-empty list")

    scenarios: list[Scenario] = []
    names: set[str] = set()
    probabilities: list[float] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise DataError(f"scenario {index} must be an object")
        name = entry.get("name", f"scenario_{index}")
        if not isinstance(name, str) or not name or name in names:
            raise DataError(f"scenario name {name!r} is missing or duplicated")
        try:
            probability = float(entry["probability"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataError(f"probability for scenario {name!r} is required") from exc
        if not np.isfinite(probability) or probability <= 0:
            raise DataError(f"probability for scenario {name!r} must be positive")
        has_factor = "load_factor" in entry
        has_load = "load" in entry
        if has_factor == has_load:
            raise DataError(f"scenario {name!r} must specify exactly one of load_factor or load")
        if has_factor:
            factor = float(entry["load_factor"])
            if factor < 0:
                raise DataError(f"load_factor for scenario {name!r} must be non-negative")
            load = factor * base_load
        else:
            load = np.asarray(entry["load"], dtype=float)
            if load.shape != (horizon,):
                raise DataError(f"load for scenario {name!r} must have {horizon} entries")
            if np.any(load < 0) or not np.all(np.isfinite(load)):
                raise DataError(f"load for scenario {name!r} must be finite and non-negative")
        names.add(name)
        probabilities.append(probability)
        scenarios.append(Scenario(name, probability, load))
    total = float(sum(probabilities))
    if not np.isclose(total, 1.0, atol=1e-9, rtol=0.0):
        raise DataError(f"scenario probabilities must sum to 1; got {total:.12g}")
    return scenarios, reserve_fraction


def load_suc_data(
    data_path: str | Path = "data/rts24/units.json",
    scenario_path: str | Path | None = None,
) -> tuple[UCData, list[Scenario]]:
    """Load validated UC data and stochastic scenarios.

    A load_factor is relative to the canonical base load, while an explicit
    ``load`` array is an absolute 24-period scenario.
    """

    data_path = Path(data_path)
    units = _load_units(data_path)
    if scenario_path is None:
        entries = [
            {"name": "low", "probability": 0.25, "load_factor": 0.9},
            {"name": "base", "probability": 0.50, "load_factor": 1.0},
            {"name": "high", "probability": 0.25, "load_factor": 1.05},
        ]
        raw = {"reserve_fraction": 0.05, "scenarios": entries}
    else:
        try:
            raw = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"cannot read scenario file {scenario_path}: {exc}") from exc
    scenarios, reserve_fraction = _read_scenarios(raw, units.base_load, units.horizon)
    available = float(np.sum(units.pmax))
    required = max(float(np.max((1.0 + reserve_fraction) * scenario.load)) for scenario in scenarios)
    if required > available + 1e-6:
        raise DataError(
            f"scenario load plus reserve exceeds total Pmax: required={required:.6g}, available={available:.6g}"
        )
    units = UCData(**{**units.__dict__, "reserve_fraction": reserve_fraction})
    return units, scenarios

