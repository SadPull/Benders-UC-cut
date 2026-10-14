"""Read the original JSON inputs without changing their physical units."""
# 模块职责：读取机组/场景 JSON，保留原物理单位，并在进入建模前检查输入边界。
# 机组参数为长度 n_units 的向量，负荷为长度 horizon 的向量；返回 d 和场景列表。
import json
from types import SimpleNamespace as NS

import numpy as np


# 加载并验证机组参数、初始状态、冷热费用、场景概率和负荷，返回 NumPy 化的数据对象。
# 初始时长的符号先决定开停机，再取绝对值；标准 UC 与 SMS 对零时长的开机口径不同。
def load_data(unit_file, scenario_file):
    with open(unit_file, encoding="utf-8") as f:
        raw = json.load(f)
    d = NS(**raw)
    profile = getattr(d, "model_profile", "")
    if profile not in ("", "azema_sms"):
        raise ValueError("Unknown model_profile")
    for key in ("n_units", "horizon"):
        value = getattr(d, key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    fields = ("pmax", "pmin", "pstart", "pshut", "min_up", "min_down", "ramp_up",
              "ramp_down", "initial_duration", "cost_a", "cost_b", "cost_c",
              "startup_cost", "shutdown_cost")
    cold_fields = ("cold_startup_cost", "cold_start_hours")
    if any(key in raw for key in cold_fields):
        if not all(key in raw for key in cold_fields):
            raise ValueError("Specify both cold_startup_cost and cold_start_hours")
        fields += cold_fields
    for key in fields:
        value = np.asarray(getattr(d, key), dtype=float)
        if value.shape != (d.n_units,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid unit vector: {key}")
        if key != "initial_duration" and (value < 0).any():
            raise ValueError(f"Negative values: {key}")
        setattr(d, key, value)
    if hasattr(d, "cold_startup_cost"):
        if (d.cold_startup_cost < d.startup_cost).any():
            raise ValueError("Cold startup cost must be at least the hot startup cost")
        if ((d.cold_start_hours != np.floor(d.cold_start_hours)).any()
                or any(isinstance(v, bool) for v in raw["cold_start_hours"])):
            raise ValueError("cold_start_hours must contain nonnegative integer hours")
    for key in ("min_up", "min_down", "initial_duration"):
        value = getattr(d, key)
        if (value != np.floor(value)).any() or (key != "initial_duration" and (value < 1).any()):
            raise ValueError(f"Invalid integer durations: {key}")
    if any((getattr(d, key) > d.pmax).any() for key in ("pmin", "pstart", "pshut")):
        raise ValueError("Output limits exceed Pmax")
    # 先读取时长符号中的初始开停机状态，再保存正的持续小时数，后续时序规则只使用持续长度。
    d.initial_on = ((d.initial_duration >= 0) if profile == "azema_sms" else (d.initial_duration > 0)).astype(float)
    d.initial_duration = np.abs(d.initial_duration)
    d.initial_output = np.asarray(getattr(d, "initial_output", d.pmin * d.initial_on), dtype=float)
    if (d.initial_output.shape != (d.n_units,) or not np.isfinite(d.initial_output).all()
            or (d.initial_output < d.pmin * d.initial_on).any()
            or (d.initial_output > d.pmax * d.initial_on).any()):
        raise ValueError("Initial output is inconsistent with commitment")
    d.base_load = _load(d.base_load, d.horizon)
    with open(scenario_file, encoding="utf-8") as f:
        raw = json.load(f)
    d.reserve_fraction = float(raw["reserve_fraction"])
    if not np.isfinite(d.reserve_fraction) or d.reserve_fraction < 0:
        raise ValueError("Invalid reserve fraction")
    if profile == "azema_sms":
        if d.reserve_fraction != 0 or d.cost_a.any() or hasattr(d, "cold_startup_cost"):
            raise ValueError("SMS requires zero reserve, linear fuel and fixed startup costs")
        if (d.ramp_up <= 0).any() or (d.ramp_down <= 0).any():
            raise ValueError("SMS ramp limits must be positive")
        for key in ("shedding_cost", "curtailment_cost"):
            value = getattr(d, key, None)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 700:
                raise ValueError("Author SMS imbalance penalties must equal 700")
    scenarios = []
    # 概率只用于费用加权；每个场景的负荷和备用约束都必须分别满足。
    for item in raw["scenarios"]:
        name, probability = item["name"], float(item["probability"])
        if not isinstance(name, str) or not name or not np.isfinite(probability) or probability <= 0:
            raise ValueError("Invalid scenario name or probability")
        has_factor = item.get("load_factor") is not None
        has_load = item.get("load") is not None
        if has_factor == has_load:
            raise ValueError("Specify exactly one of load and load_factor")
        if has_factor:
            factor = float(item["load_factor"])
            if not np.isfinite(factor) or factor < 0:
                raise ValueError("Invalid load factor")
            demand = d.base_load * factor
        else:
            demand = item["load"]
        scenarios.append(NS(name=name, probability=probability, load=_load(demand, d.horizon)))
    if (not scenarios or len({s.name for s in scenarios}) != len(scenarios)
            or abs(sum(s.probability for s in scenarios) - 1) > 1e-9):
        raise ValueError("Scenario names must be unique and probabilities must sum to one")
    if profile != "azema_sms" and max(s.load.max() for s in scenarios) * (1 + d.reserve_fraction) > d.pmax.sum() + 1e-6:
        raise ValueError("Load plus reserve exceeds installed capacity")
    return d, scenarios


# 将单个小时负荷序列转为浮点向量，要求长度正确、数值有限且非负。
def _load(value, horizon):
    value = np.asarray(value, dtype=float)
    if value.shape != (horizon,) or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("Invalid hourly load")
    return value
