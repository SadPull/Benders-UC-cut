"""Whole-day advisor context restored from the D/17 frozen implementation."""
# 模块职责：把已验证方案、场景调度、费用与对偶敏感度整理成紧凑的全天 LLM 上下文。
# unit/period 使用零起始索引；梯度是局部启发式估计，三位小数仅用于提示展示，不参与认证。
# 调用方总是传入已验证的 incumbent 及其各场景调度（含平衡/备用对偶价格）。
from dataclasses import dataclass

import numpy as np

from .validation import capacity, fixed_cost, startup_costs


# 用于提示词的候选评估摘要：来源、实测总费用及其固定/期望燃料分解和可行性。
@dataclass
class CandidateEvaluation:
    source: str
    total_cost: float
    fixed_cost: float
    expected_fuel_cost: float
    feasible: bool


# 上下文构建输入：当前方案及场景调度、历史候选评估与迭代记录。
@dataclass
class LLMContextInput:
    incumbent: object
    incumbent_results: list
    candidate_evaluations: list
    history: list


# 递归转换 NumPy 标量并将有限浮点数保留三位小数，无穷/NaN 转为 None 以便生成合法 JSON。
def _round3(value):
    if isinstance(value, dict):
        return {key: _round3(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round3(item) for item in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return round(value, 3) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


# 按总费用选取至多十个候选对照，展示费用分解和可行性。
def _candidate_comparison_rows(evaluations):
    return [dict(source=e.source, total_cost=e.total_cost, fixed_cost=e.fixed_cost,
                 expected_fuel_cost=e.expected_fuel_cost, feasible=e.feasible)
            for e in sorted(evaluations, key=lambda item: item.total_cost)[:10]]


# 汇总各场景发电、备用与启停次数；marginal_unit_count 统计出力接近上下限的开机点。
def _scenario_dispatch_summary(data, scenarios, incumbent, results):
    rows = []
    for scenario, result in zip(scenarios, results):
        p, r = np.asarray(result.p, dtype=float), np.asarray(result.r, dtype=float)
        at_bound = (np.isclose(p, data.pmin[:, None], atol=1e-5) | np.isclose(p, data.pmax[:, None], atol=1e-5))
        rows.append(dict(scenario=scenario.name, total_generation=float(np.sum(p)), total_reserve=float(np.sum(r)),
                         startup_count=int(np.sum(incumbent.v)), shutdown_count=int(np.sum(incumbent.w)),
                         marginal_unit_count=int(np.sum(at_bound & (incumbent.u == 1)))))
    return rows


# 只保留最近五轮的上界改善、割增量、候选数和 LLM 失败摘要，控制提示词长度。
def _iteration_history_rows(history):
    rows, previous = [], None
    for record in list(history)[-5:]:
        incumbent = float(record.incumbent)
        rows.append(dict(iteration=int(record.iteration), incumbent=incumbent,
                         improvement=0.0 if previous is None else max(0.0, previous - incumbent),
                         optimality_cuts=int(record.optimality_cuts_added),
                         feasibility_cuts=int(record.feasibility_cuts_added),
                         new_candidates=int(record.new_candidates), llm_failures=1 if record.llm_failed else 0))
        previous = incumbent
    return rows


# 用燃料导数 2*a*P+b 估算各机组在 Pmin、开机小时概率加权平均出力和 Pmax 处的边际费用。
# 全天未开机的机组没有当前出力，at_current_p 为 None。
def _unit_marginal_cost(data, incumbent, expected_p):
    on_hours = incumbent.u.sum(axis=1)
    current_p = (expected_p * incumbent.u).sum(axis=1) / np.maximum(on_hours, 1)
    marginal = lambda i, p: float(2.0 * data.cost_a[i] * p + data.cost_b[i])
    return [dict(unit=i, at_pmin=marginal(i, data.pmin[i]),
                 at_current_p=marginal(i, current_p[i]) if on_hours[i] else None, at_pmax=marginal(i, data.pmax[i]))
            for i in range(data.n_units)]


# 按机组拆分空载、实际冷热启动、停机及期望燃料费用，揭示全天降本来源。
def _unit_cost_decomposition(data, scenarios, incumbent, results):
    starts = startup_costs(data, incumbent.u)
    rows = []
    for i in range(data.n_units):
        fuel = float(sum(scenario.probability * np.sum(data.cost_a[i] * np.asarray(result.p)[i] ** 2
                                                       + data.cost_b[i] * np.asarray(result.p)[i])
                         for scenario, result in zip(scenarios, results)))
        rows.append(dict(unit=i, fixed_cost=float(np.sum(data.cost_c[i] * incumbent.u[i])),
                         startup_cost=float(np.sum(starts[i])),
                         shutdown_cost=float(np.sum(data.shutdown_cost[i] * incumbent.w[i])), fuel_cost=fuel))
    return rows


# 估算停掉每个已开机点的局部费用变化：-空载费用 + P*(系统价格-机组边际费用)。
# 系统价格为平衡对偶价格按场景概率重新归一化的加权平均。按估计降本排序后截取前 limit 项；
# 忽略重调度、启停和时序耦合，因此不是实测收益或可行性证明。
def _decommit_gradient_rows(data, scenarios, incumbent, results, expected_p, limit):
    weight = sum(scenario.probability for scenario in scenarios)
    price = sum(scenario.probability * np.asarray(result.balance_dual, dtype=float)
                for scenario, result in zip(scenarios, results)) / weight
    rows = []
    for i in range(data.n_units):
        for t in range(data.horizon):
            if not incumbent.u[i, t]:
                continue
            p_mw = float(expected_p[i, t])
            marginal = float(2.0 * data.cost_a[i] * p_mw + data.cost_b[i])
            fuel_change = p_mw * (float(price[t]) - marginal)
            rows.append(dict(unit=i, period=t, fixed_saving=float(data.cost_c[i]), output_mw=p_mw,
                             unit_marginal_cost=marginal, system_price=float(price[t]),
                             estimated_fuel_change=fuel_change,
                             estimated_total_change=-float(data.cost_c[i]) + fuel_change))
    rows.sort(key=lambda row: row["estimated_total_change"])
    return rows[:limit]


# 按小时记录开机数、机组编号及包含实际启动费的固定费用，用于观察尾部过度开机。
def _hourly_commitment_profile(data, incumbent, limit=24):
    starts = startup_costs(data, incumbent.u)
    rows = []
    for t in range(min(limit, data.horizon)):
        units_on = np.flatnonzero(incumbent.u[:, t]).tolist()
        hourly_fixed = float(np.sum(data.cost_c[units_on])) if units_on else 0.0
        hourly_fixed += float(np.sum(starts[:, t]))
        hourly_fixed += float(np.sum(data.shutdown_cost * incumbent.w[:, t]))
        rows.append(dict(period=t, units_on=len(units_on), unit_ids=units_on, hourly_fixed_cost=hourly_fixed))
    return rows


# 按实际开机小时占比给机组标注 base/peaker/mid_merit 展示角色，并汇总启动费用。
# 这些角色由当前方案得出，不是输入中固定的机组分类。
def _peaker_usage(data, incumbent):
    starts = startup_costs(data, incumbent.u)
    rows = []
    for i in range(data.n_units):
        on_hours = int(incumbent.u[i].sum())
        role = "base" if on_hours >= data.horizon else "peaker" if on_hours <= 0.5 * data.horizon else "mid_merit"
        rows.append(dict(unit=i, on_hours=on_hours, cost_c=float(data.cost_c[i]),
                         startup_cost=float(data.startup_cost[i]), incurred_startup_cost=float(starts[i].sum()),
                         pmax=float(data.pmax[i]), pmin=float(data.pmin[i]), role=role,
                         cost_per_on_hour=float(data.cost_c[i] + starts[i].sum() / max(1, on_hours))))
    return rows


# 接收 LLMContextInput，返回费用、候选、调度、备用裕量、敏感度及历史反馈的展示字典。
# 所有费用来自数值结果，最后才做显示精度裁剪。
def build_llm_context(data, scenarios, context_input):
    """Build the compact P0/P1 context injected into the LLM prompt."""
    incumbent, results = context_input.incumbent, context_input.incumbent_results
    fixed = fixed_cost(data, incumbent.u, incumbent.w)
    expected_fuel = float(sum(scenario.probability * result.objective for scenario, result in zip(scenarios, results)))
    expected_p = sum(scenario.probability * np.asarray(result.p, dtype=float)
                     for scenario, result in zip(scenarios, results))
    available = capacity(data, incumbent.u, incumbent.v, incumbent.w).sum(axis=0)
    return _round3({
        "incumbent_cost": dict(total=fixed + expected_fuel, fixed=fixed, expected_fuel=expected_fuel,
                               scenario_fuel={scenario.name: float(result.objective)
                                              for scenario, result in zip(scenarios, results)}),
        "scenario_duals": dict(balance=[result.balance_dual.tolist() for result in results],
                               reserve=[result.reserve_dual.tolist() for result in results]),
        "candidate_comparison": _candidate_comparison_rows(context_input.candidate_evaluations),
        "scenario_dispatch_summary": _scenario_dispatch_summary(data, scenarios, incumbent, results),
        "reserve_margin": [(available - (1.0 + data.reserve_fraction) * scenario.load).tolist()
                           for scenario in scenarios],
        "iteration_history": _iteration_history_rows(context_input.history),
        "unit_marginal_cost": _unit_marginal_cost(data, incumbent, expected_p),
        "unit_cost_decomposition": _unit_cost_decomposition(data, scenarios, incumbent, results),
        "unit_hour_decommit_gradient": _decommit_gradient_rows(data, scenarios, incumbent, results, expected_p, 60),
        "hourly_commitment_profile": _hourly_commitment_profile(data, incumbent),
        "peaker_usage": _peaker_usage(data, incumbent),
    })
