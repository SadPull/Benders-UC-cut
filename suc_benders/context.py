"""Whole-day advisor context restored from the D/17 frozen implementation."""
# 模块职责：把已验证方案、场景调度、费用与对偶敏感度整理成紧凑的全天 LLM 上下文。
# unit/period 使用零起始索引；梯度是局部启发式估计，三位小数仅用于提示展示，不参与认证。
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .validation import capacity as commitment_capacity, fixed_cost, startup_costs


# 用于提示词的候选评估摘要，保存来源、实测费用、可行性、割数量和与当前方案的距离。
@dataclass
class CandidateEvaluation:
    source: str
    total_cost: float
    fixed_cost: float
    expected_fuel_cost: float
    scenario_costs: dict[str, float]
    feasible: bool
    optimality_cuts: int
    feasibility_cuts: int
    hamming_to_incumbent: int | None


# 上下文构建输入：当前方案及场景调度、历史候选评估、迭代记录与不可行诊断。
@dataclass
class LLMContextInput:
    incumbent: Any | None
    incumbent_results: Sequence[Any]
    base_evaluations: Sequence[CandidateEvaluation]
    candidate_evaluations: Sequence[CandidateEvaluation]
    history: Sequence[Any]
    infeasible_diagnostics: Sequence[Mapping[str, Any]] = field(default_factory=list)


# 递归转换 NumPy 标量并将有限浮点数保留三位小数，无穷/NaN 转为 None 以便生成合法 JSON。
def _round3(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _round3(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round3(item) for item in value]
    if isinstance(value, tuple):
        return [_round3(item) for item in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return round(value, 3) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


# 按当前全天 u/w 重算固定运行、实际冷热启动与停机费用。
def _fixed_cost(data: Any, incumbent: Any) -> float:
    return fixed_cost(data, incumbent.u, incumbent.w)


# 读取各场景未乘概率的燃料费用，按场景名返回明细字典。
def _scenario_costs(results: Sequence[Any], scenarios: Sequence[Any]) -> dict[str, float]:
    return {
        scenario.name: float(result.objective)
        for scenario, result in zip(scenarios, results)
    }


# 对各场景燃料费用乘概率求和；固定费用在上层另外计入。
def _expected_fuel(results: Sequence[Any], scenarios: Sequence[Any]) -> float:
    return float(
        sum(scenario.probability * result.objective for scenario, result in zip(scenarios, results))
    )


# 按实测总费用保留至多十个基础方案摘要，并标出是否优于当前 incumbent。
def _base_candidate_rows(
    evaluations: Sequence[CandidateEvaluation],
    incumbent_total: float | None,
) -> list[dict]:
    rows = []
    for evaluation in sorted(evaluations, key=lambda item: item.total_cost)[:10]:
        rows.append(
            {
                "id": len(rows),
                "source": evaluation.source,
                "total_cost": evaluation.total_cost,
                "fixed_cost": evaluation.fixed_cost,
                "expected_fuel_cost": evaluation.expected_fuel_cost,
                "better_than_incumbent": (
                    incumbent_total is not None and evaluation.total_cost < incumbent_total
                ),
                "hamming_to_incumbent": evaluation.hamming_to_incumbent,
            }
        )
    return rows


# 将各场景的平衡/备用对偶价格转换成可序列化列表；缺失价格返回空列表。
def _scenario_duals(results: Sequence[Any]) -> dict:
    balance = [
        result.balance_dual.tolist() if result.balance_dual is not None else []
        for result in results
    ]
    reserve = [
        result.reserve_dual.tolist() if result.reserve_dual is not None else []
        for result in results
    ]
    return {"balance": balance, "reserve": reserve}


# 按总费用选取至多十个候选对照，展示费用分解、可行性和最优性割数量。
def _candidate_comparison_rows(
    evaluations: Sequence[CandidateEvaluation],
) -> list[dict]:
    rows = []
    for evaluation in sorted(evaluations, key=lambda item: item.total_cost)[:10]:
        rows.append(
            {
                "source": evaluation.source,
                "total_cost": evaluation.total_cost,
                "fixed_cost": evaluation.fixed_cost,
                "expected_fuel_cost": evaluation.expected_fuel_cost,
                "hamming_to_incumbent": evaluation.hamming_to_incumbent,
                "feasible": evaluation.feasible,
                "optimality_cuts": evaluation.optimality_cuts,
            }
        )
    return rows


# 汇总各场景发电、备用与启停次数；marginal_unit_count 统计出力接近上下限的开机点。
def _scenario_dispatch_summary(
    data: Any,
    scenarios: Sequence[Any],
    incumbent: Any | None,
    results: Sequence[Any],
) -> list[dict]:
    rows = []
    for scenario, result in zip(scenarios, results):
        p = np.asarray(result.p, dtype=float) if result.p is not None else None
        r = np.asarray(result.r, dtype=float) if result.r is not None else None
        marginal_count = 0
        if p is not None and incumbent is not None:
            at_min = np.isclose(p, data.pmin[:, None], atol=1e-5)
            at_max = np.isclose(p, data.pmax[:, None], atol=1e-5)
            marginal_count = int(np.sum((at_min | at_max) & (incumbent.u == 1)))
        rows.append(
            {
                "scenario": scenario.name,
                "total_generation": float(np.sum(p)) if p is not None else 0.0,
                "total_reserve": float(np.sum(r)) if r is not None else 0.0,
                "startup_count": int(np.sum(incumbent.v)) if incumbent is not None else 0,
                "shutdown_count": int(np.sum(incumbent.w)) if incumbent is not None else 0,
                "marginal_unit_count": marginal_count,
            }
        )
    return rows


# 用启停容量约束下的总可用容量减去各场景负荷及备用需求，返回场景 × 小时的裕量。
def _reserve_margin(
    data: Any,
    scenarios: Sequence[Any],
    incumbent: Any | None,
) -> list[list[float]]:
    if incumbent is None:
        return [[] for _ in scenarios]
    capacity = commitment_capacity(data, incumbent.u, incumbent.v, incumbent.w).sum(axis=0)
    return [
        (capacity - (1.0 + data.reserve_fraction) * scenario.load).tolist()
        for scenario in scenarios
    ]


# 按平衡/备用对偶价格绝对值列出至多二十个敏感时段，供 LLM 判断可能的紧约束。
def _tight_constraints(
    scenarios: Sequence[Any],
    results: Sequence[Any],
) -> list[dict]:
    rows = []
    for scenario, result in zip(scenarios, results):
        if result.balance_dual is not None:
            for period, dual in enumerate(result.balance_dual):
                rows.append(
                    {
                        "type": "balance",
                        "scenario": scenario.name,
                        "period": period,
                        "dual": float(dual),
                    }
                )
        if result.reserve_dual is not None:
            for period, dual in enumerate(result.reserve_dual):
                rows.append(
                    {
                        "type": "reserve",
                        "scenario": scenario.name,
                        "period": period,
                        "dual": float(dual),
                    }
                )
    rows.sort(key=lambda row: abs(row["dual"]), reverse=True)
    return rows[:20]


# 只保留最近五轮的上界改善、割增量、候选数和 LLM 失败摘要，控制提示词长度。
def _iteration_history_rows(history: Sequence[Any]) -> list[dict]:
    rows = []
    previous = None
    for record in list(history)[-5:]:
        incumbent = float(record.incumbent)
        improvement = 0.0 if previous is None else max(0.0, previous - incumbent)
        rows.append(
            {
                "iteration": int(record.iteration),
                "incumbent": incumbent,
                "improvement": improvement,
                "optimality_cuts": int(record.optimality_cuts_added),
                "feasibility_cuts": int(record.feasibility_cuts_added),
                "new_candidates": int(record.new_candidates),
                "llm_failures": 1 if record.llm_failed else 0,
            }
        )
        previous = incumbent
    return rows


# 用燃料导数 2*a*P+b 估算各机组在 Pmin、概率加权平均出力和 Pmax 处的边际费用。
def _unit_marginal_cost(
    data: Any,
    scenarios: Sequence[Any],
    results: Sequence[Any],
) -> list[dict]:
    if not results:
        return []
    expected_p = sum(
        scenario.probability * np.asarray(result.p, dtype=float)
        for scenario, result in zip(scenarios, results)
    )
    current_p = expected_p.mean(axis=1)
    rows = []
    for i in range(data.n_units):
        rows.append(
            {
                "unit": i,
                "at_pmin": float(2.0 * data.cost_a[i] * data.pmin[i] + data.cost_b[i]),
                "at_current_p": float(2.0 * data.cost_a[i] * current_p[i] + data.cost_b[i]),
                "at_pmax": float(2.0 * data.cost_a[i] * data.pmax[i] + data.cost_b[i]),
            }
        )
    return rows


# 按机组拆分空载、实际冷热启动、停机及期望燃料费用，揭示全天降本来源。
def _unit_cost_decomposition(
    data: Any,
    scenarios: Sequence[Any],
    incumbent: Any | None,
    results: Sequence[Any],
) -> list[dict]:
    if incumbent is None or not results:
        return []
    rows = []
    starts = startup_costs(data, incumbent.u)
    for i in range(data.n_units):
        fuel = float(
            sum(
                scenario.probability * np.sum(
                    data.cost_a[i] * np.asarray(result.p)[i] ** 2
                    + data.cost_b[i] * np.asarray(result.p)[i]
                )
                for scenario, result in zip(scenarios, results)
            )
        )
        rows.append(
            {
                "unit": i,
                "fixed_cost": float(np.sum(data.cost_c[i] * incumbent.u[i])),
                "startup_cost": float(np.sum(starts[i])),
                "shutdown_cost": float(np.sum(data.shutdown_cost[i] * incumbent.w[i])),
                "fuel_cost": fuel,
            }
        )
    return rows


# 接收 LLMContextInput，返回费用、候选、调度、备用裕量、敏感度及历史反馈的展示字典。
# 所有费用来自数值结果，最后才做显示精度裁剪。
def build_llm_context(
    data: Any,
    scenarios: Sequence[Any],
    context_input: LLMContextInput,
) -> dict:
    """Build the compact P0/P1 context injected into the LLM prompt."""

    incumbent = context_input.incumbent
    results = context_input.incumbent_results
    fixed = _fixed_cost(data, incumbent) if incumbent is not None else None
    expected_fuel = _expected_fuel(results, scenarios) if results else None
    total = fixed + expected_fuel if fixed is not None and expected_fuel is not None else None
    incumbent_cost = None
    if total is not None:
        incumbent_cost = {
            "total": total,
            "fixed": fixed,
            "expected_fuel": expected_fuel,
            "scenario_fuel": _scenario_costs(results, scenarios),
        }
    return _round3(
        {
            "incumbent_cost": incumbent_cost,
            "base_candidates": _base_candidate_rows(
                context_input.base_evaluations, total
            ),
            "scenario_duals": _scenario_duals(results),
            "candidate_comparison": _candidate_comparison_rows(
                context_input.candidate_evaluations
            ),
            "scenario_dispatch_summary": _scenario_dispatch_summary(data, scenarios, incumbent, results),
            "reserve_margin": _reserve_margin(data, scenarios, incumbent),
            "tight_constraints": _tight_constraints(scenarios, results),
            "infeasible_diagnostics": list(context_input.infeasible_diagnostics)[:20],
            "iteration_history": _iteration_history_rows(context_input.history),
            "unit_marginal_cost": _unit_marginal_cost(data, scenarios, results),
            "unit_cost_decomposition": _unit_cost_decomposition(
                data, scenarios, incumbent, results
            ),
            "unit_hour_decommit_gradient": _unit_hour_decommit_gradient(
                data, scenarios, incumbent, results, limit=60
            ),
            "top_decommit_opportunities": _unit_hour_decommit_gradient(
                data, scenarios, incumbent, results, limit=20
            ),
            "hourly_commitment_profile": _hourly_commitment_profile(data, incumbent),
            "peaker_usage": _peaker_usage(data, incumbent),
        }
    )


# 返回场景概率加权的机组 × 小时出力矩阵；没有调度结果时返回 None。
def _expected_outputs(
    scenarios: Sequence[Any], results: Sequence[Any]
) -> np.ndarray | None:
    """Probability-weighted output, shape (units, horizon)."""

    if not results:
        return None
    return sum(
        scenario.probability * np.asarray(result.p, dtype=float)
        for scenario, result in zip(scenarios, results)
    )


# 对具有平衡对偶价格的场景重新归一化概率，得到每小时的系统价格估计。
def _system_price(
    scenarios: Sequence[Any], results: Sequence[Any]
) -> np.ndarray | None:
    """Probability-weighted energy price from the balance duals, shape (horizon,)."""

    usable = [
        (scenario, result)
        for scenario, result in zip(scenarios, results)
        if result.balance_dual is not None
    ]
    if not usable:
        return None
    weight = sum(scenario.probability for scenario, _ in usable)
    if weight <= 0:
        return None
    return sum(
        scenario.probability * np.asarray(result.balance_dual, dtype=float)
        for scenario, result in usable
    ) / weight


# 估算停掉每个已开机点的局部费用变化：-空载费用 + P*(系统价格-机组边际费用)。
# 按估计降本排序；忽略重调度、启停和时序耦合，因此不是实测收益或可行性证明。
def _decommit_gradient_rows(
    data: Any,
    scenarios: Sequence[Any],
    incumbent: Any | None,
    results: Sequence[Any],
) -> list[dict]:
    """Full per-(unit, hour) decommitment gradient table (untruncated)."""

    output = _expected_outputs(scenarios, results)
    price = _system_price(scenarios, results)
    if incumbent is None or output is None or price is None:
        return []

    rows = []
    for i in range(data.n_units):
        for t in range(data.horizon):
            if not incumbent.u[i, t]:
                continue
            p_mw = float(output[i, t])
            marginal = float(2.0 * data.cost_a[i] * p_mw + data.cost_b[i])
            fuel_change = p_mw * (float(price[t]) - marginal)
            rows.append(
                {
                    "unit": i,
                    "period": t,
                    "fixed_saving": float(data.cost_c[i]),
                    "output_mw": p_mw,
                    "unit_marginal_cost": marginal,
                    "system_price": float(price[t]),
                    "estimated_fuel_change": fuel_change,
                    "estimated_total_change": -float(data.cost_c[i]) + fuel_change,
                }
            )
    rows.sort(key=lambda row: row["estimated_total_change"])
    return rows


# 截取排序后的停机敏感度表前 limit 项，作为提示词中的候选机会列表。
def _unit_hour_decommit_gradient(
    data: Any,
    scenarios: Sequence[Any],
    incumbent: Any | None,
    results: Sequence[Any],
    *,
    limit: int = 12,
) -> list[dict]:
    """Truncated view of the decommitment gradient for the LLM prompt."""

    return _decommit_gradient_rows(data, scenarios, incumbent, results)[:limit]


# 按小时记录开机数、机组编号及包含实际启动费的固定费用，用于观察尾部过度开机。
def _hourly_commitment_profile(
    data: Any,
    incumbent: Any | None,
    *,
    limit: int = 24,
) -> list[dict]:
    """Per-hour on-count and fixed cost, exposing over-commitment in the tail."""

    if incumbent is None:
        return []
    rows = []
    starts = startup_costs(data, incumbent.u)
    for t in range(min(limit, data.horizon)):
        units_on = np.flatnonzero(incumbent.u[:, t]).tolist()
        hourly_fixed = float(np.sum(data.cost_c[units_on])) if units_on else 0.0
        hourly_fixed += float(
            np.sum(starts[:, t])
        )
        hourly_fixed += float(
            np.sum(data.shutdown_cost * incumbent.w[:, t])
        )
        rows.append(
            {
                "period": t,
                "units_on": len(units_on),
                "unit_ids": units_on,
                "hourly_fixed_cost": hourly_fixed,
            }
        )
    return rows


# 按实际开机小时占比给机组标注 base/peaker/mid_merit 展示角色，并汇总启动费用。
# 这些角色由当前方案得出，不是输入中固定的机组分类。
def _peaker_usage(data: Any, incumbent: Any | None) -> list[dict]:
    """Per-unit on-hours and role, so the LLM can see which units are peakers."""

    if incumbent is None:
        return []
    rows = []
    starts = startup_costs(data, incumbent.u)
    for i in range(data.n_units):
        on_hours = int(incumbent.u[i].sum())
        if on_hours >= data.horizon:
            role = "base"
        elif on_hours <= 0.5 * data.horizon:
            role = "peaker"
        else:
            role = "mid_merit"
        rows.append(
            {
                "unit": i,
                "on_hours": on_hours,
                "cost_c": float(data.cost_c[i]),
                "startup_cost": float(data.startup_cost[i]),
                "incurred_startup_cost": float(starts[i].sum()),
                "pmax": float(data.pmax[i]),
                "pmin": float(data.pmin[i]),
                "role": role,
                "cost_per_on_hour": float(data.cost_c[i] + starts[i].sum() / max(1, on_hours)),
            }
        )
    return rows
