from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil, floor, gcd, lcm
from typing import Iterable, Sequence

import numpy as np

from .common import get_kaiwu, Scenario
from .data import UCData
from .subproblem import ScenarioResult


@dataclass(frozen=True)
class BendersCut:
    alpha: float
    beta_u: np.ndarray
    beta_v: np.ndarray
    beta_w: np.ndarray
    kind: str = "optimality"

    def value(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        return float(
            self.alpha
            + np.sum(self.beta_u * u)
            + np.sum(self.beta_v * v)
            + np.sum(self.beta_w * w)
        )


@dataclass
class MasterCandidate:
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    theta: float | None = None
    source: str = "quantum"
    move: dict | None = None
    provenance: dict = field(default_factory=dict)


@dataclass
class QuantumConfig:
    backend: str = "kaiwu-local"
    kaiwu_seed: int | None = 7
    candidate_pool_size: int = 16
    initial_temperature: float = 100.0
    alpha: float = 0.98
    cutoff_temperature: float = 0.01
    iterations_per_temperature: int = 5


def theta_bit_width(theta_cap: float, theta_step: float) -> int:
    if theta_cap <= 0 or theta_step <= 0:
        raise ValueError("theta cap and step must be positive")
    max_steps = int(np.ceil(theta_cap / theta_step))
    return max(1, int(np.ceil(np.log2(max_steps + 1))))


@dataclass(frozen=True)
class _IntegerInequality:
    """step * (constant + sum(coefficients[name] * bit)) <= 0."""

    coefficients: dict[str, int]
    constant: int
    step: Fraction
    rounding_error: float = 0.0


def _capacity_inequality(
    capacity: dict[str, Fraction], requirement: Fraction
) -> _IntegerInequality:
    """Encode a decimal capacity lattice exactly, including a fractional demand."""

    capacity = {name: value for name, value in capacity.items() if value}
    denominator = lcm(*(value.denominator for value in capacity.values()))
    integers = {name: int(value * denominator) for name, value in capacity.items()}
    divisor = gcd(*integers.values()) or 1
    step = Fraction(divisor, denominator)
    return _IntegerInequality(
        {name: -value // divisor for name, value in integers.items()},
        ceil(requirement / step),
        step,
    )


def _quantize_inequality(
    coefficients: dict[str, float], constant: float, error_budget: float,
    exact_coefficients: dict[str, Fraction] | None = None,
) -> _IntegerInequality:
    """Round a supporting plane down, with a bound valid for every binary point."""

    coefficients = {name: value for name, value in coefficients.items() if value}
    step = Fraction(str(error_budget)) / (len(coefficients) + 1)
    # Fractions of the actual floats keep flooring conservative even at grid edges.
    integers = {name: floor(Fraction(float(value)) / step) for name, value in coefficients.items()}
    integer_constant = floor(Fraction(float(constant)) / step)
    error = Fraction(float(constant)) - integer_constant * step
    error += sum(Fraction(float(value)) - integers[name] * step for name, value in coefficients.items())
    for name, value in (exact_coefficients or {}).items():
        integer = value / step
        if integer.denominator != 1:
            raise ValueError("exact coefficient is not on the quantization grid")
        integers[name] = int(integer)
    return _IntegerInequality(integers, integer_constant, step, float(error))


class QuantumMaster:
    """Kaiwu QUBO construction and decoding for the first-stage problem."""

    def __init__(
        self,
        data: UCData,
        scenarios: Sequence[Scenario],
        cuts: Iterable[BendersCut],
        theta_cap: float,
        *,
        penalty: float = 1_000_000.0,
        theta_step: float = 1.0,
        reserve_scale: float = 100.0,
    ):
        if theta_cap <= 0 or theta_step <= 0:
            raise ValueError("theta cap and step must be positive")
        self.data = data
        self.scenarios = list(scenarios)
        self.cuts = list(cuts)
        self.penalty = float(penalty)
        self.theta_step = float(theta_step)
        self.theta_cap = float(theta_cap)
        self.cost_scale = max(1.0, self.theta_cap)
        self.reserve_scale = float(reserve_scale)
        if self.reserve_scale <= 0:
            raise ValueError("reserve_scale must be positive")
        self.theta_bits = theta_bit_width(self.theta_cap, self.theta_step)
        self.kw = get_kaiwu()
        self.model = self.kw.core.QuboModel()
        self.u: dict[tuple[int, int], object] = {}
        self.v: dict[tuple[int, int], object] = {}
        self.w: dict[tuple[int, int], object] = {}
        self._encoded_inequalities: dict[str, _IntegerInequality] = {}
        self._build()

    def _sum(self, expressions):
        result = 0
        for expression in expressions:
            result = result + expression
        return result

    def _add_integer_inequality(
        self, name: str, inequality: _IntegerInequality, scale: float
    ) -> None:
        self._encoded_inequalities[name] = inequality
        lower = inequality.constant + sum(min(0, value) for value in inequality.coefficients.values())
        upper = inequality.constant + sum(max(0, value) for value in inequality.coefficients.values())
        if upper <= 0:
            return
        # A zero range needs no slack bit. An impossible inequality remains a
        # positive penalty instead of failing in the SDK's automatic encoder.
        slack = self.kw.core.Integer(f"_slack_{name}", 0, -lower) if lower < 0 else 0
        residual = self.kw.core.BinaryExpression(
            coefficient={(key,): value for key, value in inequality.coefficients.items() if value},
            offset=inequality.constant,
        ) + slack
        residual = residual * float(inequality.step / Fraction(str(scale)))
        self.model.add_constraint(residual ** 2, name=name, constr_type="hard", penalty=self.penalty)

    def _build(self) -> None:
        data = self.data
        n, horizon = data.n_units, data.horizon
        for i in range(n):
            for t in range(horizon):
                self.u[i, t] = self.kw.core.Binary(f"u_{i}_{t}")
                self.v[i, t] = self.kw.core.Binary(f"v_{i}_{t}")
                self.w[i, t] = self.kw.core.Binary(f"w_{i}_{t}")

        objective = []
        for i in range(n):
            for t in range(horizon):
                objective.append(
                    data.cost_c[i] * self.u[i, t]
                    + data.startup_cost[i] * self.v[i, t]
                    + data.shutdown_cost[i] * self.w[i, t]
                )

        theta = self._build_theta()
        objective.append(theta)

        costs = np.r_[data.cost_c, data.startup_cost, data.shutdown_cost, self.theta_step]
        self.energy_unit = float(np.min(costs[costs > 0])) / self.cost_scale

        # These temporal constraints have integer coefficients and exact SDK slack.
        for i in range(n):
            prev = 1.0 if data.initial_on[i] else 0.0
            for t in range(horizon):
                self.model.add_constraint(
                    self.u[i, t] - prev - self.v[i, t] + self.w[i, t] == 0,
                    constr_type="hard", penalty=self.penalty,
                )
                self.model.add_constraint(
                    self.v[i, t] + self.w[i, t] <= 1,
                    constr_type="hard", penalty=self.penalty,
                )
                prev = self.u[i, t]
            self.model.add_constraint(self.w[i, horizon - 1] == 0, constr_type="hard", penalty=self.penalty)
            for t in range(horizon):
                for k in range(t, min(horizon, t + int(data.min_up[i]))):
                    self.model.add_constraint(self.u[i, k] >= self.v[i, t], constr_type="hard", penalty=self.penalty)
                for k in range(t, min(horizon, t + int(data.min_down[i]))):
                    self.model.add_constraint(self.u[i, k] <= 1 - self.w[i, t], constr_type="hard", penalty=self.penalty)
            if data.initial_on[i]:
                remaining = max(0, int(data.min_up[i]) - int(data.initial_duration[i]))
                for t in range(remaining):
                    self.model.add_constraint(self.u[i, t] == 1, constr_type="hard", penalty=self.penalty)
            else:
                remaining = max(0, int(data.min_down[i]) - int(data.initial_duration[i]))
                for t in range(remaining):
                    self.model.add_constraint(self.u[i, t] == 0, constr_type="hard", penalty=self.penalty)

        for t in range(horizon):
            capacity = {}
            for i in range(n):
                pmax = Fraction(str(data.pmax[i]))
                start_reduction = pmax - Fraction(str(data.pstart[i]))
                shut_reduction = pmax - Fraction(str(data.pshut[i]))
                capacity[f"u_{i}_{t}"] = pmax
                capacity[f"v_{i}_{t}"] = -start_reduction
                if t + 1 < horizon:
                    capacity[f"w_{i}_{t + 1}"] = -shut_reduction
                    if data.min_up[i] == 1:
                        # z = v[t] AND w[t+1]. The correction makes simultaneous
                        # startup/shutdown capacity min(Pstart, Pshut).
                        name = f"_one_hour_{i}_{t}"
                        z = self.kw.core.Binary(name)
                        v, w = self.v[i, t], self.w[i, t + 1]
                        self.model.add_constraint(
                            v * w - 2 * v * z - 2 * w * z + 3 * z,
                            name=name, constr_type="hard", penalty=self.penalty,
                        )
                        capacity[name] = min(start_reduction, shut_reduction)
            for index, scenario in enumerate(self.scenarios):
                requirement = (1 + Fraction(str(data.reserve_fraction))) * Fraction(str(scenario.load[t]))
                self._add_integer_inequality(
                    f"reserve_{index}_{t}",
                    _capacity_inequality(capacity, requirement),
                    self.reserve_scale,
                )

        for index, cut in enumerate(self.cuts):
            coefficients = {
                f"{name}_{i}_{t}": float(beta[i, t])
                for name, beta in (("u", cut.beta_u), ("v", cut.beta_v), ("w", cut.beta_w))
                for i in range(n) for t in range(horizon)
            }
            if cut.kind == "optimality":
                exact = {
                    f"theta_{k}": -Fraction(str(self.theta_step)) * (1 << k)
                    for k in range(self.theta_bits)
                }
                inequality = _quantize_inequality(
                    coefficients, cut.alpha, self.theta_step, exact
                )
                scale = self.cost_scale
            else:
                inequality = _quantize_inequality(coefficients, cut.alpha, 1e-3)
                scale = self.reserve_scale
            self._add_integer_inequality(f"cut_{index}", inequality, scale)

        self.model.set_objective(self._sum(objective) * (1.0 / self.cost_scale))

    def _build_theta(self):
        return self._sum(
            (self.theta_step * (1 << k)) * self.kw.core.Binary(f"theta_{k}")
            for k in range(self.theta_bits)
        )

    def decision_variables(self) -> list[str]:
        n, horizon = self.data.n_units, self.data.horizon
        names = [f"u_{i}_{t}" for i in range(n) for t in range(horizon)]
        names += [f"v_{i}_{t}" for i in range(n) for t in range(horizon)]
        names += [f"w_{i}_{t}" for i in range(n) for t in range(horizon)]
        names += [f"theta_{k}" for k in range(self.theta_bits)]
        return names

    def variable_count(self) -> int:
        """Number of decision bits, excluding slack and one-hour auxiliary bits."""
        return len(self.decision_variables())

    def qubo_variables(self) -> list[str]:
        """Decision bits followed by all slack and one-hour auxiliary bits."""
        names = self.decision_variables()
        decisions = set(names)
        names.extend(name for name in self.model.get_variables() if name not in decisions)
        return names

    def matrix(self) -> np.ndarray:
        """Return the full QUBO matrix ordered by qubo_variables()."""
        all_variables = self.model.get_variables()
        names = self.qubo_variables()
        indices = [all_variables[name] for name in names]
        matrix = np.asarray(self.model.get_matrix(), dtype=float)
        return matrix[np.ix_(indices, indices)]

    def decode(self, binary: np.ndarray) -> MasterCandidate:
        names = self.qubo_variables()
        if len(names) != len(binary):
            raise ValueError(f"binary solution has {len(binary)} entries; expected {len(names)}")
        values = {name: int(value) for name, value in zip(names, np.asarray(binary, dtype=int))}
        n, horizon = self.data.n_units, self.data.horizon
        u = np.array([[values[f"u_{i}_{t}"] for t in range(horizon)] for i in range(n)], dtype=int)
        v = np.array([[values[f"v_{i}_{t}"] for t in range(horizon)] for i in range(n)], dtype=int)
        w = np.array([[values[f"w_{i}_{t}"] for t in range(horizon)] for i in range(n)], dtype=int)
        theta = float(
            self.theta_step * sum(values[f"theta_{k}"] << k for k in range(self.theta_bits))
        )
        return MasterCandidate(u, v, w, theta)
