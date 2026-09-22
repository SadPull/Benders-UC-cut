from __future__ import annotations

from dataclasses import dataclass, field
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
        self.reserve_scale = float(reserve_scale)
        if self.reserve_scale <= 0:
            raise ValueError("reserve_scale must be positive")
        self.theta_bits = theta_bit_width(self.theta_cap, self.theta_step)
        self.kw = get_kaiwu()
        self.model = self.kw.qubo.QuboModel()
        self.u: dict[tuple[int, int], object] = {}
        self.v: dict[tuple[int, int], object] = {}
        self.w: dict[tuple[int, int], object] = {}
        self._build()

    def _sum(self, expressions):
        result = 0
        for expression in expressions:
            result = result + expression
        return result

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

        # State-transition, min-up/down, and scenario reserve constraints are
        # linear and therefore safe in Kaiwu add_constraint.
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

        for scenario in self.scenarios:
            reserve = (1.0 + data.reserve_fraction) * np.asarray(scenario.load)
            for t in range(horizon):
                capacity = []
                for i in range(n):
                    expr = data.pmax[i] * self.u[i, t] - (data.pmax[i] - data.pstart[i]) * self.v[i, t]
                    if t + 1 < horizon:
                        expr = expr - (data.pmax[i] - data.pshut[i]) * self.w[i, t + 1]
                    capacity.append(expr * (1.0 / self.reserve_scale))
                self.model.add_constraint(
                    self._sum(capacity) >= float(reserve[t]) / self.reserve_scale,
                    constr_type="hard", penalty=self.penalty,
                )

        for cut in self.cuts:
            terms = []
            for i in range(n):
                for t in range(horizon):
                    terms.extend(
                        (
                            cut.beta_u[i, t] * self.u[i, t],
                            cut.beta_v[i, t] * self.v[i, t],
                            cut.beta_w[i, t] * self.w[i, t],
                        )
                    )
            rhs = cut.alpha
            if cut.kind == "optimality":
                self.model.add_constraint(theta - self._sum(terms) >= rhs, constr_type="hard", penalty=self.penalty)
            else:
                self.model.add_constraint(self._sum(terms) <= -rhs, constr_type="hard", penalty=self.penalty)

        self.model.set_objective(self._sum(objective))

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
        return len(self.decision_variables())

    def matrix(self) -> np.ndarray:
        """Return the QUBO matrix ordered by decision_variables()."""
        all_variables = self.model.get_variables()
        names = self.decision_variables()
        indices = [all_variables['b' + name] for name in names]
        matrix = np.asarray(self.model.get_matrix(), dtype=float)
        return matrix[np.ix_(indices, indices)]

    def decode(self, binary: np.ndarray) -> MasterCandidate:
        names = self.decision_variables()
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
