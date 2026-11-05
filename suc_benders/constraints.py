"""Constraint policies used to build the block-search QUBO."""
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np


@dataclass(frozen=True)
class ConstraintRow:
    """One local binary inequality/equality before QUBO encoding."""

    indices: np.ndarray
    values: np.ndarray
    rhs: float
    equality: bool = False
    kind: str = "hard"

    def signature(self):
        return (self.kind, float(self.rhs), tuple(self.indices.tolist()), tuple(self.values.tolist()))


def _position(b):
    position = np.full(len(b.obj), -1, dtype=int)
    position[b.binary] = np.arange(len(b.binary))
    return position


def _capacity_rows(b):
    """Return the archived integer capacity rows for the legacy policy."""
    pmax = [Fraction(str(v)) for v in b.d.pmax]
    denominator = math.lcm(*(v.denominator for v in pmax))
    step = Fraction(math.gcd(*(int(v * denominator) for v in pmax)), denominator)
    capacity_units = np.asarray([int(v / step) for v in pmax], float) if step else np.zeros(len(pmax))
    position = _position(b)
    units = position[b.u]
    rows = []
    for t, hour in enumerate(b.times):
        required = (1 + b.d.reserve_fraction) * max(s.load[hour] for s in b.scenarios)
        required_units = math.ceil(Fraction(str(required)) / step) if step else 0
        rows.append(ConstraintRow(units[:, t], -capacity_units, -required_units, kind="capacity"))
    return rows


def _feasibility_rows(cuts):
    rows = []
    for cut in cuts:
        if not cut.optimality:
            rows.append(ConstraintRow(np.arange(len(cut.beta), dtype=int), np.asarray(cut.beta, float),
                                      -float(cut.alpha), kind="feasibility"))
    return rows


def _cover_for_capacity(units, pmax, required):
    """Build one valid minimal cover: all selected units cannot be off together."""
    threshold = float(np.sum(pmax) - required)
    if threshold < 0:
        chosen = np.arange(len(units), dtype=int)
    else:
        chosen = []
        total = 0.
        for index in np.argsort(-np.asarray(pmax, float)):
            chosen.append(int(index))
            total += float(pmax[index])
            if total > threshold + 1e-9:
                break
        if total <= threshold + 1e-9:
            return None
        chosen = np.asarray(chosen, dtype=int)
    if not len(chosen):
        return None
    return ConstraintRow(np.asarray(units)[chosen], -np.ones(len(chosen)), -1., kind="capacity_cover")


def _nogood(candidate):
    candidate = np.asarray(candidate, dtype=np.uint8)
    # At least one bit must differ: sum((1-x) for v=1) + sum(x for v=0) >= 1.
    values = np.where(candidate > 0, 1., -1.)
    rhs = float(np.count_nonzero(candidate == 1) - 1)
    return ConstraintRow(np.arange(len(candidate), dtype=int), values, rhs, kind="nogood")


class LegacySquarePolicy:
    """The v2.1.0 squared-penalty policy kept for compatibility and rollback."""

    name = "legacy_square"

    def rows(self, b, cuts, options, extra_rows=()):
        return [*list(extra_rows), *_capacity_rows(b), *_feasibility_rows(cuts)]

    def observe(self, b, candidate, reason):
        return []

    def weight(self, kind, local_bound, options):
        if kind in ("capacity", "feasibility"):
            return options.sa_capacity_weight
        return options.sa_hard_weight


class CoverPolicy(LegacySquarePolicy):
    """Lazy 0/1 covers with no-good rows for repeated local assignments."""

    name = "cover"
    dominant_penalties = True

    def rows(self, b, cuts, options, extra_rows=()):
        position = _position(b)
        units = position[b.u]
        rows = list(extra_rows)
        for t, hour in enumerate(b.times):
            required = (1 + b.d.reserve_fraction) * max(s.load[hour] for s in b.scenarios)
            cover = _cover_for_capacity(units[:, t], b.d.pmax, required)
            if cover is not None:
                rows.append(cover)
        rows.extend(_feasibility_rows(cuts))
        return rows

    def observe(self, b, candidate, reason):
        candidate = np.asarray(candidate, dtype=np.uint8)
        position = _position(b)
        units = position[b.u]
        rows = []
        if reason in {"discrete_infeasible", "repeated", "repair_infeasible"}:
            rows.append(_nogood(candidate))
        if reason in {"discrete_infeasible", "repair_infeasible"}:
            for t, hour in enumerate(b.times):
                required = (1 + b.d.reserve_fraction) * max(s.load[hour] for s in b.scenarios)
                u = candidate[units[:, t]]
                if float(np.asarray(b.d.pmax) @ u) + 1e-6 >= required:
                    continue
                off = np.flatnonzero(u == 0)
                remaining = required - float(np.asarray(b.d.pmax) @ u)
                cover = _cover_for_capacity(units[:, t][off], np.asarray(b.d.pmax)[off], remaining)
                if cover is not None:
                    rows.append(cover)
        return rows

    def weight(self, kind, local_bound, options):
        if kind in ("capacity_cover", "nogood"):
            baseline = options.sa_cover_weight
        elif kind == "feasibility":
            baseline = options.sa_feasibility_weight
        else:
            baseline = options.sa_hard_weight
        return max(baseline, 1.1 * local_bound)


legacy_square_policy = LegacySquarePolicy()
cover_policy = CoverPolicy()

