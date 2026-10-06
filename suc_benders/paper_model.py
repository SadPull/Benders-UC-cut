"""Indexed whole-horizon SUC rows shared by continuous dispatch and QUBO.

Rows use residual <= 0 (or == 0). Costs are in original currency units;
scenario probabilities are incorporated exactly once, at construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from fractions import Fraction
import math
import time

import numpy as np

from .common import require_gurobi, get_kaiwu
from .master import _capacity_inequality, _quantize_inequality


def key(kind, i, t, s=-1):
    return (kind, i, t, s)


def binary(k):
    return k[0] in ("u", "v", "w")


def name(k):
    return "_".join(map(str, k)).replace("-", "m")


@dataclass
class Row:
    a: dict
    c: float = 0.0
    equality: bool = False
    scale: float = 1.0
    label: str = ""

    def value(self, x):
        return self.c + sum(v * x[k] for k, v in self.a.items())


@dataclass
class Cut:
    a: dict
    c: float
    feasible: bool = True

    def value(self, x):
        return self.c + sum(v * x[k] for k, v in self.a.items())


class IndexedUC:
    def __init__(self, data, scenarios):
        self.data, self.scenarios = data, scenarios
        self.bounds, self.linear, self.quadratic, self.rows = {}, {}, {}, []
        d = data
        for t in range(d.horizon):
            for i in range(d.n_units):
                for kind, cost in (("u", d.cost_c[i]), ("v", d.startup_cost[i]), ("w", d.shutdown_cost[i])):
                    k = key(kind, i, t)
                    self.bounds[k] = (0., 1.)
                    self.linear[k] = float(cost)
                for s, scenario in enumerate(scenarios):
                    for kind in ("p", "r"):
                        k = key(kind, i, t, s)
                        self.bounds[k] = (0., float(d.pmax[i]))
                        self.linear[k] = float(scenario.probability * d.cost_b[i]) if kind == "p" else 0.
                        self.quadratic[k] = float(scenario.probability * d.cost_a[i]) if kind == "p" else 0.
                u, v, w = (key(k, i, t) for k in ("u", "v", "w"))
                a = {u: 1., v: -1., w: 1.}
                if t:
                    a[key("u", i, t-1)] = -1.
                self.add(a, -float(d.initial_on[i]) if not t else 0., True, label="transition")
                self.add({v: 1., w: 1.}, -1., label="exclusive")
                if t == d.horizon - 1:
                    self.add({w: 1.}, equality=True, label="terminal")
                remaining = max(0, int((d.min_up if d.initial_on[i] else d.min_down)[i] - d.initial_duration[i]))
                if t < remaining:
                    self.add({u: 1.}, -float(d.initial_on[i]), True, label="initial")
                for j in range(t, min(d.horizon, t + int(d.min_up[i]))):
                    self.add({v: 1., key("u", i, j): -1.}, label="min_up")
                for j in range(t, min(d.horizon, t + int(d.min_down[i]))):
                    self.add({w: 1., key("u", i, j): 1.}, -1., label="min_down")
                for s in range(len(scenarios)):
                    p, r = key("p", i, t, s), key("r", i, t, s)
                    self.add({u: float(d.pmin[i]), p: -1.}, scale=d.pmax[i], label="pmin")
                    self.add({p: 1., r: 1., u: -float(d.pmax[i]), v: float(d.pmax[i]-d.pstart[i])}, scale=d.pmax[i], label="startup_capacity")
                    a = {p: 1., r: 1., u: -float(d.pmax[i])}
                    # Keep the continuous relaxation physically bounded.
                    self.add(a.copy(), scale=d.pmax[i], label="envelope")
                    if t + 1 < d.horizon:
                        a[key("w", i, t+1)] = float(d.pmax[i]-d.pshut[i])
                    self.add(a, scale=d.pmax[i], label="shutdown_capacity")
                    a = {p: 1.}
                    if t:
                        a[key("p", i, t-1, s)] = -1.
                    self.add(a, -float(d.ramp_up[i]) - (float(d.initial_output[i]) if not t else 0.), scale=d.pmax[i], label="ramp_up")
                    self.add({k: -v for k, v in a.items()}, -float(d.ramp_down[i]) + (float(d.initial_output[i]) if not t else 0.), scale=d.pmax[i], label="ramp_down")
            for s, scenario in enumerate(scenarios):
                self.add({key("p", i, t, s): 1. for i in range(d.n_units)}, -float(scenario.load[t]), True, label="balance")
                self.add({key("r", i, t, s): -1. for i in range(d.n_units)}, float(d.reserve_fraction*scenario.load[t]), label="reserve")

    def add(self, a, c=0., equality=False, scale=1., label=""):
        self.rows.append(Row({k: v for k, v in a.items() if v}, c, equality, float(scale), label))


def fixed_binary(rows):
    """Propagate only consequences of small integer hard rows, exactly."""
    fixed = {}
    small = [r for r in rows if all(binary(k) for k in r.a) and len(r.a) <= 3]
    changed = True
    while changed:
        changed = False
        for row in small:
            free = [k for k in row.a if k not in fixed]
            valid = []
            for bits in product((0, 1), repeat=len(free)):
                x = {**fixed, **dict(zip(free, bits))}
                value = row.value(x)
                if (abs(value) <= 1e-9 if row.equality else value <= 1e-9):
                    valid.append(bits)
            if not valid:
                raise ValueError("inconsistent local integer constraints")
            for j, k in enumerate(free):
                if all(v[j] == valid[0][j] for v in valid):
                    fixed[k] = valid[0][j]
                    changed = True
    return fixed


class Block:
    def __init__(self, indexed):
        self.uc = indexed
        self.times = tuple(range(indexed.data.horizon))
        self.keys = list(indexed.bounds)
        self.bkeys = [k for k in self.keys if binary(k)]
        self.ckeys = [k for k in self.keys if not binary(k)]
        self.rows = indexed.rows
        self.integer_rows = [r for r in self.rows if all(binary(k) for k in r.a)]
        self.fixed = fixed_binary(self.integer_rows)
        self.scale = max(1., sum(abs(indexed.linear[k])*indexed.bounds[k][1] for k in self.keys)
                         + sum(indexed.quadratic.get(k, 0.)*indexed.bounds[k][1]**2 for k in self.keys))

    def discrete_feasible(self, x):
        return all((abs(r.value(x)) <= 1e-7 if r.equality else r.value(x) <= 1e-7) for r in self.integer_rows)

    def objective_bounds(self, costs, continuous_only=False):
        keys = self.ckeys if continuous_only else self.keys
        lo = sum(min(costs[k]*self.uc.bounds[k][0], costs[k]*self.uc.bounds[k][1]) for k in keys)
        hi = sum(max(costs[k]*self.uc.bounds[k][0], costs[k]*self.uc.bounds[k][1])
                 + self.uc.quadratic.get(k, 0.)*self.uc.bounds[k][1]**2 for k in keys)
        return lo, hi

    def dispatch(self, x, costs, deadline):
        """Joint scenario convex QP: probabilities already live in costs."""
        gp = require_gurobi()
        for elastic in (False, True):
            if time.perf_counter() >= deadline:
                return None
            m = gp.Model("paper_block_feasibility" if elastic else "paper_block_dispatch")
            m.Params.OutputFlag = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(.001, deadline-time.perf_counter())
            y = {k: m.addVar(lb=self.uc.bounds[k][0], ub=self.uc.bounds[k][1], name=name(k)) for k in self.ckeys}
            records, slacks = [], []
            for row in self.rows:
                if all(binary(k) for k in row.a):
                    continue
                bx = {k: v for k, v in row.a.items() if binary(k)}
                rhs = -row.c - sum(v*x[k] for k, v in bx.items())
                expr = gp.quicksum(v*y[k] for k, v in row.a.items() if not binary(k))
                if elastic:
                    slack = m.addVar(lb=0.)
                    slacks.append(slack)
                    c = m.addConstr(expr-slack <= rhs)
                    records.append((c, bx, -1.))
                    if row.equality:
                        c = m.addConstr(-expr-slack <= -rhs)
                        records.append((c, bx, 1.))
                else:
                    c = m.addConstr(expr == rhs) if row.equality else m.addConstr(expr <= rhs)
                    records.append((c, bx, -1.))
            obj = gp.quicksum(slacks) if elastic else gp.quicksum(
                costs[k]*y[k] + self.uc.quadratic.get(k, 0.)*y[k]*y[k] for k in self.ckeys)
            m.setObjective(obj)
            m.optimize()
            if m.Status != gp.GRB.OPTIMAL:
                status = m.Status
                m.dispose()
                if not elastic and status in (gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD):
                    continue
                return None
            value = float(m.ObjVal)
            # Recompute the dual objective over the variable box. Using the
            # primal objective as a cut intercept would overstate a lower cut
            # whenever the QP still has a small primal-dual numerical gap.
            prices = [float(c.Pi) if c.Sense == "=" else min(0., float(c.Pi)) for c, _, _ in records]
            if elastic:
                by_index = {c.index: price for (c, _, _), price in zip(records, prices)}
                load = max((sum(col.getCoeff(j)*by_index[col.getConstr(j).index]
                                for j in range(col.size())) for col in (m.getCol(v) for v in slacks)), default=1.)
                prices = [price/max(1., load) for price in prices]
            linear = {k: 0. if elastic else costs[k] for k in self.ckeys}
            variable_keys = {v.index:k for k,v in y.items()}
            dual_value = 0.
            beta = {}
            for (c, bx, sign), price in zip(records, prices):
                dual_value += price*float(c.RHS)
                expression = m.getRow(c)
                for j in range(expression.size()):
                    k = variable_keys.get(expression.getVar(j).index)
                    if k is not None:
                        linear[k] -= price*expression.getCoeff(j)
                for k, v in bx.items():
                    beta[k] = beta.get(k, 0.) + sign*price*v
            for k, coefficient in linear.items():
                lo, hi = self.uc.bounds[k]
                quadratic = 0. if elastic else self.uc.quadratic.get(k, 0.)
                minimizer = min(hi, max(lo, -coefficient/(2*quadratic))) if quadratic else lo if coefficient >= 0 else hi
                dual_value += coefficient*minimizer + quadratic*minimizer**2
            cut = Cut(beta, dual_value-sum(v*x[k] for k, v in beta.items()), not elastic)
            point = {**x, **{k: float(v.X) for k, v in y.items()}}
            m.dispose()
            return {"feasible": not elastic, "cut": cut, "point": point,
                    "cost": value + sum(costs[k]*x[k] for k in self.bkeys), "recourse": value}
        return None

    def relaxation(self, costs, cuts, deadline):
        gp = require_gurobi()
        if time.perf_counter() >= deadline:
            return None
        m = gp.Model("paper_master_relaxation")
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = max(.001, deadline-time.perf_counter())
        x = {k: m.addVar(lb=self.fixed.get(k, 0.), ub=self.fixed.get(k, 1.)) for k in self.bkeys}
        qlo, _ = self.objective_bounds(costs, True)
        q = m.addVar(lb=qlo)
        for row in self.integer_rows:
            expr = row.c + gp.quicksum(v*x[k] for k, v in row.a.items())
            m.addConstr(expr == 0 if row.equality else expr <= 0)
        d = self.uc.data
        for t in self.times:
            capacity = gp.LinExpr()
            for i in range(d.n_units):
                u, v = x[key("u", i, t)], x[key("v", i, t)]
                sr, dr = float(d.pmax[i]-d.pstart[i]), float(d.pmax[i]-d.pshut[i])
                capacity += float(d.pmax[i])*u-sr*v
                if t+1 in self.times:
                    w = x[key("w", i, t+1)]
                    capacity -= dr*w
                    if d.min_up[i] == 1:
                        z = m.addVar(lb=0., ub=1.)
                        m.addConstr(z <= v)
                        m.addConstr(z <= w)
                        m.addConstr(z >= v+w-1.)
                        capacity += min(sr, dr)*z
            m.addConstr(capacity >= (1.+d.reserve_fraction)*max(s.load[t] for s in self.uc.scenarios))
        for cut in cuts:
            expr = cut.c + gp.quicksum(v*x[k] for k, v in cut.a.items())
            m.addConstr(expr <= q if cut.feasible else expr <= 0)
        m.setObjective(q + gp.quicksum(costs[k]*x[k] for k in self.bkeys))
        m.optimize()
        value = float(m.ObjVal) if m.Status == gp.GRB.OPTIMAL else None
        m.dispose()
        return value


def grid_bounds(lower, upper, step):
    if not all(math.isfinite(v) for v in (lower, upper, step)) or step <= 0:
        raise ValueError("finite encoding bounds and positive step required")
    if lower > upper + 1e-5:
        raise ValueError(f"inconsistent objective bounds: {lower} > {upper}")
    return math.floor(lower/step)*step, math.ceil(upper/step)*step


class CompactQubo:
    def __init__(self, block, costs, cuts, lower, upper, config):
        self.block, self.costs = block, costs
        self.kw = get_kaiwu()
        self.model = self.kw.core.QuboModel()
        self.config = config
        self.fixed = block.fixed
        self.vars = {k: self.kw.core.Binary(name(k)) for k in block.bkeys if k not in self.fixed}
        self.lower, self.upper = grid_bounds(lower, upper, config.theta_step)
        steps = int(round((self.upper-self.lower)/config.theta_step))
        self.weights = []
        if steps:
            n = int(math.ceil(math.log2(steps+1)))
            self.weights = [1 << j for j in range(n-1)] + [steps-((1 << (n-1))-1)]
        self.zvars = {f"z_{j}": config.theta_step*w for j, w in enumerate(self.weights)}
        z = self.lower + sum(w*self.kw.core.Binary(n) for n, w in self.zvars.items())
        self.model.set_objective(z/block.scale)
        self.energy_unit = config.theta_step/block.scale
        for j, row in enumerate(block.integer_rows):
            expr = row.c + sum(v*self.bit(k) for k, v in row.a.items())
            if row.equality:
                self.penalty(expr**2, f"hard_{j}")
            elif row.label == "exclusive":
                ks = list(row.a)
                self.penalty(self.bit(ks[0])*self.bit(ks[1]), f"hard_{j}")
            elif row.label == "min_up":
                v = next(k for k in row.a if k[0] == "v")
                u = next(k for k in row.a if k[0] == "u")
                self.penalty(self.bit(v)*(1-self.bit(u)), f"hard_{j}")
            elif row.label == "min_down":
                ks = list(row.a)
                self.penalty(self.bit(ks[0])*self.bit(ks[1]), f"hard_{j}")
        self.capacity()
        qlo, _ = block.objective_bounds(costs, True)
        for j, cut in enumerate([Cut({}, qlo)] + list(cuts)):
            a = dict(cut.a)
            c = cut.c
            if cut.feasible:
                for k in block.bkeys:
                    a[k] = a.get(k, 0.) + costs[k]
                c -= self.lower
            for k, v in self.fixed.items():
                c += a.pop(k, 0.)*v
            exact = {n: -Fraction(str(w)) for n, w in self.zvars.items()} if cut.feasible else None
            iq = _quantize_inequality({name(k): v for k, v in a.items()}, c,
                                      config.theta_step if cut.feasible else 1e-3, exact)
            self.inequality(iq, f"cut_{j}", block.scale if cut.feasible else config.reserve_scale)
        variables = self.model.get_variables()
        self.names = list(variables)
        indices = [variables[n] for n in self.names]
        self.matrix = np.asarray(self.model.get_matrix(), dtype=float)[np.ix_(indices, indices)]

    def bit(self, k):
        return self.fixed[k] if k in self.fixed else self.vars[k]

    def penalty(self, expr, label):
        if isinstance(expr, (int, float, np.number)):
            if abs(expr) > 1e-8:
                raise ValueError(f"inconsistent fixed constraint {label}")
            return
        self.model.add_constraint(expr, name=label, constr_type="hard", penalty=self.config.penalty)

    def inequality(self, iq, label, scale):
        lo = iq.constant + sum(min(0, v) for v in iq.coefficients.values())
        hi = iq.constant + sum(max(0, v) for v in iq.coefficients.values())
        if hi <= 0:
            return
        slack = self.kw.core.Integer(f"_slack_{label}", 0, -lo) if lo < 0 else 0
        expr = self.kw.core.BinaryExpression(coefficient={(k,): v for k, v in iq.coefficients.items() if v}, offset=iq.constant)
        self.penalty(((expr+slack)*float(iq.step/Fraction(str(scale))))**2, label)

    def capacity(self):
        d, uc = self.block.uc.data, self.block.uc
        for t in self.block.times:
            capacity = {}
            for i in range(d.n_units):
                u, v = key("u", i, t), key("v", i, t)
                pm = Fraction(str(d.pmax[i]))
                sr = pm-Fraction(str(d.pstart[i]))
                dr = pm-Fraction(str(d.pshut[i]))
                capacity[name(u)], capacity[name(v)] = pm, -sr
                if t+1 in self.block.times:
                    w = key("w", i, t+1)
                    capacity[name(w)] = -dr
                    if d.min_up[i] == 1:
                        an = f"and_{i}_{t}"
                        z = self.kw.core.Binary(an)
                        vb, wb = self.bit(v), self.bit(w)
                        self.penalty(vb*wb-2*vb*z-2*wb*z+3*z, an)
                        capacity[an] = min(sr, dr)
            req = (1+Fraction(str(d.reserve_fraction)))*max(Fraction(str(s.load[t])) for s in uc.scenarios)
            # Substitute fixed variables before recomputing the integer lattice.
            for k, v in self.fixed.items():
                req -= capacity.pop(name(k), Fraction(0))*v
            iq = _capacity_inequality(capacity, req)
            self.inequality(iq, f"reserve_{t}", self.config.reserve_scale)

    def decode(self, bits):
        values = dict(zip(self.names, map(int, bits)))
        point = {**self.fixed, **{k: values.get(name(k), 0) for k in self.vars}}
        z = self.lower + sum(w*values.get(n, 0) for n, w in self.zvars.items())
        return point, z-sum(self.costs[k]*point[k] for k in self.block.bkeys)
