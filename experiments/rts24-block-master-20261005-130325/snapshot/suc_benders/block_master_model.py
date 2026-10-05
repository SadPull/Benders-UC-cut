"""Time-partitioned, fixed-price master models for the SA microbenchmark."""
from __future__ import annotations

from collections import Counter
import time

from .paper_model import Block, CompactQubo, binary, fixed_binary, key, name
from .common import require_gurobi


TOL = 1e-6


def partition_rows(indexed, hours):
    if hours <= 0 or indexed.data.horizon % hours:
        raise ValueError("block length must divide the horizon")
    local = [[] for _ in range(indexed.data.horizon // hours)]
    cross = []
    for i, row in enumerate(indexed.rows):
        owners = {k[2] // hours for k in row.a}
        if len(owners) == 1:
            local[next(iter(owners))].append(i)
        else:
            cross.append(i)
    return local, cross


class TemporalBlock(Block):
    """Reuse dispatch and LP logic, retaining original global time indices."""
    def __init__(self, indexed, times, row_ids, costs, constant=0.):
        self.uc = indexed
        self.times = tuple(times)
        self.keys = [k for k in indexed.bounds if k[2] in self.times]
        self.bkeys = [k for k in self.keys if binary(k)]
        self.ckeys = [k for k in self.keys if not binary(k)]
        self.row_ids = list(row_ids)
        self.rows = [indexed.rows[i] for i in row_ids]
        self.integer_rows = [r for r in self.rows if all(binary(k) for k in r.a)]
        self.fixed = fixed_binary(self.integer_rows)
        self.costs = {k: float(costs[k]) for k in self.keys}
        self.constant = float(constant)
        # Preserve the existing physical cost scaling rule; do not tune penalties.
        self.scale = max(1., sum(abs(indexed.linear[k])*indexed.bounds[k][1] for k in self.keys)
                         + sum(indexed.quadratic.get(k, 0.)*indexed.bounds[k][1]**2 for k in self.keys))

    def capacity_violation(self, point):
        d = self.uc.data
        worst = 0.
        for t in self.times:
            capacity = 0.
            for i in range(d.n_units):
                startup = d.pmax[i]*point[key("u", i, t)] - (d.pmax[i]-d.pstart[i])*point[key("v", i, t)]
                shutdown = d.pmax[i]*point[key("u", i, t)]
                if t+1 in self.times:
                    shutdown -= (d.pmax[i]-d.pshut[i])*point[key("w", i, t+1)]
                capacity += min(startup, shutdown)
            demand = (1+d.reserve_fraction)*max(s.load[t] for s in self.uc.scenarios)
            worst = max(worst, float(demand-capacity))
        return worst

    def discrete_checks(self, point):
        violations = Counter()
        maximum = 0.
        for row in self.integer_rows:
            value = abs(row.value(point)) if row.equality else max(0., row.value(point))
            if value > TOL:
                violations[row.label] += 1
            maximum = max(maximum, value)
        capacity = self.capacity_violation(point)
        return dict(feasible=not violations and capacity <= TOL,
                    violations=dict(violations), maximum_row_violation=maximum,
                    capacity_violation_mw=capacity)


def priced_blocks(indexed, hours, multipliers):
    local, cross = partition_rows(indexed, hours)
    if len(multipliers) != len(indexed.rows):
        raise ValueError("one multiplier entry per original row required")
    costs = dict(indexed.linear)
    constants = [0.] * len(local)
    for i in cross:
        row, price = indexed.rows[i], float(multipliers[i])
        if not row.equality and price < -TOL:
            raise ValueError("inequality multiplier must be nonnegative")
        if not row.equality:
            price = max(0., price)
        for k, value in row.a.items():
            costs[k] += price*value
        constants[min(k[2] // hours for k in row.a)] += price*row.c
    blocks = [TemporalBlock(indexed, range(j*hours, (j+1)*hours), ids, costs, constants[j])
              for j, ids in enumerate(local)]
    return blocks, cross


def continuous_relaxation(indexed, deadline):
    gp = require_gurobi()
    model = gp.Model("block_benchmark_continuous_relaxation")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(.001, min(60., deadline-time.perf_counter()))
    x = {k: model.addVar(lb=lo, ub=hi, name=name(k)) for k, (lo, hi) in indexed.bounds.items()}
    constraints = []
    for row in indexed.rows:
        expression = gp.quicksum(v*x[k] for k, v in row.a.items())
        constraints.append(model.addConstr(expression == -row.c if row.equality else expression <= -row.c))
    model.setObjective(gp.quicksum(indexed.linear[k]*x[k] + indexed.quadratic.get(k, 0.)*x[k]*x[k] for k in x))
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        status = model.Status
        model.dispose()
        raise RuntimeError(f"continuous pricing relaxation not optimal: {status}")
    # Gurobi Pi is the sensitivity to RHS; our Lagrangian uses residual lhs-rhs.
    multipliers = [float(-c.Pi) if r.equality else max(0., float(-c.Pi))
                   for r, c in zip(indexed.rows, constraints)]
    result = dict(objective=float(model.ObjVal), multipliers=multipliers,
                  point={k: float(v.X) for k, v in x.items()}, seconds=float(model.Runtime))
    model.dispose()
    return result


def solve_master(block, cuts, deadline, *, integral=True):
    """Native master oracle; constant restored, but never used as an SA start."""
    gp = require_gurobi()
    model = gp.Model("block_benchmark_integer_master" if integral else "block_benchmark_lp_master")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.Seed = 7
    model.Params.MIPGap = 1e-6
    model.Params.TimeLimit = max(.001, min(60., deadline-time.perf_counter()))
    x = {k: model.addVar(lb=block.fixed.get(k, 0.), ub=block.fixed.get(k, 1.),
                        vtype=gp.GRB.BINARY if integral else gp.GRB.CONTINUOUS, name=name(k)) for k in block.bkeys}
    qlo, _ = block.objective_bounds(block.costs, True)
    q = model.addVar(lb=qlo, name="recourse")
    for row in block.integer_rows:
        expression = gp.quicksum(v*x[k] for k, v in row.a.items())
        model.addConstr(expression == -row.c if row.equality else expression <= -row.c)
    d = block.uc.data
    for t in block.times:
        capacity = gp.LinExpr()
        for i in range(d.n_units):
            u, v = x[key("u", i, t)], x[key("v", i, t)]
            sr, dr = float(d.pmax[i]-d.pstart[i]), float(d.pmax[i]-d.pshut[i])
            capacity += float(d.pmax[i])*u-sr*v
            if t+1 in block.times:
                w = x[key("w", i, t+1)]
                capacity -= dr*w
                if d.min_up[i] == 1:
                    z = model.addVar(lb=0., ub=1.)
                    model.addConstr(z <= v)
                    model.addConstr(z <= w)
                    model.addConstr(z >= v+w-1.)
                    capacity += min(sr, dr)*z
        model.addConstr(capacity >= (1+d.reserve_fraction)*max(s.load[t] for s in block.uc.scenarios))
    for cut in cuts:
        expression = cut.c + gp.quicksum(v*x[k] for k, v in cut.a.items())
        model.addConstr(expression <= q if cut.feasible else expression <= 0.)
    model.setObjective(q + gp.quicksum(block.costs[k]*x[k] for k in x) + block.constant)
    model.optimize()
    feasible = model.SolCount > 0
    bound = float(model.ObjBound) if model.IsMIP else float(model.ObjVal) if feasible else None
    if bound is not None and abs(bound) >= gp.GRB.INFINITY:
        bound = None
    result = dict(status=int(model.Status), certified=model.Status == gp.GRB.OPTIMAL,
        objective=float(model.ObjVal) if feasible else None, lower_bound=bound,
        mip_gap=float(model.MIPGap) if model.IsMIP and feasible else None,
        point={k: int(round(v.X)) if integral else float(v.X) for k, v in x.items()} if feasible else None,
        theta=float(q.X) if feasible else None, seconds=float(model.Runtime))
    model.dispose()
    return result


def fixed_master_value(block, cuts, point):
    """Classical recourse-variable recomputation; leaves all u/v/w unchanged."""
    if not block.discrete_checks(point)["feasible"]:
        return None
    if any(c.value(point) > TOL for c in cuts if not c.feasible):
        return None
    qlo, _ = block.objective_bounds(block.costs, True)
    theta = max([qlo] + [c.value(point) for c in cuts if c.feasible])
    return float(theta + sum(block.costs[k]*point[k] for k in block.bkeys) + block.constant)


class RecordedQubo(CompactQubo):
    """Same encoding, with exact integer residuals retained for auditing."""
    def __init__(self, *args, **kwargs):
        self.inequalities = []
        super().__init__(*args, **kwargs)

    def inequality(self, iq, label, scale):
        lo = iq.constant + sum(min(0, v) for v in iq.coefficients.values())
        hi = iq.constant + sum(max(0, v) for v in iq.coefficients.values())
        if hi <= 0:
            return
        slack = self.kw.core.Integer(f"_slack_{label}", 0, -lo) if lo < 0 else 0
        coefficients = dict(iq.coefficients)
        offset = iq.constant
        if not isinstance(slack, (int, float)):
            offset += int(slack.offset)
            for term, value in slack.coefficient.items():
                if len(term) != 1 or int(value) != value:
                    raise ValueError("unexpected integer slack encoding")
                coefficients[term[0]] = coefficients.get(term[0], 0) + int(value)
        self.inequalities.append(dict(label=label, coefficients=coefficients, constant=offset,
            original_coefficients=dict(iq.coefficients), original_constant=iq.constant,
            step=str(iq.step), rounding_error=iq.rounding_error, scale=scale))
        from fractions import Fraction
        expression = self.kw.core.BinaryExpression(coefficient={(k,): v for k, v in iq.coefficients.items() if v}, offset=iq.constant)
        self.penalty(((expression+slack)*float(iq.step/Fraction(str(scale))))**2, label)

    def encoded_checks(self, bits):
        values = dict(zip(self.names, map(int, bits)))
        point, _ = self.decode(bits)
        hard = self.block.discrete_feasible(point)
        and_ok = True
        for variable in self.names:
            if variable.startswith("and_"):
                _, i, t = variable.split("_")
                i, t = int(i), int(t)
                and_ok &= values[variable] == point[key("v", i, t)]*point[key("w", i, t+1)]
        residuals = {r["label"]: r["constant"] + sum(v*values[k] for k, v in r["coefficients"].items())
                     for r in self.inequalities}
        inequalities_ok = all(r["original_constant"] + sum(v*values[k] for k, v in r["original_coefficients"].items()) <= 0
                              for r in self.inequalities)
        return dict(feasible=bool(hard and and_ok and all(v == 0 for v in residuals.values())),
            quantized_inequalities_feasible=bool(inequalities_ok), and_auxiliaries_valid=bool(and_ok),
            nonzero_integer_residuals={k: v for k, v in residuals.items() if v != 0})


def row_record(row):
    return dict(coefficients=[[list(k), float(v)] for k, v in row.a.items()], constant=row.c,
                equality=row.equality, scale=row.scale, label=row.label)


def point_record(point):
    return [[list(k), float(v)] for k, v in point.items()] if point is not None else None
