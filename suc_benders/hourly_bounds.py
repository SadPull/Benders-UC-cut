"""Hourly recourse bounds (fuel plus cold premium); relax inter-hour continuous rows."""
import hashlib
from types import SimpleNamespace as NS

import numpy as np

from .benders import dispatch


class HourlyBounds:
    def __init__(self, full):
        self.full, self.parts, self.cuts, self.identities = full, [], [], set()
        X, Y = full.binary, full.continuous
        counts = full.A[:, Y].getnnz(axis=1)
        for hour in full.times:
            variables = Y[full.time[Y] == hour]
            local_counts = full.A[:, variables].getnnz(axis=1)
            rows = np.flatnonzero((local_counts > 0) & (local_counts == counts))
            indices = np.flatnonzero(full.A[rows][:, X].getnnz(axis=0))
            columns = np.r_[X[indices], variables]
            nx = len(indices)
            b = NS(A=full.A[rows][:, columns], rhs=full.rhs[rows], eq=full.eq[rows],
                obj=np.r_[np.zeros(nx), full.obj[variables]], q=np.r_[np.zeros(nx), full.q[variables]],
                lb=full.lb[columns], ub=full.ub[columns], binary=np.arange(nx),
                continuous=np.arange(nx, len(columns)), constant=0.,
                audit_cuts=bool(getattr(full, 'audit_cuts', False)) or
                    getattr(getattr(full, "d", None), "model_profile", "") == "azema_sms")
            self.parts.append(NS(hour=int(hour), b=b, indices=indices, variables=variables, cache=set()))

    def recourse(self, point):
        return np.array([self.full.obj[p.variables] @ point[p.variables]
                         + self.full.q[p.variables] @ point[p.variables]**2 for p in self.parts])

    def lower(self, x):
        values = [float(np.minimum(p.b.obj[p.b.continuous]*p.b.lb[p.b.continuous],
                                   p.b.obj[p.b.continuous]*p.b.ub[p.b.continuous]).sum()) for p in self.parts]
        for cut in self.cuts:
            hour = cut["hour"]-1
            values[hour] = max(values[hour], cut["alpha"] + np.asarray(cut["beta"]) @ x[cut["binary_indices"]])
        return float(sum(values))

    def observe(self, x, source, actual, remaining):
        for part, value in zip(self.parts, actual):
            local = x[part.indices]
            key = local.astype(np.uint8).tobytes()
            if key in part.cache:
                continue
            if remaining() <= 0:
                raise TimeoutError
            r = dispatch(part.b, local, remaining(), certified=True)
            if remaining() <= 0 or r.status == "TIME_LIMIT":
                raise TimeoutError
            if not r.feasible or r.cut is None or not r.cut.certified:
                raise RuntimeError("Hourly relaxation of a validated dispatch failed")
            cut = r.cut
            if cut.alpha + cut.beta @ local > value + 1e-5:
                raise RuntimeError("Hourly lower bound exceeds validated hourly recourse cost")
            part.cache.add(key)
            identity = hashlib.sha256(np.r_[part.hour, cut.alpha, cut.beta].astype("<f8").tobytes()).hexdigest()
            if identity in self.identities:
                continue
            self.identities.add(identity)
            row = dict(cut_id=identity, source=source, hour=part.hour+1,
                       binary_indices=part.indices.tolist(), alpha=float(cut.alpha), beta=cut.beta.tolist())
            if hasattr(cut, "proof"):
                row["proof"] = cut.proof
            self.cuts.append(row)
            yield row
