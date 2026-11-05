"""Certified cut retention and invalidation of the associated dispatch cache."""
import numpy as np


def _cache_key(x):
    return x.astype(np.uint8).tobytes()


class CutPool:
    def __init__(self, binary_count, diagnostics):
        self.binary_count, self.diagnostics = binary_count, diagnostics
        self.cuts, self.cache, self.owners = [], {}, {}
        self.totals = dict(optimality=0, feasibility=0)

    def add(self, cut, cache_key=None, *, protected=None, limit):
        if not cut.certified or cut.beta.shape != (self.binary_count,) or not (
                np.isfinite(cut.alpha) and np.isfinite(cut.beta).all()):
            raise ValueError("Whole-day search requires a finite certified full-day cut")
        self.cuts.append(cut)
        self.owners[id(cut)] = cache_key
        self.totals["optimality" if cut.optimality else "feasibility"] += 1
        if len(self.cuts) > limit:
            index = next(k for k, old in enumerate(self.cuts) if old is not protected)
            dropped = self.cuts.pop(index)
            owner = self.owners.pop(id(dropped))
            if owner in self.cache:
                del self.cache[owner]
                self.diagnostics['cache_invalidations'] += 1
            self.diagnostics["cuts_dropped"] += 1
        self.diagnostics["stored_cuts"] = len(self.cuts)
        return id(cut) in self.owners
