"""Fixed preprocessing and candidate policies for the pure-SA gap experiment."""
from fractions import Fraction


def interval_fixed_binary(rows, initial):
    """Fix only values proved by exact interval infeasibility of the alternative."""
    fixed = dict(initial)
    changed = True
    while changed:
        changed = False
        for row in rows:
            free = {k: Fraction(str(v)) for k, v in row.a.items() if k not in fixed}
            constant = Fraction(str(row.c)) + sum(Fraction(str(v))*fixed[k]
                                                 for k, v in row.a.items() if k in fixed)
            lo = constant + sum(min(0, v) for v in free.values())
            hi = constant + sum(max(0, v) for v in free.values())
            def possible(lower, upper):
                return lower <= 0 <= upper if row.equality else lower <= 0
            if not possible(lo, hi):
                raise ValueError("inconsistent binary hard-row interval")
            for k, v in free.items():
                valid = [bit for bit in (0, 1) if possible(
                    lo-min(0, v)+v*bit, hi-max(0, v)+v*bit)]
                if len(valid) == 1:
                    fixed[k] = valid[0]
                    changed = True
    return fixed


def active_cuts(cuts, best=None, samples=(), limit=16):
    """Choose encoded cuts only; the complete original list remains untouched."""
    representatives = {}
    for i, cut in enumerate(cuts):
        signature = cut.feasible, tuple(sorted((k, v) for k, v in cut.a.items() if v))
        previous = representatives.get(signature)
        if previous is None or cut.c >= previous[1].c:
            representatives[signature] = (i, cut)
    indexed = sorted(representatives.values())
    if len(indexed) <= limit:
        return [c for _, c in indexed]
    selected = []
    def add(item):
        if item[0] not in {i for i, _ in selected} and len(selected) < limit:
            selected.append(item)
    optimality = [item for item in indexed if item[1].feasible]
    if best is not None and optimality:
        add(max(optimality, key=lambda item: (item[1].value(best), item[0])))
    for item in reversed(indexed[-4:]):
        add(item)
    def violation(item):
        i, cut = item
        scale = max(1., abs(cut.c), sum(abs(v) for v in cut.a.values()))
        maximum = max((max(0., cut.value(x)-(theta if cut.feasible else 0.))/scale
                       for x, theta in samples), default=0.)
        return maximum, i
    for item in sorted(indexed, key=violation, reverse=True):
        add(item)
    return [c for _, c in sorted(selected)]
