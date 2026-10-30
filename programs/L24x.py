def propose(context, rng):
    n, h = context['n_units'], context['horizon']
    if n < 2 or not h:
        return []
    u, p = context['u'], context['p']
    mn, mx = context['pmin'], context['pmax']
    su, sd = context['pstart'], context['pshut']
    ru, rd = context['ramp_up'], context['ramp_down']
    up, down = context['min_up'], context['min_down']
    ion, age0, ip = context['initial_on'], context['initial_duration'], context['initial_output']
    ca, cb, cc = context['cost_a'], context['cost_b'], context['cost_c']
    cs, cd = context['startup_cost'], context['shutdown_cost']
    scenarios = context['scenario_loads']
    minimum = [min(s[t] for s in scenarios) for t in range(h)]
    maximum = [max(s[t] for s in scenarios) for t in range(h)]
    required = [(1.0 + context['reserve_fraction']) * x for x in maximum]
    eps, cache = 0.000001, {}

    def profile(i, row):
        key = (i, row)
        if key in cache:
            return cache[key]
        cache[key] = None
        state, age, fee = ion[i], age0[i], 0.0
        cap, lo, hi = [0.0] * h, [0.0] * h, [0.0] * h
        if state and not row[0] and ip[i] > sd[i] + eps:
            return None
        for t in range(h):
            if row[t] != state:
                if age < (up[i] if state else down[i]) or (state and t == h - 1):
                    return None
                fee += cs[i] if row[t] else cd[i]
                state, age = row[t], 0
            age += 1
            if row[t]:
                previous = ion[i] if t == 0 else row[t - 1]
                ceiling = mx[i]
                if not previous:
                    ceiling = min(ceiling, su[i])
                if t + 1 < h and not row[t + 1]:
                    ceiling = min(ceiling, sd[i])
                cap[t], lo[t], hi[t] = ceiling, mn[i], ceiling
                if previous:
                    lo[t] = max(lo[t], (ip[i] if t == 0 else lo[t - 1]) - rd[i])
                    hi[t] = min(hi[t], (ip[i] if t == 0 else hi[t - 1]) + ru[i])
                if lo[t] > hi[t] + eps:
                    return None
        for t in range(h - 2, -1, -1):
            if row[t] and row[t + 1]:
                lo[t] = max(lo[t], lo[t + 1] - ru[i])
                hi[t] = min(hi[t], hi[t + 1] + rd[i])
                if lo[t] > hi[t] + eps:
                    return None
        cache[key] = (cap, lo, hi, fee)
        return cache[key]

    base = [profile(i, u[i]) for i in range(n)]
    if any(v is None for v in base):
        return []
    totals = [[sum(v[k][t] for v in base) for t in range(h)] for k in range(3)]
    price = []
    for t in range(h):
        free = sorted(cb[i] + 2.0 * ca[i] * p[i][t] for i in range(n)
                      if u[i][t] and mn[i] + eps < p[i][t] < base[i][2][t] - eps)
        running = [cb[i] + 2.0 * ca[i] * p[i][t] for i in range(n)
                   if u[i][t] and p[i][t] > mn[i] + eps]
        price.append(free[len(free) // 2] if free else max([0.0] + running))
    net = [[(ca[i] * p[i][t] + cb[i] - price[t]) * p[i][t] + cc[i] * u[i][t]
            for t in range(h)] for i in range(n)]

    def operating(i, t, lo, hi):
        q = p[i][t] if u[i][t] else ((price[t] - cb[i]) / (2.0 * ca[i])
             if ca[i] > 0.000000000001 else (hi if price[t] >= cb[i] else lo))
        q = max(lo, min(hi, q))
        return (ca[i] * q + cb[i] - price[t]) * q + cc[i]

    def credit(i, row, v):
        return base[i][3] - v[3] + sum(net[i][t] - (operating(i, t, v[1][t], v[2][t])
                                                   if row[t] else 0.0) for t in range(h))

    rates = [[operating(i, t, mn[i], mx[i]) for t in range(h)] for i in range(n)]
    penalty = 4.0 * max([1.0] + price)
    groups = {}
    fields = ('pmin', 'pmax', 'pstart', 'pshut', 'ramp_up', 'ramp_down', 'min_up', 'min_down',
              'initial_on', 'initial_duration', 'initial_output', 'cost_a', 'cost_b', 'cost_c',
              'startup_cost', 'shutdown_cost')
    for i in range(n):
        signature = tuple(context[name][i] for name in fields)
        if signature not in groups:
            groups[signature] = []
        groups[signature].append(i)
    classes = list(groups.values())

    def canonical(rows):
        return tuple(tuple(sorted(rows.get(i, u[i]) for i in group)) for group in classes)

    def defects(cap, lo, hi):
        return [max(0.0, required[t] - cap[t]) + max(0.0, lo[t] - minimum[t])
                + max(0.0, maximum[t] - hi[t]) for t in range(h)]

    def reachable(rows, changed):
        states = [rows.get(i, u[i]) for i in range(n)]
        bounds = [changed.get(i, base[i]) for i in range(n)]
        for demand in scenarios:
            low = [[bounds[i][1][t] for i in range(n)] for t in range(h)]
            high = [[bounds[i][2][t] for i in range(n)] for t in range(h)]
            for direction in (1, -1, 1):
                for t in (range(h) if direction == 1 else range(h - 1, -1, -1)):
                    lo, hi = low[t], high[t]
                    for i in range(n):
                        if direction == 1 and states[i][t] and (ion[i] if t == 0 else states[i][t - 1]):
                            lo[i] = max(lo[i], (ip[i] if t == 0 else low[t - 1][i]) - rd[i])
                            hi[i] = min(hi[i], (ip[i] if t == 0 else high[t - 1][i]) + ru[i])
                        elif direction == -1 and t + 1 < h and states[i][t] and states[i][t + 1]:
                            lo[i] = max(lo[i], low[t + 1][i] - ru[i])
                            hi[i] = min(hi[i], high[t + 1][i] + rd[i])
                    total_lo, total_hi = sum(lo), sum(hi)
                    if total_lo > demand[t] + eps or total_hi + eps < demand[t] or any(lo[i] > hi[i] + eps for i in range(n)):
                        return False
                    low[t] = [max(lo[i], demand[t] - total_hi + hi[i]) for i in range(n)]
                    high[t] = [min(hi[i], demand[t] - total_lo + lo[i]) for i in range(n)]
        return True

    def synthesize(i, oldrow, oldprofile, capacity, low, high, pivot, forced, left, right, wave, strength, pin):
        left, right = min(left, pivot), max(right, pivot + 1)
        must = [oldrow[t] if t < left or t >= right else None for t in range(h)]
        must[pivot] = forced
        if pin is not None:
            if pin[0] == pivot and pin[1] != forced:
                return None
            must[pin[0]] = pin[1]
        scale = cc[i] + cs[i] / max(1, h)
        distance, tilt = rng.uniform(0.01, 0.12) * scale, rng.uniform(-0.35, 0.35) * scale
        charge, pressure = rng.uniform(0.85, 1.15), penalty * strength * rng.uniform(0.8, 1.2)
        residual = [[v[t] - oldprofile[k][t] for t in range(h)]
                    for k, v in enumerate((capacity, low, high))]

        def loss(t, cap, lo, hi):
            return (max(0.0, required[t] - residual[0][t] - cap)
                    + max(0.0, residual[1][t] + lo - minimum[t])
                    + max(0.0, maximum[t] - residual[2][t] - hi))

        onbad = [loss(t, mx[i], mn[i], mx[i]) for t in range(h)]
        costs, blocked = [[0.0], [0.0]], [[0], [0]]
        for bit in (0, 1):
            for t in range(h):
                value = pressure * (onbad[t] if bit else loss(t, 0.0, 0.0, 0.0))
                value += distance * (bit != oldrow[t])
                if bit:
                    value += rates[i][t] + tilt * wave[t]
                costs[bit].append(costs[bit][-1] + value)
                blocked[bit].append(blocked[bit][-1] + (must[t] is not None and must[t] != bit))
        best = [[None, None] for t in range(h + 1)]
        for bit in (0, 1):
            if bit != ion[i] and age0[i] < (up[i] if ion[i] else down[i]):
                continue
            if ion[i] and not bit and ip[i] > sd[i] + eps:
                continue
            best[0][bit] = (0.0, ())
        finish = None
        for start in range(h):
            for bit in (0, 1):
                parent = best[start][bit]
                if parent is None or (not bit and start == h - 1 and (start > 0 or ion[i])):
                    continue
                continuing = start == 0 and bit == ion[i]
                duration = max(0, (up[i] if bit else down[i]) - (age0[i] if continuing else 0))
                fee = 0.0 if continuing else (cs[i] * charge if bit else cd[i])
                for end in range(start + 1, h + 1):
                    if (end < h and end - start < duration) or blocked[bit][end] != blocked[bit][start]:
                        continue
                    value = parent[0] + fee + costs[bit][end] - costs[bit][start]
                    valid = True
                    if bit:
                        for t in ((start,) if end == start + 1 else (start, end - 1)):
                            ceiling = mx[i]
                            if t == start and not continuing:
                                ceiling = min(ceiling, su[i])
                            if t == end - 1 and end < h:
                                ceiling = min(ceiling, sd[i])
                            lo, hi = mn[i], ceiling
                            if continuing:
                                lo = max(lo, ip[i] - (t + 1) * rd[i])
                                hi = min(hi, ip[i] + (t + 1) * ru[i])
                            else:
                                hi = min(hi, min(mx[i], su[i]) + (t - start) * ru[i])
                            if end < h:
                                hi = min(hi, sd[i] + (end - 1 - t) * rd[i])
                            if lo > hi + eps:
                                valid = False
                                break
                            value += pressure * (loss(t, ceiling, lo, hi) - onbad[t])
                            value += operating(i, t, lo, hi) - rates[i][t]
                    if not valid:
                        continue
                    path = parent[1] + (bit,) * (end - start)
                    if end == h:
                        if finish is None or value < finish[0]:
                            finish = (value, path)
                    elif best[end][1 - bit] is None or value < best[end][1 - bit][0]:
                        best[end][1 - bit] = (value, path)
        return None if finish is None else finish[1]

    anchor, seen, records, draws = canonical({}), {}, [], 0
    edges = [t for t in range(h) if any(u[i][t] != (ion[i] if t == 0 else u[i][t - 1]) for i in range(n))]
    burdens = [sum(max(0.0, net[i][t]) for i in range(n)) for t in range(h)]
    economic_hours = sorted(range(h), key=lambda t: burdens[t], reverse=True)
    for trial in range(7):
        if draws >= 40:
            break
        mode = rng.random()
        if edges and mode < 0.45:
            focus = rng.choice(edges)
        elif mode < 0.8:
            focus = economic_hours[min(rng.randrange(h), rng.randrange(h))]
        else:
            focus = rng.randrange(h)
        left, right = rng.randint(0, focus), rng.randint(focus + 1, h)
        wave = [1.0 if left <= t < right else -0.5 for t in range(h)]
        if rng.random() < 0.6:
            left, right = 0, h
        target = min(n, rng.choice((3, 3, 4, 6, 8, 10)))
        rows, changed, credits, visits, pins = {}, {}, {}, {}, {}
        value = 0.0
        capacity, low, high = totals[0].copy(), totals[1].copy(), totals[2].copy()
        for step in range(target + 2):
            if draws >= 40:
                break
            gaps = defects(capacity, low, high)
            bad = sum(gaps)
            if step >= target and bad <= eps:
                break
            pivot = max(range(h), key=lambda t: gaps[t]) if bad > eps else (focus if rng.random() < 0.8 else rng.randrange(h))
            pool = []
            for i in range(n):
                if visits.get(i, 0) >= 2:
                    continue
                if i in rows and (bad <= eps or step < 2):
                    continue
                if i not in rows and len(rows) >= target:
                    continue
                oldrow, oldprofile = rows.get(i, u[i]), changed.get(i, base[i])
                forced, helpvalue = 1 - oldrow[pivot], 0.0
                if bad > eps:
                    cap = capacity[pivot] - oldprofile[0][pivot]
                    lo, hi = low[pivot] - oldprofile[1][pivot], high[pivot] - oldprofile[2][pivot]
                    offgap = max(0.0, required[pivot] - cap) + max(0.0, lo - minimum[pivot]) + max(0.0, maximum[pivot] - hi)
                    ongap = max(0.0, required[pivot] - cap - mx[i]) + max(0.0, lo + mn[i] - minimum[pivot]) + max(0.0, maximum[pivot] - hi - mx[i])
                    if abs(ongap - offgap) > eps:
                        forced = int(ongap < offgap)
                    helpvalue = gaps[pivot] - min(offgap, ongap)
                if forced == oldrow[pivot]:
                    continue
                if i in pins and pins[i][0] == pivot and pins[i][1] != forced:
                    continue
                economic = operating(i, pivot, oldprofile[1][pivot], oldprofile[2][pivot]) if oldrow[pivot] else -rates[i][pivot]
                pool.append((penalty * helpvalue + economic, i, forced, helpvalue))
            helpful = [entry for entry in pool if entry[3] > eps]
            if bad > eps and helpful:
                pool = helpful
            pool.sort(key=lambda entry: entry[0], reverse=True)
            order = sorted(range(len(pool)), key=lambda k: (k + 1) * rng.uniform(0.08, 1.0))
            chosen, bestmark = None, 0.0
            strength = 0.15 if step == 0 else min(1.4, 0.35 + 1.2 * step / max(1, target - 1))
            for position in order[:1 if step == 0 else 2]:
                if draws >= 40:
                    break
                draws += 1
                merit, i, forced, helpvalue = pool[position]
                oldrow, oldprofile = rows.get(i, u[i]), changed.get(i, base[i])
                row = synthesize(i, oldrow, oldprofile, capacity, low, high, pivot, forced,
                                 left, right, wave, strength, pins.get(i))
                if row is None or row == oldrow:
                    continue
                v = profile(i, row)
                if v is None:
                    continue
                nexts = [[old[t] + v[k][t] - oldprofile[k][t] for t in range(h)]
                         for k, old in enumerate((capacity, low, high))]
                remaining = sum(defects(nexts[0], nexts[1], nexts[2]))
                newcredit = credit(i, row, v)
                gain = newcredit - credits.get(i, 0.0)
                mark = gain - penalty * strength * remaining + rng.uniform(-0.08, 0.08) * max(1.0, abs(gain))
                if chosen is None or mark > bestmark:
                    chosen = (i, row, v, newcredit, gain, nexts, remaining, forced)
                    bestmark = mark
            if chosen is None:
                break
            i, row, v, newcredit, gain, nexts, bad, forced = chosen
            capacity, low, high = nexts
            visits[i] = visits.get(i, 0) + 1
            if step == 0:
                pins[i] = (pivot, forced)
            if row == u[i]:
                rows = {j: r for j, r in rows.items() if j != i}
                changed = {j: b for j, b in changed.items() if j != i}
                credits = {j: c for j, c in credits.items() if j != i}
            else:
                rows[i], changed[i], credits[i] = row, v, newcredit
            value += gain
            if len(rows) >= 2 and bad <= eps:
                key = canonical(rows)
                if key != anchor and key not in seen:
                    seen[key] = True
                    records.append((value, rows.copy(), changed.copy()))
    records.sort(key=lambda entry: entry[0], reverse=True)
    order = sorted(range(len(records)), key=lambda k: (k + 1) * rng.uniform(0.1, 1.0))
    preferred = []
    for lower, upper in ((2, 4), (5, 8), (9, 10)):
        choices = [k for k in order if lower <= len(records[k][1]) <= upper]
        if choices:
            preferred.append(choices[0])
    used = {tuple(sorted(records[k][1])): True for k in preferred}
    for k in order:
        tag = tuple(sorted(records[k][1]))
        if tag not in used:
            used[tag] = True
            preferred.append(k)
    preferred.extend(k for k in order if k not in preferred)
    result = []
    for k in preferred[:8]:
        value, rows, changed = records[k]
        if reachable(rows, changed):
            result.append(make_candidate(u, rows))
            if len(result) == 4:
                break
    return result

