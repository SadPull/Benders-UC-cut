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
        key = (i, tuple(row))
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

    def operating(i, t, low, high):
        if u[i][t]:
            q = p[i][t]
        elif ca[i] > 0.000000000001:
            q = (price[t] - cb[i]) / (2.0 * ca[i])
        else:
            q = high if price[t] >= cb[i] else low
        q = max(low, min(high, q))
        return (ca[i] * q + cb[i] - price[t]) * q + cc[i]

    def credit(i, row, v):
        return base[i][3] - v[3] + sum(net[i][t] - (operating(i, t, v[1][t], v[2][t])
                                                   if row[t] else 0.0) for t in range(h))

    groups = {}
    for i in range(n):
        signature = (mn[i], mx[i], su[i], sd[i], ru[i], rd[i], up[i], down[i],
                     ion[i], age0[i], ip[i], ca[i], cb[i], cc[i], cs[i], cd[i])
        if signature not in groups:
            groups[signature] = []
        groups[signature].append(i)
    classes = list(groups.values())

    def canonical(rows):
        return tuple(tuple(sorted(rows.get(i, u[i]) for i in group)) for group in classes)

    def defects(capacity, low, high):
        return [max(0.0, required[t] - capacity[t]) + max(0.0, low[t] - minimum[t])
                + max(0.0, maximum[t] - high[t]) for t in range(h)]

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

    rates = [[operating(i, t, mn[i], mx[i]) for t in range(h)] for i in range(n)]
    penalty = 4.0 * max([1.0] + price)

    def synthesize(i, capacity, low, high, pivot, forced, wave, strength):
        left, right = rng.randint(0, pivot), rng.randint(pivot + 1, h)
        if rng.random() < 0.5:
            left, right = 0, h
        must = [u[i][t] if t < left or t >= right else None for t in range(h)]
        must[pivot] = forced
        scale = cc[i] + cs[i] / max(1, h)
        distance = rng.uniform(0.02, 0.2) * scale
        tilt = rng.uniform(-0.2, 0.2) * scale
        charge = rng.uniform(0.9, 1.1)
        pressure = penalty * strength * rng.uniform(0.7, 1.3)
        residual_cap = [capacity[t] - base[i][0][t] for t in range(h)]
        residual_low = [low[t] - base[i][1][t] for t in range(h)]
        residual_high = [high[t] - base[i][2][t] for t in range(h)]

        def loss(t, cap, lo, hi):
            return (max(0.0, required[t] - residual_cap[t] - cap)
                    + max(0.0, residual_low[t] + lo - minimum[t])
                    + max(0.0, maximum[t] - residual_high[t] - hi))

        onbad = [loss(t, mx[i], mn[i], mx[i]) for t in range(h)]
        costs, blocked = [[0.0], [0.0]], [[0], [0]]
        for bit in (0, 1):
            for t in range(h):
                value = pressure * (onbad[t] if bit else loss(t, 0.0, 0.0, 0.0))
                value += distance * (bit != u[i][t])
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
                if parent is None:
                    continue
                if not bit and start == h - 1 and (start > 0 or ion[i]):
                    continue
                duration = up[i] if bit else down[i]
                continuing = start == 0 and bit == ion[i]
                if continuing:
                    duration = max(0, duration - age0[i])
                fee = (cs[i] * charge if bit else cd[i]) if not continuing else 0.0
                for end in range(start + 1, h + 1):
                    if end < h and end - start < duration:
                        continue
                    if blocked[bit][end] != blocked[bit][start]:
                        continue
                    value = parent[0] + fee + costs[bit][end] - costs[bit][start]
                    valid = True
                    if bit:
                        points = (start,) if end == start + 1 else (start, end - 1)
                        for t in points:
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
                    if end == h:
                        if finish is None or value < finish[0]:
                            finish = (value, parent[1] + (bit,) * (end - start))
                    elif best[end][1 - bit] is None or value < best[end][1 - bit][0]:
                        best[end][1 - bit] = (value, parent[1] + (bit,) * (end - start))
        return None if finish is None else finish[1]

    anchor, seen, records = canonical({}), {}, []
    edges = [t for t in range(h) if any(u[i][t] != (ion[i] if t == 0 else u[i][t - 1]) for i in range(n))]
    draws = 0
    for trial in range(10):
        if draws >= 48:
            break
        focus = rng.randrange(h)
        if edges and rng.random() < 0.35:
            focus = max(0, min(h - 1, rng.choice(edges) + rng.choice((-1, 0, 1))))
        a, b = rng.randint(0, focus), rng.randint(focus + 1, h)
        wave = [1.0 if a <= t < b else -0.5 for t in range(h)]
        rows, changed, value = {}, {}, 0.0
        capacity, low, high = totals[0].copy(), totals[1].copy(), totals[2].copy()
        target = min(n, rng.choice((2, 3, 3, 4)))
        for step in range(target):
            if draws >= 48:
                break
            gaps = defects(capacity, low, high)
            bad = sum(gaps)
            pivot = max(range(h), key=lambda t: gaps[t]) if bad > eps else (focus if rng.random() < 0.7 else rng.randrange(h))
            pool = []
            for i in range(n):
                if i in rows:
                    continue
                forced = 1 - u[i][pivot]
                helpvalue = 0.0
                if bad > eps:
                    cap = capacity[pivot] - base[i][0][pivot]
                    lo = low[pivot] - base[i][1][pivot]
                    hi = high[pivot] - base[i][2][pivot]
                    offgap = max(0.0, required[pivot] - cap) + max(0.0, lo - minimum[pivot]) + max(0.0, maximum[pivot] - hi)
                    ongap = max(0.0, required[pivot] - cap - mx[i]) + max(0.0, lo + mn[i] - minimum[pivot]) + max(0.0, maximum[pivot] - hi - mx[i])
                    if ongap < offgap - eps:
                        forced = 1
                    elif offgap < ongap - eps:
                        forced = 0
                    helpvalue = gaps[pivot] - min(offgap, ongap)
                economic = (net[i][pivot] if u[i][pivot] else -rates[i][pivot]) if forced != u[i][pivot] else 0.0
                pool.append((penalty * helpvalue + economic, i, forced, helpvalue))
            if bad > eps:
                helpful = [entry for entry in pool if entry[3] > eps]
                if helpful:
                    pool = helpful
            if not pool:
                break
            pool.sort(key=lambda entry: entry[0], reverse=True)
            order = sorted(range(len(pool)), key=lambda k: (k + 1) * rng.uniform(0.05, 1.0))
            chosen, bestmark = None, 0.0
            for position in order[:1 if step == 0 else 2]:
                if draws >= 48:
                    break
                draws += 1
                merit, i, forced, helpvalue = pool[position]
                row = synthesize(i, capacity, low, high, pivot, forced, wave, 0.25 if step == 0 else 1.0)
                if row is None or row == u[i]:
                    continue
                v = profile(i, row)
                if v is None:
                    continue
                capnext = [capacity[t] + v[0][t] - base[i][0][t] for t in range(h)]
                lonext = [low[t] + v[1][t] - base[i][1][t] for t in range(h)]
                hinext = [high[t] + v[2][t] - base[i][2][t] for t in range(h)]
                remaining = sum(defects(capnext, lonext, hinext))
                gain = credit(i, row, v)
                mark = gain - penalty * remaining + rng.uniform(-0.05, 0.05) * max(1.0, abs(gain))
                if chosen is None or mark > bestmark:
                    chosen = (i, row, v, gain, capnext, lonext, hinext, remaining)
                    bestmark = mark
            if chosen is None:
                break
            i, row, v, gain, capacity, low, high, bad = chosen
            rows[i], changed[i] = row, v
            value += gain
            if len(rows) >= 2 and bad <= eps:
                key = canonical(rows)
                if key != anchor and key not in seen:
                    seen[key] = True
                    records.append((value, rows.copy(), changed.copy()))
    records.sort(key=lambda entry: entry[0], reverse=True)
    order = sorted(range(len(records)), key=lambda k: (k + 1) * rng.uniform(0.05, 1.0))
    preferred, used = [], {}
    for k in order:
        tag = tuple(sorted(records[k][1]))
        if tag not in used:
            used[tag] = True
            preferred.append(k)
    preferred.extend(k for k in order if k not in preferred)
    result = []
    for k in preferred[:12]:
        value, rows, changed = records[k]
        if reachable(rows, changed):
            result.append(make_candidate(u, rows))
            if len(result) == 4:
                break
    return result

