def propose(context, rng):
    n, h = context['n_units'], context['horizon']
    if not n or not h:
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
    net = [[ca[i] * p[i][t] * p[i][t] + (cb[i] - price[t]) * p[i][t] + cc[i] * u[i][t]
            for t in range(h)] for i in range(n)]
    seen, records = {}, []

    def offer(updates):
        if len(seen) >= 64:
            return
        rows = {i: tuple(row) for i, row in updates.items() if tuple(row) != u[i]}
        if not rows or len(rows) > 4:
            return
        key = tuple((i, rows[i]) for i in sorted(rows))
        if key in seen:
            return
        seen[key] = True
        changed, value = {}, 0.0
        for i, row in rows.items():
            v = profile(i, row)
            if v is None:
                return
            changed[i] = v
            value += base[i][3] - v[3]
            for t in range(h):
                newnet = 0.0
                if row[t]:
                    if u[i][t]:
                        q = p[i][t]
                    elif ca[i] > 0.000000000001:
                        q = (price[t] - cb[i]) / (2.0 * ca[i])
                    else:
                        q = v[2][t] if price[t] >= cb[i] else v[1][t]
                    q = max(v[1][t], min(v[2][t], q))
                    newnet = ca[i] * q * q + (cb[i] - price[t]) * q + cc[i]
                value += net[i][t] - newnet
        for t in range(h):
            capacity, low, high = totals[0][t], totals[1][t], totals[2][t]
            for i, v in changed.items():
                capacity += v[0][t] - base[i][0][t]
                low += v[1][t] - base[i][1][t]
                high += v[2][t] - base[i][2][t]
            if capacity + eps < required[t] or low > minimum[t] + eps or high + eps < maximum[t]:
                return
        records.append((value, rows, changed))

    def reachable(rows, changed):
        states = [rows.get(i, u[i]) for i in range(n)]
        bounds = [changed.get(i, base[i]) for i in range(n)]
        for demand in scenarios:
            low = [[bounds[i][1][t] for i in range(n)] for t in range(h)]
            high = [[bounds[i][2][t] for i in range(n)] for t in range(h)]
            for direction in (1, -1):
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

    def onrow(i, left, right):
        mode = rng.randrange(3)
        if mode == 1:
            while left > 0 and not u[i][left - 1]:
                left -= 1
            while right < h and not u[i][right]:
                right += 1
        elif mode == 2:
            span = max(right - left, int(up[i]))
            lead = rng.randint(0, max(0, span - (right - left)) + min(3, left))
            left = max(0, left - lead)
            right = min(h, max(right, left + span) + rng.randint(0, min(3, h - right)))
        row = list(u[i])
        row[left:right] = [1] * (right - left)
        return row

    cuts = []
    for i in range(n):
        for start, end in intervals(u[i]):
            cuts.append((i, start, end))
            credit = age0[i] if start == 0 and ion[i] else 0
            for edge in range(start + 1, end):
                if end == h or end - edge >= up[i]:
                    cuts.append((i, start, edge))
                if edge - start + credit >= up[i] and edge != h - 1:
                    cuts.append((i, edge, end))
    cuts.sort(key=lambda v: sum(net[v[0]][v[1]:v[2]]), reverse=True)
    merit = sorted(range(n), key=lambda i: cb[i] + ca[i] * mx[i] + cc[i] / max(1.0, mx[i]))
    for trial in range(96):
        if not cuts or len(seen) >= 64:
            break
        position = rng.randrange(len(cuts))
        if rng.random() < 0.8:
            position = min(position, rng.randrange(len(cuts)))
        i, left, right = cuts[position]
        donor = list(u[i])
        donor[left:right] = [0] * (right - left)
        if profile(i, donor) is None:
            continue
        receivers = [j for j in merit if j != i and not all(u[j][left:right])]
        if not receivers or rng.random() < 0.15:
            offer({i: donor})
        if not receivers:
            continue
        position = rng.randrange(len(receivers))
        if rng.random() < 0.8:
            position = min(position, rng.randrange(len(receivers)))
        j = receivers[position]
        updates = {i: donor, j: onrow(j, left, right)}
        if len(receivers) > 1 and rng.random() < 0.22:
            others = [k for k in receivers if k != j]
            k = others[min(rng.randrange(len(others)), rng.randrange(len(others)))]
            updates[k] = onrow(k, left, right)
        offer(updates)
    records.sort(key=lambda v: v[0], reverse=True)
    order = sorted(range(len(records)), key=lambda k: (k + 1) * rng.uniform(0.1, 1.0))
    result = []
    for k in order[:14]:
        value, rows, changed = records[k]
        if reachable(rows, changed):
            result.append(make_candidate(u, rows))
            if len(result) == 4:
                break
    return result

