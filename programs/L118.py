def propose(context, rng):
    n, h = context['n_units'], context['horizon']
    if n < 1 or h < 1:
        return []
    u, p, load = context['u'], context['p'], context['load']
    mn, mx = context['pmin'], context['pmax']
    start, stop = context['pstart'], context['pshut']
    ru, rd = context['ramp_up'], context['ramp_down']
    up, down = context['min_up'], context['min_down']
    on, duration, p0 = context['initial_on'], context['initial_duration'], context['initial_output']
    ca, cb, cc = context['cost_a'], context['cost_b'], context['cost_c']
    eps = 0.000001

    def hours(value):
        return max(0, int(value) + (int(value) < value))

    def bounds(i, row):
        state, age = int(on[i]), abs(duration[i])
        low, high = [], []
        for t, value in enumerate(row):
            if value != state:
                if age + eps < (up[i] if state else down[i]) or (state and t == h - 1):
                    return None
                age = 0
            age += 1
            low.append(mn[i] if value else 0.0)
            high.append(mx[i] if value else 0.0)
            if value and not state:
                high[t] = min(high[t], start[i])
            if value and t + 1 < h and not row[t + 1]:
                high[t] = min(high[t], stop[i])
            if t == 0 and state:
                if value:
                    low[t] = max(low[t], p0[i] - rd[i])
                    high[t] = min(high[t], p0[i] + ru[i])
                elif p0[i] > stop[i] + eps:
                    return None
            if t and value and state:
                low[t] = max(low[t], low[t - 1] - rd[i])
                high[t] = min(high[t], high[t - 1] + ru[i])
            state = value
        for t in range(h - 2, -1, -1):
            if row[t] and row[t + 1]:
                low[t] = max(low[t], low[t + 1] - ru[i])
                high[t] = min(high[t], high[t + 1] + rd[i])
        if any(low[t] > high[t] + eps for t in range(h)):
            return None
        rise, fall = [0.0] * h, [0.0] * h
        for t in range(1, h):
            rise[t], fall[t] = high[t] - low[t - 1], high[t - 1] - low[t]
            if row[t] and row[t - 1]:
                rise[t], fall[t] = min(rise[t], ru[i]), min(fall[t], rd[i])
        return low, high, rise, fall

    base = [bounds(i, u[i]) for i in range(n)]
    if any(v is None for v in base):
        return []
    totals = [[sum(v[k][t] for v in base) for t in range(h)] for k in range(4)]
    required = [v * (1.0 + context['reserve_fraction']) for v in load]
    price = []
    for t in range(h):
        free = sorted(2.0 * ca[i] * p[i][t] + cb[i] for i in range(n)
                      if u[i][t] and base[i][0][t] + eps < p[i][t] < base[i][1][t] - eps)
        head = [2.0 * ca[i] * p[i][t] + cb[i] for i in range(n)
                if u[i][t] and p[i][t] + eps < base[i][1][t]]
        price.append(free[len(free) // 2] if free else min(head) if head else
                     max([2.0 * ca[i] * p[i][t] + cb[i] for i in range(n) if u[i][t]] + [0.0]))
    target = [[min(mx[i], max(mn[i], (price[t] - cb[i]) / max(eps, 2.0 * ca[i])))
               for t in range(h)] for i in range(n)]
    profit = [[(price[t] - cb[i]) * target[i][t] - ca[i] * target[i][t] * target[i][t] - cc[i]
               for t in range(h)] for i in range(n)]
    credit = [[ca[i] * p[i][t] * p[i][t] + cb[i] * p[i][t] + cc[i] * u[i][t]
               - price[t] * p[i][t] for t in range(h)] for i in range(n)]
    segments = [(i, s, e) for i in range(n) for s, e in intervals(u[i])]
    if not segments:
        return []
    gaps = [(i, s, e) for i in range(n) for s, e in intervals(tuple(1 - v for v in u[i]))]
    segments.sort(key=lambda v: sum(credit[v[0]][v[1]:v[2]]), reverse=True)
    gaps.sort(key=lambda v: sum(profit[v[0]][v[1]:v[2]]), reverse=True)
    memo, seen, fragments, records = {}, {}, {}, []

    def prepare(i, row):
        pair = (i, row)
        if pair not in memo:
            band = bounds(i, row)
            if band is None:
                memo[pair] = None
            else:
                cash, delta = 0.0, []
                for t in range(h):
                    desired = target[i][t] if row[t] and not u[i][t] else p[i][t]
                    q = min(band[1][t], max(band[0][t], desired))
                    delta.append(p[i][t] - q)
                    cash += ca[i] * (p[i][t] * p[i][t] - q * q) + cb[i] * (p[i][t] - q)
                    cash += cc[i] * (u[i][t] - row[t])
                value = cash - sum(price[t] * delta[t] for t in range(h))
                memo[pair] = (band, delta, value)
        return memo[pair]

    def consider(updates):
        key = tuple(sorted((i, tuple(row)) for i, row in updates.items() if tuple(row) != u[i]))
        if not key or len(key) > 4 or key in seen or len(seen) >= 64:
            return
        seen[key] = True
        sums, net, value = [v.copy() for v in totals], [0.0] * h, 0.0
        for i, row in key:
            item = prepare(i, row)
            if item is None:
                return
            band, delta, gain = item
            value += gain
            for t in range(h):
                net[t] += delta[t]
                for k in range(4):
                    sums[k][t] += band[k][t] - base[i][k][t]
        if any(sums[0][t] > load[t] + eps or sums[1][t] + eps < required[t] for t in range(h)):
            return
        if any(load[t] - load[t - 1] > sums[2][t] + eps or
               load[t - 1] - load[t] > sums[3][t] + eps for t in range(1, h)):
            return
        for t in range(h):
            room = sums[1][t] - load[t] if net[t] >= 0 else load[t] - sums[0][t]
            value -= 0.2 * max(1.0, abs(price[t])) * net[t] * net[t] / max(1.0, room + abs(net[t]))
        records.append((value, dict(key)))

    def build(i, left, right, bit, library):
        row = list(u[i])
        row[left:right] = [bit] * (right - left)
        row = tuple(row)
        pair = (i, row)
        if row == u[i] or pair in fragments:
            return
        fragments[pair] = True
        item = prepare(i, row)
        if item is None:
            return
        library.append((item[2], i, row, left, right, item[1]))
        # Keep row-valid fragments even if their standalone aggregate check fails.
        consider({i: row})

    adds, cuts = [], []
    for attempt in range(12):
        if not gaps:
            break
        i, s, e = rng.choice(gaps[:16] if rng.random() < 0.8 else gaps)
        left, right, style = s, e, rng.randrange(4)
        first = s + hours(down[i]) if s > 0 or on[i] else hours(down[i] - abs(duration[i]))
        if style == 1:
            last = e if e == h else e - max(1, hours(down[i]))
            if last > s:
                right = rng.randint(s + 1, last)
        elif style == 2:
            if first < e:
                left = rng.randint(first, e - 1)
        elif style == 3:
            last = e - hours(down[i]) if e < h else h
            if first >= last:
                continue
            left = rng.randint(first, last - 1)
            shortest = min(h, left + max(1, hours(up[i])))
            if shortest > last:
                continue
            right = rng.randint(shortest, last)
        if s == 0 and not on[i]:
            left = max(left, hours(down[i] - abs(duration[i])))
        if left < right:
            build(i, left, right, 1, adds)
    adds.sort(key=lambda v: v[0], reverse=True)
    for attempt in range(24):
        guided = bool(adds) and rng.random() < 0.75
        options = segments[:24] if rng.random() < 0.75 else segments
        if guided:
            receiver = rng.choice(adds[:10])
            options = [v for v in segments if v[0] != receiver[1]
                       and v[1] < receiver[4] and receiver[3] < v[2]]
            if not options:
                options, guided = segments, False
        i, s, e = rng.choice(options[:20] if guided else options)
        left, right, style = s, e, rng.randrange(4)
        length = rng.randint(1, e - s)
        if style == 1:
            right = min(e, max(s + 1, receiver[4] + rng.randint(-2, 2))) if guided else s + length
        elif style == 2:
            left = max(s, min(e - 1, receiver[3] + rng.randint(-2, 2))) if guided else e - length
        elif style == 3:
            left = max(s, receiver[3]) if guided else rng.randrange(s, e)
            right = min(e, receiver[4]) if guided else rng.randint(left + 1, e)
        build(i, left, right, 0, cuts)
    cuts.sort(key=lambda v: v[0], reverse=True)
    for attempt in range(28):
        if adds and rng.random() < 0.85:
            width = min(10, len(adds))
            receiver = adds[min(rng.randrange(width), rng.randrange(width))]
            updates, balance = {receiver[1]: receiver[2]}, receiver[5].copy()
            available = [v for v in cuts if v[1] != receiver[1]
                         and v[3] < receiver[4] and receiver[3] < v[4]]
            size = rng.choice((1, 1, 2, 3))
        else:
            updates, balance, available = {}, [0.0] * h, cuts
            size = rng.choice((2, 2, 3, 4))
        for part in range(size):
            options = [v for v in available if v[1] not in updates]
            if not options:
                break
            options.sort(key=lambda v: v[0] - 0.2 * sum(max(1.0, abs(price[t])) *
                         (abs(balance[t] + v[5][t]) - abs(balance[t])) for t in range(h)), reverse=True)
            width = min(12, len(options))
            chosen = options[min(rng.randrange(width), rng.randrange(width))]
            updates[chosen[1]] = chosen[2]
            balance = [balance[t] + chosen[5][t] for t in range(h)]
        if len(updates) >= 2:
            consider(updates)
    records.sort(key=lambda v: v[0], reverse=True)
    result = []
    for slot in range(min(4, len(records))):
        pool = [v for v in records if len(v[1]) > 1] if slot else records
        if not pool:
            pool = records
        width = min(len(pool), 6 if slot == 0 else 24)
        chosen = pool[min(rng.randrange(width), rng.randrange(width))]
        result.append(make_candidate(u, chosen[1]))
        position = records.index(chosen)
        records = records[:position] + records[position + 1:]
    return result
