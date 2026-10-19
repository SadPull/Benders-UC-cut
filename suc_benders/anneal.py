"""Kaiwu 1.4.1 local SA. QUBO energies are search scores, never bounds."""
# 模块职责：用 Kaiwu 1.4.1 模拟退火搜索 QUBO，恢复原始二进制候选并记录采样统计。
# QUBO 能量包含罚项和缩放，只是搜索分数；它既不是原问题费用，也不是认证下界。
import contextlib
import io
from time import perf_counter

import numpy as np
from scipy import sparse


# 使用本地用户标识和 SDK 授权码初始化 Kaiwu；缺少授权时立即报错。
# 屏蔽 SDK 的原始输出和异常链，避免许可证信息进入日志或返回结果。
def initialize(user_id, sdk_code):
    if not user_id or not sdk_code:
        raise ValueError("Fill kaiwu credentials in config.local.json")
    try:
        # SDK diagnostics may contain license details; expose only a safe error.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import kaiwu as kw
            kw.license.init(user_id=user_id, sdk_code=sdk_code)
    except Exception:
        raise RuntimeError("Kaiwu initialization failed; check SDK 1.4.1 and local authorization") from None


# 输入搜索 QUBO、退火参数、剩余秒数和每条链的种子，返回二进制候选列表与统计。
# 每条链是一次独立退火，只取其能量最低的终态；第 k 条链使用第 k % len(diagonals) 个锚点的线性项。
# Kaiwu 的 size_limit 返回同一轨迹上的近邻状态，iterations_per_t 是单点翻转次数，因此按位数换算扫描。
def anneal(q, o, seconds, seeds):
    import kaiwu as kw
    started = perf_counter()
    samples, chains, isings = [], [], {}
    n = q.Q.shape[0]
    for k, seed in enumerate(seeds if n else []):
        if perf_counter() - started >= seconds:
            break
        anchor = k % len(q.diagonals)
        if anchor not in isings:
            Q = q.Q + sparse.diags(q.diagonals[anchor])
            scale = max(1., float(abs(Q).max()))
            # Kaiwu requires a dense Ising matrix; materialize only the current QUBO.
            isings[anchor] = (Q, scale, kw.core.qubo_matrix_to_ising_matrix(Q.toarray() / scale)[0])
        Q, scale, ising = isings[anchor]
        try:
            optimizer = kw.classical.SimulatedAnnealingOptimizer(
                initial_temperature=o.sa_temperature * q.energy_unit / scale, alpha=o.sa_cooling,
                cutoff_temperature=o.sa_cutoff * q.energy_unit / scale,
                iterations_per_t=o.sa_sweeps * n, size_limit=1,
                rand_seed=int(seed), verbose=False)
            spins = np.atleast_2d(optimizer.solve(ising))[:1]
        except Exception as error:
            raise RuntimeError(f"Kaiwu sampling failed ({type(error).__name__}); check local SDK authorization") from None
        if not np.isin(spins, [-1, 1]).all() or spins.shape[1] not in (n, n + 1):
            raise RuntimeError("Kaiwu returned invalid spins")
        # 含额外辅助自旋时，必须先乘该自旋恢复相对符号，再映射为二进制位。
        x = (((1 + spins[:, :n] * spins[:, -1:]) // 2 if spins.shape[1] == n + 1
              else (1 + spins) // 2))[0]
        samples.append(x)
        chains.append(dict(seed=int(seed), anchor=anchor, energy=float(x @ Q @ x)))
    candidates = []
    for sample in samples:
        x = q.fixed.copy(); x[q.free] = sample[:len(q.free)]
        x = x[:q.nx].astype(np.uint8)
        candidates.append(x)
    statistics = {"bits": n, "candidates": len(candidates), "chains": chains,
                  "best_energy": min((c["energy"] for c in chains), default=np.inf),
                  "seconds": perf_counter() - started}
    statistics.update(raw_bits=[''.join(str(int(v)) for v in x) for x in samples],
                      raw_energies=[c["energy"] for c in chains])
    return candidates, statistics
