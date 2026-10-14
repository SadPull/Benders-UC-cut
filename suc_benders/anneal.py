"""Kaiwu 1.4.1 local SA. QUBO energies are search scores, never bounds."""
# 模块职责：用 Kaiwu 1.4.1 模拟退火搜索 QUBO，恢复原始二进制候选并记录采样统计。
# QUBO 能量包含罚项和缩放，只是搜索分数；它既不是原问题费用，也不是认证下界。
import contextlib
import io
from time import perf_counter

import numpy as np


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


# 输入稀疏 QUBO、退火参数、剩余秒数和随机种子，返回二进制候选列表与统计。
# 候选只恢复原 u/v/w 位；theta、AND 和松弛辅助位不作为全天方案输出，SDK 顺序保持原样。
def anneal(q, o, seconds, seed):
    import kaiwu as kw
    started = perf_counter()
    samples = []
    n = q.Q.shape[0]
    # SDK 同步调用结束后，上层仍按全局 deadline 检查结果是否迟到，不能仅依靠启动时的 seconds。
    if seconds > 0 and n:
        scale = max(1., float(abs(q.Q).max()))
        # Kaiwu requires a dense Ising matrix; materialize only the current QUBO.
        ising, _ = kw.core.qubo_matrix_to_ising_matrix(q.Q.toarray() / scale)
        # 用单次自旋翻转能量尺度调整温度，避免 QUBO 整体缩放改变退火行为。
        flip_scale = max(float(4 * np.abs(ising).sum(axis=1).max()), q.energy_unit / scale)
        try:
            optimizer = kw.classical.SimulatedAnnealingOptimizer(
                initial_temperature=o.sa_temperature * flip_scale, alpha=o.sa_cooling,
                cutoff_temperature=o.sa_cutoff * q.energy_unit / scale,
                iterations_per_t=o.sa_sweeps, size_limit=o.sa_reads,
                rand_seed=int(seed), verbose=False)
            spins = np.atleast_2d(optimizer.solve(ising))
        except Exception as error:
            raise RuntimeError(f"Kaiwu sampling failed ({type(error).__name__}); check local SDK authorization") from None
        if not np.isin(spins, [-1, 1]).all() or spins.shape[1] not in (n, n + 1):
            raise RuntimeError("Kaiwu returned invalid spins")
        # 含额外辅助自旋时，必须先乘该自旋恢复相对符号，再映射为二进制位。
        binary = ((1 + spins[:, :n] * spins[:, -1:]) // 2 if spins.shape[1] == n + 1
                  else (1 + spins) // 2)
        samples.extend(binary)
    energies = [float(x @ q.Q @ x + q.offset) for x in samples]
    candidates = []
    for sample in samples:
        x = q.fixed.copy(); x[q.free] = sample[:len(q.free)]
        x = x[:q.nx].astype(np.uint8)
        candidates.append(x)
    statistics = {"bits": n, "candidates": len(candidates),
                  "best_energy": min(energies, default=np.inf), "seconds": perf_counter() - started}
    if getattr(o, 'paper_sa', False):
        statistics.update(raw_bits=[''.join(str(int(v)) for v in x) for x in samples], raw_energies=energies)
    return candidates, statistics
