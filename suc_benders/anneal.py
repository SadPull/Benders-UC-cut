"""D-Wave local SA. QUBO energies are search scores, never bounds."""
# 模块职责：用稀疏 BQM 从随机初态退火，恢复原始二进制候选交给修复流程。
# QUBO 能量包含罚项和缩放，只是搜索分数；它既不是原问题费用，也不是认证下界。
from time import perf_counter

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from scipy import sparse


SA_BACKEND = "dwave.samplers.SimulatedAnnealingSampler"
SA_INITIALIZATION = "random_sa_repair_v1"


def binary_model(Q):
    # xᵀQx 的非对角项必须合并两个方向；完整对角向量保留孤立的零偏置变量。
    pairs = sparse.triu(Q + Q.T, k=1, format="coo")
    pairs.eliminate_zeros()
    return dimod.BinaryQuadraticModel.from_numpy_vectors(
        Q.diagonal(), (pairs.row, pairs.col, pairs.data), 0., dimod.BINARY)


# 输入搜索 QUBO、退火参数、剩余秒数和每条链的种子，返回二进制候选列表与统计。
# 退火终态直接作为原始候选，不再执行额外局部下降。
def anneal(q, o, seconds, seeds):
    started = perf_counter()
    samples, chains, bqm = [], [], None
    n = q.Q.shape[0]
    beta_schedule = []
    temperature = o.sa_temperature * q.energy_unit
    while temperature > o.sa_cutoff * q.energy_unit:
        beta_schedule.append(1. / temperature)
        temperature *= o.sa_cooling
    sampler = SimulatedAnnealingSampler()
    for seed in seeds if n else []:
        if perf_counter() - started >= seconds:
            break
        if bqm is None:
            bqm = binary_model(q.Q)
        if perf_counter() - started >= seconds:
            break
        sampler_seed = int(seed) % (2 ** 31)
        # shortcut: 单链不可硬中断；若实测超时不可接受，再引入分段退火。
        response = sampler.sample(bqm, num_reads=1, seed=sampler_seed,
            beta_schedule_type="custom", beta_schedule=beta_schedule,
            num_sweeps_per_beta=o.sa_sweeps, randomize_order=False,
            proposal_acceptance_criteria="Metropolis",
            initial_states_generator="random")
        labels = list(response.variables)
        values = response.record.sample
        if (response.vartype is not dimod.BINARY or len(labels) != n or set(labels) != set(range(n))
                or values.shape != (1, n) or not np.isin(values, [0, 1]).all()):
            raise RuntimeError("D-Wave returned invalid binary samples or variable labels")
        x = values[0, np.argsort(labels)]
        samples.append(x)
        chains.append(dict(seed=int(seed), sampler_seed=sampler_seed, energy=float(x @ q.Q @ x)))
    candidates = []
    for sample in samples:
        x = q.fixed.copy(); x[q.free] = sample[:len(q.free)]
        x = x[:q.nx].astype(np.uint8)
        candidates.append(x)
    statistics = {"sa_backend": SA_BACKEND, "sa_initialization": SA_INITIALIZATION,
                  "beta_schedule": beta_schedule,
                  "bits": n, "candidates": len(candidates), "chains": chains,
                  "best_energy": min((c["energy"] for c in chains), default=np.inf),
                  "seconds": perf_counter() - started}
    statistics["raw_bits"] = [''.join(str(int(v)) for v in x) for x in samples]
    return candidates, statistics
