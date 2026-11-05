"""D-Wave local SA. QUBO energies are search scores, never bounds."""
# 模块职责：用稀疏 BQM 从随机初态退火，恢复原始二进制候选交给修复流程。
# QUBO 能量包含罚项和缩放，只是搜索分数；它既不是原问题费用，也不是认证下界。
from time import perf_counter

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from scipy import sparse

from .qubo import polish_auxiliaries, qubo_energy


SA_BACKEND = "dwave.samplers.SimulatedAnnealingSampler"
SA_INITIALIZATION = "random_original_exact_aux_segmented_v2"
TEMPERATURES_PER_SEGMENT = 8


def binary_model(Q):
    # xᵀQx 的非对角项必须合并两个方向；完整对角向量保留孤立的零偏置变量。
    pairs = sparse.triu(Q + Q.T, k=1, format="coo")
    pairs.eliminate_zeros()
    return dimod.BinaryQuadraticModel.from_numpy_vectors(
        Q.diagonal(), (pairs.row, pairs.col, pairs.data), 0., dimod.BINARY)


def _envelope_metrics(q, x):
    e = q.envelope
    tau = e.delta * (e.tau_weights @ x[e.tau_ids])
    required = max(0., float((e.alpha + e.beta @ x[:len(q.free)]).max(initial=0.)))
    quantized = e.delta * max(0., float((e.integer_alpha + e.integer_beta @ x[:len(q.free)]).max(initial=0.)))
    return dict(tau=float(tau), envelope_violation=max(0., required-tau),
                quantized_envelope_violation=max(0., quantized-tau),
                envelope_excess=max(0., tau-required), quantized_envelope_excess=max(0., tau-quantized))


# 每段只更新 QUBO，段间精确优化辅助位；始终返回每条链已完成状态中校正能量最低者。
def anneal(q, o, seconds, seeds):
    started = perf_counter()
    deadline = started + seconds
    samples, raw_samples, chains, bqm = [], [], [], None
    n, original = q.Q.shape[0], len(q.free)
    sampler = SimulatedAnnealingSampler()
    for seed in seeds if n else []:
        if perf_counter() >= deadline:
            break
        if bqm is None:
            bqm = binary_model(q.Q)
        if perf_counter() >= deadline:
            break
        x = np.zeros(n, dtype=np.uint8)
        x[:original] = np.random.default_rng(int(seed)).integers(0, 2, original)
        x = polish_auxiliaries(q, x)
        initial = x[:original].copy()
        best, best_raw = x.copy(), x.copy()
        best_energy = best_raw_energy = qubo_energy(q, x)
        diagonal = q.Q.diagonal()
        change = (1.-2*x) * (diagonal + 2*(q.Q@x) - 2*diagonal*x)
        uphill = change[:original][change[:original] > 0]
        temperature = max(o.sa_temperature*q.energy_unit,
                          float(np.quantile(uphill, .9))/np.log(2) if len(uphill) else 0.)
        start_temperature = temperature
        beta_schedule = []
        while temperature > o.sa_cutoff*q.energy_unit:
            beta_schedule.append(1./temperature)
            temperature *= o.sa_cooling
        completed, points, chosen, max_gap = 0, 0, None, 0.
        segment_seeds = []
        for offset in range(0, len(beta_schedule), TEMPERATURES_PER_SEGMENT):
            if perf_counter() >= deadline:
                break
            segment = beta_schedule[offset:offset+TEMPERATURES_PER_SEGMENT]
            sampler_seed = int(np.random.SeedSequence([int(seed), completed]).generate_state(1)[0]) % (2**31)
            # shortcut: 单段不可硬中断；若实测超时仍不可接受，再缩短每段温度点数。
            response = sampler.sample(bqm, num_reads=1, seed=sampler_seed,
                beta_schedule_type="custom", beta_schedule=segment,
                num_sweeps_per_beta=o.sa_sweeps, randomize_order=False,
                proposal_acceptance_criteria="Metropolis", initial_states=(x.astype(np.int8), np.arange(n)),
                initial_states_generator="none")
            labels, values = list(response.variables), response.record.sample
            if (response.vartype is not dimod.BINARY or len(labels) != n or set(labels) != set(range(n))
                    or values.shape != (1, n) or not np.isin(values, [0, 1]).all()):
                raise RuntimeError("D-Wave returned invalid binary samples or variable labels")
            raw = values[0, np.argsort(labels)]
            raw_energy = qubo_energy(q, raw)
            x = polish_auxiliaries(q, raw)
            energy = qubo_energy(q, x)
            max_gap = max(max_gap, raw_energy-energy)
            if energy <= best_energy:
                best, best_raw = x.copy(), raw.copy()
                best_energy, best_raw_energy, chosen = energy, raw_energy, completed
            segment_seeds.append(sampler_seed)
            completed += 1
            points += len(segment)
        samples.append(best); raw_samples.append(best_raw)
        chains.append(dict(seed=int(seed), sampler_seed=segment_seeds[chosen] if chosen is not None else None,
            selected_segment=chosen, selected_state='initial' if chosen is None else 'sample',
            segment_seeds=segment_seeds, segments_completed=completed, temperature_points_completed=points,
            start_temperature=start_temperature, beta_schedule=beta_schedule,
            budget_truncated=points < len(beta_schedule), energy=best_energy, raw_energy=best_raw_energy,
            auxiliary_energy_gap=best_raw_energy-best_energy, max_segment_auxiliary_gap=max_gap,
            returned_auxiliary_gap=best_energy-qubo_energy(q, polish_auxiliaries(q, best)),
            hamming_from_initial=int(np.count_nonzero(best[:original] != initial)),
            **_envelope_metrics(q, best),
            **{'raw_'+key: value for key, value in _envelope_metrics(q, best_raw).items()}))
    if not n and seconds > 0:
        samples.append(np.empty(0, dtype=np.uint8)); raw_samples.append(samples[-1])
    candidates = []
    for sample in samples:
        x = q.fixed.copy(); x[q.free] = sample[:original]
        candidates.append(x[:q.nx].astype(np.uint8))
    statistics = dict(sa_backend=SA_BACKEND, sa_initialization=SA_INITIALIZATION,
        bits=n, candidates=len(candidates), chains=chains,
        best_energy=min((qubo_energy(q, x) for x in samples), default=np.inf),
        seconds=perf_counter()-started, nonzeros=q.Q.nnz,
        max_coefficient=float(np.abs(q.Q.data).max(initial=0.)), cost_unit=q.cost_unit,
        tau_bits=len(q.envelope.tau_ids), optimality_slack_bits=q.envelope.slack_bits,
        envelope_error_bound=q.envelope.error_bound,
        envelope_error_bound_cost=q.cost_unit*q.envelope.error_bound,
        feasibility_relaxation_bound=q.feasibility_relaxation_bound,
        raw_bits=[''.join(str(int(v)) for v in x) for x in raw_samples],
        corrected_bits=[''.join(str(int(v)) for v in x) for x in samples])
    return candidates, statistics
