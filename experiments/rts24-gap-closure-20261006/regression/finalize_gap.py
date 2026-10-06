"""Verify the completed frozen trial and preserve both attempts outside the worktree."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parents[2]
OLD = WORK / 'results/gap-closure-20261006-112744'
NEW = WORK / 'results/gap-closure-deferred-20261006-114608'
DEST = Path('C:/Users/12433/Desktop/LLM-benders-SUC/experiments/rts24-gap-closure-20261006')
DEV = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def verify_sums(directory):
    count = 0
    for line in (directory/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        assert sha(directory/name) == digest, name
        count += 1
    return count

started = time.perf_counter()
before = read(NEW/'summary.json')
assert before['controller']['status'] == 'completed', before['controller']
assert not DEST.exists(), f'Preserve existing destination: {DEST}'
manifest = read(NEW/'manifest.json')
excluded_regression = read(DEV/'post-regression-state.json')['seconds']
audit_started = manifest['started_clock']+before['controller']['seconds']
assert sha(OLD/'SHA256SUMS.txt') == manifest['prior_archive_sha256']
assert sha(OLD/'references/n32-stoch/result.json') == manifest['cached_reference_sha256']
env = os.environ.copy()
env['PYTHONDONTWRITEBYTECODE'] = '1'
subprocess.run([sys.executable, '-m', 'suc_benders.gap_closure_experiment', '--output-dir', str(NEW),
                '--report-only'], cwd=NEW/'source', env=env, check=True)
after = read(NEW/'summary.json')
assert before == after, 'Frozen report-only rebuild changed numeric results'
counts = {'original': verify_sums(OLD), 'revised': verify_sums(NEW)}
assert after['source_hashes_valid']
assert read(NEW/'audit.json')['all_slots_accounted_for']
for filename in ['deferred-controller.stdout.log', 'deferred-controller.stderr.log']:
    shutil.copyfile(DEV/filename, NEW/filename)
sys.path.insert(0, str(NEW/'source'))
from suc_benders.sa_scale_experiment import checksum, events, load_case
from suc_benders.validation import schedule_object, validate_schedule
checksum(NEW)
counts['revised'] = verify_sums(NEW)
shutil.copytree(OLD, DEST/'original')
shutil.copytree(NEW, DEST/'revised')
for filename in ['baseline-regression.txt', 'full-regression.txt', 'deferred-regression.txt', 'targeted-gap-tests.txt',
                 'post-regression.txt', 'post-regression-state.json']:
    target = DEST/'regression'/filename
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEV/filename, target)
best_checks = {}
for run in after['runs']:
    folder = NEW/'runs'/run['id']
    checkpoint = read(folder/'checkpoint.json') if (folder/'checkpoint.json').exists() else {}
    best = checkpoint.get('best')
    if best is None:
        continue
    assert best['clock'] <= run['state']['deadline']
    data, scenarios = load_case(NEW, run['case'])
    check = validate_schedule(data, scenarios, schedule_object(best))
    assert check['feasible_solution']
    assert abs(best['objective']-run['final_cost']) <= 1e-5
    save(DEST/'best-schedules'/f"{run['id']}.json", best)
    best_checks[run['id']] = check
for filename in ['summary.json', 'audit.json', 'runs.csv', 'blocks.csv', 'curves.csv', 'paper-gap.png', 'certified-lower.png']:
    shutil.copyfile(NEW/filename, DEST/filename)
confirmations = [r for r in after['runs'] if r['stage'] == 'confirmation']
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
curve_endpoints = []
for metric, filename, label in [('gap', 'paper-gap.png', 'Paper-style gap (%)'),
                                ('lower', 'certified-lower.png', 'Certified lower bound')]:
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in confirmations:
        bounds = [e for e in events(NEW/'runs'/row['id']/'events.jsonl') if e['kind']=='bound_update'
                  and e['clock'] <= row['state']['deadline']]
        xs = [e['clock']-row['state']['started_clock'] for e in bounds]
        ys = [e['reliable_lower'] for e in bounds]
        end = row['state']['wall_seconds']
        if metric == 'gap':
            curve_endpoints.append(dict(seed=row['seed'], last_certificate_seconds=xs[-1],
                                        carry_until_stop_seconds=end, lower=ys[-1]))
            ys = [100*(row['reference_cost']-v)/row['reference_cost'] for v in ys]
        line, = ax.step(xs+[end], ys+[ys[-1]], where='post', label=f"seed {row['seed']}")
        ax.plot(end, ys[-1], 'o', color=line.get_color(), markersize=4)
    if metric == 'gap':
        ax.axhline(1., color='grey', linestyle='--', label='1% target')
        ax.axhline(.5, color='grey', linestyle=':', label='0.5% challenge')
    else:
        ref = after['reference']['n32-stoch']['objective']
        ax.axhline(.99*ref, color='grey', linestyle='--', label='1% target lower bound')
        ax.axhline(ref, color='grey', linestyle=':', label='Reference cost')
    ax.set(xlabel='Wall seconds from process launch', ylabel=label, xlim=(0, 600))
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DEST/filename, dpi=150)
    plt.close(fig)
lines = ['# RTS-24 SUC Gap 补足实验交付', '',
    f"主目标（三个固定种子的 600 秒预算均 ≤1%）：{after['primary_passed']}；挑战目标（均 ≤0.5%）：{after['challenge_passed']}。",
    f"独立随机参考 R={after['reference']['n32-stoch']['objective']:.10f}，MIPGap=0；固定选择 {after['selection']['chosen_hours']}h 分块。", '',
    '| 种子 | 实际运行秒 | 状态 | 可行成本 | 认证下界 | 论文口径 Gap | 经典调用 | 完整外层 |',
    '|---:|---:|---|---:|---:|---:|---:|---:|']
for row in confirmations:
    lines.append(f"| {row['seed']} | {row['state']['wall_seconds']:.2f} | {row['solver_status']} | {row['final_cost']:.4f} | {row['reliable_lower']:.4f} | {row['paper_gap']*100:.6f}% | {row['native_calls']} | {row['completed_outer_rounds']} |")
lines += ['', '完整逐运行结果和贡献见 [revised/README.md](revised/README.md)、[summary.json](summary.json)；逐块与时间曲线见 CSV。', '',
    '经典点改善了局部界与可行成本，但确认运行未完成第一块的内层收敛，也未完成外层对偶协调。认证下界仍由初始连续松弛提供；这不是已证明的分解对偶上限。',
    '纯 SA 返回的大量样本只有少量去重后局部可行点；混合方法随割增加，QUBO 构建和退火开销明显增加。经典长运行则仍主要耗时于完整日候选评估。', '',
    '原尝试因完整日评估占九成以上耗时而中断。用户批准将评估移到块求解结束后的前 16 个候选池；旧尝试实际用时完整扣减，随机参考仅在模型和输入相同并再次通过物理校验后复用。',
    f"原尝试 {manifest['prior_seconds']:.6f} 秒；修订运行（含报告）{after['controller']['seconds']:.6f} 秒；累计 {after['controller']['total_attempt_seconds']:.6f} 秒。复核明细见 final-audit.json，含压缩归档的最终预算记录见 ../rts24-gap-closure-20261006.execution-audit.json。总预算 7200 秒，开发与回归不计入。", '',
    '旧尝试原样见 original/；修订后的完整日志、冻结源码与输入、参考和检查点见 revised/；每次最佳方案单独见 best-schedules/。全部失败及未达标运行保留。',
    f"开发分支 codex/gap-closure；执行源码提交 {manifest['commit']}，从 0d90bd1 开始。实施前 154 项回归、实施后 168 项回归均通过。", '',
    '重建报告：在 revised/source 目录使用同一 Python 依赖执行 `python -m suc_benders.gap_closure_experiment --output-dir <revised 的绝对路径> --report-only`。冻结代码重建后的 summary.json 数值已逐项核对一致。', '',
    '交付图将最后一个有效认证下界延用到实际停止时刻；终点圆点表示运行停止，不代表新增证书。三个种子下界相同，曲线重合；横轴保留完整 600 秒预算。原始认证事件曲线及数据保持在 revised/。', '',
    '![论文口径 Gap](paper-gap.png)', '', '![认证下界](certified-lower.png)', '']
(DEST/'README.md').write_text('\n'.join(lines), encoding='utf-8')
audit = dict(report_only_identical=True, source_hashes_valid=True, original_checksums=counts['original'],
             revised_checksums=counts['revised'], cached_reference_hash_valid=True, best_schedule_checks=best_checks,
             display_curve_endpoints=curve_endpoints,
             confirmation_allocations=[r['allocated_seconds'] for r in confirmations],
             execution_seconds=after['controller']['total_attempt_seconds'], audit_started_clock=audit_started,
             excluded_post_regression_seconds=excluded_regression)
audit['audit_seconds'] = time.perf_counter()-audit_started-excluded_regression
audit['total_experiment_seconds'] = audit['execution_seconds']+audit['audit_seconds']
audit['within_7200_seconds'] = audit['total_experiment_seconds'] <= 7200
assert audit['within_7200_seconds'], audit
save(DEST/'final-audit.json', audit)
files = sorted(p for p in DEST.rglob('*') if p.is_file() and p != DEST/'SHA256SUMS.txt')
(DEST/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(DEST).as_posix()}\n' for p in files), encoding='utf-8')
archive = DEST.with_suffix('.zip')
assert not archive.exists(), archive
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in sorted(p for p in DEST.rglob('*') if p.is_file()):
        z.write(path, f'{DEST.name}/{path.relative_to(DEST).as_posix()}')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for item in z.infolist():
        relative = item.filename.split('/', 1)[1]
        assert hashlib.sha256(z.read(item)).hexdigest() == sha(DEST/relative), relative
archive.with_suffix('.zip.sha256').write_text(f'{sha(archive)}  {archive.name}\n', encoding='utf-8')
finished = time.perf_counter()
envelope = dict(execution_seconds=after['controller']['total_attempt_seconds'],
                post_audit_and_archive_seconds=finished-audit_started-excluded_regression,
                excluded_post_regression_seconds=excluded_regression,
                total_experiment_seconds=manifest['prior_seconds']+finished-manifest['started_clock']-excluded_regression,
                archive_sha256=sha(archive), archive_crc_passed=True, archive_bytes_match=True,
                report_only_identical=True)
envelope['within_7200_seconds'] = envelope['total_experiment_seconds'] <= 7200
assert envelope['within_7200_seconds'], envelope
save(archive.with_suffix('.execution-audit.json'), envelope)
print(json.dumps({'destination':str(DEST), 'archive':str(archive), 'archive_sha256':sha(archive),
                  'elapsed_seconds':time.perf_counter()-started, 'primary_passed':after['primary_passed'],
                  'total_seconds':envelope['total_experiment_seconds']}, ensure_ascii=False))
