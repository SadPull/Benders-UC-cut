# Python：Kaiwu SA + LLM Benders SUC

**版本：v0.22.0（2026-10-07）**。变更与兼容性说明见 [CHANGELOG.md](CHANGELOG.md)。

**连续松弛初始化 → 时间分块 → Benders / QUBO / Kaiwu SA → 拉格朗日切平面协调 → 全天恢复与物理校验。**

Python 为默认 `main`，原 MATLAB 代码保存在 `MATLAB` 分支。求解策略以归档提交 `1d590f8` 中 D/17 的冻结代码为基线，保留当前时间分块、连续松弛初始化和稀疏 QUBO 构建。历史实验调度、文件交换控制器和报告工具不恢复；测试中保留少量冻结源码作为独立对照。

## 安装与运行

需要 Windows 64 位、Python 3.10、有效 Gurobi 许可证和 Kaiwu 授权。使用独立虚拟环境及提供的 SDK：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt .\kaiwu-1.4.1-cp310-none-win_amd64.whl
Copy-Item config.example.json config.local.json
```

只在首次配置时复制模板，保留已有本地配置。填写 `kaiwu.user_id`、`kaiwu.sdk_code` 后运行：

```powershell
.\.venv\Scripts\python.exe run_suc.py
# 短运行及可选保存
.\.venv\Scripts\python.exe run_suc.py --max-outer 1 --max-inner 2 --time-limit 60 --output results/result.json
.\.venv\Scripts\python.exe run_suc.py --scenario-file data/deterministic.json --max-outer 1 --max-inner 2 --time-limit 60
```

提供的 wheel SHA256：`b55b11a491232bda40d13396c0d0b4409b594092510ef0ff50d973893d35a694`。wheel、虚拟环境、真实授权和结果目录均不提交 Git。无需 MATLAB、MATPOWER 或 QPU；Kaiwu 自身依赖由 pip 安装。

```python
from run_suc import run_suc
result = run_suc({"max_outer": 2, "max_inner": 5, "time_limit": 120})
print(result.status, result.objective, result.lower_bound, result.relative_gap)
print(result.validation)
```

不传 `--output` 时不自动保存。`u/v/w` 和 `result.scenarios[k].p/r` 为“机组 × 小时”NumPy 矩阵。结果保留 `history` / `outer_history` / `llm_history`，新增 `diagnostics`，记录 SA 调用、原始离散可行数、候选修复距离、重复及拒绝原因、全天评估、来源、LLM 反馈与分项耗时；不持有求解器对象。Python 中无穷界保留 `inf`，JSON 中表示为 `null`。

## 参数与算法约定

默认 32 台机组、24 小时、3 个场景。`run_suc(options)` 的显式选项优先于本地配置，未知参数报错。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `block_hours` | 3 | 时间块长度，支持较短末块 |
| `max_outer` / `max_inner` | 20 / 20 | 外层与各块内层上限 |
| `time_limit` | 600 | 总求解预算，秒 |
| `tolerance` | 0.0001 | 相对容差，另加固定绝对容差 0.00001 |
| `seed` / `verbose` | 7 / true | 随机种子与进度输出 |
| `theta_step` / `penalty` / `reserve_scale` | 1 / 1000000 / 100 | QUBO 目标步长、罚系数和容量缩放 |
| `sa_reads` / `sa_sweeps` | 16 / 5 | Kaiwu 解池大小 `size_limit` 与 `iterations_per_t` |
| `sa_temperature` / `sa_cooling` / `sa_cutoff` | 100 / 0.98 / 0.01 | 按能量尺度调整的温度、冷却与终温系数 |

- 保留期望二次燃料成本、固定运行与启停费用、负荷、备用、最小出力、启停容量、爬坡、最小开停机时间和初始剩余约束。末小时禁止停机，最小时间约束截断至末小时。
- 变量顺序为变量类型、场景、小时、机组。内部索引从 0 开始。分块不重置初始条件，跨块约束进入协调。
- 下界来自全天连续松弛及块主问题的连续松弛，使用旧版数值余量；已移除整数主问题认证和 `certify_seconds` 参数，继续传入该参数会报未知选项。QUBO 能量不是下界，连续松弛解也不充当 SA 候选。
- Benders 割按旧版整条割误差预算向下量化：最优性割为 `theta_step`，可行性割为 `1e-3`，预算分配至各系数及常数项。容量用有理数整数格点，保留旧版 AND 罚函数及辅助位权；稀疏组装后才转为稠密 Ising 矩阵。
- 每次 SA 调用固定使用 `seed`，保留 SDK 返回顺序；不把历史种子混入 SA 样本，也不进行额外能量排序。Gurobi 单线程，Kaiwu 参数沿用旧版 SDK 调用。小 QUBO 固定种子可重复，不污染调用者的 NumPy 随机状态；不同变量排序和时间预算不保证复现旧实验轨迹。
- 原始 SA 候选取 `u` 后覆盖到整批采样开始时的全天方案，按旧版时序与备用规则修复并重建 `v/w`，再去重、调度和校验。修复可以修改块外小时，不能在块边界重新设置初始状态。只通过独立全天验证的方案更新上界。
- 块内保留割和评估缓存，维护最多 16 个较优历史方案。乘子变化后重新调度二进制方案、重建成本和割；全天物理评估缓存不依赖乘子。
- `INNER_NOT_CONVERGED` 表示块内达到上限，停止且不更新乘子。`DUAL_CONVERGED` 只表示对偶协调收敛，原问题仍看 `relative_gap`。还有 `MAX_ITERATIONS`、`TIME_LIMIT` 和带阶段的 `*_FAILED`，原因见 `message`。
- 计时覆盖模型构建与求解，不含输入加载和 SDK 授权。同步 Kaiwu/HTTP 调用不能强制中断，可能超出预算；迟到候选、调度和松弛结果不入账。

恢复旧版行为的例外是：保留默认 3 小时分块、全天连续松弛取整与全开方案择优初始化、稀疏矩阵构建，以及当前内外层循环、拉格朗日协调和单块未收敛即停止的规则。连续松弛可能保留整数间隙，因此即使已有很好的整数方案，也可能返回 `INNER_NOT_CONVERGED`；本版本不通过重新加入整数认证来强行关闭该间隙。

## 可选 LLM

在 `config.local.json` 的 `llm` 中填写完整 Responses `url`、`model`、`api_key`，将 `enabled` 改为 `true`。默认关闭，`timeout=600`、`patience=3`、`max_calls=3`；对应 `llm_*` 运行参数可覆盖本地设置。

每批 SA 的全天评估完成后，根据全天最好成本统计停滞。连续 `patience` 批改善不足 0.05%，且当前块尚未收敛时，按旧版连续请求最多 3 次，并受全局 `max_calls` 限制；每次最多 4 个全天 `u` 矩阵候选。每次完成修复、评估及反馈后才发送下一次请求。上下文恢复全天场景调度、成本、对偶敏感度和机组停机机会；原始 `unit` / `period` 为零起始索引，反馈变化区间为一开始的机组/小时编号。

接入仍使用现有完整 Responses URL、密钥和模型配置，同步非流式发送 `stream=false`、`store=false`、`reasoning={"effort":"max"}`，不设置输出 token 上限。接入模型须支持该推理档位。通过 `output` 中的 `output_text` 提取候选；HTTP 失败计入调用额度，不自动重试。无需旧版外部 Codex 文件交换控制器。

全天候选修复、连续调度和物理校验通过后才能更新上界；其块投影在当前乘子下重新调度生成局部割，不能直接复用全天割。失败后继续 SA；不保留密钥、HTTP 头或原始传输异常文本。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# 无真实授权时，先运行数值和 LLM 离线测试
.\.venv\Scripts\python.exe -m pytest -q -m "not integration"
```

真实 Kaiwu 测试需要本地授权；缺少授权时明确跳过，不能视为真实验收完成。测试不调用真实 LLM 服务。

基线为 MATLAB 提交 `aacef8a6d49ce126a28f70a694c4013f9a1c3b24`。两种场景的矩阵、右端项、目标和边界逐项一致，测试保存 MATLAB 导出的模型指纹。

| 场景 | 约束 / 变量 / 非零项 | MATLAB 固定方案成本 |
|---|---|---:|
| 确定性 | 12648 / 3840 / 30636 | 1223143.120217 |
| 随机 | 21960 / 6912 / 56996 | 1220423.773288 |

本次变更后重新运行真实 Kaiwu 检查：小 QUBO 固定种子重复结果一致；确定性与随机场景各进行一次 60 秒预算、内层两次的纯 SA 短跑（允许提前退出）。确定性成本约 883727.892202、下界约 815439.903585；随机成本约 933522.215366、下界约 841376.524844、间隙约 9.8708%。两者独立校验通过且为 `INNER_NOT_CONVERGED`，不表示全局最优，也不能把初始化收益全归于 SA。集成短跑使用较快采样参数。

离线回归直接对照 D/17 冻结代码的修复结果、整条割量化及 QUBO 矩阵/能量，另检查跨块修复、同批基准冻结、LP 下界、未收敛停止、全天 LLM 连续反馈及迟到结果排除。归档 D/17 的 885066.704 成绩由旧版全天搜索和 LLM 产生，不作为本版本成绩或验收门槛。

## 数据

四个 JSON 原样保留，来源版本、哈希和转换口径见 `data/provenance.json`。公开来源为 [MATPOWER RTS-24](https://github.com/MATPOWER/matpower/blob/8b326f00d496864ed81169f7e684568fc1f7006d/data/case24_ieee_rts.m) 和 [ucacopf RTS79](https://github.com/anyacastillo/ucacopf/blob/001cf29dc2e9b3d26e71146f82db35fa379b4dca/RTS79_ucacopf.dat)。

装机 3405 MW、基础峰值负荷 2850 MW。随机倍率 0.90/1.00/1.05、概率 0.25/0.50/0.25 和 5% 备用率为项目设定。模型为聚合无网络 SUC，不含输电约束、冷热启动成本或水电能量预算。
