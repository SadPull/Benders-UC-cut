# Python：Kaiwu SA + LLM Benders SUC

**版本：v0.24.0（2026-10-08）**。变更与兼容性说明见 [CHANGELOG.md](CHANGELOG.md)，本次对照实验见 [归档索引](experiments/README.md)。

**连续松弛初始化 → 时间分块 → Benders / QUBO / Kaiwu SA → 拉格朗日切平面协调 → 全天恢复与物理校验。**

Python 为默认 `main`，原 MATLAB 代码保存在 `MATLAB` 分支。求解策略以归档提交 `1d590f8` 中 D/17 的冻结代码为基线，保留当前时间分块、连续松弛初始化和稀疏 QUBO 构建。历史实验调度、文件交换控制器和报告工具不恢复；测试中保留少量冻结源码作为独立对照。

## 安装与运行

需要 Windows 64 位、Python 3.10、有效 Gurobi 许可证和 Kaiwu 授权。使用独立虚拟环境及提供的 SDK：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt .\kaiwu-1.4.1-cp310-none-win_amd64.whl
```

首次运行时创建 `config.local.json`，已有本地配置直接保留。最小配置如下，填写自己的 `kaiwu.user_id`、`kaiwu.sdk_code`：

```json
{
  "kaiwu": {"user_id": "", "sdk_code": ""},
  "llm": {"enabled": false, "url": "", "model": "", "api_key": "", "max_calls": 3}
}
```

随后运行：

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
| `max_outer` / `max_inner` | 20 / 20 | 乘子阶段上限 / 每阶段完整遍历轮数上限 |
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
- 固定乘子下按轮遍历全部时间块，每个未收敛块每轮最多执行一批 SA；收敛块跳过采样。块内保留割和评估缓存，维护最多 16 个较优历史方案。乘子变化后重新调度二进制方案、重建成本和割；全天物理评估缓存不依赖乘子。
- 达到遍历轮数上限后，即使局部间隙未闭合，也使用各块有效可行平面继续协调；不再返回首块触发的 `INNER_NOT_CONVERGED`。外层上限返回 `MAX_ITERATIONS`，局部间隙记录在诊断中。`DUAL_CONVERGED` 只表示对偶协调收敛，原问题仍看 `relative_gap`。还有 `MAX_ITERATIONS`、`TIME_LIMIT` 和带阶段的 `*_FAILED`，原因见 `message`。
- 计时覆盖模型构建与求解，不含输入加载和 SDK 授权。同步 Kaiwu/HTTP 调用不能强制中断，可能超出预算；迟到候选、调度和松弛结果不入账。

保留默认 3 小时分块、全天连续松弛取整与全开方案择优初始化、稀疏矩阵构建和拉格朗日协调。连续松弛可能保留整数间隙，因此不再用局部间隙是否闭合作为访问下一块的门槛，也不通过重新加入整数认证强行关闭间隙。下界汇总只使用同一乘子下的块下界。协调 LP 先求原目标上界，再在相同最优目标下选择绝对值总和最小的乘子，避免退化方向导致巨大数值；次级目标不用于收敛认证。`diagnostics.blocks` 记录逐阶段各块的访问、采样、首次访问时间和局部间隙；`rounds`、`complete_rounds`、`coordination_count` 记录完整及中断轮和协调次数。

## 可选 LLM

在 `config.local.json` 的 `llm` 中填写完整 Responses `url`、`model`、`api_key`，将 `enabled` 改为 `true`。默认关闭，`timeout=600`、`patience=3`、`max_calls=3`；对应 `llm_*` 运行参数可覆盖本地设置。`max_calls=null`（Python 中为 `llm_max_calls=None`）取消全局请求次数上限，仍受总时间预算和停止规则限制；`0` 表示不请求。

每个完整遍历轮的 SA 和块方案回填完成后，根据全天最好成本统计停滞。连续 `patience` 轮改善不足 0.05%，且仍有块未收敛时，请求一次子代理，并受全局 `max_calls` 限制；每次最多 4 个全天 `u` 矩阵候选。请求后重置计数并恢复 SA 遍历；未完成的轮不触发请求。子代理调用记录归属轮末，`block=null`。上下文恢复全天场景调度、成本、对偶敏感度和机组停机机会；原始 `unit` / `period` 为零起始索引，反馈变化区间为一开始的机组/小时编号。

接入仍使用现有完整 Responses URL、密钥和模型配置，同步非流式发送 `stream=false`、`store=false`、`reasoning={"effort":"max"}`，不设置输出 token 上限。接入模型须支持该推理档位。通过 `output` 中的 `output_text` 提取候选；HTTP 失败计入调用额度，不自动重试。无需旧版外部 Codex 文件交换控制器。

全天候选修复、连续调度和物理校验通过后才能更新上界；其所有块的投影在当前乘子下分别重新调度生成局部割，不能直接复用全天割。失败后继续 SA；不保留密钥、HTTP 头或原始传输异常文本。

## 验证

单种子对照实验（随机 RTS-24，种子 7，SA+LLM 后接固定种子 SA，各最多 1800 秒）：

```powershell
py -3.10 run_two_arm_experiment.py --output-dir results/two-arm-seed7-new
```

需要填写 `config.local.json` 中的 Kaiwu 授权，并在 Codex 中运行，由 Codex 调度全新的 Astra/max 子代理。此实验脚本通过 `connectivity/` 和 `D-7/calls/NNN/` 中的请求、候选及完成回执交换数据，不使用 HTTP，也不需要 HTTP 密钥。每次回执包含请求 ID、子代理标识和响应 SHA256。脚本首先测试子代理连通性，失败立即停止整个批次；通过后检查 Kaiwu、运行独立 Gurobi 参考，再串行运行两组。实验设置 `llm_max_calls=None`，取消子代理总调用次数上限；每次完整轮停滞触发请求一次、最多 4 个候选；采用上述轮转和协调规则，预检单独记录。输出目录必须是新目录，保存不含本地凭据的源码快照、逐组结果、独立复核、报告和成本曲线。报告分别列出参考成本差、求解器下界间隙、逐块采样与覆盖率、完整轮数和协调次数；两组全部块都被访问且未收敛块实际采样才通过覆盖验收，否则标记 `COVERAGE_FAILED`。SA 直接搜索当前块，子代理搜索全天；此实验比较相同时间预算下的完整方法，单种子结果不代表普遍优势。

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

v0.22.0 的真实 Kaiwu 检查：小 QUBO 固定种子重复结果一致；确定性与随机场景各进行一次 60 秒预算、内层两次的纯 SA 短跑（允许提前退出）。确定性成本约 883727.892202、下界约 815439.903585；随机成本约 933522.215366、下界约 841376.524844、间隙约 9.8708%。两者独立校验通过且为 `INNER_NOT_CONVERGED`，不表示全局最优，也不能把初始化收益全归于 SA。集成短跑使用较快采样参数。

离线回归直接对照 D/17 冻结代码的修复结果、整条割量化及 QUBO 矩阵/能量，另检查跨块修复、同批基准冻结、LP 下界、未收敛轮转和协调、轮末 LLM 反馈到全部块及迟到结果排除。归档 D/17 的 885066.704 成绩由旧版全天搜索和 LLM 产生，不作为本版本成绩或验收门槛。

## 分块轮转修复后的新实验

[新配对实验与审计](experiments/2026-10-08-round-robin-seed7/README.md)：两组均覆盖全部 8 块，每组上限 1800 秒、种子 7、相同冻结代码。SA＋子代理成本 885963.836139，纯 SA 925317.577290，相对降本 4.2530%。纯 SA 完成 400 轮、3200 次采样和 20 次协调；混合组完成 12 轮、96 次采样和 0 次协调后达到时间上限。80 项测试及独立复核通过；完整方法的单种子观察仍保留候选范围差异。

[v0.24.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.24.0) 保存完整实验、冻结源码、请求回执、失败批次和协调数值复现输入，以及发布源码包与 SHA256 校验文件。成本下降过程、下载链接和解压说明见本次归档。

## v0.23.0 历史实验归档

[v0.23.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.23.0) 保存发布源码包、完整实验附件和 SHA256 校验文件。仓库内的 [实验归档](experiments/2026-10-08-seed7-unlimited/README.md) 保存精简报告、曲线、参数和审计记录；原始请求、响应、全天结果、冻结源码及暂停恢复证据在 Release 附件中。

历史实验取消子代理全局次数上限并追加恢复预算后，SA+子代理成本为 885425.011365，纯 SA 为 925317.577290。两组均只执行到第一轮的第一个 3 小时块，后续 7 块与跨块乘子更新未执行；最终方案仍通过完整 24 小时物理校验。SA 直接搜索当前块，子代理提出全天候选。历史 4.3112% 降本只是一组旧协议观察，不能作为修复后比较的结论。单种子及预算变化的解释限制见归档报告。

`resume_two_arm_experiment.py` 是本次暂停点的专用恢复工具：重放前三次调用、验证暂停状态，再纳入第 2 次请求的重试；`--check-only` 只验证状态恢复。它严格检查冻结源码和输入哈希，历史实验须使用附件中对应的源码快照及原参数路径。发布版本号更新后，当前源码不会通过旧实验的逐文件哈希检查。

## 数据

四个 JSON 原样保留，来源版本、哈希和转换口径见 `data/provenance.json`。公开来源为 [MATPOWER RTS-24](https://github.com/MATPOWER/matpower/blob/8b326f00d496864ed81169f7e684568fc1f7006d/data/case24_ieee_rts.m) 和 [ucacopf RTS79](https://github.com/anyacastillo/ucacopf/blob/001cf29dc2e9b3d26e71146f82db35fa379b4dca/RTS79_ucacopf.dat)。

装机 3405 MW、基础峰值负荷 2850 MW。随机倍率 0.90/1.00/1.05、概率 0.25/0.50/0.25 和 5% 备用率为项目设定。模型为聚合无网络 SUC，不含输电约束、冷热启动成本或水电能量预算。
