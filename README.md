# Python：Kaiwu SA + LLM Benders SUC

**版本：v0.25.0（2026-10-11）**。变更见 [CHANGELOG.md](CHANGELOG.md)，实验和运行环境见 [版本归档与恢复说明](archives/README.md) 及 [GitHub Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.25.0)。

默认算例为 **Kazarlis 经典 10 机组 UC**，支持冷热启动费用，继续使用聚合 SUC。RTS-24 实验显式读取 [独立核心数据](data/rts24/README.md)。

最新论文分工 C/seed 7 已完成：整数主问题仅使用 Kaiwu SA，正式运行中的 Gurobi 优化全部为连续计算。1800 秒预算内得到 UB **933522.215366**、认证 LB **841376.524852**、gap **9.870755%**，未达到 0.1% 目标；相对独立 Gurobi 参考成本高 **5.474786%**。完成 2603 次 SA 和 16 次协调，正式整数 Gurobi 调用为零。报告与全部证据保存在 [v0.25.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.25.0)。

经典 10 机组上下界 LLM 实验已完成，正式间隙 **0.009663%**。完整上界过程、两条真实 LLM 通道的独立验收与证据见 [中文报告](https://github.com/SadPull/LLM-benders-SUC/releases/download/v0.25.0/classic10-report.md)。

SMS100 首轮采用 100 机组、24 小时、100 场景的 A／B／C／D 对照，每组 3600 秒，验收门槛为有效上下界间隙 0.1%。当前已暂停，四组正式实验尚未启动，已有预检与失败记录见 [实验报告](https://github.com/SadPull/LLM-benders-SUC/releases/download/v0.25.0/sms100-report.md)。入口为 `run_benchmark.py`，须先按归档说明恢复或重建 `.benchmark/azema/python/Scripts/python.exe` 的 Gurobi 12.0.2 隔离环境；全部数值预检和至少 4 GiB 可用内存门槛通过后才会启动正式组。

RTS-24 完整方法短测已完成：600 秒内 UB **885637.947869**、LB **881636.274031**、gap **0.451841%**，尚未达到 0.1% 目标，独立审计通过。输入为保留的 32 机组、24 小时、三场景算例，预检和审计单独计时。入口为 `run_rts24_pilot.py`，与当前 Codex 的 Astra/max 子代理调度配合使用；`llm_agent_directory` 指定文件请求目录，候选仍由原数值流程修复、评估及生成割。结果见 [短测报告](https://github.com/SadPull/LLM-benders-SUC/releases/download/v0.25.0/rts24-pilot-report.md)。

本地保留核心代码、回归测试、输入及来源、独立参考数据、版本索引和未提交的本地授权配置。`results/`、`.benchmark/` 和此前移出的缓存统一保存为 v0.25.0 Release 附件，提供逐文件清单和 SHA256；历史失败记录及冻结源码完整归档。运行实验前按 [恢复说明](archives/README.md) 准备依赖。

[SA 参数调优报告](https://github.com/SadPull/LLM-benders-SUC/releases/download/v0.25.0/sa-tuning-report.md) 保留全部 64 次试验记录；复核数据不完整，未修改生产默认参数。`sa_tuning` 源码目录在更早清理前已缺失，专用测试位于实验压缩包内 `results/sa_tuning/20261010-115014-123203/support/test_sa_tuning.py`，不参与核心测试收集。v0.23.0 / v0.24.0 的历史实验继续保留在各自标签和 Release 中。

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

默认 10 台机组、24 小时、3 个场景。`run_suc(options)` 的显式选项优先于本地配置，未知参数报错。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `block_hours` | 3 | 时间块长度，支持较短末块 |
| `max_outer` / `max_inner` | 20 / 20 | 乘子阶段上限 / 每阶段完整遍历轮数上限 |
| `time_limit` | 600 | 总求解预算，秒 |
| `bound_mode` | `relaxation` | 原连续松弛模式，或 `integer_benders` 全天整数主问题认证 |
| `bound_slice_seconds` | 30 | 全天整数主问题每次求解的时间片，秒 |
| `tolerance` | 0.0001 | 相对容差，另加固定绝对容差 0.00001 |
| `seed` / `verbose` | 7 / true | 随机种子与进度输出 |
| `theta_step` / `penalty` / `reserve_scale` | 1 / 1000000 / 100 | QUBO 目标步长、罚系数和容量缩放 |
| `sa_reads` / `sa_sweeps` | 16 / 5 | Kaiwu 解池大小 `size_limit` 与 `iterations_per_t` |
| `sa_temperature` / `sa_cooling` / `sa_cutoff` | 100 / 0.98 / 0.01 | 按能量尺度调整的温度、冷却与终温系数 |

- 保留期望二次燃料成本、固定运行费用、冷热启动和停机费用、负荷、备用、最小出力、启停容量、爬坡、最小开停机时间和初始剩余约束。末小时禁止停机，最小时间约束截断至末小时。默认算例的爬坡及启停容量均为各机组 `pmax`。
- 变量顺序为变量类型、场景、小时、机组。内部索引从 0 开始。分块不重置初始条件，跨块约束进入协调。
- 默认模式下界来自全天连续松弛及块主问题的连续松弛，使用旧版数值余量；可选全天整数主问题认证见下文。旧 `certify_seconds` 参数仍不支持。QUBO 能量不是下界，连续松弛解也不充当 SA 候选。
- Benders 割按旧版整条割误差预算向下量化：最优性割为 `theta_step`，可行性割为 `1e-3`，预算分配至各系数及常数项。容量用有理数整数格点，保留旧版 AND 罚函数及辅助位权；稀疏组装后才转为稠密 Ising 矩阵。
- 每次 SA 调用固定使用 `seed`，保留 SDK 返回顺序；不把历史种子混入 SA 样本，也不进行额外能量排序。Gurobi 单线程，Kaiwu 参数沿用旧版 SDK 调用。小 QUBO 固定种子可重复，不污染调用者的 NumPy 随机状态；不同变量排序和时间预算不保证复现旧实验轨迹。
- 原始 SA 候选取 `u` 后覆盖到整批采样开始时的全天方案，按旧版时序与备用规则修复并重建 `v/w`，再去重、调度和校验。修复可以修改块外小时，不能在块边界重新设置初始状态。只通过独立全天验证的方案更新上界。
- 固定乘子下按轮遍历全部时间块，每个未收敛块每轮最多执行一批 SA；收敛块跳过采样。块内保留割和评估缓存，维护最多 16 个较优历史方案。乘子变化后重新调度二进制方案、重建成本和割；全天物理评估缓存不依赖乘子。
- 达到遍历轮数上限后，即使局部间隙未闭合，也使用各块有效可行平面继续协调；不再返回首块触发的 `INNER_NOT_CONVERGED`。外层上限返回 `MAX_ITERATIONS`，局部间隙记录在诊断中。`DUAL_CONVERGED` 只表示对偶协调收敛，原问题仍看 `relative_gap`。还有 `MAX_ITERATIONS`、`TIME_LIMIT` 和带阶段的 `*_FAILED`，原因见 `message`。
- 计时覆盖模型构建与求解，不含输入加载和 SDK 授权。同步 Kaiwu/HTTP 调用不能强制中断，可能超出预算；迟到候选、调度和松弛结果不入账。

保留默认 3 小时分块、全天连续松弛取整与全开方案择优初始化、稀疏矩阵构建和拉格朗日协调。连续松弛可能保留整数间隙，因此不再用局部间隙是否闭合作为访问下一块的门槛，也不通过重新加入整数认证强行关闭间隙。下界汇总只使用同一乘子下的块下界。协调 LP 先求原目标上界，再在相同最优目标下选择绝对值总和最小的乘子，避免退化方向导致巨大数值；次级目标不用于收敛认证。`diagnostics.blocks` 记录逐阶段各块的访问、采样、首次访问时间和局部间隙；`rounds`、`complete_rounds`、`coordination_count` 记录完整及中断轮和协调次数。

## 可选 LLM

在 `config.local.json` 的 `llm` 中填写完整 Responses `url`、`model`、`api_key`，将 `enabled` 改为 `true`。默认关闭，`timeout=600`、`patience=3`、`max_calls=3`；对应 `llm_*` 运行参数可覆盖本地设置。`max_calls=null`（Python 中为 `llm_max_calls=None`）取消全局请求次数上限，仍受总时间预算和停止规则限制；`0` 表示不请求。

默认 `relaxation` 模式下，每个完整遍历轮的 SA 和块方案回填完成后，根据全天最好成本统计停滞。连续 `patience` 轮改善不足 0.05%，且仍有块未收敛时，请求一次子代理，并受全局 `max_calls` 限制；每次最多 4 个全天 `u` 矩阵候选。请求后重置计数并恢复 SA 遍历；未完成的轮不触发请求。子代理调用记录归属轮末，`block=null`。上下文恢复全天场景调度、成本、对偶敏感度和机组停机机会；原始 `unit` / `period` 为零起始索引，反馈变化区间为一开始的机组/小时编号。`integer_benders` 的双通道批次规则见下文。

接入仍使用现有完整 Responses URL、密钥和模型配置，同步非流式发送 `stream=false`、`store=false`、`reasoning={"effort":"max"}`，不设置输出 token 上限。接入模型须支持该推理档位。通过 `output` 中的 `output_text` 提取候选；HTTP 失败计入调用额度，不自动重试。无需旧版外部 Codex 文件交换控制器。

默认模式的全天候选修复、连续调度和物理校验通过后才能更新上界；其所有块的投影在当前乘子下分别重新调度生成局部割，不能直接复用全天割。失败后继续 SA；不保留密钥、HTTP 头或原始传输异常文本。

## 可选全天整数 Benders 下界认证

```powershell
py -3.10 run_suc.py --bound-mode integer_benders --bound-slice-seconds 30 --llm-enabled --time-limit 3600 --seed 7 --output results/certified-seed7.json
```

此模式保留 SA 分块搜索和拉格朗日协调，另外维护包含全部跨块离散约束的全天整数主问题。每个完整 SA 轮之后求解一个认证时间片；SA 达到内外层上限后，认证继续使用剩余预算。主问题、全天有效割和可行初值被保留，没有新信息时继续求解。该主问题不含完整 SUC 的连续出力变量。

下界仅取全天连续松弛的对偶下界和整数主问题 `ObjBound` 的历史最大值。整数候选目标、SA 能量和协调上界都不能用作认证下界。认证割使用未量化系数及安全数值余量，由全天连续调度的对偶信息生成，不随块乘子变化而清空。每个主问题整数方案直接调度，不进行启发式修复；可行方案可改善上界，不可行方案用于生成可行性割。

认证主问题另设每小时连续子问题成本变量，覆盖期望燃料费用与该小时的冷启动差额。小时调度松弛保留该小时的出力、备用、冷启动差额变量及所需的历史开机状态，仅删除同时含多个小时连续变量的行；全天可行方案一定满足这些松弛，因此各小时认证下界之和仍是全天连续子问题成本下界。全天调度继续完整处理爬坡和其他跨时段约束。小时割按相关二进制方案缓存，使用原始系数，记录在 `diagnostics.hourly_certificate_cuts` 中。

逐小时诊断使用 `hourly_recourse_costs` 和探测记录中的 `hourly_actual_recourse`。`expected_fuel_cost` 及场景费用明细只包含燃料；`fixed_cost` 包含空载、实际冷热启动及停机费用。总成本等于实际固定费用加期望燃料费用，不能再次叠加逐小时成本中的冷启动差额。

启用 LLM 后，上界降本和下界探测分别判断停滞。计数单位是实际进行过采样的完整 SA 轮及随后的一次认证；默认各连续三轮无显著改善才触发对应通道。上界显著改善为降本 0.05%，下界为提高 `1e-5 + tolerance * max(1, abs(UB))`。中断轮、没有采样的轮和纯认证周期不增加计数，首次认证不会直接触发请求。

每个触发阶段中，每条通道最多请求三次，两侧同时触发时按上界、下界交替，合计最多六次。每次回复完成评估后刷新上下文，某侧累计达到显著改善则提前结束该侧；失败或无效回复也占一次额度。阶段结束后重置两侧计数、返回 SA，至少再完成三轮才能再次触发。`llm_max_calls=None` 取消全局次数上限，有限整数仍是两侧共享的全局上限，默认值仍为 3。全程最多一个后台请求在途，每次最多 600 秒，仍受请求超时和总预算约束。LLM 阶段暂停新的 SA 采样，整数认证继续；SA 额度耗尽后的纯认证阶段不再触发新的 LLM 批次。

上界 LLM 读取当前全天方案、三场景调度、实际冷热费用、费用敏感度及上一轮反馈，提出至多四个降本方案。候选经修复、去重、全天调度和独立校验后择优更新上界，可行方案的投影还会反馈到各块。下界 LLM 读取低估方案、预测与实际连续成本及割反馈，提出至多四个全天探测点；重建 `v/w` 并检查离散约束后直接调度，连续调度不可行的探测可以生成可行性割。两侧建议均可产生全天割，可行的下界建议也可降低上界。上界还接收初始化、SA、块回填及整数主问题产生的可行方案，统一按原始全天目标计价。

`diagnostics.advisor_rounds`、`llm_batches` 和 `advisor_state` 保存两侧停滞、批次及恢复状态；回执增加 `channel`、`batch_id`、`batch_call` 和 `phase`。`direct_incumbent_decrease` 按建议实际评估时的上界变化计量。恢复时保留批次已发请求次数，在途请求标为取消；旧记录从新的 SA 停滞窗口开始。正式收敛后保存的成绩与早期状态补充验收分开，补充验收的请求明确标记 `phase=validation`。

只有独立全天物理校验通过且 `UB - LB <= 1e-5 + tolerance * max(1, abs(UB))` 才返回 `CONVERGED`；默认相对容差为 0.01%。对偶收敛单独记录，不终止此模式；超时返回 `TIME_LIMIT` 和最后有效界，下界超过上界则报告认证失败。`certification_history` 记录每次认证的上下界、求解状态与探测成本；`diagnostics.certificate_cuts` 保存割及来源，`bound_probes` 保存探测反馈，`llm_history` 保存不含凭据的请求和响应回执。后台请求耗时单独记录，与主线程墙钟时间重叠，不能相加解释为总时间。割来源和随后下界变化不是 LLM 加速效果的独立消融证明。

新结果在 `diagnostics.model_fingerprint` 中保存模型指纹，恢复前核对完整约束、目标、边界及场景身份，防止不同算例或启动规则混用割和下界。冷热启动模型必须具有匹配指纹。旧固定启动费用记录仍可兼容读取，并转换旧的逐小时燃料诊断字段；恢复 RTS-24 记录时使用 `tests/fixtures/rts24/` 中的原输入。

## 经典 10 机组上下界 LLM 实验（2026-10-10）

三场景、seed 7、3 小时分块、默认 SA、20×20 轮次上限、30 秒整数认证时间片，从头初始化；LLM 总次数不限，每侧每批最多三次。正式运行完成 4 轮、32 次真实 Kaiwu 采样和 7 个认证周期，全部 8 块均覆盖，**15.186 秒达到 `CONVERGED`**：UB **558785.020094**、LB **558731.025157**，相对间隙 **0.009663%**。

上界从修复后的松弛取整方案 **610827.777620** 起步，SA 候选直接降本 **2574.060418**、块回填降本 **1106.157516**、整数主问题降本 **48362.539591**。第 4 轮上界累计三轮停滞并发出 LLM 请求，但数值认证在回复前收敛；正式成绩中两条 LLM 通道的直接贡献均为零。

按提前收敛规则，用两个全新 Astra/max 子代理分别完成早期状态补充验收，每个仅接收当次请求上下文。上界接续已发请求，在请求时状态上评估四个经末小时修复的候选，直接降本 **943.434632**，新增 **4 条全天最优性割、2 条小时割**。下界从首次认证后的冻结状态直接探测四个方案，三个可行、一个备用不足，直接降本 **938.087563**，新增 **3 条全天最优性割、1 条可行性割、8 条小时割**。两侧起点不同，补充结果不改写正式成绩，也不证明 LLM 加速收敛。

实验驱动程序含补充验收、保存、核心审计及绘图共 **863.352 秒**，通过 3600 秒预算验收。最终 `10×24` 三场景调度物理校验通过，固定方案重新调度费用差为零；冷热费用独立核算、全部保留方案与割、界的单调性及 `LB ≤ UB` 均通过检查。150 项离线回归、10 项实验辅助检查和 5 项真实 Kaiwu 检查通过；这些回归关闭真实 LLM。完整调度、请求回执、模型指纹、输入／源码快照、SHA256、原子检查点、曲线和逐候选费用见 [实验报告](https://github.com/SadPull/LLM-benders-SUC/releases/download/v0.25.0/classic10-report.md) 与 [实验压缩包](https://github.com/SadPull/LLM-benders-SUC/releases/download/v0.25.0/experiments-v0.25.0.zip) 内同一实验目录下的 `summary.json`。

## RTS-24 整数认证历史实验与恢复

2026-10-09 的 **RTS-24** seed 7 本地批次使用三个真实 LLM 子代理，从头初始化，预设总预算 3600 秒（搜索 3540 秒，保存与复核预留 60 秒）。进程未写出终态文件，检查点与日志尾部出现零字节损坏；最后完整证据为第 36 个认证周期、1118.97 秒。该批次记为 `INTERRUPTED_WITHOUT_FINAL_RESULT`，不能视为完整的 3600 秒运行或正常 `TIME_LIMIT`。恢复后的 UB 为 **885066.704430**、LB 为 **880685.832405**，相对间隙 **0.494976%**；绝对间隙 4380.872025 超过允许的 88.506680，**未通过收敛验收**。

固定恢复方案重新求解全天调度，物理最大残差为 `9.09e-13 MW`、成本差为零；独立重建第 7 周期主问题，以更严格 MIP 容差再次取得相同 `ObjBound`。恢复的 45 个可行探测与 347 条全天割、2376 条小时割共 122535 个组合检查通过。另有 299 个可行评估的完整 SA、回填和初始化方案未保留，不能声称这些方案均已复核。原始损坏文件、恢复结果、审计及曲线已随历史实验移入回收站。

三个子代理共提交 12 个有效探测，产生 9 条最优性割、3 条可行性割和 17 条小时割，后者有 2 条违反当时主问题预测。两次 LLM 评估直接改善上界，合计降本 **1255.950904**；可恢复的下界自第 7 周期起停滞，不能据此声称 LLM 加速了下界收敛。第 8 至 36 周期均耗尽 30 秒认证时间片，当前主要瓶颈是整数主问题的下界证明；新增割引起的模型修改对搜索状态复用的影响仍需测量。此次复核耗时单独记录，未补记到原实验预算。

随后按用户选择重建分块状态继续 SA 与认证：保留经审计的上下界、全天割、小时割和三次子代理回执；分支定界树及分块缓存重新建立，协调乘子从连续松弛价格重新初始化。Python 入口 `run_suc(..., resume_state=...)` 接受这一经审计的恢复状态，扣除已用时间并延续轮次、采样计数与 LLM 次数。新评估日志保存完整 `u`，便于后续恢复。此次续跑从外层 2、遍历轮 17 开始，没有新增 LLM 请求。

续跑最终为 **`TIME_LIMIT`**，新增 2380.162 秒求解、69 个认证周期、552 次 SA 采样和 4 次协调；累计 105 个认证周期、840 次采样、5 次协调。最终 UB **885066.704430**、LB **880685.832405**，间隙仍为 **0.494976%**，未通过收敛验收。497 条新增全天割、1834 条新增小时割没有提高历史有效下界。547 个可恢复的可行方案与 844 条全天割、4210 条小时割共 2764538 个组合检查通过，最终方案独立物理复核通过；中断前仍有 294 个方案无法恢复，本次新增方案没有缺失。该批次的续跑结果、独立审计和上下界曲线已随历史实验移入回收站。

时间按续跑口径扣除中断停机：前段依据最后落盘时间向上取整计入 1160 秒，累计求解 3540.162 秒。336 MB 详细结果的存储耗时约 66.268 秒，耗尽了 60 秒审计预留；依据进程启动和错误日志时间估算，流程累计约 3612.992 秒，**整体 3600 秒预算验收也未通过**。独立审计在预算外补做并通过，原始错误日志与结果已随历史实验移入回收站。续跑结束后已改用紧凑序列化并保留刷盘；同一结果的保存实测 21.64 秒、约 80.44 MB，回读内容完全一致，这项修复单独记录在存储基准中。

## RTS-24 论文分工 C

`run_rts24_paper_c.py` 保留连续初始化、局部固定变量 QP、主问题 LP 松弛和乘子协调，正式整数主问题只使用 Kaiwu SA。按原始局部整数样本生成割后，再独立拼接和规则修复全天方案；正式运行断言 Gurobi 整数变量数为零。仅连续对偶证据进入认证下界，SA 能量及隔离整数诊断不作为证明。

恢复运行环境后，从新目录执行预检和正式实验；每次正式运行重新初始化，预算为 1800 秒：

```powershell
.\.benchmark\azema\python\Scripts\python.exe -B run_rts24_paper_c.py --phase preflight --seed 7 --directory results/rts24-paper-c-next
.\.benchmark\azema\python\Scripts\python.exe -B run_rts24_paper_c.py --phase run --seed 7 --directory results/rts24-paper-c-next
```

2026-10-11 的 C/7 正确性审计通过，但认证 gap 9.870755% 未达标，初始化后没有降本。原始离散可行样本为 4398/41648，完整原生主问题可行样本为零；修复后重复率 87.6735%。本次按用户明确授权取消 4 GiB 门槛，资源监控和看门狗仍启用；入口默认继续检查内存，显式 `--allow-low-memory` 才复用该例外。详细报告见 [论文分工 C 报告](https://github.com/SadPull/LLM-benders-SUC/releases/download/v0.25.0/rts24-paper-c-report.md)。

## RTS-24 C/D 整数认证改进实验（历史方法）

独立入口 `run_rts24_improved.py` 固定使用已保存的 RTS-24 派生输入，运行 C（改进数值方法）和 D（同一方法加异步双通道子代理）。每组独立初始化，墙钟预算 1800 秒，保留 600 秒成绩，认证目标为 0.1%。认证窗口按停滞情况从 30 秒升至 90、180 秒；启用分场景割、历史认证下界、三轮 SA 后协调及可复现采样种子。

```powershell
.\.benchmark\azema\python\Scripts\python.exe -B run_rts24_improved.py --phase preflight --directory results/rts24-cd-next
.\.benchmark\azema\python\Scripts\python.exe -B run_rts24_improved.py --phase all --directory results/rts24-cd-next
# 单独选择阶段、组别、种子及输出目录
.\.benchmark\azema\python\Scripts\python.exe -B run_rts24_improved.py --phase run --arm C --seed 7 --directory results/rts24-cd-next
```

预检验证核心回归、全部八块及两次协调、真实上下界子代理提案、割和上下界证明。数值启动要求至少 4 GiB 可用内存及预计峰值余量。D 的文件接口需要当前 Codex 任务派发独立上下文的 Astra/max 子代理并记录完成回执；脚本本身不会创建代理。正式运行期间继续 SA 和认证，每次最多一个在途请求，超时及取消结果单独记录。

先执行 C/7、D/7；两组审计通过且至少一组达标后，才执行种子 11、19。已有正式运行不覆盖、不续时。Gurobi 参考解只用于独立费用核对，预检解、参考解和割均不作为正式初始化。输入、冻结源码、证明、回执和结果保存在 v0.25.0 实验压缩包的 `results/rts24-improved-20261011` 中；`report.md` 给出认证 gap 与参考成本差两个不同指标。该入口包含 Gurobi 整数认证，论文分工 C 使用上面的独立入口。

本轮 C/7、D/7 均通过审计，最终成本均为 885066.704430，认证 gap 分别为 0.226751%、0.213077%，未触发多种子扩展。D 发出 5 次正式请求，其中 4 次完成评估、1 次取消；详见结果目录。正式运行结束后修正了冗余诊断打包造成的收尾超时，保留原冻结版本及成绩；当前代码回归为 193 项通过。后续运行须选择新输出目录并重新预检。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# 无真实授权时，先运行数值和 LLM 离线测试
.\.venv\Scripts\python.exe -m pytest -q -m "not integration"
```

真实 Kaiwu 测试需要本地授权；缺少授权时明确跳过，不能视为真实验收完成。测试不调用真实 LLM 服务。

经典 10 机组回归覆盖数据来源与哈希、冷热阈值前／等于／后、零冷启动小时、初始停机历史、跨块启动、多次重启及多场景单次计费。小规模穷举使用独立停机计时核算，核对全天割、小时下界及完整整数模型最优值；恢复测试核对指纹及旧固定费用记录兼容性。

2026-10-10 完成 138 项离线测试及 5 项真实 Kaiwu 检查。真实短运行使用 seed 7、每组 60 秒上限、2 个乘子阶段、每阶段 2 轮、较快采样参数，整数认证时间片为 1 秒；真实 LLM 请求关闭。四组均覆盖全部 8 个时间块，输出 `10×24` 调度，独立物理／费用校验及 `LB ≤ UB` 检查通过。

| 经典 10 机组场景 | 下界模式 | 状态 | UB | LB | 相对间隙 |
|---|---|---|---:|---:|---:|
| 确定性 | `relaxation` | `MAX_ITERATIONS` | 602192.039417 | 558309.564173 | 7.287123% |
| 随机 | `relaxation` | `MAX_ITERATIONS` | 595625.375851 | 553036.327760 | 7.150308% |
| 确定性 | `integer_benders` | `CONVERGED` | 564446.375240 | 564400.066237 | 0.008204% |
| 随机 | `integer_benders` | `CONVERGED` | 558785.020094 | 558785.019842 | < 0.000001% |

`CONVERGED` 表示达到本项目 0.01% 容差；这些短运行用于实现验收，受采样和时间预算影响，不作为文献最优值复现或算法速度比较。

**RTS-24 历史基线**为 MATLAB 提交 `aacef8a6d49ce126a28f70a694c4013f9a1c3b24`。原四个 JSON 保存在 `tests/fixtures/rts24/`，两种场景的矩阵、右端项、目标和边界逐项一致，测试继续核对 MATLAB 导出的模型指纹及固定方案费用。

| 场景 | 约束 / 变量 / 非零项 | MATLAB 固定方案成本 |
|---|---|---:|
| 确定性 | 12648 / 3840 / 30636 | 1223143.120217 |
| 随机 | 21960 / 6912 / 56996 | 1220423.773288 |

v0.22.0 的 RTS-24 真实 Kaiwu 检查：小 QUBO 固定种子重复结果一致；确定性与随机场景各进行一次 60 秒预算、内层两次的纯 SA 短跑（允许提前退出）。确定性成本约 883727.892202、下界约 815439.903585；随机成本约 933522.215366、下界约 841376.524844、间隙约 9.8708%。两者独立校验通过且为 `INNER_NOT_CONVERGED`，不表示全局最优，也不能把初始化收益全归于 SA。集成短跑使用较快采样参数。

离线回归直接对照 D/17 冻结代码的修复结果、整条割量化及 QUBO 矩阵/能量，另检查跨块修复、同批基准冻结、LP 下界、未收敛轮转和协调、轮末 LLM 反馈到全部块及迟到结果排除。归档 D/17 的 885066.704 成绩由旧版全天搜索和 LLM 产生，不作为本版本成绩或验收门槛。

## v0.24.0 RTS-24 历史实验归档

[配对实验与审计](https://github.com/SadPull/LLM-benders-SUC/blob/v0.24.0/experiments/2026-10-08-round-robin-seed7/README.md)：两组均覆盖全部 8 块，每组上限 1800 秒、种子 7、相同冻结代码。SA＋子代理成本 885963.836139，纯 SA 925317.577290，相对降本 4.2530%。纯 SA 完成 400 轮、3200 次采样和 20 次协调；混合组完成 12 轮、96 次采样和 0 次协调后达到时间上限。该发布版的 80 项测试及独立复核通过；完整方法的单种子观察仍保留候选范围差异。

[v0.24.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.24.0) 保存完整实验、冻结源码、请求回执、失败批次和协调数值复现输入，以及发布源码包与 SHA256 校验文件。成本下降过程、下载链接和解压说明见该版本归档。历史配对实验脚本、专用恢复工具及报告目录已从当前项目移除；复现历史实验请使用对应发布附件中的冻结源码和原参数。

## v0.23.0 RTS-24 历史实验归档

[v0.23.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.23.0) 保存发布源码包、完整实验附件和 SHA256 校验文件。该版本的 [实验归档](https://github.com/SadPull/LLM-benders-SUC/blob/v0.23.0/experiments/2026-10-08-seed7-unlimited/README.md) 保存精简报告、曲线、参数和审计记录；原始请求、响应、全天结果、冻结源码及暂停恢复证据在 Release 附件中。

历史实验取消子代理全局次数上限并追加恢复预算后，SA+子代理成本为 885425.011365，纯 SA 为 925317.577290。两组均只执行到第一轮的第一个 3 小时块，后续 7 块与跨块乘子更新未执行；最终方案仍通过完整 24 小时物理校验。SA 直接搜索当前块，子代理提出全天候选。历史 4.3112% 降本只是一组旧协议观察，不能作为修复后比较的结论。单种子及预算变化的解释限制见归档报告。

该发布版中的 [恢复脚本](https://github.com/SadPull/LLM-benders-SUC/blob/v0.23.0/resume_two_arm_experiment.py) 仅用于当时的暂停点：重放前三次调用、验证暂停状态，再纳入第 2 次请求的重试。它严格检查冻结源码和输入哈希，须使用附件中对应的源码快照及原参数路径。

## 数据

默认采用 **Kazarlis 经典 10 机组 UC**：24 小时、装机 1662 MW、峰负荷 1500 MW。原始文献为 [Kazarlis、Bakirtzis、Petridis（1996）](https://doi.org/10.1109/59.485989)，参数使用固定提交的 [GA10.uc](https://github.com/rogierhans/UCBenchmark/blob/8175b0e9dc884e7af5fc1cb1802a9dbe5af38806/New%20efficient%20ADMM%20algorithm%20for%20the%20Unit%20Commitment%20Problem/GA10.uc)，并与 [公开论文的表 1、表 2 及公式 (4)](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.1005577/full)核对。它是聚合 UC 算例，与原始 IEEE 39 节点／新英格兰潮流数据不同，项目不建模节点和线路。

[data/provenance.json](data/provenance.json) 记录版本、SHA256、机组／字段映射和适配假设，[data/sources/GA10.uc](data/sources/GA10.uc) 保留源文件。输入和本地源文件哈希统一在 CRLF/CR 转 LF 后按 UTF-8 计算。燃料函数为 `cost_a*p² + cost_b*p + cost_c*u`，对应 GA10 的 `c/b/a`；采用其显式二次系数（公开表中的二次系数数值需乘 `1e-3`，第 7 台为 `0.0079`）。

确定性场景使用 **10% 备用**；随机场景保留负荷倍率 `0.90/1.00/1.05`、概率 `0.25/0.50/0.25` 和 **5% 备用**。初始状态采用带符号持续小时数，正值为在线；初始出力取在线机组 `pmin`，停机机组为零。选择不附加爬坡限制的经典版本，将 `ramp_up/ramp_down/pstart/pshut` 均设为 `pmax`，不采用 GA10 的额外爬坡字段；停机费用为零。

`startup_cost` 在新数据中为热启动费用。可选向量 `cold_startup_cost` 与 `cold_start_hours` 必须同时提供，长度等于机组数、取有限非负值；冷启动小时数必须为整数，冷启动费用不低于热启动费用。二者缺省时沿用原固定启动费用。

启动前连续停机 `D` 小时、阈值 `L = min_down + cold_start_hours`：`D ≤ L` 为热启动，`D > L` 为冷启动。计时纳入时域开始前的停机历史，跨时间块不重置。热启动基费保留在 `v` 的目标系数中，有界连续变量通过最近 `L+1` 小时的开机状态表达冷启动差额，并经连续子问题及 Benders 割进入优化；差额在全部场景之外只计一次。候选排序、费用诊断和 LLM 上下文使用实际启动费用，独立校验器从开停机历史重新核算。

继续保留项目的末小时禁停和末端最小时间截断规则。这些边界规则、初始出力及随机场景属于项目口径，不能直接声称复现文献的最优目标值。原 RTS-24 输入及其来源记录保存在 [测试夹具](tests/fixtures/rts24/README.md)，用于历史固定费用回归。
