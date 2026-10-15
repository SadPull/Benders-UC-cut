# Python：Kaiwu SA + LLM Benders SUC

**最近发布：v0.30.0（2026-10-15；核心清理与模块合并，回归通过）**。变更见 [CHANGELOG.md](CHANGELOG.md)，源码、恢复包和发布校验见 [v0.30.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.30.0)。

本版收敛到单一主线：独立数值强化下界＋SA／可选 LLM 上界建议，支持经典 UC 和 RTS-24。求解包整理为 17 个模块，旧求解模式、LLM 下界通道、论文对照和 Azéma/SMS 专用实现已移除。新增手动实验入口与独立 Gurobi 参考方案比较，发布校验和恢复清单见 [v0.30.0.json](archives/v0.30.0.json)。本版未重跑正式长实验，以下历史成绩保持原口径。

v0.29.0 历史实验将强化下界接入独立数值进程，上界使用 SA、修复与回填，可选 LLM 只提供上界建议。RTS-24／seed 17 两组串行冷启动、各设 7200 秒求解与 1800 秒审计预算；下界从原输入建立固定模型，不接收上界方案或割。完整证据见 [v0.29.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.29.0)。

| 组别 | UB | 认证 LB | 认证 gap | 结论 |
|---|---:|---:|---:|---|
| A：无 LLM | 931324.966912 | 885038.481803 | 4.969961% | TIME_LIMIT；数值审计通过，预算失败 |
| B：上界 Astra/max 子代理 | 885425.011365 | 885038.481803 | 0.043655% | CONVERGED；数值与预算审计通过 |

B 首次达到 0.1% 用时 **3614.077 秒**，96 次 SA、7 次上界子代理调用；有效建议直接降本 **48082.957029**。A 共 9320 次 SA，1550.458 秒后未再改善；中途因 Windows 文件占用恢复，累计预算没有重置，最终保存使总占用达 **7217.775 秒**。因此本轮不能标为严格配对协议全部通过，也不能证明跨种子稳定收益或 LLM 下界收益。两组最终方案独立重调度、物理约束和共同下界重建均通过。

v0.28.0 的 **0.031458%** 是继承历史割池与方案的冻结状态快筛，与本次冷启动成绩分别保存；其原本仅在本地的完整证据一并纳入 v0.29.0 附件。

项目保留通用求解核心、`run_suc.py`、`run_experiment.py`、必要输入、核心测试、版本文档和恢复清单。历史专用实验控制器、完整结果和临时文件保存在相应版本归档，见 [恢复说明](archives/README.md)。授权配置与许可证保持本地。

历史 v0.26.0 整数轮次的 LB 为 880447.427679、gap 为 0.521913%，原结果和协议继续由该版本归档保存。参考完整模型的整数下界未导入本次冷启动；此前隔离的性能优化未合入。

默认算例为 **Kazarlis 经典 10 机组 UC**，RTS-24 派生输入见 [数据说明](data/rts24/README.md)。流程为连续松弛初始化、时间分块、Benders／QUBO／Kaiwu SA、拉格朗日协调和全天物理校验；下界统一使用独立强化认证。Python 为默认 main，历史 MATLAB 代码保存在 MATLAB 分支。

## 安装与运行

在 VS Code 中可打开 [run_experiment.py](run_experiment.py)，修改顶部的算例、种子、秒预算和目标 gap 百分比，保存后直接点击“运行 Python 文件”，无需命令行参数。默认关闭 LLM、使用强化下界，命令行以中文说明阶段、分块进度与指标含义，并将每次入门实验的参数、完整结果和摘要保存到新的 `results/manual-*` 目录；该入口不包含增量检查点、独立审计或子代理派发。

主算法结束后，入口会使用 Gurobi 单独求解同一完整整数问题，并在最后打印通过物理与成本校验的参考方案成本；未证明最优时会明确标注。完整参考方案（启停、各场景出力和备用）保存在 `reference.json`，状态、成本、参考求解自身的最优间隙和耗时同时写入 `summary.json`，日志见 `reference-gurobi.log`。主算法与参考求解各使用一次 `TIME_LIMIT_SECONDS` 预算（默认各 120 秒，另有建模与保存开销）；参考结果不回填主算法上下界。

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

## 代码结构

`suc_benders/` 按职责保留 17 个 Python 文件。主要入口放在各模块前部，构建、核验和反馈辅助函数随后展开。

| 职责 | 模块 | 阅读重点 |
|---|---|---|
| 输入与配置 | `config`、`data` | 参数检查、输入校验及通用原子 JSON 保存 |
| 建模与独立校验 | `model`、`validation` | 物理模型、时间分块、启停修复、实际冷热费用与独立校验 |
| 数值与 SA 搜索 | `solver`、`benders`、`optimization`、`qubo`、`anneal` | 总流程、连续调度与协调、Gurobi、QUBO 编码和 Kaiwu 采样 |
| 强化下界 | `strengthened_master`、`strengthened_certification` | 数值模型与透视割；独立进程、认证界和生命周期管理 |
| 可选上界 LLM | `upper_advisor`、`context`、`llm`、`advisor_request`、`agent_bridge` | 建议与反馈、上下文、提示与解析、HTTP／文件传输、外部派发命令 |
| 包标识 | `__init__` | 版本信息 |

本次合并的导入路径如下，旧模块文件已删除：

| 原位置 | 当前导入位置 |
|---|---|
| `startup.startup_costs`、`startup.fixed_cost` | `suc_benders.validation` |
| `advisor_feedback.summarize`、`advisor_feedback.changes` | `suc_benders.upper_advisor` |
| `lower_bound.perspective_cut`、`lower_bound.verify_cut`、`lower_bound.proof_model` | `suc_benders.strengthened_master` |
| `subagent_advisor.SubagentAdvisor` | `suc_benders.advisor_request` |
| `subagent_advisor.save` | `suc_benders.data.save_json` |

`save_json` 保留刷盘、原子替换和 Windows 文件占用重试逻辑，数值下界进程通过 `data` 保存结果。`run_suc`、`solve`、检查点格式和 `python -m suc_benders.agent_bridge` 命令继续使用原接口。归档及验证见 [模块合并记录](archives/module-merge-20261015.json)。

## 已验证的强化下界核心

`add_temporal` 添加累计最小开停机约束，`build` 将完整物理关系和透视费用接入已有 Benders 主问题，`separate` 在连续根上添加费用支撑，`certify` 进行一次 `MIPFocus=3` 整数认证。以下展示普通核心模型的调用方式；有效费用割和经过核验的认证保底可由调用方放入 `root`。

```python
from suc_benders.benders import master
from suc_benders.model import model, partition
from suc_benders.strengthened_master import add_temporal, build, separate, certify

m = model(d, scenarios)
full = partition(m, d.horizon)[0][0]
root, _ = master(full, [])
root = add_temporal(root, m)
strengthened = build(m, root, len(root.obj) - 1)
refined = separate(m, strengthened, seconds=60, epsilon=0.01)
proof = certify(refined.model, seconds=180)
print(proof.lower)  # 认证下界；predicted_objective 不是可行上界
```

`certify` 接受模型坐标中的 `warm_start` 和可选 `target_lower` 停止门槛，使用单线程、seed 7、`IntFeasTol=1e-9`。超过时间窗口、数值异常或没有有限整数界时不计新下界。最终 SUC gap 必须配合独立物理验证的可行 UB 计算。本次成功快筛的冻结割池、热启动和完整证据保存在 v0.28.0 归档，示例不承诺复现其冷启动成绩。

求解器将该数值核心放入独立进程：从原输入冷启动，连续透视分离最多 180 秒，然后在固定模型上认证；只向主搜索回报下界，不回填整数方案，也不接收 SA／LLM 反馈。上界沿用初始化、SA、修复与块回填，可选 LLM 只提供上界建议。连续两次调用失败返回 `PAUSED_LLM_FAILURE`，保留预算等待明确恢复。

本次实验专用 `run_upper_validation.py`、专用测试和全部运行目录已归入 Release，恢复步骤见 [archives/README.md](archives/README.md)。正式快照与发布后工作树分别保存。使用子代理时配置 `llm_agent_directory`，外部派发器逐请求创建全新 Astra/max 子代理；子代理保存回复后立即原子写入完成回执，回执使用实际完成时刻。当前只触发上界通道，连续三轮停滞后每批最多三次建议，`llm_max_calls=None` 可取消全局次数上限。

归档的最终实验记录器按实测检查点耗时预留保存时间，恢复时计入上一段的完整占用。这两项修复在正式两组结束后完成并通过回归，未据此重跑或改写成绩；A 的预算失败保留。

## 参数与算法约定

默认 10 台机组、24 小时、3 个场景。`run_suc(options)` 的显式选项优先于本地配置，未知参数报错。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `block_hours` | 3 | 时间块长度，支持较短末块 |
| `max_outer` / `max_inner` | 20 / 20 | 乘子阶段上限 / 每阶段完整遍历轮数上限 |
| `time_limit` | 600 | 总求解预算，秒 |
| `tolerance` | 0.0001 | 相对容差，另加固定绝对容差 0.00001 |
| `sa_enabled` | `true` | 是否执行 SA；关闭时也须关闭 LLM，无需 Kaiwu 授权 |
| `sa_budget_seconds` | `None` | SA 相关建模、搜索、修复、调度与协调累计配额；耗尽后继续认证 |
| `seed` / `verbose` | 7 / true | 随机种子与进度输出 |
| `theta_step` / `penalty` / `reserve_scale` | 1 / 1000000 / 100 | QUBO 目标步长、罚系数和容量缩放 |
| `sa_reads` / `sa_sweeps` | 16 / 5 | Kaiwu 解池大小 `size_limit` 与 `iterations_per_t` |
| `sa_temperature` / `sa_cooling` / `sa_cutoff` | 100 / 0.98 / 0.01 | 按能量尺度调整的温度、冷却与终温系数 |

- 保留期望二次燃料成本、固定运行费用、冷热启动和停机费用、负荷、备用、最小出力、启停容量、爬坡、最小开停机时间和初始剩余约束。末小时禁止停机，最小时间约束截断至末小时。默认算例的爬坡及启停容量均为各机组 `pmax`。
- 变量顺序为变量类型、场景、小时、机组。内部索引从 0 开始。分块不重置初始条件，跨块约束进入协调。
- 全局下界取全天连续松弛的认证下界与独立强化认证结果的历史最大值；块下界只服务搜索和协调诊断。QUBO 能量不是下界，整数候选目标也不是认证界，连续松弛解不充当 SA 候选。
- Benders 割按旧版整条割误差预算向下量化：最优性割为 `theta_step`，可行性割为 `1e-3`，预算分配至各系数及常数项。容量用有理数整数格点，保留旧版 AND 罚函数及辅助位权；稀疏组装后才转为稠密 Ising 矩阵。
- 每次 SA 调用固定使用 `seed`，保留 SDK 返回顺序；不把历史种子混入 SA 样本，也不进行额外能量排序。Gurobi 单线程，Kaiwu 参数沿用旧版 SDK 调用。小 QUBO 固定种子可重复，不污染调用者的 NumPy 随机状态；不同变量排序和时间预算不保证复现旧实验轨迹。
- 原始 SA 候选取 `u` 后覆盖到整批采样开始时的全天方案，按旧版时序与备用规则修复并重建 `v/w`，再去重、调度和校验。修复可以修改块外小时，不能在块边界重新设置初始状态。只通过独立全天验证的方案更新上界。
- 固定乘子下按轮遍历全部时间块，每个未收敛块每轮最多执行一批 SA；收敛块跳过采样。块内保留割和评估缓存，维护最多 16 个较优历史方案。乘子变化后重新调度二进制方案、重建成本和割；全天物理评估缓存不依赖乘子。
- SA 达到轮数或配额上限后，数值认证继续使用剩余总预算；只有经过物理验证的原问题 UB/LB 满足容差才返回 `CONVERGED`。对偶协调闭合仅记入诊断；未达标返回 `TIME_LIMIT`，失败返回带阶段名的 `*_FAILED`，两次 LLM 失败返回 `PAUSED_LLM_FAILURE`。
- 计时覆盖模型构建与求解，不含输入加载和 SDK 授权。Kaiwu 同步调用可能超出预算；HTTP 请求由可取消的工作进程执行；迟到候选、调度和松弛结果不入账。

保留默认 3 小时分块、全天连续松弛取整与全开方案择优初始化、稀疏矩阵构建和拉格朗日协调。连续松弛可能保留整数间隙，因此不再用局部间隙是否闭合作为访问下一块的门槛，协调诊断汇总只使用同一乘子下的块下界，独立强化证明不接收这些块的割或方案。协调 LP 先求原目标上界，再在相同最优目标下选择绝对值总和最小的乘子，避免退化方向导致巨大数值；次级目标不用于收敛认证。`diagnostics.blocks` 记录逐阶段各块的访问、采样、首次访问时间和局部间隙；`rounds`、`complete_rounds`、`coordination_count` 记录完整及中断轮和协调次数。

## 可选 LLM

在 `config.local.json` 的 `llm` 中填写完整 Responses `url`、`model`、`api_key`，将 `enabled` 改为 `true`。默认关闭，`timeout=600`、`patience=3`、`max_calls=3`；对应 `llm_*` 运行参数可覆盖本地设置。`max_calls=null`（Python 中为 `llm_max_calls=None`）取消全局请求次数上限，仍受总时间预算和停止规则限制；`0` 表示不请求。

每个实际采样的完整 SA 轮结束后，根据全天最好成本统计停滞。连续 `patience` 轮降本不足 0.05% 时触发一批上界建议，每批最多三次请求，每次至多四个全天二进制 `u` 矩阵。某批建议使累计降本达到 0.05% 时提前结束；每次请求前重新生成上下文和反馈。未完成轮、没有采样的轮和纯认证阶段不触发新批次。失败也计入调用额度；连续两次失败暂停，成功清零失败计数。

HTTP 使用完整 Responses URL，非流式发送 `stream=false`、`store=false`、`reasoning={"effort":"max"}`，不自动重试。也可通过 `llm_agent_directory` 使用文件交换；外部调度器使用 `python -m suc_benders.agent_bridge` 查询请求、登记派发和完成回执。文件交换检查请求和回复 SHA256、代理身份、矩阵形状及完成时间。两种传输共用严格 JSON／二进制候选解析；凭据不写入回执。

候选经过修复、去重、全天连续调度和独立物理校验后才更新上界；投影到各块时重新调度生成局部割。LLM 批次期间暂停新的 SA，数值认证继续。每次请求至多 600 秒且受总预算约束；迟到回复不评估。

## 接口与恢复

`run_suc(options, *, resume_state=None, progress=None, started_at=None)` 和核心 `solve` 保留预算、进度回调和续跑能力。`bound_mode`、`bound_slice_seconds`、`rts24_improvements`、`paper_sa` 已删除，传入时按未知参数报错；`initial_state` 参数已删除。

续跑仅接受本主线和 v0.29.0 强化模式自身的检查点，核对模型指纹、独立重调度原方案并延续累计时间、SA 配额、LLM 次数和失败计数。下界进程从原输入重建固定模型，不恢复 Gurobi 搜索树，不继承旧 Benders 割池。其他历史模式必须使用对应归档的冻结源码恢复。核心入口不自动保存增量检查点；正式实验记录器另见归档。

`hourly_recourse_costs` 从已验证的全天调度按小时汇总，含概率加权燃料和共享冷启动差额；`expected_fuel_cost` 只表示燃料，`fixed_cost` 包含实际冷热启动。总费用为实际固定费用加期望燃料，不能再次叠加冷启动差额。

## 经典 10 机组上下界 LLM 实验（2026-10-10）

三场景、seed 7、3 小时分块、默认 SA、20×20 轮次上限、30 秒整数认证时间片，从头初始化；LLM 总次数不限，每侧每批最多三次。正式运行完成 4 轮、32 次真实 Kaiwu 采样和 7 个认证周期，全部 8 块均覆盖，**15.186 秒达到 `CONVERGED`**：UB **558785.020094**、LB **558731.025157**，相对间隙 **0.009663%**。

上界从修复后的松弛取整方案 **610827.777620** 起步，SA 候选直接降本 **2574.060418**、块回填降本 **1106.157516**、整数主问题降本 **48362.539591**。第 4 轮上界累计三轮停滞并发出 LLM 请求，但数值认证在回复前收敛；正式成绩中两条 LLM 通道的直接贡献均为零。

按提前收敛规则，用两个全新 Astra/max 子代理分别完成早期状态补充验收，每个仅接收当次请求上下文。上界接续已发请求，在请求时状态上评估四个经末小时修复的候选，直接降本 **943.434632**，新增 **4 条全天最优性割、2 条小时割**。下界从首次认证后的冻结状态直接探测四个方案，三个可行、一个备用不足，直接降本 **938.087563**，新增 **3 条全天最优性割、1 条可行性割、8 条小时割**。两侧起点不同，补充结果不改写正式成绩，也不证明 LLM 加速收敛。

实验驱动程序含补充验收、保存、核心审计及绘图共 **863.352 秒**，通过 3600 秒预算验收。最终 `10×24` 三场景调度物理校验通过，固定方案重新调度费用差为零；冷热费用独立核算、全部保留方案与割、界的单调性及 `LB ≤ UB` 均通过检查。150 项离线回归、10 项实验辅助检查和 5 项真实 Kaiwu 检查通过；这些回归关闭真实 LLM。完整调度、请求回执、模型指纹、输入／源码快照、SHA256、原子检查点、曲线和逐候选费用见 [实验报告](https://github.com/SadPull/LLM-benders-SUC/releases/download/v0.25.0/classic10-report.md) 与 [实验压缩包](https://github.com/SadPull/LLM-benders-SUC/releases/download/v0.25.0/experiments-v0.25.0.zip) 内同一实验目录下的 `summary.json`。

## RTS-24 整数认证历史实验与恢复

2026-10-09 的 **RTS-24** seed 7 本地批次使用三个真实 LLM 子代理，从头初始化，预设总预算 3600 秒（搜索 3540 秒，保存与复核预留 60 秒）。进程未写出终态文件，检查点与日志尾部出现零字节损坏；最后完整证据为第 36 个认证周期、1118.97 秒。该批次记为 `INTERRUPTED_WITHOUT_FINAL_RESULT`，不能视为完整的 3600 秒运行或正常 `TIME_LIMIT`。恢复后的 UB 为 **885066.704430**、LB 为 **880685.832405**，相对间隙 **0.494976%**；绝对间隙 4380.872025 超过允许的 88.506680，**未通过收敛验收**。

固定恢复方案重新求解全天调度，物理最大残差为 `9.09e-13 MW`、成本差为零；独立重建第 7 周期主问题，以更严格 MIP 容差再次取得相同 `ObjBound`。恢复的 45 个可行探测与 347 条全天割、2376 条小时割共 122535 个组合检查通过。另有 299 个可行评估的完整 SA、回填和初始化方案未保留，不能声称这些方案均已复核。原始损坏文件、恢复结果、审计及曲线已随历史实验移入回收站。

三个子代理共提交 12 个有效探测，产生 9 条最优性割、3 条可行性割和 17 条小时割，后者有 2 条违反当时主问题预测。两次 LLM 评估直接改善上界，合计降本 **1255.950904**；可恢复的下界自第 7 周期起停滞，不能据此声称 LLM 加速了下界收敛。第 8 至 36 周期均耗尽 30 秒认证时间片，当前主要瓶颈是整数主问题的下界证明；新增割引起的模型修改对搜索状态复用的影响仍需测量。此次复核耗时单独记录，未补记到原实验预算。

随后按用户选择重建分块状态继续 SA 与认证：保留经审计的上下界、全天割、小时割和三次子代理回执；分支定界树及分块缓存重新建立，协调乘子从连续松弛价格重新初始化。当时冻结源码中的 Python 入口 `run_suc(..., resume_state=...)` 接受这一经审计的恢复状态（当前主线已不接受该旧认证模式），扣除已用时间并延续轮次、采样计数与 LLM 次数。新评估日志保存完整 `u`，便于后续恢复。此次续跑从外层 2、遍历轮 17 开始，没有新增 LLM 请求。

续跑最终为 **`TIME_LIMIT`**，新增 2380.162 秒求解、69 个认证周期、552 次 SA 采样和 4 次协调；累计 105 个认证周期、840 次采样、5 次协调。最终 UB **885066.704430**、LB **880685.832405**，间隙仍为 **0.494976%**，未通过收敛验收。497 条新增全天割、1834 条新增小时割没有提高历史有效下界。547 个可恢复的可行方案与 844 条全天割、4210 条小时割共 2764538 个组合检查通过，最终方案独立物理复核通过；中断前仍有 294 个方案无法恢复，本次新增方案没有缺失。该批次的续跑结果、独立审计和上下界曲线已随历史实验移入回收站。

时间按续跑口径扣除中断停机：前段依据最后落盘时间向上取整计入 1160 秒，累计求解 3540.162 秒。336 MB 详细结果的存储耗时约 66.268 秒，耗尽了 60 秒审计预留；依据进程启动和错误日志时间估算，流程累计约 3612.992 秒，**整体 3600 秒预算验收也未通过**。独立审计在预算外补做并通过，原始错误日志与结果已随历史实验移入回收站。续跑结束后已改用紧凑序列化并保留刷盘；同一结果的保存实测 21.64 秒、约 80.44 MB，回读内容完全一致，这项修复单独记录在存储基准中。

## RTS-24 历史实验入口

论文分工 C、连续下界筛选、时间结构强化、连续分支和整数收敛实验入口均从当前核心项目移出。2026-10-12 至 13 日的入口及其专用测试在 [v0.26.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.26.0) 的清理前源码包中；更早的 C/D 入口仍由 v0.25.0 保存。原实验必须使用原冻结源码和协议；当前核心不能冒充历史快照。

## 验证

本次清理前基线为 253 项通过，9 项集成／数值检查未运行。清理后离线回归 184 项、真实 Kaiwu 检查 3 项均通过。验证结果及外部源码归档见 [本次清理记录](archives/mainline-cleanup-20261015.json)。经典 UC 和 RTS-24 的确定性／随机场景共四组物理模型与强化模型，清理前后矩阵、目标和边界哈希一致；这不代表重跑正式长实验。

回归保留输入来源、冷热费用、初始历史、跨块修复、QUBO 能量及量化、穷举物理与费用检查、强化下界、上界反馈、迟到结果排除、传输身份／哈希和预算恢复。冻结 D/17 源码仅作为仍适用于当前修复与 QUBO 行为的独立测试依据，不进入运行路径。`tests/fixtures/v029-strengthened-checkpoint.json` 是清理前原版代码生成的小算例检查点，用于续跑兼容回归。

```powershell
py -3.10 -m pytest -q -m "not integration"
# 使用本地 Kaiwu 授权：小 QUBO 及经典 UC 确定性／随机场景短测
py -3.10 -m pytest -q -m integration -s
```

测试不调用真实 LLM；缺少 Kaiwu 授权时集成检查跳过。历史实验成绩与测试记录保留在相应版本文档和归档。

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
