# LLM Benders SUC

Python 机组组合求解器：时间分块 D-Wave 开源 SA 搜索可行上界，Codex 提供异步上界候选，独立数值进程认证下界。

v0.34.0 固定采用**随机初态 SA＋修复**，已删除锚点初态、邻近惩罚和额外 1-flip 下降的实现与配置入口。原始候选评价／割学习、全天修复与物理校验、独立下界认证继续使用原有流程。

当前默认输入为 Pisa IEEE118 派生的 **54 机组、24 小时、单场景、零备用**；启动费按 `fixedCost` 列映射为零。只修改 `data/units.json` 和 `data/stochastic.json` 即可更换输入，来源与费用口径见 `data/provenance.json`。

## 运行

环境：Windows 64 位、Python 3.10、Gurobi 许可证。SA 使用 Apache-2.0 开源的 `dwave-samplers==1.8.0`（`dimod==0.12.22`、`networkx==3.4.2`），在本地 CPU 运行，无需 Kaiwu 授权、D-Wave 云账户或量子硬件。

```powershell
py -3.10 -m pip install -r requirements.txt
```

保留自己的 `config.local.json`，不要上传授权信息。升级时删除旧的 `kaiwu` 段，LLM 配置保持不变。首次配置可使用：

```json
{
  "llm": {"enabled": false, "url": "", "model": "", "api_key": "", "max_calls": 3}
}
```

在 VS Code 中直接运行 `run_experiment.py`。预算、种子、LLM 开关等参数位于文件顶部；`CHECK_ONLY=True` 只检查输入。入口默认启用 Codex CLI，使用已有登录、`gpt-6-astra/max`，不需要在项目中填写 API 密钥。缺少 Codex 时可设置 `LLM_ENABLED=False`。

入口默认从零开始，预算 7200 秒、seed 17、目标认证 gap 0.1%。`RESUME_FROM=None` 表示冷启动；续跑时填入从归档解压的完整检查点路径，程序保留已有方案、认证 LB、累计耗时和调用计数，重建局部搜索模型，并把配置预算作为追加时间。续跑跳过参考求解；冷启动另用最多 120 秒求参考解，其结果不回填主算法。

每轮遍历全部时间块，未闭合的块各最多采样一批 SA：一批为 `sa_reads` 条独立退火链，种子由配置种子、调用序号和链号派生，再取模 `2**31` 传给 D-Wave；诊断同时保留派生种子和实际采样种子。候选拼接使用该批开始时的当前最优全天方案；保留调度去重。首次完整轮改善不足 0.05% 时可触发 LLM；`LLM_MAX_CALLS=None` 取消全局次数上限，单次最多 1800 秒、预留 60 秒。剩余时间不足完整等待窗口时不新发请求，已有在途请求仍正常处理。CLI 后续调用显式续接同一会话，恢复时选用历史中最后一次成功调用的会话；会话丢失或身份不匹配会停止，不默默新建。

结果自动写入新建的 `results/manual-seed.../` 或 `results/resume-seed.../`：

- `summary.json`：状态、UB/LB/gap、耗时和统计。
- `result.json`：精简主结果、最终方案、计时与调用摘要。
- `reference.json`：仅冷启动运行时生成的独立参考解。
- `llm_requests/`：实际触发时的请求与回执。
- `snapshot/`、`manifest.json`、`effective-options.json`：实际源码与输入快照、SHA256、依赖版本和脱敏后的完整有效配置。
- `progress.jsonl`、`latest-solution.json`：轻量进度与原子更新的最新可行方案。最新方案不含完整搜索状态，不能作为 `RESUME_FROM` 的完整检查点。

默认不再保存周期或终态的全量检查点，也不序列化全部历史采样；记录开销与最终保存耗时单列。旧实验的完整检查点保留原样。

manifest、精简结果及运行诊断记录 `sa_backend` 与 `sa_initialization`（当前为 `random_sa_repair_v1`）。采样诊断包含 beta 序列，以及每条链的原始／实际种子、退火终态与 QUBO 能量；这些诊断可经 `run_suc` 返回对象读取，不自动全量落盘。

读取精简结果：

```python
import json
with open("results/你的运行目录/result.json", encoding="utf-8") as f:
    result = json.load(f)
```

直接运行 `run_experiment.py`；`run_suc.py` 仅提供 `from run_suc import run_suc` 的 Python 接口。旧命令行参数及 `--output` JSON 输出入口已删除。

全天联合 SA 已删除，`enhanced_search` 和 `block_sa_period` 参数不再接受，结果也不再输出联合搜索统计。新主线保留恢复接口；含联合 SA 调用、批次或采样记录的旧检查点返回 `RESUME_FAILED`，请使用归档版本读取和继续旧实验。入口只恢复显式指定的完整检查点，不自动补跑；恢复时继续累计已用预算。

只接受当前完整的分块检查点、匹配的 D-Wave SA／随机初始化标识和 v2 LLM 调度状态。Kaiwu、含锚点或额外下降、以及缺少 SA 标识的旧检查点返回 `RESUME_FAILED`，使用其归档版本继续；不迁移旧状态、补齐缺失历史或重建旧统计。机组输入必须明确提供 `initial_output`。运行环境固定为上述 Windows／Python 3.10，TOML 解析直接使用已声明的 `tomli` 依赖。

## 核心文件

- `suc_benders/solver.py`：结果类型与主循环，只安排初始化、分块轮询、协调、建议和认证。
- `suc_benders/search.py`：单块搜索及共享的全天候选修复、调度、校验、去重和预算。
- `suc_benders/benders.py`：连续调度、Benders 割、块主问题和跨块乘子协调。
- `suc_benders/strengthened_*.py`：独立强化证明模型，只返回认证下界。
- `suc_benders/upper_advisor.py`：异步 LLM 建议调度；上下文与传输由其配套模块负责。
- `run_suc.py`、`run_experiment.py`：程序接口与一键运行入口。
- `data/`：当前输入、原始算例与溯源信息。
- `tests/`：当前分块主线的必要回归；执行 `py -3.10 -m pytest -q tests`。
- `config.local.json`、`.local/*.lic`：仅本地保留的授权配置。

主线顺序：输入建模 → 连续松弛与初始可行解 → 固定乘子的分块 SA 轮询 → 全天修复/调度/校验 → 跨块协调。LLM 异步提供候选，独立认证进程提供 LB；仅全天验证通过的方案能更新 UB，最终用 UB/LB 的认证间隙停止。

块 SA 使用搜索专用 QUBO（`qubo.search_qubo`）：成本取 incumbent 处最紧最优性割的线性代理，不含 theta 与最优性割松弛位；硬约束按行归一化后以 `sa_hard_weight` 加权；容量只含 `pmax·u`；跨块纯二进制约束代入 incumbent 后作为块内硬约束。能量单位是单个位的最大归一化成本，`sa_temperature`、`sa_cutoff` 同此单位，`sa_sweeps` 为每个温度的全位扫描次数。LLM 回复返回时若 incumbent 已变化，其相对请求快照改动的机组行会先整体、再逐台移植到当前 incumbent（来源 `llm_rebase`）。协调 LP 被 incumbent 截顶或失败时保留原乘子，L1 选择失败时回退主问题乘子，不再中止运行；状态见 `outer_history[*].coordination`。

每批只有一个搜索 QUBO，直接由稀疏 Q 构造 BINARY BQM，合并 `Qij+Qji`，不生成稠密 Ising 矩阵。每条链由采样器根据种子生成随机初态，每次一个 read。退火终态恢复固定变量后，先进行原始块候选评价／割学习，再走既有全天修复、调度和独立校验；通过校验才可更新 UB。温度从 `sa_temperature*energy_unit` 起逐次乘 `sa_cooling`，严格高于 `sa_cutoff*energy_unit` 时执行，倒数作为自定义 beta；每级执行 `sa_sweeps` 次顺序 Metropolis 扫描。[采样器 API](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/generated/dwave.samplers.SimulatedAnnealingSampler.sample.html)

建模和采样均计入时间；链间检查预算，单条链不能硬中断，超时按实记录。稀疏接口消除了强制稠密化，但 QUBO 本身仍可能较密。历史不同搜索策略的实验保留各自口径。

## v0.34.0 方法选择与归档

2026-10-20 对 classic10 确定性算例（10 机组、24 小时、10% 备用）做了 seed 17、每组 300 秒、关闭 LLM 的机制组合筛选。补跑 5 组，复用 3 组已验证结果；8 组均为 `TIME_LIMIT`，最终方案均通过物理与费用校验。

本版本按用户选择固定三项全关闭的普通 SA＋修复。对应已验证冷启动结果：UB **585904.353740**、认证 LB **564437.299938**、认证 gap **3.663918%**；独立参考最优值 **564446.375240**，成本差距 **3.801597%**。该参考值只用于事后比较，没有回填主算法。

组合筛选中三项全开得到最低成本 **585518.168990**，比最终选用方案低约 **0.0659%**；筛选仅含一个种子，不支持稳定优劣或普遍性能结论。完整 8 组数值及选择依据见 [archives/v0.34.0.json](archives/v0.34.0.json)。默认输入仍为当前 IEEE118 派生的 54 机组数据；classic10 输入保存在实验归档内。

[v0.34.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.34.0) 提供发布提交对应的 `core-source-v0.34.0.zip`、清理前资料 `experiment-archive-v0.34.0.zip`、逐文件 SHA256 清单和上传／清理验证记录。历史实验快照中的三个机制只作为留证材料保存；当前主线没有相关实现或开关。授权配置及许可证仅本地保留。

## v0.33.0 实验结果与归档

2026-10-19 对当前 54 机组算例只做了一次 seed 17 冷启动及一次继承终态的续跑。**0.0954555% 是续跑结果，不是冷启动收敛成绩。** 机器可读汇总见 [archives/v0.33.0.json](archives/v0.33.0.json)。

| 阶段 | 状态 | 可行 UB | 认证 LB | 认证 gap | 本段求解器计时 |
|---|---|---:|---:|---:|---:|
| 冷启动，1800 秒预算 | TIME_LIMIT | 3,815,908.922633 | 3,672,259.838667 | 3.76447884% | 1795.71 秒 |
| 继承终态，追加 7200 秒预算 | CONVERGED | 3,675,768.562529 | 3,672,259.838667 | **0.09545552%** | 5666.18 秒 |

两段累计求解器计时 7461.89 秒。冷启动结果通过最终方案及证明模型/割和数值回执核查；续跑只做最终方案独立物理与费用复核，未再做完整证明审计。续跑最大物理残差为 `4.55e-12 MW`，费用残差为零。

| 降本来源 | 冷启动阶段 | 续跑新增 |
|---|---:|---:|
| 块 SA | 58,465.20 | 8,783.55 |
| pool 拼接 | 176.84 | 3,412.23 |
| LLM 直接候选 | 113,740.71 | 97,077.12 |
| LLM 移植 | 16,708.10 | 30,867.46 |

冷启动进行了 536 次块 SA 调用和 3 次 LLM 请求（2 次成功、1 次超时）。续跑新增 1376 次块 SA 调用；第 4—11 次共 8 次 LLM 请求全部成功，并经回执确认复用同一会话。续跑开始时恢复已有 UB、LB、历史和计数，局部模型重建；没有引入参考解，也没有调整 SA 参数。两段协调分别为 3 次和 8 次，全部 `CAPPED`；运行未因协调中止，但乘子没有因此更新。

冷启动后的独立参考成本为 3,672,938.498673，求解器报告 `OPTIMAL`，自身 gap 为 0.00007637%。最终续跑 UB 相对这一参考成本高 0.07705176%；该比值与认证 gap 分列，参考方案及参考下界均未回填主算法。

冷启动的最终全量留证及结果保存等尾部工作另耗时 81.60 秒，含保存的墙钟时间并未严格限制在 1800 秒内。续跑采用轻量留证，记录开销 0.94 秒，最终保存 0.014 秒。本次同时改变了留证、LLM 会话方式及时间预算，且只有一个种子；上述来源记账不能用于单独证明会话复用或某项修改的因果增益。

完整资料发布于 [v0.33.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.33.0)。下载 `experiment-archive-v0.33.0.zip` 和 `archive-manifest-v0.33.0.json`，解压到项目外的新目录；`workspace/` 保留清理前的源码、两阶段输入/结果/请求/回执及本地诊断材料，含运行时源码快照。每个文件均有 SHA256，清理前先验证远端下载件。`core-source-v0.33.0.zip` 对应发布提交，默认 `RESUME_FROM=None`，避免引用已移入归档的本地路径。

授权文件和全局 Codex 会话库不上传；本地保留 `config.local.json` 与许可证。复用历史 LLM 会话需要原 Codex 配置目录中的会话记录；仅解压数值检查点并不能恢复该会话。`latest-solution.json` 只是方案快照，不是完整搜索检查点。

[2026-10-18 历史资料归档](https://github.com/SadPull/LLM-benders-SUC/releases/tag/archive-20261018-lean)继续保留更早实验和清理前资料，包括旧版的 `COORDINATION_FAILED` 结果。本地仅保留当前可运行代码、输入、必要测试和本次汇总；历史结果不会因清理而改写。
