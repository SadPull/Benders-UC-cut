# LLM Benders SUC

Python 机组组合求解器：时间分块 D-Wave 开源 SA 搜索可行上界，LLM（Codex CLI 或流式 HTTP Responses）提供异步上界候选，独立数值进程认证下界。

v0.35.0 新增流式 HTTP 通道（`LLM_TRANSPORT="http"`），LLM 改为只输出改动机组的全天开机区间，由程序以提示中的 `incumbent_u` 还原矩阵；候选逐个校验，不合格者单独剔除并记录原因。v0.34.0 起固定采用**随机初态 SA＋修复**，原始候选评价／割学习、全天修复与物理校验、独立下界认证继续使用原有流程。

当前默认输入为 Pisa IEEE118 派生的 **54 机组、24 小时、单场景、零备用**；启动费按 `fixedCost` 列映射为零。只修改 `data/units.json` 和 `data/stochastic.json` 即可更换输入，来源与费用口径见 `data/provenance.json`。其他算例在 `instances/<名称>`（classic10、rts24、rcuc20），把 `run_experiment.py` 的 `DATA_DIR` 改为对应目录即可。

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

在 VS Code 中直接运行 `run_experiment.py`。预算、种子、LLM 开关等参数位于文件顶部；`CHECK_ONLY=True` 只检查输入。`LLM_TRANSPORT="codex"`（默认）使用 Codex CLI 已有登录、`gpt-6-astra/max`，不需要在项目中填写 API 密钥；`LLM_TRANSPORT="http"` 以流式 SSE 调用 `config.local.json` 中 `url`（Responses 端点）与 `api_key`，模型取 `LLM_MODEL`，请求带 `text.format` JSON schema。流式是必需的：中转在约 300 秒处会切断无数据的非流式长推理请求。缺少 LLM 时可设置 `LLM_ENABLED=False`。

LLM 每个候选只列出与 `incumbent_u` 不同的机组及其全天开机区间（1 起始、首尾包含，空列表表示全天关机），程序以请求中的 `incumbent_u` 还原完整矩阵；`llm_requests/` 或 `result.json` 同时保留原样改动 `raw_changes` 和还原矩阵 `raw_u`。机组越界、重复、区间越界或重叠的候选单独剔除，记录 `invalid_reason` 并在下一次调用的反馈中告知模型；其余候选照常评估，全部不合格时该次调用记为 `NO_VALID_CANDIDATE`。

**升级层（可选）**：`LLM_ESCALATION_MODEL` 填入模型名（如 `gpt-6-astra`）即开启，常规层须为 `LLM_TRANSPORT="http"`。常规层一次调用使 UB 降幅低于 `LLM_ESCALATION_GAIN`（默认 0.5%）或调用失败时，升级层被武装，并经 Codex CLI 以推理 max 并行发起一次请求；常规层照常继续，两层回复都走同一评估与移植流程。升级层每次发送完整提示、不续接会话；剩余预算扣除预留后不足 `LLM_ESCALATION_MIN_SECONDS`（默认 700 秒）时不发起，单次最多 `LLM_ESCALATION_TIMEOUT_SECONDS` 且不超过剩余预算。升级请求有效改进后清除期间累积的武装，早于该改进发出的常规请求不再触发升级；连续两次失败只停用升级层。每次升级调用在新建的临时 `CODEX_HOME` 中运行，其中只有登录信息副本和去掉 memories、插件、hooks、MCP、桌面与项目设置的配置，不含全局 AGENTS.md，并加 `--ephemeral`；调用结束即删除，删除结果记入回执 `codex_home_cleanup`。`result.json` 的每次调用标明 `tier`（regular／escalation）、`transport`、`triggered_by` 与 `deadline_seconds`，`summary.json` 汇总升级次数、有效改进次数与停用原因。

入口每次从零开始，预算 7200 秒、seed 17、目标认证 gap 0.1%；另用最多 120 秒求参考解，其结果不回填主算法。v0.36.0 起删除了续跑（`RESUME_FROM`）：v0.33.0 起不再写全量检查点，而现存检查点都不满足当前 SA 标识，续跑旧实验请使用对应的归档版本。

每轮遍历全部时间块，未闭合的块各最多采样一批 SA：一批为 `sa_reads` 条独立退火链，种子由配置种子、调用序号和链号派生，再取模 `2**31` 传给 D-Wave；诊断同时保留派生种子和实际采样种子。候选拼接使用该批开始时的当前最优全天方案；保留调度去重。`LLM_FIRST_CALL_AT_START=True` 时初始化后立即发首个请求，之后连续 `LLM_PATIENCE` 个完整轮改善不足 0.05% 时触发；`LLM_MAX_CALLS=None` 取消全局次数上限，单次最多 `LLM_TIMEOUT_SECONDS` 秒、预留 60 秒。剩余时间不足完整等待窗口时不新发请求，已有在途请求仍正常处理。Codex 后续调用续接同一会话，只发送动态快照（`prompt_mode=delta`）；HTTP 通道每次发送完整提示。会话失效或连续两次失败时先新开一次会话，再失败则停用 LLM，剩余预算继续 SA。全天或局部调度出现 SUBOPTIMAL 等数值状态时只拒绝该候选（计入 `dispatch_failures`），不中止运行。

结果自动写入新建的 `results/manual-seed.../`：

- `summary.json`：状态、UB/LB/gap、耗时和统计。
- `result.json`：精简主结果、最终方案、计时与调用摘要（含每次 LLM 调用的 usage 与回复原文）。
- `reference.json`：仅冷启动运行时生成的独立参考解。
- `llm_requests/`：Codex 通道或升级层实际触发时的请求、提示、命令行、事件流与回执（不含登录信息）。
- `snapshot/`、`manifest.json`、`effective-options.json`：实际源码与输入快照、SHA256、依赖版本和脱敏后的完整有效配置。
- `progress.jsonl`、`latest-solution.json`：轻量进度与原子更新的最新可行方案；最新方案不含搜索状态。

不保存全量检查点，也不序列化全部历史采样；记录开销与最终保存耗时单列。

manifest、精简结果及运行诊断记录 `sa_backend` 与 `sa_initialization`（当前为 `random_sa_repair_v1`）。采样诊断包含 beta 序列，以及每条链的原始／实际种子、退火终态与 QUBO 能量；全天提议记录只保存原始方案的 SHA256 与固定位违例数，只有 LLM 提议另存 `raw_u`／`repaired_u` 以生成反馈。这些诊断可经 `run_suc` 返回对象读取，不自动全量落盘。

读取精简结果：

```python
import json
with open("results/你的运行目录/result.json", encoding="utf-8") as f:
    result = json.load(f)
```

直接运行 `run_experiment.py`；`run_suc.py` 仅提供 `from run_suc import run_suc` 的 Python 接口。旧命令行参数及 `--output` JSON 输出入口已删除。

机组输入必须明确提供 `initial_output`。运行环境固定为上述 Windows／Python 3.10，TOML 解析直接使用已声明的 `tomli` 依赖。

## 核心文件

- `suc_benders/solver.py`：结果类型与主循环，只安排初始化、分块轮询、协调、建议和认证。
- `suc_benders/search.py`：单块搜索及共享的全天候选修复、调度、校验、去重和预算。
- `suc_benders/benders.py`：连续调度、Benders 割、块主问题和跨块乘子协调。
- `suc_benders/strengthened_*.py`：独立强化证明模型，只返回认证下界。
- `suc_benders/upper_advisor.py`：异步 LLM 建议调度（常规层批次与并行升级层）；`llm.py` 负责提示、输出 schema、流式传输与候选还原，`advisor_request.py` 负责 HTTP／Codex 工作进程与升级层的隔离 `CODEX_HOME`。
- `run_suc.py`、`run_experiment.py`：程序接口与一键运行入口。
- `data/`：当前输入、原始算例与溯源信息；`instances/`：classic10、rts24、rcuc20 输入。
- `tests/`：当前分块主线的必要回归；执行 `py -3.10 -m pytest -q tests`。
- `config.local.json`、`.local/*.lic`：仅本地保留的授权配置。

主线顺序：输入建模 → 连续松弛与初始可行解 → 固定乘子的分块 SA 轮询 → 全天修复/调度/校验 → 跨块协调。LLM 异步提供候选，独立认证进程提供 LB；仅全天验证通过的方案能更新 UB，最终用 UB/LB 的认证间隙停止。

块 SA 使用搜索专用 QUBO（`qubo.search_qubo`）：成本取 incumbent 处最紧最优性割的线性代理，不含 theta 与最优性割松弛位；硬约束按行归一化后以 `sa_hard_weight` 加权；容量只含 `pmax·u`；跨块纯二进制约束代入 incumbent 后作为块内硬约束。能量单位是单个位的最大归一化成本，`sa_temperature`、`sa_cutoff` 同此单位，`sa_sweeps` 为每个温度的全位扫描次数。LLM 回复返回时若 incumbent 已变化，其相对请求快照改动的机组行会先整体、再逐台移植到当前 incumbent（来源 `llm_rebase`）。协调 LP 被 incumbent 截顶或失败时保留原乘子，L1 选择失败时回退主问题乘子，不再中止运行；状态见 `outer_history[*].coordination`。

每批只有一个搜索 QUBO，直接由稀疏 Q 构造 BINARY BQM，合并 `Qij+Qji`，不生成稠密 Ising 矩阵。每条链由采样器根据种子生成随机初态，每次一个 read。退火终态恢复固定变量后，先进行原始块候选评价／割学习，再走既有全天修复、调度和独立校验；通过校验才可更新 UB。温度从 `sa_temperature*energy_unit` 起逐次乘 `sa_cooling`，严格高于 `sa_cutoff*energy_unit` 时执行，倒数作为自定义 beta；每级执行 `sa_sweeps` 次顺序 Metropolis 扫描。[采样器 API](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/generated/dwave.samplers.SimulatedAnnealingSampler.sample.html)

建模和采样均计入时间；链间检查预算，单条链不能硬中断，超时按实记录。稀疏接口消除了强制稠密化，但 QUBO 本身仍可能较密。历史不同搜索策略的实验保留各自口径。

## v0.36.0 升级层、IEEE118 补跑与代码清理

2026-10-21 在 IEEE118 派生算例（54 机组×24 小时×1 场景）、seed 17、每组 1800 秒、12 h 分块、16 sweeps 条件下补跑 SA+DeepSeek（`deepseek-v4.1-flash`，推理 max，HTTP 流式），并试验经 Codex 的 `gpt-6-astra` 升级层。参考成本为 Gurobi 完整问题最优值 3,672,938.50，只用于事后比较，不回填主算法。机器可读汇总见 [archives/v0.36.0.json](archives/v0.36.0.json)。

| 运行 | 代码 | UB 相对最优值 | 认证 gap | LLM 调用 |
|---|---|---:|---:|---|
| 纯 SA | `ffaaf1d` | +7.515% | 7.007% | 0 |
| DeepSeek，增量输出修复后第一次 | `017d925` | +2.103% | 2.078% | 3（全部 OK） |
| DeepSeek，补全调用记录后 | `2ac87c2` | +2.046% | 2.023% | 3（全部 OK） |
| DeepSeek，可达出力缺口反馈 + 450 s 超时 | `ffaaf1d` | +2.354% | 2.317% | 5（4 OK，1 超时） |
| DeepSeek 单独（混合对照组） | `f8054a8` | +2.691% | 2.639% | 4（全部 OK） |
| DeepSeek + astra 升级层 | `f8054a8` | **+1.465%** | **1.462%** | 常规 4 + 升级 1 |

全部为 `TIME_LIMIT`；纯 SA 在四个代码版本上的结果相同（UB 3,948,959.48）。

- **增量输出修复有效**：v0.35.0 的整矩阵输出曾使 DeepSeek 组停在 +4.934%；改为增量输出与逐候选校验后，各次运行都在 +2.0%～+2.7%。
- **可达出力缺口反馈**：调度不可行的候选此前都是"计入爬坡和开机首小时后可达出力低于负荷"；把缺口小时与 MW 数反馈给模型后，候选调度可行率从 6/12 升到 15/16，但这次的最终 UB（+2.354%）并未更好。
- **升级层**：混合组 +1.465% 是目前 IEEE118 的最好结果，但 91.1% 的降幅来自第 1 次 DeepSeek 调用（升级层尚未介入），astra 只贡献 0.5%；这一对的差距不能归因于升级层。停滞点探针中，同一提示下 `gpt-6-astra` 4/4 候选可行、最好 +1.490%，DeepSeek 2/4、+1.901%（各一次调用）。
- **局限**：每组只有一个种子、一次运行；DeepSeek 推理本身有随机性，同条件下 DeepSeek 单独运行的结果在 +2.046%～+2.691% 之间。各报告见 `results/ieee118-deepseek-fix*-20261021`、`results/ieee118-hybrid-20261021` 与 `results/astra-stall-probe-20261021`。

同时做了一次代码清理：删除不可达或无读者的代码（生产代码 4900 → 4216 行），拆分最难读的函数；回归脚本在每一步逐字比对 UB 轨迹、搜索统计、172 次故障注入下的状态与 LLM 请求体，真实证明进程给出相同的界，数值行为不变。对外可见的变化：

- 删除续跑（`RESUME_FROM`、`run_suc(resume_state=)`）、手动文件交换 LLM 通道（`agent_bridge.py`）、归档的整割 `qubo()` 编码及其选项 `theta_step`／`penalty`／`reserve_scale`，以及从未启用的审计落盘钩子。
- `summary.json` 去掉 `scope`、`audit_status`、`resume_source`、`segment_*`、`new_llm_calls`、`block_sa_calls`（恒等于 `sa_calls`），`segment_wall_seconds` 改名 `solver_wall_seconds`；`manifest.json` 去掉 `cold_start`、`inherited_evidence`、`resume_source`；`progress.jsonl` 与 `latest-solution.json` 去掉 `resume_cursor`、`segment_elapsed_seconds`，运行中的状态显示 `RUNNING`。
- LLM 提示去掉恒为 0／null／空的 `optimality_cuts`、`hamming_to_incumbent`、`infeasible_diagnostics` 和 `advice_routing`（每次约少 190 个字符）；上表的运行都使用清理前的提示，此后的结果不能与之逐字比较。
- 证明进程每轮透视割分离只计算一次模型指纹（IEEE118 派生算例首轮从 0.83 秒降到 0.01 秒）；块 SA 诊断在内存中只保留标量与哈希（classic10 上 8 次调用后从 0.65 MB 降到 0.09 MB）。

本地 `results/` 现保留 12 项：8 个分析／探针目录（`deepseek-flash-pilot-20261020`、`ieee118-deepseek-20261020`、`ieee118-deepseek-fix-20261021`、`ieee118-deepseek-fix2-20261021`、`ieee118-deepseek-fix3-20261021`、`ieee118-hybrid-20261021`、`astra-stall-probe-20261021`、`schema-probe-20261020`），以及 fix3 与 hybrid 两组对照的原始运行（`manual-seed17-20261021-104528-*`、`manual-seed17-20261021-142727-*`）。fix、fix2 两轮的原始运行只保存在 Release 的实验归档中，10-20 的首轮对照在 v0.35.0 Release 中。

[v0.36.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.36.0) 提供发布提交对应的 `core-source-v0.36.0.zip`、2026-10-21 全部 13 项实验目录 `experiment-archive-v0.36.0.zip`、逐文件 SHA256 清单 `archive-manifest-v0.36.0.json` 和资产校验和 `SHA256SUMS.txt`。

## v0.35.0 LLM 通道、多算例实验与归档

2026-10-20 在 seed 17、单次运行条件下完成以下实验，均与同条件纯 SA 成对比较。参考成本为 Gurobi 完整问题最优值，只用于事后比较，不回填主算法。机器可读汇总见 [archives/v0.35.0.json](archives/v0.35.0.json)。

| 算例（机组×小时×场景） | 分块 | LLM | 纯 SA UB 差距 | SA+LLM UB 差距 | SA+LLM 状态 |
|---|---|---|---:|---:|---|
| RCUC20（20×24×1） | 3 h | gpt-6-astra，Codex | +3.616% | +0.043% | CONVERGED，1353 s |
| RCUC20（20×24×1） | 12 h，16 sweeps | gpt-6-astra，Codex | +0.903% | +0.000% | CONVERGED，1429 s |
| classic10（10×24×3） | 12 h，16 sweeps | gpt-6-astra，Codex | +3.770% | +0.051% | CONVERGED，1092 s |
| RTS-24（32×24×3） | 12 h，16 sweeps | gpt-6-astra，Codex | +5.475% | +1.108% | TIME_LIMIT，1800 s |
| RTS-24，5400 s 预算 | 12 h，16 sweeps | gpt-6-astra，Codex | +5.475% | +0.040% | CONVERGED，2255 s |
| IEEE118 派生（54×24×1） | 12 h，16 sweeps | deepseek-v4.1-flash，HTTP 流式 | +7.515% | +4.934% | TIME_LIMIT，1800 s |

- **12 h 分块**：RCUC20 敏感性筛选表明 3 h 块会被长最小停机时间锁在块坐标局部最优，12 h 块配 16 sweeps 是该算例最好的纯 SA 设置；其余算例沿用此设置，未逐算例调参。
- **LLM 延迟**：gpt-6-astra（max）单次 391–900 s，主要是推理 token 以约 28 tok/s 生成；1800 s 内只能完成 2 次调用。同一首调 prompt 下，deepseek-v4.1-flash（max）单次 156–277 s、约 220 tok/s，推理 token 为 gpt 的 3–5 倍，候选质量与 gpt 首调相当。
- **非流式 HTTP 会在约 300 秒处被中转切断（HTTP 503）**，因此 HTTP 通道改为流式；IEEE118 中 315 秒的调用传输完整。
- **整矩阵输出不可靠**：IEEE118 的 4 次 DeepSeek 调用中有 2 次在 54×24 矩阵里漏写一行（均位于连续相同的全天开机行中间，其余改动正确），旧解析规则将整个回复作废。v0.35.0 改为增量输出、JSON schema 约束和逐候选校验；同一 IEEE118 首调 prompt 的试探中 4 个候选全部合格且可行，正文输出从约 10,600 token 降到 431 token，单次调用把 UB 从 4,004,999.77 降到 3,847,762.95（+4.760%）。修复后尚未重跑完整求解。
- **Codex 通道注入用户级配置**：Codex CLI 会带入个人 memory、AGENTS.md 和插件提示（每次多约 2.4 万输入 token），其中 memory 含历史实验信息，包括 RTS-24 参考最优值。gpt 组结果应在此限制下解读；HTTP 通道不受影响。此后加入的升级层改为在隔离的临时 `CODEX_HOME` 中运行，不再带入这些内容；单独使用的 Codex 通道（`LLM_TRANSPORT="codex"`）仍照旧。

局限：每组只有一个种子、一次运行；gpt 组与 DeepSeek 组的通道、会话方式和输出格式都不同，不能直接比较模型；v0.35.0 改变了提示与输出格式，此后结果不能与整矩阵输出的历史结果直接比较。

[v0.35.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.35.0) 提供发布提交对应的 `core-source-v0.35.0.zip`、全部 34 项实验目录与驱动／分析脚本 `experiment-archive-v0.35.0.zip`、逐文件 SHA256 清单和上传／清理验证记录。本地 `results/` 只保留 DeepSeek 相关的五项：`deepseek-flash-pilot-20261020`、`ieee118-deepseek-20261020`、`manual-seed17-20261020-224136-14459f84`、`manual-seed17-20261020-224136-7326d548`、`schema-probe-20261020`。

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
