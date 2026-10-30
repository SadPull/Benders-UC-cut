# LLM Benders SUC

Python 机组组合求解器：先用 Gurobi 求完整模型的参考方案，再由时间分块 D-Wave 开源 SA 搜索可行上界，LLM（Codex CLI 或流式 HTTP Responses）提供异步候选。全局评价采用相对参考成本的差距；项目不计算下界。

v1.4.0 在 RTS24 上做程序演化训练（种子种群从 L24 起步，接口放宽到每个候选 ≤12 台），得到程序 L24x；2 种子完整验证中 SA+L24x 优于 SA+R、SA+L118 和 SA+L24，见下文“v1.4.0”一节。v1.3.0 清理仓库：只保留求解器、离线程序算子、输入和测试；论文、训练、验证、作图与修复上限实验的脚本，以及尚未封存的实验结果，都移到 v1.3.0 Release 中，见下文“v1.3.0”一节。v1.2.0 在 RTS24 上做了 10 轮闭环程序训练，得到程序 L24，通过小型验证实验的三个预登记判据，并统一了方法与程序的命名，见下文“命名”与“v1.2.0”两节。v1.1.0 加入爬坡可达出力修复，以及多场景 IEEE118 和 RTS24 上 SA 与 SA+L118（发布时称 A 与 L+SA）的对比实验，见下文“v1.1.0”一节。v1.0.0 删除了从未生效的跨块拉格朗日协调与外层循环，块搜索改为全程保留，见下文“v1.0.0”一节。v0.35.0 新增流式 HTTP 通道（`LLM_TRANSPORT="http"`），LLM 改为只输出改动机组的全天开机区间，由程序以提示中的 `incumbent_u` 还原矩阵；候选逐个校验，不合格者单独剔除并记录原因。v0.34.0 起固定采用**随机初态 SA＋修复**。当前保留原始候选评价／割学习、全天修复与物理校验，已删除透视割和独立强化证明进程；下方历史版本实验仍保留当时的认证口径。

当前默认输入为 Pisa IEEE118 派生的 **54 机组、24 小时、单场景、零备用**；启动费按 `fixedCost` 列映射为零。只修改 `data/units.json` 和 `data/stochastic.json` 即可更换输入，来源与费用口径见 `data/provenance.json`。其他算例在 `instances/<名称>`（classic10、rts24、rcuc20，以及 10 场景的 ieee118s10），把 `run_experiment.py` 的 `DATA_DIR` 改为对应目录即可。

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

入口每次从零开始，先用最多 120 秒独立预算求参考解，再给搜索完整的 7200 秒预算，seed 17、目标参考成本差距 0.1%。参考方案经过独立物理与费用校验；只有状态为 `OPTIMAL` 的有效参考才允许提前停止。仅有未证最优的可行参考时只比较、不据此提前停止；参考失败或没有有效方案时仍继续搜索，参考成本和差距为空。参考方案不用于初始化、候选注入，参考成本不加入 LLM 提示。v0.36.0 起删除了续跑（`RESUME_FROM`），续跑旧实验请使用对应归档版本。

参考差距为 `100 × (UB - reference_cost) / abs(reference_cost)`，允许负数；参考成本为零时百分比为空。停止要求有效 `OPTIMAL` 参考，且 `UB - reference_cost <= 1e-5 + tolerance × abs(reference_cost)`；零参考成本只用绝对容差。达到目标记为 `REFERENCE_TARGET_REACHED`，搜索轮次或 SA 配额耗尽并收尾当前 LLM 工作后记为 `SEARCH_EXHAUSTED`（`search_stop_reason` 为 `MAX_ROUNDS`、`SA_BUDGET` 或 `SA_DISABLED`），只有截止时间信号（`TimeoutError`）记为 `TIME_LIMIT`；其他异常无论是否已过截止时间都记为 `<阶段>_FAILED`（阶段为 `INITIALIZATION`、`SEARCH_SETUP`、`ROUND_<k>_BLOCK_<j>`、`ADVISOR`）。没有待处理工作时不空等剩余预算。

初始化后每块建立一个持久块搜索，整个运行不重建，块内割与调度缓存跨轮保留；每轮遍历全部时间块，每块采样一批 SA（`max_rounds` 限制轮数，入口不限）：一批为 `sa_reads` 条独立退火链，种子由配置种子、调用序号和链号派生，再取模 `2**31` 传给 D-Wave；诊断同时保留派生种子和实际采样种子。候选拼接使用该批开始时的当前最优全天方案；保留调度去重。`LLM_FIRST_CALL_AT_START=True` 时初始化后立即发首个请求，之后连续 `LLM_PATIENCE` 个完整轮改善不足 0.05% 时触发；`LLM_MAX_CALLS=None` 取消全局次数上限，单次最多 `LLM_TIMEOUT_SECONDS` 秒、预留 60 秒。剩余时间不足完整等待窗口时不新发请求，已有在途请求仍正常处理。Codex 后续调用续接同一会话，只发送动态快照（`prompt_mode=delta`）；HTTP 通道每次发送完整提示。会话失效或连续两次失败时先新开一次会话，再失败则停用 LLM，剩余预算继续 SA。全天或局部调度出现 SUBOPTIMAL 等数值状态时只拒绝该候选（计入 `dispatch_failures`），不中止运行。

结果自动写入新建的 `results/manual-seed.../`：

- `summary.json`：状态、UB、参考元数据、`reference_gap_percent`、耗时和统计，以及 `complete_rounds`、`search_stop_reason`、`sa_budget_exhausted`。
- `result.json`：精简主结果、最终方案、计时与调用摘要（含每次 LLM 调用的 usage 与回复原文）。
- `reference.json`：先行参考求解的状态、完整方案、校验和独立耗时；失败时保留原因。`gurobi_bound`／`gurobi_gap_percent` 是 Gurobi 自身的界与间隙，只说明参考离最优多远，不参与停止判断。
- `llm_requests/`：Codex 通道或升级层实际触发时的请求、提示、命令行、事件流与回执（不含登录信息）。
- `snapshot/`、`manifest.json`、`effective-options.json`：实际源码与输入快照、SHA256、依赖版本和脱敏后的完整有效配置。
- `progress.jsonl`、`latest-solution.json`：轻量进度与原子更新的最新可行方案；最新方案不含搜索状态。

不保存全量检查点，也不序列化全部历史采样；记录开销与最终保存耗时单列。

当前主结果删除全局 `lower_bound`／`lower`、`relative_gap`／`gap_percent` 和 `certification_history`，不计算下界：SA 只能给出上界，项目历史上的各种下界（连续松弛、精确 Benders 主问题、拉格朗日）都由 Gurobi 求得，评价直接使用 Gurobi 参考成本。`Result.reference` 保留完整参考记录，`reference_cost` 与 `reference_gap_percent` 是随当前已验证方案计算的只读属性。进度和最新方案同时带有 `reference_status`。`elapsed_seconds`／`solver_elapsed_seconds` 只计搜索，`reference_elapsed_seconds` 单列参考耗时，`total_elapsed_seconds` 计两段总耗时。历史 `results/`、`archives/` 和源码快照不迁移，旧分析脚本需适配新字段与状态。

manifest、精简结果及运行诊断记录 `sa_backend` 与 `sa_initialization`（当前为 `random_sa_repair_v1`）。采样诊断包含 beta 序列，以及每条链的原始／实际种子、退火终态与 QUBO 能量；全天提议记录只保存原始方案的 SHA256 与固定位违例数，只有 LLM 提议另存 `raw_u`／`repaired_u` 以生成反馈。这些诊断可经 `run_suc` 返回对象读取，不自动全量落盘。

读取精简结果：

```python
import json
with open("results/你的运行目录/result.json", encoding="utf-8") as f:
    result = json.load(f)
```

直接运行 `run_experiment.py`；`run_suc.py` 提供 `from run_suc import run_suc` 的 Python 接口，同样只计算一次先行参考。配置 `reference_time_limit` 默认 120 秒，对应入口顶部 `REFERENCE_TIME_LIMIT_SECONDS`；`time_limit` 为独立搜索预算。可选关键字 `reference_log_file` 指定参考日志；`started_at` 是整体调用的计时起点，不扣减参考结束后新建的搜索预算。旧命令行参数及 `--output` JSON 输出入口已删除。

机组输入必须明确提供 `initial_output`。运行环境固定为上述 Windows／Python 3.10，TOML 解析直接使用已声明的 `tomli` 依赖。

## 命名

每个方法都是 SA+Benders 加一个算子，记为 `SA+<算子>`。离线 LLM 程序按生成时所用的系统命名，早期轮次的程序再加 `r<轮>`。

| 名称 | 含义 | v1.2.0 之前的名称 |
|---|---|---|
| `SA` | 只有 SA+Benders | `A` |
| `SA+R` | 加随机区间算子 R，不用 LLM；训练中的手写对照 R 是同一个算子 | `N+SA`；训练对照 `R` |
| `SA+L118` | 加程序 L118（`programs/L118.py`），在 IEEE118 上经两轮生成 | `L+SA`、`L3+SA`；程序 `L3` |
| `SA+L24` | 加程序 L24（`programs/L24.py`），RTS24 上 10 轮闭环训练的终选程序（第 9 轮的 P9b） | `LIT+SA`；程序 `LIT` |
| `SA+L24r2` | 加程序 L24r2（`programs/L24r2.py`），同一训练第 2 轮的最佳程序（P2a） | `LIT1+SA`；程序 `LIT1` |
| `SA+L24x` | 加程序 L24x（`programs/L24x.py`），v1.4.0 程序演化训练的终选程序（第 3 轮的 P3a）；每个候选可改 ≤12 台 | — |

训练中的候选程序记为 `P{轮}{a|b}`，手写对照为 `R`、`H0`–`H3`；判据编号（论文实验的 M1／M2、验证实验的 T1–T3）和图号 F1–F6 不变。已归档的结果目录（`results/paper-*`、`results/lit-*`）及其中的报告、图和运行记录保留实验时的名称，用各自 `snapshot/` 中的代码复现。这些目录只在 Release 资产中，见“v1.3.0”一节的封存表。

## 核心文件

- `suc_benders/solver.py`：结果类型与主循环，安排参考、初始化、分块轮询和建议。
- `suc_benders/model.py`：稀疏模型、按小时分块（`partition`）、跨块定价（`priced`、`relaxed_rows`、`search_multipliers`）。
- `suc_benders/search.py`：持久的原始块搜索及共享的全天候选修复、调度、校验、去重和预算。
- `suc_benders/benders.py`：离散可行性检查、固定二进制后的连续调度与 Benders 割、块内硬约束行（`hard_rows`）。
- `suc_benders/reference.py`：Gurobi 完整模型参考求解及独立方案校验。
- `suc_benders/upper_advisor.py`：异步 LLM 建议调度（常规层批次与并行升级层）；`llm.py` 负责提示、输出 schema、流式传输与候选还原，`advisor_request.py` 负责 HTTP／Codex 工作进程与升级层的隔离 `CODEX_HOME`。
- `run_suc.py`、`run_experiment.py`：程序接口与一键运行入口。
- `data/`：当前输入、原始算例与溯源信息；`instances/`：classic10、rts24、rcuc20、ieee118s10 输入。
- `offline_pilot.py`、`offline_worker.py`、`programs/`：离线 LLM 程序 L118、L24、L24r2、L24x 的沙箱调用，以及 SA、SA+R、SA+L118、SA+L24r2、SA+L24、SA+L24x 的冷启动运行（`offline_pilot.run_trial`）；SA 与 SA+L118 与 v1.1.0 逐位一致。每个候选最多改动的机组数默认 `MAX_UNITS = 4`，`ARM_MAX_UNITS` 按方法覆盖（SA+L24x 为 12）。
- `tests/`：求解器与离线程序算子的回归；执行 `py -3.10 -m pytest -q tests`。
- 实验脚本（论文实验、程序训练与验证、作图、修复上限实验）自 v1.3.0 起不在仓库中，完整代码在 Release 资产 `experiment-source-v1.3.0.zip` 中，解压即可运行，见“v1.3.0”一节；v1.4.0 程序演化训练与验证的脚本在 `experiment-source-v1.4.0.zip` 中。
- `config.local.json`、`.local/*.lic`：仅本地保留的授权配置。

主线顺序：输入建模 → 独立参考求解 → 连续松弛与初始可行解 → 固定价格的持久分块 SA 轮询 → 全天修复/调度/校验。LLM 异步提供候选；仅全天验证通过的方案能更新 UB。

**跨块协调**：跨块行指变量落在不同时间块的约束。块之间不做乘子迭代，而是围绕当前最优全天方案（incumbent）做块坐标搜索：纯二进制跨块行（状态转移、最小开停机）把块外变量代入 incumbent 后作为块内硬约束；含连续变量的跨块行（跨边界爬坡、块末停机容量、冷启动窗口）在块费用中按初始化时连续松弛的对偶价格定价（`search_multipliers`），价格整个运行不变。块内候选一律经全天修复、调度和校验后才能更新 UB。

块 SA 使用搜索专用 QUBO（`qubo.search_qubo`）：成本取 incumbent 处最紧最优性割的线性代理，不含 theta 与最优性割松弛位；硬约束按行归一化后以 `sa_hard_weight` 加权；容量只含 `pmax·u`；跨块纯二进制约束代入 incumbent 后作为块内硬约束。能量单位是单个位的最大归一化成本，`sa_temperature`、`sa_cutoff` 同此单位，`sa_sweeps` 为每个温度的全位扫描次数。QUBO 只编码 anchor 处松弛量最小的 64 条可行割，每块至多保存 4096 条割（先进先出，丢弃数记入 `cuts_dropped`），块内缓存只存紧凑记录，使 QUBO 规模与内存不随轮数增长；回报给 LLM 的割数是累计数。LLM 回复返回时若 incumbent 已变化，其相对请求快照改动的机组行会先整体、再逐台移植到当前 incumbent（来源 `llm_rebase`）。没有块搜索时（如关闭 SA 后收尾阶段消费的回复），候选的局部可行性记为未知。

每批只有一个搜索 QUBO，直接由稀疏 Q 构造 BINARY BQM，合并 `Qij+Qji`，不生成稠密 Ising 矩阵。每条链由采样器根据种子生成随机初态，每次一个 read。退火终态恢复固定变量后，先进行原始块候选评价／割学习，再走既有全天修复、调度和独立校验；通过校验才可更新 UB。温度从 `sa_temperature*energy_unit` 起逐次乘 `sa_cooling`，严格高于 `sa_cutoff*energy_unit` 时执行，倒数作为自定义 beta；每级执行 `sa_sweeps` 次顺序 Metropolis 扫描。[采样器 API](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/generated/dwave.samplers.SimulatedAnnealingSampler.sample.html)

建模和采样均计入时间；链间检查预算，单条链不能硬中断，超时按实记录。稀疏接口消除了强制稠密化，但 QUBO 本身仍可能较密。历史不同搜索策略的实验保留各自口径。

## v1.4.0 RTS24 程序演化训练与完整验证

2026-10-30。v1.2.0 的 L24 在 rts24-minus 上停在 2.184%。只把 L24 的接口放宽到 ≤8 台，就能在 60 秒内逃出，所以问题不在 LLM 写不出这个策略，而在 v1.2.0 的训练环境没给它学的机会：
1. 提示和检查器把每个候选限制在 ≤4 台；
2. 锚点全部取自 SA+R 的轨迹，从不包含程序自己的停滞点；
3. J 只奖励前 10 秒的改善，每轮只把一个最佳程序往下传；
4. 反馈里没有停滞诊断，也没有跨轮经验。

### 训练

结果目录 `results/evo-training-rts24-20261030-101731`。变体沿用 v1.2.0 的 train1–3 与 validation1–2，验证结果不给 LLM。
- 接口：每个候选改 1..12 台，每次调用至多 4 个候选，0.5 秒截止；提示只陈述上限。
- 锚点：initial、half、final（SA+R）之外加 **stall**，即当前最佳程序在训练变体上 30 秒无改进的停滞点，每有新的最佳程序就刷新；验证集的 stall 锚点在第 0 轮用 L24 固定。
- 适应度为 4 种锚点 J 的均值。MAP-Elites 存档按候选改动台数的中位数分 ≤4、5–8、9–12 三格；每轮请求两个程序，分别从最佳精英和另一格出发。
- 反馈加入每个 stall 锚点的停滞诊断（提议、拒绝原因、评估、改进数、候选台数分布），并保留 LLM 自写的 lessons。
- 种子种群 L24、L24r2、R；LLM 为 gpt-6-astra（effort high），每轮 490–975 秒。停止规则：3–6 轮，耐心 2。

**前检与修订**：原判据要求至少 2 个训练 stall 锚点被 ≤4 台尺子证明锁死（B4 无改进、B8 严格改进），结果 0 个满足，B4 仍有 0.06–0.81% 的小改进。看到结果后、调用 LLM 之前，经用户批准改为“B8 改进 ≥ 2×B4 改进”，train1（0.81% / 2.42%）和 train3（0.06% / 1.82%）满足。这是事后修订，见 `amendments/amendment-001.json`。

**过程**：验证适应度 2.299（L24）→ 2.706 → 2.784 → 2.807，第 3 轮后 CONVERGED，终选 P3a 即 L24x。L24x 的训练适应度 3.077。第 1 轮的 lessons 在提示未建议的情况下提出跨出 2–4 台的改动。各程序的候选台数中位数都 ≤4（5–8、9–12 格为空），即小改动为主、夹杂 5–10 台的大改动。

**训练后检查**（`evaluation/`，都通过）：
- E1：从 v1.2.0 的停滞点出发，60 秒内 minus s83 从 2.184% 降到 0.200%，s101 降到 0.173%；base s17、plus s83 没有逃出。
- E2：27–41% 的候选改动超过 4 台，逃出都来自这类候选。
- E3：冷启动 SA+L24x 对 SA+L24，minus 3/3 胜（s83、s137 达到参考，s101 0.228%），base 3/3 达到参考（0.006%、0%、0%，SA+L24 约 0.57%），AUC 中位数降低 80.6%。这一组 6 对并发运行。

### 完整验证

结果目录 `results/l24x-validation-20261030-120413`。规格同 v1.2.0，但种子减为 2 个：rts24-base 种子 17、29 各 600 秒，rts24-minus、rts24-plus 种子 83、101 各 300 秒；对照 SA+R、SA+L118、SA+L24。6 个 SA+L24x 运行串行；之后按用户要求 18 个对照每次 3 个并发（各自 Threads=1），见 `protocol-amendment-parallel.json`，并发只可能拖慢对照。中断的 rts24-base-s17-SA+R 从冷启动重跑。对照的结果与 v1.2.0 归档逐次一致，没有漂移（图 V5）。

最终差距 %／AUC：

| 算例 | 种子 | SA+R | SA+L118 | SA+L24 | SA+L24x |
|---|---:|---:|---:|---:|---:|
| rts24-base | 17 | 3.545／3.564 | 0.531／1.058 | 0.564／0.603 | **0.006／0.070** |
| rts24-base | 29 | 3.545／3.565 | 3.321／3.345 | 0.571／0.609 | **0.000／0.463** |
| rts24-minus | 83 | 3.908／3.960 | 2.240／2.425 | 2.184／2.247 | **0.000／0.193** |
| rts24-minus | 101 | 3.908／3.961 | 2.240／2.437 | 2.184／2.240 | **0.228／0.317** |
| rts24-plus | 83 | 3.968／3.994 | 3.710／3.813 | 0.704／0.768 | **0.095／0.743** |
| rts24-plus | 101 | 3.968／3.990 | 0.704／0.861 | 0.704／0.780 | 0.704／**0.762** |

- SA+L24x 对 SA+R 6/6 胜，AUC 中位数降低 89.5%；对 SA+L118 5 胜 1 平，降低 86.6%；对 SA+L24 5 胜 1 平，降低 54.9%。平局都是 plus s101，几种 LLM 方法都停在 0.704%。
- 3/6 次达到参考（REFERENCE_TARGET_REACHED）。
- 图在 `figures/`：V1 收敛曲线、V2 最终差距、V3 相对 SA+R 的 AUC、V4 候选台数分布、V5 对照漂移、T1 训练学习曲线与 stall J、T2 训练候选台数，数据见 `data.json`。

**与 Gurobi 的时间**（只作尺子）：Gurobi 在 base／minus／plus 上分别 92／83／42 秒证明最优。SA+L24x 约 4–5 秒到 ≤1%，到 ≤0.01% 需要 44、171、474 秒或在时限内达不到。在 RTS24 上它找最优解并不比 Gurobi 快。

**局限**：
- 训练、验证和测试都在 RTS24 上（32 台机组、3 个场景，只改负荷），跨系统的泛化没有检验。
- 验证集同时用于停止规则和终选；验证 stall 锚点由 L24 固定。
- 前检判据是事后修订的；E1 只有 2 个逃出点；完整验证只有 2 个种子，E3 是并发运行。
- `programs/L24x.py` 按字节保存（CRLF），SHA256 为 `d884c753…0762d8`；训练记录中的 `f0cf20da…681d38` 是同一内容换成 LF 后的哈希。

## v1.3.0 清理与封存

2026-10-28。仓库只保留方法核心，见“核心文件”一节。此前没有封存的实验结果和全部实验脚本都封存到 v1.3.0 Release；本地 `results/` 在资产下载校验后删除。

**移出的代码**：先在提交 5a72fbc 中完整入库，再在 5b05cc8 中删除。
- 论文实验：`paper_pilot.py`、`run_paper_pilot.py`、`paper_figures.py`、`make_ieee118_scenarios.py`。
- 程序训练与验证：`program_training.py`、`run_program_training.py`、`program_evaluation.py`、`run_program_evaluation.py`。
- 作图：`program_figures.py`。
- 修复上限实验：`repair_headroom.py`、`run_repair_headroom.py`。
- 以上脚本的 4 个测试文件。

`tests/test_offline_pilot.py` 改为直接读取 ieee118s10，选项不变。保留部分的测试为 152 passed。

### 修复上限实验（repair-headroom）

2026-10-28，不调用 LLM。

问题：SA 的块样本先由 `validation.repair()` 机械修复。如果修复是理想的，纯 SA 样本能不能变成改进？只有能，才值得让 LLM 来做修复。

做法：
- rts24 跑 5 个种子，ieee118s10 跑 2 个种子，每次 300 秒纯 SA，记录全部样本。
- 每个种子抽取与当时 incumbent 不同的去重样本：rts24 每种子 120 个。ieee118s10 每种子计划 60 个，在 oracle 时限内只完成了 38 个。
- 每个样本做精确最小改动修复 P0：先在完整模型上找与原样本 Hamming 距离最小的可行方案，再在这个距离内取真实成本最低的方案。
- Gurobi 只作诊断标尺；是否改进以独立调度校验后的成本为准。

预登记判据（rts24）：
- PASS：至少 3/5 个种子有改进样本，且合并改进率 ≥ 1%。
- FAIL：至多 1/5 个种子有改进样本，unresolved ≤ 10%，且每个种子重放至少 60 个样本。

| 算例 | 重放样本 | P0 改进 | 原样本离散可行 | 现有修复相对 incumbent（各种子中位数） | P0 相对 incumbent（各种子中位数） |
|---|---:|---:|---:|---:|---:|
| rts24 | 600 | 0 | 0 | +16.2% 至 +16.7% | +19.1% 至 +20.6% |
| ieee118s10 | 76 | 0 | 0 | +2.1% 至 +2.2% | +2.5% 至 +2.6% |

- rts24 上，把最小改动放宽 Δ=2 的 200 个样本同样 0 个改进。所有样本都没有 unresolved。
- 现有修复从不失败，但修复后的方案都比 incumbent 贵。P0 的改动约为现有修复的一半（rts24 中位 14–16 位，现有修复为 32–37.5 位），成本反而更高。
- SA 样本离 incumbent 很远：rts24 的中位距离为 138–140 位（全天共 768 位），ieee118s10 为 184.5–186 位。
- 对照（只作描述），相对 incumbent：

  | 算例 | 只放开块 1（C_b） | 只放开块 2（C_b） | incumbent 的 Hamming-2 球 |
  |---|---:|---:|---:|
  | rts24 | −4.23% | −0.92% | −0.61% |
  | ieee118s10 | −1.71% | −1.00% | −0.19% |

  也就是说，改进就在 incumbent 附近，但 SA 从来没有采到那里。
- **结论**：FAIL。按预登记的解读规则，问题出在 SA 样本本身（QUBO 费用代理或退火），不在修复。因此不做 LLM 修复。
- **更正**：`report.md` 和 `summary.json` 写的是 INVALID。原因是复核 5 把 rts24-s67 的 `sa:138:2` 判为与重算的修复不一致，而这条记录在记录运行到截止时间时还没有评估，`commitment_id` 为空。该样本无论记为 N/A 还是排除（剩 599 个），判据都是 FAIL。说明见目录下的 `corrections/correction-001.json`，由 `corrections/correction_001.py` 生成；原文件均未改动。

### 封存位置

`results/` 不进仓库。各结果目录的完整副本：

| 目录 | Release 资产 |
|---|---|
| `results/paper-reference-20261025-124811-c6f41da2`、`results/paper-compare-20261025-125905-c264cdcf`、`results/paper-case-20261025-133936-d0ed2658` | v1.1.0 `experiment-archive-v1.1.0.zip` |
| `results/lit-training-rts24-20261026-170730-82fb090e`、`results/lit-evaluation-rts24-20261026-213117-7bb610af`、`results/_lit_logs` | v1.2.0 `experiment-archive-v1.2.0.zip` |
| `results/repair-headroom-20261028-112213-f5f97cef`（含更正） | v1.3.0 `experiment-archive-v1.3.0.zip` |
| `results/figures-v1.1.0`、`results/figures-v1.2.0`：用统一名称重绘的 v1.1.0、v1.2.0 图和 `data.md` 数据表，由 `program_figures.py` 生成 | v1.3.0 `experiment-archive-v1.3.0.zip` |
| `results/evo-training-rts24-20261030-101731`、`results/l24x-validation-20261030-120413` | v1.4.0 `experiment-archive-v1.4.0.zip`，脚本在 `experiment-source-v1.4.0.zip`，见 [v1.4.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v1.4.0) |

删除本地 `results/` 之前，三个实验归档都从 GitHub 下载回来，逐个确认本地每个文件（`__pycache__` 除外）都在其中且字节一致。

[v1.3.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v1.3.0) 提供：
- 发布提交对应的核心源码 `core-source-v1.3.0.zip`；
- 清理前的完整源码 `experiment-source-v1.3.0.zip`（提交 5a72fbc，含全部实验脚本与测试）；
- 上述新封存的结果目录 `experiment-archive-v1.3.0.zip`；
- 逐文件 SHA256 清单 `archive-manifest-v1.3.0.json`；
- 资产校验和 `SHA256SUMS.txt`。

机器可读汇总见 [archives/v1.3.0.json](archives/v1.3.0.json)。

## v1.2.0 RTS24 多轮闭环程序训练与验证

2026-10-26 至 27。L118 只经过两轮生成、一次反馈。这次在 RTS24 上重新做一次完整的闭环训练，得到程序 L24，并用一个小型验证实验检查训练效果；同时按“命名”一节统一了方法与程序的名称。v1.1.0 之后、本节之前的补充实验与交叉测试已作废，整体归档在仓库外，不随本版本发布。

**训练**（`program_training.py`、`run_program_training.py`）：
- 训练实例由 RTS24 派生，3 个场景（负荷系数 0.9／1.0／1.05，概率 .25／.5／.25），备用 5%。
  - train 用 3 个负荷扰动变体，validation 用 2 个。
  - 每个变体与测试实例在各场景负荷上的相对距离都至少 0.5%。
- 程序额外收到 `scenario_loads` 与 `scenario_probabilities`。接口不变：每个候选最多改动 4 台机组。
- 评分：在 SA+R 的 120 秒运行上取 3 个锚点（初始、成本降到一半处、终点），各做 15 秒探测，J 为前 10 秒的时间平均改善。
- 每轮请求两个程序。反馈包含此前每个程序的全部训练测量及失败原因，`closed-loop.json` 断言没有遗漏；验证分数不进入反馈。
- 停止规则：B_r 为截至第 r 轮的最佳验证均分。
  - 第 r 轮算“改进”的条件：B_r ≥ B_{r−1} + max(0.10, 3%·|B_{r−1}|)。
  - 至少 4 轮；连续 2 轮无改进即停，最多 10 轮。
  - 传输故障不计轮数，只暂停。
- 结果：跑满 10 轮（`MAX_ROUNDS`）。
  - 第 1 轮两个程序都用了沙箱禁止的 `set.add`，静态检查不通过。
  - 最佳验证均分从第 2 轮的 1.22 升到第 9 轮的 3.08。手写对照 R（即 SA+R 的算子）为 0.32。
  - L24 = 第 9 轮的 P9b（`programs/L24.py`），按残余缺口逐台做动态规划来凑伴随机组。
- 早期对照原定为第 1 轮最佳程序。因第 1 轮无有效程序，经用户批准，在任何验证运行之前修订：改取第 2 轮验证最佳的 P2a，即 L24r2（`programs/L24r2.py`），判据不变。

**验证**（`program_evaluation.py`、`run_program_evaluation.py`）：
- 实例：rts24 原工况 5 个种子，每次 600 秒，比较 SA+R、SA+L118、SA+L24r2、SA+L24。负荷扰动的 minus／plus 工况各 3 个种子，每次 300 秒，比较 SA+R、SA+L118、SA+L24。
- 运行方式：所有对照方法重新冷启动；单任务串行、单线程，在线 LLM 调用为零。

| 预登记判据 | 结果 |
|---|---|
| T1 迭代效应：SA+L24 对 SA+L24r2，原工况，≥4/5 且 AUC 差值中位数 > 0 | **通过**：4/5，AUC 差值中位数 0.041（降幅中位数约 6%） |
| T2 对无 LLM：SA+L24 对 SA+R，原工况，≥4/5 且 AUC 降幅中位数 ≥ 10% | **通过**：5/5，83.0% |
| T3 对 L118：SA+L24 对 SA+L118，全部 11 对，≥8/11 且 AUC 差值中位数 > 0 | **通过**：8/11（另 2 对终点相同），2.290 |

- 原工况：
  - SA+R 每次都停在 3.545%。
  - SA+L118 只在 s17 上走出停滞点（0.531%），其余停在 3.23%–3.32%。
  - SA+L24 与 SA+L24r2 在 5 个种子上都降到 0.53%–0.59%。
- 迭代的增益主要在收敛速度，终点差距相差很小。SA+L24r2 只在 s53 上明显较慢，AUC 为 1.48，SA+L24 为 0.61。
- minus 工况：
  - SA+L24 有 2 个种子停在 2.18%，只比 SA+L118 的 2.24% 略好；
  - s137 降到 0.23%。
- plus 工况：
  - SA+L24 在 3 个种子上都是 0.704%；
  - SA+L118 有 2 个种子也到 0.704%，这两对终点相同，不计胜。
- 复核：审计逐个重新调度 38 个终点，成本一致并通过独立校验，计分重算一致，配对初始化一致。
- 局限：只有 RTS24 一个系统，场景由本项目生成，种子少。判据是工程门槛，不是显著性检验。

**复现**：

```powershell
py -3.10 run_program_training.py                        # 每轮生成后暂停，写入审阅文件再继续
py -3.10 run_program_training.py --resume results/program-training-rts24-<时间>
py -3.10 run_program_evaluation.py --prepare results/program-training-rts24-<时间>
py -3.10 run_program_evaluation.py --run results/program-evaluation-rts24-<时间>
```

这些脚本自 v1.3.0 起移出仓库，在 `experiment-source-v1.3.0.zip` 中运行。

改名前后，固定轮数下 5 个方法的运行逐位一致。下面两个目录在改名前生成，目录名、报告、图和运行记录沿用旧名（见“命名”一节），用各自 `snapshot/` 中的代码复现。

**结果目录**：

| 目录 | 内容 |
|---|---|
| `results/lit-training-rts24-20261026-170730-82fb090e` | `report.md`、`learning-curve.json`、`closed-loop.json`、F5 学习曲线、全部程序与审阅 |
| `results/lit-evaluation-rts24-20261026-213117-7bb610af` | `report.md`、`audit.json`、F6 终点差距图；`interruptions/` 记录一次夜间暂停，`corrections/` 记录 F6 图例更正 |

`results/` 不进仓库，这两个目录与运行日志 `results/_lit_logs` 的完整副本在 Release 资产 `experiment-archive-v1.2.0.zip` 中；资产还附有 `NAMING.md` 对照表和用新名重绘的 F6。

[v1.2.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v1.2.0) 提供：
- 发布提交对应的源码 `core-source-v1.2.0.zip`；
- 上述结果目录 `experiment-archive-v1.2.0.zip`；
- 逐文件 SHA256 清单 `archive-manifest-v1.2.0.json`；
- 资产校验和 `SHA256SUMS.txt`。

机器可读汇总见 [archives/v1.2.0.json](archives/v1.2.0.json)，其中也记录了对 v1.1.0 两处表述的更正（见 v1.1.0 一节）。

## v1.1.0 可达出力修复与论文实验

2026-10-25。

### 可达出力修复

- `validation.reachable_capacity`：逐机组逐小时的可达出力上界，取两条链中的较小者：前向的开机爬坡链（初始开机机组从 `initial_output` 出发）和后向的停机链。`reachable_shortfall` 改为基于它计算。
- `repair()` 的缺口取"场景容量缺口"与"可达出力缺口"中的较大者，其余选择规则不变。
- `WholeDay.prepare()` 遇到仍有可达缺口的方案，记为 `reachable_infeasible` 并跳过调度。
- `BlockSearch.build_qubo`：行为不变的提取，供记录 QUBO 规模使用。
- 回归测试见 `tests/test_reachability.py`。

### 论文实验：SA 与 SA+L118

发布时方法称 A 与 L+SA，程序称 L3（`programs/L3.py`）；下文用统一后的名称，见“命名”一节。

**离线程序 L118**（`programs/L118.py`）是事先由 LLM 写好、此后冻结的程序。
- 它按价格信号提出最多 4 个候选，每个候选最多改动 4 台机组。
- 程序在隔离子进程（`offline_worker.py`）中运行，每次调用最多 0.5 秒；在线不调用 LLM。
- SA+L118 在每轮 SA 之后让 L118 运行同样长的时间，见 `offline_pilot.run_trial`。
- 多场景时，程序收到按概率加权的负荷与出力；候选仍要对全部场景做必要性检查，并经过全天精确调度。

**算例**：
- `instances/ieee118s10`：54 机组 × 24 小时 × 10 场景，由 `make_ieee118_scenarios.py` 生成。日系数和小时扰动的做法参照 Mahroo & Kargarian（TPWRS 2026），幅度按机组容量余量收窄。场景是本项目的扩展，不是 IEEE 源数据。
- `instances/rts24`：32 机组 × 24 小时 × 3 场景。

**参考解**只当差距尺子，由 Gurobi 单线程求解：
- ieee118s10：OPTIMAL，3,625,778.90，用时 212 秒；
- RTS24：OPTIMAL，885,066.70，用时 114 秒。

**复现**：

```powershell
py -3.10 make_ieee118_scenarios.py --check
py -3.10 run_paper_pilot.py --reference ieee118s10 rts24
py -3.10 run_paper_pilot.py --compare results/paper-reference-<时间>    # 先跑全部测试，再做 20 次 600 秒运行
py -3.10 run_paper_pilot.py --case results/paper-compare-<时间>
py -3.10 run_paper_pilot.py --audit results/paper-compare-<时间>
```

这些脚本（含生成 `instances/ieee118s10` 的 `make_ieee118_scenarios.py`）自 v1.3.0 起移出仓库，在 `experiment-source-v1.3.0.zip` 中运行。

**结果**：每次 600 秒冷启动，12 h 分块，16 sweeps，种子 17、29、41、53、67。

| 算例 | SA 终点差距 | SA+L118 终点差距 | SA+L118 更优 | AUC 降幅中位数 |
|---|---:|---:|---:|---:|
| IEEE118，10 场景 | 2.881% | 0.196%–0.633% | 5/5 | 70.8% |
| RTS24，3 场景 | 5.475% | 0.531%–3.321% | 5/5 | 39.8% |

- 两组初始化时就达到 SA 的终点成本；约 10 秒（RTS24 上约 1 秒）是初始化耗时，不作为 L118 加速成绩。SA+L118 首次严格改善该成本分别发生在约 35–42 秒和 5.5–7.3 秒。
- 两个预先登记的判据都通过：至少 4/5 个种子更优，且 AUC 降幅中位数 ≥ 10%。
- 复核通过：判据重算一致；方案重新调度后成本一致并通过独立校验；实例文件和参考解的哈希一致。
- 运行结束时，本文块 QUBO 为 1,413–2,490 位。假设 θ 和每条割的松弛寄存器均取 12 位，编码同一批已存割的计数估算为 2,537–14,143 位；另见 E2 表的一位假设。这不是无条件下界，本文最紧割费用代理也不宣称与完整多割主问题等价。场景在子问题中聚合，QUBO 不为每个场景复制一组变量。
- **案例**：在 SA 的终点方案上，L118 第 3 次调用就把 G5 在第 1–6 小时停机。那几个小时按概率加权的边际价格约为 21–24 $/MWh，这一改动使全天成本降低 21,631（−0.58%）。
- **局限**：场景由本项目生成，零备用；每个算例 5 个种子；SA 在 CPU 上运行。

**结果目录**：

| 目录 | 内容 |
|---|---|
| `results/paper-reference-20261025-124811-c6f41da2` | 参考解 |
| `results/paper-compare-20261025-125905-c264cdcf` | `report.md`；`figures/` 中的 F1 收敛曲线和 F2 QUBO 位数；`audit.json`；`CONSOLIDATION.md` |
| `results/paper-case-20261025-133936-d0ed2658` | F3 案例图 |

`results/` 不进仓库，这三个目录的完整副本在 Release 资产 `experiment-archive-v1.1.0.zip` 中。

[v1.1.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v1.1.0) 提供：
- 发布提交对应的源码 `core-source-v1.1.0.zip`；
- 上述三个结果目录 `experiment-archive-v1.1.0.zip`；
- 逐文件 SHA256 清单 `archive-manifest-v1.1.0.json`；
- 资产校验和 `SHA256SUMS.txt`。

机器可读汇总见 [archives/v1.1.0.json](archives/v1.1.0.json)。

## v1.0.0 跨块协调清理与封存

2026-10-22。重构前的代码封存为标签 `archive-20261022-pre-lagrangian`（3d3fa3e），v1.0.0 为本次封存版本。

**原有缺陷**：跨块拉格朗日协调从未更新过乘子。探针（seed 17、12 h 分块、关闭 LLM）中 RCUC20 52/52 次、IEEE118 29/29 次协调均为 `CAPPED`，2026-10-21 的 IEEE118 运行 44/44 次同样如此，乘子始终停在 LP 对偶。根因是协调平面来自边界固定为 incumbent 的 SA／修复点，而不是块拉格朗日子问题的最优点，切平面模型高估真实对偶值 2–7.5%，始终高于 UB 而被截顶。此外：外层重建清空割和缓存；块内上下界从未闭合，闭合跳过是死代码；L1 乘子选择即使接受也会降低对偶值。v0.33.0 两段共 11 次协调也全部 `CAPPED`；有记录的运行实际都在固定的 LP 对偶价格下搜索。

**试验过的修复及删除原因**：开发分支曾用 Gurobi 精确求解块子问题、以盒式信赖域 bundle 更新乘子（提交 62b8deb、a8100fb），乘子确实移动，下界在 RCUC20、classic10 上等于参考最优值，IEEE118 上差 0.043%。但块子问题全部由 Gurobi 求解，价格和下界都是经典计算的结果，与 SA 无关；IEEE118 上该阶段 119 秒仍未收敛，而 Gurobi 直接求完整问题只需 9.7 秒（RCUC20 0.8 秒、classic10 4.4 秒）；单种子冒烟中也看不出它对 UB 有帮助。SA 只能给出上界，项目中所有下界都来自 Gurobi，因此 v1.0.0 删除拉格朗日阶段，评价直接使用 Gurobi 参考成本。实现保留在 Git 历史中。

**v1.0.0 的改动**：
- 删除 `benders.coordinate`、外层循环、块上下界与闭合跳过、`planes`、`outer_history`、`coordination_upper`；`master()` 改为只含硬约束行的 `hard_rows()`（固定位与无割块 QUBO 在四个算例上与旧代码逐字一致）。
- 每块一个持久块搜索，割与调度缓存跨轮保留；QUBO 只编码 anchor 处松弛量最小的 64 条可行割，每块至多保存 4096 条割，缓存只存紧凑记录（不设上限时 RCUC20 的 QUBO 在 90 秒内由 361 位涨到 2227 位）。
- 块搜索价格只作用于含连续变量的跨块行；纯二进制跨块行价格为 0（旧代码对两类行都按 LP 对偶定价）。
- 选项 `max_outer`／`max_inner` 改为 `max_rounds`；旧选项名视为未知选项。
- 修正：只有 `TimeoutError` 记为 `TIME_LIMIT`，截止时间之后的其他异常不再被掩盖；没有块搜索时 LLM 候选的局部可行性不再空真为 True；`partition` 拒绝没有系数的行；`optimize` 的 `IntFeasTol` 改为显式参数，不再由 `mip_focus` 隐式修改。

**冒烟验证**（seed 17、12 h 分块、16 sweeps、关闭 LLM、搜索 600 秒、三个算例并行运行；单种子单次运行，UB 差异不能据此判断优劣）：

| 算例 | 封存版本 UB 差距 | 拉格朗日版本（已删除） | v1.0.0 UB 差距 | v1.0.0 SA 调用 | v1.0.0 状态 |
|---|---:|---:|---:|---:|---|
| RCUC20 | +0.903% | +0.710% | +0.710% | 747 | TIME_LIMIT |
| classic10 | +4.285% | +4.420% | +4.420% | 724 | TIME_LIMIT |
| IEEE118 派生 | +7.515% | +8.165% | +7.767% | 229 | TIME_LIMIT |

- RCUC20、classic10 上 v1.0.0 与拉格朗日版本的提议序列不同（少了拉格朗日阶段，SA 调用更多），但停在相同的 UB。
- 每次 SA 的 QUBO 位数有界：前后各十分之一调用的平均位数 RCUC20 805→891、classic10 766→999、IEEE118 1946→2069；全部运行没有调度数值失败。

机器可读汇总见 [archives/v1.0.0.json](archives/v1.0.0.json)。[v1.0.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v1.0.0) 提供发布提交对应的 `core-source-v1.0.0.zip`、实验归档 `experiment-archive-v1.0.0.zip`（重构用脚本、上述冒烟运行、离线探针与 v0.36.0 清理时的回归脚本）、逐文件 SHA256 清单 `archive-manifest-v1.0.0.json` 和资产校验和 `SHA256SUMS.txt`；[archive-20261022-pre-lagrangian Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/archive-20261022-pre-lagrangian) 提供重构前源码与协调排查探针。本地 `tmp/`、`releases/` 已在资产下载校验后清理，`results/` 未变动。

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
