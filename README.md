# SA-Benders SUC

Python 随机机组组合（SUC）求解器：先用 Gurobi 求完整模型的参考方案，再由时间分块的 D-Wave 开源 SA 与块内 Benders 割搜索可行上界；候选一律经全天修复、连续调度和独立物理／费用校验后才能更新 UB。全局评价采用相对参考成本的差距；项目不计算下界。

v2.0.0 删除了全部 LLM 组件（在线上界顾问、离线 LLM 程序及基于它们的实验脚本），只保留 SA+Benders 求解核心，见文末“历史”一节。

当前默认输入为 Pisa IEEE118 派生的 **54 机组、24 小时、单场景、零备用**；启动费按 `fixedCost` 列映射为零。只修改 `data/units.json` 和 `data/stochastic.json` 即可更换输入，来源与费用口径见 `data/provenance.json`。其他算例在 `instances/<名称>`（classic10、rts24、rcuc20，以及 10 场景的 ieee118s10），把 `run_experiment.py` 的 `DATA_DIR` 改为对应目录即可。

## 运行

环境：Windows 64 位、Python 3.10、Gurobi 许可证。SA 使用 Apache-2.0 开源的 `dwave-samplers==1.8.0`（`dimod==0.12.22`、`networkx==3.4.2`），在本地 CPU 运行，无需 Kaiwu 授权、D-Wave 云账户或量子硬件。

```powershell
py -3.10 -m pip install -r requirements.txt
```

在 VS Code 中直接运行 `run_experiment.py`。预算、种子、分块等参数位于文件顶部；`CHECK_ONLY=True` 只检查输入。

入口每次从零开始，先用最多 120 秒独立预算求参考解，再给搜索完整的 7200 秒预算，seed 17、目标参考成本差距 0.1%。参考方案经过独立物理与费用校验；只有状态为 `OPTIMAL` 的有效参考才允许提前停止。仅有未证最优的可行参考时只比较、不据此提前停止；参考失败或没有有效方案时仍继续搜索，参考成本和差距为空。参考方案不用于初始化和候选注入。v0.36.0 起删除了续跑（`RESUME_FROM`），续跑旧实验请使用对应归档版本。

参考差距为 `100 × (UB - reference_cost) / abs(reference_cost)`，允许负数；参考成本为零时百分比为空。停止要求有效 `OPTIMAL` 参考，且 `UB - reference_cost <= 1e-5 + tolerance × abs(reference_cost)`；零参考成本只用绝对容差。达到目标记为 `REFERENCE_TARGET_REACHED`，搜索轮次或 SA 配额耗尽后记为 `SEARCH_EXHAUSTED`（`search_stop_reason` 为 `MAX_ROUNDS`、`SA_BUDGET` 或 `SA_DISABLED`），只有截止时间信号（`TimeoutError`）记为 `TIME_LIMIT`；其他异常无论是否已过截止时间都记为 `<阶段>_FAILED`（阶段为 `INITIALIZATION`、`SEARCH_SETUP`、`ROUND_<k>_BLOCK_<j>`）。没有待处理工作时不空等剩余预算。

初始化后每块建立一个持久块搜索，整个运行不重建，块内割与调度缓存跨轮保留；每轮遍历全部时间块，每块采样一批 SA（`max_rounds` 限制轮数，入口不限）：一批为 `sa_reads` 条独立退火链，种子由配置种子、调用序号和链号派生，再取模 `2**31` 传给 D-Wave；诊断同时保留派生种子和实际采样种子。候选拼接使用该批开始时的当前最优全天方案；保留调度去重。全天或局部调度出现 SUBOPTIMAL 等数值状态时只拒绝该候选（计入 `dispatch_failures`），不中止运行。

结果自动写入新建的 `results/manual-seed.../`：

- `summary.json`：状态、UB、参考元数据、`reference_gap_percent`、耗时和统计，以及 `complete_rounds`、`search_stop_reason`、`sa_budget_exhausted`。
- `result.json`：精简主结果、最终方案、计时与搜索统计。
- `reference.json`：先行参考求解的状态、完整方案、校验和独立耗时；失败时保留原因。`gurobi_bound`／`gurobi_gap_percent` 是 Gurobi 自身的界与间隙，只说明参考离最优多远，不参与停止判断。
- `snapshot/`、`manifest.json`、`effective-options.json`：实际源码与输入快照、SHA256、依赖版本和完整有效配置。
- `progress.jsonl`、`latest-solution.json`：轻量进度与原子更新的最新可行方案；最新方案不含搜索状态。

不保存全量检查点，也不序列化全部历史采样；记录开销与最终保存耗时单列。

当前主结果删除全局 `lower_bound`／`lower`、`relative_gap`／`gap_percent` 和 `certification_history`，不计算下界：SA 只能给出上界，项目历史上的各种下界（连续松弛、精确 Benders 主问题、拉格朗日）都由 Gurobi 求得，评价直接使用 Gurobi 参考成本。`Result.reference` 保留完整参考记录，`reference_cost` 与 `reference_gap_percent` 是随当前已验证方案计算的只读属性。进度和最新方案同时带有 `reference_status`。`elapsed_seconds`／`solver_elapsed_seconds` 只计搜索，`reference_elapsed_seconds` 单列参考耗时，`total_elapsed_seconds` 计两段总耗时。历史 `results/`、`archives/` 和源码快照不迁移，旧分析脚本需适配新字段与状态。

manifest、精简结果及运行诊断记录 `sa_backend` 与 `sa_initialization`（当前为 `random_sa_repair_v1`）。采样诊断包含 beta 序列，以及每条链的原始／实际种子、退火终态与 QUBO 能量；全天提议记录只保存原始方案的 SHA256 与固定位违例数。这些诊断可经 `run_suc` 返回对象读取，不自动全量落盘。

读取精简结果：

```python
import json
with open("results/你的运行目录/result.json", encoding="utf-8") as f:
    result = json.load(f)
```

直接运行 `run_experiment.py`；`run_suc.py` 提供 `from run_suc import run_suc` 的 Python 接口，同样只计算一次先行参考。配置 `reference_time_limit` 默认 120 秒，对应入口顶部 `REFERENCE_TIME_LIMIT_SECONDS`；`time_limit` 为独立搜索预算。可选关键字 `reference_log_file` 指定参考日志；`started_at` 是整体调用的计时起点，不扣减参考结束后新建的搜索预算。旧命令行参数及 `--output` JSON 输出入口已删除。

机组输入必须明确提供 `initial_output`。运行环境固定为上述 Windows／Python 3.10。

## 核心文件

- `suc_benders/solver.py`：结果类型与主循环，安排参考、初始化和分块轮询。
- `suc_benders/model.py`：稀疏模型、按小时分块（`partition`）、跨块定价（`priced`、`relaxed_rows`、`search_multipliers`）。
- `suc_benders/search.py`：持久的原始块搜索及共享的全天候选修复、调度、校验、去重和预算。
- `suc_benders/benders.py`：离散可行性检查、固定二进制后的连续调度与 Benders 割、块内硬约束行（`hard_rows`）。
- `suc_benders/reference.py`：Gurobi 完整模型参考求解及独立方案校验。
- `suc_benders/qubo.py`、`anneal.py`：块搜索 QUBO 与 D-Wave SA 采样；`validation.py`：修复、可达出力与独立校验。
- `run_suc.py`、`run_experiment.py`：程序接口与一键运行入口。
- `data/`：当前输入、原始算例与溯源信息；`instances/`：classic10、rts24、rcuc20、ieee118s10 输入。
- `tests/`：求解器回归；执行 `py -3.10 -m pytest -q tests`。
- `.local/*.lic`：仅本地保留的授权文件。

主线顺序：输入建模 → 独立参考求解 → 连续松弛与初始可行解 → 固定价格的持久分块 SA 轮询 → 全天修复/调度/校验。仅全天验证通过的方案能更新 UB。

**跨块协调**：跨块行指变量落在不同时间块的约束。块之间不做乘子迭代，而是围绕当前最优全天方案（incumbent）做块坐标搜索：纯二进制跨块行（状态转移、最小开停机）把块外变量代入 incumbent 后作为块内硬约束；含连续变量的跨块行（跨边界爬坡、块末停机容量、冷启动窗口）在块费用中按初始化时连续松弛的对偶价格定价（`search_multipliers`），价格整个运行不变。块内候选一律经全天修复、调度和校验后才能更新 UB。

块 SA 使用搜索专用 QUBO（`qubo.search_qubo`）：成本取 incumbent 处最紧最优性割的线性代理，不含 theta 与最优性割松弛位；硬约束按行归一化后以 `sa_hard_weight` 加权；容量只含 `pmax·u`；跨块纯二进制约束代入 incumbent 后作为块内硬约束。能量单位是单个位的最大归一化成本，`sa_temperature`、`sa_cutoff` 同此单位，`sa_sweeps` 为每个温度的全位扫描次数。QUBO 只编码 anchor 处松弛量最小的 64 条可行割，每块至多保存 4096 条割（先进先出，丢弃数记入 `cuts_dropped`），块内缓存只存紧凑记录，使 QUBO 规模与内存不随轮数增长。

每批只有一个搜索 QUBO，直接由稀疏 Q 构造 BINARY BQM，合并 `Qij+Qji`，不生成稠密 Ising 矩阵。每条链由采样器根据种子生成随机初态，每次一个 read。退火终态恢复固定变量后，先进行原始块候选评价／割学习，再走既有全天修复、调度和独立校验；通过校验才可更新 UB。温度从 `sa_temperature*energy_unit` 起逐次乘 `sa_cooling`，严格高于 `sa_cutoff*energy_unit` 时执行，倒数作为自定义 beta；每级执行 `sa_sweeps` 次顺序 Metropolis 扫描。[采样器 API](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/generated/dwave.samplers.SimulatedAnnealingSampler.sample.html)

建模和采样均计入时间；链间检查预算，单条链不能硬中断，超时按实记录。稀疏接口消除了强制稠密化，但 QUBO 本身仍可能较密。历史不同搜索策略的实验保留各自口径。

## 历史

- v2.0.0（2026-11-04）：删除全部 LLM 组件——在线上界顾问（`llm.py`、`advisor_request.py`、`upper_advisor.py`、`context.py`）、离线 LLM 程序（`programs/`、`offline_pilot.py`、`offline_worker.py`）与邻域实验（`neighbourhood_search.py`、`run_neighbourhood_precheck.py`），以及全部 `llm_*` 参数、`config.local.json` 读取和 `tomli` 依赖。SA 只以当前最优方案为种子，其余搜索、修复与校验逻辑不变。
- v1.4.0 及更早：含 LLM 的方法、程序训练与验证实验。代码见对应 git 标签与 GitHub Release，各版本的变更、结论与封存资产记录在 `archives/v*.json`。
