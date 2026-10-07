# MATLAB：时间分块 SA + LLM Benders SUC

**最新版本：v0.21.0（待实验）。** 当前代码保留 SA 停滞后的块内融合框架；全天 LLM 提案与反馈闭环实验的计划已制定，相关实现已回退，等待后续实施与验证。

从 `LLM-benders-SUC` 的顶层 Python 源码重构，保留原算法主线：

**连续松弛初始化 → 时间分块 → 块内 Benders / QUBO / SA → 拉格朗日切平面协调 → 全天恢复与物理校验。**

运行只需要 MATLAB 和已配置许可证的 Gurobi MATLAB 接口，不依赖 Python、Kaiwu、MATPOWER 或额外 MATLAB 工具箱。已在 MATLAB R2025b、Gurobi 13.0.1 上验证。

## 运行

在 MATLAB 中进入本目录。如果 Gurobi 尚未加入 MATLAB 路径，先运行其安装目录下的 `gurobi_setup`。

```matlab
result = run_suc();
```

默认使用 RTS-24 的 32 台机组、24 小时、3 个随机场景；3 小时一块，外层最多 20 轮，每块内层最多 20 轮，总预算 600 秒。

短运行检查：

```matlab
result = run_suc(struct('max_outer', 2, 'max_inner', 5, 'time_limit', 120));
disp(result.status)
disp(result.objective)
disp(result.lower_bound)
disp(result.relative_gap)
```

确定性场景与结果保存：

```matlab
root = fileparts(which('run_suc'));
result = run_suc(struct('scenario_file', fullfile(root, 'data', 'deterministic.json')));
save('result.mat', 'result');
```

`result.u/v/w` 是“机组 × 小时”矩阵；`result.scenarios(k).p/r` 是对应场景的出力与备用。`history` 记录块内上下界、两类割的数量及 SA 统计，`outer_history` 记录外层协调。结果默认返回工作区，不自动生成实验目录。

## 精简后的文件

保留 15 个运行用 `.m` 文件、1 个测试文件和 4 个数据 JSON。

| 文件 | 职责 |
|---|---|
| `run_suc.m` | 唯一运行入口、参数检查、随机种子 |
| `llm_config.m` | 自定义 LLM 开关、API 地址、Key、模型及调用参数 |
| `+suc/load_data.m` | 读取并检查原 JSON 格式 |
| `+suc/model.m` | 全天稀疏模型及索引 |
| `+suc/partition.m` | 约束分块与跨块定价 |
| `+suc/optimize.m` | Gurobi 接口、连续松弛的对偶下界 |
| `+suc/dispatch.m` | 连续调度、最优性割、弹性可行性割 |
| `+suc/master.m` | 精确块主问题，仅输出认证下界 |
| `+suc/qubo.m` | 目标寄存器、容量格点、保守量化与松弛位 |
| `+suc/anneal.m` | MATLAB 本地 Metropolis 模拟退火 |
| `+suc/coordinate.m` | 拉格朗日切平面协调 LP |
| `+suc/repair.m` | 全天最小开停机时间及容量修复 |
| `+suc/llm.m` | 可选 HTTP 候选建议 |
| `+suc/validate.m` | 独立物理与成本校验 |
| `+suc/solve.m` | 内外层循环、候选恢复与结果 |
| `tests/test_core.m` | 数值正确性、编码、边界和端到端回归 |

未迁入：实验结果、冻结源码副本、历史报告、绘图与报告脚本、多套实验入口、Codex 文件通信、HTTP 重测框架、量子 SDK 包装，以及重复的模型实现。原项目目录保持原样，新版位于 `LLM-benders-SUC2`。

## 模型与算法约定

- 保留随机场景的期望二次燃料成本、固定开机/启动/关闭费用、负荷平衡、备用、最小出力、启动与关闭出力限制、爬坡、最小开停机时间和初始剩余约束。
- 小时之间使用原始初始出力及无条件出力差爬坡限制。最小开停机约束截断至第 24 小时；最后一小时禁止停机。分块不重置初始条件，跨块约束进入外层协调，支持短末块。
- 下界来自连续松弛的对偶值和整数块主问题的 `objbound`。认证主问题的整数解不进入 SA、割生成或全天恢复。QP 割截距用盒约束上的对偶目标重算，不直接用近似原始目标锚定。
- QUBO 中的 Benders 割采用单侧向下量化，容量用十进制整数格点编码（容量输入最多六位小数）；认证和实际调度使用原始精度。SA 能量不是下界。
- 每次更新外层乘子都会重建块内成本和割，并重新调度历史候选；全天成本始终按原目标重算，通过独立校验后才更新可行上界。
- SA 改为 MATLAB 实现，按单比特翻转能量尺度升温，冷却到目标分辨率。保留一个热启动并使用随机重启；随机数和采样实现与 Kaiwu 不同，不能期待相同搜索轨迹或耗时。
- 精简保留严格内外层停止规则。某块在内层上限内未收敛便返回 `INNER_NOT_CONVERGED`，不更新外层乘子。`DUAL_CONVERGED` 只表示对偶协调收敛；原整数问题的间隙仍看 `relative_gap`。

`tolerance` 默认 `1e-4`，绝对容差 `1e-5`。还可能返回 `MAX_ITERATIONS`、`TIME_LIMIT` 或带阶段名的 `*_FAILED`；详细原因在 `message` 中。时间预算在求解和 SA 检查点生效，模型构造和正在进行的 HTTP 调用可能造成少量超时；截止后的候选不会记入结果。

## 可选 LLM

默认关闭。启用后，连续 `llm_patience`（默认 3）个批次改善不足 0.05% 时调用一次，最多 `llm_max_calls`（默认 3）次。使用同步、非流式的 Responses API；旧版实验策略和子代理文件协议不再保留。

打开项目根目录的 `llm_config.m`，一次填写以下配置：

```matlab
o.llm_enabled = true;
o.llm_url = 'https://YOUR-SERVICE/v1/responses';
o.llm_api_key = 'YOUR-API-KEY';
o.llm_model = 'YOUR-MODEL';
```

保存后，每次直接运行：

```matlab
result = run_suc();
```

`llm_url` 必须是服务的**完整 Responses 请求 URL**（通常以 `/v1/responses` 结尾）。无需设置环境变量或每次传入参数；`llm_timeout`、`llm_patience`、`llm_max_calls` 也在同一文件中修改。已有的 `run_suc(options)` 仍可用于临时覆盖配置。

请求按 [OpenAI Responses 官方文档](https://developers.openai.com/api/docs/guides/migrate-to-responses) 使用 `model`、`instructions`、`input`、`stream=false` 和 `store=false`，从返回的 `output` 中提取 `output_text`，跳过推理等其他类型的条目。

LLM 仅提供 `u/v/w` 二进制候选，必须再通过约束检查、连续调度和全天物理校验。HTTP 失败、响应未完成、无文本或格式错误时继续 SA；状态保存在 `llm_history`。Key 只从 `llm_config.m` 读取，不读取旧项目的 `.env` 或 `LLM_API_KEY`，运行结果和日志不保存密钥或请求头。验证使用离线模拟，未调用真实 LLM 服务。

## 验证

```matlab
tests = runtests('tests');
assertSuccess(tests);
```

12 项测试全部通过，包含小实例全枚举割校验、精确主问题与完整 MIQP 对照、启动/关闭容量、QUBO 全编码枚举、容量格点与保守量化、跨块成本恒等式、自由等式乘子、Responses 请求与响应离线验证、本地 LLM 配置加载、内外层收敛、独立物理校验、截止时间及 RTS-24 可重复运行。

迁移时还直接对照了原项目 `IndexedUC`：21,960 条约束、6,912 个变量、56,996 个非零项，矩阵、右端项、目标系数和边界逐项完全相同。随机场景下，全开机固定方案成本约为 **1,220,423.773288**；两种实现的连续松弛目标约为 **841,376.53**，差约 0.0014，来自求解数值精度。

交付时的 5 轮内层短跑得到可行成本 **933,522.215366**、认证下界 **841,376.524841**、间隙 **9.8708%**，独立校验最大物理残差约 `1.36e-12 MW`。SA 生成了新的候选和割，但第一块尚未收敛，故状态为 `INNER_NOT_CONVERGED`；这一检查不表示已经求得全天最优解。

独立完整 MIQP 可使用同一精确模型，仅用于事后比较：

```matlab
root = fileparts(which('run_suc'));
[d, s] = suc.load_data(fullfile(root,'data','units.json'), ...
    fullfile(root,'data','stochastic.json'));
reference = suc.optimize(suc.model(d,s), 60, true);
% reference.objval 是可行目标（仅找到解时存在）；reference.lower 是认证下界。
```

## 数据来源

`data/units.json`、`stochastic.json`、`deterministic.json` 和 `provenance.json` 从原项目原样保留。来源固定版本、哈希、机组映射和转换说明见 `provenance.json`；精简版直接使用这些已转换数据，不附带 Python 转换器或原始数据副本。

原始公开来源：[MATPOWER IEEE RTS-24](https://github.com/MATPOWER/matpower/blob/8b326f00d496864ed81169f7e684568fc1f7006d/data/case24_ieee_rts.m) 与 [ucacopf RTS79](https://github.com/anyacastillo/ucacopf/blob/001cf29dc2e9b3d26e71146f82db35fa379b4dca/RTS79_ucacopf.dat)。总装机 3405 MW、峰值基础负荷 2850 MW；同步调相机已排除。随机倍率 0.90/1.00/1.05、概率 0.25/0.50/0.25 和 5% 备用率属于项目设定。

本模型是聚合无网络 SUC，不含输电约束、冷热启动成本或水电能量预算；这与原项目建模口径一致。

## SA 停滞后的融合实验

配置 `llm_config.m` 后，在项目根目录运行首批筛查：

```powershell
.\scripts\run_stagnation.ps1 -Directory results/stagnation-20261007
```

固定种子 7、17、27，每个首个停滞检查点最多调用一次 LLM，C/D 共享响应并分别扣除实际调用耗时。四条分支为 A 继续 SA、B 普通再升温、C LLM 候选与割、D 额外候选热启动；不运行纯 LLM 组。当前调度器串行执行数值任务和 HTTP 请求。已有校验通过的工件会被复用，失败调用不自动重试；新实验使用新的输出目录。

MATLAB 底层入口为 `run_stagnation_experiment(action, directory, seed, mode)`，支持 `capture`、`prepare`、`import`、`branch`、`diagnose`、`reference`、`report`、`online`、`online-report`。首批 HTTP 请求由 PowerShell 调度器负责，`import` 消费已落盘的响应；第二阶段在运行中的 MATLAB 内同步请求，并使用独立全局调用账本。仅重新生成报告可运行：

```matlab
suc.stagnation_report('results/stagnation-20261007');
```

输出包括 `report.md`、`summary.json` 和 `curves.csv`；全天校验尚未完成时标记 `PENDING`，不会提前决定扩展。参考解只作事后比较，不进入搜索。三个种子的结果用于筛查，不构成统计显著性结论。达到预定门槛后，调度器自动执行已授权的第二阶段：种子 107、117、127，A 与选出的融合方式，共六次完整求解调用，新增模型调用总上限六次。首批重放受 1200 秒和内层总计 200 批双上限约束；第二阶段受 1800 秒和内层 200 批双上限约束，可能提前以 `INNER_NOT_CONVERGED` 结束。第二阶段结束后可用 `suc.stagnation_online_report` 生成离线归因报告。
