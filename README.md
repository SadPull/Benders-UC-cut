# RTS-24 SUC：SA + LLM Benders

本项目研究 IEEE RTS-24 派生随机机组组合问题：32 台机组、24 个小时，
由 Kaiwu 本地模拟退火（SA）生成开停机候选，LLM 提供可选建议，
Gurobi 求解连续调度子问题并生成 Benders 割。候选必须经过修复、去重和完整物理／成本校验。
独立 Gurobi 完整模型提供参考解及下界。

当前仅维护 RTS-24 全时域动态 QUBO 流程。旧 10×24 数据、归因实验、
静态／旧版主问题、时间分块与精确块对照、独立减开机搜索及旧实验归档已移除。

当前版本为 `v0.18.0`，变更见 [CHANGELOG.md](CHANGELOG.md)。
[项目阅读教材](docs/project-textbook.md)基于提交 `bae3200` 的历史源码快照编写，
版本范围和文件哈希在教材中列明；后续实验入口及补跑结果见下方说明。

## 安装

使用 Python 3.10、Kaiwu SDK 1.4.1 和可用的 Gurobi 许可证：

```powershell
python -m pip install -r requirements.txt
```

Kaiwu 激活方式见 [SDK 安装说明](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/getting_started/sdk_installation_instructions.html)。
这里只使用本地模拟退火，不调用 QPU。

## 数据

[数据说明与来源](data/rts24/README.md)包含公开输入、转换器及校验和。
总装机容量 3405 MW，峰值负荷 2850 MW。模型采用无网络聚合系统和固定启动费用；
不包含输电约束、冷热启动费用或水电能量预算。

| 配置 | 场景 | 备用 |
|---|---|---|
| `data/rts24/deterministic.json` | 公开小时负荷，概率 1 | 0 |
| `data/rts24/stochastic.json` | 负荷倍率 0.90／1.00／1.05，概率 0.25／0.50／0.25 | 5% |

随机场景及备用率是项目设定。默认输入是 `data/rts24/units.json`，
不传 `--scenario-file` 时采用上述随机场景。

```powershell
python data/rts24/convert.py --check
```

## 运行

纯 SA：

```powershell
python -m suc_benders --strategy sa --rounds 3 --output-dir results/rts24-sa-new
```

SA 与 Astra 固定交替（需在 Codex 中调度子代理）：

```powershell
python -m suc_benders --strategy fixed --rounds 2 --output-dir results/rts24-astra-new
```

连续三轮 SA 降本不足 0.05% 后，顺序调用最多三次 LLM；每次调用后先验证并更新反馈：

```powershell
python -m suc_benders --strategy stagnation_burst --rounds 10 --http-base-url https://api.deepseek.com --http-model deepseek-flash --output-dir results/rts24-http-new
```

HTTP 模式从进程环境读取 `LLM_API_KEY`。Astra 使用文件通信，普通终端运行会等待
Codex 回传候选。两种方式均保存请求、响应、校验和反馈。

| `--strategy` | 行为 |
|---|---|
| `sa` | 纯 SA |
| `fixed` | 每轮 SA 后调用一次 LLM |
| `stagnation` | 首轮调用，之后连续三轮 SA 改善不足时调用一次 |
| `stagnation_burst` | 连续三轮 SA 改善不足时顺序调用三次 |
| `llm` | LLM 与连续子问题，无 SA |

可设置 `--time-limit-seconds`、`--max-llm-calls`、`--seed`、`--scenario-file`。
输出目录必须是新目录。`python -m suc_benders.astra_run` 使用同一入口。
完整文件通信、取消、自动反馈及审计方式见 [运行说明](docs/astra-module.md)。

`--target-reference-gap 0.005` 表示独立参考成本差严格小于 0.5% 才停止。
该模式忽略轮数，不与时间／调用次数上限组合；参考成本用于停止判断和报告，
不进入 LLM 输入或候选生成。参考解会先固定开停机并重新验证连续调度。

## 独立参考与基线验证

```powershell
python -m suc_benders.full --scenario-file data/rts24/deterministic.json --time-limit 600 --mip-gap 0.0001
python -m suc_benders.rts24_experiment --output-dir results/rts24-baseline-new
```

基线验证要求干净且已提交的工作区和新目录。它串行完成两个配置的 SA（种子 7、
候选池 16、最多 3 轮／600 秒），再求独立参考（单线程、600 秒、MIPGap 1e-4），
并校验每个完整方案。该命令不会调用 LLM。

## 保留的 RTS-24 结果

最小必要性对照实验（15 次搜索 + 1 次参考）使用独立入口：

```powershell
python -m suc_benders.minimal_experiment --output-dir experiments/rts24-minimal-new
```

该入口冻结源码和输入后串行运行 A 固定种子 SA、B 逐轮换种子 SA、C LLM-only、
D SA+LLM、E SA+规则启发式，各取种子 7/17/27，搜索预算 1800 秒。
C/D 各最多三次全新 Astra/max 子代理调用，需要 Codex 按 `subagent.claim/receipt`
调度；C 用完额度结束，其余组继续 SA 至截止。E 默认总共三轮规则提案，
`--rule-max-calls 0` 可选择每次停滞均触发三轮；每轮至多四个不同候选。
规则候选采用提示词中相同的收益表，尝试整段关机及两端修剪。
运行过程中生成 `report.md`、`summary.json` 和 `curves.csv`，包括截止时刻成绩、
原始/修复/调度逐级统计、贡献来源、分项耗时及独立物理复核。
参考解只用于事后比较，不用于停止或生成候选。三个重复仅用于初步筛查。

| 归档 | 结果 |
|---|---|
| [中断实验补跑](experiments/rts24-recovery-20261004-resume/README.md) | C／17、D／7、C／27、D／17 均按 1800 秒预算完成，数值审计通过；实时监测，累计恢复会话 17／18 次，完整响应为 2、1、2、2 次；先前失败、中断及全部结果保留 |
| [最小必要性实验](experiments/rts24-minimal-20261003/analysis.md) | 15 个预定结果及 2 次标记补跑；数值审计通过，但 C／17 补跑、D／7 无模型响应，协同证据仍不足；全部 17 次尝试保留 |
| [SA 基线](experiments/rts24-baseline-20261002/README.md) | 三轮后确定性／随机配置的参考成本差分别为 46.89%／32.70% |
| [Astra burst](experiments/rts24-sa-llm-burst-20261002/report.md) | 5 次 SA、2 次 Astra；成本 889097.785912，参考差 0.455455%，达到严格 0.5% 目标 |
| [DeepSeek burst](experiments/rts24-deepseek-burst-20261002/report.md) | 用户停止；成本 907432.581261，参考差 2.527027%，未达到修订后的 1% 目标 |

归档保留当次源码、输入、运行记录与审计，文件校验和保持不变。复现当次结果应使用
压缩包中的源码快照；当前源码清理后不能冒充原版本或恢复原指纹的运行。
这些单种子实验不构成速度优势或全局最优性证明。

## 实现与验证

- `solver.py` 定义配置、结果和统一入口；`paper_solver.py` 实现当前动态全时域循环。
- `paper_model.py` 实现索引约束、连续松弛和紧凑 QUBO；文件名沿用既有实现。
- `candidates.py` 负责确定性修复；`context.py` 保留供 LLM 使用的成本、对偶及减开机启发信息。
- `llm.py`、`subagent.py`、`advisor_feedback.py` 负责候选通信和自动反馈。
- `validation.py` 独立校验物理约束和成本；`full_model.py` 提供完整 Gurobi 参考。

启动／关闭容量分别处理；备用使用精确容量格点，进入 QUBO 的割采用单侧保守量化。
连续模型和实际成本保持原精度。SA 搜索是启发式，QUBO 能量不能作为全局下界。
时间限制在检查点生效，正在执行的 SA SDK 调用可能产生超时；截止后的候选不计入结果。

```powershell
python -m unittest discover -s tests -v
```
