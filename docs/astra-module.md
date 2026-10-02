# Astra 求解模块与 Codex 调度

## 启动

在 Codex 中请求运行；Codex 启动下面的程序并服务文件请求。普通终端独立运行会等待 Codex 回传，不会自行创建代理。

```powershell
python -u -m suc_benders.astra_run --data data/rts24/units.json --seed 7 --rounds 2 --output-dir results/astra-new-run
```

可指定 `--scenario-file`、`--time-limit-seconds`、`--max-llm-calls`。默认增强反馈；`--original-feedback` 用于旧提示词对照。输出目录必须不存在。没有隐式重跑或恢复，失败目录保留。

可选配置（每次使用不同输出目录）：

| 组 | 参数 |
|---|---|
| 纯 SA | `--strategy sa` |
| 原反馈固定交替 | `--strategy fixed --original-feedback` |
| 增强反馈固定交替 | `--strategy fixed` |
| 增强反馈停滞触发 | `--strategy stagnation` |
| 连续停滞三轮后连续调用三次 | `--strategy stagnation_burst` |
| 增强反馈无 SA | `--strategy llm` |

停滞模式在首次 SA 后调用一次；随后连续 3 次 SA 的改善不足该轮 SA 前最佳成本的 0.05% 时调用。显著改善或一次调用结束会重置计数；失败也消耗调用额度。达到 `--max-llm-calls` 后，混合模式继续 SA，无 SA 模式停止。默认调用额度不限，但仍受轮数和可选总时间约束。

`stagnation_burst` 从 SA 开始，没有首次调用特例。每轮 SA 完整候选验证后比较最佳成本，连续 3 轮降本不足轮前最佳成本的 0.05% 才进入 LLM；降本达到 0.05% 清零计数。每组顺序调用 3 次，每次完成候选验证、最佳解更新和自动反馈后重新构造下一次输入，组内不运行 SA。组末清零停滞计数并回到 SA。调用目录按全局调用序号编号；事件同时记录 SA 轮数、调用序号及组内序号。失败调用消耗本组一次额度。

本次 RTS-24 三场景实验命令（种子 7、候选池 16）：

```powershell
python -u -m suc_benders.astra_run --data data/rts24/units.json --scenario-file data/rts24/stochastic.json --seed 7 --strategy stagnation_burst --target-reference-gap 0.005 --output-dir results/rts24-sa-llm-burst-new
```

同协议 HTTP 模型对照可同时传入 `--http-base-url` 和 `--http-model`，密钥仅从进程环境 `LLM_API_KEY` 读取。下例使用 DeepSeek 官方 Chat Completions 接口；不设置温度和输出 token 上限，使用 `reasoning_effort=max`，每次仅一次 HTTP 请求，无重试。模型列表和参数兼容性应先以独立试请求核对，试请求不能计作正式优化调用。

```powershell
python -u -m suc_benders.astra_run --data data/rts24/units.json --scenario-file data/rts24/stochastic.json --seed 7 --strategy stagnation_burst --target-reference-gap 0.005 --http-base-url https://api.deepseek.com --http-model deepseek-flash --output-dir results/rts24-deepseek-new
```

HTTP 调用自动运行；每次的提示词、无认证头请求体、响应、模型/用量、解析修复诊断和验证反馈保存在 `calls/`。配置、提示词和求解/停止规则沿用原实验。不同服务的默认生成参数和传输/调度开销仍可能不同；总耗时不能直接当作纯模型推理速度对比。

任意候选完成全部场景调度及物理/成本验证后，参考差严格小于 0.5% 立即停止；等于 0.5% 继续。无需完成该候选后的候选或本组三次调用。目标模式不设 SA 轮数、总时间或调用次数上限；参考求解仍有独立的 600 秒时限。

无 SA 模式复用同一连续评估流程，不构建或采样 QUBO，独立入口为上述 `--strategy llm`。该模式不报告认证下界。轮数不是跨策略的等价资源预算；正式比较应另行冻结端到端预算。

## Codex 调度约定

### 按参考成本达标停止

```powershell
python -u -m suc_benders.astra_run --data data/rts24/units.json --seed 7 --strategy fixed --target-reference-gap 0.001 --output-dir results/astra-target-new
```

该模式重新求解同一算例的 Gurobi 完整模型（单线程、同种子、MIPGap=1e-6，参考求解最多 600 秒），再固定其开停机方案重算连续调度并通过原有物理验证。原始与复核结果分别保存为 `reference-raw.json` 和 `reference.json`；保留独立下界。参考值只进入停止判断和报告，不进入模型输入。

`0.001` 表示相对参考成本差严格小于 0.1%；等于阈值不停止。该模式忽略 `--rounds`，拒绝同时配置时间/调用上限，每个候选完整验证后立即判断；仅正常达标才返回 `TARGET_REACHED`。用户取消、计算故障或无可行初始化仍可能中断，不能标记为达标。参考求解的 600 秒限制不限制 SA＋LLM 的运行时间。

每个完整可行候选的参考差写入事件，最终比较写入 `comparison.json`。相对参考成本差与算法自身认证 gap 分开解释。

1. 等待输出 `WAIT_SUBAGENT`，读取对应 `request.json`；调用 `suc_benders.subagent.claim(root, iteration)`，将返回的程序生成指令原样作为子代理任务。该函数独占创建 `dispatch.json`，已认领或已关闭的请求不能再次启动。每个请求仅启动一个子代理，串行处理；若在认领后启动工具失败，停止本次运行并保存现场，不自动重试。
2. 调用 `collaboration.spawn_agent`，明确设置 `model="gpt-6-astra"`、`reasoning_effort="max"`、`fork_turns="none"`。只授权读取该目录的 `system.txt` 和 `user.txt`，以及写入该目录的 `candidates.json`；禁止读取其他文件、求解器调用、外部 API、其他代理和优化程序执行。
3. 业务任务全部来自输入文件；主代理不得新增数学建议或手工修改候选。子代理输出最多 4 个带 name/reason/u 的候选，u 为完整二元矩阵。
4. 等待该子代理完成，再调用 `suc_benders.subagent.receipt(root, iteration, agent_id, "completed")`。参数 agent_id 使用实际返回的代理 ID 或规范任务路径；失败则用 `"failed"`。该函数自动计算输入输出哈希并原子发布回执。
5. 主代理继续服务下一请求，直至求解程序退出；不得向子代理提供评测参考成本、参考解或停止阈值。每次新建代理，不使用 followup 延续上下文。
6. 取消时先使用 `publish(root / "cancel.json", {})` 发出程序取消信号，终止正在运行的子代理，并等候求解程序退出。总预算截止同样需要 Codex 终止仍活跃的子代理；迟到结果不回传。

模型不额外设置 token 上限或温度。子代理工具不可得时记录失败，不切换到 HTTP 或其他模型。不宣称未知 token 用量为零或无限。

## 自动反馈和结果

`run.json` 保存运行 ID、参数、输入与源码哈希。每次调用保存提示词、请求、候选、回执、消费状态及程序生成的 `feedback.json`。区间中的机组和小时均从 1 开始，起止小时均包含。

反馈关联原始提案、修复、缓存、验证状态和真实成本。成本变化相对该次调用开始前的最佳方案；负值代表降低。未知修复/不可行原因保留 null。未完成验证的 feasible 为 null，不能作为不可行证据。重复方案可以引用已完成的缓存结果，但不会再次记为刷新最佳解。

`events.jsonl` 保存全流程，`best-checkpoint.json` 在每次完整验证后的最佳解更新时立即写入；异常或中断也保留此文件。正常结束另有 `result.json` 和 `state.json`。已消费请求不会再次启动程序侧交换。

运行后执行：

```powershell
python -m suc_benders.astra_audit results/astra-new-run
```

审计核对请求/回执身份、哈希、模型配置、自动反馈跨轮一致性，输出 `audit.json`。它不调用模型。端到端耗时包含聊天调度、等待、文件传输和求解；`agent_wait_seconds` 是程序等待交接的时长，不是纯模型推理时间。

时间截止在求解器和交接检查点生效；正在执行的 SA SDK 调用不能由 Python 在内部立即中断，超时量保存在结果中。已有完整验证的最佳解立即写入检查点，截止后返回的模型结果不接受。

默认数据为 RTS-24 三场景配置。旧实验适配器已删除；`python -m suc_benders` 与本页的 `suc_benders.astra_run` 使用同一入口。
