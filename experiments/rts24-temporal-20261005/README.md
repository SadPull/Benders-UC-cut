# 时间分解与拉格朗日协调验收（2026-10-05）

默认算法已切换为连续松弛初始化、3 小时分块、块内 Benders/SA、外层拉格朗日切平面。
枚举候选端到端测试验证 1/2/3 小时块均能完成内外层收敛；真实 SA 检查仅验证接入、停止语义、下界与可行方案。

两次真实 SA 各仅运行一次：种子 7、候选池 16、禁用 LLM、总预算 60 秒，默认 20 外层／20 内层，容差 1e-4 + 1e-5。
每个配置都是 8 个三小时块；首块达到 20 次内层上限后按协议结束，没有更新外层乘子。
`INNER_NOT_CONVERGED` 是未收敛结果。既不表示所有 8 块均已求解，也不构成速度优势证据。

| 配置 | 状态 | 秒 | 全天可行成本 | 认证下界 | 原问题 gap | SA 次数 |
|---|---|---:|---:|---:|---:|---:|
| deterministic | INNER_NOT_CONVERGED | 11.141 | 883727.892202 | 815439.903182 | 7.7273% | 20 |
| stochastic | INNER_NOT_CONVERGED | 21.475 | 933522.215367 | 841376.523177 | 9.8708% | 20 |

认证下界包含完整连续松弛的对偶下界，以及已取得块下界与其余块解析下界的有效总和。
协调尚未发生，因此协调上界为空。原问题 gap 使用 `(全天可行成本 - 认证下界) / max(1, abs(全天可行成本))`。
两次运行的每个可行全天事件均已独立物理验证；最大残差分别约 9.1e-13 MW 和 1.4e-12 MW。
逐块上下界、内层诊断、采样记录、可行方案、源文件快照及运行输入哈希见各配置目录；audit.json 为验收结果。

运行命令：

```powershell
python -m suc_benders --strategy sa --scenario-file data/rts24/deterministic.json --seed 7 --time-limit-seconds 60 --output-dir experiments/rts24-temporal-new/deterministic
python -m suc_benders --strategy sa --scenario-file data/rts24/stochastic.json --seed 7 --time-limit-seconds 60 --output-dir experiments/rts24-temporal-new/stochastic
```

真实 SA 的 source/ 保存运行时原样源码；求解器此后补充了失败阶段、LLM-only 额度优先终止及原始二元值检查，未为改善成绩重跑。
最终源码的 138 项回归测试全部通过，完整输出见 [validation-v0.19.0.txt](../../docs/validation-v0.19.0.txt)。
既有归档、冻结源码和历史校验和未改动；教材仅补充新版入口说明。
