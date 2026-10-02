# RTS-24 SA / LLM 停滞触发实验

本次三场景 SUC 实验已达标结束；[报告](report.md)记录成本与完整阶段时间线。

- 5 次 SA；第 3–5 轮连续进展不足，触发第一组 LLM。
- 第 1 次 LLM 验证 4 个候选；第 2 次 LLM 的第 1 个候选验证后参考差为 0.455455%，立即停止。
- 没有运行第 3 次 LLM；第 2 次剩余 3 个提案保留为未验证，不能解读为不可行。
- 完整物理、成本、调度与调用审计通过；173 项回归测试通过。

`raw-run.zip` 保存完整运行目录：冻结源码、输入、参考解、事件、完整调度、检查点、所有调用输入/输出/回执/反馈、审计与日志。
内部 `SHA256SUMS.json` 校验各文件，外部 `checksums.json` 校验归档及方便查阅的副本。
运行源码提交为 `e5fc1b213714bf195aa2857ae890f8fc1059bdd2`，后续提交仅归档和文档。

复现实验需由 Codex 服务文件请求，详见[模块约定](../../docs/astra-module.md)。使用新的输出目录：

```powershell
python -u -m suc_benders.astra_run --data data/rts24/units.json --scenario-file data/rts24/stochastic.json --seed 7 --strategy stagnation_burst --target-reference-gap 0.005 --output-dir results/rts24-sa-llm-burst-new
```

离线审计不调用模型或求解器。将归档内容解压到仓库的 `results/rts24-sa-llm-burst-20261002` 后，从仓库根目录执行：

```powershell
python -c "import runpy; runpy.run_path('results/rts24-sa-llm-burst-20261002/audit-run.py', run_name='__main__')"
```

LLM 输出并非确定性结果；单种子实验不证明普遍精度或速度优势。参考差与基于独立下界的认证差距分开报告。
