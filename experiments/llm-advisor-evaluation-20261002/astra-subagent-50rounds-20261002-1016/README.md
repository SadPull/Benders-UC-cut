# Astra：种子 7 提前停止试验

目录名称保留原始 50 轮计划，实际按用户修改后的条件在第 7 轮停止。完整轮次为 6，子代理调用为 7；原始 5×50 计划未完成。

[实验报告](report.md)、[审计](audit.json)、[完整原始归档](reproducibility.zip)。总览与比较边界见[上级说明](../README.md)。

压缩包中的文件相对于原工作目录 `results/astra-subagent-50rounds-20261002-1016/` 保存，内容逐字节保留。源文件与归档哈希见上级 `manifest.json` 和 `SHA256SUMS`。在新的检出中解压可以查看原始证据，不会自动运行实验。

[最终汇总](final-summary.json)、[最佳调度](best-schedule.json)、[修改后的停止规则](protocol-amendment.json)及[逐轮输入输出](runs/sa_llm__s7/calls/)可直接查看。驱动源码仍保存原始 50 轮配置；实际停止由当前聊天执行的 `accept_and_observe.py` 控制，脚本和记录位于压缩包内。
