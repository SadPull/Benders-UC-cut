# RTS-24 DeepSeek Flash 对比实验（用户停止）

[完整报告](report.md) / [汇总数据](summary.json) / [Astra 对照](../rts24-sa-llm-burst-20261002/report.md)

本次未达到目标：最佳成本 **907432.581261**，相对独立参考差 **2.527027%**。
共 14 次 SA；11 次 LLM 调用启动，其中 10 次完成，最后一次在等待响应时按用户要求中断。
第 8、9、10 次连续达到服务端生成上限，未产出可验证候选。第 11 次最终 token 用量未知。

初始停止门槛为严格小于 0.5%；用户在第 8 次请求期间改为小于等于 1%，随后要求直接停止。
`protocol-amendment.json` 和 `user-stop.json` 保留这两次变更。没有重启实验，也没有伪造原生完成结果。

`raw-run.zip` 包含完整运行目录：初始代码及输入快照、参考调度、事件、最佳检查点、已完成调用的请求/响应/反馈、部分调用、审计、协议变更与停止证据。
内部 `SHA256SUMS.json` 校验原始文件，外部 `checksums.json` 校验归档及查阅副本。
API 密钥不在归档中。运行源码提交为 `adf784ef28ffdf46d797f14d8bb4a7bcbf861eff`。

观察到中断请求的请求体未在响应前落盘后，另行修复了后续版本：HTTP 输入现在在发送前保存。
这项修复没有用于重跑或改写本次实验；本次第 11 次请求体、响应及最终用量仍按未知处理。
运行前及修复后的 32 项相关回归测试均通过，分别见 regression.txt 与 poststop-regression.txt。

离线审计不调用 LLM 或求解器。将归档解压到仓库的 `results/rts24-deepseek-burst-20261002` 后，从仓库根目录运行：

```powershell
python -c "import runpy; runpy.run_path('results/rts24-deepseek-burst-20261002/audit-run.py', run_name='__main__')"
```

停止监督器源码为归档内 stop-supervisor.py。运行参数和初始命令保存在 run.json；本次的中途条件变更应结合上述协议记录解读。
与 Astra 的首次业务提示词逐字节一致，算法与起始 SA 行为一致；模型完整上下文、传输和服务缺省参数并不相同。
这是单种子、主动停止的实验，不据此宣称普遍优劣或纯模型速度优势。
