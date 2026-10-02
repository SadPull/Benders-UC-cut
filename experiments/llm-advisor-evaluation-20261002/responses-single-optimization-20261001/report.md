# Responses API 单次优化试验

唯一请求已结束，仍然截断，没有产生可用候选。

## 本次设置

- 接口：`https://api.farluna.tech/v1/responses`；模型：`farluna/deepseek-v4.1-flash`。
- 重放种子 17 第 15 轮的同一份 system/user 文本，逐项核对一致，历史事件文件哈希保持不变。
- 参数：`reasoning.effort=max`、`max_output_tokens=65536`、`temperature=0.2`、`stream=false`、`store=false`。
- 无客户端超时；仅发起 1 次 HTTP 请求，没有预检、重试或后续迭代。
- 优化前最佳可行成本：583,861.4308。

## 结果

|项目|历史 Chat Completions|本次 Responses|
|---|---:|---:|
|请求耗时（秒）|472.39|516.26|
|输入 token|13,561|13,587|
|输出 token|65,536|65,536|
|其中推理 token|65,536|65,536|
|正文 token|0|0|
|结束标记|length|incomplete / max_output_tokens|
|可用候选|0|0|

本次比历史请求多等待约 43.87 秒。两次请求并非同时进行，不能据此推断协议本身一定更慢。

响应中没有正文，也没有候选进入物理与成本验证。因此本次 SA 调用 0 次、连续调度求解 0 次；最佳成本保持 583,861.4308，改善为 0。原五种子实验继续保持停止状态。

这次样本表明：单纯改用 Responses 没有解决长等待和输出截断。当前配置仍然在推理阶段耗尽预算，未生成可用于优化的正文。不追加第二次请求。

## 代码与验证

通用客户端已新增 Responses 支持，可通过 `--llm-api-format responses` 选择。请求与返回解析按[官方文档](https://developers.openai.com/api/reference/resources/responses/methods/create)实现，只接受完成响应，并明确记录截断和其他不完整响应。现有 Chat Completions 和 Messages 支持保留。

28 项相关测试通过，见 [tests.log](tests.log)。请求数量、原始文本一致性、历史轨迹哈希及密钥排除检查通过，见 [audit.json](audit.json)。运行时的源码快照保存在 source 目录。

原始证据：[请求](request.json)、[响应](response.json)、[诊断](diagnostics.json)、[摘要](summary.json)、[上下文来源](provenance.json)。
