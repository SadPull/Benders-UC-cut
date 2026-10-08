# 种子 7 实验归档

对应代码版本：[v0.23.0](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.23.0)。本目录是精简发布归档；[report.md](report.md) 保持实验结束时的报告内容，原始证据在 Release 附件中。

## 结果

| 方法 | 成本 | 相对独立参考成本差 | SA 调用 | 子代理尝试 / 成功 / 新增可行候选 | 停止原因 |
|---|---:|---:|---:|---:|---|
| SA＋子代理 | 885425.011365 | 0.0405% | 9 | 7 / 5 / 18 | 总时间上限 |
| 纯 SA | 925317.577290 | 4.5478% | 20 | 0 / 0 / 0 | 第一个块内层尚未收敛 |

配置为 24 小时分成 8 个 3 小时块。两组实际都只运行到外层第一轮第一个块（第 1–3 小时），后续块与跨块乘子更新没有执行。SA 直接搜索当前块；子代理直接提出完整 24 小时开机方案；修复可以跨块，最终成本与物理可行性按全天计算。

本次取消全局调用次数上限，但保留每次停滞触发最多 3 次连续请求。暂停点累计有效时间 1642.668 秒，追加 1800 秒包含第 2 次超时请求的重试等待，总上限为 3442.668 秒；原失败等待保留，暂停与等待用户回复的时间排除。纯 SA 提前停止，实际运行约 11.23 秒。

相对纯 SA 降本 4.3112%，相对原三次上限的混合方法降本 0.2283%。次数上限、时间预算和响应序列均有变化；这是一组种子 7 结果，不能将成本变化全部归因于解除次数上限。

## 文件与下载

- [report.md](report.md)、[cost_curve.png](cost_curve.png)、[curves.csv](curves.csv)：实验结束时的报告与曲线。
- [summary.json](summary.json)、[comparison.json](comparison.json)：最终成绩与原三次上限对照。
- [manifest.json](manifest.json)、[restoration.json](restoration.json)、[protocol_audit.json](protocol_audit.json)：实际执行时的参数、冻结源码哈希、状态重建与审计。
- [release_validation.txt](release_validation.txt)：整理发布后的代码验证。
- [archive.json](archive.json)：归档附件、目录、大小和 SHA256 索引。
- [完整实验 ZIP](https://github.com/SadPull/LLM-benders-SUC/releases/download/v0.23.0/LLM-benders-SUC-v0.23.0-experiments.zip)：原三次上限实验、无限次暂停段（含重试）、追加预算完成段及逐文件清单。
- [发布源码 ZIP](https://github.com/SadPull/LLM-benders-SUC/releases/download/v0.23.0/LLM-benders-SUC-v0.23.0-source.zip) 与 [附件 SHA256](https://github.com/SadPull/LLM-benders-SUC/releases/download/v0.23.0/SHA256SUMS.txt)。

实验 ZIP 内保留 `results/` 相对目录及文件原始字节，解压后可按 `FILE_SHA256.json` 核验。快照中的版本号仍是执行实验时的 0.22.0；发布源码为 0.23.0，版本号变更不追写到历史快照。`manifest.json` 的原始绝对路径和当时 Git 状态作为来源证据保留，重放时需使用匹配的快照和参数路径。

新实验运行器依赖 Codex 调度子代理：它通过文件交换等待响应，单独执行 Python 不会自动创建代理。本次专用恢复工具保留在发布源码和完成段快照中，不是任意实验的通用续跑入口。

原始本地配置、许可证、Kaiwu wheel、虚拟环境和无关临时分析文件不包含在发布附件中。
