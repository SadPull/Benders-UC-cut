# 版本与实验恢复

当前版本为 [v0.32.0](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.32.0)，发布索引见 [v0.32.0.json](v0.32.0.json)。RCUC20／种子 17 在 873.64 秒达到认证 gap 0.054364%。代码、输入、紧凑结果和报告保存在 Git；完整运行记录保存在同版本 Release 附件。

## 当前版本附件

- `core-source-v0.32.0.zip` 从 Git 标签的原始 blob 构建，代表实际发布源码。
- `rcuc20-experiments-v0.32.0.zip` 保存 300／1800 秒无 LLM 诊断、单请求提示词小试验、一次早期记录脚本中止和最终冷启动收敛实验；附有逐文件 SHA256。用于小试验的历史第 2 次请求及运行证据也一并保存。
- `rcuc20-report-v0.32.0.md`、`rcuc20-summary-v0.32.0.json`、`rcuc20-comparison-v0.32.0.json` 提供改动归因、正式结果与历史对照。
- `release-validation-v0.32.0.json` 记录待发布树回归、最终方案复核和归档核验；`release-manifest.json`、`SHA256SUMS.txt` 记录附件身份、大小及哈希。

先核对附件 SHA256，再解压到新目录。实验 ZIP 中 `rcuc20/` 下的内容可恢复到仓库根目录，对应报告中的 `results/` 链接。每份证据保留实际运行时源码和输入；发布后的版本号、文档或开关不回写到历史记录。保存的文件不是 Gurobi 搜索树或在途工作进程。

本版仅验证一次种子 17 的完整冷启动，未做多种子或提示词单因素验证。工作树中原有计时小试验、教学页面和传输层纯提取重构不属于此次代码提交；实际运行中的传输源码保留在原始证据快照，发布树单独回归验证。

## 旧版本和回退

v0.31.0 RCUC20 初始搜索、v0.30.0 核心合并、v0.29.0 RTS-24、v0.28.0 冻结状态及更早实验，分别保存在各自 Git 标签与 GitHub Release。原始恢复清单随相应旧标签和附件保留，当前主线不重新导入旧算法或改写旧成绩。

仅停用新搜索时设置入口 `ENHANCED_SEARCH=False` 或核心 `enhanced_search=False`。需要恢复整个旧版本时，在独立目录检出相应标签，按其依赖与数据运行。Gurobi、Kaiwu 授权及 Codex 登录由使用者本地配置，发布包不包含凭据、许可证或 Kaiwu wheel。
