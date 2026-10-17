# 版本与实验恢复

当前版本为 [v0.31.0](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.31.0)，发布索引见 [v0.31.0.json](v0.31.0.json)。代码、输入、紧凑结果和报告保存在 Git；完整实验日志、历史冻结源码与发布校验保存在同版本 Release 附件。

## 当前版本附件

- `core-source-v0.31.0.zip` 从 Git 标签构建，代表实际发布源码。
- `rcuc20-experiments-v0.31.0.zip` 保存初始化失败、初始化专项、1800 秒含 LLM 运行及两组 300 秒配对的完整记录；内含逐文件 SHA256。
- `rcuc20-report-v0.31.0.md`、`rcuc20-summary-v0.31.0.json` 和 `rcuc20-comparison-v0.31.0.json` 分别提供报告、总表及同预算配对指标。
- `release-validation-v0.31.0.json` 记录发布源码回归和调度复核；`release-manifest.json`、`SHA256SUMS.txt` 记录附件身份、大小及哈希。

先核对下载文件的 SHA256，再解压到新的审计目录。输入、原始日志和冻结试验代码应保持对应；发布后工作树中的版本号或默认开关，不应回写到历史证据中。实验结果可用于复核调度与费用，不代表保存了 Gurobi 搜索树或在途 SDK 状态。

## 旧版本和回退

v0.30.0 核心合并、v0.29.0 RTS-24、v0.28.0 冻结状态及更早实验，分别保存在各自 Git 标签与 GitHub Release。原始恢复清单随相应旧标签和附件保留，当前主线不重新导入旧算法或改写旧成绩。

仅停用新搜索时设置入口 `ENHANCED_SEARCH=False` 或核心 `enhanced_search=False`。需要恢复整个旧版本时，在独立目录检出相应标签，按其依赖与数据运行。Gurobi、Kaiwu 授权及 Codex 登录由使用者本地配置，发布包不包含凭据、许可证或 Kaiwu wheel。
