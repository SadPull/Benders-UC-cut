# RTS-24 核心数据

`inputs/` 是本项目实际使用的 RTS-24 派生算例：32 台机组、24 小时、三个场景、5% 备用，保留二次燃料费用。它与 `results/rts24-pilot-20261011/inputs/` 的原文件逐字节一致。C/D 和论文分工 C 的入口直接读取此目录，不依赖本地历史实验目录。

`reference/manifest.json` 和 `reference/result.json` 原样保留独立 Gurobi MIQP 的输入哈希、模型指纹、开机矩阵、认证界和费用审计。参考方案成本为 **885066.704430**，参考认证下界为 **885065.981876**。参考数据仅用于独立对照，不作为正式搜索的初始化、割或下界输入。

原始参考脚本、日志和冻结源码位于 [v0.25.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.25.0) 的 `experiments-v0.25.0.zip`，路径为 `results/rts24-pilot-20261011/reference/`。原 manifest 的 `../snapshot` 保持历史含义，应在解压后的历史目录中解析。

默认经典 10 机组输入仍位于上级 `data/`；`tests/fixtures/rts24/` 保留独立的历史回归夹具。
