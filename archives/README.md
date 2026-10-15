# v0.30.0 核心清理发布与恢复

[v0.30.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.30.0) 保存当前 17 模块核心、手动实验入口、发布验证及两份清理前源码。完整测试 191 项通过，包含 3 项真实 Kaiwu 检查；本版没有重跑正式长实验，不改写 v0.29.0 及更早成绩。

| 发布附件 | 用途 |
|---|---|
| `core-source-v0.30.0.zip` | 标签对应的当前源码、输入、测试、文档及便携 VS Code 设置 |
| `source-before-mainline-cleanup-v0.30.0.zip` 及同名 `-manifest.json` | 80 文件的主线清理前快照及逐文件 SHA256 |
| `source-before-module-merge-v0.30.0.zip` 及同名 `-manifest.json` | 72 文件的模块合并前快照及逐文件 SHA256 |
| `release-validation-v0.30.0.json` | 本版测试、数值一致性、导入和私密文件排除检查 |
| `release-manifest.json`、`SHA256SUMS.txt` | 附件名称、大小、SHA256 和发布源码提交 |

先核对整包 SHA256，再解压到空的独立目录。两份历史源码包按各自 manifest 逐文件核对，不覆盖当前源码；当前核心 ZIP 使用 `LLM-benders-SUC-v0.30.0/` 顶层目录。所有附件均排除真实授权配置和许可证，须自行准备运行授权。清理时记录的 `published=false` 保留其当时状态，本次远端转存以 [v0.30.0.json](v0.30.0.json) 和 Release 清单为准。

# 2026-10-15 模块合并与恢复

模块合并前的 21 文件版本保存在 `C:/Users/12433/Desktop/LLM-benders-SUC-archive/module-merge-20261015-173242`。`source-before-merge.zip` 包含 72 个当前源码、测试、数据及文档文件，已按 `manifest.json` 逐文件回读核验；`working-tree.patch` 和 `status-before.txt` 保存合并前已有修改。授权配置、许可证和 `.vscode` 未打入源码包，单独记录哈希并核对原文件。

[模块合并记录](module-merge-20261015.json) 保存归档哈希、四组模型一致性和回归结果。需要恢复合并前导入路径时，将 ZIP 解压到独立空目录，核对清单后再使用；对应源码包和 manifest 已纳入 v0.30.0 发布附件。工作区期间的其他并行改动单独记录并保留，既有主线清理与历史实验归档继续有效。

# 2026-10-15 主线清理与恢复

本次清理前的源码、测试、数据及未提交改动保存在外部目录 `C:/Users/12433/Desktop/LLM-benders-SUC-archive/mainline-cleanup-20261015-165807`。`source-before-cleanup.zip` 包含 80 个文件，`manifest.json` 保存逐文件大小及 SHA256，归档已逐文件回读核验；`working-tree.patch` 和 `status-before.txt` 记录原有修改。授权配置和许可证排除在外。

[本次记录](mainline-cleanup-20261015.json) 保存整包哈希、模型一致性和测试结果。需要恢复旧模式时，将 ZIP 解压到空的独立目录，先核对整包及逐文件哈希，再按该快照准备运行环境；不覆盖当前主线。对应源码包和 manifest 已纳入 v0.30.0 发布附件，既有 v0.29.0 及更早实验的发布附件与恢复清单继续保留。

# v0.29.0 冷启动实验归档与恢复

[v0.29.0 Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.29.0) 保存本轮全部现有结果、预检与失败批次，以及此前仅在本地的 v0.28.0 快筛证据。仓库为私有仓库，下载需要有权限的账户。各包 SHA256、文件数和范围见 [v0.29.0.json](v0.29.0.json)；完整发布附件及远端校验记录见 Release 的 `release-manifest.json`、`release-transfer.json` 和 `SHA256SUMS.txt`。

| 附件 | 内容 |
|---|---|
| `source-before-cleanup-v0.29.0-part001.zip` | 清理前源码、实验控制器与专用测试、输入和原文档 |
| `experiments-v0.29.0-partNNN.zip` | 全部现有 `results/`，含正式快照、恢复段、原始采样、请求回执、模型、割和独立审计 |
| `historical-v0.28.0-v0.29.0-part001.zip` | 原 v0.28.0 外部归档，置于 `historical/v0.28.0/`，含原 ZIP 与核验记录 |
| `core-source-v0.29.0.zip` | 本标签的精简核心源码 |
| `experiment-report-v0.29.0.md`、JSON 和 PNG | 中文报告、机器可读结果及上下界曲线 |

每包小于 1 GiB；内部 `_archive/<包名>-manifest.json` 保存逐文件 SHA256，独立解压时不会覆盖其他包的清单。所有包均逐文件回读核验；上传后重新下载核对整包 SHA256，完成后才迁出项目文件。完整本地副本位于 `C:/Users/12433/Desktop/LLM-benders-SUC-archive/v0.29.0`，本地授权配置和许可证不在附件中。

恢复时先下载包及 `SHA256SUMS.txt`，用 `Get-FileHash -Algorithm SHA256` 核对，再解压到空的独立目录。需要重建本轮全貌时同时解压源码包与全部实验分包。主报告位于 `results/upper-validation-20261015-085155-a4fceb4c/report.md`；正式运行入口在该目录的 `snapshot/`，A 的恢复版本链、计时和偏差见 `recovery.json`、`audit-amendment.json` 与 `postflight/`。不要用当前发布源码替代正式冻结源码。

本轮环境为 Windows、Python 3.10、Gurobi 13.0.1、Kaiwu 1.4.1；有效授权须自行准备。历史协议保留原绝对路径，跨机使用前须核对。新实验使用新结果目录并重新预检；不要自动续跑本轮已完成批次。A 数值通过但预算未通过，B 冷启动收敛至 0.043655%；归档和清理不改变该结论。

# v0.28.0 下界强化核心与本地证据

源码固定为 [v0.28.0 Git 标签](https://github.com/SadPull/LLM-benders-SUC/tree/v0.28.0)。本地归档位于 `C:/Users/12433/Desktop/LLM-benders-SUC-archive/v0.28.0`，ZIP 名称、SHA256、成绩和归因范围见本目录 `v0.28.0.json`。逐文件清单和回读校验保留在归档 `manifest.json`。

`before-cleanup-v0.28.0.zip` 保存清理前工作区、两轮快筛、完整模型／割／证明和必要恢复辅助。解压到独立目录并核对逐文件哈希；最新后续快筛的 `snapshot` 自包含核心数值代码，历史第一轮入口仍含原绝对路径，不能直接当作当前入口运行。原数值结果使用 Python 3.10、Gurobi 13.0.1，源于共同历史证据，不代表冷启动成绩。授权配置和许可证保持本地，不进入包中。

核心与已计分 M1 的数值矩阵一致，验证见归档 `verification/core-numerical-identity.json`。一份变动 MPS 副本已保留在 `verification/corrupted-source-base.mps`，由匹配协议的原始副本恢复并重验；原因未确定，记录见 `verification/source-recovery.json`，不改写成绩。

# v0.26.0 归档与恢复

[GitHub Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.26.0) 保存 2026-10-12 至 13 日全部现有实验证据和清理前源码。仓库为私有仓库，下载需要有权限的账户。附件名称、字节数、SHA256、原始文件数与分包内容见本目录 v0.26.0.json 及 Release 的 release-manifest.json。

| 附件 | 内容 |
|---|---|
| source-before-cleanup-v0.26.0-part001.zip | 清理前源码、全部原测试、计划、输入、历史参考数据和原文档 |
| experiments-v0.26.0-partNNN.zip | 所有现有 results 文件，保留原项目相对路径；各分包独立可解压 |
| experiment-summary-v0.26.0.md 和 JSON | 阶段结果、正式计分、独立审计、LLM 归因限制及停滞诊断 |
| integer-bounds-v0.26.0.png 和 continuous-bounds-v0.26.0.png | 原始上下界及连续分支增益曲线 |
| release-validation-v0.26.0.json | 清理前与核心测试、入口检查及数值源码身份验证 |
| SHA256SUMS.txt | 所有发布附件的 SHA256 |

每包小于 1 GiB；内部 `_archive/<包名>-manifest.json` 保存逐文件哈希。分包之间的清单名称不同，解压不会相互覆盖。所有文件都经过压缩包回读校验；完成远端附件回读校验后才移出原项目。

## 恢复实验

1. 下载清理前源码包、实验分包和 SHA256SUMS.txt，逐包用 Get-FileHash 核对。
2. 解压到一个空的独立目录，不覆盖当前核心代码。按清单核对所需文件；需要全套证据时解压全部实验分包。
3. 准备 Python 3.10、Kaiwu 1.4.1 及自己的授权。当前核心 requirements 保持原 Gurobi 13.0.1 默认；此次冻结实验和发布回归使用 Gurobi 12.0.2，应在独立环境安装依赖后单独切换到 gurobipy==12.0.2。
4. 历史复核在对应 snapshot 目录使用原协议和原输入。协议含原机绝对路径，跨机器恢复须核对路径引用；本发布不声称运行环境可无条件迁移。新的实验必须创建新的结果目录，不自动续跑已停止的批次。

完整本地归档位于 `C:/Users/12433/Desktop/LLM-benders-SUC-archive/v0.26.0`。旧 Python 环境、基线备份和缓存仅保留在该目录，不作为远端附件；config.local.json 和许可证仍保留本地，未上传。

## 当前核心范围

保留 run_suc.py、suc_benders 求解包、必要输入及来源、核心测试与夹具、依赖和版本文档，以及 Git 历史和本地授权配置。benchmarks、实验专用模块、计划、参考求解结果、结果目录和运行环境已移出。历史性能工作树没有合入或清理。

## v0.25.0 历史归档与恢复

[GitHub Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.25.0) 保存非核心文件。仓库是私有仓库，下载需要使用有权限的 GitHub 账户。源码由 `v0.25.0` 标签固定；本目录的 `v0.25.0.json` 记录附件名称、字节数、SHA256 和原始文件数。

v0.25.1 已移除旧 SMS100、短测及整数认证 C/D 入口。当时的 C 组只使用 Python 环境，按根目录 README 安装；以下 Julia 和旧入口恢复说明适用于 v0.25.0 源码或历史冻结源码。

| 附件 | 内容 | 何时需要 |
|---|---|---|
| `experiments-v0.25.0.zip` | 本次清理前全部 `results/`：正式与失败批次、预检、审计、原始采样、代理回执、冻结源码、清理记录 | 查阅或复核历史实验 |
| `runtime-julia-v0.25.0.zip` | `.benchmark/azema/julia-1.11.5/` | 恢复原 Julia 环境 |
| `runtime-julia-depot-v0.25.0.zip` | `.benchmark/azema/julia-depot/` 包、二进制依赖和缓存 | 恢复原 Julia 环境 |
| `runtime-python-and-sources-v0.25.0.zip` | Python 环境、上游固定源码、Julia Project/Manifest、环境清单与 profiling 证据 | 恢复原实验环境 |
| `previous-cleanup-v0.25.0.zip` | 此前移出的 742 个 `.pyc` 和 Julia 安装压缩包，使用原项目相对路径 | 仅用于还原清理前全貌；运行不需要 |

各 ZIP 内的 `_archive/manifest.json` 列出每个文件的 SHA256。归档时逐文件回读校验，上传后下载附件再次核对整包 SHA256，再删除本地实验与运行环境副本。报告另有小型 Markdown 附件，便于直接阅读。

## 按需恢复

1. 下载所需 ZIP、`SHA256SUMS.txt`，用 `Get-FileHash -Algorithm SHA256 <文件>` 对照本仓库清单；不要只凭文件名判断版本。
2. 在项目根目录解压相应 ZIP，内部路径会恢复为 `results/...` 或 `.benchmark/...`。已有同名实验时先解压到空目录比对，避免覆盖原始证据。每个压缩包的 `_archive/manifest.json` 独立保存，提取多个包时不要覆盖掉上一包的清单。
3. 在原 Windows 路径恢复运行环境时，需要三个 `runtime-*` 包；`previous-cleanup` 包为可选的历史冗余文件。`.local/gurobi-runtime.lic` 保留原 Python 包内许可证；如需复原该包，将其复制到 `.benchmark/azema/python/Lib/site-packages/gurobipy/gurobi.lic`。有效的个人 Gurobi/Kaiwu 授权仍须自行配置。
4. 历史审计应使用实验目录内冻结源码和原参数。当前源码更新了数据入口，不能冒充当时源码；新实验使用新输出目录并重新预检。

运行环境包保留了这台 Windows 机器的路径，不是通用安装包。Python 虚拟环境使用 `include-system-site-packages=true`，依赖系统 Python 3.10 与已授权 Kaiwu 1.4.1；本归档不包含系统 Python、Kaiwu 私有 wheel 或个人授权。Julia 包缓存可重建，其中一个文档符号链接已按目标内容存为普通文件，清单保留原链接信息。项目代码没有运行时符号链接依赖。

## 重建依赖

经典 10 机组使用根目录 README 的安装步骤。需要复现 v0.25.0 的 RTS-24/SMS100 框架时，先把该标签源码或实验冻结源码解压到独立目录，准备 Python 3.10、Kaiwu SDK 和有效许可证，再从该历史源码根目录运行：

```powershell
py -3.10 -B -m benchmarks.azema.setup
.\.benchmark\azema\python\Scripts\python.exe -B run_rts24_paper_c.py --help
```

此历史命令按固定上游提交及输入校验和重建 `.benchmark/azema/`，使用 Gurobi 12.0.2 与 Julia 1.11.5，只准备环境，不启动求解或 LLM 请求。需要网络；新机器优先重建，避免沿用旧机器的虚拟环境绝对路径。`config.local.json`、`.local/` 和 `*.lic` 均被 Git 忽略，本地配置不在 Release 附件内。

## v0.25.0 当时的保留范围

`suc_benders/`、精简后的 `benchmarks/`、`run_suc.py`、`run_rts24_paper_c.py`、`tests/`、`data/`、原始 Azema 输入包、依赖和文档，以及 `.git/` 与本地授权配置。回归夹具属于核心验证数据，保留历史对照源码；历史大文件仍可通过原标签和 Release 找回。本次归档只包含清理时实际存在的项目文件和已记录的外部缓存暂存目录，不声称恢复更早已丢失的文件。
