# v0.25.0 归档与恢复

[GitHub Release](https://github.com/SadPull/LLM-benders-SUC/releases/tag/v0.25.0) 保存非核心文件。仓库是私有仓库，下载需要使用有权限的 GitHub 账户。源码由 `v0.25.0` 标签固定；本目录的 `v0.25.0.json` 记录附件名称、字节数、SHA256 和原始文件数。

v0.25.1 已移除旧 SMS100、短测及整数认证 C/D 入口。当前 C 只使用 Python 环境，按根目录 README 安装；以下 Julia 和旧入口恢复说明适用于 v0.25.0 源码或历史冻结源码。

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

经典 10 机组和当前 C 组使用根目录 README 的安装步骤。需要复现 v0.25.0 的 RTS-24/SMS100 框架时，先把该标签源码或实验冻结源码解压到独立目录，准备 Python 3.10、Kaiwu SDK 和有效许可证，再从该历史源码根目录运行：

```powershell
py -3.10 -B -m benchmarks.azema.setup
.\.benchmark\azema\python\Scripts\python.exe -B run_rts24_paper_c.py --help
```

此历史命令按固定上游提交及输入校验和重建 `.benchmark/azema/`，使用 Gurobi 12.0.2 与 Julia 1.11.5，只准备环境，不启动求解或 LLM 请求。需要网络；新机器优先重建，避免沿用旧机器的虚拟环境绝对路径。`config.local.json`、`.local/` 和 `*.lic` 均被 Git 忽略，本地配置不在 Release 附件内。

## 本地保留范围

`suc_benders/`、精简后的 `benchmarks/`、`run_suc.py`、`run_rts24_paper_c.py`、`tests/`、`data/`、原始 Azema 输入包、依赖和文档，以及 `.git/` 与本地授权配置。回归夹具属于核心验证数据，保留历史对照源码；历史大文件仍可通过原标签和 Release 找回。本次归档只包含清理时实际存在的项目文件和已记录的外部缓存暂存目录，不声称恢复更早已丢失的文件。
