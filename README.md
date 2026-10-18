# LLM Benders SUC

Python 机组组合求解器：Kaiwu SA 搜索可行上界，Codex 提供异步上界候选，独立数值进程认证下界。

当前默认输入为 Pisa IEEE118 派生的 **54 机组、24 小时、单场景、零备用**；启动费按 `fixedCost` 列映射为零。只修改 `data/units.json` 和 `data/stochastic.json` 即可更换输入，来源与费用口径见 `data/provenance.json`。

## 运行

环境：Windows 64 位、Python 3.10、Gurobi 许可证、Kaiwu 1.4.1 授权。Kaiwu Windows wheel 由用户自行提供，不在仓库中分发。

```powershell
py -3.10 -m pip install -r requirements.txt C:\path\kaiwu-1.4.1-cp310-none-win_amd64.whl
```

保留自己的 `config.local.json`，不要上传授权信息。首次配置可使用：

```json
{
  "kaiwu": {"user_id": "", "sdk_code": ""},
  "llm": {"enabled": false, "url": "", "model": "", "api_key": "", "max_calls": 3}
}
```

在 VS Code 中直接运行 `run_experiment.py`。预算、种子、LLM 开关等参数位于文件顶部；`CHECK_ONLY=True` 只检查输入。入口默认启用 Codex CLI，使用已有登录、`gpt-6-astra/max`，不需要在项目中填写 API 密钥。缺少 Codex 时可设置 `LLM_ENABLED=False`。

主算法默认预算 1800 秒、seed 17、目标认证 gap 0.1%；之后另用 120 秒求参考解，参考结果不回填主算法。联合搜索开启时，块 SA 每三轮执行一次；首次完整轮改善不足 0.05% 时可触发 LLM，最多 3 次、单次最多 1200 秒、预留 60 秒。

结果自动写入新建的 `results/manual-seed.../`：

- `summary.json`：状态、UB/LB/gap、耗时和统计。
- `result.json.gz`：无损压缩的完整主结果与原始诊断。
- `reference.json`：独立参考解。
- `llm_requests/`：实际触发时的请求与回执。

读取完整结果：

```python
import gzip, json
with gzip.open("results/你的运行目录/result.json.gz", "rt", encoding="utf-8") as f:
    result = json.load(f)
```

程序接口仍为 `from run_suc import run_suc`。命令行 `run_suc.py --output path.json` 保持显式 JSON 输出兼容；普通运行优先使用上述一键入口。

## 核心文件

- `suc_benders/`：物理模型、QUBO/SA、修复调度、认证下界和 LLM 协作。
- `run_suc.py`、`run_experiment.py`：程序接口与一键运行入口。
- `data/`：当前输入、原始算例与溯源信息。
- `config.local.json`、`.local/*.lic`：仅本地保留的授权配置。

## 已知状态与归档

54 机组最近一次修复后运行在 2637.87 秒返回 `COORDINATION_FAILED`，认证 gap **0.980144%**，未达到 0.1%。最终方案和独立下界证据核查通过；第 5 次协调的最小 L1 乘子选择 LP 返回 INFEASIBLE，该问题尚未修复。QUBO 六组冻结输入逐项等价，中位编码提速 8.35 倍；不能将它表述为整体算法的收敛加速。

[2026-10-18 完整资料归档](https://github.com/SadPull/LLM-benders-SUC/releases/tag/archive-20261018-lean)保存全部现有实验结果、证明、LLM 记录、分析脚本、报告、完整测试、历史算例以及清理前源码与未提交改动。原始文件按 SHA256 核验后才从本地移除。

下载 Release 中的 `research-archive-20261018.zip` 与 `archive-manifest.json`，解压到项目外的新目录。`workspace/` 保留原始相对路径；`before-cleanup/` 保留本次压缩输出与清理前的对应文件。授权文件不在归档中，需要自行配置。恢复后的工作快照可运行 `py -3.10 -m pytest -q -m "not integration"`；如未安装测试依赖，先安装 `pytest>=8,<10`。

本地只保留运行所需文件，测试与历史资料从上述归档按需恢复。已有正式实验的状态、预算和结果没有因清理而改写。
