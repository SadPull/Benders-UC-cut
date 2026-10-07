# Python：Kaiwu SA + LLM Benders SUC

**连续松弛初始化 → 时间分块 → Benders / QUBO / Kaiwu SA → 拉格朗日切平面协调 → 全天恢复与物理校验。**

Python 为默认 `main`，原 MATLAB 代码保存在 `MATLAB` 分支。保留原数学模型，删除实验调度、检查点重放、归因报告、冻结源码和重复实现；旧 Python 历史仍可追溯。

## 安装与运行

需要 Windows 64 位、Python 3.10、有效 Gurobi 许可证和 Kaiwu 授权。使用独立虚拟环境及提供的 SDK：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt .\kaiwu-1.4.1-cp310-none-win_amd64.whl
Copy-Item config.example.json config.local.json
```

只在首次配置时复制模板，保留已有本地配置。填写 `kaiwu.user_id`、`kaiwu.sdk_code` 后运行：

```powershell
.\.venv\Scripts\python.exe run_suc.py
# 短运行及可选保存
.\.venv\Scripts\python.exe run_suc.py --max-outer 1 --max-inner 2 --time-limit 60 --output results/result.json
.\.venv\Scripts\python.exe run_suc.py --scenario-file data/deterministic.json --max-outer 1 --max-inner 2 --time-limit 60
```

提供的 wheel SHA256：`b55b11a491232bda40d13396c0d0b4409b594092510ef0ff50d973893d35a694`。wheel、虚拟环境、真实授权和结果目录均不提交 Git。无需 MATLAB、MATPOWER 或 QPU；Kaiwu 自身依赖由 pip 安装。

```python
from run_suc import run_suc
result = run_suc({"max_outer": 2, "max_inner": 5, "time_limit": 120})
print(result.status, result.objective, result.lower_bound, result.relative_gap)
print(result.validation)
```

不传 `--output` 时不自动保存。`u/v/w` 和 `result.scenarios[k].p/r` 为“机组 × 小时”NumPy 矩阵。结果提供状态、成本、上下界、间隙、独立校验和精简的 `history` / `outer_history` / `llm_history`，不持有求解器对象或完整退火轨迹。Python 中无穷界保留 `inf`，JSON 中表示为 `null`。

## 参数与算法约定

默认 32 台机组、24 小时、3 个场景。`run_suc(options)` 的显式选项优先于本地配置，未知参数报错。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `block_hours` | 3 | 时间块长度，支持较短末块 |
| `max_outer` / `max_inner` | 20 / 20 | 外层与各块内层上限 |
| `time_limit` / `certify_seconds` | 600 / 30 | 总求解与单次认证预算，秒 |
| `tolerance` | 0.0001 | 相对容差，另加固定绝对容差 0.00001 |
| `seed` / `verbose` | 7 / true | 随机种子与进度输出 |
| `theta_step` / `penalty` / `reserve_scale` | 1 / 1000000 / 100 | QUBO 目标步长、罚系数和容量缩放 |
| `sa_reads` / `sa_sweeps` | 16 / 5 | Kaiwu 解池大小 `size_limit` 与 `iterations_per_t` |
| `sa_temperature` / `sa_cooling` / `sa_cutoff` | 100 / 0.98 / 0.01 | 按能量尺度调整的温度、冷却与终温系数 |

- 保留期望二次燃料成本、固定运行与启停费用、负荷、备用、最小出力、启停容量、爬坡、最小开停机时间和初始剩余约束。末小时禁止停机，最小时间约束截断至末小时。
- 变量顺序为变量类型、场景、小时、机组。内部索引从 0 开始。分块不重置初始条件，跨块约束进入协调。
- 下界仅来自重算的连续对偶目标与整数认证 `ObjBound`。认证整数解不进入候选、割、SA 或全天恢复，QUBO 能量不是下界。
- Benders 割向下量化，容量使用最多六位小数的整数格点；认证和调度使用原始精度。乘子更新后重建成本和割、重新调度历史候选；只有通过独立校验的全天方案更新上界。
- Gurobi 单线程、Kaiwu 单进程；仅当前 QUBO 在 SDK 调用时转为稠密 Ising 矩阵。固定种子可重复，不污染调用者的 NumPy 随机状态，不要求复现 MATLAB 轨迹。
- `INNER_NOT_CONVERGED` 表示块内达到上限，停止且不更新乘子。`DUAL_CONVERGED` 只表示对偶协调收敛，原问题仍看 `relative_gap`。还有 `MAX_ITERATIONS`、`TIME_LIMIT` 和带阶段的 `*_FAILED`，原因见 `message`。
- 计时覆盖模型构建与求解，不含输入加载和 SDK 授权。同步 Kaiwu/HTTP 调用不能强制中断，可能超出预算；迟到候选和认证结果不入账。

## 可选 LLM

在 `config.local.json` 的 `llm` 中填写完整 Responses `url`、`model`、`api_key`，将 `enabled` 改为 `true`。默认关闭，`timeout=600`、`patience=3`、`max_calls=3`；对应 `llm_*` 运行参数可覆盖本地设置。

连续 `patience` 批改善不足 0.05% 时请求最多三个候选，每次求解最多调用 `max_calls` 次。按 [Responses 协议](https://developers.openai.com/api/docs/guides/migrate-to-responses) 同步非流式调用，设置 `stream=false`、`store=false`，提取 `output` 中消息的 `output_text`。提示的小时从 1 开始，变量和稀疏矩阵索引明确标记从 0 开始。

候选须经过格式、离散约束、连续调度、全天恢复和物理校验。失败后继续 SA，失败调用也计数；只保存不含密钥的状态摘要。保留默认候选/割融合，不保留实验账本、重放或特殊再升温。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# 无真实授权时，先运行数值和 LLM 离线测试
.\.venv\Scripts\python.exe -m pytest -q -m "not integration"
```

真实 Kaiwu 测试需要本地授权；缺少授权时明确跳过，不能视为真实验收完成。测试不调用真实 LLM 服务。

基线为 MATLAB 提交 `aacef8a6d49ce126a28f70a694c4013f9a1c3b24`。两种场景的矩阵、右端项、目标和边界逐项一致，测试保存 MATLAB 导出的模型指纹。

| 场景 | 约束 / 变量 / 非零项 | MATLAB 固定方案成本 |
|---|---|---:|
| 确定性 | 12648 / 3840 / 30636 | 1223143.120217 |
| 随机 | 21960 / 6912 / 56996 | 1220423.773288 |

真实 Kaiwu 两轮短跑：确定性成本约 883727.892202、下界约 815439.903584；随机成本约 933522.215366、下界约 841376.524840、间隙约 9.8708%。两者独立校验通过且为 `INNER_NOT_CONVERGED`，不表示已达全局最优。固定种子重复结果一致。集成短跑使用较快采样参数，另验证默认采样参数的命令行运行。

## 数据

四个 JSON 原样保留，来源版本、哈希和转换口径见 `data/provenance.json`。公开来源为 [MATPOWER RTS-24](https://github.com/MATPOWER/matpower/blob/8b326f00d496864ed81169f7e684568fc1f7006d/data/case24_ieee_rts.m) 和 [ucacopf RTS79](https://github.com/anyacastillo/ucacopf/blob/001cf29dc2e9b3d26e71146f82db35fa379b4dca/RTS79_ucacopf.dat)。

装机 3405 MW、基础峰值负荷 2850 MW。随机倍率 0.90/1.00/1.05、概率 0.25/0.50/0.25 和 5% 备用率为项目设定。模型为聚合无网络 SUC，不含输电约束、冷热启动成本或水电能量预算。
