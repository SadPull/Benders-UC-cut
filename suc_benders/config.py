"""Core options plus ignored local LLM credentials."""
# 模块职责：定义运行参数，合并被 Git 忽略的本地授权配置，并检查参数合法性。
# 显式 overrides 优先于 config.local.json；所有求解/请求时间参数均以秒计。
from dataclasses import dataclass, field, fields
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# 求解配置的数据容器：参数分为分块/预算、QUBO/SA 和 LLM/授权三组。
# 授权与敏感 URL 字段使用 repr=False，避免对象打印时泄漏凭据。
@dataclass
class Options:
    # 输入路径以仓库根目录为基准，避免命令行工作目录变化时选错算例。
    data_file: str = str(ROOT / "data" / "units.json")
    scenario_file: str = str(ROOT / "data" / "stochastic.json")
    block_hours: int = 3
    max_outer: int = 20
    # max_inner 限制固定乘子下遍历全部块的完整轮数，不是单个块独占的采样次数。
    max_inner: int = 20
    tolerance: float = 1e-4
    time_limit: float = 600
    seed: int = 7
    verbose: bool = True
    # 搜索 QUBO 以"单个位的最大归一化成本"为能量单位；以下权重、温度均以该单位计。
    # sa_reads 为独立退火链数，sa_sweeps 为每个温度对全部位的扫描次数。
    sa_hard_weight: float = 8
    sa_capacity_weight: float = 4
    sa_reads: int = 8
    sa_enabled: bool = True
    # 辅助工作配额独立于总 time_limit；耗尽后可继续整数认证，None 表示不另设此配额。
    sa_budget_seconds: float | None = None
    sa_sweeps: int = 4
    sa_temperature: float = 4
    sa_cooling: float = .95
    sa_cutoff: float = .05
    # 上界 LLM 请求由 llm_max_calls 限额，单次超时和连续停滞触发次数分别由后续字段控制。
    llm_enabled: bool = False
    llm_url: str = field(default="", repr=False)
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    llm_agent_directory: str = ""
    llm_codex_command: str = ""
    llm_timeout: float = 600
    llm_reserve_seconds: float = 0.
    llm_patience: int = 3
    llm_max_calls: int | None = 3
    # True：初始化后立即发首个请求，不等待 SA 停滞；之后仍按 llm_patience 触发。
    llm_first_call_at_start: bool = False
    # 升级层：常规层一次调用降幅低于 llm_escalation_gain（比例）或失败时，并行发起一次 Codex 请求（隔离用户配置、不续接）；
    # 模型为空即关闭。剩余预算扣除预留后不足 min_seconds 时不发起，单次最多 timeout 秒且不超过剩余预算。
    llm_escalation_model: str = ""
    llm_escalation_command: str = ""
    llm_escalation_directory: str = ""
    llm_escalation_timeout: float = 1800
    llm_escalation_min_seconds: float = 700
    llm_escalation_gain: float = .005


# 整数参数至少为 1，其中 seed 与 llm_max_calls 可为 0（llm_max_calls 为 None 表示不限次数）。
INTEGER_OPTIONS = ("block_hours", "max_outer", "max_inner", "sa_reads", "sa_sweeps", "llm_patience", "seed",
                   "llm_max_calls")
NONNEGATIVE_INTEGERS = ("seed", "llm_max_calls")
# 数值参数须有限且为正，ZERO_ALLOWED 中的参数还可为 0。
NUMERIC_OPTIONS = ("time_limit", "sa_temperature", "sa_cutoff", "llm_timeout", "llm_reserve_seconds", "tolerance",
                   "sa_hard_weight", "sa_capacity_weight", "llm_escalation_timeout", "llm_escalation_min_seconds",
                   "llm_escalation_gain")
ZERO_ALLOWED = ("tolerance", "llm_reserve_seconds", "llm_escalation_min_seconds", "llm_escalation_gain")
BOOLEAN_OPTIONS = ("llm_enabled", "verbose", "llm_first_call_at_start", "sa_enabled")
STRING_OPTIONS = ("llm_url", "llm_model", "llm_api_key", "llm_agent_directory", "llm_codex_command",
                  "llm_escalation_model", "llm_escalation_command", "llm_escalation_directory")


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (float, int)) and np.isfinite(value)


# 按默认值 → 本地配置 → 显式覆盖构造 Options，拒绝未知字段、非法数值和不兼容模式。
# llm_max_calls=None 表示取消全局次数上限，0 表示不发请求，其他预算与停止规则仍适用。
def load_options(overrides=None, config_path=None):
    o = Options()
    names = {f.name for f in fields(o)}
    path = ROOT / "config.local.json" if config_path is None else Path(config_path)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if set(raw) - {"llm"}:
            raise ValueError("Unknown local configuration section")
        for section, values in raw.items():
            for name, value in values.items():
                key = f"{section}_{name}"
                if key not in names:
                    raise ValueError(f"Unknown configuration option: {key}")
                setattr(o, key, value)
    for name, value in (overrides or {}).items():
        if name not in names:
            raise ValueError(f"Unknown option: {name}")
        setattr(o, name, value)
    for name in INTEGER_OPTIONS:
        v = getattr(o, name)
        if name == "llm_max_calls" and v is None:
            continue
        minimum = 0 if name in NONNEGATIVE_INTEGERS else 1
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < minimum:
            raise ValueError(f"Invalid integer option: {name}")
    for name in NUMERIC_OPTIONS:
        v = getattr(o, name)
        if not _number(v) or v < 0 or (v == 0 and name not in ZERO_ALLOWED):
            raise ValueError(f"Invalid numeric option: {name}")
    if not 0 < o.sa_cooling < 1 or not o.sa_cutoff < o.sa_temperature:
        raise ValueError("Require 0 < sa_cooling < 1 and sa_cutoff < sa_temperature")
    for name in BOOLEAN_OPTIONS:
        if not isinstance(getattr(o, name), bool):
            raise ValueError(f"{name} must be boolean")
    if o.sa_budget_seconds is not None and (not _number(o.sa_budget_seconds) or o.sa_budget_seconds < 0):
        raise ValueError('SA budget requires a nonnegative finite value')
    if not o.sa_enabled and o.llm_enabled:
        raise ValueError("Disabling SA requires LLM disabled")
    for name in STRING_OPTIONS:
        if not isinstance(getattr(o, name), str):
            raise ValueError(f"{name} must be a string")
    if o.llm_codex_command and not o.llm_agent_directory:
        raise ValueError("Codex tasks require llm_agent_directory for request evidence")
    if o.llm_enabled and o.llm_escalation_model and not (o.llm_escalation_command and o.llm_escalation_directory):
        raise ValueError("LLM escalation requires llm_escalation_command and llm_escalation_directory")
    if o.llm_enabled and not o.llm_model:
        raise ValueError("Set llm_model before enabling LLM")
    if o.llm_enabled and not o.llm_codex_command and not all((o.llm_url, o.llm_api_key)):
        raise ValueError("Fill LLM URL and API key before enabling the HTTP channel")
    return o
