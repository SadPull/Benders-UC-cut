"""Core solver options and their validation."""
# 模块职责：定义运行参数并检查参数合法性；所有求解时间参数均以秒计。
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# 求解配置的数据容器：参数分为分块/预算与 QUBO/SA 两组。
@dataclass
class Options:
    # 输入路径以仓库根目录为基准，避免命令行工作目录变化时选错算例。
    data_file: str = str(ROOT / "data" / "units.json")
    scenario_file: str = str(ROOT / "data" / "stochastic.json")
    block_hours: int = 3
    # max_rounds 限制遍历全部块的完整轮数（每轮每块一批 SA），不是单个块独占的采样次数。
    max_rounds: int = 400
    tolerance: float = 1e-4
    time_limit: float = 600
    reference_time_limit: float = 120
    seed: int = 7
    verbose: bool = True
    # 基线总费用及割差斜率按最大绝对值归一化；以下权重、温度均以该单位计。
    # sa_reads 为独立退火链数，sa_sweeps 为每个温度对全部位的扫描次数。
    sa_hard_weight: float = 8
    sa_capacity_weight: float = 4
    sa_reads: int = 8
    sa_enabled: bool = True
    # 辅助工作配额独立于搜索 time_limit；耗尽后结束搜索，None 表示不另设此配额。
    sa_budget_seconds: float | None = None
    sa_sweeps: int = 4
    sa_temperature: float = 4  # 自适应起温的下限；实际起温还覆盖原始自由位的典型单比特能垒。
    sa_cooling: float = .95
    sa_cutoff: float = .05


# 整数参数至少为 1，其中 seed 可为 0。
INTEGER_OPTIONS = ("block_hours", "max_rounds", "sa_reads", "sa_sweeps", "seed")
NONNEGATIVE_INTEGERS = ("seed",)
# 数值参数须有限且为正，ZERO_ALLOWED 中的参数还可为 0。
NUMERIC_OPTIONS = ("time_limit", "reference_time_limit", "sa_temperature", "sa_cutoff", "tolerance",
                   "sa_hard_weight", "sa_capacity_weight")
ZERO_ALLOWED = ("tolerance",)
BOOLEAN_OPTIONS = ("verbose", "sa_enabled")


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (float, int)) and np.isfinite(value)


# 按默认值 → 显式覆盖构造 Options，拒绝未知字段和非法数值。
def load_options(overrides=None):
    o = Options()
    names = {f.name for f in fields(o)}
    for name, value in (overrides or {}).items():
        if name not in names:
            raise ValueError(f"Unknown option: {name}")
        setattr(o, name, value)
    for name in INTEGER_OPTIONS:
        v = getattr(o, name)
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
    return o
