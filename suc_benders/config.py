"""Core options plus ignored local SDK/LLM credentials."""
from dataclasses import dataclass, field, fields
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Options:
    data_file: str = str(ROOT / "data" / "units.json")
    scenario_file: str = str(ROOT / "data" / "stochastic.json")
    block_hours: int = 3
    max_outer: int = 20
    max_inner: int = 20
    tolerance: float = 1e-4
    time_limit: float = 600
    seed: int = 7
    verbose: bool = True
    theta_step: float = 1
    penalty: float = 1e6
    reserve_scale: float = 100
    sa_reads: int = 16
    sa_sweeps: int = 5
    sa_temperature: float = 100
    sa_cooling: float = .98
    sa_cutoff: float = .01
    kaiwu_user_id: str = field(default="", repr=False)
    kaiwu_sdk_code: str = field(default="", repr=False)
    llm_enabled: bool = False
    llm_url: str = field(default="", repr=False)
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    llm_timeout: float = 600
    llm_patience: int = 3
    llm_max_calls: int | None = 3


def load_options(overrides=None, config_path=None):
    o = Options()
    path = ROOT / "config.local.json" if config_path is None else Path(config_path)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if set(raw) - {"kaiwu", "llm"}:
            raise ValueError("Unknown local configuration section")
        for section, values in raw.items():
            for name, value in values.items():
                key = f"{section}_{name}"
                if key not in {f.name for f in fields(o)}:
                    raise ValueError(f"Unknown configuration option: {key}")
                setattr(o, key, value)
    for name, value in (overrides or {}).items():
        if name not in {f.name for f in fields(o)}:
            raise ValueError(f"Unknown option: {name}")
        setattr(o, name, value)
    for name in ("block_hours", "max_outer", "max_inner", "sa_reads", "sa_sweeps", "llm_patience", "seed", "llm_max_calls"):
        v = getattr(o, name)
        if name == "llm_max_calls" and v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < (0 if name in ("seed", "llm_max_calls") else 1):
            raise ValueError(f"Invalid integer option: {name}")
    for name in ("time_limit", "theta_step", "penalty", "reserve_scale", "sa_temperature", "sa_cutoff", "llm_timeout", "tolerance"):
        v = getattr(o, name)
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or v < 0 or (v == 0 and name != "tolerance"):
            raise ValueError(f"Invalid numeric option: {name}")
    if not 0 < o.sa_cooling < 1 or not o.sa_cutoff < o.sa_temperature:
        raise ValueError("Require 0 < sa_cooling < 1 and sa_cutoff < sa_temperature")
    if not isinstance(o.llm_enabled, bool) or not isinstance(o.verbose, bool):
        raise ValueError("llm_enabled and verbose must be boolean")
    for name in ("kaiwu_user_id", "kaiwu_sdk_code", "llm_url", "llm_model", "llm_api_key"):
        if not isinstance(getattr(o, name), str):
            raise ValueError(f"{name} must be a string")
    if o.llm_enabled and not all((o.llm_url, o.llm_model, o.llm_api_key)):
        raise ValueError("Fill LLM URL, model and API key before enabling LLM")
    return o
