"""Frozen recovery control with imports from the frozen solver snapshot."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "snapshot" / "suc_benders"))
