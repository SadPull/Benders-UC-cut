"""Azema SMS100 benchmark against the pinned author's model."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / ".benchmark" / "azema"
COMMIT = "5b9cf5ec2795b481f4d65c364be37eacfd0b585b"
JULIA_VERSION = "1.11.5"
GUROBI_VERSION = "12.0.2"
