from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil, floor, gcd, lcm

import numpy as np


@dataclass
class MasterCandidate:
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    source: str = "quantum"
    provenance: dict = field(default_factory=dict)


@dataclass
class QuantumConfig:
    backend: str = "kaiwu-local"
    kaiwu_seed: int | None = 7
    candidate_pool_size: int = 16
    initial_temperature: float = 100.0
    alpha: float = 0.98
    cutoff_temperature: float = 0.01
    iterations_per_temperature: int = 5


@dataclass(frozen=True)
class _IntegerInequality:
    """step * (constant + sum(coefficients[name] * bit)) <= 0."""

    coefficients: dict[str, int]
    constant: int
    step: Fraction
    rounding_error: float = 0.0


def _capacity_inequality(
    capacity: dict[str, Fraction], requirement: Fraction
) -> _IntegerInequality:
    """Encode a decimal capacity lattice exactly, including a fractional demand."""

    capacity = {name: value for name, value in capacity.items() if value}
    denominator = lcm(*(value.denominator for value in capacity.values()))
    integers = {name: int(value * denominator) for name, value in capacity.items()}
    divisor = gcd(*integers.values()) or 1
    step = Fraction(divisor, denominator)
    return _IntegerInequality(
        {name: -value // divisor for name, value in integers.items()},
        ceil(requirement / step),
        step,
    )


def _quantize_inequality(
    coefficients: dict[str, float], constant: float, error_budget: float,
    exact_coefficients: dict[str, Fraction] | None = None,
) -> _IntegerInequality:
    """Round a supporting plane down, with a bound valid for every binary point."""

    coefficients = {name: value for name, value in coefficients.items() if value}
    step = Fraction(str(error_budget)) / (len(coefficients) + 1)
    # Fractions of the actual floats keep flooring conservative even at grid edges.
    integers = {name: floor(Fraction(float(value)) / step) for name, value in coefficients.items()}
    integer_constant = floor(Fraction(float(constant)) / step)
    error = Fraction(float(constant)) - integer_constant * step
    error += sum(Fraction(float(value)) - integers[name] * step for name, value in coefficients.items())
    for name, value in (exact_coefficients or {}).items():
        integer = value / step
        if integer.denominator != 1:
            raise ValueError("exact coefficient is not on the quantization grid")
        integers[name] = int(integer)
    return _IntegerInequality(integers, integer_constant, step, float(error))
