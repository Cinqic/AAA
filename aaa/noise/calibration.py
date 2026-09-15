"""Causal observation-prediction interval calibration.

The calibrator sees only the forecast and the newly revealed noisy target.
Latent truth is never passed to it.  It is intentionally a separate object
from point-predictor state so interval adaptation cannot be mistaken for
point-prediction learning.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np


class CalibrationError(ValueError):
    """Raised when a causal calibration input is invalid."""


class CausalResidualCalibrator:
    """Symmetric residual-quantile intervals from past noisy residuals only."""

    def __init__(self, *, levels: tuple[float, ...] = (0.90, 0.95), minimum_samples: int = 8) -> None:
        if not levels or any(not 0.0 < level < 1.0 for level in levels):
            raise CalibrationError("interval levels must lie strictly between zero and one")
        if minimum_samples < 1:
            raise CalibrationError("minimum_samples must be positive")
        self.levels = tuple(float(level) for level in levels)
        self.minimum_samples = int(minimum_samples)
        self._absolute_residuals: deque[float] = deque(maxlen=4096)

    @property
    def sample_count(self) -> int:
        return len(self._absolute_residuals)

    def intervals(self, forecast: float) -> dict[str, dict[str, float] | None]:
        if not math.isfinite(forecast):
            raise CalibrationError("forecast must be finite")
        if self.sample_count < self.minimum_samples:
            return {f"{level:g}": None for level in self.levels}
        residuals = np.asarray(self._absolute_residuals, dtype=float)
        result: dict[str, dict[str, float] | None] = {}
        for level in self.levels:
            radius = float(np.quantile(residuals, level, method="higher"))
            result[f"{level:g}"] = {
                "lower": float(forecast - radius),
                "upper": float(forecast + radius),
            }
        return result

    def update(self, forecast: float, noisy_target: float) -> None:
        if not math.isfinite(forecast) or not math.isfinite(noisy_target):
            raise CalibrationError("forecast and noisy target must be finite")
        self._absolute_residuals.append(abs(float(noisy_target) - float(forecast)))

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": "aaa.causal_residual_calibrator.v1",
            "levels": list(self.levels),
            "minimum_samples": self.minimum_samples,
            "absolute_residuals": list(self._absolute_residuals),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> CausalResidualCalibrator:
        if state.get("format_version") != "aaa.causal_residual_calibrator.v1":
            raise CalibrationError("unsupported calibrator checkpoint")
        result = cls(
            levels=tuple(float(value) for value in state["levels"]),
            minimum_samples=int(state["minimum_samples"]),
        )
        for value in state["absolute_residuals"]:
            if not math.isfinite(float(value)) or float(value) < 0:
                raise CalibrationError("invalid residual in calibrator checkpoint")
            result._absolute_residuals.append(float(value))
        return result


def interval_score(interval: dict[str, float], target: float, level: float) -> float:
    """Gneiting-Raftery interval score for a central prediction interval."""

    lower = float(interval["lower"])
    upper = float(interval["upper"])
    if not lower <= upper or not math.isfinite(target):
        raise CalibrationError("invalid interval or target")
    alpha = 1.0 - float(level)
    penalty = 0.0
    if target < lower:
        penalty += (2.0 / alpha) * (lower - target)
    if target > upper:
        penalty += (2.0 / alpha) * (target - upper)
    return (upper - lower) + penalty
