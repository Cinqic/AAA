"""Small, explicit error bookkeeping shared by the session and the experiment.

Nothing here is a benchmark metric. These are the plain arithmetic helpers the
exploratory Playground uses so that the numbers on screen and the numbers in
the JSON evidence come from the same code path and can be recomputed by hand.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


def mean_absolute_error(values: Sequence[float]) -> float | None:
    """Mean of already-computed absolute errors, or ``None`` when empty."""

    if not values:
        return None
    total = 0.0
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("cannot average a non-finite error")
        total += abs(number)
    return total / len(values)


@dataclass
class ErrorTrack:
    """Cumulative and rolling absolute error for one named predictor.

    Errors are stored *normalized by the interval width* so the panels and the
    reported numbers are on one comparable scale.
    """

    name: str
    window: int = 25
    errors: list[float] = field(default_factory=list)
    _rolling: deque[float] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("rolling window must be at least one step")
        self._rolling = deque(self.errors[-self.window :], maxlen=self.window)

    def add(self, error: float) -> None:
        value = abs(float(error))
        if not math.isfinite(value):
            raise ValueError(f"predictor {self.name} produced a non-finite error")
        self.errors.append(value)
        self._rolling.append(value)

    @property
    def count(self) -> int:
        return len(self.errors)

    @property
    def cumulative_mae(self) -> float | None:
        return mean_absolute_error(self.errors)

    @property
    def rolling_mae(self) -> float | None:
        return mean_absolute_error(list(self._rolling))

    def reset(self) -> None:
        self.errors = []
        self._rolling = deque(maxlen=self.window)


def summarize(errors: Iterable[float]) -> dict[str, float | int | None]:
    """A compact, recomputable summary of one error sequence."""

    values = [abs(float(value)) for value in errors]
    return {
        "count": len(values),
        "mae": mean_absolute_error(values),
        "max_absolute_error": max(values) if values else None,
    }
