"""Frozen sensor schedules and the latent/observed boundary for noise runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _label_words(label: str) -> list[int]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return [int.from_bytes(digest[index : index + 4], "little") for index in range(0, 32, 4)]


def derive_seed(root_seed: int, namespace: str, *parts: object) -> int:
    """Derive a stable uint32 seed without Python's process-randomized hash."""

    words = [int(root_seed) & 0xFFFFFFFF, *_label_words(namespace)]
    for part in parts:
        words.extend(_label_words(str(part)))
    return int(np.random.SeedSequence(words).generate_state(1, dtype=np.uint32)[0])


def scale_key(scale: float) -> int:
    """Represent a frozen decimal scale as integer micro-RMS units."""

    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    key = round(scale * 1_000_000)
    if abs(scale - key / 1_000_000) > 1e-12:
        raise ValueError(f"scale {scale!r} is not representable in micro-RMS units")
    return key


@dataclass(frozen=True)
class NoiseSchedule:
    channel: str
    scale: float
    values: tuple[float, ...]
    seed: int
    scale_path: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.channel not in {"gaussian", "uniform", "correlated", "impulsive"}:
            raise ValueError(f"unknown channel {self.channel!r}")
        if not math.isfinite(self.scale) or self.scale < 0:
            raise ValueError("schedule scale must be finite and non-negative")
        if not self.values:
            raise ValueError("schedule must contain at least one timestamp")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("schedule values must be finite")
        if self.scale_path is not None and len(self.scale_path) != len(self.values):
            raise ValueError("scale_path must cover every timestamp")

    def digest(self) -> str:
        payload = {
            "channel": self.channel,
            "scale": self.scale,
            "seed": self.seed,
            "scale_path": self.scale_path,
            "values": self.values,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stationary_unit_process(channel: str, count: int, seed: int) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, *_label_words(channel)]))
    if channel == "gaussian":
        return rng.standard_normal(count)
    if channel == "uniform":
        return rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), count)
    if channel == "correlated":
        values = np.empty(count, dtype=float)
        values[0] = rng.standard_normal()
        for index in range(1, count):
            values[index] = 0.8 * values[index - 1] + 0.6 * rng.standard_normal()
        return values
    if channel == "impulsive":
        z = rng.standard_normal(count)
        hit = rng.binomial(1, 0.01, count)
        sign = rng.choice(np.asarray([-1.0, 1.0]), count)
        return (z + 10.0 * hit * sign) / math.sqrt(2.0)
    raise ValueError(f"unknown channel {channel!r}")


def generate_schedule(
    channel: str,
    scale: float,
    count: int,
    *,
    seed: int,
    scale_path: tuple[float, ...] | None = None,
) -> NoiseSchedule:
    """Generate and return the actual timestamp schedule before execution."""

    if scale_path is not None:
        if len(scale_path) != count:
            raise ValueError("scale_path must have one value per timestamp")
        if any(not math.isfinite(item) or item < 0 for item in scale_path):
            raise ValueError("scale_path values must be finite and non-negative")
        unit = _stationary_unit_process(channel, count, seed)
        values = unit * np.asarray(scale_path, dtype=float)
    elif scale == 0:
        values = np.zeros(count, dtype=float)
    else:
        values = _stationary_unit_process(channel, count, seed) * scale
    return NoiseSchedule(
        channel=channel,
        scale=float(scale),
        values=tuple(float(item) for item in values),
        seed=int(seed),
        scale_path=scale_path,
    )


def generator_validation(seed: int) -> dict[str, Any]:
    """Check declared distribution moments on a separate non-benchmark sample."""

    count = 20_000
    scale = 0.002
    results: dict[str, Any] = {"sample_count": count, "scale": scale, "channels": {}}
    for channel in ("gaussian", "uniform", "correlated", "impulsive"):
        values = _stationary_unit_process(channel, count, seed)
        results["channels"][channel] = {
            "mean": float(np.mean(values)),
            "variance": float(np.var(values)),
            "finite": bool(np.all(np.isfinite(values))),
            "declared_unit_variance_tolerance": 0.08,
        }
    return results


class NoisyObservationEnvironment:
    """Wrap an unchanged latent environment with a cached additive sensor."""

    def __init__(self, latent: Any, schedule: NoiseSchedule) -> None:
        expected = latent.config.steps_per_episode + 1
        if len(schedule.values) != expected:
            raise ValueError(f"schedule has {len(schedule.values)} timestamps; expected {expected}")
        self.latent = latent
        self.schedule = schedule
        self.scenario = latent.scenario
        self.seed = latent.seed
        self.config = latent.config
        self._cursor = -1
        self._observed = math.nan

    @property
    def position(self) -> float:
        """Latent evaluator truth; the prediction runner never passes this value."""

        return float(self.latent.position)

    @property
    def velocity(self) -> float:
        return float(self.latent.velocity)

    @property
    def change_step(self) -> int | None:
        return self.latent.change_step

    def reset(self) -> float:
        latent_position = float(self.latent.reset())
        self._cursor = 0
        self._observed = latent_position + self.config.width * self.schedule.values[0]
        return float(self._observed)

    def observe(self) -> float:
        if self._cursor < 0:
            raise RuntimeError("environment must be reset before observe")
        return float(self._observed)

    def advance(self):
        if self._cursor < 0:
            raise RuntimeError("environment must be reset before advance")
        transition = self.latent.advance()
        self._cursor += 1
        if self._cursor >= len(self.schedule.values):
            raise RuntimeError("sensor schedule exhausted")
        self._observed = transition.position + self.config.width * self.schedule.values[self._cursor]
        return transition
