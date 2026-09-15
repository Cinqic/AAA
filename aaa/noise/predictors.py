"""Predictors used by the observation-noise phase."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..predictors import (
    ConstantMotionPredictor,
    OnlineRLSPredictor,
    PersistencePredictor,
    Predictor,
    ReflectedConstantMotionPredictor,
    reflect_prediction,
)


def _finite_history(history: Sequence[float], minimum: int = 1) -> None:
    if len(history) < minimum:
        raise ValueError(f"predictor requires at least {minimum} observations")
    if any(not math.isfinite(float(item)) for item in history[-minimum:]):
        raise ValueError("predictor history must be finite")


class CausalSmoothingMotionPredictor(Predictor):
    """Two-window causal smoothing followed by public-boundary reflection."""

    name = "causal_smoothing_motion"

    def __init__(self, *, lower_bound: float, upper_bound: float, window: int = 2) -> None:
        if window < 1 or upper_bound <= lower_bound:
            raise ValueError("invalid smoothing baseline configuration")
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.window = int(window)

    def predict(self, history: Sequence[float]) -> float:
        _finite_history(history, self.window * 2)
        values = np.asarray(history[-self.window * 2 :], dtype=float)
        previous = float(np.mean(values[: self.window]))
        current = float(np.mean(values[self.window :]))
        return reflect_prediction(current + (current - previous), self.lower_bound, self.upper_bound)


class AlphaBetaFilter(Predictor):
    """Fixed-parameter causal state estimator with no parameter learning."""

    name = "alpha_beta_filter"
    update_enabled = True

    def __init__(
        self,
        *,
        lower_bound: float,
        upper_bound: float,
        dt: float,
        alpha: float = 0.72,
        beta: float = 0.18,
        position: float | None = None,
        velocity: float = 0.0,
    ) -> None:
        if not 0 < alpha <= 1 or not 0 < beta <= 1 or dt <= 0 or upper_bound <= lower_bound:
            raise ValueError("invalid alpha-beta configuration")
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.position = position
        self.velocity = float(velocity)
        self.update_enabled = True

    def predict(self, history: Sequence[float]) -> float:
        _finite_history(history, 2)
        if self.position is None:
            self.position = float(history[-1])
            self.velocity = float(history[-1] - history[-2]) / self.dt
        predicted = self.position + self.velocity * self.dt
        return reflect_prediction(predicted, self.lower_bound, self.upper_bound)

    def update(self, history: Sequence[float], target_position: float) -> None:
        if not self.update_enabled:
            return
        _finite_history(history, 1)
        if not math.isfinite(target_position):
            raise ValueError("target must be finite")
        if self.position is None:
            self.position = float(history[-1])
        predicted = self.position + self.velocity * self.dt
        residual = float(target_position) - predicted
        self.position = predicted + self.alpha * residual
        self.velocity += self.beta * residual / self.dt
        if not math.isfinite(self.position) or not math.isfinite(self.velocity):
            raise FloatingPointError("alpha-beta state became non-finite")

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": "aaa.alpha_beta_filter.v1",
            "name": self.name,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "dt": self.dt,
            "alpha": self.alpha,
            "beta": self.beta,
            "position": self.position,
            "velocity": self.velocity,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], *, name: str | None = None) -> AlphaBetaFilter:
        if state.get("format_version") != "aaa.alpha_beta_filter.v1":
            raise ValueError("unsupported alpha-beta checkpoint")
        model = cls(
            lower_bound=float(state["lower_bound"]),
            upper_bound=float(state["upper_bound"]),
            dt=float(state["dt"]),
            alpha=float(state["alpha"]),
            beta=float(state["beta"]),
            position=None if state.get("position") is None else float(state["position"]),
            velocity=float(state["velocity"]),
        )

        model.name = name or str(state.get("name", cls.name))
        return model


def incumbent_from_v21(*, name: str, update_enabled: bool) -> OnlineRLSPredictor:
    """Construct the incumbent from the unchanged v2.1 candidate identity."""

    from ..benchmark.families import make_candidate
    from ..benchmark.spec import load_spec

    return make_candidate(load_spec(), name=name, update_enabled=update_enabled)


def make_predictors(
    *, lower_bound: float, upper_bound: float, dt: float, update_enabled: bool
) -> list[Predictor]:
    """Return the fixed comparator order used by every noise cell."""

    return [
        PersistencePredictor(),
        ConstantMotionPredictor(),
        ReflectedConstantMotionPredictor(lower_bound=lower_bound, upper_bound=upper_bound),
        CausalSmoothingMotionPredictor(lower_bound=lower_bound, upper_bound=upper_bound),
        AlphaBetaFilter(lower_bound=lower_bound, upper_bound=upper_bound, dt=dt),
        incumbent_from_v21(name="incumbent_square_root_rls", update_enabled=update_enabled),
        incumbent_from_v21(name="no_learning_control", update_enabled=False),
    ]


def clone_predictor(predictor: Predictor, *, update_enabled: bool) -> Predictor:
    from .candidates import InnovationClipRLSPredictor

    if isinstance(predictor, InnovationClipRLSPredictor):
        return InnovationClipRLSPredictor.from_state_dict(
            predictor.state_dict(), name=predictor.name, update_enabled=update_enabled
        )
    if isinstance(predictor, OnlineRLSPredictor):
        return OnlineRLSPredictor.from_state_dict(
            predictor.state_dict(), name=predictor.name, update_enabled=update_enabled
        )
    if isinstance(predictor, AlphaBetaFilter):
        clone = AlphaBetaFilter.from_state_dict(predictor.state_dict(), name=predictor.name)
        clone.update_enabled = update_enabled
        return clone
    if isinstance(predictor, CausalSmoothingMotionPredictor):
        return CausalSmoothingMotionPredictor(
            lower_bound=predictor.lower_bound,
            upper_bound=predictor.upper_bound,
            window=predictor.window,
        )
    if isinstance(predictor, ReflectedConstantMotionPredictor):
        return ReflectedConstantMotionPredictor(
            lower_bound=predictor.lower_bound,
            upper_bound=predictor.upper_bound,
            name=predictor.name,
        )
    if isinstance(predictor, ConstantMotionPredictor):
        return ConstantMotionPredictor(name=predictor.name)
    if isinstance(predictor, PersistencePredictor):
        return PersistencePredictor()
    raise TypeError(f"unsupported predictor type {type(predictor).__name__}")


def state_fingerprint(predictor: Predictor) -> str:
    if hasattr(predictor, "state_dict"):
        import json

        state = dict(predictor.state_dict())
        state.pop("name", None)
        return json.dumps(state, sort_keys=True, separators=(",", ":"))
    return predictor.name
