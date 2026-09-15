"""Fresh non-confirmatory diagnostic for the observation-noise EIV structure.

This script consumes no confirmation identity, reserves no batch, writes
nothing into ``benchmarks/``, ``results/`` or ``docs/evidence/``, and prints a
JSON object on stdout. It exists to let a reviewer check the claims in
``existing_aaa_behavior.md`` without trusting this branch's prose.

Run from the repository root:

    python research/aaa134/analysis/eiv_diagnostic.py
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from aaa.benchmark.families import make_candidate
from aaa.benchmark.spec import load_spec
from aaa.config import WorldConfig

PROTOTYPE_LABEL = "DEVELOPMENT_DIAGNOSTIC_NOT_CONFIRMATION"
SCALES = (0.0, 0.0005, 0.002, 0.01)
STEPS = 4000
SPEEDS = (0.12, 0.2, 0.32)


def _theory(config: WorldConfig, displacement_scale: float) -> dict[str, float]:
    """Closed-form limits for the single-speed constant-velocity regression.

    The learner regresses ``z_t = (y_{t+1} - y_t) / L`` on
    ``phi_t = (y_t - y_{t-1}) / D``. With ``y_t = x_t + L * eta_t`` and white
    ``eta``, the same ``eta_t`` enters ``phi_t`` with ``+L/D`` and ``z_t`` with
    ``-1``. So the regressor error and the target error are negatively
    correlated: this is not only classical attenuation.
    """

    width = config.width
    ratio = width / displacement_scale
    return {
        "true_displacement_weight": displacement_scale / width,
        "noise_only_asymptote": -displacement_scale / (2.0 * width),
        "regressor_noise_variance_per_unit_sigma_squared": 2.0 * ratio * ratio,
        "regressor_target_covariance_per_unit_sigma_squared": -ratio,
    }


def _least_squares_weight(scale: float, speed: float, seed: int, config: WorldConfig) -> float:
    """Batch OLS on the learner's own (feature, target) pair, noise only."""

    rng = np.random.default_rng(seed)
    width = config.width
    displacement_scale = config.dt * float(load_spec().candidate.displacement_scale_speed)
    true_positions = np.arange(STEPS + 2, dtype=float) * speed * config.dt
    noise = rng.standard_normal(STEPS + 2) * scale
    observed = true_positions + width * noise
    phi = (observed[1:-1] - observed[:-2]) / displacement_scale
    target = (observed[2:] - observed[1:-1]) / width
    design = np.column_stack([np.ones_like(phi), phi])
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(solution[1])


def _online_weights(scale: float, seed: int, config: WorldConfig) -> dict[str, Any]:
    """Drive the real incumbent through a noisy latent path that never bounces.

    A sinusoid well inside the interval gives the displacement feature genuine
    variation while keeping every wall policy (reflection, unfolding, straddle
    skip) inactive, so the measured attenuation is the EIV term alone.
    """

    spec = load_spec()
    predictor = make_candidate(spec, name="incumbent_square_root_rls", update_enabled=True)
    rng = np.random.default_rng(seed)
    width = config.width
    midpoint = (config.lower_bound + config.upper_bound) / 2.0
    index_grid = np.arange(STEPS + 1, dtype=float)
    latent = midpoint + 0.30 * width * np.sin(2.0 * math.pi * index_grid / 200.0)
    observed = latent + width * rng.standard_normal(STEPS + 1) * scale
    history = [float(value) for value in observed[:4]]
    for index in range(4, STEPS + 1):
        predictor.predict(tuple(history[-4:]))
        predictor.update(tuple(history[-4:]), float(observed[index]))
        history.append(float(observed[index]))
    weights = [float(value) for value in predictor.weights]
    return {
        "intercept_weight": weights[0],
        "displacement_weight": weights[1],
        "centered_position_weight": weights[2],
        "reflection_skips": int(predictor.reflection_skips),
    }


def main() -> dict[str, Any]:
    config = WorldConfig()
    spec = load_spec()
    displacement_scale = config.dt * float(spec.candidate.displacement_scale_speed)
    theory = _theory(config, displacement_scale)
    batch: list[dict[str, Any]] = []
    for scale in SCALES:
        for speed in SPEEDS:
            estimates = [_least_squares_weight(scale, speed, 9_000 + index, config) for index in range(5)]
            batch.append(
                {
                    "scale": scale,
                    "speed": speed,
                    "mean_estimated_displacement_weight": float(np.mean(estimates)),
                    "min": float(np.min(estimates)),
                    "max": float(np.max(estimates)),
                }
            )
    online = [{"scale": scale, **_online_weights(scale, 4_100, config)} for scale in SCALES]
    causal_contract = {
        "predict_signature": "predict(history) -> float",
        "update_signature": "update(history, target_position) -> None",
        "latent_truth_reaches_predictor": False,
        "checked_against": "aaa/noise/runner.py run_stationary_segment / run_noisy_segment",
    }
    return {
        "schema_version": "aaa.research.aaa134.eiv_diagnostic.v1",
        "label": PROTOTYPE_LABEL,
        "interval_width": config.width,
        "dt": config.dt,
        "displacement_scale": displacement_scale,
        "theory": theory,
        "batch_least_squares": batch,
        "online_incumbent_weights": online,
        "causal_contract": causal_contract,
        "finite": all(math.isfinite(row["mean_estimated_displacement_weight"]) for row in batch),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
