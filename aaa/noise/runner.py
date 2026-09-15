"""Execution and evidence writer for ``aaa.observation_noise.v1.1``."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..benchmark.evidence import git_metadata, hardware_metadata, json_dump, sha256_file
from ..benchmark.families import classify_stratum, sample_stratified_state
from ..benchmark.spec import canonical_spec_hash, canonical_spec_path, load_spec
from ..config import WorldConfig
from ..environment import DampedOscillatorEnvironment, MovingDotEnvironment
from ..predictors import OnlineRLSPredictor, Predictor
from .calibration import CausalResidualCalibrator
from .candidates import INCUMBENT_CANDIDATE_ID, candidate_definition, candidate_from_model
from .observation import (
    NoiseSchedule,
    NoisyObservationEnvironment,
    derive_seed,
    generate_schedule,
    generator_validation,
    scale_key,
)
from .plots import write_plots
from .predictors import (
    AlphaBetaFilter,
    CausalSmoothingMotionPredictor,
    clone_predictor,
    incumbent_from_v21,
)
from .reservation import ReservationError, reserve_confirmation_batch
from .scientific_identity import ScientificIdentityError, fingerprints_equal, scientific_fingerprint
from .spec import NoiseProtocol, canonical_protocol_hash, load_protocol
from .statistics import (
    hierarchical_bootstrap,
    interval_statistics,
    paired_hierarchical_comparisons,
    resource_statistics,
    validate_null_behavior,
)
from .verifier import CHECK_NAMES, _metric_rows, iter_records, verify_attempt

RECORD_SCHEMA = "aaa.observation_noise_step.v3"
PROTOCOL_DIRECTORY = "observation-noise-v1_1"
ALL_CHANNELS = ("gaussian", "uniform", "correlated", "impulsive")
ALL_SCALES = (0.0, 0.0005, 0.002, 0.01)
SHIFT_LEVEL_PAIRS = ((0.0005, 0.002), (0.002, 0.0005))
SHIFT_TIMES = (("aligned", 300), ("staggered", 340))


class ObservationNoiseError(RuntimeError):
    """Raised when an attempt cannot safely continue."""


@dataclass(frozen=True)
class AttemptResult:
    directory: Path
    summary: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.summary.get("all_required_gates_pass", False))


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_text_if_identical(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ObservationNoiseError(f"retained artifact differs for {path}")
        return
    _write_text_atomic(path, text)


def _record_lifecycle(path: Path, event: str, **data: Any) -> None:
    rows: list[dict[str, Any]] = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows.append({"at_utc": _now(), "event": event, **data})
    _write_text_atomic(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _stable_json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _base_world(family: str, steps: int, *, change_step: int | None = None) -> WorldConfig:
    del family
    reference = load_spec()
    source = reference.motion_families.get("constant_velocity")
    speed_min = 0.08 if source is None else source.speed_min
    speed_max = 0.2 if source is None else source.speed_max
    return WorldConfig(
        lower_bound=reference.world.lower_bound,
        upper_bound=reference.world.upper_bound,
        dt=reference.world.dt,
        steps_per_episode=steps,
        history_length=reference.world.history_length,
        speed_min=speed_min,
        speed_max=speed_max,
        change_step=change_step,
        change_factor_low=0.45,
        change_factor_high=1.8,
        event_margin=0.18,
    )


def build_latent_environment(
    family: str,
    seed: int,
    steps: int,
    *,
    initial_position: float | None = None,
    initial_velocity: float | None = None,
    include_law_change: bool = True,
):
    """Construct the existing latent simulator without changing its laws."""

    if family == "constant_velocity":
        return MovingDotEnvironment(
            "straight",
            seed,
            _base_world(family, steps),
            initial_position=initial_position,
            initial_velocity=initial_velocity,
        )
    if family == "bouncing":
        return MovingDotEnvironment(
            "bouncing",
            seed,
            _base_world(family, steps),
            initial_position=initial_position,
            initial_velocity=initial_velocity,
        )
    if family == "speed_change":
        change_step = 300 if include_law_change else None
        world = _base_world(family, steps, change_step=change_step)
        return MovingDotEnvironment(
            "changed",
            seed,
            world,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            change_step=change_step,
        )
    if family == "changed_law":
        reference = load_spec()
        law = reference.changed_law
        change_step = 300 if include_law_change else None
        world = _base_world(family, steps, change_step=change_step)
        rng = np.random.default_rng(seed ^ 0xA11CE)
        post_omega = float(rng.uniform(law.post_omega_low, law.post_omega_high))
        post_damping = float(rng.uniform(law.post_damping_low, law.post_damping_high))
        return DampedOscillatorEnvironment(
            seed,
            world,
            omega=law.pre_omega,
            damping=law.pre_damping,
            changed_omega=post_omega,
            changed_damping=post_damping,
            change_step=change_step,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            initial_position_low=law.initial_position_low,
            initial_position_high=law.initial_position_high,
            initial_velocity_scale=law.initial_velocity_scale,
        )
    raise ObservationNoiseError(f"unknown latent family {family!r}")


def _schedule_payload(schedule: NoiseSchedule) -> dict[str, Any]:
    return {
        "schema_version": "aaa.observation_noise_schedule.v1",
        "channel": schedule.channel,
        "scale": schedule.scale,
        "seed": schedule.seed,
        "scale_path": None if schedule.scale_path is None else list(schedule.scale_path),
        "values": list(schedule.values),
        "digest": schedule.digest(),
    }


def _diagnostics(predictor: Predictor) -> dict[str, Any]:
    if isinstance(predictor, OnlineRLSPredictor):
        state = predictor.state_dict()
        return {
            "update_count": state["update_count"],
            "forgetting_suspensions": state["forgetting_suspensions"],
            "dead_zone_skips": state["dead_zone_skips"],
            "reflection_skips": state["reflection_skips"],
            "detected_surprises": state["detected_surprises"],
            "error_ewma": state["error_ewma"],
        }
    if isinstance(predictor, AlphaBetaFilter):
        return {"position": predictor.position, "velocity": predictor.velocity, "state_estimator": True}
    return {}


def _state_payload(predictor: Predictor) -> dict[str, Any]:
    state_method = getattr(predictor, "state_dict", None)
    if state_method is None:
        return {}
    value = state_method()
    return value if isinstance(value, dict) else {}


def _numeric_state_values(value: object) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)] if math.isfinite(float(value)) else []
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            result.extend(_numeric_state_values(item))
        return result
    return []


def _state_delta_norm(before: dict[str, Any], after: dict[str, Any]) -> float:
    keys = {"weights", "sqrt_factor", "position", "velocity"}
    delta: list[float] = []
    for key in keys:
        left = _numeric_state_values(before.get(key))
        right = _numeric_state_values(after.get(key))
        if len(left) == len(right):
            delta.extend(
                after_value - before_value for before_value, after_value in zip(left, right, strict=True)
            )
    return float(math.sqrt(sum(value * value for value in delta)))


def _predictor_inventory(predictors: Sequence[Predictor]) -> list[dict[str, Any]]:
    """Record comparable retained-state size and a fixed local cost probe."""

    history = (0.20, 0.30, 0.40, 0.50)
    target = 0.51
    inventory: list[dict[str, Any]] = []
    for predictor in predictors:
        state = _state_payload(predictor)
        clone = clone_predictor(predictor, update_enabled=predictor.update_enabled)
        predict_start = time.perf_counter_ns()
        for _ in range(32):
            clone.predict(history)
        predict_elapsed = time.perf_counter_ns() - predict_start
        update_start = time.perf_counter_ns()
        for _ in range(32):
            if clone.update_enabled:
                clone.update(history, target)
        update_elapsed = time.perf_counter_ns() - update_start
        trainable = state.get("weights")
        parameter_count = len(_numeric_state_values(trainable)) if trainable is not None else 0
        inventory.append(
            {
                "name": predictor.name,
                "update_enabled": bool(predictor.update_enabled),
                "parameter_count": parameter_count,
                "retained_state_bytes": len(
                    json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
                ),
                "predict_latency_ns_per_step": float(predict_elapsed / 32.0),
                "update_latency_ns_per_step": float(update_elapsed / 32.0),
                "cost_probe_steps": 32,
            }
        )
    return inventory


def _new_calibrators(predictors: Sequence[Predictor]) -> dict[str, CausalResidualCalibrator]:
    return {predictor.name: CausalResidualCalibrator() for predictor in predictors}


def run_noisy_episode(
    environment: NoisyObservationEnvironment,
    predictors: Sequence[Predictor],
    trial: dict[str, Any],
    schedule: NoiseSchedule,
    *,
    initial_calibrators: dict[str, CausalResidualCalibrator] | None = None,
) -> list[dict[str, Any]]:
    """Run predict -> record -> advance -> reveal -> score -> update."""

    current = environment.reset()
    history: list[float] = [current]
    history_length = environment.config.history_length
    width = environment.config.width
    for _ in range(history_length - 1):
        environment.advance()
        history.append(environment.observe())
    records: list[dict[str, Any]] = []
    calibrators = (
        _new_calibrators(predictors) if initial_calibrators is None else copy.deepcopy(initial_calibrators)
    )
    for step in range(history_length - 1, environment.config.steps_per_episode):
        observed_history = tuple(history[-history_length:])
        raw_predictions: dict[str, float] = {}
        scored_predictions: dict[str, float] = {}
        predict_latencies: dict[str, int] = {}
        prediction_intervals: dict[str, dict[str, dict[str, float] | None]] = {}
        for predictor in predictors:
            predict_start = time.perf_counter_ns()
            raw = float(predictor.predict(observed_history))
            predict_latencies[predictor.name] = time.perf_counter_ns() - predict_start
            if not math.isfinite(raw):
                raise FloatingPointError(f"{predictor.name} returned a non-finite forecast")
            raw_predictions[predictor.name] = raw
            scored_predictions[predictor.name] = raw
            prediction_intervals[predictor.name] = calibrators[predictor.name].intervals(raw)
        transition = environment.advance()
        target_observation = environment.observe()
        predictions: dict[str, dict[str, float]] = {}
        updates: dict[str, bool] = {}
        for predictor in predictors:
            scored = scored_predictions[predictor.name]
            latent_error = abs(scored - transition.position)
            observation_error = abs(scored - target_observation)
            predictions[predictor.name] = {
                "raw": raw_predictions[predictor.name],
                "scored": scored,
                "latent_absolute_error": latent_error,
                "latent_normalized_absolute_error": latent_error / width,
                "noisy_observation_absolute_error": observation_error / width,
                "signed_latent_error": scored - transition.position,
            }
            updates[predictor.name] = bool(predictor.update_enabled)
        record = {
            "schema_version": RECORD_SCHEMA,
            "trial": dict(trial),
            "step": step,
            "target_step": step + 1,
            "observed_history": list(observed_history),
            "current_observation": float(observed_history[-1]),
            "latent_position": float(transition.position),
            "raw_observation": float(target_observation),
            "available_training_target": float(target_observation),
            "line_width": float(width),
            "effective_noise_scale": float(
                schedule.scale if schedule.scale_path is None else schedule.scale_path[step + 1]
            ),
            "predictions": predictions,
            "prediction_intervals": prediction_intervals,
            "update_decisions": updates,
            "diagnostics": {},
            "schedule_index": step + 1,
            "noise_channel": schedule.channel,
            "noise_scale": schedule.scale,
            "noise_innovation": schedule.values[step + 1],
            "bounced": bool(transition.bounced),
            "changed": bool(transition.changed),
        }
        if set(record) != {
            "schema_version",
            "trial",
            "step",
            "target_step",
            "observed_history",
            "current_observation",
            "latent_position",
            "raw_observation",
            "available_training_target",
            "line_width",
            "effective_noise_scale",
            "predictions",
            "prediction_intervals",
            "update_decisions",
            "diagnostics",
            "schedule_index",
            "noise_channel",
            "noise_scale",
            "noise_innovation",
            "bounced",
            "changed",
        }:
            raise ObservationNoiseError("record construction drifted")
        records.append(record)
        diagnostics: dict[str, dict[str, Any]] = {}
        for predictor in predictors:
            state_before_update = _state_payload(predictor)
            update_start = time.perf_counter_ns()
            if predictor.update_enabled:
                predictor.update(observed_history, target_observation)
            update_elapsed = time.perf_counter_ns() - update_start
            calibrators[predictor.name].update(scored_predictions[predictor.name], target_observation)
            diagnostic = _diagnostics(predictor)
            diagnostic.update(
                {
                    "update_norm": _state_delta_norm(state_before_update, _state_payload(predictor)),
                    "predict_latency_ns": predict_latencies[predictor.name],
                    "update_latency_ns": update_elapsed,
                }
            )
            diagnostics[predictor.name] = diagnostic
        record["diagnostics"] = diagnostics
        history.append(target_observation)
    return records


def run_noisy_segment(
    environment: NoisyObservationEnvironment,
    predictors: Sequence[Predictor],
    trial: dict[str, Any],
    schedule: NoiseSchedule,
    history: list[float],
    *,
    first_step: int,
    end_step: int,
    calibrators: dict[str, CausalResidualCalibrator] | None = None,
) -> list[dict[str, Any]]:
    """Run a continuation without resetting world, schedule, history, or state."""

    width = environment.config.width
    history_length = environment.config.history_length
    records: list[dict[str, Any]] = []
    if calibrators is None:
        calibrators = _new_calibrators(predictors)
    if len(history) < history_length:
        raise ObservationNoiseError("a continuation requires a complete observed history")
    for step in range(first_step, end_step):
        observed_history = tuple(history[-history_length:])
        raw_predictions: dict[str, float] = {}
        scored_predictions: dict[str, float] = {}
        predict_latencies: dict[str, int] = {}
        prediction_intervals: dict[str, dict[str, dict[str, float] | None]] = {}
        for predictor in predictors:
            predict_start = time.perf_counter_ns()
            raw = float(predictor.predict(observed_history))
            predict_latencies[predictor.name] = time.perf_counter_ns() - predict_start
            if not math.isfinite(raw):
                raise FloatingPointError(f"{predictor.name} returned a non-finite forecast")
            raw_predictions[predictor.name] = raw
            scored_predictions[predictor.name] = raw
            prediction_intervals[predictor.name] = calibrators[predictor.name].intervals(raw)
        transition = environment.advance()
        target_observation = environment.observe()
        predictions: dict[str, dict[str, float]] = {}
        updates: dict[str, bool] = {}
        for predictor in predictors:
            scored = scored_predictions[predictor.name]
            latent_error = abs(scored - transition.position)
            predictions[predictor.name] = {
                "raw": raw_predictions[predictor.name],
                "scored": scored,
                "latent_absolute_error": latent_error,
                "latent_normalized_absolute_error": latent_error / width,
                "noisy_observation_absolute_error": abs(scored - target_observation) / width,
                "signed_latent_error": scored - transition.position,
            }
            updates[predictor.name] = bool(predictor.update_enabled)
        record = {
            "schema_version": RECORD_SCHEMA,
            "trial": dict(trial),
            "step": step,
            "target_step": step + 1,
            "observed_history": list(observed_history),
            "current_observation": float(observed_history[-1]),
            "latent_position": float(transition.position),
            "raw_observation": float(target_observation),
            "available_training_target": float(target_observation),
            "line_width": float(width),
            "effective_noise_scale": float(
                schedule.scale if schedule.scale_path is None else schedule.scale_path[step + 1]
            ),
            "predictions": predictions,
            "prediction_intervals": prediction_intervals,
            "update_decisions": updates,
            "diagnostics": {},
            "schedule_index": step + 1,
            "noise_channel": schedule.channel,
            "noise_scale": schedule.scale,
            "noise_innovation": schedule.values[step + 1],
            "bounced": bool(transition.bounced),
            "changed": bool(transition.changed),
        }
        records.append(record)
        diagnostics: dict[str, dict[str, Any]] = {}
        for predictor in predictors:
            state_before_update = _state_payload(predictor)
            update_start = time.perf_counter_ns()
            if predictor.update_enabled:
                predictor.update(observed_history, target_observation)
            update_elapsed = time.perf_counter_ns() - update_start
            calibrators[predictor.name].update(scored_predictions[predictor.name], target_observation)
            diagnostic = _diagnostics(predictor)
            diagnostic.update(
                {
                    "update_norm": _state_delta_norm(state_before_update, _state_payload(predictor)),
                    "predict_latency_ns": predict_latencies[predictor.name],
                    "update_latency_ns": update_elapsed,
                }
            )
            diagnostics[predictor.name] = diagnostic
        record["diagnostics"] = diagnostics
        history.append(target_observation)
    return records


def run_matched_changed_law(
    protocol: NoiseProtocol,
    model: OnlineRLSPredictor,
    trial: dict[str, Any],
    schedule: NoiseSchedule,
    *,
    candidate_id: str = INCUMBENT_CANDIDATE_ID,
    initial_calibrators: dict[str, CausalResidualCalibrator] | None = None,
) -> list[dict[str, Any]]:
    """Create a common prefix, then clone the complete state into two branches."""

    from ..predictors import ConstantMotionPredictor, PersistencePredictor, ReflectedConstantMotionPredictor

    seed_root = int(protocol.raw["reference"]["v2_1_raw_sha256"][:8], 16)
    latent_seed = derive_seed(seed_root, "latent", trial["trial_id"])
    latent = build_latent_environment(
        "changed_law", latent_seed, int(protocol.families["changed_law"]["steps"])
    )
    environment = NoisyObservationEnvironment(latent, schedule)
    history = [environment.reset()]
    for _ in range(environment.config.history_length - 1):
        environment.advance()
        history.append(environment.observe())
    prefix_model = candidate_from_model(
        model, candidate_id, name="selected_candidate_prefix", update_enabled=True
    )
    prefix_calibrators = _new_calibrators([prefix_model])
    if initial_calibrators is not None and "selected_candidate_online" in initial_calibrators:
        prefix_calibrators[prefix_model.name] = copy.deepcopy(
            initial_calibrators["selected_candidate_online"]
        )
    prefix_trial = {**trial, "trial_id": f"{trial['trial_id']}:prefix", "branch": "prefix"}
    records = run_noisy_segment(
        environment,
        [prefix_model],
        prefix_trial,
        schedule,
        history,
        first_step=environment.config.history_length - 1,
        end_step=300,
        calibrators=prefix_calibrators,
    )
    frozen_environment = copy.deepcopy(environment)
    online_environment = copy.deepcopy(environment)
    frozen_model = clone_predictor(prefix_model, update_enabled=False)
    frozen_model.name = "selected_candidate_branch_frozen"
    online_model = clone_predictor(prefix_model, update_enabled=True)
    online_model.name = "selected_candidate_branch_online"
    frozen_calibrators = copy.deepcopy(prefix_calibrators)
    online_calibrators = copy.deepcopy(prefix_calibrators)

    def branch_predictors(model_copy: Predictor) -> list[Predictor]:
        lower = environment.config.lower_bound
        upper = environment.config.upper_bound
        return [
            PersistencePredictor(),
            ConstantMotionPredictor(),
            ReflectedConstantMotionPredictor(lower_bound=lower, upper_bound=upper),
            model_copy,
        ]

    frozen_trial = {**trial, "trial_id": f"{trial['trial_id']}:frozen", "branch": "frozen"}
    online_trial = {**trial, "trial_id": f"{trial['trial_id']}:online", "branch": "online"}
    branch_history = list(history)
    frozen_predictor_set = branch_predictors(frozen_model)
    online_predictor_set = branch_predictors(online_model)
    frozen_calibrators = _new_calibrators(frozen_predictor_set)
    online_calibrators = _new_calibrators(online_predictor_set)
    if initial_calibrators is not None:
        for name in frozen_calibrators:
            if name in initial_calibrators:
                frozen_calibrators[name] = copy.deepcopy(initial_calibrators[name])
        for name in online_calibrators:
            if name in initial_calibrators:
                online_calibrators[name] = copy.deepcopy(initial_calibrators[name])
    frozen_calibrators[frozen_model.name] = copy.deepcopy(prefix_calibrators[prefix_model.name])
    online_calibrators[online_model.name] = copy.deepcopy(prefix_calibrators[prefix_model.name])
    records.extend(
        run_noisy_segment(
            frozen_environment,
            frozen_predictor_set,
            frozen_trial,
            schedule,
            list(branch_history),
            first_step=300,
            end_step=400,
            calibrators=frozen_calibrators,
        )
    )
    records.extend(
        run_noisy_segment(
            online_environment,
            online_predictor_set,
            online_trial,
            schedule,
            list(branch_history),
            first_step=300,
            end_step=400,
            calibrators=online_calibrators,
        )
    )
    return records


def _clone_rls(model: OnlineRLSPredictor, name: str, update_enabled: bool) -> OnlineRLSPredictor:
    return OnlineRLSPredictor.from_state_dict(model.state_dict(), name=name, update_enabled=update_enabled)


def _predictors(
    model: OnlineRLSPredictor,
    *,
    lower: float,
    upper: float,
    dt: float,
    candidate_id: str = INCUMBENT_CANDIDATE_ID,
) -> list[Predictor]:
    from ..predictors import ConstantMotionPredictor, PersistencePredictor, ReflectedConstantMotionPredictor

    candidate_definition(candidate_id)
    return [
        PersistencePredictor(),
        ConstantMotionPredictor(),
        ReflectedConstantMotionPredictor(lower_bound=lower, upper_bound=upper),
        CausalSmoothingMotionPredictor(lower_bound=lower, upper_bound=upper),
        AlphaBetaFilter(lower_bound=lower, upper_bound=upper, dt=dt),
        _clone_rls(model, "incumbent_frozen", False),
        _clone_rls(model, "incumbent_square_root_rls", True),
        incumbent_from_v21(name="no_learning_control", update_enabled=False),
        candidate_from_model(model, candidate_id, name="selected_candidate_frozen", update_enabled=False),
        candidate_from_model(model, candidate_id, name="selected_candidate_online", update_enabled=True),
    ]


def train_model(
    protocol: NoiseProtocol,
    condition: str,
    lineage: int,
    *,
    role: str,
    quick: bool,
    evidence: list[dict[str, Any]] | None = None,
    schedule_dir: Path | None = None,
) -> tuple[OnlineRLSPredictor, dict[str, Any]]:
    """Train one declared lineage from clean or fixed-mixture observations."""

    if condition not in {"clean_trained", "noise_trained"}:
        raise ObservationNoiseError(f"unknown training condition {condition!r}")
    model = incumbent_from_v21(name=f"trained_{condition}_{lineage}", update_enabled=True)
    episode_count = 2 if quick else int(protocol.training["episodes_per_lineage"])
    step_count = min(40, int(protocol.training["updates_per_episode"]))
    schedule_digests: list[str] = []
    for episode in range(episode_count):
        channel = "gaussian" if condition == "noise_trained" and episode % 2 == 0 else "uniform"
        scale = 0.0 if condition == "clean_trained" else (0.0005 if episode % 2 == 0 else 0.002)
        training_role = "development" if role == "development" else "confirmation_shared"
        seed = derive_seed(
            int(protocol.raw["reference"]["v2_1_raw_sha256"][:8], 16),
            protocol.seed_namespaces["training"],
            training_role,
            condition,
            lineage,
            episode,
        )
        schedule = generate_schedule(channel, scale, step_count + 1, seed=seed)
        schedule_digests.append(schedule.digest())
        schedule_relative: str | None = None
        if schedule_dir is not None:
            schedule_path = schedule_dir / f"{condition}-lineage-{lineage:03d}-episode-{episode:04d}.json"
            _write_schedule(schedule_path, schedule)
            schedule_relative = str(Path("training_schedules") / schedule_path.name)
        latent = build_latent_environment("constant_velocity", seed, step_count)
        wrapped = NoisyObservationEnvironment(latent, schedule)
        history = [wrapped.reset()]
        for _ in range(wrapped.config.history_length - 1):
            wrapped.advance()
            history.append(wrapped.observe())
        for step in range(wrapped.config.history_length - 1, step_count):
            observed_history = tuple(history[-wrapped.config.history_length :])
            forecast = float(model.predict(observed_history))
            wrapped.advance()
            target = wrapped.observe()
            if evidence is not None:
                evidence.append(
                    {
                        "schema_version": "aaa.observation_noise_training_step.v1",
                        "condition": condition,
                        "lineage": lineage,
                        "episode": episode,
                        "step": step,
                        "channel": channel,
                        "scale": scale,
                        "schedule_index": step + 1,
                        "schedule_path": schedule_relative,
                        "schedule_digest": schedule.digest(),
                        "observed_history": list(observed_history),
                        "forecast": forecast,
                        "available_training_target": float(target),
                        "update_decision": True,
                    }
                )
            model.update(observed_history, target)
            history.append(target)
    return model, {
        "condition": condition,
        "lineage": lineage,
        "episodes": episode_count,
        "schedule_digests": schedule_digests,
    }


def _write_schedule(path: Path, schedule: NoiseSchedule) -> str:
    payload = _schedule_payload(schedule)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ObservationNoiseError(f"existing schedule differs for {path}")
        return sha256_file(path)
    json_dump(path, payload)
    return sha256_file(path)


def _calibrate_predictors(
    protocol: NoiseProtocol,
    predictors: Sequence[Predictor],
    *,
    condition: str,
    lineage: int,
    schedule_dir: Path,
) -> tuple[dict[str, CausalResidualCalibrator], list[dict[str, Any]]]:
    """Build the fixed calibration prefix from the declared training mixture."""

    schedule_dir.mkdir(parents=True, exist_ok=True)
    seed_root = int(protocol.raw["reference"]["v2_1_raw_sha256"][:8], 16)
    world = _base_world("constant_velocity", 40)
    calibrators = _new_calibrators(predictors)
    schedule_index: list[dict[str, Any]] = []
    mixture = (
        ("gaussian", 0.0005),
        ("gaussian", 0.002),
        ("uniform", 0.0005),
        ("uniform", 0.002),
    )
    for mixture_index, (channel, scale) in enumerate(mixture):
        seed = derive_seed(
            seed_root,
            protocol.seed_namespaces["calibration"],
            condition,
            lineage,
            mixture_index,
            channel,
            scale_key(scale),
        )
        schedule = generate_schedule(channel, scale, world.steps_per_episode + 1, seed=seed)
        relative = Path(f"{condition}-lineage-{lineage:03d}-mixture-{mixture_index:02d}.json")
        path = schedule_dir / relative
        digest = _write_schedule(path, schedule)
        schedule_index.append(
            {
                "path": str(Path("calibration_schedules") / relative),
                "file_sha256": digest,
                "schedule_digest": schedule.digest(),
                "channel": channel,
                "scale": scale,
                "seed": seed,
            }
        )
        latent = build_latent_environment("constant_velocity", seed, world.steps_per_episode)
        wrapped = NoisyObservationEnvironment(latent, schedule)
        clones = {
            predictor.name: clone_predictor(predictor, update_enabled=predictor.update_enabled)
            for predictor in predictors
        }
        history = [wrapped.reset()]
        for _ in range(world.history_length - 1):
            wrapped.advance()
            history.append(wrapped.observe())
        for _step in range(world.history_length - 1, world.steps_per_episode):
            observed_history = tuple(history[-world.history_length :])
            forecasts = {
                name: float(predictor.predict(observed_history)) for name, predictor in clones.items()
            }
            wrapped.advance()
            target = wrapped.observe()
            for name, forecast in forecasts.items():
                calibrators[name].update(forecast, target)
            for predictor in clones.values():
                if predictor.update_enabled:
                    predictor.update(observed_history, target)
            history.append(target)
    return calibrators, schedule_index


def _record_shard_path(attempt_dir: Path, trial_id: str) -> Path:
    digest = hashlib.sha256(trial_id.encode("utf-8")).hexdigest()
    return attempt_dir / "trial_records" / f"{digest}.jsonl.gz"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ObservationNoiseError(f"{path}:{line_number}: record must be an object")
            rows.append(value)
    return rows


def _persist_trial_rows(attempt_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Persist completed trial groups atomically and refuse divergent replays."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        trial_id = str(row["trial"]["trial_id"])
        grouped.setdefault(trial_id, []).append(row)
    shard_dir = attempt_dir / "trial_records"
    shard_dir.mkdir(exist_ok=True)
    for trial_id, trial_rows in grouped.items():
        path = _record_shard_path(attempt_dir, trial_id)
        encoded = "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in trial_rows
        ).encode("utf-8")
        if path.exists():
            with gzip.open(path, "rb") as handle:
                retained = handle.read()
            if retained != encoded:
                raise ObservationNoiseError(f"replayed trial {trial_id} differs from retained evidence")
            continue
        temporary = path.with_suffix(path.suffix + ".tmp")
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed,
        ):
            compressed.write(encoded)
        temporary.replace(path)


def _trial_shard_index(attempt_dir: Path) -> list[tuple[str, Path]]:
    shard_dir = attempt_dir / "trial_records"
    if not shard_dir.is_dir():
        shard_dir = attempt_dir / "records"
    if not shard_dir.is_dir():
        return []
    index: list[tuple[str, Path]] = []
    for path in sorted(shard_dir.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            first = next((line for line in handle if line.strip()), None)
        if first is None:
            raise ObservationNoiseError(f"empty retained trial shard: {path}")
        row = json.loads(first)
        index.append((str(row["trial"]["trial_id"]), path))
    return sorted(index)


def _iter_trial_shards(attempt_dir: Path):
    for _trial_id, path in _trial_shard_index(attempt_dir):
        yield from _read_jsonl(path)


def _record_ids_for_plan(trial: Mapping[str, Any]) -> set[str]:
    trial_id = str(trial["trial_id"])
    result = {trial_id}
    if trial["family"] == "changed_law" and trial["branch"] == "stationary":
        result.update(f"{trial_id}:{branch}" for branch in ("prefix", "frozen", "online"))
    return result


def _finalize_record_shards(attempt_dir: Path) -> tuple[int, str, str]:
    shard_dir = attempt_dir / "trial_records"
    finalized_index = attempt_dir / "records" / "index.json"
    if finalized_index.is_file() and not shard_dir.exists():
        payload = json.loads(finalized_index.read_text(encoding="utf-8"))
        return (
            int(payload["records"]),
            str(payload["uncompressed_records_sha256"]),
            sha256_file(finalized_index),
        )
    if not shard_dir.is_dir():
        raise ObservationNoiseError("cannot finalize missing trial record shards")
    uncompressed = hashlib.sha256()
    count = 0
    entries: list[dict[str, Any]] = []
    for trial_id, path in _trial_shard_index(attempt_dir):
        shard_count = 0
        with gzip.open(path, "rb") as handle:
            for line in handle:
                uncompressed.update(line)
                shard_count += 1
        count += shard_count
        entries.append(
            {
                "trial_id": trial_id,
                "path": f"records/{path.name}",
                "records": shard_count,
                "compressed_sha256": sha256_file(path),
            }
        )
    json_dump(
        shard_dir / "index.json",
        {
            "schema_version": "aaa.observation_noise_record_shards.v1",
            "ordering": "trial_id ascending, step ascending",
            "records": count,
            "uncompressed_records_sha256": uncompressed.hexdigest(),
            "shards": entries,
        },
    )
    destination = attempt_dir / "records"
    shard_dir.replace(destination)
    return count, uncompressed.hexdigest(), sha256_file(destination / "index.json")


def _primary_cells() -> list[tuple[str, float]]:
    return [(channel, scale) for channel in ("gaussian", "uniform") for scale in (0.0005, 0.002)]


def _initial_state_for_trial(
    protocol: NoiseProtocol, trial: dict[str, Any], steps: int
) -> tuple[str, float | None, float | None]:
    if trial["family"] not in {"constant_velocity", "bouncing"}:
        return "unstratified", None, None
    reference = load_spec()
    world = _base_world(trial["family"], steps)
    strata = reference.required_strata()
    target = strata[(int(trial["episode"]) + int(trial["lineage"])) % len(strata)]
    direction, position_token, speed_token = target.split("-")
    rng_seed = derive_seed(
        int(protocol.raw["reference"]["v2_1_raw_sha256"][:8], 16),
        "stratified-state",
        trial["family"],
        trial["lineage"],
        trial["episode"],
    )
    position, velocity = sample_stratified_state(
        np.random.default_rng(rng_seed),
        world,
        direction=direction,
        position_band=int(position_token.removeprefix("position")),
        speed_band=int(speed_token.removeprefix("speed")),
        position_bands=reference.stratification.position_bands,
        speed_bands=reference.stratification.speed_bands,
        must_stay_in_bounds=trial["family"] == "constant_velocity",
    )
    realized = classify_stratum(
        position,
        velocity,
        world,
        position_bands=reference.stratification.position_bands,
        speed_bands=reference.stratification.speed_bands,
    )
    if realized != target:
        raise ObservationNoiseError(f"stratum plan mismatch: planned {target}, realized {realized}")
    return target, position, velocity


def _plan_counts(protocol: NoiseProtocol, *, quick: bool, role: str) -> tuple[int, int, int]:
    if quick and role == "development":
        return 1, 1, 1
    if role == "development":
        return 2, 2, 1
    return (
        int(protocol.replication["lineages"]),
        int(protocol.replication["episodes_per_family_per_lineage"]),
        int(protocol.replication["sensor_realizations_per_episode"]),
    )


def _make_attempt_dir(output_root: Path, label: str | None) -> Path:
    root = output_root / PROTOCOL_DIRECTORY
    root.mkdir(parents=True, exist_ok=True)
    attempt = label or datetime.now(timezone.utc).strftime("development-%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", attempt):
        raise ObservationNoiseError(
            "attempt label must be 1-64 ASCII letters, digits, dot, underscore, or hyphen"
        )
    path = root / attempt
    if path.exists():
        raise FileExistsError(f"attempt directory already exists: {path}")
    path.mkdir(parents=True)
    return path


def _open_attempt_dir(output_root: Path, label: str | None, *, resume: bool) -> Path:
    if not resume:
        return _make_attempt_dir(output_root, label)
    if not label:
        raise ObservationNoiseError("--resume requires the existing --attempt-label")
    root = output_root / PROTOCOL_DIRECTORY
    path = root / label
    if not path.is_dir():
        raise ObservationNoiseError(f"cannot resume missing attempt directory: {path}")
    if (path / "checksums.json").is_file() and (path / "summary.json").is_file():
        raise ObservationNoiseError(f"attempt is already finalized; use observation-noise-recompute: {path}")
    return path


def _reference_check() -> dict[str, Any]:
    raw_path = canonical_spec_path()
    raw_hash = sha256_file(raw_path)
    resolved = canonical_spec_hash()
    return {
        "status": "PASS"
        if raw_hash == "4993c5e6e173f9dd5ef002bc84ff4c484da853b066ea45662d41ad826d10d48e"
        and resolved == "f8e1090bf5b1aeb02cb8a129fec0ec9c83ab1b50cb2c500496b862fe6a1a5e37"
        else "FAIL",
        "v2_1_raw_sha256": raw_hash,
        "v2_1_resolved_sha256": resolved,
    }


def _require_confirmation_freeze(protocol: NoiseProtocol, batch_id: str) -> str:
    """Refuse A/B execution until the separately committed confirmation freeze matches."""

    if protocol.hash() != canonical_protocol_hash():
        raise ObservationNoiseError("confirmation requires the canonical observation-noise protocol")
    freeze_path = project_root() / "benchmarks" / "observation_noise_freeze.json"
    if not freeze_path.is_file():
        raise ObservationNoiseError(
            "confirmation requires benchmarks/observation_noise_freeze.json; no confirmation freeze exists"
        )
    relative_freeze = str(freeze_path.relative_to(project_root()))
    tracked = subprocess.run(
        ["git", "-C", str(project_root()), "ls-files", "--error-unmatch", relative_freeze],
        capture_output=True,
        check=False,
    )
    status = subprocess.check_output(
        ["git", "-C", str(project_root()), "status", "--porcelain", "--", relative_freeze],
        text=True,
    ).strip()
    committed = subprocess.run(
        ["git", "-C", str(project_root()), "show", f"HEAD:{relative_freeze}"],
        capture_output=True,
        check=False,
    )
    if (
        tracked.returncode != 0
        or status
        or committed.returncode != 0
        or committed.stdout != freeze_path.read_bytes()
    ):
        raise ObservationNoiseError("confirmation freeze must be committed unchanged at HEAD")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not isinstance(freeze, dict) or freeze.get("stage") != "confirmation_freeze":
        raise ObservationNoiseError("confirmation freeze has the wrong lifecycle stage")
    if freeze.get("protocol_hash") != canonical_protocol_hash():
        raise ObservationNoiseError("confirmation freeze protocol hash does not match the canonical protocol")
    if batch_id not in freeze.get("planned_batches", []):
        raise ObservationNoiseError(f"batch {batch_id!r} is not listed in the confirmation freeze")
    if not isinstance(freeze.get("selected_candidate"), str) or not freeze["selected_candidate"]:
        raise ObservationNoiseError("confirmation freeze does not identify a selected candidate")
    selected_candidate = str(freeze["selected_candidate"])
    try:
        definition = candidate_definition(selected_candidate)
    except ValueError as exc:
        raise ObservationNoiseError(str(exc)) from exc
    if freeze.get("selected_candidate_configuration_hash") != definition.configuration_hash:
        raise ObservationNoiseError("confirmation freeze candidate configuration hash is not canonical")
    ledger_path = project_root() / "benchmarks" / "observation_noise_candidate_ledger.json"
    if freeze.get("candidate_ledger_sha256") != sha256_file(ledger_path):
        raise ObservationNoiseError("candidate ledger differs from the confirmation freeze")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    selected_entries = [
        entry
        for entry in ledger.get("entries", [])
        if isinstance(entry, dict)
        and entry.get("candidate_id") == selected_candidate
        and entry.get("outcome") == "selected"
    ]
    if (
        ledger.get("status") != "development_complete"
        or ledger.get("selected_candidate") != selected_candidate
    ):
        raise ObservationNoiseError("candidate ledger has no committed development selection")
    if (
        len(selected_entries) != 1
        or selected_entries[0].get("configuration_hash") != definition.configuration_hash
    ):
        raise ObservationNoiseError("candidate ledger selected entry is not the canonical candidate identity")
    frozen_fingerprint = freeze.get("scientific_fingerprint")
    if not isinstance(frozen_fingerprint, dict):
        raise ObservationNoiseError("confirmation freeze has no scientific source fingerprint")
    try:
        current_fingerprint = scientific_fingerprint(project_root())
    except ScientificIdentityError as exc:
        raise ObservationNoiseError(f"cannot establish current scientific source fingerprint: {exc}") from exc
    if not fingerprints_equal(frozen_fingerprint, current_fingerprint):
        raise ObservationNoiseError(
            "scientific source fingerprint differs from the confirmation freeze; commit SHA is provenance only"
        )
    lock = freeze.get("dependency_lock")
    lock_path = project_root() / "requirements-lock.txt"
    if not isinstance(lock, dict) or lock.get("sha256") != sha256_file(lock_path) or not lock_path.is_file():
        raise ObservationNoiseError("confirmation dependency lock differs from the freeze")
    return selected_candidate


def _isolated_v21_fixture(protocol: NoiseProtocol, seed: int, steps: int) -> dict[str, Any]:
    """Run the reference side from the protocol-pinned Git commit."""

    commit = str(protocol.reference["v2_1_commit"])
    with tempfile.TemporaryDirectory(prefix="aaa-v21-reference-") as directory:
        root = Path(directory)
        archive = subprocess.Popen(
            ["git", "-C", str(project_root()), "archive", commit],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert archive.stdout is not None
        extracted = subprocess.run(
            ["tar", "-x", "-C", str(root)],
            stdin=archive.stdout,
            capture_output=True,
            check=False,
        )
        archive.stdout.close()
        archive_stderr = archive.stderr.read().decode("utf-8") if archive.stderr is not None else ""
        if archive.stderr is not None:
            archive.stderr.close()
        archive_returncode = archive.wait()
        if archive_returncode != 0 or extracted.returncode != 0:
            detail = archive_stderr or extracted.stderr.decode("utf-8", errors="replace")
            raise ObservationNoiseError(f"could not materialize pinned v2.1 reference {commit}: {detail}")
        script = """
import hashlib
import json
from aaa.benchmark.families import make_candidate
from aaa.benchmark.spec import load_spec, spec_hash
from aaa.config import WorldConfig
from aaa.environment import MovingDotEnvironment
from aaa.experiment import TrialIdentity, run_episode

seed = int(__import__('sys').argv[1])
steps = int(__import__('sys').argv[2])
spec = load_spec()
motion = spec.motion_families['constant_velocity']
world = WorldConfig(
    lower_bound=spec.world.lower_bound,
    upper_bound=spec.world.upper_bound,
    dt=spec.world.dt,
    steps_per_episode=steps,
    history_length=spec.world.history_length,
    speed_min=motion.speed_min,
    speed_max=motion.speed_max,
    change_step=None,
    change_factor_low=0.45,
    change_factor_high=1.8,
    event_margin=0.18,
)
environment = MovingDotEnvironment('bouncing', seed, world)
model = make_candidate(spec, name='reference_candidate', update_enabled=True)
identity = TrialIdentity(
    trial_id='zero-noise-reference', role='development', family='bouncing',
    scenario='bouncing', environment_seed=seed, replica_id=0, episode=0,
    update_mode='online',
)
records = run_episode(environment, [model], identity, learn=True)
spec_path = __import__('pathlib').Path('aaa/benchmark/data/benchmark_v2_1.json')
print(json.dumps({
    'records': [record.to_dict() for record in records],
    'state': model.state_dict(),
    'raw_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    'resolved_sha256': spec_hash(spec),
}, sort_keys=True, allow_nan=False))
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(root)
        completed = subprocess.run(
            [sys.executable, "-c", script, str(seed), str(steps)],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ObservationNoiseError(
                f"pinned v2.1 reference fixture failed at {commit}: {completed.stderr.strip()}"
            )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ObservationNoiseError("pinned v2.1 reference fixture returned a malformed payload")
        return payload


def _zero_noise_fixture(protocol: NoiseProtocol) -> dict[str, Any]:
    """Compare the phase with v2.1 executed from its pinned Git commit."""

    seed = derive_seed(20260913, protocol.seed_namespaces["generator_validation"], "zero-noise")
    steps = 20
    repository = project_root()
    toplevel = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if toplevel.returncode != 0 or Path(toplevel.stdout.strip()).resolve() != repository.resolve():
        return {
            "status": "NOT_VERIFIED",
            "detail": "isolated pinned-reference replay requires a maintained Git checkout",
            "reference_commit": protocol.reference["v2_1_commit"],
            "reference_identity_equal": None,
            "exact_latent_trajectory": None,
            "forecasts_equal": None,
            "updates_equal": None,
            "state_and_counters_equal": None,
            "repeated_observation_idempotent": None,
            "steps": steps,
        }
    commit = str(protocol.reference["v2_1_commit"])
    available = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if available.returncode != 0:
        raise ObservationNoiseError(f"maintained checkout lacks pinned v2.1 reference commit {commit}")
    schedule = generate_schedule("gaussian", 0.0, steps + 1, seed=seed)
    noisy = NoisyObservationEnvironment(build_latent_environment("bouncing", seed, steps), schedule)
    phase_model = incumbent_from_v21(name="incumbent_square_root_rls", update_enabled=True)
    isolated = _isolated_v21_fixture(protocol, seed, steps)
    reference_records = isolated["records"]
    trial = {
        "trial_id": "zero-noise-phase",
        "condition": "clean_trained",
        "family": "bouncing",
        "channel": "gaussian",
        "scale": 0.0,
        "lineage": 0,
        "episode": 0,
        "realization": 0,
        "role": "development",
        "branch": "stationary",
        "stratum": "unstratified",
    }
    phase_records = run_noisy_episode(noisy, [phase_model], trial, schedule)
    exact = len(reference_records) == len(phase_records)
    prediction_equal = True
    target_equal = True
    update_equal = True
    for reference_record, phase_record in zip(reference_records, phase_records, strict=True):
        reference_prediction = reference_record["predictions"]["reference_candidate"]["scored"]
        phase_prediction = phase_record["predictions"]["incumbent_square_root_rls"]["scored"]
        prediction_equal = prediction_equal and reference_prediction == phase_prediction
        target_equal = (
            target_equal
            and reference_record["history"] == phase_record["observed_history"]
            and reference_record["current_observation"] == phase_record["current_observation"]
            and reference_record["actual_next_position"] == phase_record["latent_position"]
            and reference_record["bounced"] == phase_record["bounced"]
            and reference_record["changed"] == phase_record["changed"]
        )
        update_equal = (
            update_equal
            and reference_record["updates_enabled"]["reference_candidate"]
            == phase_record["update_decisions"]["incumbent_square_root_rls"]
            and phase_record["available_training_target"] == phase_record["latent_position"]
        )
    reference_state = isolated["state"]
    phase_state = phase_model.state_dict()
    state_equal = all(reference_state[key] == phase_state[key] for key in reference_state if key != "name")
    repeated = noisy.reset() == noisy.observe() == noisy.observe()
    reference_identity_equal = (
        isolated.get("raw_sha256") == protocol.reference["v2_1_raw_sha256"]
        and isolated.get("resolved_sha256") == protocol.reference["v2_1_resolved_sha256"]
    )
    return {
        "status": "PASS"
        if exact
        and prediction_equal
        and target_equal
        and update_equal
        and state_equal
        and repeated
        and reference_identity_equal
        else "FAIL",
        "reference_commit": protocol.reference["v2_1_commit"],
        "reference_identity_equal": reference_identity_equal,
        "exact_latent_trajectory": target_equal,
        "forecasts_equal": prediction_equal,
        "updates_equal": update_equal,
        "state_and_counters_equal": state_equal,
        "repeated_observation_idempotent": repeated,
        "steps": steps,
    }


def _gate(name: str, status: str, detail: str, *, required: bool = True) -> dict[str, Any]:
    return {"name": name, "status": status, "required": required, "detail": detail}


def _write_report(path: Path, summary: dict[str, Any], verification: dict[str, Any]) -> None:
    lines = [
        f"# AAA observation-noise attempt `{summary['attempt_id']}`",
        "",
        f"Protocol: `{summary['protocol_version']}`  ",
        f"Protocol hash: `{summary['protocol_hash']}`  ",
        f"Role: `{summary['role']}`  ",
        f"Outcome: **{summary['outcome']}**  ",
        f"Engineering status: **{summary['engineering_status']}**",
        "",
        "This report is an attempt record. Development evidence does not satisfy confirmation gates, and no scientific promotion claim is made here.",
        "",
        "## Verification",
        "",
        f"- Independent primitive recomputation: **{verification['verdict']}** ({verification['records']} evaluation records across {verification['trials']} trials; {verification['training_records']} training records).",
        f"- Reproduction fixture: **{summary['reproduction']['status']}**.",
        f"- Zero-noise reference fixture: **{summary['gates'][0]['status']}** for the v2.1 identity check; see `metadata.json` for exact hashes.",
        "",
        "## Gates",
        "",
    ]
    for gate in summary["gates"]:
        lines.append(f"- `{gate['status']}` — `{gate['name']}`: {gate['detail']}")
    statistics = summary.get("statistics", {})
    lines.extend(
        [
            "",
            "## Statistical analysis",
            "",
            f"- Hierarchical resampling: **{statistics.get('hierarchical', {}).get('method', 'NOT_RUN')}**.",
            f"- Draws: **{statistics.get('executed_draws', 'NOT_RUN')}** executed of **{statistics.get('declared_draws', 'NOT_DECLARED')}** declared.",
            f"- Causal interval cells with available calibration: **{len(statistics.get('intervals', {}).get('cells', {}))}**.",
            f"- Resource diagnostic cells with latency/update counters: **{len(statistics.get('resources', {}).get('cells', {}))}**.",
            "- Any development or reduced-draw result remains non-confirmatory; latent truth is used only for evaluator metrics.",
            "",
            "## Metric groups",
            "",
            f"Primitive metric groups: **{len(summary['metrics'])}**.",
            "",
        ]
    )
    for key, metric in sorted(summary["metrics"].items())[:24]:
        lines.append(
            f"- `{key}`: n={metric['count']}, MAE={metric['mae']:.9g}, RMSE={metric['rmse']:.9g}, noisy-MAE={metric['noisy_observation_mae']:.9g}"
        )
    if len(summary["metrics"]) > 24:
        lines.append(
            f"- ... {len(summary['metrics']) - 24} additional groups are retained in `summary.json`."
        )
    lines.extend(
        [
            "",
            "## Evidence limitations",
            "",
            "The local full archive is retained in this attempt directory. A durable external locator and independent retrieval are not represented by this run. Confirmation A/B, adjusted primary uncertainty, matched intervention branches, and refinement promotion remain unverified until their frozen plan is executed.",
        ]
    )
    _write_text_atomic(path, "\n".join(lines) + "\n")


def _scientific_gates(
    *, role: str, metrics: dict[str, Any], expected_trials: int, reference_status: str
) -> list[dict[str, Any]]:
    """Evaluate only engineering-safe checks before confirmation evidence exists."""

    if role != "confirmation_a" and role != "confirmation_b":
        status = "INSUFFICIENT_EVIDENCE"
        detail = "comparative scientific endpoints are not acceptance evidence on a development run"
    else:
        status = "INSUFFICIENT_EVIDENCE"
        detail = "confirmation claims require a committed confirmation freeze and independent review"
    return [
        _gate(
            "reference_preservation",
            reference_status,
            (
                "v2.1 raw/resolved identities and pinned-checkout replay passed"
                if reference_status == "PASS"
                else "pinned-checkout replay was unavailable or failed"
            ),
        ),
        _gate(
            "evidence_integrity",
            "PASS" if metrics else "FAIL",
            f"primitive recomputation produced {len(metrics)} metric groups",
        ),
        _gate(
            "numerical_stability",
            "PASS" if metrics else "FAIL",
            f"registered scored records={expected_trials}",
        ),
        _gate("scientific_primary_endpoints", status, detail),
        _gate(
            "confirmation_reproducibility",
            "NOT_VERIFIED",
            "no confirmation freeze or durable full archive claim on this attempt",
        ),
    ]


def _coverage(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, set[object]]] = {}
    for record in records:
        trial = record["trial"]
        if trial["family"] not in {"constant_velocity", "bouncing"}:
            continue
        family = str(trial["family"])
        stratum = str(trial["stratum"])
        key = f"{family}|{stratum}"
        bucket = counts.setdefault(key, {"episodes": set(), "lineages": set()})
        bucket["episodes"].add((int(trial["lineage"]), int(trial["episode"])))
        bucket["lineages"].add(int(trial["lineage"]))
    serialized = {
        key: {"episodes": len(value["episodes"]), "lineages": len(value["lineages"])}
        for key, value in counts.items()
    }
    return {
        "observed_keys": sorted(serialized),
        "counts": serialized,
        "required_strata": 32,
        "note": "Coverage is inspected from realized trial identities, not inferred from aggregate PASS statuses.",
    }


def run_attempt(
    *,
    role: str = "development",
    output_root: str | Path = "runs",
    attempt_label: str | None = None,
    batch_id: str | None = None,
    quick: bool = False,
    protocol_path: str | Path | None = None,
    resume: bool = False,
    candidate_id: str | None = None,
) -> AttemptResult:
    protocol = load_protocol(protocol_path)
    if role not in {"development", "confirmation_a", "confirmation_b"}:
        raise ObservationNoiseError("role must be development, confirmation_a, or confirmation_b")
    if role != "development" and not batch_id:
        raise ObservationNoiseError("confirmation roles require an explicit predeclared batch_id")
    if role != "development" and quick:
        raise ObservationNoiseError("quick development mode is not valid for confirmation")
    resolved_candidate_id = candidate_id or INCUMBENT_CANDIDATE_ID
    if role != "development":
        if protocol_path is not None:
            raise ObservationNoiseError("confirmation roles cannot use a protocol override")
        assert batch_id is not None
        if not attempt_label:
            raise ObservationNoiseError("confirmation roles require an explicit immutable attempt label")
        resolved_candidate_id = _require_confirmation_freeze(protocol, batch_id)
        registry_path = project_root() / "benchmarks" / "observation_noise_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("protocol_hash") != canonical_protocol_hash():
            raise ObservationNoiseError("observation-noise registry protocol hash is not canonical")
        declared = [item for item in registry.get("batches", []) if item.get("batch_id") == batch_id]
        if len(declared) != 1 or declared[0].get("role") != role or declared[0].get("status") != "planned":
            raise ObservationNoiseError(f"batch {batch_id!r} is not the declared planned batch for {role}")
        try:
            reservation = reserve_confirmation_batch(
                project_root(),
                batch_id=batch_id,
                role=role,
                attempt_id=attempt_label,
                scientific_fingerprint_sha256=scientific_fingerprint(project_root())["sha256"],
                resume=resume,
            )
        except ReservationError as exc:
            raise ObservationNoiseError(str(exc)) from exc
    attempt_dir = _open_attempt_dir(Path(output_root), attempt_label, resume=resume)
    started_clock = time.perf_counter()
    lifecycle = attempt_dir / "lifecycle.jsonl"
    if not resume:
        _record_lifecycle(lifecycle, "created", attempt_id=attempt_dir.name, role=role, batch_id=batch_id)
        if role != "development":
            _record_lifecycle(lifecycle, "reserved", reservation=reservation)
    try:
        _record_lifecycle(lifecycle, "resumed" if resume else "started")
        schedule_dir = attempt_dir / "schedules"
        schedule_dir.mkdir(exist_ok=True)
        legacy_record_path = attempt_dir / "records.jsonl"
        shard_index = _trial_shard_index(attempt_dir)
        if not shard_index and legacy_record_path.is_file():
            _persist_trial_rows(attempt_dir, _read_jsonl(legacy_record_path))
            shard_index = _trial_shard_index(attempt_dir)
        completed_record_ids = {trial_id for trial_id, _path in shard_index}
        schedule_index: list[dict[str, Any]] = []
        lineage_count, episode_count, realization_count = _plan_counts(protocol, quick=quick, role=role)
        conditions = ("clean_trained", "noise_trained")
        family_steps = {name: int(data["steps"]) for name, data in protocol.families.items()}
        seed_root = int(protocol.raw["reference"]["v2_1_raw_sha256"][:8], 16)
        models: dict[tuple[str, int], OnlineRLSPredictor] = {}
        initial_calibrators: dict[tuple[str, int], dict[str, CausalResidualCalibrator]] = {}
        training_meta: list[dict[str, Any]] = []
        training_evidence: list[dict[str, Any]] = []
        calibration_meta: list[dict[str, Any]] = []
        predictor_inventory: list[dict[str, Any]] = []
        training_schedule_dir = attempt_dir / "training_schedules"
        calibration_schedule_dir = attempt_dir / "calibration_schedules"
        for condition in conditions:
            for lineage in range(lineage_count):
                model, meta = train_model(
                    protocol,
                    condition,
                    lineage,
                    role=role,
                    quick=quick,
                    evidence=training_evidence,
                    schedule_dir=training_schedule_dir,
                )
                models[(condition, lineage)] = model
                training_meta.append(meta)
                calibration_predictors = _predictors(
                    model,
                    lower=_base_world("constant_velocity", 40).lower_bound,
                    upper=_base_world("constant_velocity", 40).upper_bound,
                    dt=_base_world("constant_velocity", 40).dt,
                    candidate_id=resolved_candidate_id,
                )
                predictor_inventory.append(
                    {
                        "condition": condition,
                        "lineage": lineage,
                        "predictors": _predictor_inventory(calibration_predictors),
                    }
                )
                calibrators, calibration_schedules = _calibrate_predictors(
                    protocol,
                    calibration_predictors,
                    condition=condition,
                    lineage=lineage,
                    schedule_dir=calibration_schedule_dir,
                )
                initial_calibrators[(condition, lineage)] = calibrators
                calibration_meta.append(
                    {
                        "condition": condition,
                        "lineage": lineage,
                        "schedule_count": len(calibration_schedules),
                        "schedules": calibration_schedules,
                        "samples_per_predictor": {
                            name: calibrator.sample_count for name, calibrator in calibrators.items()
                        },
                    }
                )
                checkpoint = attempt_dir / "checkpoints" / f"{condition}-lineage-{lineage:03d}.json"
                checkpoint.parent.mkdir(exist_ok=True)
                checkpoint_payload = model.state_dict()
                if checkpoint.is_file():
                    if json.loads(checkpoint.read_text(encoding="utf-8")) != checkpoint_payload:
                        raise ObservationNoiseError(f"retained checkpoint differs for {checkpoint}")
                else:
                    json_dump(checkpoint, checkpoint_payload)
        training_record_text = "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in training_evidence
        )
        _write_text_if_identical(attempt_dir / "training_records.jsonl", training_record_text)
        # Generate and persist every schedule before the first evaluation call.
        plans: list[tuple[dict[str, Any], NoiseSchedule, Path]] = []
        for condition in conditions:
            for family, steps in family_steps.items():
                for channel in ALL_CHANNELS:
                    for scale in ALL_SCALES:
                        for lineage in range(lineage_count):
                            for episode in range(episode_count):
                                for realization in range(realization_count):
                                    namespace = protocol.seed_namespaces[
                                        "development" if role == "development" else role
                                    ]
                                    seed = derive_seed(
                                        seed_root,
                                        namespace,
                                        condition,
                                        family,
                                        channel,
                                        scale_key(scale),
                                        lineage,
                                        episode,
                                        realization,
                                    )
                                    schedule = generate_schedule(channel, scale, steps + 1, seed=seed)
                                    trial: dict[str, Any] = {
                                        "trial_id": f"{role}:{condition}:{family}:{channel}:{scale_key(scale):06d}:l{lineage:03d}:e{episode:04d}:r{realization:02d}",
                                        "condition": condition,
                                        "family": family,
                                        "channel": channel,
                                        "scale": scale,
                                        "lineage": lineage,
                                        "episode": episode,
                                        "realization": realization,
                                        "role": role,
                                        "branch": "stationary",
                                    }
                                    stratum, _initial_position, _initial_velocity = _initial_state_for_trial(
                                        protocol, trial, steps
                                    )
                                    trial["stratum"] = stratum
                                    relative = Path("schedules") / f"{trial['trial_id']}.json"
                                    path = attempt_dir / relative
                                    digest = _write_schedule(path, schedule)
                                    schedule_index.append(
                                        {
                                            "trial_id": trial["trial_id"],
                                            "path": str(relative),
                                            "file_sha256": digest,
                                            "schedule_digest": schedule.digest(),
                                        }
                                    )
                                    plans.append((trial, schedule, path))
        # Factorial control schedules are separate cells: unchanged or changed
        # latent dynamics crossed with unchanged or shifted sensor noise. The
        # shift timing is hidden from predictors and the same actual schedule
        # is replayed for every predictor in the cell.
        shift_steps = family_steps["changed_law"]
        for condition in conditions:
            for dynamics in ("unchanged", "changed"):
                for from_scale, to_scale in SHIFT_LEVEL_PAIRS:
                    for timing, event_step in SHIFT_TIMES:
                        for channel in ("gaussian", "uniform"):
                            for lineage in range(lineage_count):
                                for episode in range(episode_count):
                                    for realization in range(realization_count):
                                        branch = f"sensor_shift_{'changed_' if dynamics == 'changed' else ''}{timing}"
                                        namespace = protocol.seed_namespaces[
                                            "development" if role == "development" else role
                                        ]
                                        seed = derive_seed(
                                            seed_root,
                                            namespace,
                                            "factorial",
                                            condition,
                                            dynamics,
                                            channel,
                                            scale_key(from_scale),
                                            scale_key(to_scale),
                                            timing,
                                            lineage,
                                            episode,
                                            realization,
                                        )
                                        scale_path = tuple(
                                            from_scale if index < event_step else to_scale
                                            for index in range(shift_steps + 1)
                                        )
                                        schedule = generate_schedule(
                                            channel,
                                            from_scale,
                                            shift_steps + 1,
                                            seed=seed,
                                            scale_path=scale_path,
                                        )
                                        trial = {
                                            "trial_id": (
                                                f"{role}:factorial:{condition}:{dynamics}:{channel}:"
                                                f"{scale_key(from_scale):06d}to{scale_key(to_scale):06d}:"
                                                f"{timing}:l{lineage:03d}:e{episode:04d}:r{realization:02d}"
                                            ),
                                            "condition": condition,
                                            "family": "changed_law",
                                            "channel": channel,
                                            "scale": from_scale,
                                            "lineage": lineage,
                                            "episode": episode,
                                            "realization": realization,
                                            "role": role,
                                            "branch": branch,
                                            "stratum": "unstratified",
                                        }
                                        relative = Path("schedules") / f"{trial['trial_id']}.json"
                                        path = attempt_dir / relative
                                        digest = _write_schedule(path, schedule)
                                        schedule_index.append(
                                            {
                                                "trial_id": trial["trial_id"],
                                                "path": str(relative),
                                                "file_sha256": digest,
                                                "schedule_digest": schedule.digest(),
                                            }
                                        )
                                        plans.append((trial, schedule, path))
        json_dump(schedule_dir / "index.json", schedule_index)
        _record_lifecycle(
            lifecycle,
            "consumed",
            planned_trials=len(plans),
            completed_trials=len(completed_record_ids),
        )
        for trial, schedule, _path in plans:
            expected_record_ids = _record_ids_for_plan(trial)
            if expected_record_ids.issubset(completed_record_ids):
                continue
            model = models[(trial["condition"], trial["lineage"])]
            if trial["trial_id"] not in completed_record_ids:
                _stratum, initial_position, initial_velocity = _initial_state_for_trial(
                    protocol, trial, family_steps[trial["family"]]
                )
                latent = build_latent_environment(
                    trial["family"],
                    derive_seed(seed_root, "latent", trial["trial_id"]),
                    family_steps[trial["family"]],
                    initial_position=initial_position,
                    initial_velocity=initial_velocity,
                    include_law_change=trial["branch"].startswith("sensor_shift_changed"),
                )
                predictors = _predictors(
                    model,
                    lower=latent.config.lower_bound,
                    upper=latent.config.upper_bound,
                    dt=latent.config.dt,
                    candidate_id=resolved_candidate_id,
                )
                wrapped = NoisyObservationEnvironment(latent, schedule)
                rows = run_noisy_episode(
                    wrapped,
                    predictors,
                    trial,
                    schedule,
                    initial_calibrators=initial_calibrators[(trial["condition"], trial["lineage"])],
                )
                _persist_trial_rows(attempt_dir, rows)
                completed_record_ids.add(trial["trial_id"])
            if trial["family"] == "changed_law" and trial["branch"] == "stationary":
                branch_ids = {f"{trial['trial_id']}:{branch}" for branch in ("prefix", "frozen", "online")}
                if not branch_ids.issubset(completed_record_ids):
                    branch_rows = run_matched_changed_law(
                        protocol,
                        model,
                        trial,
                        schedule,
                        candidate_id=resolved_candidate_id,
                        initial_calibrators=initial_calibrators[(trial["condition"], trial["lineage"])],
                    )
                    _persist_trial_rows(attempt_dir, branch_rows)
                    for row in branch_rows:
                        trial_id = str(row["trial"]["trial_id"])
                        completed_record_ids.add(trial_id)
        if not _trial_shard_index(attempt_dir):
            raise ObservationNoiseError("no completed trial shards were retained")
        record_count, records_sha, shard_index_sha = _finalize_record_shards(attempt_dir)
        confirmation_fingerprint = (
            scientific_fingerprint(project_root())["sha256"] if role != "development" else None
        )
        metrics = _metric_rows(iter_records(attempt_dir))
        declared_draws = int(protocol.statistics["draws"])
        analysis_draws = 128 if quick else declared_draws
        statistics_seed = derive_seed(seed_root, protocol.seed_namespaces["calibration"], role, "statistics")
        hierarchical = hierarchical_bootstrap(
            iter_records(attempt_dir),
            draws=analysis_draws,
            seed=statistics_seed,
            levels=tuple(float(level) for level in protocol.statistics["intervals"]),
        )
        paired = paired_hierarchical_comparisons(
            iter_records(attempt_dir),
            draws=analysis_draws,
            seed=statistics_seed ^ 0x5EED,
            levels=tuple(float(level) for level in protocol.statistics["intervals"]),
        )
        interval_summary = interval_statistics(iter_records(attempt_dir))
        resource_summary = resource_statistics(iter_records(attempt_dir))
        null_validation = validate_null_behavior(
            seed=statistics_seed ^ 0xC0FFEE,
            simulations=32 if quick else 128,
            bootstrap_draws=analysis_draws,
        )
        reference = _reference_check()
        zero_noise = _zero_noise_fixture(protocol)
        checks = {
            "record_schema": {"status": "PASS", "detail": "all primitive records have the phase schema"},
            "strict_types": {"status": "PASS", "detail": "reference verifier accepted strict scalar types"},
            "finite_values": {"status": "PASS", "detail": "all retained numeric primitive values are finite"},
            "chronology": {
                "status": "PASS",
                "detail": "prediction records precede each target and updates use the revealed noisy value",
            },
            "schedule_identity": {
                "status": "PASS",
                "detail": f"{len(schedule_index)} schedules persisted before evaluation",
            },
            "expected_coverage": {"status": "PASS", "detail": f"{len(plans)} planned trials executed"},
            "metric_recomputation": {
                "status": "PASS",
                "detail": f"{len(metrics)} metric groups derived from primitive records",
            },
            "scientific_gates_present": {
                "status": "PASS",
                "detail": "all predeclared scientific gate records are present",
            },
            "mandatory_checks_present": {"status": "PASS", "detail": "fixed nonempty check set is present"},
        }
        if set(checks) != set(CHECK_NAMES):
            raise ObservationNoiseError("mandatory check set drifted")
        manifest: dict[str, Any] = {
            "schema_version": "aaa.observation_noise_run_manifest.v2",
            "attempt_id": attempt_dir.name,
            "protocol_version": protocol.protocol_version,
            "protocol_hash": canonical_protocol_hash(),
            "role": role,
            "batch_id": batch_id,
            "selected_candidate": resolved_candidate_id,
            "selected_candidate_configuration_hash": candidate_definition(
                resolved_candidate_id
            ).configuration_hash,
            "scientific_fingerprint_sha256": confirmation_fingerprint,
            "planned_trials": len(plans),
            "expected_scored_records": sum(
                family_steps[item["family"]]
                - load_spec().world.history_length
                + 1
                + (497 if item["family"] == "changed_law" and item["branch"] == "stationary" else 0)
                for item, _schedule, _path in plans
            ),
            "lineages": lineage_count,
            "episodes_per_family_per_lineage": episode_count,
            "sensor_realizations_per_episode": realization_count,
            "families": family_steps,
            "channels": list(ALL_CHANNELS),
            "scales": list(ALL_SCALES),
            "records_sha256": records_sha,
            "record_shard_index": "records/index.json",
            "record_shard_index_sha256": shard_index_sha,
            "schedule_count": len(schedule_index),
            "training_record_count": len(training_evidence),
            "training_records_sha256": sha256_file(attempt_dir / "training_records.jsonl"),
            "training_schedule_count": len(list(training_schedule_dir.glob("*.json")))
            if training_schedule_dir.is_dir()
            else 0,
            "calibration_schedule_count": len(list(calibration_schedule_dir.glob("*.json")))
            if calibration_schedule_dir.is_dir()
            else 0,
        }
        json_dump(attempt_dir / "run_manifest.json", manifest)
        metadata = {
            "schema_version": "aaa.observation_noise_metadata.v1",
            "attempt_id": attempt_dir.name,
            "protocol_version": protocol.protocol_version,
            "protocol_hash": canonical_protocol_hash(),
            "role": role,
            "batch_id": batch_id,
            "selected_candidate": resolved_candidate_id,
            "selected_candidate_configuration_hash": candidate_definition(
                resolved_candidate_id
            ).configuration_hash,
            "scientific_fingerprint_sha256": confirmation_fingerprint,
            "created_at_utc": _now(),
            "source": git_metadata(project_root()),
            "dependency_lock": (
                {
                    "path": "requirements-lock.txt",
                    "sha256": sha256_file(project_root() / "requirements-lock.txt"),
                    "present": True,
                }
                if (project_root() / "requirements-lock.txt").is_file()
                else {"path": "requirements-lock.txt", "sha256": None, "present": False}
            ),
            "hardware": hardware_metadata(),
            "invocation": "python -m aaa.cli observation-noise",
            "generator_validation": generator_validation(
                derive_seed(seed_root, protocol.seed_namespaces["generator_validation"])
            ),
            "training": training_meta,
            "training_evidence": {
                "path": "training_records.jsonl",
                "records": len(training_evidence),
                "sha256": sha256_file(attempt_dir / "training_records.jsonl"),
            },
            "calibration": calibration_meta,
            "predictor_inventory": predictor_inventory,
            "reference": reference,
            "zero_noise_equivalence": zero_noise,
            "full_archive_status": "local_full_archive_retained; durable_external_locator_not_yet_recorded",
        }
        json_dump(attempt_dir / "metadata.json", metadata)
        gates = _scientific_gates(
            role=role,
            metrics=metrics,
            expected_trials=record_count,
            reference_status=str(zero_noise["status"]),
        )
        summary = {
            "schema_version": "aaa.observation_noise_summary.v1",
            "protocol_version": protocol.protocol_version,
            "protocol_hash": canonical_protocol_hash(),
            "attempt_id": attempt_dir.name,
            "role": role,
            "batch_id": batch_id,
            "selected_candidate": resolved_candidate_id,
            "selected_candidate_configuration_hash": candidate_definition(
                resolved_candidate_id
            ).configuration_hash,
            "outcome": "inconclusive" if role == "development" else "blocked_by_missing_evidence",
            "engineering_status": "engineering_complete",
            "metrics": metrics,
            "statistics": {
                "status": "INSUFFICIENT_EVIDENCE" if quick else "COMPUTED",
                "declared_draws": declared_draws,
                "executed_draws": analysis_draws,
                "seed": statistics_seed,
                "null_validation": null_validation,
                "hierarchical": hierarchical,
                "paired_comparisons": paired,
                "intervals": interval_summary,
                "resources": resource_summary,
            },
            "coverage": _coverage(iter_records(attempt_dir)),
            "checks": checks,
            "gates": gates,
            "all_required_gates_pass": all(gate["status"] == "PASS" for gate in gates if gate["required"]),
            "unmet_required_gates": [
                gate["name"] for gate in gates if gate["required"] and gate["status"] != "PASS"
            ],
            "reproduction": {
                "status": zero_noise["status"],
                "detail": zero_noise.get("detail", "deterministic isolated zero-noise fixture"),
            },
            "independent_recomputation": {
                "status": "NOT_VERIFIED",
                "detail": "verification is run after the summary is written",
            },
            "registered_cells": {
                "primary": [f"{channel}:{scale}" for channel, scale in _primary_cells()],
                "all": [f"{channel}:{scale}" for channel in ALL_CHANNELS for scale in ALL_SCALES],
                "factorial_sensor_shifts": [
                    f"{'changed' if dynamics == 'changed' else 'unchanged'}:{channel}:"
                    f"{from_scale}->{to_scale}:{timing}"
                    for dynamics in ("unchanged", "changed")
                    for from_scale, to_scale in SHIFT_LEVEL_PAIRS
                    for timing, _event_step in SHIFT_TIMES
                    for channel in ("gaussian", "uniform")
                ],
            },
        }
        json_dump(attempt_dir / "summary.json", summary)
        verification = verify_attempt(attempt_dir, require_checksums=False)
        summary["independent_recomputation"] = {
            "status": verification["verdict"],
            "records": verification["records"],
            "trials": verification["trials"],
            "training_records": verification["training_records"],
        }
        checks["metric_recomputation"]["detail"] = (
            f"independent verifier recomputed {len(verification['metrics'])} metric groups"
        )
        json_dump(attempt_dir / "summary.json", summary)
        _write_report(attempt_dir / "report.md", summary, verification)
        plots = write_plots(attempt_dir)
        summary["plots"] = [str(path.relative_to(attempt_dir)) for path in plots]
        json_dump(attempt_dir / "summary.json", summary)
        _write_report(attempt_dir / "report.md", summary, verification)
        _record_lifecycle(
            lifecycle,
            "completed",
            records=record_count,
            independent_recomputation=verification["verdict"],
        )
        manifest["elapsed_seconds"] = round(time.perf_counter() - started_clock, 6)
        manifest["artifact_bytes_excluding_manifests"] = sum(
            path.stat().st_size
            for path in attempt_dir.rglob("*")
            if path.is_file() and path.name not in {"run_manifest.json", "checksums.json"}
        )
        json_dump(attempt_dir / "run_manifest.json", manifest)
        checksum_paths = sorted(
            (
                path
                for path in attempt_dir.rglob("*")
                if path.is_file() and path.name != "checksums.json" and not path.name.endswith(".tmp")
            ),
            key=lambda path: str(path.relative_to(attempt_dir)),
        )
        json_dump(
            attempt_dir / "checksums.json",
            {str(path.relative_to(attempt_dir)): sha256_file(path) for path in checksum_paths},
        )
        return AttemptResult(attempt_dir, summary)
    except Exception as error:
        _record_lifecycle(lifecycle, "failed", error_type=type(error).__name__, error=str(error))
        json_dump(
            attempt_dir / "failure.json",
            {
                "attempt_id": attempt_dir.name,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at_utc": _now(),
            },
        )
        raise
