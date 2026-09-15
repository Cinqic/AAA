"""First-class candidate identities for the observation-noise phase.

Candidate selection is deliberately separate from the mandatory comparator
inventory.  A candidate has a stable identity and configuration hash before
it is trained or evaluated.  Development outcomes are ledger data; they are
never inferred from a confirmation command-line argument.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..benchmark.spec import BenchmarkSpec, load_spec
from ..predictors import (
    MAX_CONDITION_NUMBER,
    PSD_TOLERANCE,
    SYMMETRY_TOLERANCE,
    OnlineRLSPredictor,
)
from .spec import load_protocol

INCUMBENT_CANDIDATE_ID = "incumbent-no-refinement-v1"
CLIP_CANDIDATE_IDS = (
    "causal-innovation-clip-025-v1",
    "causal-innovation-clip-050-v1",
    "causal-innovation-clip-100-v1",
)


def _hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _incumbent_parameters(spec: BenchmarkSpec) -> dict[str, Any]:
    candidate = spec.candidate
    return {
        "model": candidate.model,
        "feature_set": candidate.feature_set,
        "features": list(candidate.features),
        "displacement_scale_speed": candidate.displacement_scale_speed,
        "forgetting": candidate.forgetting,
        "forgetting_mode": candidate.forgetting_mode,
        "ridge": candidate.ridge,
        "trace_bound": candidate.trace_bound,
        "reflect": candidate.reflect,
        "unfold_target": candidate.unfold_target,
        "skip_after_reflected_prediction": candidate.skip_after_reflected_prediction,
        "dead_zone": candidate.dead_zone,
        "detector_multiplier": candidate.detector_multiplier,
        "detector_floor": candidate.detector_floor,
        "detector_decay": candidate.detector_decay,
        "boundary_policy": candidate.boundary_policy,
    }


@dataclass(frozen=True)
class CandidateDefinition:
    """Immutable scientific identity plus the frozen development disposition."""

    candidate_id: str
    parent_candidate_id: str
    mechanism_description: str
    parameters: dict[str, Any]
    trainable_state: dict[str, Any]
    fixed_state: dict[str, Any]
    training_condition: str
    mechanism_change_count: int
    development_attempts: tuple[str, ...]
    outcome: str
    outcome_reason: str
    configuration_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_candidate_id": self.parent_candidate_id,
            "mechanism_description": self.mechanism_description,
            "parameters": self.parameters,
            "trainable_state": self.trainable_state,
            "fixed_state": self.fixed_state,
            "training_condition": self.training_condition,
            "mechanism_change_count": self.mechanism_change_count,
            "development_attempts": list(self.development_attempts),
            "outcome": self.outcome,
            "outcome_reason": self.outcome_reason,
            "configuration_hash": self.configuration_hash,
        }


def _definition(
    candidate_id: str,
    *,
    parameters: dict[str, Any],
    mechanism_description: str,
    mechanism_change_count: int,
) -> CandidateDefinition:
    config: dict[str, Any] = {
        "candidate_id": candidate_id,
        "parent_candidate_id": "v2.1-incumbent",
        "mechanism_description": mechanism_description,
        "parameters": parameters,
        "trainable_state": {
            "model_parameters": "OnlineRLSPredictor weights and square-root covariance are trained per declared condition",
            "update_enabled": True,
        },
        "fixed_state": {
            "observation_boundary_policy": "public scalar reflection only",
            "evaluator_event_labels": False,
            "law_change_labels": False,
        },
        "training_condition": "clean_trained or noise_trained, selected before evaluation",
        "mechanism_change_count": mechanism_change_count,
    }
    return CandidateDefinition(
        candidate_id=candidate_id,
        parent_candidate_id="v2.1-incumbent",
        mechanism_description=mechanism_description,
        parameters=parameters,
        trainable_state=config["trainable_state"],
        fixed_state=config["fixed_state"],
        training_condition=str(config["training_condition"]),
        mechanism_change_count=mechanism_change_count,
        development_attempts=(),
        outcome="unattempted",
        outcome_reason="",
        configuration_hash=_hash(config),
    )


def candidate_catalog(spec: BenchmarkSpec | None = None) -> dict[str, CandidateDefinition]:
    """Return the bounded candidate catalog in deterministic order."""

    reference = load_spec() if spec is None else spec
    incumbent = _incumbent_parameters(reference)
    definitions = [
        _definition(
            INCUMBENT_CANDIDATE_ID,
            parameters=incumbent,
            mechanism_description="Unchanged v2.1 square-root RLS incumbent; explicit no-refinement control.",
            mechanism_change_count=0,
        )
    ]
    for candidate_id, limit in zip(CLIP_CANDIDATE_IDS, (0.25, 0.50, 1.00), strict=True):
        definitions.append(
            _definition(
                candidate_id,
                parameters={**incumbent, "max_normalized_innovation": limit},
                mechanism_description=(
                    "Causal bounded innovation influence: clip the newly revealed noisy target's "
                    "innovation relative to the model's own raw forecast before the incumbent update."
                ),
                mechanism_change_count=1,
            )
        )
    return {definition.candidate_id: definition for definition in definitions}


def candidate_definition(candidate_id: str) -> CandidateDefinition:
    try:
        return candidate_catalog()[candidate_id]
    except KeyError as exc:
        raise ValueError(f"unknown observation-noise candidate {candidate_id!r}") from exc


def candidate_from_model(
    model: OnlineRLSPredictor,
    candidate_id: str,
    *,
    name: str,
    update_enabled: bool,
) -> OnlineRLSPredictor:
    """Construct a predictor from a trained model and an immutable candidate ID."""

    definition = candidate_definition(candidate_id)
    if candidate_id == INCUMBENT_CANDIDATE_ID:
        return OnlineRLSPredictor.from_state_dict(
            model.state_dict(), name=name, update_enabled=update_enabled
        )
    limit = float(definition.parameters["max_normalized_innovation"])
    return InnovationClipRLSPredictor.from_model(
        model,
        name=name,
        update_enabled=update_enabled,
        candidate_id=candidate_id,
        max_normalized_innovation=limit,
    )


class InnovationClipRLSPredictor(OnlineRLSPredictor):
    """Incumbent RLS with one causal, preregistered innovation bound."""

    format_version = "aaa.observation_noise_innovation_clip.v1"

    def __init__(self, *, candidate_id: str, max_normalized_innovation: float, **kwargs: Any) -> None:
        if max_normalized_innovation <= 0:
            raise ValueError("max_normalized_innovation must be positive")
        super().__init__(**kwargs)
        self.candidate_id = str(candidate_id)
        self.max_normalized_innovation = float(max_normalized_innovation)

    def update(self, history: Any, target_position: float) -> None:
        if not self.update_enabled:
            return
        raw = self.raw_predict(history)
        width = self.upper_bound - self.lower_bound
        clipped_target = raw + max(
            -self.max_normalized_innovation * width,
            min(self.max_normalized_innovation * width, float(target_position) - raw),
        )
        super().update(history, clipped_target)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "format_version": self.format_version,
                "candidate_id": self.candidate_id,
                "max_normalized_innovation": self.max_normalized_innovation,
            }
        )
        return state

    @classmethod
    def from_model(
        cls,
        model: OnlineRLSPredictor,
        *,
        name: str,
        update_enabled: bool,
        candidate_id: str,
        max_normalized_innovation: float,
        symmetry_tolerance: float = SYMMETRY_TOLERANCE,
        psd_tolerance: float = PSD_TOLERANCE,
        max_condition_number: float = MAX_CONDITION_NUMBER,
    ) -> InnovationClipRLSPredictor:
        state = model.state_dict()
        return cls(
            candidate_id=candidate_id,
            max_normalized_innovation=max_normalized_innovation,
            symmetry_tolerance=symmetry_tolerance,
            psd_tolerance=psd_tolerance,
            max_condition_number=max_condition_number,
            lower_bound=float(state["lower_bound"]),
            upper_bound=float(state["upper_bound"]),
            displacement_scale=float(state["displacement_scale"]),
            forgetting=float(state["forgetting"]),
            forgetting_mode=str(state.get("forgetting_mode", "exponential")),
            ridge=float(state["ridge"]),
            feature_set=str(state["feature_set"]),
            reflect=bool(state.get("reflect", True)),
            unfold_target=bool(state.get("unfold_target", False)),
            skip_after_reflected_prediction=bool(state.get("skip_after_reflected_prediction", False)),
            trace_bound=float(state.get("trace_bound", 1e5)),
            dead_zone=float(state.get("dead_zone", 0.0)),
            detector_multiplier=float(state.get("detector_multiplier", 0.0)),
            detector_floor=float(state.get("detector_floor", 1e-6)),
            detector_decay=float(state.get("detector_decay", 0.05)),
            name=name,
            update_enabled=update_enabled,
            weights=state["weights"],
            covariance=state["covariance"],
            sqrt_factor=state.get("sqrt_factor"),
            update_count=int(state.get("update_count", 0)),
            forgetting_suspensions=int(state.get("forgetting_suspensions", 0)),
            dead_zone_skips=int(state.get("dead_zone_skips", 0)),
            detected_surprises=int(state.get("detected_surprises", 0)),
            reflection_skips=int(state.get("reflection_skips", 0)),
            previous_prediction_reflected=bool(state.get("previous_prediction_reflected", False)),
            error_ewma=float(state.get("error_ewma", 0.0)),
        )

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        name: str | None = None,
        update_enabled: bool = False,
        symmetry_tolerance: float = SYMMETRY_TOLERANCE,
        psd_tolerance: float = PSD_TOLERANCE,
        max_condition_number: float = MAX_CONDITION_NUMBER,
    ) -> InnovationClipRLSPredictor:
        if state.get("format_version") != cls.format_version:
            raise ValueError("unsupported innovation-clip checkpoint")
        base_state = dict(state)
        base_state["format_version"] = OnlineRLSPredictor.format_version
        return cls.from_model(
            OnlineRLSPredictor.from_state_dict(
                base_state,
                name=name or str(state.get("name", cls.name)),
                update_enabled=update_enabled,
                symmetry_tolerance=symmetry_tolerance,
                psd_tolerance=psd_tolerance,
                max_condition_number=max_condition_number,
            ),
            name=name or str(state.get("name", cls.name)),
            update_enabled=update_enabled,
            candidate_id=str(state["candidate_id"]),
            max_normalized_innovation=float(state["max_normalized_innovation"]),
            symmetry_tolerance=symmetry_tolerance,
            psd_tolerance=psd_tolerance,
            max_condition_number=max_condition_number,
        )


def selected_candidate_from_freeze() -> str:
    """Resolve the selected ID from the committed confirmation freeze only."""

    path = load_protocol().artifacts["freeze_manifest"]
    freeze_path = Path(path)
    if not freeze_path.is_absolute():
        freeze_path = Path(__file__).resolve().parents[2] / freeze_path
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    candidate_id = payload.get("selected_candidate")
    if not isinstance(candidate_id, str) or candidate_id not in candidate_catalog():
        raise ValueError("confirmation freeze does not contain a catalogued selected candidate")
    return candidate_id
