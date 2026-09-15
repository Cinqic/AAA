"""Strict executable protocol for the observation-noise phase.

This module deliberately does not import the v2.1 specification or mutate its
meaning.  The noise phase has its own protocol identity, command surface,
artifact namespace, and freeze files.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "aaa.observation_noise.v1.1"
_REQUIRED_TOP_LEVEL = {
    "protocol_version",
    "status",
    "title",
    "reference",
    "observation_model",
    "noise_channels",
    "families",
    "schedules",
    "training",
    "baselines",
    "replication",
    "statistics",
    "acceptance",
    "search_budget",
    "artifacts",
    "seed_namespaces",
    "compute_storage",
    "hypotheses",
    "endpoints",
}


class ProtocolError(ValueError):
    """Raised when the executable protocol is incomplete or contradictory."""


def _strict(value: Mapping[str, Any], expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{path}: expected an object")
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ProtocolError(f"{path}: unknown={unknown}, missing={missing}")
    return dict(value)


def _number(value: Any, path: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{path}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError(f"{path}: must be finite")
    if positive and number <= 0:
        raise ProtocolError(f"{path}: must be positive")
    if nonnegative and number < 0:
        raise ProtocolError(f"{path}: must be non-negative")
    return number


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{path}: expected integer >= {minimum}")
    return value


@dataclass(frozen=True)
class NoiseChannel:
    name: str
    definition: str
    variance: str
    primary: bool
    stress: bool


@dataclass(frozen=True)
class NoiseProtocol:
    """Validated protocol with the complete JSON retained for audit output."""

    raw: dict[str, Any]
    channels: tuple[NoiseChannel, ...]

    @property
    def protocol_version(self) -> str:
        return str(self.raw["protocol_version"])

    @property
    def status(self) -> str:
        return str(self.raw["status"])

    @property
    def reference(self) -> Mapping[str, Any]:
        return self.raw["reference"]

    @property
    def observation_model(self) -> Mapping[str, Any]:
        return self.raw["observation_model"]

    @property
    def schedules(self) -> Mapping[str, Any]:
        return self.raw["schedules"]

    @property
    def training(self) -> Mapping[str, Any]:
        return self.raw["training"]

    @property
    def baselines(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.raw["baselines"])

    @property
    def replication(self) -> Mapping[str, Any]:
        return self.raw["replication"]

    @property
    def statistics(self) -> Mapping[str, Any]:
        return self.raw["statistics"]

    @property
    def acceptance(self) -> Mapping[str, Any]:
        return self.raw["acceptance"]

    @property
    def artifacts(self) -> Mapping[str, Any]:
        return self.raw["artifacts"]

    @property
    def seed_namespaces(self) -> Mapping[str, str]:
        return self.raw["seed_namespaces"]

    @property
    def families(self) -> Mapping[str, Any]:
        return self.raw["families"]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.raw, sort_keys=True))

    def hash(self) -> str:
        encoded = json.dumps(self.raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate(raw_value: Mapping[str, Any]) -> NoiseProtocol:
    raw = _strict(raw_value, _REQUIRED_TOP_LEVEL, "protocol")
    if raw["protocol_version"] != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol.protocol_version must be {PROTOCOL_VERSION!r}")
    if raw["status"] not in {"design_frozen", "confirmation_frozen", "executed"}:
        raise ProtocolError("protocol.status is not an executable lifecycle status")

    reference = _strict(
        raw["reference"],
        {
            "v2_1_commit",
            "v2_1_spec_path",
            "v2_1_raw_sha256",
            "v2_1_resolved_sha256",
            "zero_noise_tolerance",
        },
        "protocol.reference",
    )
    for key in ("v2_1_commit", "v2_1_spec_path", "v2_1_raw_sha256", "v2_1_resolved_sha256"):
        if not isinstance(reference[key], str) or not reference[key]:
            raise ProtocolError(f"protocol.reference.{key}: required nonempty string")
    if len(reference["v2_1_commit"]) != 40 or any(
        character not in "0123456789abcdef" for character in reference["v2_1_commit"]
    ):
        raise ProtocolError("protocol.reference.v2_1_commit must be a lowercase 40-character Git SHA")
    _number(reference["zero_noise_tolerance"], "protocol.reference.zero_noise_tolerance", nonnegative=True)

    observation = _strict(
        raw["observation_model"],
        {
            "equation",
            "latent_truth_field",
            "raw_observation_field",
            "predictor_visible_field",
            "no_clipping_or_reflection_of_sensor",
            "one_sample_per_timestamp",
            "idempotent_repeated_observation",
            "update_target",
            "primary_channels",
            "primary_scales",
            "stress_channels",
            "all_scales",
        },
        "protocol.observation_model",
    )
    if observation["equation"] != "y_t = x_t + L * eta_t":
        raise ProtocolError("protocol.observation_model.equation is not the declared observation model")
    for key in (
        "no_clipping_or_reflection_of_sensor",
        "one_sample_per_timestamp",
        "idempotent_repeated_observation",
    ):
        if observation[key] is not True:
            raise ProtocolError(f"protocol.observation_model.{key} must be true")
    if observation["update_target"] != "newly_revealed_noisy_observation":
        raise ProtocolError("predictors must update from the newly revealed noisy observation")
    all_scales = tuple(
        _number(item, "protocol.observation_model.all_scales", nonnegative=True)
        for item in observation["all_scales"]
    )
    if all_scales != (0.0, 0.0005, 0.002, 0.01):
        raise ProtocolError("protocol.observation_model.all_scales must be the frozen four-level scale set")
    for key in ("latent_truth_field", "raw_observation_field", "predictor_visible_field"):
        if not isinstance(observation[key], str) or not observation[key]:
            raise ProtocolError(f"protocol.observation_model.{key}: required nonempty string")

    channels: list[NoiseChannel] = []
    channel_map = raw["noise_channels"]
    if not isinstance(channel_map, Mapping) or set(channel_map) != {
        "gaussian",
        "uniform",
        "correlated",
        "impulsive",
    }:
        raise ProtocolError("protocol.noise_channels must contain exactly the four frozen channels")
    for name in ("gaussian", "uniform", "correlated", "impulsive"):
        item = _strict(
            channel_map[name],
            {"definition", "variance", "primary", "stress"},
            f"protocol.noise_channels.{name}",
        )
        if not isinstance(item["definition"], str) or not isinstance(item["variance"], str):
            raise ProtocolError(f"protocol.noise_channels.{name}: definitions must be strings")
        if not isinstance(item["primary"], bool) or not isinstance(item["stress"], bool):
            raise ProtocolError(f"protocol.noise_channels.{name}: primary/stress must be booleans")
        channels.append(
            NoiseChannel(name, item["definition"], item["variance"], item["primary"], item["stress"])
        )
    if tuple(observation["primary_channels"]) != ("gaussian", "uniform"):
        raise ProtocolError("primary channel declaration drifted")
    if tuple(observation["stress_channels"]) != ("correlated", "impulsive"):
        raise ProtocolError("stress channel declaration drifted")
    if tuple(observation["primary_scales"]) != (0.0005, 0.002):
        raise ProtocolError("primary scale declaration drifted")

    families = raw["families"]
    if not isinstance(families, Mapping) or set(families) != {
        "constant_velocity",
        "bouncing",
        "speed_change",
        "changed_law",
    }:
        raise ProtocolError("protocol.families must cover the four declared latent environment families")
    for name, item in families.items():
        data = _strict(
            item,
            {"simulator_family", "steps", "prefix_steps", "branch_steps", "stratified", "description"},
            f"protocol.families.{name}",
        )
        if not isinstance(data["simulator_family"], str) or not isinstance(data["description"], str):
            raise ProtocolError(f"protocol.families.{name}: description fields must be strings")
        _integer(data["steps"], f"protocol.families.{name}.steps", minimum=1)
        _integer(data["prefix_steps"], f"protocol.families.{name}.prefix_steps", minimum=0)
        _integer(data["branch_steps"], f"protocol.families.{name}.branch_steps", minimum=0)
        if not isinstance(data["stratified"], bool):
            raise ProtocolError(f"protocol.families.{name}.stratified must be boolean")
    if families["changed_law"]["prefix_steps"] != 300 or families["changed_law"]["branch_steps"] != 100:
        raise ProtocolError("changed-law timing must remain 300 + 100")

    schedules = _strict(
        raw["schedules"],
        {
            "generator",
            "stream_derivation",
            "floating_point_encoding",
            "stationary",
            "sensor_shifts",
            "event_times",
            "correlated_shift_rule",
            "pairing",
        },
        "protocol.schedules",
    )
    for key in (
        "generator",
        "stream_derivation",
        "floating_point_encoding",
        "correlated_shift_rule",
        "pairing",
    ):
        if not isinstance(schedules[key], str) or not schedules[key]:
            raise ProtocolError(f"protocol.schedules.{key}: required nonempty string")
    stationary = _strict(
        schedules["stationary"],
        {"channels", "scales", "save_actual_values", "hash_before_execution"},
        "protocol.schedules.stationary",
    )
    if tuple(stationary["channels"]) != ("gaussian", "uniform", "correlated", "impulsive") or tuple(
        stationary["scales"]
    ) != (0.0, 0.0005, 0.002, 0.01):
        raise ProtocolError("stationary schedule declaration drifted")
    if stationary["save_actual_values"] is not True or stationary["hash_before_execution"] is not True:
        raise ProtocolError("actual schedules must be saved and hashed before execution")
    shifts = schedules["sensor_shifts"]
    if shifts != [{"from": 0.0005, "to": 0.002}, {"from": 0.002, "to": 0.0005}]:
        raise ProtocolError("both predeclared sensor shifts are required")
    event_times = _strict(
        schedules["event_times"],
        {"law_change", "sensor_aligned", "sensor_staggered", "post_event_window"},
        "protocol.schedules.event_times",
    )
    if event_times != {
        "law_change": 300,
        "sensor_aligned": 300,
        "sensor_staggered": 340,
        "post_event_window": 50,
    }:
        raise ProtocolError("event timing declaration drifted")

    training = _strict(
        raw["training"],
        {
            "conditions",
            "noise_mixture",
            "episodes_per_lineage",
            "updates_per_episode",
            "calibration_policy",
            "calibration_allocation",
            "freeze_parameters_vs_filter_state",
        },
        "protocol.training",
    )
    if tuple(training["conditions"]) != ("clean_trained", "noise_trained"):
        raise ProtocolError("both clean-trained and noise-trained conditions are required")
    if not isinstance(training["noise_mixture"], Mapping) or set(training["noise_mixture"]) != {
        "gaussian",
        "uniform",
    }:
        raise ProtocolError("noise-trained mixture must be fixed to the two primary channels")
    for channel, levels in training["noise_mixture"].items():
        if tuple(levels) != (0.0005, 0.002):
            raise ProtocolError(f"training mixture for {channel} drifted")
    _integer(training["episodes_per_lineage"], "protocol.training.episodes_per_lineage", minimum=1)
    _integer(training["updates_per_episode"], "protocol.training.updates_per_episode", minimum=1)
    for key in ("calibration_policy", "freeze_parameters_vs_filter_state"):
        if not isinstance(training[key], str) or not training[key]:
            raise ProtocolError(f"protocol.training.{key}: required nonempty string")
    if not isinstance(training["calibration_allocation"], str) or not training["calibration_allocation"]:
        raise ProtocolError("protocol.training.calibration_allocation: required nonempty string")

    baselines = raw["baselines"]
    required_baselines = {
        "persistence",
        "constant_motion",
        "constant_motion_reflected",
        "causal_smoothing_motion",
        "alpha_beta_filter",
        "incumbent_square_root_rls",
        "no_learning_control",
        "primary_deployable_baseline",
    }
    if set(baselines) != required_baselines:
        raise ProtocolError(f"protocol.baselines must contain exactly {sorted(required_baselines)}")
    if any(not isinstance(item, str) or not item for item in baselines.values()):
        raise ProtocolError("protocol.baselines descriptions must be nonempty strings")

    replication = _strict(
        raw["replication"],
        {
            "lineages",
            "episodes_per_family_per_lineage",
            "sensor_realizations_per_episode",
            "existing_strata",
            "a_batch",
            "b_batch",
            "shared_training_lineages",
            "disjoint_evaluation_streams",
            "independent_retraining_required_for_claim",
        },
        "protocol.replication",
    )
    for key in (
        "lineages",
        "episodes_per_family_per_lineage",
        "sensor_realizations_per_episode",
        "existing_strata",
    ):
        _integer(replication[key], f"protocol.replication.{key}", minimum=1)
    if (
        replication["lineages"] != 10
        or replication["episodes_per_family_per_lineage"] != 32
        or replication["sensor_realizations_per_episode"] != 3
        or replication["existing_strata"] != 32
    ):
        raise ProtocolError("minimum replication plan drifted")
    for key in (
        "shared_training_lineages",
        "disjoint_evaluation_streams",
        "independent_retraining_required_for_claim",
    ):
        if not isinstance(replication[key], bool):
            raise ProtocolError(f"protocol.replication.{key} must be boolean")
    for key in ("a_batch", "b_batch"):
        _strict(replication[key], {"id", "namespace", "role"}, f"protocol.replication.{key}")

    statistics = _strict(
        raw["statistics"],
        {
            "primary_metric",
            "secondary_metrics",
            "resampling_units",
            "draws",
            "intervals",
            "multiplicity_family",
            "null_validation",
            "undefined_ratio_rule",
        },
        "protocol.statistics",
    )
    if statistics["primary_metric"] != "latent_next_position_mae_normalized_by_line_length":
        raise ProtocolError("primary metric declaration drifted")
    if tuple(statistics["resampling_units"]) != ("training_lineage", "latent_episode", "sensor_realization"):
        raise ProtocolError("hierarchical resampling units drifted")
    _integer(statistics["draws"], "protocol.statistics.draws", minimum=1000)
    if tuple(statistics["intervals"]) != (0.90, 0.95):
        raise ProtocolError("both interval levels are required")
    for key in ("multiplicity_family", "null_validation", "undefined_ratio_rule"):
        if not isinstance(statistics[key], str) or not statistics[key]:
            raise ProtocolError(f"protocol.statistics.{key}: required nonempty string")
    if not isinstance(statistics["secondary_metrics"], list) or not statistics["secondary_metrics"]:
        raise ProtocolError("secondary metrics must be explicitly listed")

    acceptance = _strict(
        raw["acceptance"],
        {
            "reference_preservation",
            "evidence_integrity",
            "numerical_stability",
            "moderate_noise_adaptation",
            "baseline_competitiveness",
            "unchanged_law_retention",
            "absolute_utility",
            "added_mechanism_promotion",
            "outcome_vocabulary",
            "machine_criteria",
        },
        "protocol.acceptance",
    )
    for key in (
        "reference_preservation",
        "evidence_integrity",
        "numerical_stability",
        "moderate_noise_adaptation",
        "baseline_competitiveness",
        "unchanged_law_retention",
        "absolute_utility",
        "added_mechanism_promotion",
    ):
        if not isinstance(acceptance[key], str) or not acceptance[key]:
            raise ProtocolError(f"protocol.acceptance.{key}: required rule")
    if tuple(acceptance["outcome_vocabulary"]) != (
        "engineering_complete",
        "scientifically_supported",
        "negative",
        "inconclusive",
        "blocked_by_missing_evidence",
    ):
        raise ProtocolError("outcome vocabulary drifted")
    criteria = acceptance["machine_criteria"]
    if not isinstance(criteria, Mapping) or set(criteria) != {
        "absolute_utility",
        "baseline_competitiveness",
        "moderate_noise_adaptation",
        "unchanged_law_retention",
        "added_mechanism_promotion",
    }:
        raise ProtocolError("protocol.acceptance.machine_criteria is incomplete")
    required_criteria_fields = {
        "estimate",
        "bound",
        "confidence_level",
        "adjustment",
        "operator",
        "threshold",
    }
    for name, criterion in criteria.items():
        if not isinstance(criterion, Mapping) or not required_criteria_fields.issubset(criterion):
            raise ProtocolError(f"protocol.acceptance.machine_criteria.{name} is incomplete")
        if not isinstance(criterion["estimate"], str) or not criterion["estimate"]:
            raise ProtocolError(f"protocol.acceptance.machine_criteria.{name}.estimate is required")
        if criterion["bound"] not in {"upper_confidence_bound", "lower_confidence_bound"}:
            raise ProtocolError(f"protocol.acceptance.machine_criteria.{name}.bound is invalid")
        _number(
            criterion["confidence_level"],
            f"protocol.acceptance.machine_criteria.{name}.confidence_level",
        )
        if criterion["confidence_level"] != 0.95:
            raise ProtocolError(f"protocol.acceptance.machine_criteria.{name} must use 95% bounds")
        if not isinstance(criterion["adjustment"], str) or "Holm once" not in criterion["adjustment"]:
            raise ProtocolError(
                f"protocol.acceptance.machine_criteria.{name} must declare joint Holm adjustment"
            )
        if criterion["operator"] not in {"<=", ">"}:
            raise ProtocolError(f"protocol.acceptance.machine_criteria.{name}.operator is invalid")
        _number(
            criterion["threshold"], f"protocol.acceptance.machine_criteria.{name}.threshold", nonnegative=True
        )

    search = _strict(
        raw["search_budget"],
        {
            "max_candidate_configurations",
            "max_substantive_mechanism_changes",
            "baseline_search_separate",
            "candidate_ledger_file",
            "development_selection_plan",
            "design_freeze",
            "confirmation_freeze",
        },
        "protocol.search_budget",
    )
    if (
        search["max_candidate_configurations"] != 12
        or search["max_substantive_mechanism_changes"] != 3
        or search["baseline_search_separate"] is not True
    ):
        raise ProtocolError("candidate search budget drifted")
    for key in (
        "candidate_ledger_file",
        "development_selection_plan",
        "design_freeze",
        "confirmation_freeze",
    ):
        if not isinstance(search[key], str) or not search[key]:
            raise ProtocolError(f"protocol.search_budget.{key}: required path")

    artifacts = _strict(
        raw["artifacts"],
        {
            "root",
            "protocol",
            "schedule_dir",
            "records",
            "reports",
            "registry",
            "freeze_manifest",
            "source_freeze_manifest",
            "checksum_manifest",
            "candidate_ledger",
            "full_archive_required",
            "durable_storage_policy",
            "raw_record_encoding",
        },
        "protocol.artifacts",
    )
    for key, value in artifacts.items():
        if not isinstance(value, (str, bool)) or value == "":
            raise ProtocolError(f"protocol.artifacts.{key}: invalid value")
    if artifacts["full_archive_required"] is not True:
        raise ProtocolError("full archive is mandatory for an independent-verification claim")

    namespaces = raw["seed_namespaces"]
    if not isinstance(namespaces, Mapping) or set(namespaces) != {
        "training",
        "development",
        "calibration",
        "confirmation_a",
        "confirmation_b",
        "generator_validation",
    }:
        raise ProtocolError("all disjoint seed namespaces must be declared")
    if any(not isinstance(value, str) or not value for value in namespaces.values()):
        raise ProtocolError("seed namespace labels must be nonempty strings")

    compute = _strict(
        raw["compute_storage"],
        {"platform", "lock_required", "estimate", "raw_record_encoding", "retention"},
        "protocol.compute_storage",
    )
    for key in ("platform", "estimate", "raw_record_encoding", "retention"):
        if not isinstance(compute[key], str) or not compute[key]:
            raise ProtocolError(f"protocol.compute_storage.{key}: required nonempty string")
    if compute["lock_required"] is not True:
        raise ProtocolError("formal execution must require the repository lock")
    for section in ("hypotheses", "endpoints"):
        if (
            not isinstance(raw[section], Mapping)
            or not raw[section]
            or any(not isinstance(v, str) or not v for v in raw[section].values())
        ):
            raise ProtocolError(f"protocol.{section}: every declared item needs readable text")
    return NoiseProtocol(raw=dict(raw), channels=tuple(channels))


def canonical_protocol_path() -> Path:
    path = Path(__file__).resolve().parent / "data" / "observation_noise_v1.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_protocol(path: str | Path | None = None) -> NoiseProtocol:
    source = Path(path) if path is not None else canonical_protocol_path()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProtocolError(f"{source}: malformed JSON: {error}") from error
    return _validate(raw)


def canonical_protocol_hash() -> str:
    return load_protocol().hash()
