"""Independent joint A+B evaluator for observation-noise confirmation."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .candidates import candidate_definition
from .spec import canonical_protocol_hash, load_protocol
from .statistics import holm_bonferroni
from .verifier import VerificationError, iter_records, verify_attempt

PRIMARY_CHANNELS = ("gaussian", "uniform")
PRIMARY_SCALES = (0.0005, 0.002)
PRIMARY_FAMILIES = ("constant_velocity", "bouncing", "speed_change", "changed_law")
UNCHANGED_FAMILIES = ("constant_velocity", "bouncing")
MANDATORY_PREDICTORS = {
    "persistence",
    "constant_motion",
    "constant_motion_reflected",
    "causal_smoothing_motion",
    "alpha_beta_filter",
    "incumbent_frozen",
    "incumbent_square_root_rls",
    "no_learning_control",
    "selected_candidate_frozen",
    "selected_candidate_online",
}


class JointEvaluationError(ValueError):
    """Raised when A+B archives cannot support a joint claim."""


@dataclass(frozen=True)
class _Claim:
    claim_id: str
    batch_id: str
    endpoint: str
    cell: dict[str, Any]
    values: np.ndarray
    threshold: float
    direction: str
    denominator: np.ndarray | None = None


def _dense(values: dict[tuple[int, int, int], float], label: str) -> np.ndarray:
    if not values:
        raise JointEvaluationError(f"{label}: no paired hierarchy values")
    lineages = sorted({identity[0] for identity in values})
    episodes = sorted({identity[1] for identity in values})
    realizations = sorted({identity[2] for identity in values})
    if lineages != list(range(10)) or episodes != list(range(32)) or realizations != list(range(3)):
        raise JointEvaluationError(f"{label}: expected 10 lineages x 32 episodes x 3 realizations")
    expected = {
        (lineage, episode, realization)
        for lineage in lineages
        for episode in episodes
        for realization in realizations
    }
    if set(values) != expected:
        raise JointEvaluationError(f"{label}: hierarchy has missing or unexpected identities")
    return np.asarray(
        [
            [
                [values[(lineage, episode, realization)] for realization in realizations]
                for episode in episodes
            ]
            for lineage in lineages
        ],
        dtype=float,
    )


def _hierarchical_draws(array: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    if array.shape != (10, 32, 3):
        raise JointEvaluationError(f"hierarchical array has wrong shape: {array.shape}")
    selected_lineages = rng.integers(0, 10, size=(draws, 10))
    selected_episodes = rng.integers(0, 32, size=(draws, 10, 32))
    selected_realizations = rng.integers(0, 3, size=(draws, 10, 32, 3))
    sampled = array[
        selected_lineages[:, :, None, None],
        selected_episodes[:, :, :, None],
        selected_realizations,
    ]
    return np.mean(np.mean(np.mean(sampled, axis=3), axis=2), axis=1)


def _ratio_of_means(numerator: np.ndarray, denominator: np.ndarray) -> float:
    if numerator.shape != denominator.shape or numerator.shape != (10, 32, 3):
        raise JointEvaluationError("ratio-of-means arrays must share the frozen 10 x 32 x 3 hierarchy")
    denominator_mean = float(np.mean(denominator))
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all() or denominator_mean == 0.0:
        raise JointEvaluationError("adaptation ratio has non-finite values or a zero mean denominator")
    return float(np.mean(numerator) / denominator_mean)


def _hierarchical_ratio_draws(
    numerator: np.ndarray,
    denominator: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap the declared ratio of means with shared hierarchical indices."""

    if numerator.shape != denominator.shape or numerator.shape != (10, 32, 3):
        raise JointEvaluationError("ratio-of-means arrays must share the frozen 10 x 32 x 3 hierarchy")
    selected_lineages = rng.integers(0, 10, size=(draws, 10))
    selected_episodes = rng.integers(0, 32, size=(draws, 10, 32))
    selected_realizations = rng.integers(0, 3, size=(draws, 10, 32, 3))
    indices = (
        selected_lineages[:, :, None, None],
        selected_episodes[:, :, :, None],
        selected_realizations,
    )
    numerator_means = np.mean(numerator[indices], axis=(1, 2, 3))
    denominator_means = np.mean(denominator[indices], axis=(1, 2, 3))
    if not np.isfinite(numerator_means).all() or not np.isfinite(denominator_means).all():
        raise JointEvaluationError("adaptation bootstrap produced non-finite means")
    if np.any(denominator_means == 0.0):
        raise JointEvaluationError("adaptation bootstrap produced a zero mean denominator")
    return numerator_means / denominator_means


def _trial_values(
    records: Any,
    predictor: str,
    *,
    branch: str = "stationary",
    first50: bool = False,
) -> dict[tuple[str, str, str, float], dict[tuple[int, int, int], float]]:
    grouped: dict[
        tuple[str, str, str, float],
        dict[tuple[int, int, int], list[float]],
    ] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for record in records:
        trial = record["trial"]
        if trial["branch"] != branch or predictor not in record["predictions"]:
            continue
        if first50 and not 300 <= int(record["step"]) < 350:
            continue
        cell = (
            str(trial["condition"]),
            str(trial["family"]),
            str(trial["channel"]),
            float(trial["scale"]),
        )
        identity = (int(trial["lineage"]), int(trial["episode"]), int(trial["realization"]))
        bucket = grouped[cell][identity]
        bucket[0] += float(record["predictions"][predictor]["latent_normalized_absolute_error"])
        bucket[1] += 1.0
    return {
        cell: {identity: values[0] / values[1] for identity, values in identities.items()}
        for cell, identities in grouped.items()
    }


def _archive(path: str | Path, expected_role: str, expected_batch: str) -> dict[str, Any]:
    directory = Path(path)
    try:
        verification = verify_attempt(directory)
    except (VerificationError, OSError, json.JSONDecodeError) as exc:
        raise JointEvaluationError(f"{directory}: independent archive verification failed: {exc}") from exc
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if metadata.get("role") != expected_role or metadata.get("batch_id") != expected_batch:
        raise JointEvaluationError(
            f"{directory}: role or batch identity does not match the requested archive"
        )
    if (
        metadata.get("protocol_hash") != canonical_protocol_hash()
        or manifest.get("protocol_hash") != canonical_protocol_hash()
    ):
        raise JointEvaluationError(f"{directory}: protocol hash is not canonical")
    candidate_id = metadata.get("selected_candidate")
    if not isinstance(candidate_id, str):
        raise JointEvaluationError(f"{directory}: selected candidate identity is missing")
    try:
        definition = candidate_definition(candidate_id)
    except ValueError as exc:
        raise JointEvaluationError(str(exc)) from exc
    if metadata.get("selected_candidate_configuration_hash") != definition.configuration_hash:
        raise JointEvaluationError(f"{directory}: selected candidate configuration hash is not canonical")
    stationary_record = next(
        (record for record in iter_records(directory) if record["trial"]["branch"] == "stationary"), None
    )
    if stationary_record is None:
        raise JointEvaluationError(f"{directory}: archive has no primitive records")
    stationary_predictors = set(stationary_record["predictions"])
    if not MANDATORY_PREDICTORS.issubset(stationary_predictors):
        raise JointEvaluationError(f"{directory}: mandatory predictors are missing")
    return {
        "directory": directory,
        "verification": verification,
        "metadata": metadata,
        "manifest": manifest,
        "summary": summary,
        "candidate_id": candidate_id,
        "candidate_definition": definition,
    }


def _claims_for_archive(archive: dict[str, Any], protocol: Any) -> list[_Claim]:
    batch_id = str(archive["metadata"]["batch_id"])
    directory = archive["directory"]
    candidate = "selected_candidate_online"
    candidate_values = _trial_values(iter_records(directory), candidate)
    incumbent_values = _trial_values(iter_records(directory), "incumbent_square_root_rls")
    baseline_values = _trial_values(iter_records(directory), "constant_motion_reflected")
    frozen_values = _trial_values(iter_records(directory), "selected_candidate_frozen")
    online_branch_values = _trial_values(
        iter_records(directory), "selected_candidate_branch_online", branch="online", first50=True
    )
    frozen_branch_values = _trial_values(
        iter_records(directory), "selected_candidate_branch_frozen", branch="frozen", first50=True
    )
    claims: list[_Claim] = []
    criteria = protocol.acceptance["machine_criteria"]
    for condition in ("clean_trained", "noise_trained"):
        for family in PRIMARY_FAMILIES:
            for channel in PRIMARY_CHANNELS:
                for scale in PRIMARY_SCALES:
                    cell = {
                        "condition": condition,
                        "family": family,
                        "channel": channel,
                        "scale": scale,
                        "branch": "stationary",
                    }
                    key = (condition, family, channel, scale)
                    candidate_array = _dense(candidate_values.get(key, {}), f"{batch_id} candidate {key}")
                    baseline_array = _dense(baseline_values.get(key, {}), f"{batch_id} baseline {key}")
                    incumbent_array = _dense(incumbent_values.get(key, {}), f"{batch_id} incumbent {key}")
                    claims.append(
                        _Claim(
                            f"{batch_id}|absolute_utility|{condition}|{family}|{channel}|{scale:.7g}",
                            batch_id,
                            "absolute_utility",
                            cell,
                            candidate_array,
                            float(criteria["absolute_utility"]["threshold"]),
                            "upper",
                        )
                    )
                    if archive["candidate_definition"].mechanism_change_count > 0:
                        claims.append(
                            _Claim(
                                f"{batch_id}|added_mechanism_promotion|{condition}|{family}|{channel}|{scale:.7g}",
                                batch_id,
                                "added_mechanism_promotion",
                                cell,
                                incumbent_array - candidate_array,
                                float(criteria["added_mechanism_promotion"]["threshold"]),
                                "lower",
                            )
                        )
                    claims.append(
                        _Claim(
                            f"{batch_id}|baseline_competitiveness|{condition}|{family}|{channel}|{scale:.7g}",
                            batch_id,
                            "baseline_competitiveness",
                            cell,
                            candidate_array - 1.10 * baseline_array,
                            float(criteria["baseline_competitiveness"]["threshold"]),
                            "upper",
                        )
                    )
                    if family in UNCHANGED_FAMILIES:
                        frozen_array = _dense(frozen_values.get(key, {}), f"{batch_id} frozen {key}")
                        claims.append(
                            _Claim(
                                f"{batch_id}|unchanged_law_retention|{condition}|{family}|{channel}|{scale:.7g}",
                                batch_id,
                                "unchanged_law_retention",
                                cell,
                                candidate_array - frozen_array,
                                float(criteria["unchanged_law_retention"]["threshold"]),
                                "upper",
                            )
                        )
                    if family == "changed_law":
                        online_array = _dense(
                            online_branch_values.get(key, {}), f"{batch_id} branch online {key}"
                        )
                        frozen_array = _dense(
                            frozen_branch_values.get(key, {}), f"{batch_id} branch frozen {key}"
                        )
                        claims.append(
                            _Claim(
                                f"{batch_id}|moderate_noise_adaptation|{condition}|{family}|{channel}|{scale:.7g}",
                                batch_id,
                                "moderate_noise_adaptation",
                                {**cell, "branch": "first50_online_vs_frozen"},
                                frozen_array - online_array,
                                float(criteria["moderate_noise_adaptation"]["threshold"]),
                                "lower",
                                frozen_array,
                            )
                        )
    return claims


def _formal_batch_ids() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    freeze_path = root / "benchmarks" / "observation_noise_freeze.json"
    registry_path = root / "benchmarks" / "observation_noise_registry.json"
    if not freeze_path.is_file():
        raise JointEvaluationError("formal evaluation requires the committed confirmation freeze")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if (
        freeze.get("stage") != "confirmation_freeze"
        or freeze.get("protocol_hash") != canonical_protocol_hash()
        or registry.get("protocol_hash") != canonical_protocol_hash()
    ):
        raise JointEvaluationError("confirmation freeze or registry identity is not canonical")
    frozen_batches = freeze.get("planned_batches")
    if not isinstance(frozen_batches, list) or len(frozen_batches) != 2:
        raise JointEvaluationError("confirmation freeze must name exactly one A and one B batch")
    resolved: dict[str, str] = {}
    for batch_id in frozen_batches:
        matches = [
            row
            for row in registry.get("batches", [])
            if isinstance(row, dict) and row.get("batch_id") == batch_id
        ]
        if len(matches) != 1 or matches[0].get("role") not in {"confirmation_a", "confirmation_b"}:
            raise JointEvaluationError(f"frozen batch {batch_id!r} is not uniquely declared")
        role = str(matches[0]["role"])
        if role in resolved:
            raise JointEvaluationError(f"confirmation freeze declares multiple batches for {role}")
        resolved[role] = str(batch_id)
    if set(resolved) != {"confirmation_a", "confirmation_b"}:
        raise JointEvaluationError("confirmation freeze does not declare both A and B roles")
    return resolved


def evaluate_joint_archives(
    archive_a: str | Path,
    archive_b: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify A and B independently, then reconstruct one joint claim family."""

    protocol = load_protocol()
    draw_count = int(protocol.statistics["draws"])
    batches = _formal_batch_ids()
    a = _archive(archive_a, "confirmation_a", batches["confirmation_a"])
    b = _archive(archive_b, "confirmation_b", batches["confirmation_b"])
    for archive in (a, b):
        statistics = archive["summary"].get("statistics", {})
        if statistics.get("declared_draws") != draw_count or statistics.get("executed_draws") != draw_count:
            raise JointEvaluationError(
                f"{archive['directory']}: formal archive did not execute the frozen draw count"
            )
    if a["candidate_id"] != b["candidate_id"]:
        raise JointEvaluationError("A and B selected candidate IDs differ")
    if a["metadata"].get("selected_candidate_configuration_hash") != b["metadata"].get(
        "selected_candidate_configuration_hash"
    ):
        raise JointEvaluationError("A and B candidate configuration hashes differ")
    if a["metadata"].get("scientific_fingerprint_sha256") != b["metadata"].get(
        "scientific_fingerprint_sha256"
    ):
        raise JointEvaluationError("A and B scientific fingerprints differ")
    checkpoints_a = sorted((a["directory"] / "checkpoints").glob("*.json"))
    checkpoints_b = sorted((b["directory"] / "checkpoints").glob("*.json"))
    if [path.name for path in checkpoints_a] != [path.name for path in checkpoints_b] or any(
        left.read_bytes() != right.read_bytes()
        for left, right in zip(checkpoints_a, checkpoints_b, strict=True)
    ):
        raise JointEvaluationError("A and B shared training checkpoint archives differ")
    claims = _claims_for_archive(a, protocol) + _claims_for_archive(b, protocol)
    rng = np.random.default_rng(derive_joint_seed(protocol))
    rows: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    samples: dict[str, np.ndarray] = {}
    for claim in claims:
        sample = (
            _hierarchical_draws(claim.values, draw_count, rng)
            if claim.denominator is None
            else _hierarchical_ratio_draws(claim.values, claim.denominator, draw_count, rng)
        )
        raw_p = (
            float((np.count_nonzero(sample >= claim.threshold) + 1) / (draw_count + 1))
            if claim.direction == "upper"
            else float((np.count_nonzero(sample <= claim.threshold) + 1) / (draw_count + 1))
        )
        raw_pvalues[claim.claim_id] = raw_p
        samples[claim.claim_id] = sample
        rows[claim.claim_id] = {
            "batch_id": claim.batch_id,
            "endpoint": claim.endpoint,
            "cell": claim.cell,
            "estimand": "hierarchical_mean_of_lineage_episode_sensor_realization_values",
            "point": (
                float(np.mean(claim.values))
                if claim.denominator is None
                else _ratio_of_means(claim.values, claim.denominator)
            ),
            "threshold": claim.threshold,
            "direction": claim.direction,
            "draws": draw_count,
            "raw_p_value": raw_p,
            "candidate_id": a["candidate_id"],
            "evidence_status": "PASS",
        }
    adjusted = holm_bonferroni(raw_pvalues)
    ordered = sorted(raw_pvalues, key=lambda key: (raw_pvalues[key], key))
    family_size = len(ordered)
    for rank, claim_id in enumerate(ordered, 1):
        row = rows[claim_id]
        alpha = 0.05 / (family_size - rank + 1)
        sample = samples[claim_id]
        if row["direction"] == "upper":
            bound = float(np.quantile(sample, 1.0 - alpha, method="linear"))
            passed = bound <= row["threshold"] and adjusted[claim_id] <= 0.05
        else:
            bound = float(np.quantile(sample, alpha, method="linear"))
            passed = bound > row["threshold"] and adjusted[claim_id] <= 0.05
        row.update(
            {
                "holm_rank": rank,
                "family_size": family_size,
                "holm_adjusted_p_value": adjusted[claim_id],
                "adjusted_alpha": alpha,
                "adjusted_bound": bound,
                "status": "PASS" if passed else "FAIL",
            }
        )
    gates: list[dict[str, Any]] = []
    for endpoint in (
        "absolute_utility",
        "baseline_competitiveness",
        "moderate_noise_adaptation",
        "unchanged_law_retention",
    ):
        endpoint_rows = [row for row in rows.values() if row["endpoint"] == endpoint]
        gates.append(
            {
                "name": endpoint,
                "required": True,
                "status": "PASS"
                if endpoint_rows and all(row["status"] == "PASS" for row in endpoint_rows)
                else "FAIL",
                "claim_count": len(endpoint_rows),
                "detail": "Every enumerated primary A+B claim passed the joint Holm-adjusted bound."
                if endpoint_rows and all(row["status"] == "PASS" for row in endpoint_rows)
                else "At least one enumerated primary A+B claim failed its preregistered bound.",
            }
        )
    definition = a["candidate_definition"]
    promotion_rows = [row for row in rows.values() if row["endpoint"] == "added_mechanism_promotion"]
    promotion_passed = bool(promotion_rows) and all(row["status"] == "PASS" for row in promotion_rows)
    gates.append(
        {
            "name": "added_mechanism_promotion",
            "required": definition.mechanism_change_count > 0,
            "status": (
                "NOT_APPLICABLE"
                if definition.mechanism_change_count == 0
                else ("PASS" if promotion_passed else "FAIL")
            ),
            "claim_count": len(promotion_rows),
            "detail": (
                "Not applicable: the selected candidate is the preregistered unchanged-incumbent control."
                if definition.mechanism_change_count == 0
                else (
                    "Every promotion claim passed its joint Holm-adjusted practical-gain bound."
                    if promotion_passed
                    else "At least one required promotion claim failed its adjusted practical-gain bound."
                )
            ),
        }
    )
    all_required_pass = all(gate["status"] == "PASS" for gate in gates if gate["required"])
    outcome = "scientifically_supported" if all_required_pass else "negative"
    result = {
        "schema_version": "aaa.observation_noise_joint_evaluation.v1",
        "protocol_version": protocol.protocol_version,
        "protocol_hash": canonical_protocol_hash(),
        "archives": {
            "A": {
                "path": str(a["directory"]),
                "batch_id": a["metadata"]["batch_id"],
                "verification": a["verification"],
            },
            "B": {
                "path": str(b["directory"]),
                "batch_id": b["metadata"]["batch_id"],
                "verification": b["verification"],
            },
        },
        "candidate_id": a["candidate_id"],
        "candidate_configuration_hash": definition.configuration_hash,
        "records_and_training": {
            "A_records_sha256": a["manifest"]["records_sha256"],
            "B_records_sha256": b["manifest"]["records_sha256"],
            "shared_training_checkpoints": True,
            "full_state_branch_pairing": True,
        },
        "multiplicity": {
            "method": "Holm step-down",
            "family_scope": "all primary endpoint x cell x batch claims across A+B",
            "family_size": family_size,
            "claim_order": ordered,
        },
        "claims": rows,
        "gates": gates,
        "all_required_gates_pass": all_required_pass,
        "outcome": outcome,
        "conclusion_recomputed_from_primitive_records": True,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def derive_joint_seed(protocol: Any) -> int:
    """Use a fixed seed namespace independent of stored summary conclusions."""

    value = protocol.raw["reference"]["v2_1_raw_sha256"]
    return int(value[:16], 16) ^ 0xA17B0B1
