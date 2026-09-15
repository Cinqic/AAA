"""Small reference verifier for primitive observation-noise artifacts.

The verifier intentionally owns its own arithmetic.  It does not import the
production benchmark collector, production gates, or stored summary booleans
as evidence.  A reproduction verdict is supplied separately by the runner.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from array import array
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO

from .candidates import candidate_definition

RECORD_SCHEMA = "aaa.observation_noise_step.v3"
LEGACY_RECORD_SCHEMAS = {"aaa.observation_noise_step.v2"}
CHECK_NAMES = (
    "record_schema",
    "strict_types",
    "finite_values",
    "chronology",
    "schedule_identity",
    "expected_coverage",
    "metric_recomputation",
    "scientific_gates_present",
    "mandatory_checks_present",
)
METRIC_FIELDS = {
    "raw",
    "scored",
    "latent_absolute_error",
    "latent_normalized_absolute_error",
    "noisy_observation_absolute_error",
    "signed_latent_error",
}


class VerificationError(ValueError):
    """Raised when primitive evidence is malformed or incomplete."""


def _safe_archive_file(run_dir: Path, relative: str, label: str) -> Path:
    """Resolve one archive member without permitting traversal or symlinks."""

    relative_path = Path(relative)
    if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
        raise VerificationError(f"{label} path is unsafe: {relative}")
    current = run_dir
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise VerificationError(f"{label} path is a symlink: {relative}")
    root = run_dir.resolve()
    resolved = (run_dir / relative_path).resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise VerificationError(f"{label} path is unsafe or missing: {relative}")
    return resolved


def _record_paths(run_dir: Path) -> list[Path]:
    flat = [name for name in ("records.jsonl", "records.jsonl.gz") if (run_dir / name).exists()]
    index_path = run_dir / "records" / "index.json"
    if flat and index_path.exists():
        raise VerificationError("archive ambiguously contains flat and sharded primitive records")
    if len(flat) > 1:
        raise VerificationError("archive ambiguously contains compressed and uncompressed primitive records")
    if flat:
        return [_safe_archive_file(run_dir, flat[0], "primitive record")]
    if index_path.is_file():
        _safe_archive_file(run_dir, "records/index.json", "record shard index")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            not isinstance(index, dict)
            or index.get("schema_version") != "aaa.observation_noise_record_shards.v1"
            or index.get("ordering") != "trial_id ascending, step ascending"
            or not isinstance(index.get("shards"), list)
            or not index["shards"]
        ):
            raise VerificationError("record shard index is malformed")
        paths: list[Path] = []
        prior_trial = ""
        total = 0
        for entry in index["shards"]:
            if not isinstance(entry, dict) or set(entry) != {
                "trial_id",
                "path",
                "records",
                "compressed_sha256",
            }:
                raise VerificationError("record shard index entry is malformed")
            trial_id = entry["trial_id"]
            relative = entry["path"]
            if not isinstance(trial_id, str) or trial_id <= prior_trial or not isinstance(relative, str):
                raise VerificationError("record shard identities are not strictly ordered")
            path = _safe_archive_file(run_dir, relative, "record shard")
            if _schedule_digest(path) != entry["compressed_sha256"]:
                raise VerificationError("record shard compressed checksum mismatch")
            if (
                isinstance(entry["records"], bool)
                or not isinstance(entry["records"], int)
                or entry["records"] < 1
            ):
                raise VerificationError("record shard count is invalid")
            total += entry["records"]
            paths.append(path)
            prior_trial = trial_id
        if index.get("records") != total:
            raise VerificationError("record shard index total is inconsistent")
        return paths
    raise VerificationError(f"{run_dir}: no primitive records file")


def _decode_lines(handle: TextIO, path: Path) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise VerificationError(f"{path}:{line_number}: malformed JSON") from error
        if not isinstance(value, dict):
            raise VerificationError(f"{path}:{line_number}: record must be an object")
        yield value


def iter_records(run_dir: str | Path) -> Iterator[dict[str, Any]]:
    for path in _record_paths(Path(run_dir)):
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                yield from _decode_lines(handle, path)
        else:
            with path.open("r", encoding="utf-8") as handle:
                yield from _decode_lines(handle, path)


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise VerificationError(f"{path}: expected a finite number")
    return float(value)


def _finite_close(stored: Any, recomputed: Any, path: str, *, tolerance: float = 1e-12) -> None:
    """Reject non-finite cached values before applying tolerance arithmetic."""

    stored_value = _finite(stored, f"{path}.stored")
    recomputed_value = _finite(recomputed, f"{path}.recomputed")
    if abs(stored_value - recomputed_value) > tolerance:
        raise VerificationError(f"{path} mismatch")


def validate_record(record: dict[str, Any]) -> None:
    required = {
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
        "update_decisions",
        "diagnostics",
        "schedule_index",
        "noise_channel",
        "noise_scale",
        "noise_innovation",
        "bounced",
        "changed",
    }
    schema_version = record.get("schema_version")
    if schema_version == RECORD_SCHEMA:
        required.add("prediction_intervals")
    elif schema_version not in LEGACY_RECORD_SCHEMAS:
        raise VerificationError("unsupported observation-noise record schema")
    unknown = sorted(set(record) - required)
    missing = sorted(required - set(record))
    if unknown or missing:
        raise VerificationError(f"record fields invalid: unknown={unknown}, missing={missing}")
    trial = record["trial"]
    if not isinstance(trial, dict):
        raise VerificationError("trial identity must be an object")
    trial_required = {
        "trial_id",
        "condition",
        "family",
        "channel",
        "scale",
        "lineage",
        "episode",
        "realization",
        "role",
        "branch",
        "stratum",
    }
    if set(trial) != trial_required:
        raise VerificationError("trial identity has missing or unknown fields")
    for name in ("trial_id", "condition", "family", "channel", "role", "branch", "stratum"):
        if not isinstance(trial[name], str) or not trial[name]:
            raise VerificationError(f"trial.{name} must be a nonempty string")
    if trial["condition"] not in {"clean_trained", "noise_trained"}:
        raise VerificationError("unknown training condition")
    if trial["family"] not in {"constant_velocity", "bouncing", "speed_change", "changed_law"}:
        raise VerificationError("unknown latent family")
    if trial["channel"] not in {"gaussian", "uniform", "correlated", "impulsive"}:
        raise VerificationError("unknown trial noise channel")
    if trial["role"] not in {"development", "confirmation_a", "confirmation_b"}:
        raise VerificationError("unknown trial role")
    if trial["branch"] not in {
        "stationary",
        "prefix",
        "frozen",
        "online",
        "sensor_shift_aligned",
        "sensor_shift_staggered",
        "sensor_shift_changed_aligned",
        "sensor_shift_changed_staggered",
    }:
        raise VerificationError("unknown trial branch")
    for name in ("lineage", "episode", "realization"):
        if isinstance(trial[name], bool) or not isinstance(trial[name], int) or trial[name] < 0:
            raise VerificationError(f"trial.{name} must be a non-negative integer")
    trial_scale = _finite(trial["scale"], "trial.scale")
    if trial_scale not in {0.0, 0.0005, 0.002, 0.01}:
        raise VerificationError("trial.scale is not a registered level")
    for name in ("step", "target_step", "schedule_index"):
        if isinstance(record[name], bool) or not isinstance(record[name], int) or record[name] < 0:
            raise VerificationError(f"record.{name} must be a non-negative integer")
    if record["target_step"] != record["step"] + 1:
        raise VerificationError("target_step is not the next transition")
    history = record["observed_history"]
    if not isinstance(history, list) or len(history) < 2:
        raise VerificationError("observed_history must contain at least two observations")
    for index, value in enumerate(history):
        _finite(value, f"observed_history[{index}]")
    current = _finite(record["current_observation"], "current_observation")
    if current != float(history[-1]):
        raise VerificationError("current_observation disagrees with observed history")
    latent_position = _finite(record["latent_position"], "latent_position")
    raw_observation = _finite(record["raw_observation"], "raw_observation")
    target = _finite(record["available_training_target"], "available_training_target")
    if target != float(record["raw_observation"]):
        raise VerificationError("available training target is not the revealed raw observation")
    line_width = _finite(record["line_width"], "line_width")
    if line_width <= 0:
        raise VerificationError("line_width must be positive")
    if not isinstance(record["noise_channel"], str) or record["noise_channel"] not in {
        "gaussian",
        "uniform",
        "correlated",
        "impulsive",
    }:
        raise VerificationError("unknown noise channel")
    noise_scale = _finite(record["noise_scale"], "noise_scale")
    if record["noise_channel"] != trial["channel"] or noise_scale != trial_scale:
        raise VerificationError("record noise identity disagrees with trial identity")
    innovation = _finite(record["noise_innovation"], "noise_innovation")
    effective_scale = _finite(record["effective_noise_scale"], "effective_noise_scale")
    if effective_scale < 0:
        raise VerificationError("effective_noise_scale must be non-negative")
    if abs(raw_observation - (latent_position + line_width * innovation)) > 1e-15:
        raise VerificationError("raw observation disagrees with latent truth and sensor innovation")
    if record["schedule_index"] != record["target_step"]:
        raise VerificationError("schedule_index must identify the revealed target timestamp")
    if not isinstance(record["bounced"], bool) or not isinstance(record["changed"], bool):
        raise VerificationError("bounced and changed must be booleans")
    predictions = record["predictions"]
    updates = record["update_decisions"]
    if (
        not isinstance(predictions, dict)
        or not predictions
        or not isinstance(updates, dict)
        or set(predictions) != set(updates)
    ):
        raise VerificationError("prediction and update maps must be nonempty and have equal keys")
    for name, metrics in predictions.items():
        if not isinstance(name, str) or not isinstance(metrics, dict) or set(metrics) != METRIC_FIELDS:
            raise VerificationError(f"invalid metric shape for predictor {name!r}")
        for field, value in metrics.items():
            _finite(value, f"predictions.{name}.{field}")
        if not isinstance(updates[name], bool):
            raise VerificationError(f"update_decisions.{name} must be boolean")
        expected_latent = abs(metrics["scored"] - record["latent_position"])
        if abs(metrics["latent_absolute_error"] - expected_latent) > 1e-15:
            raise VerificationError(f"cached latent error disagrees for {name}")
        expected_signed = metrics["scored"] - record["latent_position"]
        if abs(metrics["signed_latent_error"] - expected_signed) > 1e-15:
            raise VerificationError(f"cached signed error disagrees for {name}")
        expected_normalized = expected_latent / line_width
        if abs(metrics["latent_normalized_absolute_error"] - expected_normalized) > 1e-15:
            raise VerificationError(f"cached normalized latent error disagrees for {name}")
        expected_noisy = abs(metrics["scored"] - raw_observation) / line_width
        if abs(metrics["noisy_observation_absolute_error"] - expected_noisy) > 1e-15:
            raise VerificationError(f"cached noisy-observation error disagrees for {name}")
    if schema_version == RECORD_SCHEMA:
        intervals = record["prediction_intervals"]
        if not isinstance(intervals, dict) or set(intervals) != set(predictions):
            raise VerificationError("prediction intervals must cover every predictor")
        for name, levels in intervals.items():
            if not isinstance(levels, dict) or set(levels) != {"0.9", "0.95"}:
                raise VerificationError(f"invalid interval levels for predictor {name!r}")
            for level, interval in levels.items():
                if interval is None:
                    continue
                if not isinstance(interval, dict) or set(interval) != {"lower", "upper"}:
                    raise VerificationError(f"invalid {level} interval for predictor {name!r}")
                lower = _finite(interval["lower"], f"prediction_intervals.{name}.{level}.lower")
                upper = _finite(interval["upper"], f"prediction_intervals.{name}.{level}.upper")
                if lower > upper:
                    raise VerificationError(f"interval bounds are reversed for predictor {name!r}")
    diagnostics = record.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise VerificationError("diagnostics must be an object")


def _metric_rows(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, float, str, str, str, str], dict[str, Any]] = {}
    for record in records:
        trial = record["trial"]
        key = (
            trial["condition"],
            trial["family"],
            trial["channel"],
            float(trial["scale"]),
            trial["role"],
            trial["branch"],
            trial["stratum"],
        )
        for name, prediction in record["predictions"].items():
            identity = (key[0], key[1], key[2], key[3], key[4], key[5], key[6], name)
            bucket = grouped.setdefault(
                identity,
                {
                    "errors": array("d"),
                    "sum_squared": 0.0,
                    "sum_signed": 0.0,
                    "sum_noisy": 0.0,
                },
            )
            error = float(prediction["latent_normalized_absolute_error"])
            bucket["errors"].append(error)
            bucket["sum_squared"] += error * error
            bucket["sum_signed"] += float(prediction["signed_latent_error"])
            bucket["sum_noisy"] += float(prediction["noisy_observation_absolute_error"])
    output: dict[str, Any] = {}
    for (condition, family, channel, scale, role, branch, stratum, predictor), bucket in sorted(
        grouped.items()
    ):
        errors = bucket["errors"]
        if not errors:
            raise VerificationError("empty metric group")
        ordered = sorted(errors)
        label = f"{condition}|{family}|{channel}|{scale:.7g}|{role}|{branch}|{stratum}|{predictor}"
        output[label] = {
            "condition": condition,
            "family": family,
            "channel": channel,
            "scale": scale,
            "role": role,
            "branch": branch,
            "stratum": stratum,
            "predictor": predictor,
            "count": len(errors),
            "mae": float(sum(errors) / len(errors)),
            "rmse": float(math.sqrt(bucket["sum_squared"] / len(errors))),
            "signed_bias": float(bucket["sum_signed"] / len(errors)),
            "noisy_observation_mae": float(bucket["sum_noisy"] / len(errors)),
            "p95": float(ordered[min(len(errors) - 1, math.ceil(0.95 * len(errors)) - 1)]),
            "p99": float(ordered[min(len(errors) - 1, math.ceil(0.99 * len(errors)) - 1)]),
        }
    return output


def _interval_rows(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Recompute interval coverage and scores with verifier-owned arithmetic."""

    grouped: dict[tuple[str, str, str, float, str, str, str, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0]
    )
    for record in records:
        intervals = record.get("prediction_intervals")
        if intervals is None:
            continue
        trial = record["trial"]
        target = _finite(record["raw_observation"], "raw_observation")
        for predictor, levels in intervals.items():
            for level, interval in levels.items():
                if interval is None:
                    continue
                lower = _finite(interval["lower"], "interval.lower")
                upper = _finite(interval["upper"], "interval.upper")
                alpha = 1.0 - float(level)
                penalty = 0.0
                if target < lower:
                    penalty += 2.0 * (lower - target) / alpha
                if target > upper:
                    penalty += 2.0 * (target - upper) / alpha
                key = (
                    str(trial["condition"]),
                    str(trial["family"]),
                    str(trial["channel"]),
                    float(trial["scale"]),
                    str(trial["role"]),
                    str(trial["branch"]),
                    str(predictor),
                    str(level),
                )
                bucket = grouped[key]
                bucket[0] += 1.0
                bucket[1] += float(lower <= target <= upper)
                bucket[2] += upper - lower
                bucket[3] += upper - lower + penalty
    output: dict[str, Any] = {}
    for key, (count, covered, width, score) in sorted(grouped.items(), key=str):
        label = "|".join((*map(str, key[:3]), f"{key[3]:.7g}", *map(str, key[4:])))
        output[label] = {
            "condition": key[0],
            "family": key[1],
            "channel": key[2],
            "scale": key[3],
            "role": key[4],
            "branch": key[5],
            "predictor": key[6],
            "level": key[7],
            "count": int(count),
            "coverage": covered / count,
            "width": width / count,
            "interval_score": score / count,
            "evidence_status": "PASS",
        }
    return output


def _verify_training_evidence(root: Path, metadata: dict[str, Any], manifest: dict[str, Any]) -> int:
    """Verify the retained training primitive rows without using latent truth."""

    artifact = metadata.get("training_evidence")
    if not isinstance(artifact, dict) or artifact.get("path") != "training_records.jsonl":
        raise VerificationError("training evidence metadata is missing")
    path = root / "training_records.jsonl"
    if not path.is_file() or manifest.get("training_records_sha256") != _schedule_digest(path):
        raise VerificationError("training evidence checksum is missing or incorrect")
    with path.open("r", encoding="utf-8") as handle:
        rows = list(_decode_lines(handle, path))
    if artifact.get("records") != len(rows) or manifest.get("training_record_count") != len(rows):
        raise VerificationError("training evidence record count disagrees with metadata")
    required = {
        "schema_version",
        "condition",
        "lineage",
        "episode",
        "step",
        "channel",
        "scale",
        "schedule_index",
        "schedule_path",
        "schedule_digest",
        "observed_history",
        "forecast",
        "available_training_target",
        "update_decision",
    }
    for row in rows:
        if set(row) != required or row["schema_version"] != "aaa.observation_noise_training_step.v1":
            raise VerificationError("training primitive schema is invalid")
        if row["condition"] not in {"clean_trained", "noise_trained"}:
            raise VerificationError("training primitive condition is invalid")
        if row["channel"] not in {"gaussian", "uniform"}:
            raise VerificationError("training primitive channel is invalid")
        if _finite(row["scale"], "training.scale") < 0:
            raise VerificationError("training scale must be non-negative")
        schedule_path = row["schedule_path"]
        if not isinstance(schedule_path, str) or not schedule_path:
            raise VerificationError("training schedule path is invalid")
        schedule_file = (root / schedule_path).resolve()
        if root.resolve() not in schedule_file.parents or not schedule_file.is_file():
            raise VerificationError("training schedule path escapes the attempt or is missing")
        schedule = json.loads(schedule_file.read_text(encoding="utf-8"))
        if not isinstance(row["schedule_digest"], str) or schedule.get("digest") != row["schedule_digest"]:
            raise VerificationError("training schedule digest is not bound to the primitive row")
        for name in ("lineage", "episode", "step", "schedule_index"):
            if isinstance(row[name], bool) or not isinstance(row[name], int) or row[name] < 0:
                raise VerificationError(f"training.{name} must be a non-negative integer")
        history = row["observed_history"]
        if not isinstance(history, list) or len(history) < 2:
            raise VerificationError("training history is invalid")
        for value in history:
            _finite(value, "training.observed_history")
        _finite(row["forecast"], "training.forecast")
        _finite(row["available_training_target"], "training.available_training_target")
        if not isinstance(row["update_decision"], bool) or not row["update_decision"]:
            raise VerificationError("training update decision must be true")
    return len(rows)


def _resource_rows(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Recompute latency, update-norm and counter aggregates independently."""

    grouped: dict[tuple[str, str, str, float, str, str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for record in records:
        trial = record["trial"]
        diagnostics = record["diagnostics"]
        for predictor, diagnostic in diagnostics.items():
            key = (
                str(trial["condition"]),
                str(trial["family"]),
                str(trial["channel"]),
                float(trial["scale"]),
                str(trial["role"]),
                str(trial["branch"]),
                str(predictor),
            )
            bucket = grouped[key]
            for name in ("update_norm", "predict_latency_ns", "update_latency_ns", "detected_surprises"):
                value = _finite(diagnostic.get(name, 0.0), f"diagnostics.{name}")
                bucket[f"sum_{name}"] += value
                bucket[f"max_{name}"] = max(bucket[f"max_{name}"], value)
            for name in ("dead_zone_skips", "reflection_skips", "forgetting_suspensions"):
                value = _finite(diagnostic.get(name, 0.0), f"diagnostics.{name}")
                bucket[f"max_{name}"] = max(bucket[f"max_{name}"], value)
            bucket["count"] += 1.0
    output: dict[str, Any] = {}
    for key, values in sorted(grouped.items(), key=str):
        label = "|".join((*map(str, key[:3]), f"{key[3]:.7g}", *map(str, key[4:])))
        output[label] = {
            "condition": key[0],
            "family": key[1],
            "channel": key[2],
            "scale": key[3],
            "role": key[4],
            "branch": key[5],
            "predictor": key[6],
            "count": int(values["count"]),
            "mean_update_norm": values["sum_update_norm"] / values["count"],
            "mean_predict_latency_ns": values["sum_predict_latency_ns"] / values["count"],
            "mean_update_latency_ns": values["sum_update_latency_ns"] / values["count"],
            "max_detected_surprises": values["max_detected_surprises"],
            "max_dead_zone_skips": values["max_dead_zone_skips"],
            "max_reflection_skips": values["max_reflection_skips"],
            "max_forgetting_suspensions": values["max_forgetting_suspensions"],
        }
    return output


def _schedule_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _uncompressed_records_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _schedule_payload_digest(payload: dict[str, Any]) -> str:
    identity = {
        "channel": payload["channel"],
        "scale": payload["scale"],
        "seed": payload["seed"],
        "scale_path": payload["scale_path"],
        "values": payload["values"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _expected_confirmation_signatures(role: str) -> set[tuple[Any, ...]]:
    signatures: set[tuple[Any, ...]] = set()
    for condition in ("clean_trained", "noise_trained"):
        for family in ("constant_velocity", "bouncing", "speed_change", "changed_law"):
            for channel in ("gaussian", "uniform", "correlated", "impulsive"):
                for scale in (0.0, 0.0005, 0.002, 0.01):
                    for lineage in range(10):
                        for episode in range(32):
                            for realization in range(3):
                                signatures.add(
                                    (
                                        condition,
                                        family,
                                        channel,
                                        scale,
                                        lineage,
                                        episode,
                                        realization,
                                        role,
                                        "stationary",
                                    )
                                )
                                if family == "changed_law":
                                    for branch in ("prefix", "frozen", "online"):
                                        signatures.add(
                                            (
                                                condition,
                                                family,
                                                channel,
                                                scale,
                                                lineage,
                                                episode,
                                                realization,
                                                role,
                                                branch,
                                            )
                                        )
        for dynamics in ("unchanged", "changed"):
            for channel in ("gaussian", "uniform"):
                for from_scale in (0.0005, 0.002):
                    for timing in ("aligned", "staggered"):
                        branch = f"sensor_shift_{'changed_' if dynamics == 'changed' else ''}{timing}"
                        for lineage in range(10):
                            for episode in range(32):
                                for realization in range(3):
                                    signatures.add(
                                        (
                                            condition,
                                            "changed_law",
                                            channel,
                                            from_scale,
                                            lineage,
                                            episode,
                                            realization,
                                            role,
                                            branch,
                                        )
                                    )
    return signatures


def verify_attempt(run_dir: str | Path, *, require_checksums: bool = True) -> dict[str, Any]:
    """Verify primitive records and return a distinct recomputation verdict."""

    root = Path(run_dir)
    metadata_path = root / "metadata.json"
    manifest_path = root / "run_manifest.json"
    summary_path = root / "summary.json"
    if not metadata_path.is_file() or not manifest_path.is_file() or not summary_path.is_file():
        raise VerificationError("metadata.json, run_manifest.json, and summary.json are required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (metadata, manifest, summary)):
        raise VerificationError("metadata, manifest, and summary must be objects")
    is_v3 = metadata.get("protocol_version") == "aaa.observation_noise.v1.1"
    if is_v3:
        candidate_ids = {
            metadata.get("selected_candidate"),
            manifest.get("selected_candidate"),
            summary.get("selected_candidate"),
        }
        if len(candidate_ids) != 1 or None in candidate_ids:
            raise VerificationError("selected candidate identities disagree or are missing")
        try:
            definition = candidate_definition(str(next(iter(candidate_ids))))
        except ValueError as exc:
            raise VerificationError(str(exc)) from exc
        candidate_hashes = {
            metadata.get("selected_candidate_configuration_hash"),
            manifest.get("selected_candidate_configuration_hash"),
            summary.get("selected_candidate_configuration_hash"),
        }
        if candidate_hashes != {definition.configuration_hash}:
            raise VerificationError("selected candidate configuration hash is not canonical")
        if metadata.get("role") in {"confirmation_a", "confirmation_b"} and not isinstance(
            metadata.get("scientific_fingerprint_sha256"), str
        ):
            raise VerificationError("confirmation metadata is missing the scientific fingerprint")
    attempt_ids = {metadata.get("attempt_id"), manifest.get("attempt_id"), summary.get("attempt_id")}
    if len(attempt_ids) != 1 or None in attempt_ids:
        raise VerificationError("metadata, manifest, and summary attempt identities disagree")
    record_paths = _record_paths(root)
    if manifest.get("records_sha256") != _uncompressed_records_digest(record_paths):
        raise VerificationError("primitive record checksum disagrees with run manifest")
    compressed_path = root / "records.jsonl.gz"
    if compressed_path.is_file() and manifest.get("compressed_records_sha256") != _schedule_digest(
        compressed_path
    ):
        raise VerificationError("compressed primitive record checksum disagrees with run manifest")
    shard_index = root / "records" / "index.json"
    if shard_index.is_file() and (
        manifest.get("record_shard_index") != "records/index.json"
        or manifest.get("record_shard_index_sha256") != _schedule_digest(shard_index)
    ):
        raise VerificationError("record shard index identity disagrees with run manifest")
    if shard_index.is_file():
        shard_payload = json.loads(shard_index.read_text(encoding="utf-8"))
        if shard_payload.get("uncompressed_records_sha256") != manifest.get("records_sha256"):
            raise VerificationError("record shard semantic identity disagrees with run manifest")
    record_count = 0
    trial_count = 0
    previous_identity: tuple[str, int] | None = None
    has_v3_records = False
    observed_signatures: set[tuple[Any, ...]] = set()
    for record in iter_records(root):
        validate_record(record)
        identity = (str(record["trial"]["trial_id"]), int(record["step"]))
        if previous_identity is not None and identity <= previous_identity:
            raise VerificationError("primitive trial/step identities are duplicate or not canonical-sorted")
        if previous_identity is None or identity[0] != previous_identity[0]:
            trial_count += 1
            trial = record["trial"]
            observed_signatures.add(
                (
                    trial["condition"],
                    trial["family"],
                    trial["channel"],
                    float(trial["scale"]),
                    int(trial["lineage"]),
                    int(trial["episode"]),
                    int(trial["realization"]),
                    trial["role"],
                    trial["branch"],
                )
            )
        previous_identity = identity
        record_count += 1
        has_v3_records = has_v3_records or record["schema_version"] == RECORD_SCHEMA
    if record_count == 0:
        raise VerificationError("no primitive records retained")
    training_records = 0
    if has_v3_records:
        training_records = _verify_training_evidence(root, metadata, manifest)
        for directory_name, manifest_key in (
            ("training_schedules", "training_schedule_count"),
            ("calibration_schedules", "calibration_schedule_count"),
        ):
            directory = root / directory_name
            count = len(list(directory.glob("*.json"))) if directory.is_dir() else 0
            if manifest.get(manifest_key) != count or count == 0:
                raise VerificationError(f"{directory_name} count is missing or inconsistent")
    expected = int(manifest["expected_scored_records"])
    if record_count != expected:
        raise VerificationError(f"record count {record_count} does not equal expected {expected}")
    role = metadata.get("role")
    if role in {"confirmation_a", "confirmation_b"}:
        expected_manifest = {
            "lineages": 10,
            "episodes_per_family_per_lineage": 32,
            "sensor_realizations_per_episode": 3,
            "planned_trials": 153600,
            "schedule_count": 153600,
            "expected_scored_records": 59043840,
        }
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            raise VerificationError("formal confirmation manifest does not match the frozen complete plan")
        expected_signatures = _expected_confirmation_signatures(str(role))
        if observed_signatures != expected_signatures:
            raise VerificationError("formal confirmation trial cells are incomplete or unexpected")
    schedule_index = json.loads((root / "schedules" / "index.json").read_text(encoding="utf-8"))
    if not isinstance(schedule_index, list) or not schedule_index:
        raise VerificationError("schedule index must be a nonempty list")
    schedules: dict[str, dict[str, Any]] = {}
    for item in schedule_index:
        if not isinstance(item, dict) or set(item) != {
            "trial_id",
            "path",
            "file_sha256",
            "schedule_digest",
        }:
            raise VerificationError("schedule index entry has missing or unknown fields")
        if not all(
            isinstance(item[key], str) and item[key]
            for key in ("trial_id", "path", "file_sha256", "schedule_digest")
        ):
            raise VerificationError("schedule index identity fields must be nonempty strings")
        if item["trial_id"] in schedules:
            raise VerificationError("duplicate schedule trial identity")
        path = _safe_archive_file(root, item["path"], "schedule")
        if _schedule_digest(path) != item["file_sha256"]:
            raise VerificationError(f"schedule checksum mismatch for {item['path']}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "channel",
            "scale",
            "seed",
            "scale_path",
            "values",
            "digest",
        }:
            raise VerificationError(f"invalid schedule payload for {item['trial_id']}")
        if payload["schema_version"] != "aaa.observation_noise_schedule.v1":
            raise VerificationError("unsupported schedule schema")
        if not isinstance(payload["channel"], str) or payload["channel"] not in {
            "gaussian",
            "uniform",
            "correlated",
            "impulsive",
        }:
            raise VerificationError("invalid schedule channel")
        payload_scale = _finite(payload["scale"], "schedule.scale")
        if payload_scale not in {0.0, 0.0005, 0.002, 0.01}:
            raise VerificationError("schedule.scale is not registered")
        if isinstance(payload["seed"], bool) or not isinstance(payload["seed"], int):
            raise VerificationError("schedule.seed must be an integer")
        values = payload["values"]
        if not isinstance(values, list) or not values:
            raise VerificationError("schedule.values must be a nonempty list")
        for index, value in enumerate(values):
            _finite(value, f"schedule.values[{index}]")
        scale_path = payload["scale_path"]
        if scale_path is not None:
            if not isinstance(scale_path, list) or len(scale_path) != len(values):
                raise VerificationError("schedule.scale_path must cover schedule.values")
            for index, value in enumerate(scale_path):
                if _finite(value, f"schedule.scale_path[{index}]") < 0:
                    raise VerificationError("schedule.scale_path values must be non-negative")
        if not isinstance(payload["digest"], str) or payload["digest"] != _schedule_payload_digest(payload):
            raise VerificationError(f"schedule content digest mismatch for {item['trial_id']}")
        if item["schedule_digest"] != payload["digest"]:
            raise VerificationError(f"schedule index digest mismatch for {item['trial_id']}")
        schedules[item["trial_id"]] = payload
    if manifest.get("schedule_count") != len(schedules):
        raise VerificationError("manifest schedule count disagrees with schedule index")
    for record in iter_records(root):
        trial = record["trial"]
        schedule_id = (
            trial["trial_id"]
            if trial["branch"] not in {"prefix", "frozen", "online"}
            else trial["trial_id"].rsplit(":", 1)[0]
        )
        payload = schedules.get(schedule_id)
        if payload is None:
            raise VerificationError(f"no schedule is registered for record trial {trial['trial_id']}")
        if payload["channel"] != trial["channel"] or float(payload["scale"]) != float(trial["scale"]):
            raise VerificationError(f"schedule identity disagrees for record trial {trial['trial_id']}")
        values = payload["values"]
        index = record["schedule_index"]
        if index >= len(values) or record["noise_innovation"] != values[index]:
            raise VerificationError(
                f"record noise innovation disagrees with schedule for {trial['trial_id']}"
            )
        scale_path = payload["scale_path"]
        expected_scale = payload["scale"] if scale_path is None else scale_path[index]
        if record["effective_noise_scale"] != expected_scale:
            raise VerificationError(f"record effective scale disagrees with schedule for {trial['trial_id']}")
        if trial["branch"] == "stationary" and scale_path is not None:
            raise VerificationError("stationary trial unexpectedly has a time-varying scale path")
        if trial["branch"].startswith("sensor_shift") and scale_path is None:
            raise VerificationError("sensor-shift trial is missing its time-varying scale path")
    metrics = _metric_rows(iter_records(root))
    stored_metrics = summary.get("metrics")
    if not isinstance(stored_metrics, dict) or set(stored_metrics) != set(metrics):
        raise VerificationError("stored summary metric keys do not match primitive recomputation")
    for key, recomputed in metrics.items():
        stored = stored_metrics[key]
        for field in ("count", "mae", "rmse", "signed_bias", "noisy_observation_mae", "p95", "p99"):
            if field not in stored or field not in recomputed:
                raise VerificationError(f"metric {key} is missing {field}")
            if field == "count":
                if isinstance(stored[field], bool) or not isinstance(stored[field], int):
                    raise VerificationError(f"metric {key} count must be a strict integer")
                if stored[field] != recomputed[field]:
                    raise VerificationError(f"metric {key} count mismatch")
            else:
                _finite_close(stored[field], recomputed[field], f"metric {key} {field}")
    if has_v3_records:
        statistics = summary.get("statistics")
        if not isinstance(statistics, dict):
            raise VerificationError("v3 evidence requires a statistics artifact")
        declared_draws = statistics.get("declared_draws")
        executed_draws = statistics.get("executed_draws")
        if (
            isinstance(declared_draws, bool)
            or not isinstance(declared_draws, int)
            or declared_draws < 1
            or isinstance(executed_draws, bool)
            or not isinstance(executed_draws, int)
            or executed_draws < 1
            or executed_draws > declared_draws
        ):
            raise VerificationError("statistics draw declarations are invalid")
        hierarchical = statistics.get("hierarchical")
        paired = statistics.get("paired_comparisons")
        if not isinstance(hierarchical, dict) or hierarchical.get("method") != "hierarchical_bootstrap":
            raise VerificationError("hierarchical statistics method is missing")
        if not isinstance(paired, dict) or paired.get("method") != "paired_hierarchical_bootstrap":
            raise VerificationError("paired statistics method is missing")
        null_validation = statistics.get("null_validation")
        if (
            not isinstance(null_validation, dict)
            or null_validation.get("status") != "DIAGNOSTIC_ONLY"
            or not isinstance(null_validation.get("simulations"), int)
            or null_validation["simulations"] < 1
            or not isinstance(null_validation.get("bootstrap_draws"), int)
            or null_validation["bootstrap_draws"] < 1
        ):
            raise VerificationError("known-null finite-sample diagnostic is missing")
        interval_artifact = statistics.get("intervals")
        if (
            not isinstance(interval_artifact, dict)
            or interval_artifact.get("method") != "causal_residual_quantile"
        ):
            raise VerificationError("causal interval statistics are missing")
        recomputed_intervals = _interval_rows(iter_records(root))
        stored_intervals = interval_artifact.get("cells")
        if not isinstance(stored_intervals, dict) or set(stored_intervals) != set(recomputed_intervals):
            raise VerificationError("stored interval cells do not match primitive recomputation")
        for key, recomputed in recomputed_intervals.items():
            stored = stored_intervals[key]
            for field in ("count", "coverage", "width", "interval_score"):
                if field not in stored:
                    raise VerificationError(f"interval cell {key} is missing {field}")
                if field == "count":
                    if stored[field] != recomputed[field]:
                        raise VerificationError(f"interval cell {key} count mismatch")
                else:
                    _finite_close(stored[field], recomputed[field], f"interval cell {key} {field}")
        resource_artifact = statistics.get("resources")
        if (
            not isinstance(resource_artifact, dict)
            or resource_artifact.get("method") != "primitive_diagnostic_aggregation"
        ):
            raise VerificationError("resource diagnostics are missing")
        recomputed_resources = _resource_rows(iter_records(root))
        stored_resources = resource_artifact.get("cells")
        if not isinstance(stored_resources, dict) or set(stored_resources) != set(recomputed_resources):
            raise VerificationError("stored resource cells do not match primitive recomputation")
        for key, recomputed in recomputed_resources.items():
            stored = stored_resources[key]
            for field in (
                "count",
                "mean_update_norm",
                "mean_predict_latency_ns",
                "mean_update_latency_ns",
                "max_detected_surprises",
                "max_dead_zone_skips",
                "max_reflection_skips",
                "max_forgetting_suspensions",
            ):
                if field not in stored:
                    raise VerificationError(f"resource cell {key} is missing {field}")
                if field == "count":
                    if stored[field] != recomputed[field]:
                        raise VerificationError(f"resource cell {key} count mismatch")
                else:
                    _finite_close(stored[field], recomputed[field], f"resource cell {key} {field}")
    checks = summary.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CHECK_NAMES):
        raise VerificationError("mandatory verifier checks are missing or unknown")
    if any(
        not isinstance(value, dict)
        or set(value) != {"status", "detail"}
        or value.get("status") not in {"PASS", "FAIL", "NOT_VERIFIED", "INSUFFICIENT_EVIDENCE"}
        or not isinstance(value.get("detail"), str)
        or not value["detail"]
        for value in checks.values()
    ):
        raise VerificationError("invalid mandatory check result")
    required_gate_names = {
        "reference_preservation",
        "evidence_integrity",
        "numerical_stability",
        "scientific_primary_endpoints",
        "confirmation_reproducibility",
    }
    gates = summary.get("gates")
    if (
        not isinstance(gates, list)
        or {gate.get("name") for gate in gates if isinstance(gate, dict)} != required_gate_names
    ):
        raise VerificationError("mandatory scientific gates are missing or unknown")
    if any(
        not isinstance(gate, dict)
        or set(gate) != {"name", "status", "required", "detail"}
        or not isinstance(gate["name"], str)
        or gate["status"] not in {"PASS", "FAIL", "NOT_VERIFIED", "INSUFFICIENT_EVIDENCE"}
        or not isinstance(gate["required"], bool)
        or not isinstance(gate["detail"], str)
        or not gate["detail"]
        for gate in gates
    ):
        raise VerificationError("invalid scientific gate result")
    checksum_manifest = root / "checksums.json"
    if require_checksums and not checksum_manifest.is_file():
        raise VerificationError("checksums.json is required for archive verification")
    if checksum_manifest.is_file():
        checksum_rows = json.loads(checksum_manifest.read_text(encoding="utf-8"))
        if not isinstance(checksum_rows, dict) or not checksum_rows:
            raise VerificationError("checksum manifest must be a nonempty object")
        for relative, expected_hash in checksum_rows.items():
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise VerificationError("checksum manifest entries must be string pairs")
            path = (root / relative).resolve()
            if root.resolve() not in path.parents or not path.is_file():
                raise VerificationError(f"checksum manifest references an invalid path: {relative}")
            if _schedule_digest(path) != expected_hash:
                raise VerificationError(f"checksum manifest mismatch for {relative}")
        expected_paths = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path.name != "checksums.json" and not path.name.endswith(".tmp")
        }
        if set(checksum_rows) != expected_paths:
            missing = sorted(expected_paths - set(checksum_rows))
            unexpected = sorted(set(checksum_rows) - expected_paths)
            raise VerificationError(
                f"checksum manifest coverage mismatch: missing={missing}, unexpected={unexpected}"
            )
    return {
        "verdict": "PASS",
        "records": record_count,
        "trials": trial_count,
        "metrics": metrics,
        "training_records": training_records,
        "metadata_attempt_id": metadata.get("attempt_id"),
        "reproduction_verdict": summary.get("reproduction", {"status": "NOT_VERIFIED"}),
    }
