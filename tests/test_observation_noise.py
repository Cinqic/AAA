"""Adversarial and lifecycle tests for the separately versioned noise phase."""

from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aaa.noise.calibration import CausalResidualCalibrator, interval_score
from aaa.noise.observation import (
    NoiseSchedule,
    NoisyObservationEnvironment,
    generate_schedule,
    generator_validation,
)
from aaa.noise.runner import (
    ObservationNoiseError,
    _finalize_record_shards,
    _persist_trial_rows,
    _zero_noise_fixture,
    build_latent_environment,
    run_attempt,
    run_noisy_episode,
    train_model,
)
from aaa.noise.spec import ProtocolError, canonical_protocol_hash, load_protocol
from aaa.noise.statistics import (
    _draw_hierarchical_many,
    _hierarchical_mean,
    hierarchical_bootstrap,
    interval_statistics,
    paired_hierarchical_comparisons,
    validate_null_behavior,
)
from aaa.noise.verifier import (
    CHECK_NAMES,
    VerificationError,
    _metric_rows,
    _schedule_payload_digest,
    validate_record,
    verify_attempt,
)
from aaa.predictors import Predictor


class ObservationNoiseProtocolTests(unittest.TestCase):
    def test_protocol_identity_and_frozen_choices(self):
        protocol = load_protocol()
        self.assertEqual(protocol.protocol_version, "aaa.observation_noise.v1.1")
        self.assertEqual(protocol.status, "design_frozen")
        self.assertEqual(
            [channel.name for channel in protocol.channels],
            ["gaussian", "uniform", "correlated", "impulsive"],
        )
        self.assertEqual(canonical_protocol_hash(), protocol.hash())
        self.assertEqual(protocol.replication["lineages"], 10)
        self.assertEqual(protocol.replication["episodes_per_family_per_lineage"], 32)

    def test_zero_noise_reference_runs_from_the_pinned_v21_commit(self):
        protocol = load_protocol()
        result = _zero_noise_fixture(protocol)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["reference_commit"], protocol.reference["v2_1_commit"])
        self.assertTrue(result["reference_identity_equal"])

    def test_installed_package_reports_checkout_only_reference_as_not_verified(self):
        protocol = load_protocol()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("aaa.noise.runner.project_root", return_value=Path(directory)),
        ):
            result = _zero_noise_fixture(protocol)
        self.assertEqual(result["status"], "NOT_VERIFIED")
        self.assertIn("maintained Git checkout", result["detail"])

    def test_unknown_protocol_fields_fail_closed(self):
        protocol = load_protocol().to_dict()
        protocol["unexpected"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaises(ProtocolError):
                load_protocol(path)

    def test_generator_validation_is_separate_from_benchmark_draws(self):
        result = generator_validation(991)
        self.assertEqual(result["sample_count"], 20_000)
        for channel in ("gaussian", "uniform", "correlated", "impulsive"):
            self.assertTrue(result["channels"][channel]["finite"])
            self.assertAlmostEqual(result["channels"][channel]["variance"], 1.0, delta=0.08)

    def test_confirmation_roles_reuse_training_lineage_seeds(self):
        protocol = load_protocol()
        model_a, metadata_a = train_model(protocol, "noise_trained", 0, role="confirmation_a", quick=True)
        model_b, metadata_b = train_model(protocol, "noise_trained", 0, role="confirmation_b", quick=True)
        self.assertEqual(model_a.state_dict(), model_b.state_dict())
        self.assertEqual(metadata_a["schedule_digests"], metadata_b["schedule_digests"])

    def test_factorial_control_can_disable_the_law_change(self):
        unchanged = build_latent_environment("changed_law", 91, 400, include_law_change=False)
        changed = build_latent_environment("changed_law", 91, 400, include_law_change=True)
        self.assertIsNone(unchanged.change_step)
        self.assertEqual(changed.change_step, 300)

    def test_confirmation_rejects_a_batch_outside_the_confirmation_freeze(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ObservationNoiseError):
            run_attempt(
                role="confirmation_a",
                batch_id="unfrozen-confirmation-batch",
                attempt_label="must-not-run",
                output_root=directory,
            )


class ObservationBoundaryTests(unittest.TestCase):
    def test_schedule_is_deterministic_and_zero_noise_is_exact(self):
        first = generate_schedule("correlated", 0.002, 20, seed=41)
        second = generate_schedule("correlated", 0.002, 20, seed=41)
        self.assertEqual(first, second)
        clean = generate_schedule("impulsive", 0.0, 20, seed=41)
        self.assertTrue(all(value == 0.0 for value in clean.values))

    def test_sensor_is_cached_and_is_not_clipped_at_the_physical_boundary(self):
        latent = build_latent_environment("constant_velocity", 5, 4)
        schedule = NoiseSchedule("gaussian", 2.0, (2.0,) * 5, 5)
        environment = NoisyObservationEnvironment(latent, schedule)
        observation = environment.reset()
        self.assertEqual(observation, environment.observe())
        self.assertGreater(observation, environment.config.upper_bound)

    def test_temporal_boundary_updates_after_target_reveal(self):
        class Spy(Predictor):
            name = "spy"
            update_enabled = True

            def __init__(self):
                self.events: list[str] = []
                self.targets: list[float] = []

            def predict(self, history):
                self.events.append("predict")
                return float(history[-1])

            def update(self, history, target_position):
                self.events.append("update")
                self.targets.append(float(target_position))

        latent = build_latent_environment("constant_velocity", 6, 8)
        schedule = generate_schedule("gaussian", 0.002, 9, seed=6)
        environment = NoisyObservationEnvironment(latent, schedule)
        spy = Spy()
        records = run_noisy_episode(
            environment,
            [spy],
            {
                "trial_id": "spy",
                "condition": "clean_trained",
                "family": "constant_velocity",
                "channel": "gaussian",
                "scale": 0.002,
                "lineage": 0,
                "episode": 0,
                "realization": 0,
                "role": "development",
                "branch": "stationary",
                "stratum": "unstratified",
            },
            schedule,
        )
        self.assertEqual(len(records), 5)
        self.assertEqual(spy.events, ["predict", "update"] * 5)
        self.assertEqual(spy.targets, [record["available_training_target"] for record in records])
        self.assertNotEqual(spy.targets[0], records[0]["latent_position"])
        self.assertIn("prediction_intervals", records[0])
        validate_record(records[0])

    def test_interval_is_constructed_before_the_target_advance(self):
        events: list[str] = []

        class LoggedEnvironment(NoisyObservationEnvironment):
            def advance(self):
                events.append("advance")
                return super().advance()

        class LoggedPredictor(Predictor):
            name = "logged"
            update_enabled = True

            def predict(self, history):
                events.append("predict")
                return float(history[-1])

            def update(self, history, target_position):
                events.append("update")

        class LoggedCalibrator:
            def __deepcopy__(self, memo):
                return self

            def intervals(self, forecast):
                events.append("interval")
                return {"0.9": None, "0.95": None}

            def update(self, forecast, noisy_target):
                events.append("calibrate")

        latent = build_latent_environment("constant_velocity", 8, 8)
        schedule = generate_schedule("gaussian", 0.002, 9, seed=8)
        run_noisy_episode(
            LoggedEnvironment(latent, schedule),
            [LoggedPredictor()],
            {
                "trial_id": "logged",
                "condition": "clean_trained",
                "family": "constant_velocity",
                "channel": "gaussian",
                "scale": 0.002,
                "lineage": 0,
                "episode": 0,
                "realization": 0,
                "role": "development",
                "branch": "stationary",
                "stratum": "unstratified",
            },
            schedule,
            initial_calibrators={"logged": LoggedCalibrator()},
        )
        first_prediction = events.index("predict")
        self.assertEqual(
            events[first_prediction : first_prediction + 5],
            [
                "predict",
                "interval",
                "advance",
                "update",
                "calibrate",
            ],
        )


class ObservationStatisticsTests(unittest.TestCase):
    def test_vectorized_balanced_draws_preserve_the_hierarchical_estimand(self):
        nested = {
            0: {0: {0: 1.0, 1: 3.0}, 1: {0: 5.0, 1: 7.0}},
            1: {0: {0: 11.0, 1: 13.0}, 1: {0: 15.0, 1: 17.0}},
        }
        first = _draw_hierarchical_many(nested, np.random.default_rng(41), 100_000)
        second = _draw_hierarchical_many(nested, np.random.default_rng(41), 100_000)
        self.assertTrue(np.array_equal(first, second))
        self.assertAlmostEqual(float(first.mean()), _hierarchical_mean(nested), delta=0.05)

    @staticmethod
    def _records() -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for lineage in range(2):
            for episode in range(2):
                for realization in range(2):
                    trial = {
                        "trial_id": f"l{lineage}-e{episode}-r{realization}",
                        "condition": "clean_trained",
                        "family": "constant_velocity",
                        "channel": "gaussian",
                        "scale": 0.002,
                        "lineage": lineage,
                        "episode": episode,
                        "realization": realization,
                        "role": "development",
                        "branch": "stationary",
                        "stratum": "unstratified",
                    }
                    predictions = {}
                    for name, error in (
                        ("constant_motion_reflected", 0.20 + 0.01 * episode),
                        ("incumbent_square_root_rls", 0.10 + 0.01 * lineage),
                    ):
                        predictions[name] = {
                            "latent_normalized_absolute_error": error,
                        }
                    records.append({"trial": trial, "predictions": predictions})
        return records

    def test_hierarchy_retains_all_three_resampling_levels(self):
        result = hierarchical_bootstrap(self._records(), draws=16, seed=4)
        cell = result["cells"][
            "clean_trained|constant_velocity|gaussian|0.002|development|stationary|incumbent_square_root_rls"
        ]
        self.assertEqual(cell["lineages"], 2)
        self.assertEqual(cell["episodes"], 4)
        self.assertEqual(cell["sensor_realizations"], 8)
        self.assertEqual(cell["draws"], 16)
        self.assertEqual(set(cell["intervals"]), {"0.9", "0.95"})

    def test_paired_comparison_and_holm_adjustment_are_recorded(self):
        result = paired_hierarchical_comparisons(self._records(), draws=16, seed=9)
        key = (
            "clean_trained|constant_velocity|gaussian|0.002|development|stationary|incumbent_square_root_rls"
        )
        self.assertIn(key, result["cells"])
        self.assertIn("holm_adjusted_p_value", result["cells"][key])
        self.assertEqual(result["multiplicity"], "holm_bonferroni_with_familywise_bounds")

    def test_calibration_uses_only_past_noisy_residuals(self):
        calibrator = CausalResidualCalibrator(minimum_samples=2)
        self.assertIsNone(calibrator.intervals(0.5)["0.9"])
        calibrator.update(0.5, 0.6)
        calibrator.update(0.5, 0.7)
        interval = calibrator.intervals(0.5)["0.9"]
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertAlmostEqual(interval["lower"], 0.3)
        self.assertAlmostEqual(interval["upper"], 0.7)
        self.assertAlmostEqual(interval_score(interval, 0.7, 0.9), 0.4)

    def test_interval_summary_excludes_unavailable_warmup(self):
        record = self._records()[0]
        record["raw_observation"] = 0.5
        record["prediction_intervals"] = {
            "incumbent_square_root_rls": {"0.9": None, "0.95": {"lower": 0.4, "upper": 0.6}},
            "constant_motion_reflected": {"0.9": None, "0.95": None},
        }
        result = interval_statistics([record])
        self.assertEqual(len(result["cells"]), 1)
        row = next(iter(result["cells"].values()))
        self.assertEqual(row["count"], 1)
        self.assertEqual(row["coverage"], 1.0)

    def test_known_null_validation_is_separate_diagnostic_evidence(self):
        result = validate_null_behavior(seed=13, simulations=4, bootstrap_draws=8)
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(result["simulations"], 4)
        self.assertAlmostEqual(result["zero_in_interval_rate"] + result["false_positive_rate"], 1.0)


def _valid_record() -> dict[str, object]:
    return {
        "schema_version": "aaa.observation_noise_step.v2",
        "trial": {
            "trial_id": "fixture",
            "condition": "clean_trained",
            "family": "constant_velocity",
            "channel": "gaussian",
            "scale": 0.002,
            "lineage": 0,
            "episode": 0,
            "realization": 0,
            "role": "development",
            "branch": "stationary",
            "stratum": "unstratified",
        },
        "step": 3,
        "target_step": 4,
        "observed_history": [0.1, 0.2, 0.3, 0.4],
        "current_observation": 0.4,
        "latent_position": 0.5,
        "raw_observation": 0.51,
        "available_training_target": 0.51,
        "line_width": 1.0,
        "effective_noise_scale": 0.002,
        "predictions": {
            "fixture_predictor": {
                "raw": 0.6,
                "scored": 0.6,
                "latent_absolute_error": 0.1,
                "latent_normalized_absolute_error": 0.1,
                "noisy_observation_absolute_error": 0.09,
                "signed_latent_error": 0.1,
            }
        },
        "update_decisions": {"fixture_predictor": True},
        "diagnostics": {},
        "schedule_index": 4,
        "noise_channel": "gaussian",
        "noise_scale": 0.002,
        "noise_innovation": 0.01,
        "bounced": False,
        "changed": False,
    }


def _write_verifier_fixture(directory: Path, record: dict[str, object]) -> None:
    schedule_identity = {
        "channel": "gaussian",
        "scale": 0.002,
        "seed": 17,
        "scale_path": None,
        "values": [0.0, 0.0, 0.0, 0.0, 0.01],
    }
    schedule = {
        "schema_version": "aaa.observation_noise_schedule.v1",
        **schedule_identity,
        "digest": _schedule_payload_digest(schedule_identity),
    }
    schedules = directory / "schedules"
    schedules.mkdir()
    schedule_path = schedules / "fixture.json"
    schedule_path.write_text(json.dumps(schedule, sort_keys=True) + "\n", encoding="utf-8")
    schedule_sha = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    (schedules / "index.json").write_text(
        json.dumps(
            [
                {
                    "trial_id": "fixture",
                    "path": "schedules/fixture.json",
                    "file_sha256": schedule_sha,
                    "schedule_digest": schedule["digest"],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    record_path = directory / "records.jsonl"
    record_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    record_sha = hashlib.sha256(record_path.read_bytes()).hexdigest()
    (directory / "metadata.json").write_text(json.dumps({"attempt_id": "fixture"}) + "\n", encoding="utf-8")
    (directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "attempt_id": "fixture",
                "expected_scored_records": 1,
                "records_sha256": record_sha,
                "schedule_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checks = {name: {"status": "PASS", "detail": "fixture"} for name in CHECK_NAMES}
    gates = [
        {
            "name": name,
            "status": "PASS",
            "required": True,
            "detail": "fixture",
        }
        for name in (
            "reference_preservation",
            "evidence_integrity",
            "numerical_stability",
            "scientific_primary_endpoints",
            "confirmation_reproducibility",
        )
    ]
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "attempt_id": "fixture",
                "metrics": _metric_rows([record]),
                "checks": checks,
                "gates": gates,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checksum_paths = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name != "checksums.json"
    )
    (directory / "checksums.json").write_text(
        json.dumps(
            {
                str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in checksum_paths
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class ObservationNoiseVerifierTests(unittest.TestCase):
    def test_valid_primitive_record_is_accepted(self):
        validate_record(_valid_record())

    def test_boolean_integer_substitution_is_rejected(self):
        record = _valid_record()
        record["step"] = True
        with self.assertRaises(VerificationError):
            validate_record(record)

    def test_stale_cached_error_is_rejected(self):
        record = _valid_record()
        predictions = record["predictions"]
        assert isinstance(predictions, dict)
        fixture = predictions["fixture_predictor"]
        assert isinstance(fixture, dict)
        fixture["latent_absolute_error"] = 0.0
        with self.assertRaises(VerificationError):
            validate_record(record)

    def test_unknown_record_field_is_rejected(self):
        record = _valid_record()
        record["latent_velocity"] = 0.1
        with self.assertRaises(VerificationError):
            validate_record(record)

    def test_cached_normalized_or_noisy_error_is_rejected(self):
        record = _valid_record()
        predictions = record["predictions"]
        assert isinstance(predictions, dict)
        fixture = predictions["fixture_predictor"]
        assert isinstance(fixture, dict)
        fixture["latent_normalized_absolute_error"] = 0.2
        with self.assertRaises(VerificationError):
            validate_record(record)

        record = _valid_record()
        predictions = record["predictions"]
        assert isinstance(predictions, dict)
        fixture = predictions["fixture_predictor"]
        assert isinstance(fixture, dict)
        fixture["noisy_observation_absolute_error"] = 0.0
        with self.assertRaises(VerificationError):
            validate_record(record)

    def test_sensor_equation_and_schedule_index_are_checked(self):
        record = _valid_record()
        record["raw_observation"] = 0.52
        record["available_training_target"] = 0.52
        with self.assertRaises(VerificationError):
            validate_record(record)

    def test_rehashed_corrupt_evidence_fails_independent_arithmetic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _valid_record()
            _write_verifier_fixture(root, record)
            self.assertEqual(verify_attempt(root)["verdict"], "PASS")

            predictions = record["predictions"]
            assert isinstance(predictions, dict)
            fixture = predictions["fixture_predictor"]
            assert isinstance(fixture, dict)
            fixture["latent_absolute_error"] = 0.0
            fixture["latent_normalized_absolute_error"] = 0.0
            record_path = root / "records.jsonl"
            record_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            manifest["records_sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
            (root / "run_manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            summary["metrics"] = _metric_rows([record])
            (root / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
            with self.assertRaises(VerificationError):
                verify_attempt(root)

    def test_missing_checksum_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_verifier_fixture(root, _valid_record())
            (root / "checksums.json").unlink()
            with self.assertRaisesRegex(VerificationError, "checksums.json is required"):
                verify_attempt(root)

    def test_schedule_index_cannot_escape_archive_or_follow_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_verifier_fixture(root, _valid_record())
            index_path = root / "schedules" / "index.json"
            schedule_index = json.loads(index_path.read_text(encoding="utf-8"))
            schedule_index[0]["path"] = "../outside.json"
            index_path.write_text(json.dumps(schedule_index) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "schedule path is unsafe"):
                verify_attempt(root, require_checksums=False)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_verifier_fixture(root, _valid_record())
            schedule_path = root / "schedules" / "fixture.json"
            target = root / "schedule-target.json"
            schedule_path.replace(target)
            schedule_path.symlink_to(target)
            with self.assertRaisesRegex(VerificationError, "schedule path is a symlink"):
                verify_attempt(root, require_checksums=False)

    def test_nonfinite_stored_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_verifier_fixture(root, _valid_record())
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            first_metric = next(iter(summary["metrics"].values()))
            first_metric["mae"] = float("nan")
            (root / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
            checksum_rows = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
            checksum_rows["summary.json"] = hashlib.sha256((root / "summary.json").read_bytes()).hexdigest()
            (root / "checksums.json").write_text(json.dumps(checksum_rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "expected a finite number"):
                verify_attempt(root)

    def test_gzip_only_archive_uses_uncompressed_record_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_verifier_fixture(root, _valid_record())
            record_path = root / "records.jsonl"
            compressed_path = root / "records.jsonl.gz"
            with (
                record_path.open("rb") as source,
                gzip.GzipFile(filename=str(compressed_path), mode="wb", mtime=0) as target,
            ):
                target.write(source.read())
            record_path.unlink()
            manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            manifest["compressed_records_sha256"] = hashlib.sha256(compressed_path.read_bytes()).hexdigest()
            (root / "run_manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            checksum_rows = {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*")
                if path.is_file() and path.name != "checksums.json"
            }
            (root / "checksums.json").write_text(json.dumps(checksum_rows) + "\n", encoding="utf-8")
            self.assertEqual(verify_attempt(root)["verdict"], "PASS")

    def test_finalized_record_shards_are_verified_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _valid_record()
            _write_verifier_fixture(root, record)
            (root / "records.jsonl").unlink()
            _persist_trial_rows(root, [record])
            count, records_sha, index_sha = _finalize_record_shards(root)
            self.assertEqual(count, 1)

            manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            manifest.update(
                {
                    "records_sha256": records_sha,
                    "record_shard_index": "records/index.json",
                    "record_shard_index_sha256": index_sha,
                }
            )
            (root / "run_manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            checksum_rows = {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*")
                if path.is_file() and path.name != "checksums.json"
            }
            (root / "checksums.json").write_text(json.dumps(checksum_rows) + "\n", encoding="utf-8")
            self.assertEqual(verify_attempt(root)["verdict"], "PASS")

            index = json.loads((root / "records" / "index.json").read_text(encoding="utf-8"))
            shard = root / index["shards"][0]["path"]
            shard.write_bytes(shard.read_bytes() + b"tamper")
            with self.assertRaisesRegex(VerificationError, "record shard compressed checksum mismatch"):
                verify_attempt(root, require_checksums=False)

    def test_missing_scientific_gate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_verifier_fixture(root, _valid_record())
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            summary.pop("gates")
            (root / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
            with self.assertRaises(VerificationError):
                verify_attempt(root)

        record = _valid_record()
        record["schedule_index"] = 3
        with self.assertRaises(VerificationError):
            validate_record(record)
