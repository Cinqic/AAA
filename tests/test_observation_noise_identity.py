"""Adversarial tests for the non-self-referential noise scientific identity."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aaa.noise.candidates import (
    CLIP_CANDIDATE_IDS,
    INCUMBENT_CANDIDATE_ID,
    candidate_catalog,
    candidate_from_model,
)
from aaa.noise.joint import (
    JointEvaluationError,
    _hierarchical_draws,
    _hierarchical_ratio_draws,
    _ratio_of_means,
    evaluate_joint_archives,
)
from aaa.noise.predictors import incumbent_from_v21
from aaa.noise.reservation import ReservationError, reserve_confirmation_batch
from aaa.noise.scientific_identity import (
    ScientificIdentityError,
    fingerprints_equal,
    scientific_fingerprint,
)
from aaa.noise.selection import _complete_primary_cells, _evidence_destination


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)


class ScientificIdentityTests(unittest.TestCase):
    def _repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        _git(root, "init", "--initial-branch=main")
        _git(root, "config", "user.email", "test@example.invalid")
        _git(root, "config", "user.name", "AAA tests")
        (root / "aaa").mkdir()
        (root / "aaa" / "runner.py").write_text("scientific runner\n", encoding="utf-8")
        (root / "analysis.py").write_text("analysis\n", encoding="utf-8")
        (root / "candidate.py").write_text("candidate\n", encoding="utf-8")
        (root / "verifier.py").write_text("verifier\n", encoding="utf-8")
        (root / "benchmarks").mkdir()
        (root / "benchmarks" / "observation_noise_registry.json").write_text(
            json.dumps(
                {
                    "batches": [
                        {
                            "batch_id": "a",
                            "namespace": "frozen-namespace",
                            "status": "planned",
                            "consumed_by": None,
                            "outcome": None,
                            "claimed_by": None,
                            "claim_started_at": None,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        _git(root, "add", ".")
        return temporary, root

    def test_scientific_edits_change_the_fingerprint(self):
        temporary, root = self._repo()
        with temporary:
            first = scientific_fingerprint(root, require_project_shape=False)
            for relative in ("aaa/runner.py", "candidate.py", "analysis.py", "verifier.py"):
                path = root / relative
                path.write_text(path.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
                changed = scientific_fingerprint(root, require_project_shape=False)
                self.assertNotEqual(first["sha256"], changed["sha256"], relative)
                path.write_text(path.read_text(encoding="utf-8").replace("changed\n", ""), encoding="utf-8")

    def test_registry_status_is_normalized_but_batch_declaration_is_not(self):
        temporary, root = self._repo()
        with temporary:
            first = scientific_fingerprint(root, require_project_shape=False)
            path = root / "benchmarks" / "observation_noise_registry.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["batches"][0].update(
                {
                    "status": "consumed",
                    "consumed_by": "attempt-a",
                    "outcome": "failed",
                    "claimed_by": "runner",
                    "claim_started_at": "2026-09-14T00:00:00Z",
                }
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(
                fingerprints_equal(first, scientific_fingerprint(root, require_project_shape=False))
            )
            payload["batches"][0]["namespace"] = "changed-namespace"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertNotEqual(
                first["sha256"], scientific_fingerprint(root, require_project_shape=False)["sha256"]
            )

    def test_generated_and_exact_freeze_files_do_not_self_invalidate(self):
        temporary, root = self._repo()
        with temporary:
            first = scientific_fingerprint(root, require_project_shape=False)
            freeze = root / "benchmarks" / "observation_noise_freeze.json"
            source_freeze = root / "benchmarks" / "observation_noise_source_freeze.json"
            freeze.write_text("freeze one\n", encoding="utf-8")
            source_freeze.write_text("source one\n", encoding="utf-8")
            (root / "results").mkdir()
            (root / "results" / "generated.json").write_text("result\n", encoding="utf-8")
            self.assertTrue(
                fingerprints_equal(first, scientific_fingerprint(root, require_project_shape=False))
            )
            freeze.write_text("freeze two\n", encoding="utf-8")
            source_freeze.write_text("source two\n", encoding="utf-8")
            self.assertTrue(
                fingerprints_equal(first, scientific_fingerprint(root, require_project_shape=False))
            )

    def test_nonignored_untracked_scientific_file_and_symlink_fail_closed(self):
        temporary, root = self._repo()
        with temporary:
            (root / "new_scientific.py").write_text("new\n", encoding="utf-8")
            with self.assertRaises(ScientificIdentityError):
                scientific_fingerprint(root, require_project_shape=False)
            (root / "new_scientific.py").unlink()
            target = root / "candidate.py"
            target.unlink()
            target.symlink_to("analysis.py")
            _git(root, "add", "candidate.py")
            with self.assertRaises(ScientificIdentityError):
                scientific_fingerprint(root, require_project_shape=False)

    def test_candidate_catalog_is_immutable_and_refinement_is_causal(self):
        catalog = candidate_catalog()
        self.assertEqual(list(catalog), [INCUMBENT_CANDIDATE_ID, *CLIP_CANDIDATE_IDS])
        self.assertEqual(len({entry.configuration_hash for entry in catalog.values()}), 4)
        model = incumbent_from_v21(name="base", update_enabled=True)
        incumbent = candidate_from_model(model, INCUMBENT_CANDIDATE_ID, name="incumbent", update_enabled=True)
        clipped = candidate_from_model(model, CLIP_CANDIDATE_IDS[0], name="clipped", update_enabled=True)
        history = (0.20, 0.30, 0.40, 0.50)
        incumbent.predict(history)
        clipped.predict(history)
        incumbent.update(history, 2.0)
        clipped.update(history, 2.0)
        self.assertNotEqual(incumbent.state_dict()["weights"], clipped.state_dict()["weights"])
        restored = type(clipped).from_state_dict(clipped.state_dict(), update_enabled=True)
        self.assertEqual(clipped.state_dict(), restored.state_dict())

    def test_selection_cell_coverage_accepts_legitimate_zero_metrics(self):
        cell = {
            "selected_candidate_online": [0.0, 1.0],
            "selected_candidate_frozen": [0.0, 1.0],
            "constant_motion_reflected": [0.0, 1.0],
            "incumbent_square_root_rls": [0.0, 1.0],
        }
        self.assertTrue(_complete_primary_cells({("cell",): cell}, expected_cells=1))
        del cell["constant_motion_reflected"]
        self.assertFalse(_complete_primary_cells({("cell",): cell}, expected_cells=1))

    def test_quick_selection_cannot_replace_canonical_full_evidence(self):
        canonical = Path("docs/evidence/observation_noise_development_selection.json")
        with self.assertRaisesRegex(ValueError, "cannot overwrite"):
            _evidence_destination(canonical, quick=True)
        self.assertEqual(_evidence_destination(canonical, quick=False), canonical)

    def test_joint_hierarchy_sampler_and_archive_admission_fail_closed(self):
        array = np.ones((10, 32, 3), dtype=float)
        sample = _hierarchical_draws(array, 32, np.random.default_rng(5))
        self.assertEqual(sample.shape, (32,))
        self.assertTrue((sample == 1.0).all())
        with self.assertRaises(JointEvaluationError):
            evaluate_joint_archives("/does/not/exist/a", "/does/not/exist/b")

    def test_adaptation_uses_ratio_of_means_not_mean_of_ratios(self):
        frozen = np.ones((10, 32, 3), dtype=float)
        online = np.ones((10, 32, 3), dtype=float)
        frozen[5:] = 3.0
        online[5:] = 1.5
        numerator = frozen - online
        self.assertAlmostEqual(_ratio_of_means(numerator, frozen), 0.375)
        self.assertAlmostEqual(float(np.mean(numerator / frozen)), 0.25)
        draws = _hierarchical_ratio_draws(numerator, frozen, 64, np.random.default_rng(7))
        self.assertEqual(draws.shape, (64,))
        self.assertTrue(np.isfinite(draws).all())

    def test_remote_reservation_is_atomic_across_separate_clones(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "remote.git"
            source = root / "source"
            clone_a = root / "clone-a"
            clone_b = root / "clone-b"
            _git(root, "init", "--bare", str(remote))
            _git(root, "init", "--initial-branch=main", str(source))
            _git(source, "config", "user.email", "test@example.invalid")
            _git(source, "config", "user.name", "AAA tests")
            (source / "README").write_text("fixture\n", encoding="utf-8")
            _git(source, "add", "README")
            _git(source, "commit", "-m", "fixture")
            _git(source, "remote", "add", "origin", str(remote))
            _git(source, "push", "-u", "origin", "main")
            _git(root, "clone", "--branch", "main", str(remote), str(clone_a))
            _git(root, "clone", "--branch", "main", str(remote), str(clone_b))
            for clone in (clone_a, clone_b):
                _git(clone, "config", "user.email", "test@example.invalid")
                _git(clone, "config", "user.name", "AAA tests")
            claim = reserve_confirmation_batch(
                clone_a,
                batch_id="a-0001",
                role="confirmation_a",
                attempt_id="attempt-a",
                scientific_fingerprint_sha256="1" * 64,
                resume=False,
            )
            self.assertEqual(claim["attempt_id"], "attempt-a")
            with self.assertRaises(ReservationError):
                reserve_confirmation_batch(
                    clone_b,
                    batch_id="a-0001",
                    role="confirmation_a",
                    attempt_id="attempt-b",
                    scientific_fingerprint_sha256="1" * 64,
                    resume=False,
                )
            resumed = reserve_confirmation_batch(
                clone_b,
                batch_id="a-0001",
                role="confirmation_a",
                attempt_id="attempt-a",
                scientific_fingerprint_sha256="1" * 64,
                resume=True,
            )
            self.assertEqual(resumed, claim)


if __name__ == "__main__":
    unittest.main()
