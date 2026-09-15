"""Tests for the headless tiny neural AAA experiment.

The experiment's whole value rests on one structural property: the frozen and
online arms must start from the same numbers and then see the same world. If
that is not true, every number it prints is meaningless, so it is tested
directly rather than assumed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aaa.experiment import read_step_records
from playground.experiment import (
    BASELINE_ARMS,
    BRANCH_ARMS,
    DEVELOPMENT_SEEDS,
    EVALUATION_SEEDS,
    LEARNING_RATE_GRID,
    TinyExperimentConfig,
    _state_hash,
    build_parser,
    main,
    model_seed_for,
    normalized_errors,
    run_experiment,
    run_seed,
    select_learning_rate,
)
from playground.metrics import mean_absolute_error
from playground.neural import PARAMETER_COUNT

TINY = TinyExperimentConfig(prefix_steps=40, branch_steps=20, first_post_change_window=10)


class SeedDisciplineTests(unittest.TestCase):
    def test_development_and_evaluation_seeds_are_disjoint(self) -> None:
        self.assertEqual(set(DEVELOPMENT_SEEDS) & set(EVALUATION_SEEDS), set())

    def test_model_seeds_never_collide_with_environment_seeds(self) -> None:
        config = TinyExperimentConfig()
        environment = set(DEVELOPMENT_SEEDS) | set(EVALUATION_SEEDS)
        model = {model_seed_for(config, seed) for seed in environment}
        self.assertEqual(environment & model, set())
        self.assertEqual(len(model), len(environment))


class BranchStructureTests(unittest.TestCase):
    """The matched frozen/online design, verified rather than asserted."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.summary, cls.prefix, cls.branch = run_seed(TINY, 701, role="playground_development")

    def test_both_branches_start_from_one_identical_learner_state(self) -> None:
        # The branch state hash is computed once, before cloning, and both arms
        # are built from that same state dictionary.
        self.assertIsInstance(self.summary["branch_state_hash"]["tiny_nn"], str)
        self.assertEqual(len(self.summary["branch_state_hash"]["tiny_nn"]), 64)
        first = self.branch[0]
        # At the very first branch transition neither arm has updated yet, so
        # identical starting state must produce an identical prediction.
        self.assertAlmostEqual(
            first.predictions["tiny_nn_frozen"]["raw"],
            first.predictions["tiny_nn_online"]["raw"],
            places=15,
        )
        self.assertAlmostEqual(
            first.predictions["rls_frozen"]["raw"], first.predictions["rls_online"]["raw"], places=15
        )

    def test_both_branches_consume_identical_post_change_observations(self) -> None:
        # One environment serves both arms, so the revealed trajectory is by
        # construction the same sequence for every arm in the branch.
        for record in self.branch:
            self.assertEqual(set(record.predictions), set(BRANCH_ARMS) | set(BASELINE_ARMS))
        positions = [record.actual_next_position for record in self.branch]
        self.assertEqual(len(positions), len(set(range(len(positions)))))
        self.assertTrue(all(np.isfinite(positions)))

    def test_the_branch_continues_the_prefix_history(self) -> None:
        tail = self.prefix[-1]
        expected = (*tail.history[1:], tail.actual_next_position)
        self.assertEqual(self.branch[0].history, expected)
        self.assertEqual(self.branch[0].step, TINY.prefix_steps)

    def test_the_frozen_arm_does_not_update_and_the_online_arm_does(self) -> None:
        self.assertEqual(self.summary["branch_updates"]["tiny_nn_frozen"], 0)
        self.assertEqual(self.summary["branch_updates"]["tiny_nn_online"], len(self.branch))
        self.assertEqual(self.summary["branch_updates"]["rls_frozen"], 0)
        self.assertGreater(self.summary["branch_updates"]["rls_online"], 0)

    def test_the_frozen_arm_diverges_from_the_online_arm_over_the_branch(self) -> None:
        # If the two arms produced the same predictions throughout, the
        # experiment would be measuring nothing.
        differences = [
            abs(record.predictions["tiny_nn_online"]["raw"] - record.predictions["tiny_nn_frozen"]["raw"])
            for record in self.branch
        ]
        self.assertGreater(max(differences), 0.0)

    def test_no_numerical_failures_and_no_leaked_metadata(self) -> None:
        self.assertEqual(self.summary["failures"], [])
        for record in self.prefix + self.branch:
            # ``history`` is the entire predictor input. Event flags live on the
            # record for reporting and are not part of it.
            self.assertEqual(len(record.history), 4)
            self.assertTrue(all(np.isfinite(record.history)))

    def test_prefix_arms_actually_learned_during_the_prefix(self) -> None:
        self.assertEqual(self.summary["prefix_updates"]["tiny_nn"], len(self.prefix))
        self.assertGreater(self.summary["tiny_cumulative_gradient_norm"]["prefix"], 0.0)

    def test_metric_arithmetic_recomputes_from_the_retained_records(self) -> None:
        window = TINY.first_post_change_window
        for arm in BRANCH_ARMS:
            errors = normalized_errors(self.branch, arm)
            self.assertAlmostEqual(
                self.summary["post_change_mae"][arm], mean_absolute_error(errors), places=15
            )
            self.assertAlmostEqual(
                self.summary[f"first_{window}_post_change_mae"][arm],
                mean_absolute_error(errors[:window]),
                places=15,
            )
        prefix_errors = normalized_errors(self.prefix, "tiny_nn_prefix")
        self.assertAlmostEqual(
            self.summary["pre_change_mae"]["tiny_nn"], mean_absolute_error(prefix_errors), places=15
        )
        self.assertAlmostEqual(
            self.summary["overall_mae"]["tiny_nn_online"],
            mean_absolute_error(prefix_errors + normalized_errors(self.branch, "tiny_nn_online")),
            places=15,
        )

    def test_per_step_errors_match_the_recorded_predictions(self) -> None:
        for record in self.branch:
            width = record.interval_width
            for prediction in record.predictions.values():
                self.assertAlmostEqual(
                    prediction["normalized_absolute_error"],
                    abs(prediction["scored"] - record.actual_next_position) / width,
                    places=15,
                )


class DeterminismTests(unittest.TestCase):
    def test_a_seed_reproduces_exactly(self) -> None:
        first, _, first_branch = run_seed(TINY, 702, role="playground_development")
        second, _, second_branch = run_seed(TINY, 702, role="playground_development")
        self.assertEqual(
            json.dumps(first, sort_keys=True, default=str),
            json.dumps(second, sort_keys=True, default=str),
        )
        self.assertEqual(
            [record.actual_next_position for record in first_branch],
            [record.actual_next_position for record in second_branch],
        )

    def test_different_seeds_produce_different_trajectories(self) -> None:
        _, _, first = run_seed(TINY, 702, role="playground_development")
        _, _, second = run_seed(TINY, 703, role="playground_development")
        self.assertNotEqual(
            [record.actual_next_position for record in first],
            [record.actual_next_position for record in second],
        )

    def test_state_hash_ignores_only_the_arm_name(self) -> None:
        base = {"name": "a", "value": 1}
        self.assertEqual(_state_hash(base), _state_hash({"name": "b", "value": 1}))
        self.assertNotEqual(_state_hash(base), _state_hash({"name": "a", "value": 2}))


class DocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.document = run_experiment(
            TINY,
            (701, 702),
            role="playground_development",
            output_root=Path(cls.directory.name),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_the_document_declares_itself_exploratory(self) -> None:
        self.assertEqual(self.document["status"], "exploratory")
        self.assertIn("not a confirmation run", self.document["not_a_benchmark"])
        self.assertEqual(self.document["model"]["parameter_count"], PARAMETER_COUNT)

    def test_the_document_carries_provenance_and_configuration(self) -> None:
        source = self.document["source"]
        self.assertIn("commit", source)
        self.assertIn("dirty", source)
        self.assertEqual(self.document["configuration"]["prefix_steps"], TINY.prefix_steps)
        self.assertEqual(self.document["environment_seeds"], [701, 702])
        self.assertEqual(self.document["model_seeds"], [model_seed_for(TINY, 701), model_seed_for(TINY, 702)])

    def test_per_seed_values_are_reported_alongside_aggregates(self) -> None:
        self.assertEqual(len(self.document["per_seed"]), 2)
        window = TINY.first_post_change_window
        aggregate = self.document["aggregate"][f"first_{window}_post_change_mae"]
        for arm in BRANCH_ARMS:
            self.assertIn(arm, aggregate)
            # Variability is reported, not collapsed into a single number.
            self.assertIn("min", aggregate[arm])
            self.assertIn("max", aggregate[arm])
            self.assertEqual(aggregate[arm]["seeds"], 2)

    def test_the_fair_reflected_baseline_is_always_present(self) -> None:
        # The tiny network reflects, so the reflected baseline must be in every
        # comparison table, not only the weaker unreflected one.
        for key in ("pre_change_mae", "post_change_mae", "overall_mae"):
            for seed in self.document["per_seed"]:
                self.assertIn("constant_motion_reflected", seed[key])
                self.assertIn("constant_motion", seed[key])

    def test_claim_boundaries_are_kept_separate(self) -> None:
        boundaries = self.document["claim_boundaries"]
        self.assertEqual(
            set(boundaries),
            {
                "implementation_works",
                "model_learns_on_some_stream",
                "model_improves_over_its_frozen_copy",
                "model_generalizes",
                "model_beats_a_baseline",
            },
        )
        self.assertIn("NOT addressed", boundaries["model_generalizes"])

    def test_no_overclaiming_language_appears_outside_a_disclaimer(self) -> None:
        """Banned phrases may appear only inside a sentence that negates them."""

        banned = (
            "understands motion",
            "understand motion",
            "learned physics",
            "is intelligent",
            "general intelligence",
            "proves general adaptation",
            "benchmark v2.2",
            "validation of aaa",
            "evidence of understanding",
        )
        negations = ("not ", "nothing here", "never", "no acceptance", "not addressed")
        text = json.dumps(self.document).lower().replace("\\n", " ")
        sentences = [part for chunk in text.split(".") for part in chunk.split('","')]
        for phrase in banned:
            for sentence in sentences:
                if phrase not in sentence:
                    continue
                with self.subTest(phrase=phrase, sentence=sentence[:120]):
                    self.assertTrue(
                        any(marker in sentence for marker in negations),
                        f"{phrase!r} appears without a disclaimer in: {sentence[:200]}",
                    )

    def test_the_document_states_its_interpretation_limits(self) -> None:
        interpretation = self.document["interpretation"].lower()
        self.assertIn("fixed toy protocol", interpretation)
        self.assertIn("nothing here is evidence", interpretation)

    def test_generated_evidence_is_written_and_reloadable(self) -> None:
        root = Path(self.directory.name)
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["experiment"], self.document["experiment"])
        for seed in (701, 702):
            prefix = read_step_records(root / "steps" / f"seed-{seed}-prefix.jsonl")
            branch = read_step_records(root / "steps" / f"seed-{seed}-changed-law.jsonl")
            self.assertEqual(len(prefix), TINY.prefix_steps - 3)
            # ``continue_episode`` is handed a full observation window, so the
            # branch has no warm-up: every transition is scored.
            self.assertEqual(len(branch), TINY.branch_steps)
            self.assertEqual(branch[0].identity.branch, "changed-law")

    def test_head_to_head_comparisons_report_both_directions_honestly(self) -> None:
        comparisons = self.document["head_to_head_online_vs_frozen"]
        self.assertEqual(len(comparisons), 2)
        for item in comparisons:
            self.assertEqual(item["left"], "tiny_nn_online")
            self.assertEqual(item["right"], "tiny_nn_frozen")
            self.assertIsInstance(item["left_lower"], bool)
            self.assertAlmostEqual(item["difference"], item["left_mae"] - item["right_mae"], places=15)


class SelectionTests(unittest.TestCase):
    def test_learning_rate_selection_records_every_attempt(self) -> None:
        record = select_learning_rate(TINY, rates=(0.01, 0.1), seeds=(701,))
        self.assertEqual([attempt["learning_rate"] for attempt in record["attempts"]], [0.01, 0.1])
        self.assertIn(record["selected_learning_rate"], (0.01, 0.1))
        self.assertIn("Evaluation seeds were not consulted", record["note"])

    def test_a_diverging_rate_is_recorded_rather_than_crashing(self) -> None:
        record = select_learning_rate(TINY, rates=(0.01, 0.1, 1e6), seeds=(701,))
        failing = [attempt for attempt in record["attempts"] if attempt["failures"]]
        self.assertEqual(len(failing), 1)
        self.assertEqual(failing[0]["learning_rate"], 1e6)
        self.assertEqual(record["divergence_point"], 1e6)
        # The stability margin excludes the largest surviving rate.
        self.assertEqual(record["admissible_learning_rates"], [0.01])
        self.assertEqual(record["selected_learning_rate"], 0.01)

    def test_the_declared_grid_brackets_the_divergence_point(self) -> None:
        self.assertEqual(LEARNING_RATE_GRID[0], 0.003)
        self.assertEqual(LEARNING_RATE_GRID[-1], 3.0)
        self.assertEqual(list(LEARNING_RATE_GRID), sorted(LEARNING_RATE_GRID))


class CommandLineTests(unittest.TestCase):
    def test_quick_configuration_is_smaller_but_identical_in_shape(self) -> None:
        full = TinyExperimentConfig()
        quick = full.quick()
        self.assertLess(quick.prefix_steps, full.prefix_steps)
        self.assertLess(quick.branch_steps, full.branch_steps)
        self.assertEqual(quick.learning_rate, full.learning_rate)
        self.assertEqual(quick.reflect, full.reflect)

    def test_parser_defaults_to_the_evaluation_seed_group(self) -> None:
        arguments = build_parser().parse_args([])
        self.assertEqual(arguments.seeds, "evaluation")
        self.assertFalse(arguments.quick)

    def test_quick_development_run_completes_cleanly(self) -> None:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["--quick", "--seeds", "development", "--no-write"])
        self.assertEqual(code, 0)
        document = json.loads(buffer.getvalue())
        self.assertEqual(document["failures"], [])
        self.assertEqual(document["environment_seeds"], list(DEVELOPMENT_SEEDS))
        self.assertEqual(document["role"], "playground_development")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
