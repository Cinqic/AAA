"""Tests for the 17-parameter tiny neural learner.

These are correctness tests for an exploratory learner. They check the things
that would silently invalidate every number the Playground reports: the
parameter count, determinism, the causal input contract, the gradients, and
the frozen/online distinction.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from aaa.predictors import reflect_prediction, unfold_observation
from playground.neural import (
    HIDDEN_SIZE,
    INPUT_SIZE,
    OUTPUT_SIZE,
    PARAMETER_COUNT,
    TinyMLPPredictor,
)

BOUNDS = {"lower_bound": 0.0, "upper_bound": 1.0}
SCALE = 0.02 * 0.2


def make_model(**overrides) -> TinyMLPPredictor:
    keywords = {
        "model_seed": 900_711,
        "displacement_scale": SCALE,
        "learning_rate": 0.05,
        **BOUNDS,
    }
    keywords.update(overrides)
    return TinyMLPPredictor(**keywords)


class ArchitectureTests(unittest.TestCase):
    def test_parameter_count_is_exactly_seventeen(self) -> None:
        # The experiment is specifically about a 17-parameter learner. If this
        # number moves, the experiment is a different experiment.
        model = make_model()
        self.assertEqual(PARAMETER_COUNT, 17)
        self.assertEqual(model.parameter_count, 17)
        self.assertEqual(model.W1.shape, (HIDDEN_SIZE, INPUT_SIZE))
        self.assertEqual(model.b1.shape, (HIDDEN_SIZE,))
        self.assertEqual(model.W2.shape, (OUTPUT_SIZE, HIDDEN_SIZE))
        self.assertEqual(model.b2.shape, (OUTPUT_SIZE,))
        self.assertEqual(model.W1.size + model.b1.size + model.W2.size + model.b2.size, 17)
        self.assertEqual(model.parameter_vector().shape, (17,))

    def test_architecture_constants_match_the_declared_shape(self) -> None:
        self.assertEqual((INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE), (2, 4, 1))


class DeterminismTests(unittest.TestCase):
    def test_same_model_seed_gives_identical_initialization(self) -> None:
        first = make_model(model_seed=1234)
        second = make_model(model_seed=1234)
        np.testing.assert_array_equal(first.parameter_vector(), second.parameter_vector())

    def test_different_model_seed_can_give_a_different_initialization(self) -> None:
        first = make_model(model_seed=1234)
        second = make_model(model_seed=5678)
        self.assertFalse(np.array_equal(first.parameter_vector(), second.parameter_vector()))

    def test_reset_restores_the_exact_initial_state(self) -> None:
        model = make_model()
        initial = model.parameter_vector().copy()
        for step in range(20):
            model.update((0.4, 0.42 + 0.001 * step), 0.44 + 0.001 * step)
        self.assertGreater(model.update_count, 0)
        self.assertFalse(np.array_equal(model.parameter_vector(), initial))
        model.reset()
        np.testing.assert_array_equal(model.parameter_vector(), initial)
        self.assertEqual(model.update_count, 0)
        self.assertEqual(model.cumulative_gradient_norm, 0.0)
        self.assertIsNone(model.last_loss)

    def test_model_rng_is_independent_of_the_global_numpy_state(self) -> None:
        np.random.seed(0)
        first = make_model(model_seed=42).parameter_vector()
        np.random.seed(999)
        second = make_model(model_seed=42).parameter_vector()
        np.testing.assert_array_equal(first, second)


class PredictionTests(unittest.TestCase):
    def test_prediction_is_finite_and_inside_the_bounds_when_reflecting(self) -> None:
        model = make_model()
        value = model.predict((0.4, 0.45))
        self.assertTrue(math.isfinite(value))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_prediction_does_not_mutate_trainable_state(self) -> None:
        model = make_model()
        before = model.parameter_vector().copy()
        for _ in range(5):
            model.predict((0.3, 0.33))
        np.testing.assert_array_equal(model.parameter_vector(), before)
        self.assertEqual(model.update_count, 0)
        self.assertIsNone(model.last_loss)
        self.assertEqual(model.cumulative_gradient_norm, 0.0)

    def test_reflection_uses_the_shared_public_aaa_utility(self) -> None:
        model = make_model()
        history = (0.9, 0.98)
        raw = model.raw_predict(history)
        reflected = model.predict(history)
        self.assertAlmostEqual(reflected, reflect_prediction(raw, 0.0, 1.0), places=15)
        self.assertAlmostEqual(model.last_raw_prediction or 0.0, raw, places=15)

    def test_reflection_can_be_disabled_and_then_raw_is_returned(self) -> None:
        model = make_model(reflect=False)
        history = (0.9, 0.99)
        self.assertAlmostEqual(model.predict(history), model.raw_predict(history), places=15)

    def test_features_use_only_the_last_two_observations(self) -> None:
        model = make_model()
        short = model.features((0.30, 0.34))
        long = model.features((0.01, 0.99, 0.30, 0.34))
        np.testing.assert_array_equal(short, long)

    def test_features_are_the_declared_public_normalizations(self) -> None:
        model = make_model()
        history = (0.30, 0.34)
        features = model.features(history)
        self.assertAlmostEqual(features[0], (0.34 - 0.30) / SCALE, places=12)
        self.assertAlmostEqual(features[1], (0.34 - 0.5) / 1.0, places=12)

    def test_history_shorter_than_two_is_rejected(self) -> None:
        model = make_model()
        with self.assertRaises(ValueError):
            model.predict((0.4,))

    def test_non_finite_history_is_rejected(self) -> None:
        model = make_model()
        with self.assertRaises(ValueError):
            model.predict((0.4, float("nan")))


class CausalInputTests(unittest.TestCase):
    def test_the_predictor_interface_accepts_only_a_history_sequence(self) -> None:
        # The model must be unable to receive evaluator metadata, because there
        # is nowhere to put it: predict() and update() take a history and a
        # revealed target, and nothing else.
        import inspect

        predict = inspect.signature(TinyMLPPredictor.predict)
        update = inspect.signature(TinyMLPPredictor.update)
        self.assertEqual(list(predict.parameters), ["self", "history"])
        self.assertEqual(list(update.parameters), ["self", "history", "target_position"])

    def test_no_evaluator_metadata_field_exists_on_the_model(self) -> None:
        model = make_model()
        model.predict((0.4, 0.45))
        forbidden = ("scenario", "velocity", "bounced", "changed", "change_step", "omega", "damping")
        attributes = set(vars(model))
        for name in forbidden:
            self.assertNotIn(name, attributes)


class GradientTests(unittest.TestCase):
    """Finite-difference verification of the handwritten backpropagation.

    Handwritten gradients that *look* right are exactly the kind of thing that
    silently produces a plausible-looking but wrong result, so the analytical
    gradient is compared against central differences of the same loss. The
    tolerance is tight for float64 central differences at this step size.
    """

    TOLERANCE = 1e-8

    def _numerical_gradient(
        self, model: TinyMLPPredictor, history, target: float, epsilon: float = 1e-6
    ) -> np.ndarray:
        base = model.parameter_vector().copy()
        gradient = np.zeros_like(base)
        for index in range(base.size):
            forward = base.copy()
            forward[index] += epsilon
            model.set_parameter_vector(forward)
            high = model.loss_for(history, target)

            backward = base.copy()
            backward[index] -= epsilon
            model.set_parameter_vector(backward)
            low = model.loss_for(history, target)

            gradient[index] = (high - low) / (2 * epsilon)
        model.set_parameter_vector(base)
        return gradient

    def _analytical_vector(self, model: TinyMLPPredictor, history, target: float) -> np.ndarray:
        grads, _ = model.gradients(history, target)
        return np.concatenate(
            [
                grads["W1"].reshape(-1),
                grads["b1"].reshape(-1),
                grads["W2"].reshape(-1),
                grads["b2"].reshape(-1),
            ]
        )

    def test_analytical_gradients_match_finite_differences(self) -> None:
        cases = [
            ((0.30, 0.34), 0.38),
            ((0.62, 0.58), 0.54),
            ((0.50, 0.50), 0.50),
            ((0.11, 0.09), 0.07),
        ]
        for seed in (11, 4242):
            for history, target in cases:
                with self.subTest(seed=seed, history=history):
                    model = make_model(model_seed=seed)
                    # Move off the initialization so the check is not run at a
                    # point where several gradients happen to be zero.
                    model.update((0.4, 0.45), 0.5)
                    analytical = self._analytical_vector(model, history, target)
                    numerical = self._numerical_gradient(model, history, target)
                    np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=self.TOLERANCE)

    def test_analytical_gradients_match_finite_differences_without_reflection(self) -> None:
        model = make_model(model_seed=7, reflect=False, unfold_target=False)
        model.update((0.4, 0.45), 0.5)
        history, target = (0.72, 0.80), 0.88
        np.testing.assert_allclose(
            self._analytical_vector(model, history, target),
            self._numerical_gradient(model, history, target),
            rtol=1e-5,
            atol=self.TOLERANCE,
        )

    def test_gradient_sign_moves_the_output_toward_the_target(self) -> None:
        model = make_model(model_seed=3)
        history, target = (0.40, 0.44), 0.50
        before = model.loss_for(history, target)
        model.update(history, target)
        after = model.loss_for(history, target)
        self.assertLess(after, before)

    def test_zero_error_produces_zero_gradient(self) -> None:
        model = make_model()
        history = (0.40, 0.44)
        exact = model.raw_predict(history)
        grads, loss = model.gradients(history, exact)
        self.assertAlmostEqual(loss, 0.0, places=18)
        for array in grads.values():
            np.testing.assert_allclose(array, np.zeros_like(array), atol=1e-15)


class TargetTests(unittest.TestCase):
    def test_unfolded_target_uses_the_shared_public_aaa_utility(self) -> None:
        model = make_model()
        history = (0.90, 0.99)
        target_position = 0.97  # a folded observation after a wall contact
        raw = model.raw_predict(history)
        expected = (unfold_observation(target_position, raw, 0.0, 1.0) - history[-1]) / 1.0
        self.assertAlmostEqual(model.normalized_target(history, target_position), expected, places=15)

    def test_unfolding_can_be_disabled(self) -> None:
        model = make_model(unfold_target=False)
        history = (0.90, 0.99)
        self.assertAlmostEqual(model.normalized_target(history, 0.97), (0.97 - 0.99) / 1.0, places=15)

    def test_non_finite_target_is_rejected(self) -> None:
        model = make_model()
        with self.assertRaises(ValueError):
            model.update((0.4, 0.45), float("inf"))


class UpdateTests(unittest.TestCase):
    def test_update_changes_parameters_and_counters(self) -> None:
        model = make_model()
        before = model.parameter_vector().copy()
        model.update((0.40, 0.44), 0.50)
        self.assertFalse(np.array_equal(model.parameter_vector(), before))
        self.assertEqual(model.update_count, 1)
        self.assertIsNotNone(model.last_loss)
        self.assertGreater(model.cumulative_gradient_norm, 0.0)

    def test_a_frozen_instance_never_changes(self) -> None:
        model = make_model(update_enabled=False)
        before = model.parameter_vector().copy()
        for step in range(50):
            model.predict((0.4, 0.42 + 0.001 * step))
            model.update((0.4, 0.42 + 0.001 * step), 0.9)
        np.testing.assert_array_equal(model.parameter_vector(), before)
        self.assertEqual(model.update_count, 0)
        self.assertEqual(model.cumulative_gradient_norm, 0.0)

    def test_a_diverging_learning_rate_fails_loudly_rather_than_resetting(self) -> None:
        model = make_model(learning_rate=1e9, reflect=False, unfold_target=False)
        with self.assertRaises((FloatingPointError, ValueError, OverflowError)):
            for step in range(200):
                model.update((0.4, 0.44), 0.9 - 0.001 * step)


class SerializationTests(unittest.TestCase):
    def test_state_round_trip_is_numerically_exact(self) -> None:
        model = make_model()
        for step in range(15):
            model.update((0.40, 0.42 + 0.002 * step), 0.46 + 0.002 * step)
        restored = TinyMLPPredictor.from_state_dict(model.state_dict(), update_enabled=True)
        np.testing.assert_array_equal(restored.parameter_vector(), model.parameter_vector())
        self.assertEqual(restored.update_count, model.update_count)
        self.assertEqual(restored.learning_rate, model.learning_rate)
        self.assertEqual(restored.model_seed, model.model_seed)
        self.assertEqual(restored.reflect, model.reflect)
        self.assertEqual(restored.unfold_target, model.unfold_target)
        self.assertEqual(restored.cumulative_gradient_norm, model.cumulative_gradient_norm)
        self.assertEqual(restored.predict((0.5, 0.52)), model.predict((0.5, 0.52)))

    def test_state_dict_declares_the_architecture_and_parameter_count(self) -> None:
        state = make_model().state_dict()
        self.assertEqual(state["architecture"], [2, 4, 1])
        self.assertEqual(state["parameter_count"], 17)
        self.assertEqual(state["format_version"], TinyMLPPredictor.format_version)

    def test_clone_begins_numerically_identical(self) -> None:
        model = make_model()
        for step in range(10):
            model.update((0.40, 0.42 + 0.002 * step), 0.46)
        frozen = model.clone(name="frozen", update_enabled=False)
        online = model.clone(name="online", update_enabled=True)
        np.testing.assert_array_equal(frozen.parameter_vector(), model.parameter_vector())
        np.testing.assert_array_equal(online.parameter_vector(), frozen.parameter_vector())
        self.assertFalse(frozen.update_enabled)
        self.assertTrue(online.update_enabled)

    def test_clones_are_independent_of_each_other(self) -> None:
        model = make_model()
        frozen = model.clone(name="frozen", update_enabled=False)
        online = model.clone(name="online", update_enabled=True)
        before = frozen.parameter_vector().copy()
        for step in range(30):
            online.update((0.40, 0.42 + 0.002 * step), 0.50)
            frozen.update((0.40, 0.42 + 0.002 * step), 0.50)
        np.testing.assert_array_equal(frozen.parameter_vector(), before)
        self.assertFalse(np.array_equal(online.parameter_vector(), before))

    def test_unknown_format_version_is_rejected(self) -> None:
        state = make_model().state_dict()
        state["format_version"] = "something.else.v9"
        with self.assertRaises(ValueError):
            TinyMLPPredictor.from_state_dict(state)

    def test_missing_required_field_is_rejected(self) -> None:
        state = make_model().state_dict()
        del state["parameters"]
        with self.assertRaises(ValueError):
            TinyMLPPredictor.from_state_dict(state)

    def test_a_different_architecture_is_rejected(self) -> None:
        state = make_model().state_dict()
        state["architecture"] = [2, 8, 1]
        with self.assertRaises(ValueError):
            TinyMLPPredictor.from_state_dict(state)

    def test_non_finite_parameters_are_rejected(self) -> None:
        state = make_model().state_dict()
        state["parameters"]["W1"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            TinyMLPPredictor.from_state_dict(state)

    def test_wrong_parameter_shape_is_rejected(self) -> None:
        state = make_model().state_dict()
        state["parameters"]["b1"] = [0.0, 0.0]
        with self.assertRaises(ValueError):
            TinyMLPPredictor.from_state_dict(state)


class ConstructionTests(unittest.TestCase):
    def test_invalid_construction_arguments_are_rejected(self) -> None:
        for overrides in (
            {"model_seed": True},
            {"model_seed": "seven"},
            {"learning_rate": 0.0},
            {"learning_rate": -1.0},
            {"displacement_scale": 0.0},
            {"lower_bound": 1.0, "upper_bound": 1.0},
            {"update_count": -1},
            {"cumulative_gradient_norm": -1.0},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                make_model(**overrides)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
