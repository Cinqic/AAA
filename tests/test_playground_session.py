"""Tests for the headless Playground session and the rendering smoke path.

Everything up to :class:`RenderingSmokeTests` runs with no Matplotlib import at
all, which is the point of keeping the state machine separate from the
renderer. The rendering test uses the Agg backend and never needs a display.
"""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace

import numpy as np

from aaa.predictors import (
    ConstantMotionPredictor,
    PersistencePredictor,
    ReflectedConstantMotionPredictor,
)
from playground.metrics import ErrorTrack, mean_absolute_error
from playground.session import (
    MODEL_SEED_OFFSET,
    PLAYGROUND_SCENARIOS,
    PREDICTOR_NAMES,
    PlaygroundSession,
    playground_world,
)


def run_steps(session: PlaygroundSession, count: int) -> None:
    for _ in range(count):
        session.step()


class ConstructionTests(unittest.TestCase):
    def test_every_declared_scenario_constructs_and_runs(self) -> None:
        for scenario in PLAYGROUND_SCENARIOS:
            with self.subTest(scenario=scenario):
                session = PlaygroundSession(scenario=scenario, seed=711)
                run_steps(session, 30)
                self.assertEqual(session.step_index, 30)

    def test_unknown_scenario_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PlaygroundSession(scenario="teleporting")

    def test_model_seed_is_derived_far_from_the_environment_seed(self) -> None:
        session = PlaygroundSession(seed=711)
        self.assertEqual(session.model_seed, MODEL_SEED_OFFSET + 711)
        self.assertNotEqual(session.model_seed, session.environment_seed)

    def test_explicit_model_seed_is_honoured(self) -> None:
        session = PlaygroundSession(seed=711, model_seed=4242)
        self.assertEqual(session.model_seed, 4242)
        self.assertEqual(session.tiny.model_seed, 4242)

    def test_the_session_imports_no_gui_toolkit(self) -> None:
        # The headless state machine must be usable with no renderer at all.
        import playground.session as module

        self.assertFalse(hasattr(module, "plt"))
        self.assertNotIn("matplotlib", [name for name in dir(module) if not name.startswith("_")])


class SteppingTests(unittest.TestCase):
    def test_one_step_is_exactly_one_environment_transition(self) -> None:
        session = PlaygroundSession(scenario="bouncing", seed=5)
        for expected in range(1, 25):
            session.step()
            self.assertEqual(session.step_index, expected)
            self.assertEqual(len(session.history), expected + 1)

    def test_predictions_are_issued_before_the_target_is_revealed(self) -> None:
        # The session's recorded prediction must be reproducible from the
        # history that existed *before* the transition, using the model state
        # that existed before the update. Recomputing it after the reveal
        # would produce a different number.
        session = PlaygroundSession(scenario="bouncing", seed=9)
        run_steps(session, 6)
        length = session.world.history_length
        window = tuple(session.history[-length:])
        state_before = session.tiny.state_dict()
        expected = {name: model.predict(window) for name, model in session.predictors.items()}

        frame = session.step()
        assert frame is not None
        self.assertTrue(frame.scored)
        for name, value in expected.items():
            self.assertAlmostEqual(frame.predictions[name], value, places=15)
        # ...and the window used was genuinely the pre-reveal window.
        self.assertEqual(window, tuple(session.history[-length - 1 : -1]))
        self.assertNotEqual(session.tiny.state_dict()["parameters"], state_before["parameters"])

    def test_scored_error_matches_the_recorded_prediction_and_target(self) -> None:
        session = PlaygroundSession(scenario="changed", seed=3)
        run_steps(session, 10)
        frame = session.last_frame
        assert frame is not None
        width = session.world.width
        for name, prediction in frame.predictions.items():
            self.assertAlmostEqual(
                frame.normalized_errors[name], abs(prediction - frame.actual) / width, places=15
            )

    def test_update_happens_after_the_reveal_not_before(self) -> None:
        session = PlaygroundSession(scenario="bouncing", seed=11)
        run_steps(session, 8)
        length = session.world.history_length
        window = tuple(session.history[-length:])
        model = session.tiny.clone(name="shadow", update_enabled=True)

        frame = session.step()
        assert frame is not None
        # An independent copy, updated with the same pre-reveal window and the
        # revealed target, must land on exactly the session's parameters.
        model.update(window, frame.actual)
        np.testing.assert_array_equal(model.parameter_vector(), session.tiny.parameter_vector())

    def test_warm_up_steps_are_not_scored(self) -> None:
        session = PlaygroundSession(scenario="straight", seed=2)
        length = session.world.history_length
        for _ in range(length - 1):
            frame = session.step()
            assert frame is not None
            self.assertFalse(frame.scored)
            self.assertEqual(frame.predictions, {})
        frame = session.step()
        assert frame is not None
        self.assertTrue(frame.scored)
        self.assertEqual(session.tiny.update_count, 1)

    def test_baseline_predictions_are_causally_aligned(self) -> None:
        # Persistence, constant motion and reflected constant motion are closed
        # forms of the pre-reveal window, so their recorded values must equal
        # the analytic value of that window exactly.
        session = PlaygroundSession(scenario="bouncing", seed=17)
        bounds = {"lower_bound": session.world.lower_bound, "upper_bound": session.world.upper_bound}
        for _ in range(40):
            length = session.world.history_length
            if len(session.history) >= length:
                window = tuple(session.history[-length:])
                expected = {
                    "persistence": PersistencePredictor().predict(window),
                    "constant_motion": ConstantMotionPredictor().predict(window),
                    "constant_motion_reflected": ReflectedConstantMotionPredictor(**bounds).predict(window),
                }
            else:
                expected = {}
            frame = session.step()
            assert frame is not None
            for name, value in expected.items():
                self.assertAlmostEqual(frame.predictions[name], value, places=15)

    def test_end_of_horizon_is_clean(self) -> None:
        session = PlaygroundSession(scenario="straight", seed=4)
        total = session.world.steps_per_episode
        run_steps(session, total + 25)
        self.assertEqual(session.step_index, total)
        self.assertTrue(session.finished)
        terminal = session.step()
        assert terminal is not None
        self.assertTrue(terminal.finished)
        self.assertFalse(terminal.scored)
        self.assertEqual(terminal.event, "end")
        self.assertEqual(len(session.history), total + 1)

    def test_error_tracks_have_one_entry_per_scored_step(self) -> None:
        session = PlaygroundSession(scenario="bouncing", seed=6)
        run_steps(session, 40)
        length = session.world.history_length
        scored = 40 - (length - 1)
        for name in PREDICTOR_NAMES:
            self.assertEqual(len(session.error_series(name)), scored)


class PauseTests(unittest.TestCase):
    def test_pause_changes_no_simulation_or_model_state(self) -> None:
        session = PlaygroundSession(scenario="dynamics_change", seed=8)
        run_steps(session, 15)
        session.toggle_pause()
        before = {
            "step": session.step_index,
            "history": list(session.history),
            "parameters": session.tiny.parameter_vector().copy(),
            "updates": session.tiny.update_count,
            "environment": session.environment.position,
            "errors": list(session.error_series("tiny_nn")),
        }
        for _ in range(30):
            self.assertIsNone(session.step())
        self.assertEqual(session.step_index, before["step"])
        self.assertEqual(session.history, before["history"])
        np.testing.assert_array_equal(session.tiny.parameter_vector(), before["parameters"])
        self.assertEqual(session.tiny.update_count, before["updates"])
        self.assertEqual(session.environment.position, before["environment"])
        self.assertEqual(list(session.error_series("tiny_nn")), before["errors"])

    def test_single_step_advances_exactly_once_while_paused(self) -> None:
        session = PlaygroundSession(scenario="bouncing", seed=12)
        run_steps(session, 10)
        session.toggle_pause()
        session.single_step()
        self.assertEqual(session.step_index, 11)
        self.assertTrue(session.paused)
        self.assertIsNone(session.step())

    def test_toggle_pause_reports_and_resumes(self) -> None:
        session = PlaygroundSession(seed=1)
        self.assertTrue(session.toggle_pause())
        self.assertFalse(session.toggle_pause())
        run_steps(session, 3)
        self.assertEqual(session.step_index, 3)


class ResetTests(unittest.TestCase):
    def test_reset_is_deterministic(self) -> None:
        session = PlaygroundSession(scenario="dynamics_change", seed=711)
        run_steps(session, 40)
        first_history = list(session.history)
        first_parameters = session.tiny.parameter_vector().copy()

        session.reset()
        self.assertEqual(session.step_index, 0)
        self.assertEqual(len(session.history), 1)
        self.assertEqual(session.tiny.update_count, 0)
        self.assertEqual(session.error_series("tiny_nn"), ())

        run_steps(session, 40)
        self.assertEqual(session.history, first_history)
        np.testing.assert_array_equal(session.tiny.parameter_vector(), first_parameters)

    def test_reset_restores_the_initial_model_parameters(self) -> None:
        session = PlaygroundSession(seed=711)
        initial = session.tiny.parameter_vector().copy()
        run_steps(session, 30)
        self.assertFalse(np.array_equal(session.tiny.parameter_vector(), initial))
        session.reset()
        np.testing.assert_array_equal(session.tiny.parameter_vector(), initial)

    def test_two_sessions_with_the_same_seeds_agree_exactly(self) -> None:
        first = PlaygroundSession(scenario="changed", seed=77)
        second = PlaygroundSession(scenario="changed", seed=77)
        run_steps(first, 60)
        run_steps(second, 60)
        self.assertEqual(first.history, second.history)
        np.testing.assert_array_equal(first.tiny.parameter_vector(), second.tiny.parameter_vector())
        self.assertEqual(first.error_series("tiny_nn"), second.error_series("tiny_nn"))

    def test_set_seed_and_set_scenario_reset_the_session(self) -> None:
        session = PlaygroundSession(scenario="bouncing", seed=21)
        run_steps(session, 20)
        session.set_seed(22)
        self.assertEqual(session.step_index, 0)
        self.assertEqual(session.environment_seed, 22)
        self.assertEqual(session.model_seed, MODEL_SEED_OFFSET + 22)

        run_steps(session, 20)
        session.set_scenario("straight")
        self.assertEqual(session.scenario, "straight")
        self.assertEqual(session.step_index, 0)
        self.assertEqual(session.world.steps_per_episode, playground_world("straight").steps_per_episode)


class OnlineFrozenTests(unittest.TestCase):
    def test_a_frozen_tiny_network_does_not_learn(self) -> None:
        session = PlaygroundSession(scenario="dynamics_change", seed=713, online=False)
        before = session.tiny.parameter_vector().copy()
        run_steps(session, 60)
        np.testing.assert_array_equal(session.tiny.parameter_vector(), before)
        self.assertEqual(session.tiny.update_count, 0)
        # ...while the AAA RLS learner, which is shown as the project's online
        # learner, keeps updating.
        self.assertGreater(session.rls.update_count, 0)

    def test_an_online_tiny_network_learns(self) -> None:
        session = PlaygroundSession(scenario="dynamics_change", seed=713, online=True)
        before = session.tiny.parameter_vector().copy()
        run_steps(session, 60)
        self.assertFalse(np.array_equal(session.tiny.parameter_vector(), before))
        self.assertGreater(session.tiny.update_count, 0)

    def test_freezing_midway_stops_learning_from_that_point(self) -> None:
        session = PlaygroundSession(scenario="dynamics_change", seed=714)
        run_steps(session, 30)
        session.set_online(False)
        frozen_at = session.tiny.parameter_vector().copy()
        updates = session.tiny.update_count
        run_steps(session, 30)
        np.testing.assert_array_equal(session.tiny.parameter_vector(), frozen_at)
        self.assertEqual(session.tiny.update_count, updates)

    def test_frames_report_which_arms_updated(self) -> None:
        session = PlaygroundSession(seed=1, online=False)
        run_steps(session, 10)
        frame = session.last_frame
        assert frame is not None
        self.assertFalse(frame.updated["tiny_nn"])
        self.assertFalse(frame.updated["persistence"])
        self.assertTrue(frame.updated["adaptive_rls"])


class ProtocolMarkingTests(unittest.TestCase):
    def test_changing_the_learning_rate_mid_run_marks_the_session(self) -> None:
        session = PlaygroundSession(seed=1)
        run_steps(session, 10)
        self.assertFalse(session.modified_during_run)
        session.set_learning_rate(0.05)
        self.assertTrue(session.modified_during_run)
        self.assertEqual(session.tiny.learning_rate, 0.05)

    def test_setting_the_learning_rate_before_any_step_is_not_a_modification(self) -> None:
        session = PlaygroundSession(seed=1)
        session.set_learning_rate(0.05)
        self.assertFalse(session.modified_during_run)

    def test_reset_clears_the_modified_marking(self) -> None:
        session = PlaygroundSession(seed=1)
        run_steps(session, 5)
        session.set_learning_rate(0.05)
        session.reset()
        self.assertFalse(session.modified_during_run)

    def test_a_non_positive_learning_rate_is_rejected(self) -> None:
        session = PlaygroundSession(seed=1)
        with self.assertRaises(ValueError):
            session.set_learning_rate(0.0)


class KeyboardTests(unittest.TestCase):
    def test_keyboard_contract(self) -> None:
        session = PlaygroundSession(seed=1)
        self.assertEqual(session.handle_key(None), "ignored")
        self.assertEqual(session.handle_key("q"), "ignored")
        self.assertEqual(session.handle_key(" "), "paused")
        self.assertEqual(session.handle_key(" "), "resumed")
        self.assertEqual(session.handle_key("right"), "stepped")
        self.assertEqual(session.step_index, 1)
        self.assertEqual(session.handle_key("f"), "frozen")
        self.assertEqual(session.handle_key("f"), "online")
        self.assertEqual(session.handle_key("R"), "reset")
        self.assertEqual(session.step_index, 0)


class SnapshotTests(unittest.TestCase):
    def test_snapshot_is_read_only(self) -> None:
        session = PlaygroundSession(scenario="bouncing", seed=31)
        run_steps(session, 20)
        before = {
            "step": session.step_index,
            "parameters": session.tiny.parameter_vector().copy(),
            "history": list(session.history),
        }
        for _ in range(5):
            session.snapshot()
        self.assertEqual(session.step_index, before["step"])
        np.testing.assert_array_equal(session.tiny.parameter_vector(), before["parameters"])
        self.assertEqual(session.history, before["history"])

    def test_snapshot_reports_the_real_model_state(self) -> None:
        session = PlaygroundSession(seed=711)
        run_steps(session, 20)
        snapshot = session.snapshot()
        np.testing.assert_array_equal(np.asarray(snapshot.tiny_parameters["W1"]), session.tiny.W1)
        np.testing.assert_array_equal(np.asarray(snapshot.tiny_parameters["W2"]), session.tiny.W2)
        self.assertEqual(snapshot.tiny_update_count, session.tiny.update_count)
        assert snapshot.tiny_hidden is not None
        self.assertEqual(len(snapshot.tiny_hidden), 4)
        assert snapshot.tiny_input is not None
        self.assertEqual(len(snapshot.tiny_input), 2)

    def test_event_steps_are_recorded_only_after_the_reveal(self) -> None:
        session = PlaygroundSession(scenario="dynamics_change", seed=711)
        run_steps(session, session.world.change_step + 5)
        events = dict(session.snapshot().event_steps)
        # The change transition is the one at index change_step, revealed as
        # step change_step + 1.
        self.assertEqual(events.get(session.world.change_step + 1), "change")
        for step in events:
            self.assertLessEqual(step, session.step_index)


class MetricsTests(unittest.TestCase):
    def test_mean_absolute_error_of_nothing_is_none(self) -> None:
        self.assertIsNone(mean_absolute_error([]))

    def test_mean_absolute_error_uses_magnitudes(self) -> None:
        self.assertAlmostEqual(mean_absolute_error([1.0, -3.0]), 2.0)

    def test_rolling_window_only_covers_the_trailing_window(self) -> None:
        track = ErrorTrack("x", window=3)
        for value in (1.0, 1.0, 1.0, 4.0):
            track.add(value)
        self.assertAlmostEqual(track.rolling_mae or 0.0, 2.0)
        self.assertAlmostEqual(track.cumulative_mae or 0.0, 1.75)

    def test_non_finite_error_is_rejected(self) -> None:
        track = ErrorTrack("x")
        with self.assertRaises(ValueError):
            track.add(float("nan"))


class RenderingSmokeTests(unittest.TestCase):
    """Headless figure construction and refresh. No display is required."""

    def setUp(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)

    def test_dashboard_builds_refreshes_and_closes(self) -> None:
        import matplotlib.pyplot as plt

        from playground.rendering import PlaygroundRenderer

        session = PlaygroundSession(scenario="dynamics_change", seed=711)
        renderer = PlaygroundRenderer(session)
        try:
            for _ in range(20):
                session.step()
            renderer.refresh()
            self.assertEqual(len(renderer.network.hidden_nodes), 4)
            self.assertEqual(len(renderer.network.input_nodes), 2)
            self.assertEqual(len(renderer.network.output_nodes), 1)
            self.assertEqual(sum(len(row) for row in renderer.network.w1_lines), 8)
            self.assertEqual(sum(len(row) for row in renderer.network.w2_lines), 4)
            # The error panel actually carries data.
            series = renderer.error_artists["tiny_nn"].get_ydata()
            self.assertEqual(len(series), len(session.error_series("tiny_nn")))
            # The state panel names the real numbers.
            text = renderer.state_text.get_text()
            self.assertIn("711", text)
            self.assertIn(str(session.model_seed), text)
            renderer.figure.canvas.draw()
        finally:
            plt.close(renderer.figure)

    def test_the_network_graph_tracks_real_weight_changes(self) -> None:
        import matplotlib.pyplot as plt

        from playground.rendering import PlaygroundRenderer

        session = PlaygroundSession(scenario="dynamics_change", seed=712)
        renderer = PlaygroundRenderer(session)
        try:
            for _ in range(10):
                session.step()
            renderer.refresh()
            before = [line.get_linewidth() for row in renderer.network.w1_lines for line in row]
            for _ in range(80):
                session.step()
            renderer.refresh()
            after = [line.get_linewidth() for row in renderer.network.w1_lines for line in row]
            self.assertNotEqual(before, after)
        finally:
            plt.close(renderer.figure)

    def test_a_frozen_session_is_visibly_marked(self) -> None:
        import matplotlib.pyplot as plt

        from playground.rendering import PlaygroundRenderer

        session = PlaygroundSession(seed=711, online=False)
        renderer = PlaygroundRenderer(session)
        try:
            for _ in range(10):
                session.step()
            renderer.refresh()
            self.assertIn("FROZEN", renderer.warning_text.get_text())
            session.set_online(True)
            session.set_learning_rate(0.05)
            renderer.refresh()
            self.assertIn("MODIFIED DURING RUN", renderer.warning_text.get_text())
        finally:
            plt.close(renderer.figure)

    def test_the_renderer_does_not_cause_learning(self) -> None:
        import matplotlib.pyplot as plt

        from playground.rendering import PlaygroundRenderer

        session = PlaygroundSession(seed=711)
        renderer = PlaygroundRenderer(session)
        try:
            for _ in range(15):
                session.step()
            before = session.tiny.parameter_vector().copy()
            updates = session.tiny.update_count
            step = session.step_index
            for _ in range(10):
                renderer.refresh()
            np.testing.assert_array_equal(session.tiny.parameter_vector(), before)
            self.assertEqual(session.tiny.update_count, updates)
            self.assertEqual(session.step_index, step)
        finally:
            plt.close(renderer.figure)

    def test_the_change_marker_lands_on_the_revealed_step_not_beside_it(self) -> None:
        import matplotlib.pyplot as plt

        from playground.rendering import PlaygroundRenderer

        session = PlaygroundSession(scenario="dynamics_change", seed=711)
        renderer = PlaygroundRenderer(session)
        try:
            change_step = session.world.change_step
            assert change_step is not None
            run_steps(session, change_step + 10)
            renderer.refresh()
            self.assertEqual(len(renderer.error_events), 1)
            marker = renderer.error_events[0].get_xdata()[0]
            # The change transition is revealed as session step change_step + 1,
            # which is scored step change_step + 1 - (history_length - 1).
            offset = session.world.history_length - 1
            self.assertEqual(marker, change_step + 1 - offset)
            # And that scored step is genuinely the one the session labelled.
            self.assertEqual(dict(session.snapshot().event_steps)[change_step + 1], "change")
        finally:
            plt.close(renderer.figure)

    def test_matplotlib_default_key_bindings_are_released(self) -> None:
        """The documented shortcuts must be the only ones bound.

        Matplotlib binds f/r/right to fullscreen/home/forward by default, which
        are exactly this Playground's freeze/reset/single-step keys. In a real
        window both handlers fire, so pressing f froze the model and threw the
        window fullscreen at the same time.
        """

        import matplotlib.pyplot as plt

        from playground.rendering import PlaygroundRenderer, release_default_keymap

        colliding = set()
        for setting in ("keymap.fullscreen", "keymap.home", "keymap.forward"):
            colliding.update(str(key).lower() for key in plt.rcParams[setting])
        # If matplotlib ever stops binding these, this test should be revisited
        # rather than silently passing for the wrong reason.
        self.assertTrue({"f", "r", "right"} <= colliding)

        session = PlaygroundSession(seed=1)
        renderer = PlaygroundRenderer(session)
        try:
            manager = renderer.figure.canvas.manager
            self.assertIsNotNone(manager)
            handler_id = manager.key_press_handler_id
            self.assertNotIn(
                handler_id, renderer.figure.canvas.callbacks.callbacks.get("key_press_event", {})
            )
            # Releasing an already-released figure is a no-op, not an error.
            release_default_keymap(renderer.figure)
        finally:
            plt.close(renderer.figure)

    def test_the_documented_shortcuts_still_reach_the_session(self) -> None:
        import matplotlib.pyplot as plt

        from playground.rendering import PlaygroundRenderer

        session = PlaygroundSession(seed=1)
        renderer = PlaygroundRenderer(session)
        try:
            renderer.figure.canvas.mpl_connect("key_press_event", renderer.on_key)
            run_steps(session, 5)
            self.assertTrue(session.tiny_online)
            renderer.on_key(SimpleNamespace(key="f"))
            self.assertFalse(session.tiny_online)
            renderer.on_key(SimpleNamespace(key=" "))
            self.assertTrue(session.paused)
            before = session.step_index
            renderer.on_key(SimpleNamespace(key="right"))
            self.assertEqual(session.step_index, before + 1)
            renderer.on_key(SimpleNamespace(key="r"))
            self.assertEqual(session.step_index, 0)
        finally:
            plt.close(renderer.figure)

    def test_rolling_series_matches_a_direct_computation(self) -> None:
        from playground.rendering import rolling_series

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(rolling_series(values, 2), [1.0, 1.5, 2.5, 3.5, 4.5])
        self.assertEqual(rolling_series([], 3), [])

    def test_entry_point_parses_arguments_without_opening_a_window(self) -> None:
        from playground.__main__ import build_parser

        arguments = build_parser().parse_args(["--scenario", "bouncing", "--seed", "5", "--frozen"])
        self.assertEqual(arguments.scenario, "bouncing")
        self.assertEqual(arguments.seed, 5)
        self.assertTrue(arguments.frozen)
        self.assertIn("playground.__main__", sys.modules)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
