"""Headless Playground state machine.

This module contains no Matplotlib import and no GUI concept. It follows the
same separation :class:`aaa.animation.AnimationSession` already established::

    headless state machine
            |
            v
    renderer consumes state

and the same temporal contract :mod:`aaa.experiment` owns::

    observe available history
    -> predict the next observation
    -> permanently record that prediction
    -> advance the environment
    -> reveal the next observation
    -> score the pre-reveal prediction
    -> update the online learner
    -> append the revealed observation

The renderer never causes learning: it calls :meth:`PlaygroundSession.step`
and reads :meth:`PlaygroundSession.snapshot`, and nothing else. Pausing freezes
the simulation completely — no environment transition, no prediction, no
update.

Event labels (``bounce``, ``change``) are **evaluator-side display metadata**.
They are attached to a frame only after the corresponding observation has been
revealed, and they are never routed into a predictor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aaa.config import WorldConfig
from aaa.environment import (
    DampedOscillatorEnvironment,
    Environment,
    MovingDotEnvironment,
    as_scenario,
)
from aaa.predictors import (
    ConstantMotionPredictor,
    OnlineRLSPredictor,
    PersistencePredictor,
    Predictor,
    ReflectedConstantMotionPredictor,
)

from .metrics import ErrorTrack
from .neural import DEFAULT_LEARNING_RATE, TinyMLPPredictor

#: Scenarios the Playground exposes. Every one of them is an existing public
#: AAA world; the Playground invents no dynamics of its own, precisely so that
#: the neural learner cannot be handed a world built to flatter it.
PLAYGROUND_SCENARIOS: tuple[str, ...] = ("straight", "bouncing", "changed", "dynamics_change")

#: Display order, which is also the legend order in the renderer.
PREDICTOR_NAMES: tuple[str, ...] = (
    "tiny_nn",
    "adaptive_rls",
    "constant_motion_reflected",
    "constant_motion",
    "persistence",
)

#: Offset used to derive a model seed from an environment seed when the caller
#: does not supply one. Deliberately large and fixed, so a model seed can never
#: silently coincide with an environment seed.
MODEL_SEED_OFFSET = 900_000

#: Oscillator coefficients for the live ``dynamics_change`` world. These match
#: the pre-change law declared in the v2.1 specification; the post-change pair
#: is one fixed, documented choice inside the specification's declared ranges,
#: so that what a viewer watches is reproducible rather than resampled.
LIVE_PRE_OMEGA = 1.5
LIVE_PRE_DAMPING = 0.10
LIVE_POST_OMEGA = 8.0
LIVE_POST_DAMPING = 0.15


def playground_world(scenario: str) -> WorldConfig:
    """The public world configuration used for one live scenario."""

    name = as_scenario(scenario)
    if name == "dynamics_change":
        # A long enough prefix for the learners to identify the pre-change law,
        # then an unannounced coefficient change with room to watch recovery.
        return WorldConfig(steps_per_episode=420, change_step=300, speed_min=0.08, speed_max=0.20)
    if name == "changed":
        return WorldConfig(steps_per_episode=240, change_step=120)
    return WorldConfig(steps_per_episode=240, change_step=None)


def build_environment(scenario: str, seed: int, world: WorldConfig) -> Environment:
    """Construct the existing public AAA world for ``scenario``.

    The Playground never reimplements motion physics; it only selects which
    already-reviewed environment to instantiate.
    """

    name = as_scenario(scenario)
    if name == "dynamics_change":
        return DampedOscillatorEnvironment(
            seed,
            world,
            omega=LIVE_PRE_OMEGA,
            damping=LIVE_PRE_DAMPING,
            changed_omega=LIVE_POST_OMEGA,
            changed_damping=LIVE_POST_DAMPING,
        )
    return MovingDotEnvironment(scenario=name, seed=seed, config=world)


def build_rls(world: WorldConfig, *, name: str = "adaptive_rls", online: bool = True) -> OnlineRLSPredictor:
    """The current AAA RLS learner, configured for this world.

    The hyperparameters are the ones the v2.1 candidate declares. The
    displacement scale is taken from *this* world so that the RLS and the tiny
    network normalize their displacement input identically; comparing two
    learners on differently scaled inputs would not be a fair comparison.
    """

    return OnlineRLSPredictor(
        lower_bound=world.lower_bound,
        upper_bound=world.upper_bound,
        displacement_scale=world.dt * world.speed_max,
        forgetting=0.3,
        forgetting_mode="exponential",
        ridge=1e-4,
        feature_set="displacement_position",
        reflect=True,
        unfold_target=True,
        skip_after_reflected_prediction=True,
        trace_bound=1e5,
        dead_zone=0.0,
        detector_multiplier=8.0,
        detector_floor=1e-6,
        detector_decay=0.05,
        name=name,
        update_enabled=online,
    )


@dataclass(frozen=True)
class PlaygroundFrame:
    """Everything one completed transition produced.

    ``predictions`` were all issued *before* ``actual`` was revealed. The
    renderer displays them as issued; nothing is recomputed after the reveal.
    """

    step_index: int
    target_step: int
    actual: float
    scored: bool
    event: str
    finished: bool
    predictions: dict[str, float] = field(default_factory=dict)
    normalized_errors: dict[str, float] = field(default_factory=dict)
    updated: dict[str, bool] = field(default_factory=dict)
    tiny_raw_prediction: float | None = None
    tiny_reflected_prediction: float | None = None
    tiny_loss: float | None = None


@dataclass(frozen=True)
class PlaygroundSnapshot:
    """Read-only view of session state for a renderer or a test."""

    scenario: str
    environment_seed: int
    model_seed: int
    step_index: int
    total_steps: int
    paused: bool
    tiny_online: bool
    modified_during_run: bool
    learning_rate: float
    history: tuple[float, ...]
    lower_bound: float
    upper_bound: float
    last_frame: PlaygroundFrame | None
    cumulative_mae: dict[str, float | None]
    rolling_mae: dict[str, float | None]
    tiny_update_count: int
    tiny_last_loss: float | None
    tiny_cumulative_gradient_norm: float
    tiny_parameters: dict[str, Any]
    tiny_hidden: tuple[float, ...] | None
    tiny_input: tuple[float, ...] | None
    event_steps: tuple[tuple[int, str], ...]


class PlaygroundSession:
    """One live, deterministic, fully inspectable AAA prediction loop.

    Exploratory tooling. It is not a benchmark, it produces no acceptance
    evidence, and a run whose hyperparameters were changed mid-flight is
    marked :attr:`modified_during_run` for exactly that reason.
    """

    def __init__(
        self,
        *,
        scenario: str = "dynamics_change",
        seed: int = 711,
        model_seed: int | None = None,
        online: bool = True,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        rolling_window: int = 25,
    ) -> None:
        self.scenario = as_scenario(scenario)
        self.environment_seed = int(seed)
        self.model_seed = int(model_seed) if model_seed is not None else MODEL_SEED_OFFSET + int(seed)
        self.tiny_online = bool(online)
        self.learning_rate = float(learning_rate)
        self.rolling_window = int(rolling_window)
        self.modified_during_run = False
        self.paused = False
        self.reset()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.world = playground_world(self.scenario)
        self.environment = build_environment(self.scenario, self.environment_seed, self.world)
        self.tiny = TinyMLPPredictor(
            model_seed=self.model_seed,
            lower_bound=self.world.lower_bound,
            upper_bound=self.world.upper_bound,
            displacement_scale=self.world.dt * self.world.speed_max,
            learning_rate=self.learning_rate,
            name="tiny_nn",
            update_enabled=self.tiny_online,
        )
        self.rls = build_rls(self.world)
        self.predictors: dict[str, Predictor] = {
            "tiny_nn": self.tiny,
            "adaptive_rls": self.rls,
            "constant_motion_reflected": ReflectedConstantMotionPredictor(
                lower_bound=self.world.lower_bound, upper_bound=self.world.upper_bound
            ),
            "constant_motion": ConstantMotionPredictor(),
            "persistence": PersistencePredictor(),
        }
        if tuple(self.predictors) != PREDICTOR_NAMES:
            raise RuntimeError("predictor display order and construction order disagree")

    # ------------------------------------------------------------------
    # controls
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Deterministically restore environment, models, history and counters."""

        self._build()
        self.history: list[float] = [self.environment.reset()]
        self.step_index = 0
        self.last_frame: PlaygroundFrame | None = None
        self.tracks: dict[str, ErrorTrack] = {
            name: ErrorTrack(name, window=self.rolling_window) for name in PREDICTOR_NAMES
        }
        self.event_steps: list[tuple[int, str]] = []
        self.modified_during_run = False
        self.paused = False

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def set_scenario(self, scenario: str) -> None:
        self.scenario = as_scenario(scenario)
        self.reset()

    def set_seed(self, seed: int, *, model_seed: int | None = None) -> None:
        self.environment_seed = int(seed)
        self.model_seed = int(model_seed) if model_seed is not None else MODEL_SEED_OFFSET + int(seed)
        self.reset()

    def set_online(self, online: bool) -> None:
        """Freeze or unfreeze the tiny network without disturbing anything else.

        The AAA RLS learner is left updating: it is shown as the project's
        current online learner, not as a second frozen arm.
        """

        self.tiny_online = bool(online)
        self.tiny.update_enabled = self.tiny_online

    def set_learning_rate(self, learning_rate: float) -> None:
        """Change the tiny network's learning rate mid-run.

        This invalidates the run as fixed-protocol evidence, which is why it
        latches :attr:`modified_during_run`. The flag is never cleared except
        by :meth:`reset`.
        """

        value = float(learning_rate)
        if value <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = value
        self.tiny.learning_rate = value
        if self.step_index > 0:
            self.modified_during_run = True

    def handle_key(self, key: str | None) -> str:
        """Keyboard contract shared by the GUI and the tests."""

        if key is None:
            return "ignored"
        lowered = key.lower()
        if lowered == " ":
            return "paused" if self.toggle_pause() else "resumed"
        if lowered == "right":
            self.single_step()
            return "stepped"
        if lowered == "r":
            self.reset()
            return "reset"
        if lowered == "f":
            self.set_online(not self.tiny_online)
            return "online" if self.tiny_online else "frozen"
        return "ignored"

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    @property
    def finished(self) -> bool:
        return self.step_index >= self.world.steps_per_episode

    def single_step(self) -> PlaygroundFrame | None:
        """Advance exactly one transition even while paused."""

        was_paused = self.paused
        self.paused = False
        try:
            return self.step()
        finally:
            self.paused = was_paused

    def step(self) -> PlaygroundFrame | None:
        """Advance exactly one environment transition, or do nothing.

        Returns ``None`` while paused. While paused no environment transition,
        no prediction and no update happens: the simulation is frozen, not
        merely undrawn.
        """

        if self.paused:
            return None
        if self.finished:
            # End of horizon: a distinct, unambiguous terminal frame rather
            # than a silent ``None`` (which means "paused") or a replayed copy
            # of the last real frame (which a renderer would draw twice).
            return PlaygroundFrame(
                step_index=self.step_index,
                target_step=self.step_index,
                actual=self.history[-1],
                scored=False,
                event="end",
                finished=True,
            )

        length = self.world.history_length
        if len(self.history) < length:
            # Warm-up: not scored, so every predictor starts on exactly the
            # same observation window. Still exactly one transition.
            transition = self.environment.advance()
            self.step_index += 1
            self.history.append(transition.position)
            frame = PlaygroundFrame(
                step_index=self.step_index,
                target_step=self.step_index,
                actual=transition.position,
                scored=False,
                event=_event_label(transition.changed, transition.bounced),
                finished=self.finished,
            )
            self.last_frame = frame
            return frame

        window = tuple(self.history[-length:])

        # 1-3. Predict from causally available history and record it. Nothing
        # has advanced yet; no predictor can see the target.
        predictions = {name: float(model.predict(window)) for name, model in self.predictors.items()}
        tiny_raw = self.tiny.last_raw_prediction
        tiny_reflected = self.tiny.last_reflected_prediction

        # 4-5. Advance and reveal.
        transition = self.environment.advance()
        self.step_index += 1
        target = float(transition.position)

        # 6. Score the pre-reveal predictions.
        width = self.world.width
        normalized_errors = {name: abs(value - target) / width for name, value in predictions.items()}
        for name, error in normalized_errors.items():
            self.tracks[name].add(error)

        # The tiny network's a-priori loss on the just-revealed target. This is
        # a pure read of already-revealed data at the *pre-update* parameters,
        # so it is defined identically whether the model is online or frozen.
        tiny_loss = self.tiny.loss_for(window, target)

        # 7. Only now may an enabled learner update.
        updated: dict[str, bool] = {}
        for name, model in self.predictors.items():
            enabled = bool(model.update_enabled)
            if enabled:
                model.update(window, target)
            updated[name] = enabled

        # 8. Append the revealed observation.
        self.history.append(target)

        event = _event_label(transition.changed, transition.bounced)
        if event != "steady":
            self.event_steps.append((self.step_index, event))
        frame = PlaygroundFrame(
            step_index=self.step_index,
            target_step=self.step_index,
            actual=target,
            scored=True,
            event=event,
            finished=self.finished,
            predictions=predictions,
            normalized_errors=normalized_errors,
            updated=updated,
            tiny_raw_prediction=tiny_raw,
            tiny_reflected_prediction=tiny_reflected,
            tiny_loss=tiny_loss,
        )
        self.last_frame = frame
        return frame

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------
    def snapshot(self) -> PlaygroundSnapshot:
        """A read-only view. Calling it changes nothing."""

        return PlaygroundSnapshot(
            scenario=self.scenario,
            environment_seed=self.environment_seed,
            model_seed=self.model_seed,
            step_index=self.step_index,
            total_steps=self.world.steps_per_episode,
            paused=self.paused,
            tiny_online=self.tiny_online,
            modified_during_run=self.modified_during_run,
            learning_rate=self.tiny.learning_rate,
            history=tuple(self.history),
            lower_bound=self.world.lower_bound,
            upper_bound=self.world.upper_bound,
            last_frame=self.last_frame,
            cumulative_mae={name: track.cumulative_mae for name, track in self.tracks.items()},
            rolling_mae={name: track.rolling_mae for name, track in self.tracks.items()},
            tiny_update_count=self.tiny.update_count,
            tiny_last_loss=self.tiny.last_loss,
            tiny_cumulative_gradient_norm=self.tiny.cumulative_gradient_norm,
            tiny_parameters={
                "W1": self.tiny.W1.tolist(),
                "b1": self.tiny.b1.tolist(),
                "W2": self.tiny.W2.tolist(),
                "b2": self.tiny.b2.tolist(),
            },
            tiny_hidden=(None if self.tiny.last_hidden is None else tuple(self.tiny.last_hidden.tolist())),
            tiny_input=(None if self.tiny.last_input is None else tuple(self.tiny.last_input.tolist())),
            event_steps=tuple(self.event_steps),
        )

    def error_series(self, name: str) -> tuple[float, ...]:
        """Normalized absolute error per scored step, for the error panel."""

        return tuple(self.tracks[name].errors)


def _event_label(changed: bool, bounced: bool) -> str:
    """Evaluator-side display label for an already-revealed transition."""

    if changed:
        return "change"
    if bounced:
        return "bounce"
    return "steady"
