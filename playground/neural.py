"""A deliberately tiny online neural learner for the AAA causal loop.

:class:`TinyMLPPredictor` is an **experiment**, not a proposal. It is not a
replacement for :class:`aaa.predictors.OnlineRLSPredictor`, it is not "the new
AAA architecture", and nothing it produces is evidence about AAA's benchmark
claims. The single question it exists to probe is narrow:

    can a small nonlinear online learner participate in exactly the same
    observe / predict / reveal / score / update contract the AAA runner
    enforces, without ever seeing evaluator metadata?

Architecture
------------

::

    2 inputs  ->  4 tanh hidden units  ->  1 linear output

    W1: 4 x 2 = 8      b1: 4
    W2: 1 x 4 = 4      b2: 1
    ----------------------------
    total             = 17 trainable parameters

Seventeen parameters is the point, not a limitation to be apologized for. The
world has one dot on a line; a learner small enough to print in full is a
learner whose every weight, activation and gradient a reviewer can check by
hand. Backpropagation is written out explicitly in NumPy for the same reason,
and is verified against finite differences in the test suite.

Causal contract
---------------

The only inputs are the two most recent observed positions and public
constants (the interval bounds and the declared displacement scale). No
velocity, scenario name, bounce flag, change flag, change schedule, oscillator
coefficient, future observation or evaluator label enters this module —
exactly as :mod:`aaa.predictors` states for the scientific learners.

Public boundary reflection
--------------------------

When ``reflect`` is enabled the raw prediction is folded back into the public
bounds with :func:`aaa.predictors.reflect_prediction`, the *same* utility every
other predictor uses, and the update target is unfolded with
:func:`aaa.predictors.unfold_observation`. Both are **programmed public
knowledge of the observation format, not learned intelligence**. That is why
:class:`aaa.predictors.ReflectedConstantMotionPredictor` is the like-for-like
baseline: without it a boundary transform is trivially mistaken for learned
bounce anticipation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from aaa.predictors import Predictor, reflect_prediction, unfold_observation

#: Architecture constants. These are fixed by design, not configurable: the
#: experiment is specifically about a 17-parameter learner.
INPUT_SIZE = 2
HIDDEN_SIZE = 4
OUTPUT_SIZE = 1
PARAMETER_COUNT = (
    HIDDEN_SIZE * INPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE * HIDDEN_SIZE + OUTPUT_SIZE
)  # 8 + 4 + 4 + 1 = 17

#: Frozen default, chosen on **development seeds only** by
#: :func:`playground.experiment.select_learning_rate`. Implementation started at
#: a conservative 0.01; on development evidence the criterion selected 0.3 from
#: the declared grid, with 3.0 diverging loudly and 1.0 excluded by the declared
#: one-step stability margin. Every attempted rate is recorded in
#: ``playground/evidence/learning_rate_selection.json``. Evaluation seeds were
#: not consulted at any point in that choice.
DEFAULT_LEARNING_RATE = 0.3

#: Initialization scales. Small, documented, deliberately boring: the hidden
#: layer gets enough spread to break symmetry, the output layer starts near
#: zero so the untrained model predicts close to persistence rather than
#: flinging the dot across the interval on step one.
INPUT_WEIGHT_SCALE = 0.5
OUTPUT_WEIGHT_SCALE = 0.1


class TinyMLPPredictor(Predictor):
    """17-parameter online MLP over causally available position history.

    The model predicts the *next displacement normalized by the interval
    width* and adds it to ``x[t]``, which is the same target representation the
    AAA RLS candidate uses. Loss is ``0.5 * (predicted - target)^2`` on that
    normalized displacement, and the update is plain stochastic gradient
    descent on a single revealed example.
    """

    format_version = "aaa.playground.tiny_mlp.v1"

    def __init__(
        self,
        *,
        model_seed: int,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        displacement_scale: float = 0.01,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        reflect: bool = True,
        unfold_target: bool = True,
        name: str = "tiny_nn",
        update_enabled: bool = True,
        parameters: dict[str, Any] | None = None,
        update_count: int = 0,
        cumulative_gradient_norm: float = 0.0,
        last_loss: float | None = None,
    ) -> None:
        if isinstance(model_seed, bool) or not isinstance(model_seed, (int, np.integer)):
            raise ValueError("model_seed must be an integer")
        for label, value in (
            ("lower_bound", lower_bound),
            ("upper_bound", upper_bound),
            ("displacement_scale", displacement_scale),
            ("learning_rate", learning_rate),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        if upper_bound <= lower_bound:
            raise ValueError("upper_bound must exceed lower_bound")
        if displacement_scale <= 0:
            raise ValueError("displacement_scale must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if int(update_count) < 0:
            raise ValueError("update_count must be non-negative")
        if not math.isfinite(float(cumulative_gradient_norm)) or cumulative_gradient_norm < 0:
            raise ValueError("cumulative_gradient_norm must be finite and non-negative")

        self.model_seed = int(model_seed)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.displacement_scale = float(displacement_scale)
        self.learning_rate = float(learning_rate)
        self.reflect = bool(reflect)
        self.unfold_target = bool(unfold_target)
        self.name = name
        self.update_enabled = bool(update_enabled)

        self.update_count = int(update_count)
        self.cumulative_gradient_norm = float(cumulative_gradient_norm)
        self.last_loss: float | None = None if last_loss is None else float(last_loss)

        # Latest forward-pass inspectables, for the visualizer. They are
        # presentation state only; nothing reads them back into learning.
        self.last_input: np.ndarray | None = None
        self.last_hidden: np.ndarray | None = None
        self.last_normalized_output: float | None = None
        self.last_raw_prediction: float | None = None
        self.last_reflected_prediction: float | None = None

        if parameters is None:
            self.W1, self.b1, self.W2, self.b2 = self._initial_parameters(self.model_seed)
        else:
            self.W1, self.b1, self.W2, self.b2 = self._parameters_from(parameters)
        self._check_parameters()

    # ------------------------------------------------------------------
    # construction and state
    # ------------------------------------------------------------------
    @staticmethod
    def _initial_parameters(model_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Deterministic initialization from a dedicated generator.

        The environment RNG is never touched: a model seed and an environment
        seed that shared a stream would couple learner initialization to the
        trajectory being learned.
        """

        rng = np.random.default_rng(int(model_seed))
        weights_1 = rng.normal(0.0, INPUT_WEIGHT_SCALE, size=(HIDDEN_SIZE, INPUT_SIZE))
        bias_1 = np.zeros(HIDDEN_SIZE, dtype=float)
        weights_2 = rng.normal(0.0, OUTPUT_WEIGHT_SCALE, size=(OUTPUT_SIZE, HIDDEN_SIZE))
        bias_2 = np.zeros(OUTPUT_SIZE, dtype=float)
        return weights_1, bias_1, weights_2, bias_2

    @staticmethod
    def _parameters_from(
        parameters: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        missing = [key for key in ("W1", "b1", "W2", "b2") if key not in parameters]
        if missing:
            raise ValueError(f"tiny MLP parameters are missing required keys: {sorted(missing)}")
        weights_1 = np.asarray(parameters["W1"], dtype=float).copy()
        bias_1 = np.asarray(parameters["b1"], dtype=float).copy()
        weights_2 = np.asarray(parameters["W2"], dtype=float).copy()
        bias_2 = np.asarray(parameters["b2"], dtype=float).copy()
        return weights_1, bias_1, weights_2, bias_2

    def _check_parameters(self) -> None:
        shapes = {
            "W1": (self.W1, (HIDDEN_SIZE, INPUT_SIZE)),
            "b1": (self.b1, (HIDDEN_SIZE,)),
            "W2": (self.W2, (OUTPUT_SIZE, HIDDEN_SIZE)),
            "b2": (self.b2, (OUTPUT_SIZE,)),
        }
        for label, (array, expected) in shapes.items():
            if array.shape != expected:
                raise ValueError(f"tiny MLP parameter {label} must have shape {expected}, got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"tiny MLP parameter {label} contains a non-finite value")

    @property
    def parameter_count(self) -> int:
        """Number of trainable scalars. Fixed at 17 by the architecture."""

        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def parameter_vector(self) -> np.ndarray:
        """Flat copy of every trainable parameter, in ``W1 b1 W2 b2`` order."""

        return np.concatenate(
            [self.W1.reshape(-1), self.b1.reshape(-1), self.W2.reshape(-1), self.b2.reshape(-1)]
        )

    def set_parameter_vector(self, values: Sequence[float]) -> None:
        """Overwrite every trainable parameter from a flat vector.

        Used by the finite-difference gradient check. It is deliberately not
        part of the learning path.
        """

        flat = np.asarray(values, dtype=float)
        if flat.shape != (PARAMETER_COUNT,):
            raise ValueError(f"parameter vector must contain exactly {PARAMETER_COUNT} values")
        offset = 0
        for label, array in (("W1", self.W1), ("b1", self.b1), ("W2", self.W2), ("b2", self.b2)):
            size = array.size
            array[...] = flat[offset : offset + size].reshape(array.shape)
            offset += size
            del label
        self._check_parameters()

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------
    def features(self, history: Sequence[float]) -> np.ndarray:
        """Two causally available, publicly normalized inputs.

        ``recent_displacement`` uses the declared public displacement scale
        ``dt * speed_max``; ``centered_position`` uses the public interval.
        Neither uses a statistic computed from the complete episode, and
        neither carries an event indicator.
        """

        if len(history) < 2:
            raise ValueError("tiny MLP requires at least two observations")
        recent = np.asarray(history[-2:], dtype=float)
        if not np.all(np.isfinite(recent)):
            raise ValueError("history contains a non-finite observation")
        width = self.upper_bound - self.lower_bound
        midpoint = (self.lower_bound + self.upper_bound) / 2.0
        recent_displacement = (recent[-1] - recent[-2]) / self.displacement_scale
        centered_position = (recent[-1] - midpoint) / width
        return np.asarray([recent_displacement, centered_position], dtype=float)

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------
    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, float]:
        """Return ``(hidden activations, normalized displacement)``.

        Pure: it records nothing and mutates nothing. The inspectable copies
        the visualizer reads are stored by :meth:`predict`.
        """

        pre_activation = self.W1 @ inputs + self.b1
        hidden = np.tanh(pre_activation)
        output = float((self.W2 @ hidden + self.b2)[0])
        if not np.all(np.isfinite(hidden)) or not math.isfinite(output):
            raise FloatingPointError("tiny MLP forward pass produced a non-finite value")
        return hidden, output

    def raw_predict(self, history: Sequence[float]) -> float:
        """Prediction before any public boundary policy is applied."""

        inputs = self.features(history)
        _, normalized = self.forward(inputs)
        width = self.upper_bound - self.lower_bound
        return float(history[-1] + width * normalized)

    def predict(self, history: Sequence[float]) -> float:
        """Predict ``x[t+1]``, recording inspectable forward-pass state.

        Calling this never changes a trainable parameter, the update count or
        the loss. The only fields it touches are the presentation-only
        ``last_*`` inspectables the renderer reads.
        """

        inputs = self.features(history)
        hidden, normalized = self.forward(inputs)
        width = self.upper_bound - self.lower_bound
        raw = float(history[-1] + width * normalized)
        if not math.isfinite(raw):
            raise FloatingPointError("tiny MLP produced a non-finite prediction")
        reflected = reflect_prediction(raw, self.lower_bound, self.upper_bound) if self.reflect else raw

        self.last_input = inputs
        self.last_hidden = hidden
        self.last_normalized_output = normalized
        self.last_raw_prediction = raw
        self.last_reflected_prediction = reflected
        return reflected

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def normalized_target(self, history: Sequence[float], target_position: float) -> float:
        """Target displacement, normalized by the interval width.

        With ``unfold_target`` enabled the revealed observation is first
        unfolded through the *same* public reflection map, using the model's
        own raw prediction as the branch reference, so the regression target
        lives in the coordinate the raw prediction lives in. No evaluator
        bounce label is involved.
        """

        if not math.isfinite(float(target_position)):
            raise ValueError("target_position must be finite")
        effective = float(target_position)
        if self.unfold_target and self.reflect:
            effective = unfold_observation(
                effective, self.raw_predict(history), self.lower_bound, self.upper_bound
            )
        width = self.upper_bound - self.lower_bound
        return float((effective - float(history[-1])) / width)

    def gradients(
        self, history: Sequence[float], target_position: float
    ) -> tuple[dict[str, np.ndarray], float]:
        """Analytical gradients of ``0.5 * (output - target)^2``, written out.

        Verified against central finite differences in
        ``tests/test_playground_neural.py``.
        """

        inputs = self.features(history)
        hidden, output = self.forward(inputs)
        target = self.normalized_target(history, target_position)
        loss = 0.5 * (output - target) ** 2

        # ``target`` is treated as a constant with respect to the parameters.
        # That is exact, not an approximation: unfolding selects a branch of a
        # piecewise map by proximity, so it is locally constant in the
        # parameters everywhere except on the measure-zero set where the branch
        # switches. This is the same treatment the AAA RLS candidate applies to
        # the same public map.
        d_output = output - target
        grad_w2 = d_output * hidden.reshape(1, HIDDEN_SIZE)
        grad_b2 = np.asarray([d_output], dtype=float)
        d_hidden = d_output * self.W2.reshape(HIDDEN_SIZE)
        d_pre_activation = d_hidden * (1.0 - hidden**2)
        grad_w1 = np.outer(d_pre_activation, inputs)
        grad_b1 = d_pre_activation

        grads = {"W1": grad_w1, "b1": grad_b1, "W2": grad_w2, "b2": grad_b2}
        for label, array in grads.items():
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(f"tiny MLP gradient {label} is non-finite")
        if not math.isfinite(loss):
            raise FloatingPointError("tiny MLP loss is non-finite")
        return grads, float(loss)

    def loss_for(self, history: Sequence[float], target_position: float) -> float:
        """Loss at the current parameters, without touching learner state."""

        inputs = self.features(history)
        _, output = self.forward(inputs)
        target = self.normalized_target(history, target_position)
        return float(0.5 * (output - target) ** 2)

    def update(self, history: Sequence[float], target_position: float) -> None:
        """One plain SGD step on a single revealed example.

        The caller is responsible for the temporal contract: this must be
        invoked only *after* the prediction for this transition was recorded
        and the target was revealed.
        """

        if not self.update_enabled:
            return
        grads, loss = self.gradients(history, target_position)
        self.W1 -= self.learning_rate * grads["W1"]
        self.b1 -= self.learning_rate * grads["b1"]
        self.W2 -= self.learning_rate * grads["W2"]
        self.b2 -= self.learning_rate * grads["b2"]
        # Fail loudly. A broken model is evidence, not something to silently
        # reset back to a state that never produced the failure.
        for label, array in (("W1", self.W1), ("b1", self.b1), ("W2", self.W2), ("b2", self.b2)):
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(f"tiny MLP update produced non-finite {label}")
        norm = float(math.sqrt(sum(float(np.sum(np.square(array))) for array in grads.values())))
        self.cumulative_gradient_norm += norm
        self.last_loss = loss
        self.update_count += 1

    # ------------------------------------------------------------------
    # serialization and copies
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "name": self.name,
            "architecture": [INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE],
            "parameter_count": self.parameter_count,
            "activation": "tanh",
            "target": "next_displacement_normalized_by_interval_width",
            "inputs": ["recent_displacement_over_displacement_scale", "centered_position_over_width"],
            "model_seed": self.model_seed,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "displacement_scale": self.displacement_scale,
            "learning_rate": self.learning_rate,
            "reflect": self.reflect,
            "unfold_target": self.unfold_target,
            "parameters": {
                "W1": [[float(value) for value in row] for row in self.W1],
                "b1": [float(value) for value in self.b1],
                "W2": [[float(value) for value in row] for row in self.W2],
                "b2": [float(value) for value in self.b2],
            },
            "update_count": self.update_count,
            "cumulative_gradient_norm": self.cumulative_gradient_norm,
            "last_loss": self.last_loss,
        }

    REQUIRED_STATE_FIELDS = (
        "model_seed",
        "lower_bound",
        "upper_bound",
        "displacement_scale",
        "learning_rate",
        "parameters",
    )

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        name: str | None = None,
        update_enabled: bool = False,
    ) -> TinyMLPPredictor:
        if state.get("format_version") != cls.format_version:
            raise ValueError(
                f"unsupported tiny MLP checkpoint format {state.get('format_version')!r}; "
                f"expected {cls.format_version!r}"
            )
        missing = [field for field in cls.REQUIRED_STATE_FIELDS if field not in state]
        if missing:
            raise ValueError(f"tiny MLP checkpoint is missing required fields: {sorted(missing)}")
        architecture = state.get("architecture")
        if architecture is not None and list(architecture) != [INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE]:
            raise ValueError("tiny MLP checkpoint declares a different architecture")
        model = cls(
            model_seed=int(state["model_seed"]),
            lower_bound=float(state["lower_bound"]),
            upper_bound=float(state["upper_bound"]),
            displacement_scale=float(state["displacement_scale"]),
            learning_rate=float(state["learning_rate"]),
            reflect=bool(state.get("reflect", True)),
            unfold_target=bool(state.get("unfold_target", True)),
            name=name or str(state.get("name", "tiny_nn")),
            update_enabled=update_enabled,
            parameters=dict(state["parameters"]),
            update_count=int(state.get("update_count", 0)),
            cumulative_gradient_norm=float(state.get("cumulative_gradient_norm", 0.0)),
            last_loss=(None if state.get("last_loss") is None else float(state["last_loss"])),
        )
        if model.parameter_count != PARAMETER_COUNT:
            raise ValueError("tiny MLP checkpoint does not describe a 17-parameter model")
        return model

    def clone(self, *, name: str, update_enabled: bool) -> TinyMLPPredictor:
        """An independent copy that starts numerically identical to this one."""

        return self.from_state_dict(self.state_dict(), name=name, update_enabled=update_enabled)

    def reset(self) -> None:
        """Restore the exact initial parameters implied by ``model_seed``."""

        self.W1, self.b1, self.W2, self.b2 = self._initial_parameters(self.model_seed)
        self.update_count = 0
        self.cumulative_gradient_norm = 0.0
        self.last_loss = None
        self.last_input = None
        self.last_hidden = None
        self.last_normalized_output = None
        self.last_raw_prediction = None
        self.last_reflected_prediction = None
        self._check_parameters()
