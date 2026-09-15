"""Matplotlib dashboard for the AAA Playground.

The renderer is a *consumer*. It calls :meth:`PlaygroundSession.step` to
advance and :meth:`PlaygroundSession.snapshot` to read, and it never touches a
model, a parameter, an environment or a counter. Every number and every line
on screen comes from session state that was produced under the causal
contract; nothing is recomputed after the target was revealed.

Panels
------
``environment``   the bounded line world, the actual dot, and each predictor's
                  previously issued next-position prediction.
``error``         rolling normalized absolute error per predictor, plus the
                  current cumulative MAE in the legend.
``network``       the real 2 -> 4 -> 1 tiny network: its actual weights (sign
                  by colour, magnitude by thickness), its actual hidden
                  activations, and its actual biases.
``state``         step, scenario, seeds, online/frozen, loss, update count and
                  every MAE, as text.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.widgets import Button, RadioButtons, Slider

from .neural import HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE, PARAMETER_COUNT
from .session import PLAYGROUND_SCENARIOS, PREDICTOR_NAMES, PlaygroundSession

#: One colour per predictor, stable across every panel.
PREDICTOR_COLORS: dict[str, str] = {
    "tiny_nn": "#d1495b",
    "adaptive_rls": "#1b4965",
    "constant_motion_reflected": "#2a9d8f",
    "constant_motion": "#8d99ae",
    "persistence": "#c9ada7",
}

#: Human labels. The reflected baseline is spelled out because it is the
#: like-for-like comparison for a learner that is allowed to reflect.
PREDICTOR_LABELS: dict[str, str] = {
    "tiny_nn": "TinyMLP (17 params)",
    "adaptive_rls": "AAA RLS (3 params)",
    "constant_motion_reflected": "constant motion + reflection",
    "constant_motion": "constant motion (raw)",
    "persistence": "persistence",
}

POSITIVE_WEIGHT_COLOR = "#1d6fb8"
NEGATIVE_WEIGHT_COLOR = "#c1121f"


def rolling_series(values: Sequence[float], window: int) -> list[float]:
    """Rolling mean of ``values`` over a trailing ``window``."""

    if window < 1:
        raise ValueError("window must be at least one step")
    output: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += float(value)
        if index >= window:
            total -= float(values[index - window])
        span = min(index + 1, window)
        output.append(total / span)
    return output


def _format(value: float | None, digits: int = 5) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


class NetworkGraph:
    """Live drawing of the actual tiny network, from its actual parameters."""

    def __init__(self, axis: Any) -> None:
        self.axis = axis
        # Room on both flanks so the input and output labels stay inside the
        # panel when the window is narrow.
        axis.set_xlim(-0.85, 2.95)
        axis.set_ylim(-0.15, 1.15)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(
            f"TinyMLP {INPUT_SIZE}→{HIDDEN_SIZE} tanh→{OUTPUT_SIZE} ({PARAMETER_COUNT} trainable parameters)",
            fontsize=9,
        )

        self.input_positions = _layer_positions(INPUT_SIZE, 0.0)
        self.hidden_positions = _layer_positions(HIDDEN_SIZE, 1.2)
        self.output_positions = _layer_positions(OUTPUT_SIZE, 2.1)

        # Connections are drawn first so the nodes sit on top of them.
        self.w1_lines: list[list[Line2D]] = [
            [
                axis.plot(
                    [self.input_positions[i][0], self.hidden_positions[j][0]],
                    [self.input_positions[i][1], self.hidden_positions[j][1]],
                    color=POSITIVE_WEIGHT_COLOR,
                    linewidth=1.0,
                    zorder=1,
                )[0]
                for i in range(INPUT_SIZE)
            ]
            for j in range(HIDDEN_SIZE)
        ]
        self.w2_lines: list[list[Line2D]] = [
            [
                axis.plot(
                    [self.hidden_positions[j][0], self.output_positions[k][0]],
                    [self.hidden_positions[j][1], self.output_positions[k][1]],
                    color=POSITIVE_WEIGHT_COLOR,
                    linewidth=1.0,
                    zorder=1,
                )[0]
                for j in range(HIDDEN_SIZE)
            ]
            for k in range(OUTPUT_SIZE)
        ]

        self.input_nodes = [_add_node(axis, point) for point in self.input_positions]
        self.hidden_nodes = [_add_node(axis, point) for point in self.hidden_positions]
        self.output_nodes = [_add_node(axis, point) for point in self.output_positions]

        self.input_labels = [
            axis.text(point[0] - 0.08, point[1], text, ha="right", va="center", fontsize=7)
            for point, text in zip(
                self.input_positions, ("recent\ndisplacement", "centered\nposition"), strict=True
            )
        ]
        # One line, on a white plate: two stacked lines collided with the node
        # below at this spacing, which made the activations unreadable.
        self.hidden_texts = [
            axis.text(
                point[0],
                point[1] - 0.10,
                "",
                ha="center",
                va="top",
                fontsize=6,
                zorder=3,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.6},
            )
            for point in self.hidden_positions
        ]
        self.output_text = axis.text(
            self.output_positions[0][0] + 0.09,
            self.output_positions[0][1],
            "",
            ha="left",
            va="center",
            fontsize=7,
        )
        self.caption = axis.text(
            0.5,
            -0.11,
            "",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color="#444444",
        )

    def refresh(self, snapshot: Any) -> None:
        parameters = snapshot.tiny_parameters
        weights_1 = parameters["W1"]
        bias_1 = parameters["b1"]
        weights_2 = parameters["W2"]
        bias_2 = parameters["b2"]

        scale_1 = max((abs(value) for row in weights_1 for value in row), default=1.0) or 1.0
        for j in range(HIDDEN_SIZE):
            for i in range(INPUT_SIZE):
                _style_connection(self.w1_lines[j][i], weights_1[j][i], scale_1)
        scale_2 = max((abs(value) for row in weights_2 for value in row), default=1.0) or 1.0
        for k in range(OUTPUT_SIZE):
            for j in range(HIDDEN_SIZE):
                _style_connection(self.w2_lines[k][j], weights_2[k][j], scale_2)

        inputs = snapshot.tiny_input
        hidden = snapshot.tiny_hidden
        for index, node in enumerate(self.input_nodes):
            value = None if inputs is None else inputs[index]
            # Inputs are unbounded; squash only for the colour, never for the
            # value that is reported.
            node.set_facecolor(_activation_color(None if value is None else _squash(value)))
        for index, node in enumerate(self.hidden_nodes):
            value = None if hidden is None else hidden[index]
            node.set_facecolor(_activation_color(value))
            self.hidden_texts[index].set_text(
                f"h={value:+.2f}  b={bias_1[index]:+.2f}" if value is not None else ""
            )
        output_value = None
        frame = snapshot.last_frame
        if frame is not None and frame.tiny_raw_prediction is not None:
            output_value = frame.tiny_raw_prediction
        self.output_nodes[0].set_facecolor(_activation_color(None))
        self.output_text.set_text(f"raw x̂ {_format(output_value, 3)}\nb={bias_2[0]:+.2f}")
        self.caption.set_text(
            "colour = weight sign (blue pos, red neg)  ·  thickness = |weight|  ·  fill = activation"
        )


def _layer_positions(count: int, x: float) -> list[tuple[float, float]]:
    if count == 1:
        return [(x, 0.5)]
    span = 0.86
    top = 0.5 + span / 2
    gap = span / (count - 1)
    return [(x, top - index * gap) for index in range(count)]


def _add_node(axis: Any, point: tuple[float, float]) -> Circle:
    circle = Circle(point, 0.085, facecolor="#f2f2f2", edgecolor="#333333", linewidth=1.0, zorder=2)
    axis.add_patch(circle)
    return circle


def _style_connection(line: Line2D, weight: float, scale: float) -> None:
    line.set_color(POSITIVE_WEIGHT_COLOR if weight >= 0 else NEGATIVE_WEIGHT_COLOR)
    line.set_linewidth(0.4 + 3.6 * min(abs(weight) / scale, 1.0))


def _squash(value: float) -> float:
    return float(max(-1.0, min(1.0, value)))


def _activation_color(value: float | None) -> str | tuple[float, float, float]:
    """Diverging fill for an activation in ``[-1, 1]``; grey when unknown."""

    if value is None:
        return "#f2f2f2"
    bounded = _squash(value)
    if bounded >= 0:
        shade = 1.0 - 0.75 * bounded
        return (shade, shade, 1.0)
    shade = 1.0 + 0.75 * bounded
    return (1.0, shade, shade)


def release_default_keymap(figure: Figure) -> bool:
    """Drop Matplotlib's built-in figure key bindings. Returns whether it did.

    Matplotlib binds ``f`` to fullscreen, ``r`` to home and ``right`` to
    forward by default -- the same three keys this Playground documents for
    freeze, reset and single step. In a real window those fire *as well as* the
    Playground's handler: pressing ``f`` froze the tiny network and threw the
    window to fullscreen at the same time. Disconnecting the default handler
    leaves the documented controls as the only key bindings on this figure.

    Save, zoom, pan and the rest stay available on the toolbar, and the window
    manager still closes the window, so nothing a viewer needs is lost.
    """

    manager = getattr(figure.canvas, "manager", None)
    handler_id = getattr(manager, "key_press_handler_id", None)
    if handler_id is None:
        return False
    figure.canvas.mpl_disconnect(handler_id)
    return True


class PlaygroundRenderer:
    """Builds and refreshes the dashboard for one :class:`PlaygroundSession`.

    Construction and :meth:`refresh` both work under a headless backend, which
    is what the rendering smoke test exercises.
    """

    def __init__(self, session: PlaygroundSession, *, figure: Figure | None = None) -> None:
        self.session = session
        self.figure = figure if figure is not None else plt.figure(figsize=(14.0, 8.0))
        release_default_keymap(self.figure)
        self.figure.suptitle(
            "AAA Playground — exploratory visualizer. Not benchmark v2.1 evidence.",
            fontsize=11,
        )
        grid = GridSpec(
            2,
            2,
            figure=self.figure,
            left=0.075,
            right=0.975,
            top=0.875,
            bottom=0.27,
            hspace=0.50,
            wspace=0.18,
        )
        self.world_axis = self.figure.add_subplot(grid[0, 0])
        self.network_axis = self.figure.add_subplot(grid[0, 1])
        self.error_axis = self.figure.add_subplot(grid[1, 0])
        self.state_axis = self.figure.add_subplot(grid[1, 1])

        self._build_world_panel()
        self._build_error_panel()
        self.network = NetworkGraph(self.network_axis)
        self._build_state_panel()
        self._build_controls()
        self.refresh()

    # ------------------------------------------------------------------
    # panels
    # ------------------------------------------------------------------
    def _build_world_panel(self) -> None:
        axis = self.world_axis
        snapshot = self.session.snapshot()
        axis.set_xlim(snapshot.lower_bound, snapshot.upper_bound)
        axis.set_ylim(-0.45, 1.25)
        axis.set_yticks([])
        axis.set_xlabel("position — predictions were issued before this position was revealed")
        axis.axhline(0.0, color="#dddddd", linewidth=1.0, zorder=0)
        (self.actual_artist,) = axis.plot(
            [], [], "o", color="black", markersize=12, label="actual x[t+1]", zorder=5
        )
        # Each predictor gets its own row. Stacking every marker on one line
        # hid them all behind the actual dot, which is exactly when they agree
        # and exactly when the viewer needs to see that they do.
        self.prediction_rows: dict[str, float] = {
            name: 0.25 + 0.20 * index for index, name in enumerate(PREDICTOR_NAMES)
        }
        self.prediction_artists: dict[str, Line2D] = {}
        for name in PREDICTOR_NAMES:
            row = self.prediction_rows[name]
            axis.axhline(row, color="#f0f0f0", linewidth=0.8, zorder=0)
            (artist,) = axis.plot(
                [],
                [],
                "x",
                color=PREDICTOR_COLORS[name],
                markersize=9,
                mew=2,
                label=PREDICTOR_LABELS[name],
                zorder=4,
            )
            self.prediction_artists[name] = artist
        (self.tiny_raw_artist,) = axis.plot(
            [],
            [],
            "+",
            color=PREDICTOR_COLORS["tiny_nn"],
            markersize=10,
            mew=1.4,
            alpha=0.55,
            label="TinyMLP raw (before public reflection)",
            zorder=3,
        )
        self.event_artist = axis.text(0.01, 0.02, "", transform=axis.transAxes, fontsize=7, color="#444444")
        # Above the axes: predictions travel the whole interval, so any in-axes
        # legend eventually sits on top of the markers it is labelling.
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            fontsize=6.5,
            ncol=3,
            framealpha=0.9,
            borderaxespad=0.0,
        )

    def _build_error_panel(self) -> None:
        axis = self.error_axis
        axis.set_xlabel("scored step")
        axis.set_ylabel(f"rolling MAE / L (window {self.session.rolling_window})", fontsize=8)
        axis.set_title("prediction error — linear scale, common axis, no per-series rescaling", fontsize=9)
        axis.grid(alpha=0.25, linewidth=0.5)
        self.error_artists: dict[str, Line2D] = {}
        for name in PREDICTOR_NAMES:
            (artist,) = axis.plot(
                [], [], color=PREDICTOR_COLORS[name], linewidth=1.4, label=PREDICTOR_LABELS[name]
            )
            self.error_artists[name] = artist
        self.error_events: list[Line2D] = []
        axis.legend(loc="upper right", fontsize=6.5, ncol=2, framealpha=0.9)

    def _build_state_panel(self) -> None:
        axis = self.state_axis
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title("live state", fontsize=9)
        self.state_text = axis.text(
            0.01, 0.97, "", transform=axis.transAxes, va="top", ha="left", fontsize=8, family="monospace"
        )
        # Figure level, in the clear strip under the suptitle above the right
        # column. Inside the state panel it collided with the readout, and a
        # status banner a viewer has to read through other text is not a
        # status banner.
        self.warning_text = self.figure.text(
            0.76,
            0.955,
            "",
            ha="center",
            va="top",
            fontsize=9,
            color="#b00020",
            weight="bold",
        )

    def _build_controls(self) -> None:
        figure = self.figure
        self.play_button = Button(figure.add_axes((0.06, 0.13, 0.085, 0.05)), "Pause")
        self.step_button = Button(figure.add_axes((0.155, 0.13, 0.085, 0.05)), "Step")
        self.reset_button = Button(figure.add_axes((0.25, 0.13, 0.085, 0.05)), "Reset")
        self.freeze_button = Button(figure.add_axes((0.345, 0.13, 0.115, 0.05)), "Freeze TinyMLP")
        self.seed_button = Button(figure.add_axes((0.47, 0.13, 0.095, 0.05)), "Next seed")
        self.scenario_radio = RadioButtons(
            figure.add_axes((0.06, 0.02, 0.16, 0.095)),
            PLAYGROUND_SCENARIOS,
            active=PLAYGROUND_SCENARIOS.index(self.session.scenario),
        )
        for label in self.scenario_radio.labels:
            label.set_fontsize(7)
        self.speed_slider = Slider(
            figure.add_axes((0.47, 0.055, 0.30, 0.025)),
            "steps / frame",
            valmin=1,
            valmax=8,
            valinit=1,
            valstep=1,
        )
        self.speed_slider.label.set_fontsize(7)
        self.speed_slider.valtext.set_fontsize(7)

        self.play_button.on_clicked(lambda _event: self._on_play())
        self.step_button.on_clicked(lambda _event: self._on_step())
        self.reset_button.on_clicked(lambda _event: self._on_reset())
        self.freeze_button.on_clicked(lambda _event: self._on_freeze())
        self.seed_button.on_clicked(lambda _event: self._on_next_seed())
        self.scenario_radio.on_clicked(self._on_scenario)

    # ------------------------------------------------------------------
    # control callbacks — each one only drives the session, never a model
    # ------------------------------------------------------------------
    def _on_play(self) -> None:
        self.session.toggle_pause()
        self.refresh()

    def _on_step(self) -> None:
        self.session.single_step()
        self.refresh()

    def _on_reset(self) -> None:
        self.session.reset()
        self._reset_panels()
        self.refresh()

    def _on_freeze(self) -> None:
        self.session.set_online(not self.session.tiny_online)
        self.refresh()

    def _on_next_seed(self) -> None:
        self.session.set_seed(self.session.environment_seed + 1)
        self._reset_panels()
        self.refresh()

    def _on_scenario(self, label: str | None) -> None:
        if label is None:
            return
        self.session.set_scenario(label)
        self._reset_panels()
        self.refresh()

    def _reset_panels(self) -> None:
        for artist in self.error_artists.values():
            artist.set_data([], [])
        for marker in self.error_events:
            marker.remove()
        self.error_events = []
        snapshot = self.session.snapshot()
        self.world_axis.set_xlim(snapshot.lower_bound, snapshot.upper_bound)

    def on_key(self, event: Any) -> None:
        """Keyboard handling, delegated entirely to the session."""

        outcome = self.session.handle_key(getattr(event, "key", None))
        if outcome == "reset":
            self._reset_panels()
        if outcome != "ignored":
            self.refresh()

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def advance(self) -> None:
        """Advance the session by the configured number of steps and redraw."""

        steps = int(self.speed_slider.val)
        for _ in range(max(1, steps)):
            if self.session.paused or self.session.finished:
                break
            self.session.step()
        self.refresh()

    def refresh(self) -> None:
        """Redraw every panel from current session state. Changes nothing."""

        snapshot = self.session.snapshot()
        frame = snapshot.last_frame

        if frame is not None and frame.scored:
            self.actual_artist.set_data([frame.actual], [0.0])
            for name, artist in self.prediction_artists.items():
                value = frame.predictions.get(name)
                if value is None:
                    artist.set_data([], [])
                else:
                    artist.set_data([value], [self.prediction_rows[name]])
            raw = frame.tiny_raw_prediction
            reflected = frame.tiny_reflected_prediction
            # The raw marker is drawn only when public reflection actually
            # moved the prediction, so the two are never confused with one
            # another when they coincide.
            if raw is not None and reflected is not None and raw != reflected:
                self.tiny_raw_artist.set_data([raw], [self.prediction_rows["tiny_nn"]])
            else:
                self.tiny_raw_artist.set_data([], [])
            label = {"change": "law change revealed", "bounce": "wall contact revealed", "steady": ""}.get(
                frame.event, frame.event
            )
            self.event_artist.set_text(
                f"step {frame.step_index}: {label} (evaluator display metadata; never a predictor input)"
                if label
                else f"step {frame.step_index}"
            )
        elif frame is not None:
            self.actual_artist.set_data([frame.actual], [0.0])
            self.event_artist.set_text(f"step {frame.step_index}: warm-up (not scored)")

        window = self.session.rolling_window
        maximum = 0.0
        longest = 1
        for name, artist in self.error_artists.items():
            series = rolling_series(self.session.error_series(name), window)
            artist.set_data(range(1, len(series) + 1), series)
            if series:
                maximum = max(maximum, max(series))
                longest = max(longest, len(series))
        self.error_axis.set_xlim(0, max(longest, 10))
        self.error_axis.set_ylim(0.0, max(maximum * 1.15, 1e-4))
        self._mark_change_events(snapshot)

        self.network.refresh(snapshot)
        self.state_text.set_text(self._state_lines(snapshot))
        warnings = []
        if snapshot.modified_during_run:
            warnings.append("EXPLORATORY / MODIFIED DURING RUN")
        if not snapshot.tiny_online:
            warnings.append("TinyMLP FROZEN — no updates are being applied")
        self.warning_text.set_text("\n".join(warnings))
        self.play_button.label.set_text("Play" if snapshot.paused else "Pause")
        self.freeze_button.label.set_text("Freeze TinyMLP" if snapshot.tiny_online else "Unfreeze TinyMLP")

    def _mark_change_events(self, snapshot: Any) -> None:
        """Mark revealed law-change transitions on the error panel.

        A scored step ``n`` (1-based on this axis) is the session step
        ``n + history_length - 1``, because the first ``history_length - 1``
        transitions are unscored warm-up. Inverting that is what keeps the
        marker on the step the change was actually revealed on, rather than one
        to the side of it. Bounces are deliberately not marked: they are
        frequent enough to bury the curves they are meant to explain.
        """

        offset = self.session.world.history_length - 1
        positions = [step - offset for step, event in snapshot.event_steps if event == "change"]
        positions = [value for value in positions if value >= 1]
        if len(positions) == len(self.error_events):
            return
        for marker in self.error_events:
            marker.remove()
        self.error_events = [
            self.error_axis.axvline(
                value,
                color="#b00020",
                linewidth=1.0,
                linestyle="--",
                alpha=0.7,
                label="law change revealed" if index == 0 else None,
            )
            for index, value in enumerate(positions)
        ]

    def _state_lines(self, snapshot: Any) -> str:
        frame = snapshot.last_frame
        event = "—" if frame is None else frame.event
        lines = [
            f"step               {snapshot.step_index} / {snapshot.total_steps}",
            f"scenario           {snapshot.scenario}",
            f"environment seed   {snapshot.environment_seed}",
            f"model seed         {snapshot.model_seed}",
            f"TinyMLP            {'online' if snapshot.tiny_online else 'FROZEN'}"
            f"  (lr {snapshot.learning_rate:g})",
            f"TinyMLP updates    {snapshot.tiny_update_count}",
            f"TinyMLP loss       {_format(None if frame is None else frame.tiny_loss, 6)}",
            f"grad-norm total    {snapshot.tiny_cumulative_gradient_norm:.4f}",
            "",
            "cumulative MAE (normalized by interval width L)",
        ]
        for name in PREDICTOR_NAMES:
            lines.append(f"  {PREDICTOR_LABELS[name]:<32} {_format(snapshot.cumulative_mae[name])}")
        lines.extend(
            [
                "",
                f"current event      {event}",
                "                   (evaluator display metadata only)",
            ]
        )
        return "\n".join(lines)


def launch(
    session: PlaygroundSession, *, interval_ms: int = 60
) -> None:  # pragma: no cover - needs a display
    """Open the interactive Playground window.

    Space plays/pauses, Right single-steps, ``r`` resets and ``f`` toggles the
    tiny network between online and frozen.
    """

    from matplotlib.animation import FuncAnimation

    renderer = PlaygroundRenderer(session)

    def draw(_frame: int) -> None:
        renderer.advance()

    renderer.figure.canvas.mpl_connect("key_press_event", renderer.on_key)
    animation = FuncAnimation(
        renderer.figure, draw, interval=interval_ms, cache_frame_data=False, save_count=0
    )
    # Keep a reference so the animation is not garbage collected.
    renderer.figure._aaa_playground_animation = animation  # type: ignore[attr-defined]
    plt.show()
