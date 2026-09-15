"""The tiny neural AAA experiment: a headless, deterministic, exploratory probe.

What this is
------------
A **tiny neural AAA experiment**. It asks one narrow, mechanical question on a
fixed toy protocol:

    starting from one identical learner state at an unannounced change in the
    world's law, does a 17-parameter neural learner that keeps updating end up
    with lower next-position error than a bitwise-identical copy of itself that
    is frozen at that instant?

What this is not
----------------
It is not benchmark v2.2. It is not a confirmation run. It is not validation of
AAA, evidence of understanding, or evidence of general adaptation. It produces
no acceptance evidence for anything, and its output says so in the artifact
itself.

Protocol
--------
For each declared evaluation seed:

1. a fresh :class:`~playground.neural.TinyMLPPredictor` is built from a
   declared model seed;
2. it observes and learns causally through a common pre-change prefix under the
   oscillator's pre-change law;
3. immediately before the changed-law period its complete state is cloned into
   ``tiny_nn_frozen`` (updates disabled) and ``tiny_nn_online`` (updates
   enabled); the clone is hashed so a reviewer can confirm the two branches
   started identical without trusting the runtime;
4. both branches consume the *same* subsequent trajectory, from the same
   continued world state;
5. neither branch is ever told that a change occurred. The evaluator knows the
   change point solely in order to branch and to report.

The AAA RLS learner is carried through exactly the same construction so the
neural arm has a semantically matched comparison, and the reflected
constant-motion baseline is carried because the tiny network is allowed to use
the same public reflection map.

Seed discipline
---------------
Development seeds are for debugging and for choosing a learning rate.
Evaluation seeds are run once, after the configuration is frozen, and the
result is retained whatever it says.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from aaa.config import WorldConfig
from aaa.environment import DampedOscillatorEnvironment
from aaa.experiment import StepRecord, TrialIdentity, continue_episode, run_episode, write_step_records
from aaa.predictors import (
    ConstantMotionPredictor,
    OnlineRLSPredictor,
    PersistencePredictor,
    Predictor,
    ReflectedConstantMotionPredictor,
)

from . import PLAYGROUND_VERSION
from .metrics import mean_absolute_error
from .neural import DEFAULT_LEARNING_RATE, PARAMETER_COUNT, TinyMLPPredictor
from .session import MODEL_SEED_OFFSET, build_rls

EXPERIMENT_ID = "aaa.playground.tiny_neural_experiment.v1"

#: Development seeds. Debugging, numerical-failure hunting and the learning
#: rate choice happen here and only here.
DEVELOPMENT_SEEDS: tuple[int, ...] = (701, 702, 703)

#: Evaluation seeds. Run once after the configuration is frozen.
EVALUATION_SEEDS: tuple[int, ...] = (711, 712, 713, 714, 715)

#: Learning rates considered during development. Recorded so the choice is
#: auditable rather than asserted. The grid was extended upward twice during
#: development because the criterion kept landing on the boundary; it now
#: brackets the divergence point, which is the honest place for a grid to stop.
LEARNING_RATE_GRID: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)

#: Pre-change law, matching the v2.1 changed-law family's declared prefix.
PRE_OMEGA = 1.5
PRE_DAMPING = 0.10

#: Post-change coefficient ranges, matching the v2.1 declared ranges. The
#: sampling stream is derived from the environment seed alone, exactly as the
#: benchmark family does it, so a seed fully determines the episode.
POST_OMEGA_RANGE = (5.0, 11.0)
POST_DAMPING_RANGE = (0.05, 0.3)
COEFFICIENT_SEED_MASK = 0x5EED

#: The four arms whose branch behaviour is the point of the experiment.
BRANCH_ARMS: tuple[str, ...] = ("tiny_nn_frozen", "tiny_nn_online", "rls_frozen", "rls_online")
BASELINE_ARMS: tuple[str, ...] = ("persistence", "constant_motion", "constant_motion_reflected")


@dataclass(frozen=True)
class TinyExperimentConfig:
    """Everything that determines a run. Serialized verbatim into the output."""

    prefix_steps: int = 300
    branch_steps: int = 100
    first_post_change_window: int = 50
    learning_rate: float = DEFAULT_LEARNING_RATE
    reflect: bool = True
    unfold_target: bool = True
    model_seed_offset: int = MODEL_SEED_OFFSET
    speed_min: float = 0.08
    speed_max: float = 0.20

    def quick(self) -> TinyExperimentConfig:
        """A smaller, still-deterministic smoke form of the same protocol."""

        return replace(self, prefix_steps=80, branch_steps=40, first_post_change_window=20)

    def worlds(self) -> tuple[WorldConfig, WorldConfig]:
        prefix = WorldConfig(
            steps_per_episode=self.prefix_steps,
            change_step=None,
            speed_min=self.speed_min,
            speed_max=self.speed_max,
        )
        branch = replace(prefix, steps_per_episode=self.branch_steps, change_step=0)
        return prefix, branch

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_seed_for(config: TinyExperimentConfig, environment_seed: int) -> int:
    """Model seeds are offset far from environment seeds so they cannot collide."""

    return int(config.model_seed_offset) + int(environment_seed)


def _baselines(world: WorldConfig) -> list[Predictor]:
    return [
        PersistencePredictor(),
        ConstantMotionPredictor(),
        ReflectedConstantMotionPredictor(lower_bound=world.lower_bound, upper_bound=world.upper_bound),
    ]


def _tiny(config: TinyExperimentConfig, world: WorldConfig, seed: int, *, name: str) -> TinyMLPPredictor:
    return TinyMLPPredictor(
        model_seed=model_seed_for(config, seed),
        lower_bound=world.lower_bound,
        upper_bound=world.upper_bound,
        displacement_scale=world.dt * world.speed_max,
        learning_rate=config.learning_rate,
        reflect=config.reflect,
        unfold_target=config.unfold_target,
        name=name,
        update_enabled=True,
    )


def _state_hash(state: dict[str, Any]) -> str:
    """Hash of learner state with the arm name removed.

    The name is the only field that legitimately differs between a frozen and
    an online clone, so excluding it lets the hash prove the two branches began
    from the same numbers.
    """

    payload = {key: value for key, value in state.items() if key != "name"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def normalized_errors(records: Sequence[StepRecord], name: str) -> list[float]:
    """Per-step normalized absolute error for one arm, from the retained records."""

    return [float(record.predictions[name]["normalized_absolute_error"]) for record in records]


def run_seed(
    config: TinyExperimentConfig, environment_seed: int, *, role: str
) -> tuple[dict[str, Any], list[StepRecord], list[StepRecord]]:
    """Run the full prefix/branch protocol for one environment seed."""

    prefix_world, branch_world = config.worlds()
    coefficient_rng = np.random.default_rng(environment_seed ^ COEFFICIENT_SEED_MASK)
    post_omega = float(coefficient_rng.uniform(*POST_OMEGA_RANGE))
    post_damping = float(coefficient_rng.uniform(*POST_DAMPING_RANGE))

    prefix_environment = DampedOscillatorEnvironment(
        environment_seed,
        prefix_world,
        omega=PRE_OMEGA,
        damping=PRE_DAMPING,
        changed_omega=PRE_OMEGA,
        changed_damping=PRE_DAMPING,
        change_step=None,
    )
    tiny_prefix = _tiny(config, prefix_world, environment_seed, name="tiny_nn_prefix")
    rls_prefix = build_rls(prefix_world, name="rls_prefix", online=True)
    prefix_identity = TrialIdentity(
        trial_id=f"playground:tiny_nn:prefix:s{environment_seed}",
        role=role,
        family="playground_tiny_neural",
        scenario="dynamics_change",
        environment_seed=environment_seed,
        replica_id=0,
        episode=0,
        branch="prefix",
        update_mode="mixed",
    )
    prefix_records = run_episode(
        prefix_environment,
        [*_baselines(prefix_world), tiny_prefix, rls_prefix],
        prefix_identity,
        learn=True,
    )
    if not prefix_records:
        raise RuntimeError("tiny neural experiment prefix produced no scored records")

    # The branch continues the *realized* world from the prefix's final state
    # and the prefix's own observation history, so both branches see exactly the
    # trajectory the prefix was heading into.
    history = [*prefix_records[-1].history[1:], prefix_records[-1].actual_next_position]
    start_position = prefix_environment.position
    start_velocity = prefix_environment.velocity

    tiny_state = tiny_prefix.state_dict()
    rls_state = rls_prefix.state_dict()
    branch_environment = DampedOscillatorEnvironment(
        environment_seed,
        branch_world,
        omega=post_omega,
        damping=post_damping,
        changed_omega=post_omega,
        changed_damping=post_damping,
        change_step=0,
        initial_position=start_position,
        initial_velocity=start_velocity,
    )
    tiny_frozen = TinyMLPPredictor.from_state_dict(tiny_state, name="tiny_nn_frozen", update_enabled=False)
    tiny_online = TinyMLPPredictor.from_state_dict(tiny_state, name="tiny_nn_online", update_enabled=True)
    rls_frozen = OnlineRLSPredictor.from_state_dict(rls_state, name="rls_frozen", update_enabled=False)
    rls_online = OnlineRLSPredictor.from_state_dict(rls_state, name="rls_online", update_enabled=True)
    frozen_before = tiny_frozen.state_dict()

    branch_identity = TrialIdentity(
        trial_id=f"playground:tiny_nn:changed-law:s{environment_seed}",
        role=role,
        family="playground_tiny_neural",
        scenario="dynamics_change",
        environment_seed=environment_seed,
        replica_id=0,
        episode=0,
        branch="changed-law",
        update_mode="mixed",
    )
    branch_records = continue_episode(
        branch_environment,
        [*_baselines(branch_world), tiny_frozen, tiny_online, rls_frozen, rls_online],
        history,
        branch_identity,
        learn=True,
        step_offset=config.prefix_steps,
    )
    if not branch_records:
        raise RuntimeError("tiny neural experiment branch produced no scored records")

    window = config.first_post_change_window
    prefix_errors = {
        "tiny_nn": normalized_errors(prefix_records, "tiny_nn_prefix"),
        "rls": normalized_errors(prefix_records, "rls_prefix"),
        **{name: normalized_errors(prefix_records, name) for name in BASELINE_ARMS},
    }
    branch_errors = {
        **{name: normalized_errors(branch_records, name) for name in BRANCH_ARMS},
        **{name: normalized_errors(branch_records, name) for name in BASELINE_ARMS},
    }

    failures: list[str] = []
    for label, series in list(prefix_errors.items()) + list(branch_errors.items()):
        if any(not np.isfinite(value) for value in series):
            failures.append(f"non-finite error in {label}")
    frozen_after = tiny_frozen.state_dict()
    if _state_hash(frozen_after) != _state_hash(frozen_before):
        failures.append("frozen tiny network changed state during the branch")
    if tiny_frozen.update_count != tiny_online.update_count - len(branch_records):
        failures.append("online/frozen update counts are inconsistent with the branch length")

    summary: dict[str, Any] = {
        "environment_seed": environment_seed,
        "model_seed": model_seed_for(config, environment_seed),
        "post_omega": post_omega,
        "post_damping": post_damping,
        "prefix_scored_steps": len(prefix_records),
        "branch_scored_steps": len(branch_records),
        "branch_state_hash": {
            "tiny_nn": _state_hash(tiny_state),
            "rls": _state_hash(rls_state),
        },
        "prefix_updates": {
            "tiny_nn": int(tiny_state["update_count"]),
            "rls": int(rls_state["update_count"]),
        },
        "branch_updates": {
            "tiny_nn_frozen": tiny_frozen.update_count - int(tiny_state["update_count"]),
            "tiny_nn_online": tiny_online.update_count - int(tiny_state["update_count"]),
            "rls_frozen": rls_frozen.update_count - int(rls_state["update_count"]),
            "rls_online": rls_online.update_count - int(rls_state["update_count"]),
        },
        "tiny_cumulative_gradient_norm": {
            "prefix": tiny_prefix.cumulative_gradient_norm,
            "branch_online": tiny_online.cumulative_gradient_norm,
            "branch_frozen": tiny_frozen.cumulative_gradient_norm,
        },
        # Pre-change: one shared learner per model family, plus the baselines.
        "pre_change_mae": {name: mean_absolute_error(series) for name, series in prefix_errors.items()},
        # Post-change: the whole changed-law branch.
        "post_change_mae": {name: mean_absolute_error(series) for name, series in branch_errors.items()},
        # The first window of the changed-law branch, where adaptation either
        # shows up or does not.
        f"first_{window}_post_change_mae": {
            name: mean_absolute_error(series[:window]) for name, series in branch_errors.items()
        },
        # Overall: the prefix the arm actually lived through, concatenated with
        # its own branch. The frozen and online arms share the prefix by
        # construction, which is exactly what makes them comparable.
        "overall_mae": {
            "tiny_nn_frozen": mean_absolute_error(prefix_errors["tiny_nn"] + branch_errors["tiny_nn_frozen"]),
            "tiny_nn_online": mean_absolute_error(prefix_errors["tiny_nn"] + branch_errors["tiny_nn_online"]),
            "rls_frozen": mean_absolute_error(prefix_errors["rls"] + branch_errors["rls_frozen"]),
            "rls_online": mean_absolute_error(prefix_errors["rls"] + branch_errors["rls_online"]),
            **{
                name: mean_absolute_error(prefix_errors[name] + branch_errors[name]) for name in BASELINE_ARMS
            },
        },
        "failures": failures,
    }
    return summary, prefix_records, branch_records


def _comparison(summary: dict[str, Any], key: str, left: str, right: str) -> dict[str, Any]:
    """One explicit head-to-head, reported as a difference and a ratio."""

    a = summary[key].get(left)
    b = summary[key].get(right)
    if a is None or b is None:
        return {"left": left, "right": right, "metric": key, "available": False}
    return {
        "left": left,
        "right": right,
        "metric": key,
        "available": True,
        "left_mae": a,
        "right_mae": b,
        "difference": a - b,
        "ratio": (a / b) if b > 0 else None,
        "left_lower": a < b,
    }


def _aggregate(per_seed: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, float | None]]:
    """Mean/min/max across seeds. Variability is reported, never collapsed away."""

    names = sorted({name for seed in per_seed for name in seed[key]})
    output: dict[str, dict[str, float | None]] = {}
    for name in names:
        values = [seed[key][name] for seed in per_seed if seed[key].get(name) is not None]
        output[name] = {
            "mean": float(np.mean(values)) if values else None,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
            "seeds": len(values),
        }
    return output


def git_provenance(root: Path) -> dict[str, Any]:
    """Source commit and dirty-tree status, or an explicit unavailable marker."""

    def _run(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode(
                "utf-8"
            )
        except (subprocess.CalledProcessError, OSError):
            return None

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {
        "commit": None if commit is None else commit.strip(),
        "dirty": None if status is None else bool(status.strip()),
        "available": commit is not None and status is not None,
    }


CLAIM_BOUNDARIES = {
    "implementation_works": (
        "Supported when the run completes with no numerical failures. It says the loop, the "
        "gradients and the serialization behave; it says nothing about learning."
    ),
    "model_learns_on_some_stream": (
        "Supported only by the pre-change MAE of the updating arm relative to its own "
        "untrained starting point. It is a statement about this stream, not about streams."
    ),
    "model_improves_over_its_frozen_copy": (
        "This is the one claim the protocol is actually built to test, and only on this fixed "
        "toy protocol, at these seeds, on this changed-law world."
    ),
    "model_generalizes": (
        "NOT addressed. No held-out family, no unfamiliar law, no distribution shift beyond the "
        "single declared coefficient change."
    ),
    "model_beats_a_baseline": (
        "Addressed descriptively against persistence, raw constant motion, reflected constant "
        "motion and the AAA RLS learner. Reflected constant motion is the like-for-like "
        "comparison because the tiny network is allowed the same public reflection map. No "
        "hypothesis test is performed and none should be inferred."
    ),
}

INTERPRETATION = (
    "Under this fixed toy protocol, at the declared seeds, these numbers describe next-position "
    "error for a 17-parameter neural learner that kept updating versus a bitwise-identical copy "
    "frozen at the same instant. Nothing here is evidence that AAA understands motion, learned "
    "physics, is intelligent, or generalizes."
)


def run_experiment(
    config: TinyExperimentConfig,
    seeds: Sequence[int],
    *,
    role: str = "playground_evaluation",
    output_root: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Run the protocol over ``seeds`` and return the complete result document."""

    if not seeds:
        raise ValueError("at least one seed is required")
    root = project_root or Path(__file__).resolve().parent.parent
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        summary, prefix_records, branch_records = run_seed(config, int(seed), role=role)
        per_seed.append(summary)
        if output_root is not None:
            steps = Path(output_root) / "steps"
            write_step_records(prefix_records, steps / f"seed-{seed}-prefix.jsonl")
            write_step_records(branch_records, steps / f"seed-{seed}-changed-law.jsonl")

    window = config.first_post_change_window
    first_key = f"first_{window}_post_change_mae"
    comparisons = [_comparison(seed, first_key, "tiny_nn_online", "tiny_nn_frozen") for seed in per_seed]
    document: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "playground_version": PLAYGROUND_VERSION,
        "status": "exploratory",
        "not_a_benchmark": (
            "This is a tiny neural AAA experiment. It is not benchmark v2.2, not a confirmation "
            "run, not validation of AAA, and not evidence of understanding or general adaptation."
        ),
        "source": git_provenance(root),
        "role": role,
        "configuration": config.as_dict(),
        "environment_seeds": [int(seed) for seed in seeds],
        "model_seeds": [model_seed_for(config, int(seed)) for seed in seeds],
        "model": {
            "architecture": "2 inputs -> 4 tanh hidden -> 1 linear output",
            "parameter_count": PARAMETER_COUNT,
            "learning_rate": config.learning_rate,
            "optimizer": "plain SGD on 0.5 * (prediction - target)^2",
            "target": "next displacement normalized by interval width",
            "inputs": [
                "(x[t] - x[t-1]) / (dt * speed_max)",
                "(x[t] - midpoint) / interval_width",
            ],
            "public_boundary_policy": (
                "Predictions are reflected into the public bounds and update targets are unfolded "
                "through the inverse of the same public map. Both are programmed public knowledge "
                "of the observation format, not learned capability."
            ),
            "learning_rate_grid_considered": list(LEARNING_RATE_GRID),
            "learning_rate_selected_on": "development seeds only",
        },
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "per_seed": per_seed,
        "aggregate": {
            "pre_change_mae": _aggregate(per_seed, "pre_change_mae"),
            "post_change_mae": _aggregate(per_seed, "post_change_mae"),
            first_key: _aggregate(per_seed, first_key),
            "overall_mae": _aggregate(per_seed, "overall_mae"),
        },
        "head_to_head_online_vs_frozen": comparisons,
        "seeds_where_online_beat_its_frozen_clone": sum(
            1 for item in comparisons if item.get("left_lower") is True
        ),
        "failures": [failure for seed in per_seed for failure in seed["failures"]],
        "claim_boundaries": CLAIM_BOUNDARIES,
        "interpretation": INTERPRETATION,
    }
    if output_root is not None:
        destination = Path(output_root) / "summary.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def select_learning_rate(
    config: TinyExperimentConfig,
    *,
    rates: Sequence[float] = LEARNING_RATE_GRID,
    seeds: Sequence[int] = DEVELOPMENT_SEEDS,
) -> dict[str, Any]:
    """Development-only learning-rate selection.

    Every attempted rate is recorded, including the ones that lost and the ones
    that blew up. Evaluation seeds are never consulted.

    The criterion has two parts, both fixed before any evaluation seed was run:

    1. **stability margin.** A rate is admissible only if it is strictly below
       the smallest rate that produced a numerical failure on development
       seeds. Picking the best-scoring rate that happens to sit immediately
       below the divergence point would make the evaluation run a coin flip on
       whether an unseen seed diverges, and a hyperparameter chosen that way is
       not a choice, it is a gamble.
    2. **score.** Among admissible rates, the lowest mean first-window
       post-change MAE of ``tiny_nn_online`` across development seeds.
    """

    window = config.first_post_change_window
    first_key = f"first_{window}_post_change_mae"
    attempts: list[dict[str, Any]] = []
    for rate in sorted(float(value) for value in rates):
        trial = replace(config, learning_rate=rate)
        try:
            summaries = [run_seed(trial, int(seed), role="playground_development")[0] for seed in seeds]
        except (FloatingPointError, ValueError) as error:
            # A diverging rate is evidence, not a crash. Record it and carry on.
            attempts.append(
                {
                    "learning_rate": rate,
                    "development_seeds": [int(seed) for seed in seeds],
                    "per_seed_first_window_mae": None,
                    "mean_first_window_mae": None,
                    "failures": [f"{type(error).__name__}: {error}"],
                }
            )
            continue
        values = [summary[first_key]["tiny_nn_online"] for summary in summaries]
        attempts.append(
            {
                "learning_rate": rate,
                "development_seeds": [int(seed) for seed in seeds],
                "per_seed_first_window_mae": values,
                "mean_first_window_mae": float(np.mean(values)),
                "failures": [failure for summary in summaries for failure in summary["failures"]],
            }
        )

    failing = [attempt["learning_rate"] for attempt in attempts if attempt["failures"]]
    divergence_point = min(failing) if failing else None
    admissible = [
        attempt
        for attempt in attempts
        if not attempt["failures"]
        and attempt["mean_first_window_mae"] is not None
        and (divergence_point is None or attempt["learning_rate"] < divergence_point)
    ]
    # The stability margin: drop the largest admissible rate, so the selected
    # rate is at least one grid step away from the observed divergence point.
    if divergence_point is not None and len(admissible) > 1:
        largest = max(attempt["learning_rate"] for attempt in admissible)
        admissible = [attempt for attempt in admissible if attempt["learning_rate"] < largest]
    if not admissible:
        raise RuntimeError("no admissible learning rate: every attempted rate failed")
    chosen = min(admissible, key=lambda attempt: attempt["mean_first_window_mae"])
    return {
        "experiment": EXPERIMENT_ID,
        "stage": "development_learning_rate_selection",
        "criterion": (
            "among rates with no numerical failure and at least one grid step below the smallest "
            "diverging rate, the lowest mean first-window post-change MAE of tiny_nn_online "
            "across development seeds"
        ),
        "attempts": attempts,
        "divergence_point": divergence_point,
        "admissible_learning_rates": [attempt["learning_rate"] for attempt in admissible],
        "selected_learning_rate": chosen["learning_rate"],
        "note": "Evaluation seeds were not consulted at any point in this selection.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="playground.experiment",
        description="Tiny neural AAA experiment (exploratory; not a benchmark).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Small deterministic smoke form at a truncated horizon. It is a check that the "
            "protocol runs, not the evaluation result, whichever seed group it is given."
        ),
    )
    parser.add_argument(
        "--seeds",
        choices=("evaluation", "development"),
        default="evaluation",
        help="Which declared seed group to run.",
    )
    parser.add_argument(
        "--select-learning-rate",
        action="store_true",
        help="Development-only learning-rate sweep; never touches evaluation seeds.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to write generated evidence (default: runs/playground/<label>).",
    )
    parser.add_argument("--label", default=None, help="Run directory label.")
    parser.add_argument("--no-write", action="store_true", help="Print the summary without writing files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = TinyExperimentConfig()
    if arguments.quick:
        config = config.quick()

    if arguments.select_learning_rate:
        document = select_learning_rate(config)
        label = arguments.label or "learning-rate-selection"
    else:
        seeds = EVALUATION_SEEDS if arguments.seeds == "evaluation" else DEVELOPMENT_SEEDS
        role = "playground_evaluation" if arguments.seeds == "evaluation" else "playground_development"
        label = arguments.label or f"{arguments.seeds}{'-quick' if arguments.quick else ''}"
        output_root = None
        if not arguments.no_write:
            output_root = arguments.output_root or Path("runs") / "playground" / label
        document = run_experiment(config, seeds, role=role, output_root=output_root)

    if arguments.select_learning_rate and not arguments.no_write:
        destination = (arguments.output_root or Path("runs") / "playground" / label) / "summary.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(document, indent=2, sort_keys=True))
    failures = document.get("failures") or []
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
