"""Launch the interactive AAA Playground: ``python -m playground``.

Exploratory tooling, run from a repository checkout. It is deliberately not
wired into ``aaa.cli`` and deliberately not packaged: the scientific package
and its command surface are under independent review and must not acquire new
entry points to accommodate a visualizer.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .neural import DEFAULT_LEARNING_RATE
from .session import PLAYGROUND_SCENARIOS, PlaygroundSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="playground",
        description=(
            "AAA Playground — an exploratory live visualizer. Not benchmark v2.1, "
            "not evidence for any AAA claim."
        ),
    )
    parser.add_argument("--scenario", choices=PLAYGROUND_SCENARIOS, default="dynamics_change")
    parser.add_argument("--seed", type=int, default=711, help="Environment seed.")
    parser.add_argument(
        "--model-seed", type=int, default=None, help="Model seed (default: 900000 + environment seed)."
    )
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="Start with the tiny network frozen, so its learning contributes nothing.",
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--rolling-window", type=int, default=25)
    parser.add_argument(
        "--interval-ms", type=int, default=60, help="Wall-clock milliseconds between animation frames."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    session = PlaygroundSession(
        scenario=arguments.scenario,
        seed=arguments.seed,
        model_seed=arguments.model_seed,
        online=not arguments.frozen,
        learning_rate=arguments.learning_rate,
        rolling_window=arguments.rolling_window,
    )
    # Imported here so that the headless session and the tests never pull in a
    # GUI toolkit just to be constructed.
    from .rendering import launch

    print(
        "AAA Playground (exploratory). Space play/pause, Right single step, "
        "r reset, f freeze/unfreeze the tiny network.",
        file=sys.stderr,
    )
    launch(session, interval_ms=arguments.interval_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
