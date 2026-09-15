"""AAA Playground: an isolated, exploratory live visualizer and tiny neural experiment.

This package is **not** part of the AAA scientific benchmark. It is a
repository-local research utility that reuses the public, already-reviewed
components of :mod:`aaa` (environment dynamics, world configuration, the
public boundary reflection map, the existing baselines and the current RLS
learner) to let a human watch one prediction/learning loop unfold, and to run
a deliberately tiny neural experiment under the same causal contract.

Nothing here defines, redefines, weakens or supersedes benchmark v2.1, the
frozen confirmation evidence, or the separate observation-noise research
phase. Results produced by this package are exploratory and are labelled as
such wherever they are emitted.
"""

from __future__ import annotations

__all__ = [
    "PLAYGROUND_VERSION",
]

#: Identity of this exploratory tooling. Deliberately *not* a benchmark
#: version string: it must never be mistaken for ``aaa.benchmark.v2.x``.
PLAYGROUND_VERSION = "aaa.playground.exploratory.v0.1"
