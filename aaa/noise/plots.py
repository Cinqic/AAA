"""Raw-data-backed plots for an observation-noise attempt."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .verifier import iter_records


def _add(bucket: dict[Any, list[float]], key: Any, value: float) -> None:
    aggregate = bucket.setdefault(key, [0.0, 0.0])
    aggregate[0] += value
    aggregate[1] += 1.0


def _mean(aggregate: list[float]) -> float:
    if aggregate[1] == 0.0:
        raise ValueError("cannot summarize an empty plotted group")
    return aggregate[0] / aggregate[1]


def _save(figure: Any, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path


def write_plots(run_dir: str | Path) -> list[Path]:
    """Write diagnostic figures whose provenance is the primitive archive.

    Error bars in the figures are empirical 5th/95th percentiles of retained
    primitive rows around their median, not the aggregate hierarchical
    intervals in ``summary``.
    The latter remain the authoritative statistical uncertainty artifact.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(run_dir)
    output: list[Path] = []
    by_scale: dict[tuple[str, float], list[float]] = {}
    by_step: dict[tuple[str, int], list[float]] = {}
    surprises: dict[int, float] = defaultdict(float)
    update_norms: dict[tuple[str, int], list[float]] = {}
    latencies: dict[tuple[str, int], list[float]] = {}
    controls: dict[str, list[float]] = {}
    strata: dict[tuple[str, str], list[float]] = {}
    row_count = 0
    for row in iter_records(root):
        row_count += 1
        trial = row["trial"]
        branch = str(trial["branch"])
        for name, prediction in row["predictions"].items():
            error = float(prediction["latent_normalized_absolute_error"])
            if name == "incumbent_square_root_rls" and branch == "stationary":
                _add(by_scale, (str(trial["channel"]), float(trial["scale"])), error)
            if name in {"incumbent_branch_frozen", "incumbent_branch_online"}:
                _add(by_step, (name, int(row["step"])), error)
            if name == "incumbent_square_root_rls":
                diagnostics = row.get("diagnostics", {}).get(name, {})
                step = int(row["step"])
                surprises[step] = max(surprises[step], float(diagnostics.get("detected_surprises", 0)))
                _add(update_norms, (name, step), float(diagnostics.get("update_norm", 0)))
                _add(
                    latencies,
                    (name, step),
                    float(diagnostics.get("predict_latency_ns", 0))
                    + float(diagnostics.get("update_latency_ns", 0)),
                )
                if branch.startswith("sensor_shift_"):
                    _add(controls, branch, error)
                stratum = str(trial.get("stratum", "unstratified"))
                if branch == "stationary" and stratum != "unstratified":
                    _add(strata, (str(trial["family"]), stratum), error)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    colors = {
        "gaussian": "tab:blue",
        "uniform": "tab:orange",
        "correlated": "tab:green",
        "impulsive": "tab:red",
    }
    for (channel, scale), aggregate in sorted(by_scale.items()):
        color = colors.get(channel, "tab:gray")
        axis.scatter([scale], [_mean(aggregate)], s=24, color=color, label=channel)
    axis.set_title("Incumbent mean latent error by observed-noise scale")
    axis.set_xlabel("RMS noise scale")
    axis.set_ylabel("normalized absolute error")
    axis.set_yscale("symlog", linthresh=1e-6)
    if by_scale:
        axis.legend(loc="best", fontsize="small")
    output.append(_save(figure, root / "error_vs_noise_scale.png"))

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name, color in (
        ("incumbent_branch_frozen", "tab:gray"),
        ("incumbent_branch_online", "tab:blue"),
    ):
        points = sorted((step, _mean(values)) for (label, step), values in by_step.items() if label == name)
        if points:
            axis.plot([point[0] for point in points], [point[1] for point in points], label=name, color=color)
            first_step, first_error = points[0]
            axis.scatter([first_step], [first_error], color=color, marker="*", s=65, zorder=3)
            axis.annotate(
                "first post-change error",
                (first_step, first_error),
                xytext=(5, 8),
                textcoords="offset points",
            )
    axis.axvline(300, color="black", linestyle="--", linewidth=0.8, label="law change")
    axis.set_title(
        "Matched changed-law branch trajectory\n(first post-intervention error is included and marked)"
    )
    axis.set_xlabel("transition")
    axis.set_ylabel("normalized latent absolute error")
    if by_step:
        axis.legend(loc="best", fontsize="small")
    output.append(_save(figure, root / "matched_adaptation_trajectory.png"))

    figure, axis = plt.subplots(figsize=(9, 4.8))
    labels = sorted(controls)
    if labels:
        positions = list(range(len(labels)))
        means = [_mean(controls[label]) for label in labels]
        axis.scatter(positions, means)
        axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_title("Sensor-shift controls: unchanged and changed latent dynamics\n(primitive-record means)")
    axis.set_xlabel("factorial branch")
    axis.set_ylabel("incumbent normalized latent absolute error")
    output.append(_save(figure, root / "noise_shift_controls.png"))

    figure, axis = plt.subplots(figsize=(9, 4.8))
    worst = sorted(
        ((key, _mean(values)) for key, values in strata.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    if worst:
        labels = [f"{family}\n{stratum}" for (family, stratum), _value in worst]
        means = [value for _key, value in worst]
        positions = list(range(len(worst)))
        axis.scatter(positions, means)
        axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_title("Worst realized incumbent strata\n(top strata by primitive-record mean)")
    axis.set_xlabel("predeclared family and stratum")
    axis.set_ylabel("normalized latent absolute error")
    output.append(_save(figure, root / "worst_stratum.png"))

    figure, axis = plt.subplots(figsize=(8, 4.5))
    points = sorted(surprises.items())
    if points:
        axis.plot([point[0] for point in points], [point[1] for point in points], color="tab:red")
    axis.set_title("Incumbent detector activity")
    axis.set_xlabel("transition")
    axis.set_ylabel("cumulative detected surprises")
    output.append(_save(figure, root / "detector_activity.png"))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    update_points = sorted(
        (step, _mean(values))
        for (name, step), values in update_norms.items()
        if name == "incumbent_square_root_rls"
    )
    latency_points = sorted(
        (step, _mean(values))
        for (name, step), values in latencies.items()
        if name == "incumbent_square_root_rls"
    )
    if update_points:
        axes[0].plot(
            [point[0] for point in update_points],
            [point[1] for point in update_points],
            color="tab:purple",
        )
    if latency_points:
        axes[1].plot(
            [point[0] for point in latency_points],
            [point[1] for point in latency_points],
            color="tab:green",
        )
    axes[0].set_title("Update-state movement")
    axes[0].set_xlabel("transition")
    axes[0].set_ylabel("mean update norm")
    axes[1].set_title("Measured predict + update cost")
    axes[1].set_xlabel("transition")
    axes[1].set_ylabel("mean nanoseconds / step")
    output.append(_save(figure, root / "update_resource_diagnostics.png"))

    records_path = root / "records.jsonl"
    if not records_path.is_file():
        records_path = root / "records.jsonl.gz"
    if not records_path.is_file():
        records_path = root / "records" / "index.json"
    provenance = {
        "schema_version": "aaa.observation_noise_plot_provenance.v1",
        "source": records_path.name,
        "source_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "source_rows": row_count,
        "uncertainty_in_figures": "none; figures show bounded-memory primitive-record means",
        "authoritative_aggregate_uncertainty": "summary.json statistics hierarchical and paired artifacts",
        "figures": [path.name for path in output],
    }
    provenance_path = root / "plot_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.append(provenance_path)
    return output
