"""Bounded, development-only observation-noise candidate selection."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..benchmark.evidence import sha256_file
from .candidates import CandidateDefinition, candidate_catalog
from .runner import AttemptResult, project_root, run_attempt
from .spec import canonical_protocol_hash, load_protocol
from .statistics import paired_hierarchical_comparisons
from .verifier import iter_records, verify_attempt

_REQUIRED_SELECTION_PREDICTORS = {
    "selected_candidate_online",
    "selected_candidate_frozen",
    "constant_motion_reflected",
    "incumbent_square_root_rls",
}


def _complete_primary_cells(
    by_cell: dict[tuple[Any, ...], dict[str, list[float]]], *, expected_cells: int = 32
) -> bool:
    """Require every predictor key while accepting legitimate zero metrics."""

    return len(by_cell) == expected_cells and all(
        all(_aggregate_mean(cell_values.get(name)) is not None for name in _REQUIRED_SELECTION_PREDICTORS)
        for cell_values in by_cell.values()
    )


def _evidence_destination(path: str | Path, *, quick: bool) -> Path:
    destination = Path(path)
    canonical = project_root() / "docs/evidence/observation_noise_development_selection.json"
    if quick and destination.resolve() == canonical.resolve():
        raise ValueError("quick selection cannot overwrite the canonical full-selection evidence")
    return destination


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _accumulate(bucket: dict[str, list[float]], key: str, value: float) -> None:
    aggregate = bucket.setdefault(key, [0.0, 0.0])
    aggregate[0] += value
    aggregate[1] += 1.0


def _aggregate_mean(value: list[float] | None) -> float | None:
    return None if value is None or value[1] == 0.0 else value[0] / value[1]


def _candidate_metrics(attempt: AttemptResult) -> dict[str, Any]:
    """Compute descriptive development metrics from primitive records."""

    records = iter_records(attempt.directory)
    by_cell: dict[tuple[str, str, str, float, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    adaptation: dict[tuple[str, str, float], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        trial = record["trial"]
        if trial["channel"] not in {"gaussian", "uniform"} or float(trial["scale"]) not in {
            0.0005,
            0.002,
        }:
            continue
        if trial["branch"] == "stationary" and trial["family"] in {
            "constant_velocity",
            "bouncing",
            "changed_law",
            "speed_change",
        }:
            cell_key = (
                str(trial["condition"]),
                str(trial["family"]),
                str(trial["channel"]),
                float(trial["scale"]),
                str(trial["branch"]),
            )
            for predictor in (
                "selected_candidate_online",
                "selected_candidate_frozen",
                "constant_motion_reflected",
                "incumbent_square_root_rls",
            ):
                if predictor in record["predictions"]:
                    _accumulate(
                        by_cell[cell_key],
                        predictor,
                        float(record["predictions"][predictor]["latent_normalized_absolute_error"]),
                    )
        if (
            trial["family"] == "changed_law"
            and trial["branch"] in {"frozen", "online"}
            and int(record["step"]) < 350
        ):
            adaptation_key = (str(trial["condition"]), str(trial["channel"]), float(trial["scale"]))
            predictor = (
                "selected_candidate_branch_online"
                if trial["branch"] == "online"
                else "selected_candidate_branch_frozen"
            )
            if predictor in record["predictions"]:
                _accumulate(
                    adaptation[adaptation_key],
                    predictor,
                    float(record["predictions"][predictor]["latent_normalized_absolute_error"]),
                )
    cell_rows: list[dict[str, Any]] = []
    for cell_key, cell_values in sorted(by_cell.items(), key=str):
        candidate = _aggregate_mean(cell_values.get("selected_candidate_online"))
        frozen_candidate = _aggregate_mean(cell_values.get("selected_candidate_frozen"))
        baseline = _aggregate_mean(cell_values.get("constant_motion_reflected"))
        incumbent = _aggregate_mean(cell_values.get("incumbent_square_root_rls"))
        cell_rows.append(
            {
                "condition": cell_key[0],
                "family": cell_key[1],
                "channel": cell_key[2],
                "scale": cell_key[3],
                "branch": cell_key[4],
                "candidate_mae": candidate,
                "frozen_candidate_mae": frozen_candidate,
                "reflected_baseline_mae": baseline,
                "incumbent_mae": incumbent,
                "candidate_minus_1_10_baseline": (
                    None if candidate is None or baseline is None else candidate - 1.10 * baseline
                ),
                "incumbent_minus_candidate": (
                    None if candidate is None or incumbent is None else incumbent - candidate
                ),
            }
        )
    adaptation_rows: list[dict[str, Any]] = []
    for adaptation_key, adaptation_values in sorted(adaptation.items(), key=str):
        frozen = _aggregate_mean(adaptation_values.get("selected_candidate_branch_frozen"))
        online = _aggregate_mean(adaptation_values.get("selected_candidate_branch_online"))
        adaptation_rows.append(
            {
                "condition": adaptation_key[0],
                "channel": adaptation_key[1],
                "scale": adaptation_key[2],
                "frozen_first50_mae": frozen,
                "online_first50_mae": online,
                "point_reduction": (
                    None if frozen in {None, 0.0} or online is None else (frozen - online) / frozen
                ),
            }
        )
    candidates = [row["candidate_mae"] for row in cell_rows if row["candidate_mae"] is not None]
    baselines = [
        row["candidate_minus_1_10_baseline"]
        for row in cell_rows
        if row["candidate_minus_1_10_baseline"] is not None
    ]
    incumbent_gains = [
        row["incumbent_minus_candidate"] for row in cell_rows if row["incumbent_minus_candidate"] is not None
    ]
    reductions = [row["point_reduction"] for row in adaptation_rows if row["point_reduction"] is not None]
    retention = [
        row["candidate_mae"] - row["frozen_candidate_mae"]
        for row in cell_rows
        if row["family"] in {"constant_velocity", "bouncing"}
        and row["candidate_mae"] is not None
        and row["frozen_candidate_mae"] is not None
    ]
    complete_cells = _complete_primary_cells(by_cell)
    primary_records = (
        record
        for record in iter_records(attempt.directory)
        if record["trial"]["branch"] == "stationary"
        and record["trial"]["channel"] in {"gaussian", "uniform"}
        and float(record["trial"]["scale"]) in {0.0005, 0.002}
        and record["trial"]["family"] in {"constant_velocity", "bouncing", "changed_law", "speed_change"}
    )
    promotion = paired_hierarchical_comparisons(
        primary_records,
        baseline="selected_candidate_online",
        targets=("incumbent_square_root_rls",),
        draws=4000,
        seed=int(canonical_protocol_hash()[:16], 16),
        levels=(0.95,),
    )
    promotion_rows = list(promotion["cells"].values())
    practical_margin = 0.00001
    promotion_supported = len(promotion_rows) == 32 and all(
        row["evidence_status"] == "PASS"
        and row["holm_adjusted_p_value"] <= 0.05
        and row["familywise_intervals"]["0.95"]["lower"] > practical_margin
        for row in promotion_rows
    )
    verification = verify_attempt(attempt.directory)
    return {
        "primary_candidate_mae_mean": _mean(candidates),
        "baseline_margin_mean": _mean(baselines),
        "incumbent_gain_mean": _mean(incumbent_gains),
        "adaptation_reduction_mean": _mean(reductions),
        "retention_delta_mean": _mean(retention),
        "complete_primary_cell_coverage": complete_cells,
        "resource_and_numerical_stability": verification["verdict"] == "PASS",
        "promotion_adjusted_positive_evidence": promotion_supported,
        "promotion_comparison": promotion,
        "cells": cell_rows,
        "adaptation_cells": adaptation_rows,
        "primitive_verifier": verification["verdict"],
    }


def _ledger_entry(
    definition: CandidateDefinition,
    attempt: AttemptResult,
    metrics: dict[str, Any],
    selected_candidate: str,
) -> dict[str, Any]:
    manifest = json.loads((attempt.directory / "run_manifest.json").read_text(encoding="utf-8"))
    outcome = "selected" if definition.candidate_id == selected_candidate else "rejected"
    reason = (
        "Selected by the preregistered conjunctive development ranking; confirmation remains unexecuted."
        if outcome == "selected"
        else "Not selected by the preregistered development ranking; retained as a rejected bounded attempt."
    )
    entry = definition.to_dict()
    entry.update(
        {
            "development_attempts": [attempt.directory.name],
            "outcome": outcome,
            "outcome_reason": reason,
            "development_evidence": {
                "attempt_id": attempt.directory.name,
                "archive_locator": "transient_local_selection_root_not_committed",
                "summary_sha256": sha256_file(attempt.directory / "summary.json"),
                "records_sha256": manifest["records_sha256"],
                "record_shard_index_sha256": sha256_file(attempt.directory / "records/index.json"),
                "metrics": metrics,
            },
        }
    )
    return entry


def run_development_selection(
    *,
    output_root: str | Path,
    evidence_path: str | Path,
    quick: bool,
    reuse_root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate the committed catalog without consuming confirmation data."""

    protocol = load_protocol()
    plan_path = project_root() / "benchmarks/observation_noise_development_selection.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("status") != "selection_plan_frozen"
        or plan.get("frozen_before_comparative_development") is not True
    ):
        raise ValueError("development selection plan is not frozen")
    catalog = candidate_catalog()
    if list(catalog) != plan.get("candidates"):
        raise ValueError("development plan candidate catalog differs from code")
    if len(catalog) > int(protocol.raw["search_budget"]["max_candidate_configurations"]):
        raise ValueError("candidate search exceeds the frozen configuration budget")
    sample = plan.get("sample", {})
    expected_counts = (
        int(sample.get("lineages", -1)),
        int(sample.get("episodes_per_family_per_lineage", -1)),
        int(sample.get("sensor_realizations_per_episode", -1)),
    )
    if expected_counts != (2, 2, 1):
        raise ValueError("development plan counts differ from implemented full-plan counts")
    attempts: dict[str, AttemptResult] = {}
    metrics_by_candidate: dict[str, dict[str, Any]] = {}
    for candidate_id in catalog:
        if reuse_root is None:
            attempt = run_attempt(
                role="development",
                output_root=output_root,
                attempt_label=f"selection-{candidate_id}",
                quick=quick,
                candidate_id=candidate_id,
            )
        else:
            directory = Path(reuse_root) / "observation-noise-v1_1" / f"selection-{candidate_id}"
            summary_path = directory / "summary.json"
            if not summary_path.is_file():
                raise ValueError(f"cannot reuse incomplete selection attempt: {directory}")
            attempt = AttemptResult(
                directory=directory, summary=json.loads(summary_path.read_text(encoding="utf-8"))
            )
        attempts[candidate_id] = attempt
        manifest = json.loads((attempt.directory / "run_manifest.json").read_text(encoding="utf-8"))
        actual_counts = (
            manifest.get("lineages"),
            manifest.get("episodes_per_family_per_lineage"),
            manifest.get("sensor_realizations_per_episode"),
        )
        if actual_counts != ((1, 1, 1) if quick else expected_counts):
            raise ValueError(f"selection archive {attempt.directory} has wrong hierarchy counts")
        if (
            manifest.get("selected_candidate") != candidate_id
            or manifest.get("protocol_hash") != canonical_protocol_hash()
            or manifest.get("role") != "development"
        ):
            raise ValueError(f"selection archive {attempt.directory} has incompatible identity")
        metrics_by_candidate[candidate_id] = _candidate_metrics(attempt)
    # The incumbent is the no-refinement control. A refinement is eligible only
    # if its development point estimates satisfy every declared practical
    # constraint; ties resolve by configuration hash then ID.
    eligible = [
        candidate_id
        for candidate_id, metrics in metrics_by_candidate.items()
        if candidate_id != "incumbent-no-refinement-v1"
        and metrics["complete_primary_cell_coverage"] is True
        and metrics["resource_and_numerical_stability"] is True
        and metrics["primary_candidate_mae_mean"] is not None
        and metrics["primary_candidate_mae_mean"] <= 0.02
        and metrics["baseline_margin_mean"] is not None
        and metrics["baseline_margin_mean"] <= 0.00001
        and metrics["retention_delta_mean"] is not None
        and metrics["retention_delta_mean"] <= 0.00001
        and metrics["adaptation_cells"]
        and all(
            row["point_reduction"] is not None and row["point_reduction"] > 0.10
            for row in metrics["adaptation_cells"]
        )
        and metrics["incumbent_gain_mean"] is not None
        and metrics["incumbent_gain_mean"] > float(plan["ranking"]["practical_margin"])
        and metrics["promotion_adjusted_positive_evidence"] is True
    ]
    selected = min(
        eligible or ["incumbent-no-refinement-v1"],
        key=lambda candidate_id: (catalog[candidate_id].configuration_hash, candidate_id),
    )
    selection = {
        "schema_version": "aaa.observation_noise_development_selection_evidence.v1",
        "protocol_version": protocol.protocol_version,
        "protocol_hash": canonical_protocol_hash(),
        "plan_path": str(plan_path.relative_to(project_root())),
        "quick": quick,
        "status": "development_smoke" if quick else "development_complete",
        "selected_candidate": None if quick else selected,
        "provisional_candidate": selected if quick else None,
        "candidate_order": list(catalog),
        "attempts": {candidate_id: attempts[candidate_id].directory.name for candidate_id in catalog},
        "archive_locator": "transient_local_selection_root_not_committed",
        "metrics": metrics_by_candidate,
        "ranking": {
            "eligible_refinements": eligible,
            "tie_break": "configuration_hash ascending then candidate_id ascending",
            "practical_margin": 0.00001,
            "decision": "No refinement is selected unless all preregistered development constraints are met.",
        },
    }
    evidence_destination = _evidence_destination(evidence_path, quick=quick)
    evidence_destination.parent.mkdir(parents=True, exist_ok=True)
    evidence_destination.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if quick:
        return selection
    ledger_path = project_root() / protocol.raw["search_budget"]["candidate_ledger_file"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["protocol_version"] = protocol.protocol_version
    ledger["protocol_hash"] = canonical_protocol_hash()
    ledger["status"] = "development_complete"
    ledger["selected_candidate"] = selected
    ledger["selection_evidence"] = str(evidence_destination)
    ledger["entries"] = []
    for candidate_id, definition in catalog.items():
        ledger["entries"].append(
            _ledger_entry(definition, attempts[candidate_id], metrics_by_candidate[candidate_id], selected)
        )
    ledger["notes"] = "All bounded development attempts are retained. Confirmation data was not inspected."
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return selection
