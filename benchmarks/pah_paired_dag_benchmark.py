"""Run one bounded paired-DAG audit on the synthetic PAH-scale fixture."""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from pymerlin import (
    audit_founder_couple_key_effect,
    audit_paired_dag_transition,
    build_founder_couple_quotient,
    build_founder_orientation_quotient,
    family_marker_likelihood_tree,
    format_founder_couple_key_audit,
    format_paired_dag_transition_audit,
    load_merlin_inputs,
    reduce_founder_orientation_tree,
)
from pymerlin.map import haldane_recombination_fraction, map_distance_cm
from tests.pah_scale_fixture import build_pah_scale_inputs

logger = logging.getLogger(__name__)

_OUTPUT_PATH_ENVIRONMENT_VARIABLE = "PYMERLIN_BENCHMARK_RESULT_PATH"
_SOURCE_SIGNATURE_ENVIRONMENT_VARIABLE = "PYMERLIN_BENCHMARK_SOURCE_SIGNATURE"
_STATE_LIMIT_ENVIRONMENT_VARIABLE = "PYMERLIN_PAIRED_DAG_STATE_LIMIT"


def run_pah_paired_dag_benchmark(
    output_path: Path,
    state_limit: int,
    source_signature: str,
) -> None:
    """Build two PAH marker trees and write one bounded audit atomically.

    Inputs are the deterministic synthetic genotypes generated from the
    anonymized PAH pedigree topology. The output is a planning diagnostic, not
    a linkage result. The caller owns atomic promotion of this partial file.
    """

    if state_limit <= 0:
        raise ValueError("The paired-DAG state limit must be positive.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_started_at = perf_counter()
    with TemporaryDirectory(
        prefix="pymerlin-pah-paired-dag-",
        dir=output_path.parent,
    ) as temporary_directory:
        logger.info("Building deterministic PAH-scale benchmark inputs.")
        input_paths = build_pah_scale_inputs(Path(temporary_directory))
        dataset = load_merlin_inputs(
            input_paths.ped,
            input_paths.dat,
            input_paths.map,
            input_paths.freq,
        )
        family = dataset.families[0]
        orientation_quotient = build_founder_orientation_quotient(family)
        couple_quotient = build_founder_couple_quotient(
            family,
            orientation_quotient,
        )

        marker_tree_started_at = perf_counter()
        orientation_trees = tuple(
            reduce_founder_orientation_tree(
                family_marker_likelihood_tree(family, marker),
                orientation_quotient,
            )
            for marker in dataset.markers[:2]
        )
        marker_tree_seconds = perf_counter() - marker_tree_started_at

        theta = haldane_recombination_fraction(
            map_distance_cm(
                float(dataset.markers[0].position_cm),
                float(dataset.markers[1].position_cm),
            )
        )
        key_audit_started_at = perf_counter()
        founder_couple_key_audit = audit_founder_couple_key_effect(
            orientation_trees[0],
            orientation_trees[1],
            theta,
            couple_quotient,
        )
        key_audit_seconds = perf_counter() - key_audit_started_at
        logger.info(
            "Auditing at most %s unique paired-DAG subproblems.",
            f"{state_limit:,}",
        )
        audit_started_at = perf_counter()
        audit = audit_paired_dag_transition(
            orientation_trees[0],
            orientation_trees[1],
            theta,
            founder_quotient=orientation_quotient,
            maximum_unique_subproblems=state_limit,
        )
        audit_seconds = perf_counter() - audit_started_at

    total_seconds = perf_counter() - total_started_at
    result_text = _format_benchmark_result(
        source_signature=source_signature,
        state_limit=state_limit,
        marker_tree_seconds=marker_tree_seconds,
        key_audit_seconds=key_audit_seconds,
        audit_seconds=audit_seconds,
        total_seconds=total_seconds,
        founder_couple_key_audit_report=format_founder_couple_key_audit(
            founder_couple_key_audit
        ),
        audit_report=format_paired_dag_transition_audit(audit),
    )
    output_path.write_text(result_text, encoding="utf-8")
    logger.info("Wrote partial benchmark result to %s.", output_path)


def _format_benchmark_result(
    *,
    source_signature: str,
    state_limit: int,
    marker_tree_seconds: float,
    key_audit_seconds: float,
    audit_seconds: float,
    total_seconds: float,
    founder_couple_key_audit_report: str,
    audit_report: str,
) -> str:
    """Return one deterministic-schema tabular benchmark artifact."""

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "not_slurm")
    metadata_lines = (
        "benchmark_name\tpah_paired_dag",
        "benchmark_completed\ttrue",
        f"source_signature\t{source_signature}",
        f"slurm_job_id\t{slurm_job_id}",
        f"python_version\t{platform.python_version()}",
        f"platform\t{platform.platform()}",
        f"requested_state_limit\t{state_limit}",
        f"marker_tree_seconds\t{marker_tree_seconds:.6f}",
        f"founder_couple_key_audit_seconds\t{key_audit_seconds:.6f}",
        f"audit_seconds\t{audit_seconds:.6f}",
        f"total_seconds\t{total_seconds:.6f}",
    )
    return "\n".join(
        (
            *metadata_lines,
            founder_couple_key_audit_report,
            audit_report,
            "",
        )
    )


def main() -> None:
    """Read the fixed benchmark contract from the Slurm wrapper environment."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
    )
    output_value = os.environ.get(_OUTPUT_PATH_ENVIRONMENT_VARIABLE)
    if not output_value:
        raise RuntimeError(
            f"{_OUTPUT_PATH_ENVIRONMENT_VARIABLE} must name the result file."
        )
    state_limit_value = os.environ.get(_STATE_LIMIT_ENVIRONMENT_VARIABLE)
    if not state_limit_value:
        raise RuntimeError(f"{_STATE_LIMIT_ENVIRONMENT_VARIABLE} must be set.")
    try:
        state_limit = int(state_limit_value)
    except ValueError as error:
        raise ValueError(
            f"{_STATE_LIMIT_ENVIRONMENT_VARIABLE} must be an integer."
        ) from error

    source_signature = os.environ.get(
        _SOURCE_SIGNATURE_ENVIRONMENT_VARIABLE,
        "local-unversioned",
    )
    run_pah_paired_dag_benchmark(
        Path(output_value),
        state_limit,
        source_signature,
    )


if __name__ == "__main__":
    main()
