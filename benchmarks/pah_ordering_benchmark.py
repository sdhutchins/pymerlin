"""Compare exact meiosis coordinate orders on the synthetic PAH fixture."""

from __future__ import annotations

import logging
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from pymerlin import (
    MarkerTreeBudgetExceeded,
    audit_paired_dag_transition,
    build_founder_orientation_quotient,
    family_marker_likelihood_tree,
    load_merlin_inputs,
    order_family_meioses,
    reduce_founder_orientation_tree,
)
from pymerlin.map import haldane_recombination_fraction, map_distance_cm
from pymerlin.meiosis_ordering import MeiosisOrderingName
from pymerlin.models import Family, Marker
from tests.pah_scale_fixture import build_pah_scale_inputs

logger = logging.getLogger(__name__)

_OUTPUT_PATH_ENVIRONMENT_VARIABLE = "PYMERLIN_ORDERING_RESULT_PATH"
_SOURCE_SIGNATURE_ENVIRONMENT_VARIABLE = "PYMERLIN_ORDERING_SOURCE_SIGNATURE"
_STATE_LIMIT_ENVIRONMENT_VARIABLE = "PYMERLIN_ORDERING_STATE_LIMIT"
_MARKER_NODE_LIMIT_ENVIRONMENT_VARIABLE = "PYMERLIN_ORDERING_MARKER_NODE_LIMIT"
_MARKER_TIME_LIMIT_ENVIRONMENT_VARIABLE = "PYMERLIN_ORDERING_MARKER_TIME_LIMIT"
_ORDERING_NAMES: tuple[MeiosisOrderingName, ...] = (
    "current",
    "individual_identifier",
    "parent_before_child",
)
_RESULT_COLUMNS = (
    "ordering",
    "marker_trees_complete",
    "marker_tree_seconds",
    "failure",
    "maximum_recursive_nodes",
    "maximum_emission_unique_nodes",
    "full_active_bits",
    "reduced_bits",
    "reduced_active_bits",
    "examined_unique_subproblems",
    "maximum_frontier_states",
    "deepest_bit_index",
    "audit_complete",
)


@dataclass(frozen=True)
class OrderingBenchmarkResult:
    """One bounded ordering outcome on the same two-marker PAH interval."""

    ordering_name: MeiosisOrderingName
    completed_marker_trees: bool
    marker_tree_seconds: float
    failure: str | None
    maximum_recursive_node_count: int | None = None
    maximum_emission_unique_node_count: int | None = None
    full_active_bit_count: int | None = None
    reduced_bit_count: int | None = None
    reduced_active_bit_count: int | None = None
    examined_unique_subproblem_count: int | None = None
    maximum_frontier_state_count: int | None = None
    deepest_bit_index_reached: int | None = None
    audit_complete: bool | None = None


def run_pah_ordering_benchmark(
    output_path: Path,
    *,
    state_limit: int,
    marker_node_limit: int,
    marker_time_limit_seconds: float,
    source_signature: str,
) -> None:
    """Run bounded exact coordinate-order comparisons and write one artifact."""

    _validate_positive_integer(state_limit, "State limit")
    _validate_positive_integer(marker_node_limit, "Marker node limit")
    if (
        not math.isfinite(marker_time_limit_seconds)
        or marker_time_limit_seconds <= 0.0
    ):
        raise ValueError("Marker time limit must be finite and positive.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_started_at = perf_counter()
    with TemporaryDirectory(
        prefix="pymerlin-pah-ordering-",
        dir=output_path.parent,
    ) as temporary_directory:
        input_paths = build_pah_scale_inputs(Path(temporary_directory))
        dataset = load_merlin_inputs(
            input_paths.ped,
            input_paths.dat,
            input_paths.map,
            input_paths.freq,
        )
        source_family = dataset.families[0]
        theta = haldane_recombination_fraction(
            map_distance_cm(
                float(dataset.markers[0].position_cm),
                float(dataset.markers[1].position_cm),
            )
        )
        results = tuple(
            _benchmark_ordering(
                ordering_name,
                source_family,
                dataset.markers[:2],
                theta,
                state_limit,
                marker_node_limit,
                marker_time_limit_seconds,
            )
            for ordering_name in _ORDERING_NAMES
        )

    result_text = _format_benchmark_results(
        results,
        source_signature=source_signature,
        state_limit=state_limit,
        marker_node_limit=marker_node_limit,
        marker_time_limit_seconds=marker_time_limit_seconds,
        total_seconds=perf_counter() - total_started_at,
    )
    output_path.write_text(result_text, encoding="utf-8")
    logger.info("Wrote bounded ordering result to %s.", output_path)


def _benchmark_ordering(
    ordering_name: MeiosisOrderingName,
    source_family: Family,
    markers: tuple[Marker, ...],
    theta: float,
    state_limit: int,
    marker_node_limit: int,
    marker_time_limit_seconds: float,
) -> OrderingBenchmarkResult:
    """Return one result while preserving bounded candidate failures."""

    logger.info("Building marker trees for ordering %s.", ordering_name)
    family = order_family_meioses(source_family, ordering_name).family
    marker_tree_started_at = perf_counter()
    try:
        marker_trees = tuple(
            family_marker_likelihood_tree(
                family,
                marker,
                node_limit=marker_node_limit,
                time_limit_seconds=marker_time_limit_seconds,
            )
            for marker in markers
        )
    except MarkerTreeBudgetExceeded as error:
        return OrderingBenchmarkResult(
            ordering_name=ordering_name,
            completed_marker_trees=False,
            marker_tree_seconds=perf_counter() - marker_tree_started_at,
            failure=str(error),
        )
    marker_tree_seconds = perf_counter() - marker_tree_started_at

    full_audit = audit_paired_dag_transition(
        marker_trees[0],
        marker_trees[1],
        theta,
        maximum_unique_subproblems=1,
    )
    quotient = build_founder_orientation_quotient(family)
    reduced_trees = tuple(
        reduce_founder_orientation_tree(tree, quotient)
        for tree in marker_trees
    )
    reduced_audit = audit_paired_dag_transition(
        reduced_trees[0],
        reduced_trees[1],
        theta,
        founder_quotient=quotient,
        maximum_unique_subproblems=state_limit,
    )
    recursive_node_counts = tuple(
        tree.build_statistics.recursive_node_count
        for tree in marker_trees
        if tree.build_statistics is not None
    )
    if len(recursive_node_counts) != len(marker_trees):
        raise RuntimeError("Marker trees are missing build statistics.")

    return OrderingBenchmarkResult(
        ordering_name=ordering_name,
        completed_marker_trees=True,
        marker_tree_seconds=marker_tree_seconds,
        failure=None,
        maximum_recursive_node_count=max(recursive_node_counts),
        maximum_emission_unique_node_count=max(
            tree.unique_node_count() for tree in marker_trees
        ),
        full_active_bit_count=full_audit.active_bit_count,
        reduced_bit_count=quotient.reduced_bit_count,
        reduced_active_bit_count=reduced_audit.active_bit_count,
        examined_unique_subproblem_count=(
            reduced_audit.examined_unique_subproblem_count
        ),
        maximum_frontier_state_count=(
            reduced_audit.maximum_frontier_state_count
        ),
        deepest_bit_index_reached=reduced_audit.deepest_bit_index_reached,
        audit_complete=reduced_audit.complete,
    )


def _format_benchmark_results(
    results: tuple[OrderingBenchmarkResult, ...],
    *,
    source_signature: str,
    state_limit: int,
    marker_node_limit: int,
    marker_time_limit_seconds: float,
    total_seconds: float,
) -> str:
    """Format one deterministic-schema TSV benchmark artifact."""

    metadata_lines = (
        "benchmark_name\tpah_meiosis_ordering",
        "benchmark_completed\ttrue",
        f"source_signature\t{source_signature}",
        f"slurm_job_id\t{os.environ.get('SLURM_JOB_ID', 'not_slurm')}",
        f"python_version\t{platform.python_version()}",
        f"platform\t{platform.platform()}",
        f"requested_state_limit\t{state_limit}",
        f"marker_node_limit\t{marker_node_limit}",
        f"marker_time_limit_seconds\t{marker_time_limit_seconds:.6f}",
        f"total_seconds\t{total_seconds:.6f}",
    )
    header = "\t".join(_RESULT_COLUMNS)
    result_lines = tuple(_format_result_row(result) for result in results)
    return "\n".join((*metadata_lines, header, *result_lines, ""))


def _format_result_row(result: OrderingBenchmarkResult) -> str:
    """Format one ordering row without ambiguous tabs or newlines."""

    failure = (result.failure or "").replace("\t", " ").replace("\n", " ")
    values = (
        result.ordering_name,
        str(result.completed_marker_trees).lower(),
        f"{result.marker_tree_seconds:.6f}",
        failure,
        _optional_value(result.maximum_recursive_node_count),
        _optional_value(result.maximum_emission_unique_node_count),
        _optional_value(result.full_active_bit_count),
        _optional_value(result.reduced_bit_count),
        _optional_value(result.reduced_active_bit_count),
        _optional_value(result.examined_unique_subproblem_count),
        _optional_value(result.maximum_frontier_state_count),
        _optional_value(result.deepest_bit_index_reached),
        (
            ""
            if result.audit_complete is None
            else str(result.audit_complete).lower()
        ),
    )
    return "\t".join(values)


def _optional_value(value: int | None) -> str:
    """Format absent metrics as empty TSV fields."""

    return "" if value is None else str(value)


def _validate_positive_integer(value: int, label: str) -> None:
    """Reject booleans and nonpositive benchmark bounds."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")


def _required_environment_value(variable_name: str) -> str:
    """Return one required nonempty environment value."""

    value = os.environ.get(variable_name)
    if not value:
        raise RuntimeError(f"{variable_name} must be set.")
    return value


def main() -> None:
    """Read the fixed benchmark contract from environment variables."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
    )
    run_pah_ordering_benchmark(
        Path(_required_environment_value(_OUTPUT_PATH_ENVIRONMENT_VARIABLE)),
        state_limit=int(
            _required_environment_value(_STATE_LIMIT_ENVIRONMENT_VARIABLE)
        ),
        marker_node_limit=int(
            _required_environment_value(
                _MARKER_NODE_LIMIT_ENVIRONMENT_VARIABLE
            )
        ),
        marker_time_limit_seconds=float(
            _required_environment_value(
                _MARKER_TIME_LIMIT_ENVIRONMENT_VARIABLE
            )
        ),
        source_signature=os.environ.get(
            _SOURCE_SIGNATURE_ENVIRONMENT_VARIABLE,
            "local-unversioned",
        ),
    )


if __name__ == "__main__":
    main()
