"""Tests for the bounded PAH meiosis-ordering benchmark."""

from benchmarks.pah_ordering_benchmark import (
    OrderingBenchmarkResult,
    _benchmark_ordering,
    _format_benchmark_results,
)
from pymerlin import load_merlin_inputs


def test_ordering_benchmark_completes_and_preserves_bounded_failures() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    family = dataset.families[0]
    markers = dataset.markers[:2]

    completed = _benchmark_ordering(
        "current",
        family,
        markers,
        0.1,
        state_limit=1_000,
        marker_node_limit=1_000_000,
        marker_time_limit_seconds=5.0,
    )
    bounded_failure = _benchmark_ordering(
        "current",
        family,
        markers,
        0.1,
        state_limit=1_000,
        marker_node_limit=1,
        marker_time_limit_seconds=5.0,
    )

    assert completed.completed_marker_trees
    assert completed.failure is None
    assert completed.full_active_bit_count is not None
    assert completed.reduced_active_bit_count is not None
    assert not bounded_failure.completed_marker_trees
    assert "node budget reached" in (bounded_failure.failure or "")
    assert bounded_failure.examined_unique_subproblem_count is None


def test_ordering_benchmark_format_retains_failed_candidate() -> None:
    results = (
        OrderingBenchmarkResult(
            ordering_name="current",
            completed_marker_trees=True,
            marker_tree_seconds=1.25,
            failure=None,
            maximum_recursive_node_count=10,
            maximum_emission_unique_node_count=20,
            full_active_bit_count=4,
            reduced_bit_count=3,
            reduced_active_bit_count=3,
            examined_unique_subproblem_count=50,
            maximum_frontier_state_count=12,
            deepest_bit_index_reached=3,
            audit_complete=True,
        ),
        OrderingBenchmarkResult(
            ordering_name="individual_identifier",
            completed_marker_trees=False,
            marker_tree_seconds=2.5,
            failure="Marker tree\tbudget\nreached.",
        ),
    )

    report = _format_benchmark_results(
        results,
        source_signature="test-signature",
        state_limit=50,
        marker_node_limit=100,
        marker_time_limit_seconds=3.0,
        total_seconds=4.0,
    )

    assert "benchmark_name\tpah_meiosis_ordering" in report
    assert "current\ttrue\t1.250000\t\t10\t20\t4\t3\t3\t50\t12\t3\ttrue" in report
    assert (
        "individual_identifier\tfalse\t2.500000\t"
        "Marker tree budget reached.\t\t\t\t\t\t\t\t\t"
    ) in report
