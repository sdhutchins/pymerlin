import pytest

from pymerlin.benchmark import (
    benchmark_error_detection,
    benchmark_marker,
    benchmark_tree_multipoint,
)
from pymerlin.likelihood import MarkerTreeBudgetExceeded


def test_benchmark_marker_reports_timing_fields():
    summary = benchmark_marker(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "some_marker",
        "examples/basic2.map",
        "examples/basic2.freq",
        repeats=1,
    )

    assert summary["marker"] == "some_marker"
    assert summary["families"] == 1
    assert summary["individuals"] == 6
    assert summary["ibd_rows"] == 15
    assert summary["workers"] == 1
    assert summary["load_seconds"] >= 0.0
    assert summary["best_run_seconds"] >= 0.0


def test_benchmark_marker_can_duplicate_families_for_scaling():
    summary = benchmark_marker(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "some_marker",
        "examples/basic2.map",
        "examples/basic2.freq",
        repeats=1,
        family_copies=3,
    )

    assert summary["families"] == 3
    assert summary["individuals"] == 18
    assert summary["ibd_rows"] == 45


def test_error_benchmark_reports_parallel_workload_fields() -> None:
    summary = benchmark_error_detection(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
        repeats=1,
        workers=2,
    )

    assert summary["workload"] == "error"
    assert summary["families"] == 1
    assert summary["errors"] == 0
    assert summary["workers"] == 2
    assert summary["best_run_seconds"] >= 0.0


def test_tree_benchmark_reports_phase_and_reduction_fields() -> None:
    summary = benchmark_tree_multipoint(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
        marker_limit=2,
        workers=1,
    )

    assert summary["workload"] == "tree-multipoint"
    assert summary["family"] == "1"
    assert summary["markers"] == 2
    assert summary["workers"] == 1
    assert summary["emission_unique_nodes_max"] >= 1
    assert summary["posterior_unique_nodes_max"] >= 1
    assert summary["suffix_cache_hits"] >= 0
    assert summary["suffix_cache_misses"] >= 0
    assert summary["peeled_components"] >= 0
    assert summary["peeled_constraints"] >= 0
    assert summary["zero_peeled_factors"] >= 0
    assert summary["normalized_cache_reuses"] >= 0
    assert summary["peeled_factor_cache_hits"] >= 0
    assert summary["peeled_factor_cache_misses"] >= 0
    assert summary["scaled_tree_cache_hits"] >= 0
    assert summary["recursive_nodes"] >= 1
    assert summary["maximum_recursion_depth"] >= 0
    assert summary["contradiction_prunes"] >= 0
    assert summary["founder_orientation_reductions"] >= 0
    assert summary["founder_couple_reductions"] >= 0
    assert summary["counting_reductions"] >= 0
    assert summary["invariant_reductions"] >= 0
    assert summary["emission_seconds"] >= 0.0
    assert summary["forward_backward_seconds"] >= 0.0
    assert summary["posterior_seconds"] >= 0.0


def test_tree_benchmark_can_stop_at_a_diagnostic_node_budget() -> None:
    messages: list[str] = []

    with pytest.raises(MarkerTreeBudgetExceeded, match="node budget"):
        benchmark_tree_multipoint(
            "examples/basic2.ped",
            "examples/basic2.dat",
            "examples/basic2.map",
            "examples/basic2.freq",
            marker_limit=1,
            workers=1,
            progress=messages.append,
            heartbeat_node_interval=1,
            emission_node_limit=1,
        )

    assert any("marker tree heartbeat" in message for message in messages)
    assert any("nodes=1" in message for message in messages)
