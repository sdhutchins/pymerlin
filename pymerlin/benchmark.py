"""Small benchmark harness for backend and scaling decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from time import perf_counter

import numpy as np

from .chain_reduction import detect_untyped_chains
from .error_detection import detect_unlikely_genotypes
from .ibd import estimate_ibd
from .io import load_merlin_inputs
from .likelihood import (
    _marker_relevant_individual_ids,
    _marker_relevant_meiosis_indices,
)
from .map import haldane_recombination_fraction, map_distance_cm
from .models import Dataset, Family
from .multipoint import (
    _family_marker_likelihood_trees,
    _tree_forward_backward_trees,
)
from .parallel import validate_workers


def benchmark_marker(
    ped_path: str,
    dat_path: str,
    marker: str,
    map_path: str | None = None,
    freq_path: str | None = None,
    repeats: int = 3,
    family_copies: int = 1,
    workers: int = 1,
) -> dict[str, float | int | str]:
    """Profile the current reference path for one marker.

    The benchmark intentionally times the public API. Later GPU prototypes
    should beat this end-to-end path before replacing any backend code.
    """

    _validate_benchmark_arguments(repeats, family_copies)
    workers = validate_workers(workers)
    load_start = perf_counter()
    dataset = load_merlin_inputs(ped_path, dat_path, map_path, freq_path)
    load_seconds = perf_counter() - load_start
    if family_copies > 1:
        dataset = repeat_families(dataset, family_copies)

    run_seconds = []
    row_count = 0
    for _ in range(repeats):
        run_start = perf_counter()
        result = estimate_ibd(dataset, marker, workers=workers)
        run_seconds.append(perf_counter() - run_start)
        row_count = len(result.rows)

    return {
        "marker": marker,
        "families": len(dataset.families),
        "individuals": sum(len(family.individuals) for family in dataset.families),
        "ibd_rows": row_count,
        "workers": workers,
        "load_seconds": load_seconds,
        "best_run_seconds": min(run_seconds),
        "mean_run_seconds": float(np.mean(run_seconds)),
    }


def benchmark_error_detection(
    ped_path: str,
    dat_path: str,
    map_path: str,
    freq_path: str | None = None,
    repeats: int = 3,
    family_copies: int = 1,
    workers: int = 1,
) -> dict[str, float | int | str]:
    """Profile multipoint genotype error detection end to end."""

    _validate_benchmark_arguments(repeats, family_copies)
    workers = validate_workers(workers)
    load_start = perf_counter()
    dataset = load_merlin_inputs(ped_path, dat_path, map_path, freq_path)
    load_seconds = perf_counter() - load_start
    if family_copies > 1:
        dataset = repeat_families(dataset, family_copies)

    run_seconds = []
    error_count = 0
    for _ in range(repeats):
        run_start = perf_counter()
        errors = detect_unlikely_genotypes(dataset, workers=workers)
        run_seconds.append(perf_counter() - run_start)
        error_count = len(errors)

    return {
        "workload": "error",
        "families": len(dataset.families),
        "individuals": sum(
            len(family.individuals) for family in dataset.families
        ),
        "errors": error_count,
        "workers": workers,
        "load_seconds": load_seconds,
        "best_run_seconds": min(run_seconds),
        "mean_run_seconds": float(np.mean(run_seconds)),
    }


def benchmark_tree_multipoint(
    ped_path: str,
    dat_path: str,
    map_path: str,
    freq_path: str | None = None,
    marker_limit: int = 5,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
    heartbeat_node_interval: int | None = 10_000,
    emission_node_limit: int | None = None,
    emission_time_limit_seconds: float | None = None,
) -> dict[str, float | int | str]:
    """Profile exact tree phases on the largest pedigree in one dataset.

    The largest-meiosis family is the most informative scaling diagnostic for
    a MERLIN-style inheritance tree. Timings are intentionally separated so a
    slow emission reduction is not mistaken for a transition or worker issue.
    """

    if marker_limit < 1:
        raise ValueError("marker_limit must be at least 1.")
    workers = validate_workers(workers)
    load_start = perf_counter()
    dataset = load_merlin_inputs(ped_path, dat_path, map_path, freq_path)
    load_seconds = perf_counter() - load_start
    if progress is not None:
        progress(f"load complete\t{load_seconds:.6f} seconds")
    if not dataset.families:
        raise ValueError("A tree multipoint benchmark requires one family.")

    family = max(dataset.families, key=lambda item: len(item.meioses))
    markers = tuple(dataset.markers[:marker_limit])
    if not markers:
        raise ValueError("A tree multipoint benchmark requires one marker.")
    recombination_fractions = tuple(
        haldane_recombination_fraction(
            map_distance_cm(
                float(left_marker.position_cm),
                float(right_marker.position_cm),
            )
        )
        for left_marker, right_marker in zip(markers, markers[1:])
    )
    relevant_individual_ids_by_marker = tuple(
        _marker_relevant_individual_ids(family, marker.name)
        for marker in markers
    )
    relevant_individual_counts = tuple(
        len(individual_ids)
        for individual_ids in relevant_individual_ids_by_marker
    )
    relevant_meiosis_counts = tuple(
        len(
            _marker_relevant_meiosis_indices(family, individual_ids)
        )
        for individual_ids in relevant_individual_ids_by_marker
    )
    counting_chains = detect_untyped_chains(family)
    if progress is not None:
        progress(
            "emission plan\t"
            f"family={family.family_id}\t"
            f"markers={len(markers)}\tworkers={workers}\t"
            f"people={len(family.individuals)}\t"
            f"meioses={len(family.meioses)}\t"
            "relevant_people="
            f"{min(relevant_individual_counts)}-"
            f"{max(relevant_individual_counts)}\t"
            "relevant_meioses="
            f"{min(relevant_meiosis_counts)}-"
            f"{max(relevant_meiosis_counts)}\t"
            f"counting_chains={len(counting_chains)}"
        )

    emission_start = perf_counter()
    emission_trees = _family_marker_likelihood_trees(
        family,
        markers,
        workers,
        progress=(
            None
            if progress is None
            else lambda completed, total: progress(
                f"emissions complete\t{completed}/{total} markers"
            )
        ),
        diagnostic_progress=progress,
        heartbeat_node_interval=heartbeat_node_interval,
        emission_node_limit=emission_node_limit,
        emission_time_limit_seconds=emission_time_limit_seconds,
    )
    emission_seconds = perf_counter() - emission_start
    if progress is not None:
        progress(f"emission phase complete\t{emission_seconds:.6f} seconds")
    build_statistics = tuple(
        tree.build_statistics
        for tree in emission_trees
        if tree.build_statistics is not None
    )

    propagation_start = perf_counter()
    forward_trees, backward_trees = _tree_forward_backward_trees(
        family,
        markers,
        recombination_fractions,
        emission_trees=emission_trees,
        workers=workers,
    )
    propagation_seconds = perf_counter() - propagation_start
    if progress is not None:
        progress(
            "forward-backward phase complete\t"
            f"{propagation_seconds:.6f} seconds"
        )

    posterior_start = perf_counter()
    posterior_trees = tuple(
        forward_tree.pointwise_multiply(backward_tree).normalize()
        for forward_tree, backward_tree in zip(
            forward_trees,
            backward_trees,
        )
    )
    posterior_seconds = perf_counter() - posterior_start
    if progress is not None:
        progress(f"posterior phase complete\t{posterior_seconds:.6f} seconds")

    return {
        "workload": "tree-multipoint",
        "family": family.family_id,
        "individuals": len(family.individuals),
        "meioses": len(family.meioses),
        "markers": len(markers),
        "workers": workers,
        "relevant_meioses_min": min(relevant_meiosis_counts),
        "relevant_meioses_max": max(relevant_meiosis_counts),
        "relevant_individuals_min": min(relevant_individual_counts),
        "relevant_individuals_max": max(relevant_individual_counts),
        "counting_chains": len(counting_chains),
        "counted_selectors": sum(
            chain.selector_count for chain in counting_chains
        ),
        "suffix_cache_hits": sum(
            statistics.suffix_cache_hits
            for statistics in build_statistics
        ),
        "suffix_cache_misses": sum(
            statistics.suffix_cache_misses
            for statistics in build_statistics
        ),
        "cached_suffix_states": sum(
            statistics.cached_suffix_count
            for statistics in build_statistics
        ),
        "recursive_nodes": sum(
            statistics.recursive_node_count
            for statistics in build_statistics
        ),
        "maximum_recursion_depth": max(
            statistics.maximum_recursion_depth
            for statistics in build_statistics
        ),
        "contradiction_prunes": sum(
            statistics.contradiction_prune_count
            for statistics in build_statistics
        ),
        "founder_orientation_reductions": sum(
            statistics.founder_orientation_reduction_count
            for statistics in build_statistics
        ),
        "founder_couple_reductions": sum(
            statistics.founder_couple_reduction_count
            for statistics in build_statistics
        ),
        "counting_reductions": sum(
            statistics.counting_reduction_count
            for statistics in build_statistics
        ),
        "invariant_reductions": sum(
            statistics.invariant_reduction_count
            for statistics in build_statistics
        ),
        "peeled_components": sum(
            statistics.peeled_component_count
            for statistics in build_statistics
        ),
        "peeled_constraints": sum(
            statistics.peeled_constraint_count
            for statistics in build_statistics
        ),
        "zero_peeled_factors": sum(
            statistics.zero_peeled_factor_count
            for statistics in build_statistics
        ),
        "normalized_cache_reuses": sum(
            statistics.normalized_cache_reuse_count
            for statistics in build_statistics
        ),
        "peeled_factor_cache_hits": sum(
            statistics.peeled_factor_cache_hit_count
            for statistics in build_statistics
        ),
        "peeled_factor_cache_misses": sum(
            statistics.peeled_factor_cache_miss_count
            for statistics in build_statistics
        ),
        "scaled_tree_cache_hits": sum(
            statistics.scaled_tree_cache_hit_count
            for statistics in build_statistics
        ),
        "emission_unique_nodes_max": max(
            tree.unique_node_count() for tree in emission_trees
        ),
        "posterior_unique_nodes_max": max(
            tree.unique_node_count() for tree in posterior_trees
        ),
        "load_seconds": load_seconds,
        "emission_seconds": emission_seconds,
        "forward_backward_seconds": propagation_seconds,
        "posterior_seconds": posterior_seconds,
        "total_compute_seconds": (
            emission_seconds + propagation_seconds + posterior_seconds
        ),
    }


def repeat_families(dataset: Dataset, copies: int) -> Dataset:
    """Create a synthetic larger dataset by duplicating independent families."""

    if copies < 1:
        raise ValueError("copies must be at least 1.")

    families: list[Family] = []
    for copy_index in range(copies):
        for family in dataset.families:
            new_family_id = f"{family.family_id}_copy{copy_index + 1}"
            individuals = tuple(
                replace(person, family_id=new_family_id)
                for person in family.individuals
            )
            families.append(Family(new_family_id, individuals, family.meioses))
    return Dataset(dataset.markers, tuple(families), dataset.affection_names)


def _validate_benchmark_arguments(repeats: int, family_copies: int) -> None:
    """Reject empty timings and invalid synthetic scaling factors."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if family_copies < 1:
        raise ValueError("family_copies must be at least 1.")
