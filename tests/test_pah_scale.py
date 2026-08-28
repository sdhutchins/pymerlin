import math
from pathlib import Path
from time import perf_counter

import pytest

from pymerlin import (
    analyze_pedigree_reduction,
    exponential_kong_cox,
    linear_kong_cox,
    load_merlin_inputs,
    multipoint_information_content,
    multipoint_npl_pairs,
    multipoint_tree_posteriors_at_positions,
)
from pymerlin.benchmark import benchmark_tree_multipoint
from pymerlin.positions import merlin_analysis_positions
from tests.pah_scale_fixture import (
    build_pah_marker_benchmark_inputs,
    build_pah_parity_inputs,
    build_pah_scale_inputs,
)

pytestmark = pytest.mark.pah_scale


def test_pah_scale_fixture_preserves_target_structure(tmp_path: Path) -> None:
    """Require the anonymized fixture to represent the intended PAH scale."""

    input_paths = build_pah_scale_inputs(tmp_path)
    dataset = load_merlin_inputs(
        input_paths.ped,
        input_paths.dat,
        input_paths.map,
        input_paths.freq,
    )
    family = dataset.families[0]
    typed_people = [
        person
        for person in family.individuals
        if all(
            allele is not None
            for genotype in person.genotypes.values()
            for allele in genotype
        )
    ]
    affected_people = [
        person for person in family.individuals if person.phenotypes["HPAH"] == "2"
    ]

    assert len(dataset.families) == 1
    assert len(family.individuals) == 912
    assert len(typed_people) == 23
    assert len(affected_people) == 37
    assert len(family.meioses) == 1_336
    assert len(dataset.markers) == 3


def test_pah_reduction_report_exposes_information_retention(
    tmp_path: Path,
) -> None:
    """Separate computationally bounded pairs from marker-data retention."""

    input_paths = build_pah_scale_inputs(tmp_path)
    dataset = load_merlin_inputs(
        input_paths.ped,
        input_paths.dat,
        input_paths.map,
        input_paths.freq,
    )

    report = analyze_pedigree_reduction(dataset, "PAH")

    assert report.full_individual_count == 912
    assert report.full_meiosis_count == 1_336
    assert report.full_merlin_bit_count == 1_091
    assert report.full_marker_relevant_meiosis_count == 82
    assert report.full_typed_person_count == 23
    assert report.full_affected_person_count == 37
    assert len(report.candidates) == 666
    assert all(candidate.within_bit_limit for candidate in report.candidates)
    assert all(
        candidate.connected_component_count == 1 for candidate in report.candidates
    )
    assert len(report.review_candidates) == 153
    assert max(candidate.typed_person_count for candidate in report.candidates) == 6


def test_pah_parity_fixture_is_bounded_and_informative(
    tmp_path: Path,
) -> None:
    """Require a representative PAH branch that external MERLIN can run."""

    input_paths = build_pah_parity_inputs(tmp_path)
    dataset = load_merlin_inputs(
        input_paths.ped,
        input_paths.dat,
        input_paths.map,
        input_paths.freq,
    )
    family = dataset.families[0]
    typed_people = [
        person
        for person in family.individuals
        if all(
            allele is not None
            for genotype in person.genotypes.values()
            for allele in genotype
        )
    ]
    affected_people = [
        person for person in family.individuals if person.phenotypes["HPAH"] == "2"
    ]

    assert len(family.individuals) == 9
    assert len(typed_people) == 5
    assert len(affected_people) == 3
    assert len(family.meioses) == 10
    people_by_id = family.by_id
    assert people_by_id["P0803"].father_id == "P0793"
    assert people_by_id["P0803"].mother_id == "P0907"
    assert people_by_id["P0902"].father_id == "P0793"
    assert people_by_id["P0902"].mother_id == "P0907"

    positions = merlin_analysis_positions(dataset)
    npl_result = multipoint_npl_pairs(
        dataset,
        positions,
        workers=1,
        engine="tree",
    )
    linear_result = linear_kong_cox(npl_result)
    assert all(row.lod_score != 0.0 for row in linear_result.analyses[0].rows)


def test_pah_scale_tree_engine_completes_requested_analyses(
    tmp_path: Path,
) -> None:
    """Exercise the requested PAH analyses without dense state enumeration."""

    input_paths = build_pah_scale_inputs(tmp_path)
    dataset = load_merlin_inputs(
        input_paths.ped,
        input_paths.dat,
        input_paths.map,
        input_paths.freq,
    )
    positions = merlin_analysis_positions(dataset)
    tree_posteriors = multipoint_tree_posteriors_at_positions(
        dataset,
        positions,
        workers=1,
    )
    posterior_trees = tree_posteriors.families[0].trees
    full_state_count = 1 << len(dataset.families[0].meioses)

    assert posterior_trees
    assert all(tree.weighted_sum() > 0.0 for tree in posterior_trees)
    assert all(tree.node_count() < full_state_count for tree in posterior_trees)

    npl_result = multipoint_npl_pairs(
        dataset,
        positions,
        workers=1,
        engine="tree",
        tree_posteriors=tree_posteriors,
    )
    information_result = multipoint_information_content(
        dataset,
        positions,
        workers=1,
        engine="tree",
        tree_posteriors=tree_posteriors,
    )
    linear_result = linear_kong_cox(npl_result)
    exponential_result = exponential_kong_cox(npl_result)

    assert all(math.isfinite(value) for value in information_result.values)
    assert linear_result.analyses[0].informative_family_count == 1
    assert exponential_result.analyses[0].informative_family_count == 1


def test_pah_scale_four_workers_complete_5_marker_posteriors(
    tmp_path: Path,
) -> None:
    """Measure every phase of a four-worker PAH-scale marker workload."""

    input_paths = build_pah_marker_benchmark_inputs(
        tmp_path,
        marker_count=5,
    )
    summary = benchmark_tree_multipoint(
        str(input_paths.ped),
        str(input_paths.dat),
        str(input_paths.map),
        str(input_paths.freq),
        marker_limit=5,
        workers=4,
        progress=lambda message: print(message, flush=True),
    )

    print(
        "PAH five-marker phase benchmark:\n"
        + "\n".join(f"{key}\t{value}" for key, value in summary.items())
    )
    assert summary["markers"] == 5
    assert summary["workers"] == 4
    assert summary["meioses"] == 1_336
    assert summary["posterior_unique_nodes_max"] >= 1


def test_pah_scale_four_workers_complete_25_marker_posteriors(
    tmp_path: Path,
) -> None:
    """Measure four marker workers on one PAH-scale family."""

    _assert_pah_marker_benchmark_completes(
        tmp_path,
        marker_count=25,
        workers=4,
    )


def _assert_pah_marker_benchmark_completes(
    tmp_path: Path,
    marker_count: int,
    workers: int,
) -> None:
    """Require valid posterior trees for one PAH-scale marker benchmark."""

    input_paths = build_pah_marker_benchmark_inputs(
        tmp_path,
        marker_count=marker_count,
    )
    dataset = load_merlin_inputs(
        input_paths.ped,
        input_paths.dat,
        input_paths.map,
        input_paths.freq,
    )
    positions = merlin_analysis_positions(dataset)

    parallel_start = perf_counter()
    parallel_posteriors = multipoint_tree_posteriors_at_positions(
        dataset,
        positions,
        workers=workers,
    )
    parallel_seconds = perf_counter() - parallel_start
    posterior_trees = tuple(
        tree
        for family_posteriors in parallel_posteriors.families
        for tree in family_posteriors.trees
    )

    print(
        f"PAH {marker_count}-marker {workers}-worker benchmark: {parallel_seconds=:.6f}"
    )
    assert len(positions) == marker_count
    assert len(posterior_trees) == marker_count
    assert all(tree.bit_count == 1_336 for tree in posterior_trees)
    posterior_weights = tuple(tree.weighted_sum() for tree in posterior_trees)
    assert all(math.isfinite(weight) and weight > 0.0 for weight in posterior_weights)
