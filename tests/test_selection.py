import pytest

from pymerlin import (
    Dataset,
    Marker,
    estimate_ibd_for_markers,
    load_merlin_inputs,
    partition_dataset_by_chromosome,
)
from pymerlin.selection import select_markers


def test_select_markers_by_map_interval():
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    selected = select_markers(dataset, chromosome="24", start_cm=130.0, end_cm=140.0)

    assert [marker.name for marker in selected] == ["another_marker"]


def test_estimate_ibd_for_multiple_markers():
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    results = estimate_ibd_for_markers(dataset, marker_names=["some_marker", "another_marker"])

    assert [result.marker_name for result in results] == ["some_marker", "another_marker"]
    assert all(len(result.rows) == 15 for result in results)


def test_select_markers_rejects_unknown_marker():
    dataset = load_merlin_inputs("examples/basic2.ped", "examples/basic2.dat")

    with pytest.raises(ValueError, match="Unknown marker"):
        select_markers(dataset, marker_names=["missing_marker"])


def test_partitions_markers_in_merlin_chromosome_and_position_order() -> None:
    dataset = Dataset(
        markers=(
            Marker("chr10_marker", "10", 3.0),
            Marker("chr2_right", "2", 20.0),
            Marker("chr1_marker", "1", 10.0),
            Marker("chr2_left", "2", 5.0),
            Marker("x_marker", "X", 1.0),
        ),
        families=(),
    )

    chromosome_datasets = partition_dataset_by_chromosome(dataset)

    assert [
        chromosome_dataset.markers[0].chromosome
        for chromosome_dataset in chromosome_datasets
    ] == ["1", "2", "10", "X"]
    assert [
        marker.name for marker in chromosome_datasets[1].markers
    ] == [
        "chr2_left",
        "chr2_right",
    ]
