from pathlib import Path

import pytest

from pymerlin import load_merlin_inputs, multipoint_ibd
from pymerlin.benchmark import repeat_families
from pymerlin.map import haldane_recombination_fraction, map_distance_cm
from pymerlin.models import Dataset
from pymerlin.multipoint import two_marker_multipoint_ibd


MerlinIbdRow = tuple[str, str, str, str, float, float, float]


def test_haldane_recombination_fraction_uses_centimorgans():
    assert haldane_recombination_fraction(0.0) == pytest.approx(0.0)
    assert haldane_recombination_fraction(50.0) == pytest.approx(0.31606027941427883)


def test_map_distance_preserves_decimal_coordinates() -> None:
    assert map_distance_cm(123.4, 129.8) == 6.4
    assert map_distance_cm(129.8, 136.2) == 6.4


def test_two_marker_multipoint_basic2_matches_merlin_default_ibd_subset():
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    left, right = two_marker_multipoint_ibd(dataset, "some_marker", "another_marker")
    right_pair = next(row for row in right.rows if row["family_id"] == "1" and row["id1"] == "5" and row["id2"] == "6")

    assert right_pair["z0"] == pytest.approx(0.37806, abs=1e-5)
    assert right_pair["z1"] == pytest.approx(0.47361, abs=1e-5)
    assert right_pair["z2"] == pytest.approx(0.14833, abs=1e-5)


def test_two_marker_multipoint_basic2_matches_all_merlin_default_ibd_rows():
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    results = {
        result.marker_name: result
        for result in two_marker_multipoint_ibd(dataset, "some_marker", "another_marker")
    }

    expected = _read_merlin_fixture(
        "tests/fixtures/merlin/basic2_multipoint_freq.ibd",
        {"123.400": "some_marker", "136.200": "another_marker"},
    )
    for marker_name, family_id, id1, id2, expected_z0, expected_z1, expected_z2 in expected:
        if id1 == id2:
            continue
        result = results[marker_name]
        row = next(
            row
            for row in result.rows
            if row["family_id"] == family_id
            and {row["id1"], row["id2"]} == {id1, id2}
        )
        assert row["z0"] == pytest.approx(expected_z0, abs=1e-5)
        assert row["z1"] == pytest.approx(expected_z1, abs=1e-5)
        assert row["z2"] == pytest.approx(expected_z2, abs=1e-5)


def test_multipoint_normalizes_each_family_independently() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    repeated_dataset = repeat_families(dataset, copies=2)

    results = multipoint_ibd(repeated_dataset)

    for result in results:
        for row in result.rows:
            assert row["z0"] + row["z1"] + row["z2"] == pytest.approx(1.0)


def test_multipoint_parallel_result_is_identical_to_serial() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    assert multipoint_ibd(dataset, workers=2) == multipoint_ibd(
        dataset,
        workers=1,
    )


def test_three_marker_multipoint_matches_all_merlin_rows() -> None:
    dataset = load_merlin_inputs(
        "examples/sibs.ped",
        "examples/sibs.dat",
        "examples/sibs.map",
    )
    family = next(family for family in dataset.families if family.family_id == "1")
    family_dataset = Dataset(
        dataset.markers,
        (family,),
        dataset.affection_names,
    )
    results = {
        result.marker_name: result for result in multipoint_ibd(family_dataset)
    }

    expected = _read_merlin_fixture(
        "tests/fixtures/merlin/sibs_three_marker_multipoint.ibd",
        {"0.000": "SNP_1", "0.100": "SNP_2", "0.200": "SNP_3"},
    )
    for marker_name, family_id, id1, id2, expected_z0, expected_z1, expected_z2 in expected:
        if id1 == id2:
            continue
        result = results[marker_name]
        row = next(
            row
            for row in result.rows
            if row["family_id"] == family_id
            and {row["id1"], row["id2"]} == {id1, id2}
        )
        assert row["z0"] == pytest.approx(expected_z0, abs=1e-5)
        assert row["z1"] == pytest.approx(expected_z1, abs=1e-5)
        assert row["z2"] == pytest.approx(expected_z2, abs=1e-5)


def _read_merlin_fixture(
    path: str,
    position_to_marker: dict[str, str],
) -> list[MerlinIbdRow]:
    rows: list[MerlinIbdRow] = []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith("FAMILY"):
                continue
            family_id, id1, id2, marker, z0, z1, z2 = line.split()
            rows.append(
                (
                    position_to_marker.get(marker, marker),
                    family_id,
                    id1,
                    id2,
                    float(z0),
                    float(z1),
                    float(z2),
                )
            )
    return rows
