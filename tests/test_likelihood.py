import pytest

from pymerlin import (
    Dataset,
    Family,
    Individual,
    Marker,
    Meiosis,
    estimate_ibd,
    load_merlin_inputs,
    single_marker_likelihood,
)


def test_single_marker_likelihood_prunes_impossible_states():
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    result = single_marker_likelihood(dataset, "some_marker")

    assert result.likelihood > 0.0
    assert 0 < len(result.states) < 16
    assert sum(state.posterior_weight for state in result.states) == pytest.approx(1.0)


def test_single_marker_parallel_result_is_identical_to_serial() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    serial_result = single_marker_likelihood(
        dataset,
        "some_marker",
        workers=1,
    )
    parallel_result = single_marker_likelihood(
        dataset,
        "some_marker",
        workers=2,
    )

    assert parallel_result == serial_result


def test_mendelian_incompatibility_is_uninformative_for_one_family_marker() -> None:
    marker = Marker(
        name="inconsistent_marker",
        chromosome="1",
        position_cm=0.0,
        allele_frequencies={"1": 0.5, "2": 0.5},
    )
    family = Family(
        family_id="1",
        individuals=(
            Individual(
                family_id="1",
                individual_id="1",
                father_id=None,
                mother_id=None,
                sex="1",
                phenotypes={},
                genotypes={marker.name: (None, None)},
            ),
            Individual(
                family_id="1",
                individual_id="2",
                father_id=None,
                mother_id=None,
                sex="2",
                phenotypes={},
                genotypes={marker.name: ("1", "1")},
            ),
            Individual(
                family_id="1",
                individual_id="3",
                father_id="1",
                mother_id="2",
                sex="1",
                phenotypes={},
                genotypes={marker.name: ("2", "2")},
            ),
        ),
        meioses=(
            Meiosis(parent_id="1", child_id="3", parent_sex="1"),
            Meiosis(parent_id="2", child_id="3", parent_sex="2"),
        ),
    )
    dataset = Dataset(markers=(marker,), families=(family,))

    result = single_marker_likelihood(dataset, marker.name)

    assert {state.bits for state in result.states} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert all(state.likelihood == 1.0 for state in result.states)
    assert all(state.posterior_weight == 0.25 for state in result.states)
    assert result.likelihood == 4.0


def test_parent_child_ibd_is_one_shared_allele_in_informative_basic_family():
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    result = estimate_ibd(dataset, "some_marker")
    pair = next(row for row in result.rows if row["family_id"] == "1" and row["id1"] == "3" and row["id2"] == "5")

    assert pair["z0"] == pytest.approx(0.0)
    assert pair["z1"] == pytest.approx(1.0)
    assert pair["z2"] == pytest.approx(0.0)
    assert pair["kinship"] == pytest.approx(0.25)


def test_sibling_pair_has_valid_ibd_distribution():
    dataset = load_merlin_inputs("examples/sibs.ped", "examples/sibs.dat", "examples/sibs.map")

    result = estimate_ibd(dataset, "SNP_1")
    pair = next(row for row in result.rows if row["family_id"] == "1" and row["id1"] == "3" and row["id2"] == "4")

    assert pair["z0"] + pair["z1"] + pair["z2"] == pytest.approx(1.0)
    assert 0.0 <= pair["kinship"] <= 0.5
