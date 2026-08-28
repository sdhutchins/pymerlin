import math
from itertools import product

import pytest
from gmpy2 import mpfr

import pymerlin.ibd as ibd_module
import pymerlin.npl as npl_module
from pymerlin import (
    Dataset,
    Family,
    Individual,
    Marker,
    Meiosis,
    family_marker_likelihood_tree,
    inheritance_origins,
    npl_pairs_score,
    single_marker_likelihood,
)
from pymerlin.founder_symmetry import (
    _select_tree_by_bit,
    _swap_tree_bit_inputs,
    _toggle_tree_bit_input,
    build_founder_couple_symmetry_plan,
    build_founder_orientation_symmetry_plan,
)
from pymerlin.inheritance_tree import InheritanceTree
from pymerlin.ibd import _pair_ibd_indicator_trees, _shared_allele_count
from pymerlin.npl import _npl_pairs_score_tree
from tests.oracles.mpfr_multipoint import mpfr_marker_state_likelihoods


def test_founder_symmetry_plan_tracks_later_founder_transmissions() -> None:
    marker, family = _multigenerational_founder_family()
    del marker

    plan = build_founder_orientation_symmetry_plan(
        family,
        frozenset(range(len(family.meioses))),
    )

    assert plan.descendant_flip_indices(0) == frozenset({2})
    assert plan.descendant_flip_indices(1) == frozenset({3})
    assert plan.descendant_flip_indices(7) == frozenset()
    assert all(
        plan.descendant_flip_indices(bit_index) is None
        for bit_index in (2, 3, 4, 5, 6)
    )

    couple_plan = build_founder_couple_symmetry_plan(
        family,
        frozenset(range(len(family.meioses))),
    )
    assert len(couple_plan.symmetries) == 1
    couple_symmetry = couple_plan.symmetries[0]
    assert couple_symmetry.founder_ids == ("1", "2")
    assert couple_symmetry.shared_child_ids == ("3", "4")
    assert couple_symmetry.representative_bit_index == 4
    assert couple_symmetry.swapped_bit_pairs == ((0, 1), (2, 3))
    assert couple_symmetry.toggled_bit_indices == frozenset({4, 5})


def test_founder_symmetry_preserves_every_marker_likelihood_state() -> None:
    marker, family = _multigenerational_founder_family()
    dataset = Dataset(markers=(marker,), families=(family,))
    explicit_result = single_marker_likelihood(dataset, marker.name)
    expected_likelihoods = {
        state.bits: state.likelihood for state in explicit_result.states
    }
    oracle_likelihoods = mpfr_marker_state_likelihoods(family, marker)

    tree = family_marker_likelihood_tree(family, marker)

    failures: list[str] = []
    for bits in product((0, 1), repeat=len(family.meioses)):
        tree_value = tree.value_at(bits)
        explicit_value = expected_likelihoods.get(bits, 0.0)
        oracle_value = oracle_likelihoods.get(bits, mpfr(0))
        tree_error = abs(mpfr(tree_value) - oracle_value)
        explicit_error = abs(mpfr(explicit_value) - oracle_value)
        representation_unit = mpfr(math.ulp(float(oracle_value)))
        if tree_error > max(explicit_error, representation_unit):
            failures.append(
                f"Founder symmetry is less accurate for {bits=}: "
                f"{tree_error=} > {explicit_error=}, "
                f"{representation_unit=}"
            )

    assert not failures, "\n".join(failures)


def test_founder_symmetry_preserves_every_ibd_and_npl_state() -> None:
    marker, family = _multigenerational_founder_family()
    del marker
    first_id = "5"
    second_id = "6"
    affected_ids = (first_id, second_id)
    indicator_trees = _pair_ibd_indicator_trees(
        family,
        first_id,
        second_id,
    )
    labeled_founder_indicator_trees = _pair_ibd_indicator_trees(
        family,
        "1",
        second_id,
    )
    score_tree = _npl_pairs_score_tree(family, affected_ids)

    for bits in product((0, 1), repeat=len(family.meioses)):
        origins = inheritance_origins(family, bits)
        expected_ibd_state = _shared_allele_count(
            origins[first_id],
            origins[second_id],
        )
        expected_labeled_founder_ibd_state = _shared_allele_count(
            origins["1"],
            origins[second_id],
        )

        assert tuple(
            tree.value_at(bits) for tree in indicator_trees
        ) == tuple(
            float(ibd_state == expected_ibd_state)
            for ibd_state in range(3)
        )
        assert tuple(
            tree.value_at(bits)
            for tree in labeled_founder_indicator_trees
        ) == tuple(
            float(ibd_state == expected_labeled_founder_ibd_state)
            for ibd_state in range(3)
        )
        assert score_tree.value_at(bits) == npl_pairs_score(
            origins,
            affected_ids,
        )


def test_founder_symmetry_scores_one_representative_per_symmetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker, family = _multigenerational_founder_family()
    del marker
    original_ibd_origins = ibd_module.inheritance_origins
    original_npl_origins = npl_module.inheritance_origins
    ibd_leaf_count = 0
    npl_leaf_count = 0

    def count_ibd_origins(
        evaluated_family: Family,
        bits: tuple[int, ...],
    ) -> dict[str, tuple[tuple[str, int], tuple[str, int]]]:
        nonlocal ibd_leaf_count
        ibd_leaf_count += 1
        return original_ibd_origins(evaluated_family, bits)

    def count_npl_origins(
        evaluated_family: Family,
        bits: tuple[int, ...],
    ) -> dict[str, tuple[tuple[str, int], tuple[str, int]]]:
        nonlocal npl_leaf_count
        npl_leaf_count += 1
        return original_npl_origins(evaluated_family, bits)

    monkeypatch.setattr(ibd_module, "inheritance_origins", count_ibd_origins)
    monkeypatch.setattr(npl_module, "inheritance_origins", count_npl_origins)

    _pair_ibd_indicator_trees(family, "5", "6")
    _npl_pairs_score_tree(family, ("5", "6"))

    canonical_leaf_count = 2 ** (
        len(family.meioses) - len(family.founders) - 1
    )
    assert ibd_leaf_count == canonical_leaf_count
    assert npl_leaf_count == canonical_leaf_count


def test_founder_couple_tree_transformations_preserve_bit_coordinates() -> None:
    source_tree = InheritanceTree.from_dense(tuple(float(i) for i in range(8)))

    swapped_tree = InheritanceTree(
        bit_count=3,
        root=_swap_tree_bit_inputs(source_tree.root, 0, 2),
    )
    toggled_tree = InheritanceTree(
        bit_count=3,
        root=_toggle_tree_bit_input(source_tree.root, 1),
    )
    zero_tree = InheritanceTree.from_dense(
        tuple(float(10 * bits[0] + bits[2]) for bits in _three_bit_states())
    )
    one_tree = InheritanceTree.from_dense(
        tuple(
            float(100 + 10 * bits[0] + bits[2])
            for bits in _three_bit_states()
        )
    )
    selected_tree = InheritanceTree(
        bit_count=3,
        root=_select_tree_by_bit(zero_tree.root, one_tree.root, 1),
    )

    for bits in _three_bit_states():
        assert swapped_tree.value_at(bits) == source_tree.value_at(
            (bits[2], bits[1], bits[0])
        )
        assert toggled_tree.value_at(bits) == source_tree.value_at(
            (bits[0], 1 - bits[1], bits[2])
        )
        expected_selected_value = (
            zero_tree.value_at(bits)
            if bits[1] == 0
            else one_tree.value_at(bits)
        )
        assert selected_tree.value_at(bits) == expected_selected_value


def _three_bit_states() -> tuple[tuple[int, ...], ...]:
    return tuple(product((0, 1), repeat=3))


def _multigenerational_founder_family() -> tuple[Marker, Family]:
    marker = Marker(
        name="founder_symmetry_marker",
        chromosome="1",
        position_cm=0.0,
        allele_frequencies={"1": 0.4, "2": 0.6},
    )
    observed_genotypes = {
        "3": ("1", "1"),
        "4": ("1", "2"),
        "6": ("2", "2"),
    }

    def person(
        individual_id: str,
        father_id: str | None,
        mother_id: str | None,
        sex: str,
    ) -> Individual:
        return Individual(
            family_id="1",
            individual_id=individual_id,
            father_id=father_id,
            mother_id=mother_id,
            sex=sex,
            phenotypes={},
            genotypes={
                marker.name: observed_genotypes.get(
                    individual_id,
                    (None, None),
                )
            },
        )

    family = Family(
        family_id="1",
        individuals=(
            person("1", None, None, "1"),
            person("2", None, None, "2"),
            person("3", "1", "2", "1"),
            person("4", "1", "2", "2"),
            person("5", "3", "4", "1"),
            person("7", None, None, "2"),
            person("6", "5", "7", "1"),
        ),
        meioses=(
            Meiosis(parent_id="1", child_id="3", parent_sex="1"),
            Meiosis(parent_id="2", child_id="3", parent_sex="2"),
            Meiosis(parent_id="1", child_id="4", parent_sex="1"),
            Meiosis(parent_id="2", child_id="4", parent_sex="2"),
            Meiosis(parent_id="3", child_id="5", parent_sex="1"),
            Meiosis(parent_id="4", child_id="5", parent_sex="2"),
            Meiosis(parent_id="5", child_id="6", parent_sex="1"),
            Meiosis(parent_id="7", child_id="6", parent_sex="2"),
        ),
    )
    return marker, family
