"""Tests for exact founder-orientation quotient coordinates."""

from itertools import product

import pytest

from pymerlin.founder_orientation_quotient import (
    build_founder_orientation_quotient,
    founder_group_transition_probability,
    reduce_founder_orientation_tree,
)
from pymerlin.inheritance_tree import InheritanceTree
from pymerlin.models import Family, Individual, Meiosis


def test_quotient_projects_founder_bits_relative_to_representative() -> None:
    family = _quotient_family()
    quotient = build_founder_orientation_quotient(family)

    assert quotient.full_bit_count == 5
    assert quotient.reduced_bit_count == 3
    assert quotient.reduced_bit_index_by_full_bit == (None, None, 0, 1, 2)
    assert tuple(
        (
            group.founder_id,
            group.member_full_bit_indices,
            group.reduced_member_bit_indices,
        )
        for group in quotient.groups
    ) == (
        ("1", (0, 2), (0,)),
        ("2", (1, 3), (1,)),
    )

    full_bits = (1, 0, 0, 1, 1)
    assert quotient.project(full_bits) == (1, 1, 1)
    assert quotient.expand_canonical((1, 1, 1)) == (0, 0, 1, 1, 1)
    assert quotient.project(quotient.expand_canonical((1, 0, 1))) == (1, 0, 1)


def test_tree_reduction_preserves_every_quotient_value() -> None:
    family = _quotient_family()
    quotient = build_founder_orientation_quotient(family)
    reduced_values = tuple(float(index + 1) for index in range(8))
    full_values = tuple(
        reduced_values[_binary_index(quotient.project(full_bits))]
        for full_bits in product((0, 1), repeat=quotient.full_bit_count)
    )
    full_tree = InheritanceTree.from_dense(full_values)

    reduced_tree = reduce_founder_orientation_tree(full_tree, quotient)

    assert reduced_tree.bit_count == quotient.reduced_bit_count
    assert reduced_tree.dense_values() == reduced_values


def test_founder_transition_matches_exhaustive_full_orbit_sum() -> None:
    theta = 0.13
    relative_vectors = tuple(product((0, 1), repeat=3))

    for previous_bits in relative_vectors:
        row_sum = 0.0
        for next_bits in relative_vectors:
            observed = founder_group_transition_probability(
                previous_bits,
                next_bits,
                theta,
            )
            canonical_source = (0, *previous_bits)
            full_targets = (
                (0, *next_bits),
                (1, *(1 - bit for bit in next_bits)),
            )
            expected = sum(
                _full_transition_probability(
                    canonical_source,
                    full_target,
                    theta,
                )
                for full_target in full_targets
            )
            assert observed == pytest.approx(expected)
            row_sum += observed
        assert row_sum == pytest.approx(1.0)


def test_two_transmissions_use_merlin_effective_recombination() -> None:
    theta = 0.2

    assert founder_group_transition_probability(
        (0,),
        (1,),
        theta,
    ) == pytest.approx(2.0 * theta * (1.0 - theta))


@pytest.mark.parametrize(
    ("previous_bits", "next_bits", "theta", "message"),
    [
        ((0,), (), 0.1, "same number"),
        ((2,), (0,), 0.1, "only zero and one"),
        ((0,), (0,), -0.1, "between 0 and 0.5"),
    ],
)
def test_founder_transition_rejects_invalid_inputs(
    previous_bits: tuple[int, ...],
    next_bits: tuple[int, ...],
    theta: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        founder_group_transition_probability(
            previous_bits,
            next_bits,
            theta,
        )


def test_transition_groups_retain_multi_relative_founder_scope() -> None:
    family = _three_child_founder_family()

    quotient = build_founder_orientation_quotient(family)

    assert quotient.transition_interaction_groups == ((0, 1), (2, 3))


def _quotient_family() -> Family:
    return Family(
        family_id="1",
        individuals=(
            _person("1", None, None),
            _person("2", None, None),
            _person("3", "1", "2"),
            _person("4", "1", "2"),
            _person("5", "3", None),
        ),
        meioses=(
            Meiosis("1", "3", "1"),
            Meiosis("2", "3", "2"),
            Meiosis("1", "4", "1"),
            Meiosis("2", "4", "2"),
            Meiosis("3", "5", "1"),
        ),
    )


def _three_child_founder_family() -> Family:
    return Family(
        family_id="1",
        individuals=(
            _person("1", None, None),
            _person("2", None, None),
            _person("3", "1", "2"),
            _person("4", "1", "2"),
            _person("5", "1", "2"),
        ),
        meioses=(
            Meiosis("1", "3", "1"),
            Meiosis("2", "3", "2"),
            Meiosis("1", "4", "1"),
            Meiosis("1", "5", "1"),
            Meiosis("2", "4", "2"),
            Meiosis("2", "5", "2"),
        ),
    )


def _person(
    individual_id: str,
    father_id: str | None,
    mother_id: str | None,
) -> Individual:
    return Individual(
        family_id="1",
        individual_id=individual_id,
        father_id=father_id,
        mother_id=mother_id,
        sex="1",
        phenotypes={},
        genotypes={},
    )


def _binary_index(bits: tuple[int, ...]) -> int:
    return sum(bit << (len(bits) - bit_index - 1) for bit_index, bit in enumerate(bits))


def _full_transition_probability(
    previous_bits: tuple[int, ...],
    next_bits: tuple[int, ...],
    theta: float,
) -> float:
    probability = 1.0
    for previous_bit, next_bit in zip(previous_bits, next_bits):
        probability *= 1.0 - theta if previous_bit == next_bit else theta
    return probability
