"""Regression tests for exact marker-specific coordinate reductions."""

from itertools import product

import pytest

from pymerlin.coordinate_reduction import (
    CoordinateTransitionBudgetExceeded,
    exact_partial_transition,
    marker_coordinate_map,
    reduce_inheritance_tree,
    reference_state_pair_count,
)
from pymerlin.inheritance_tree import InheritanceTree


def test_marker_coordinate_map_finds_fixed_and_pair_parity_bits() -> None:
    full_vectors = tuple(product((0, 1), repeat=3))
    values = tuple(
        1.0 if bits[0] == 0 and bits[1] != bits[2] else 0.0 for bits in full_vectors
    )
    tree = InheritanceTree.from_dense(values)

    coordinate_map = marker_coordinate_map(tree)

    assert coordinate_map.full_bit_count == 3
    assert coordinate_map.fixed_bit_count == 1
    assert coordinate_map.parity_reduced_bit_count == 1
    assert coordinate_map.reduced_bit_count == 1
    assert coordinate_map.coordinate_by_bit[0] is None
    assert coordinate_map.xor_offset_by_bit[0] == 0

    for full_bits, value in zip(full_vectors, values):
        assert coordinate_map.is_compatible(full_bits) is (value != 0.0)
        if value != 0.0:
            reduced_bits = coordinate_map.project(full_bits)
            assert coordinate_map.expand(reduced_bits) == full_bits


def test_higher_order_affine_constraint_remains_as_zero_states() -> None:
    full_vectors = tuple(product((0, 1), repeat=3))
    values = tuple(
        2.0 if bits[2] == bits[0] ^ bits[1] else 0.0 for bits in full_vectors
    )
    tree = InheritanceTree.from_dense(values)

    coordinate_map = marker_coordinate_map(tree)
    reduced_message = reduce_inheritance_tree(tree, coordinate_map)

    assert coordinate_map.fixed_bit_count == 0
    assert coordinate_map.parity_reduced_bit_count == 0
    assert coordinate_map.reduced_bit_count == 3
    assert reduced_message.tree.dense_values() == values


def test_empty_support_reduces_to_one_zero_state() -> None:
    tree = InheritanceTree.from_dense((0.0,) * 8)

    coordinate_map = marker_coordinate_map(tree)
    reduced_message = reduce_inheritance_tree(tree, coordinate_map)

    assert coordinate_map.fixed_bit_count == 3
    assert coordinate_map.reduced_bit_count == 0
    assert reduced_message.tree.dense_values() == (0.0,)


@pytest.mark.parametrize("recombination_fraction", [0.0, 0.2, 0.5])
def test_exact_partial_transition_matches_full_tree_transition(
    recombination_fraction: float,
) -> None:
    full_vectors = tuple(product((0, 1), repeat=4))
    current_values = tuple(
        float(index + 1) if bits[0] == 0 and bits[1] != bits[2] else 0.0
        for index, bits in enumerate(full_vectors)
    )
    next_values = tuple(
        float(2 * index + 1) if bits[2] == 1 and bits[0] == bits[3] else 0.0
        for index, bits in enumerate(full_vectors)
    )
    current_tree = InheritanceTree.from_dense(current_values)
    next_tree = InheritanceTree.from_dense(next_values)
    current_map = marker_coordinate_map(current_tree)
    next_map = marker_coordinate_map(next_tree)
    current_message = reduce_inheritance_tree(current_tree, current_map)

    reduced_result = exact_partial_transition(
        current_message,
        next_tree,
        next_map,
        recombination_fraction,
    )
    expected_tree = current_tree.transition(recombination_fraction).pointwise_multiply(
        next_tree
    )

    for full_bits in full_vectors:
        expected_value = expected_tree.value_at(full_bits)
        if next_map.is_compatible(full_bits):
            reduced_bits = next_map.project(full_bits)
            actual_value = reduced_result.tree.value_at(reduced_bits)
        else:
            actual_value = 0.0
        assert actual_value == pytest.approx(expected_value, rel=1e-12)


def test_reference_state_pair_count_uses_both_reduced_spaces() -> None:
    unrestricted_tree = InheritanceTree.from_dense(tuple(range(1, 17)))
    restricted_tree = InheritanceTree.from_dense(
        tuple(
            1.0 if bits[0] == 0 and bits[1] == bits[2] else 0.0
            for bits in product((0, 1), repeat=4)
        )
    )

    unrestricted_map = marker_coordinate_map(unrestricted_tree)
    restricted_map = marker_coordinate_map(restricted_tree)

    assert unrestricted_map.reduced_bit_count == 4
    assert restricted_map.reduced_bit_count == 2
    assert reference_state_pair_count(unrestricted_map, restricted_map) == 64


def test_dense_reference_operations_enforce_explicit_budgets() -> None:
    tree = InheritanceTree.from_dense(tuple(range(1, 17)))
    coordinate_map = marker_coordinate_map(tree)

    with pytest.raises(CoordinateTransitionBudgetExceeded, match="16 states"):
        reduce_inheritance_tree(tree, coordinate_map, state_limit=15)

    current_message = reduce_inheritance_tree(tree, coordinate_map)
    with pytest.raises(
        CoordinateTransitionBudgetExceeded,
        match="256 source-target pairs",
    ):
        exact_partial_transition(
            current_message,
            tree,
            coordinate_map,
            recombination_fraction=0.1,
            state_pair_limit=255,
        )
