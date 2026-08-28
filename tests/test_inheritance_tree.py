import math
import pickle
import sys
from itertools import product

import pytest

from pymerlin import (
    InheritanceTree,
    LeafNode,
    SharedNode,
    SplitNode,
    ZeroNode,
)


def test_uniform_dense_values_collapse_to_one_leaf() -> None:
    tree = InheritanceTree.from_dense([0.25] * 8)

    assert tree.bit_count == 3
    assert tree.root == LeafNode(0.25)
    assert tree.node_count() == 1
    assert tree.dense_values() == (0.25,) * 8


def test_bit_invariant_dense_values_use_a_shared_branch() -> None:
    tree = InheritanceTree.from_dense([0.0, 1.0, 0.0, 1.0])

    assert tree.root == SharedNode(
        SplitNode(
            zero_child=ZeroNode(),
            one_child=LeafNode(1.0),
        )
    )
    assert tree.node_count() == 4


def test_sparse_tree_preserves_every_inheritance_vector_value() -> None:
    dense_values = (0.0, 0.0, 0.0, 0.0, 0.0, 3.5, 0.0, 0.0)
    tree = InheritanceTree.from_dense(dense_values)

    observed = tuple(
        tree.value_at(bits)
        for bits in product((0, 1), repeat=tree.bit_count)
    )

    assert observed == dense_values
    assert tree.node_count() < 2 ** (tree.bit_count + 1) - 1


def test_weighted_sum_matches_merlin_uniform_branch_weighting() -> None:
    tree = InheritanceTree.from_dense([0.0, 1.0, 2.0, 3.0])

    assert tree.weighted_sum() == 1.5


def test_pointwise_multiply_and_mean_product_match_dense_values() -> None:
    first = InheritanceTree.from_dense([0.0, 1.0, 2.0, 3.0])
    second = InheritanceTree.from_dense([4.0, 5.0, 0.5, 2.0])
    expected_products = (0.0, 5.0, 1.0, 6.0)

    multiplied = first.pointwise_multiply(second)

    assert multiplied.dense_values() == expected_products
    assert first.mean_product(second) == math.fsum(expected_products) / 4.0
    assert multiplied.weighted_sum() == first.mean_product(second)


def test_pointwise_multiply_preserves_a_compressed_zero_tree() -> None:
    zero = InheritanceTree(bit_count=4, root=ZeroNode())
    other = InheritanceTree.from_dense([1.0] * 16)

    multiplied = zero.pointwise_multiply(other)

    assert multiplied.root == ZeroNode()
    assert multiplied.node_count() == 1


def test_compact_pickle_preserves_values_and_shared_node_identity() -> None:
    """Require worker serialization to retain the exact immutable DAG."""

    shared_child = LeafNode(0.375)
    source_tree = InheritanceTree(
        bit_count=1,
        root=SplitNode(shared_child, shared_child),
    )

    restored_tree = pickle.loads(pickle.dumps(source_tree))

    assert restored_tree.dense_values() == source_tree.dense_values()
    assert isinstance(restored_tree.root, SplitNode)
    assert restored_tree.root.zero_child is restored_tree.root.one_child
    assert restored_tree.unique_node_count() == 2


def test_internal_tree_operations_do_not_repeat_full_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require depth-preserving operations to use trusted construction."""

    first = InheritanceTree.from_dense((1.0, 2.0, 3.0, 4.0))
    second = InheritanceTree.from_dense((0.5, 1.5, 2.5, 3.5))

    def unexpected_validation(*args: object) -> None:
        raise AssertionError("Internal tree operation repeated validation.")

    monkeypatch.setattr(
        "pymerlin.inheritance_tree._validate_node_depth",
        unexpected_validation,
    )

    multiplied = first.pointwise_multiply(second)
    transitioned = multiplied.transition(0.1)
    normalized = transitioned.normalize()

    assert normalized.weighted_sum() == pytest.approx(1.0)


def test_transition_matches_exhaustive_recombination_weights() -> None:
    values = (0.0, 1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 8.0)
    tree = InheritanceTree.from_dense(values)
    theta = 0.25
    inheritance_vectors = tuple(product((0, 1), repeat=tree.bit_count))
    expected = tuple(
        math.fsum(
            source_value
            * math.prod(
                theta if source_bit != target_bit else 1.0 - theta
                for source_bit, target_bit in zip(source_bits, target_bits)
            )
            for source_bits, source_value in zip(
                inheritance_vectors,
                values,
            )
        )
        for target_bits in inheritance_vectors
    )

    transitioned = tree.transition(theta)

    assert transitioned.dense_values() == expected


def test_transition_preserves_shared_structure() -> None:
    tree = InheritanceTree.from_dense([0.0, 1.0, 0.0, 1.0])

    transitioned = tree.transition(0.25)

    assert transitioned.root == SharedNode(
        SplitNode(
            zero_child=LeafNode(0.25),
            one_child=LeafNode(0.75),
        )
    )
    assert transitioned.node_count() == 4


def test_transition_handles_recombination_boundaries() -> None:
    tree = InheritanceTree.from_dense([0.0, 1.0, 2.0, 3.0])

    unmoved = tree.transition(0.0)
    unlinked = tree.transition(0.5)

    assert unmoved is tree
    assert unlinked.root == LeafNode(tree.weighted_sum())


@pytest.mark.parametrize("theta", [-0.1, 0.500001, math.nan, math.inf])
def test_transition_rejects_invalid_recombination_fraction(theta: float) -> None:
    tree = InheritanceTree.from_dense([0.0, 1.0])

    with pytest.raises(ValueError, match="between 0 and 0.5"):
        tree.transition(theta)


def test_binary_tree_operations_require_matching_meiosis_bits() -> None:
    one_bit = InheritanceTree.from_dense([0.0, 1.0])
    two_bits = InheritanceTree.from_dense([0.0, 1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="same ordered meiosis bits"):
        one_bit.pointwise_multiply(two_bits)
    with pytest.raises(ValueError, match="same ordered meiosis bits"):
        one_bit.mean_product(two_bits)


def test_normalize_scales_tree_to_unit_weighted_sum() -> None:
    tree = InheritanceTree.from_dense([0.0, 2.0, 6.0, 0.0])

    normalized = tree.normalize()

    assert normalized.weighted_sum() == 1.0
    assert normalized.dense_values() == (0.0, 1.0, 3.0, 0.0)


def test_binary_rescale_uses_an_exact_power_of_two() -> None:
    tree = InheritanceTree.from_dense([0.0, 2.0, 6.0, 0.0])

    rescaled = tree.binary_rescale()

    assert rescaled.dense_values() == (0.0, 0.5, 1.5, 0.0)
    assert rescaled.weighted_sum() == 0.5


def test_value_mapping_preserves_uniform_value_probabilities() -> None:
    tree = InheritanceTree.from_dense([1.0, 1.0, 2.0, 3.0])

    indicator = tree.map_values(lambda value: float(value >= 2.0))

    assert tree.value_probabilities() == {
        1.0: 0.5,
        2.0: 0.25,
        3.0: 0.25,
    }
    assert indicator.dense_values() == (0.0, 0.0, 1.0, 1.0)
    assert indicator.value_probabilities() == {0.0: 0.5, 1.0: 0.5}


def test_trim_converts_identical_split_subtrees_to_a_shared_branch() -> None:
    repeated_child = SplitNode(
        zero_child=LeafNode(2.0),
        one_child=ZeroNode(),
    )
    tree = InheritanceTree(
        bit_count=2,
        root=SplitNode(
            zero_child=repeated_child,
            one_child=repeated_child,
        ),
    )

    trimmed = tree.trim()

    assert trimmed.root == SharedNode(repeated_child)
    assert trimmed.dense_values() == (2.0, 0.0, 2.0, 0.0)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "at least one"),
        ([0.0, 1.0, 2.0], "power of two"),
        ([0.0, math.nan], "finite"),
    ],
)
def test_dense_tree_rejects_invalid_values(
    values: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        InheritanceTree.from_dense(values)


def test_value_lookup_requires_a_complete_binary_vector() -> None:
    tree = InheritanceTree.from_dense([0.0, 1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="length"):
        tree.value_at((0,))
    with pytest.raises(ValueError, match="zero or one"):
        tree.value_at((0, 2))


def test_zero_tree_cannot_be_normalized() -> None:
    tree = InheritanceTree(bit_count=3, root=ZeroNode())

    with pytest.raises(ValueError, match="positive finite"):
        tree.normalize()
    with pytest.raises(ValueError, match="positive finite"):
        tree.binary_rescale()


def test_deep_shared_tree_uses_and_restores_scoped_recursion_budget() -> None:
    """Allow PAH-scale depth without permanently changing Python settings."""

    original_limit = sys.getrecursionlimit()
    bit_count = 1_336
    root = LeafNode(0.5)
    for _ in range(bit_count):
        root = SharedNode(root)

    tree = InheritanceTree(bit_count=bit_count, root=root)

    assert tree.weighted_sum() == 0.5
    assert tree.node_count() == bit_count + 1
    assert sys.getrecursionlimit() == original_limit
