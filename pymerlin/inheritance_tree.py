"""Compressed likelihood values over binary inheritance vectors.

MERLIN stores functions of inheritance vectors as trees whose branches are
inheritance bits. Constant and bit-invariant subtrees can terminate early or
share one child. This module provides the immutable reference representation
needed before marker likelihoods and multipoint conditioning are moved away
from explicit ``2 ** meioses`` enumeration.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from itertools import product

from .chain_reduction import CountingPartition, UntypedChain


@dataclass(frozen=True)
class ZeroNode:
    """A subtree whose value is zero for every remaining inheritance vector."""


@dataclass(frozen=True)
class LeafNode:
    """A constant value for every inheritance-vector suffix below this node."""

    value: float


@dataclass(frozen=True)
class _ScaledNode:
    """A build-time exact Decimal scale applied without consuming a bit."""

    child: TreeNode
    factor: Decimal


@dataclass(frozen=True)
class SharedNode:
    """A bit-invariant branch whose zero and one choices share one child."""

    child: TreeNode


@dataclass(frozen=True)
class SplitNode:
    """A branch with distinct children for zero and one inheritance bits."""

    zero_child: TreeNode
    one_child: TreeNode


TreeNode = ZeroNode | LeafNode | _ScaledNode | SharedNode | SplitNode
_PackedTreeRecord = tuple[int] | tuple[int, float] | tuple[int, int] | tuple[
    int,
    int,
    int,
]


@dataclass(frozen=True)
class _DecimalLeafNode:
    """A temporary high-precision leaf used during one transition."""

    value: Decimal


@dataclass(frozen=True)
class _DecimalSharedNode:
    """A temporary bit-invariant high-precision branch."""

    child: _DecimalTreeNode


@dataclass(frozen=True)
class _DecimalSplitNode:
    """A temporary high-precision binary branch."""

    zero_child: _DecimalTreeNode
    one_child: _DecimalTreeNode


_DecimalTreeNode = (
    ZeroNode | _DecimalLeafNode | _DecimalSharedNode | _DecimalSplitNode
)


_RECURSION_BASE_FRAMES = 512
_RECURSION_FRAMES_PER_BIT = 4
_MAX_INHERITANCE_RECURSION_LIMIT = 100_000


@dataclass(frozen=True)
class TreeBuildStatistics:
    """Marker-emission reductions recorded without affecting tree equality."""

    relevant_individual_count: int
    relevant_meiosis_count: int
    suffix_cache_hits: int
    suffix_cache_misses: int
    cached_suffix_count: int
    counting_chain_count: int
    counted_selector_count: int
    recursive_node_count: int = 0
    maximum_recursion_depth: int = 0
    contradiction_prune_count: int = 0
    founder_orientation_reduction_count: int = 0
    founder_couple_reduction_count: int = 0
    counting_reduction_count: int = 0
    invariant_reduction_count: int = 0
    peeled_component_count: int = 0
    peeled_constraint_count: int = 0
    zero_peeled_factor_count: int = 0
    normalized_cache_reuse_count: int = 0
    peeled_factor_cache_hit_count: int = 0
    peeled_factor_cache_miss_count: int = 0
    scaled_tree_cache_hit_count: int = 0


def _required_inheritance_recursion_limit(bit_count: int) -> int:
    """Return the bounded Python recursion limit for one tree depth."""

    if bit_count < 0:
        raise ValueError("Inheritance-tree bit count cannot be negative.")
    required_limit = (
        _RECURSION_BASE_FRAMES + _RECURSION_FRAMES_PER_BIT * bit_count
    )
    if required_limit > _MAX_INHERITANCE_RECURSION_LIMIT:
        raise ValueError(
            "Inheritance tree exceeds the supported scoped recursion budget: "
            f"{bit_count=}."
        )
    return required_limit


@contextmanager
def _inheritance_recursion_budget(bit_count: int) -> Iterator[None]:
    """Temporarily allow recursion proportional to inheritance-tree depth.

    Tree algorithms recurse once per ordered meiosis bit. Python's default
    limit is therefore an implementation constraint for large pedigrees, not
    a scientific complexity limit. The multiplier leaves room for helper
    frames called while one tree frame is active.
    """

    required_limit = _required_inheritance_recursion_limit(bit_count)

    previous_limit = sys.getrecursionlimit()
    if required_limit <= previous_limit:
        yield
        return

    sys.setrecursionlimit(required_limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(previous_limit)


@dataclass(frozen=True)
class InheritanceTree:
    """A compressed float64-compatible function of binary inheritance bits."""

    bit_count: int
    root: TreeNode
    build_statistics: TreeBuildStatistics | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.bit_count < 0:
            raise ValueError("Inheritance-tree bit count cannot be negative.")
        with _inheritance_recursion_budget(self.bit_count):
            _validate_node_depth(self.root, self.bit_count, set())

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Pickle a shared tree as one flat record per unique DAG node."""

        records, root_index = _pack_tree_nodes(self.root)
        return (
            _restore_packed_inheritance_tree,
            (
                self.bit_count,
                records,
                root_index,
                self.build_statistics,
            ),
        )

    @classmethod
    def _from_validated_root(
        cls,
        bit_count: int,
        root: TreeNode,
        build_statistics: TreeBuildStatistics | None = None,
    ) -> InheritanceTree:
        """Construct a tree whose root came from a depth-preserving operation.

        Public construction continues to validate arbitrary roots. Internal
        operations preserve depth by construction, so rescanning a large
        shared DAG after every operation is redundant.
        """

        tree = object.__new__(cls)
        object.__setattr__(tree, "bit_count", bit_count)
        object.__setattr__(tree, "root", root)
        object.__setattr__(tree, "build_statistics", build_statistics)
        return tree

    @classmethod
    def from_dense(
        cls,
        values: Sequence[float],
        bit_count: int | None = None,
    ) -> InheritanceTree:
        """Compress dense values ordered like ``itertools.product((0, 1))``."""

        dense_values = tuple(float(value) for value in values)
        if not dense_values:
            raise ValueError("A dense inheritance tree requires at least one value.")
        if any(not math.isfinite(value) for value in dense_values):
            raise ValueError("Inheritance-tree values must be finite.")

        inferred_bit_count = _power_of_two_exponent(len(dense_values))
        if bit_count is None:
            bit_count = inferred_bit_count
        elif bit_count < 0:
            raise ValueError("Inheritance-tree bit count cannot be negative.")
        elif len(dense_values) != 1 << bit_count:
            raise ValueError(
                "Dense value count must equal 2 raised to the bit count."
            )

        with _inheritance_recursion_budget(bit_count):
            return cls._from_validated_root(
                bit_count,
                _build_dense_node(dense_values, bit_count),
            )

    def value_at(self, inheritance_bits: Sequence[int]) -> float:
        """Return the value associated with one complete inheritance vector."""

        bits = tuple(inheritance_bits)
        if len(bits) != self.bit_count:
            raise ValueError(
                "Inheritance-vector length does not match the tree bit count."
            )
        if any(bit not in (0, 1) for bit in bits):
            raise ValueError("Inheritance bits must be zero or one.")

        node = self.root
        bit_index = 0
        while True:
            if isinstance(node, ZeroNode):
                return 0.0
            if isinstance(node, LeafNode):
                return node.value
            if isinstance(node, SharedNode):
                node = node.child
            else:
                node = (
                    node.zero_child
                    if bits[bit_index] == 0
                    else node.one_child
                )
            bit_index += 1

    def weighted_sum(self) -> float:
        """Return MERLIN's uniform inheritance-vector weighted sum."""

        with _inheritance_recursion_budget(self.bit_count):
            return _weighted_sum_node(self.root, {})

    def pointwise_multiply(
        self,
        other: InheritanceTree,
    ) -> InheritanceTree:
        """Multiply values for corresponding inheritance vectors."""

        self._require_matching_bits(other)
        with _inheritance_recursion_budget(self.bit_count):
            return self._from_validated_root(
                self.bit_count,
                _multiply_nodes(
                    self.root,
                    other.root,
                    self.bit_count,
                    {},
                ),
            )

    def mean_product(self, other: InheritanceTree) -> float:
        """Return MERLIN's uniform mean of corresponding value products."""

        self._require_matching_bits(other)
        with _inheritance_recursion_budget(self.bit_count):
            return math.fsum(
                _mean_product_terms(
                    self.root,
                    other.root,
                    weight=1.0,
                )
            )

    def transition(
        self,
        recombination_fraction: float,
        *,
        extended_precision: bool = False,
    ) -> InheritanceTree:
        """Move likelihoods through independent meiosis recombination."""

        theta = float(recombination_fraction)
        if not math.isfinite(theta) or not 0.0 <= theta <= 0.5:
            raise ValueError(
                "Recombination fraction must be finite and between 0 and 0.5."
            )
        if theta == 0.0:
            return self
        if theta == 0.5:
            mean = self.weighted_sum()
            root: TreeNode = ZeroNode() if mean == 0.0 else LeafNode(mean)
            return self._from_validated_root(self.bit_count, root)

        with _inheritance_recursion_budget(self.bit_count):
            return self._from_validated_root(
                self.bit_count,
                _transition_node(
                    self.root,
                    theta,
                    self.bit_count,
                    extended_precision,
                    {},
                ),
            )

    def transition_counting_chains(
        self,
        recombination_fraction: float,
        chains: Sequence[UntypedChain],
        *,
        extended_precision: bool = False,
    ) -> InheritanceTree:
        """Transition exact untyped chains through their counting quotient.

        Ordinary meioses retain the binary transition used by ``transition``.
        Every detected chain is held fixed during that pass and then mixed
        through its exact ``r + 1`` count-transition matrix. The returned tree
        still uses the original binary meiosis coordinates, so existing output
        statistics do not need a reduced-state-specific representation.
        """

        theta = float(recombination_fraction)
        if not math.isfinite(theta) or not 0.0 <= theta <= 0.5:
            raise ValueError(
                "Recombination fraction must be finite and between 0 and 0.5."
            )
        normalized_chains = _validate_counting_chains(self.bit_count, chains)
        if not normalized_chains:
            return self.transition(
                theta,
                extended_precision=extended_precision,
            )
        if theta == 0.0:
            return self
        if theta == 0.5:
            mean = self.weighted_sum()
            root: TreeNode = ZeroNode() if mean == 0.0 else LeafNode(mean)
            return self._from_validated_root(self.bit_count, root)

        selector_indices = frozenset(
            bit_index
            for chain in normalized_chains
            for bit_index in chain.selector_bit_indices
        )
        with _inheritance_recursion_budget(self.bit_count):
            with localcontext() as decimal_context:
                decimal_context.prec = 80
                decimal_theta = Decimal.from_float(theta)
                decimal_root = _transition_decimal_node_excluding_bits(
                    _to_decimal_tree_node(self.root),
                    decimal_theta,
                    bit_index=0,
                    remaining_bits=self.bit_count,
                    excluded_bit_indices=selector_indices,
                )
                for chain in normalized_chains:
                    decimal_root = _transition_decimal_counting_chain(
                        decimal_root,
                        self.bit_count,
                        chain,
                        theta,
                    )
                root = _to_float_tree_node(decimal_root)
            return self._from_validated_root(self.bit_count, root)

    def normalize(self) -> InheritanceTree:
        """Scale leaves so the uniform inheritance-vector weighted sum is one."""

        total = self.weighted_sum()
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError(
                "A tree requires a positive finite weighted sum to normalize."
            )
        with _inheritance_recursion_budget(self.bit_count):
            return self._from_validated_root(
                self.bit_count,
                _trim_node(
                    _scale_node(self.root, 1.0 / total, {}),
                    {},
                ),
            )

    def binary_rescale(self) -> InheritanceTree:
        """Rescale likelihoods exactly so their weighted sum is near one."""

        total = self.weighted_sum()
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError(
                "A tree requires a positive finite weighted sum to rescale."
            )
        _, exponent = math.frexp(total)
        with _inheritance_recursion_budget(self.bit_count):
            return self._from_validated_root(
                self.bit_count,
                _trim_node(
                    _ldexp_node(self.root, -exponent, {}),
                    {},
                ),
            )

    def trim(self) -> InheritanceTree:
        """Collapse exact zero, constant, and bit-invariant branches."""

        with _inheritance_recursion_budget(self.bit_count):
            return self._from_validated_root(
                self.bit_count,
                _trim_node(self.root, {}),
            )

    def map_values(
        self,
        transform: Callable[[float], float],
    ) -> InheritanceTree:
        """Transform terminal values while preserving compressed branches."""

        with _inheritance_recursion_budget(self.bit_count):
            return self._from_validated_root(
                self.bit_count,
                _map_node_values(self.root, transform, {}),
            )

    def value_probabilities(self) -> dict[float, float]:
        """Return each value's probability under uniform inheritance bits."""

        with _inheritance_recursion_budget(self.bit_count):
            probability_terms: dict[float, list[float]] = {}
            _accumulate_value_probabilities(
                self.root,
                probability=1.0,
                probability_terms=probability_terms,
            )
            return {
                value: math.fsum(probability_terms[value])
                for value in sorted(probability_terms)
            }

    def node_count(self) -> int:
        """Count stored nodes, including shared children only once per reference."""

        with _inheritance_recursion_budget(self.bit_count):
            return _node_count(self.root)

    def unique_node_count(self) -> int:
        """Count unique node objects in the shared inheritance DAG."""

        with _inheritance_recursion_budget(self.bit_count):
            return _unique_node_count(self.root, set())

    def dense_values(self) -> tuple[float, ...]:
        """Expand values in reference inheritance-vector order for validation."""

        return tuple(
            self.value_at(bits)
            for bits in product((0, 1), repeat=self.bit_count)
        )

    def _require_matching_bits(self, other: InheritanceTree) -> None:
        if self.bit_count != other.bit_count:
            raise ValueError(
                "Inheritance trees must have the same ordered meiosis bits."
            )


def _power_of_two_exponent(value_count: int) -> int:
    if value_count & (value_count - 1):
        raise ValueError("Dense inheritance-tree value count must be a power of two.")
    return value_count.bit_length() - 1


def _pack_tree_nodes(
    root: TreeNode,
) -> tuple[tuple[_PackedTreeRecord, ...], int]:
    """Encode an immutable tree DAG in child-before-parent order."""

    node_index_by_id: dict[int, int] = {}
    records: list[_PackedTreeRecord] = []
    stack: list[tuple[TreeNode, bool]] = [(root, False)]

    while stack:
        node, children_visited = stack.pop()
        node_id = id(node)
        if node_id in node_index_by_id:
            continue
        if not children_visited and isinstance(node, (SharedNode, SplitNode)):
            stack.append((node, True))
            if isinstance(node, SharedNode):
                stack.append((node.child, False))
            else:
                stack.append((node.one_child, False))
                stack.append((node.zero_child, False))
            continue

        if isinstance(node, ZeroNode):
            record: _PackedTreeRecord = (0,)
        elif isinstance(node, LeafNode):
            record = (1, node.value)
        elif isinstance(node, SharedNode):
            record = (2, node_index_by_id[id(node.child)])
        else:
            record = (
                3,
                node_index_by_id[id(node.zero_child)],
                node_index_by_id[id(node.one_child)],
            )
        node_index_by_id[node_id] = len(records)
        records.append(record)

    return tuple(records), node_index_by_id[id(root)]


def _restore_packed_inheritance_tree(
    bit_count: int,
    records: tuple[_PackedTreeRecord, ...],
    root_index: int,
    build_statistics: TreeBuildStatistics | None,
) -> InheritanceTree:
    """Rebuild an identity-sharing tree from compact worker records."""

    nodes: list[TreeNode] = []
    for record in records:
        node_kind = record[0]
        if node_kind == 0:
            node: TreeNode = ZeroNode()
        elif node_kind == 1:
            node = LeafNode(record[1])
        elif node_kind == 2:
            node = SharedNode(nodes[record[1]])
        elif node_kind == 3:
            node = SplitNode(nodes[record[1]], nodes[record[2]])
        else:
            raise ValueError(f"Unknown packed inheritance node kind: {node_kind}.")
        nodes.append(node)

    if not 0 <= root_index < len(nodes):
        raise ValueError("Packed inheritance-tree root is outside its records.")
    return InheritanceTree._from_validated_root(
        bit_count,
        nodes[root_index],
        build_statistics,
    )


def _build_dense_node(values: tuple[float, ...], remaining_bits: int) -> TreeNode:
    first_value = values[0]
    if all(value == first_value for value in values):
        return ZeroNode() if first_value == 0.0 else LeafNode(first_value)
    if remaining_bits == 0:
        raise ValueError("A zero-bit tree cannot contain multiple values.")

    midpoint = len(values) // 2
    zero_child = _build_dense_node(values[:midpoint], remaining_bits - 1)
    one_child = _build_dense_node(values[midpoint:], remaining_bits - 1)
    return _combine_children(zero_child, one_child)


def _combine_children(zero_child: TreeNode, one_child: TreeNode) -> TreeNode:
    if not _tree_nodes_equal(zero_child, one_child):
        return SplitNode(zero_child=zero_child, one_child=one_child)
    if _is_terminal_node(zero_child):
        return zero_child
    return SharedNode(child=zero_child)


def _is_terminal_node(node: TreeNode) -> bool:
    """Return whether a build-time node is constant over remaining bits."""

    while isinstance(node, _ScaledNode):
        node = node.child
    return isinstance(node, (ZeroNode, LeafNode))


def _scaled_node(node: TreeNode, factor: Decimal) -> TreeNode:
    """Create one flattened lazy scale for a build-time subtree."""

    if factor == 0 or isinstance(node, ZeroNode):
        return ZeroNode()
    if factor == 1:
        return node
    if isinstance(node, _ScaledNode):
        return _scaled_node(node.child, node.factor * factor)
    return _ScaledNode(child=node, factor=factor)


def _materialize_scaled_tree(node: TreeNode) -> TreeNode:
    """Apply all build-time scales with one float rounding per terminal."""

    memo: dict[tuple[int, Decimal], TreeNode] = {}

    def materialize(current: TreeNode, factor: Decimal) -> TreeNode:
        while isinstance(current, _ScaledNode):
            factor *= current.factor
            current = current.child
        cache_key = (id(current), factor)
        cached_node = memo.get(cache_key)
        if cached_node is not None:
            return cached_node
        if isinstance(current, ZeroNode):
            result: TreeNode = current
        elif isinstance(current, LeafNode):
            value = float(Decimal.from_float(current.value) * factor)
            result = ZeroNode() if value == 0.0 else LeafNode(value)
        elif isinstance(current, SharedNode):
            child = materialize(current.child, factor)
            result = _combine_children(child, child)
        else:
            if not isinstance(current, SplitNode):
                raise TypeError(
                    f"Unsupported inheritance-tree node: {type(current)!r}"
                )
            result = _combine_children(
                materialize(current.zero_child, factor),
                materialize(current.one_child, factor),
            )
        memo[cache_key] = result
        return result

    return materialize(node, Decimal(1))


def _tree_nodes_equal(first: TreeNode, second: TreeNode) -> bool:
    """Compare shared trees without revisiting the same DAG node pair.

    Dataclass equality recursively follows every incoming edge. That repeats
    work exponentially when two separately built trees share many internal
    nodes. Tracking compared identity pairs keeps equality proportional to the
    unique subgraphs while retaining structural compression.
    """

    pending_pairs = [(first, second)]
    compared_pairs: set[tuple[int, int]] = set()
    while pending_pairs:
        first_node, second_node = pending_pairs.pop()
        if first_node is second_node:
            continue
        comparison_key = (id(first_node), id(second_node))
        if comparison_key in compared_pairs:
            continue
        compared_pairs.add(comparison_key)
        if type(first_node) is not type(second_node):
            return False
        if isinstance(first_node, ZeroNode):
            continue
        if isinstance(first_node, LeafNode):
            if not isinstance(second_node, LeafNode):
                return False
            if first_node.value != second_node.value:
                return False
            continue
        if isinstance(first_node, _ScaledNode):
            if not isinstance(second_node, _ScaledNode):
                return False
            if first_node.factor != second_node.factor:
                return False
            pending_pairs.append((first_node.child, second_node.child))
            continue
        if isinstance(first_node, SharedNode):
            if not isinstance(second_node, SharedNode):
                return False
            pending_pairs.append((first_node.child, second_node.child))
            continue
        if not isinstance(first_node, SplitNode) or not isinstance(
            second_node,
            SplitNode,
        ):
            return False
        pending_pairs.extend(
            (
                (first_node.zero_child, second_node.zero_child),
                (first_node.one_child, second_node.one_child),
            )
        )
    return True


def _finite_terminal_node(value: float) -> TreeNode:
    """Create a terminal while retaining public finite-value guarantees."""

    if not math.isfinite(value):
        raise ValueError("Inheritance-tree leaf values must be finite.")
    return ZeroNode() if value == 0.0 else LeafNode(value)


def _weighted_sum_node(
    node: TreeNode,
    memo: dict[int, float],
) -> float:
    cached_value = memo.get(id(node))
    if cached_value is not None:
        return cached_value
    if isinstance(node, ZeroNode):
        return 0.0
    if isinstance(node, LeafNode):
        return node.value
    if isinstance(node, SharedNode):
        value = _weighted_sum_node(node.child, memo)
    else:
        value = math.fsum(
            (
                0.5 * _weighted_sum_node(node.zero_child, memo),
                0.5 * _weighted_sum_node(node.one_child, memo),
            )
        )
    memo[id(node)] = value
    return value


def _multiply_nodes(
    first: TreeNode,
    second: TreeNode,
    remaining_bits: int,
    memo: dict[tuple[object, ...], TreeNode],
) -> TreeNode:
    cache_key = ("multiply", id(first), id(second), remaining_bits)
    cached_node = memo.get(cache_key)
    if cached_node is not None:
        return cached_node

    if isinstance(first, ZeroNode) or isinstance(second, ZeroNode):
        result: TreeNode = ZeroNode()
    elif isinstance(first, LeafNode):
        result = _trim_node(
            _scale_node(second, first.value, memo),
            memo,
        )
    elif isinstance(second, LeafNode):
        result = _trim_node(
            _scale_node(first, second.value, memo),
            memo,
        )
    else:
        first_zero, first_one = _branch_children(first)
        second_zero, second_one = _branch_children(second)
        result = _combine_children(
            _multiply_nodes(
                first_zero,
                second_zero,
                remaining_bits - 1,
                memo,
            ),
            _multiply_nodes(
                first_one,
                second_one,
                remaining_bits - 1,
                memo,
            ),
        )
    memo[cache_key] = result
    return result


def _mean_product_terms(
    first: TreeNode,
    second: TreeNode,
    weight: float,
) -> Iterator[float]:
    if isinstance(first, ZeroNode) or isinstance(second, ZeroNode):
        return
    if isinstance(first, LeafNode) and isinstance(second, LeafNode):
        yield weight * first.value * second.value
        return
    if isinstance(first, SharedNode) and isinstance(second, SharedNode):
        yield from _mean_product_terms(
            first.child,
            second.child,
            weight,
        )
        return
    if isinstance(first, SharedNode):
        second_zero, second_one = _children_or_constant(second)
        yield from _mean_product_terms(
            first.child,
            second_zero,
            weight * 0.5,
        )
        yield from _mean_product_terms(
            first.child,
            second_one,
            weight * 0.5,
        )
        return
    if isinstance(second, SharedNode):
        first_zero, first_one = _children_or_constant(first)
        yield from _mean_product_terms(
            first_zero,
            second.child,
            weight * 0.5,
        )
        yield from _mean_product_terms(
            first_one,
            second.child,
            weight * 0.5,
        )
        return

    first_zero, first_one = _children_or_constant(first)
    second_zero, second_one = _children_or_constant(second)
    yield from _mean_product_terms(
        first_zero,
        second_zero,
        weight * 0.5,
    )
    yield from _mean_product_terms(
        first_one,
        second_one,
        weight * 0.5,
    )


def _transition_node(
    node: TreeNode,
    theta: float,
    remaining_bits: int,
    extended_precision: bool,
    memo: dict[tuple[object, ...], TreeNode],
) -> TreeNode:
    cache_key = ("transition", id(node), remaining_bits)
    cached_node = memo.get(cache_key)
    if cached_node is not None:
        return cached_node
    if isinstance(node, (ZeroNode, LeafNode)):
        return node

    zero_child, one_child = _branch_children(node)
    if isinstance(node, SharedNode):
        transitioned_zero = _transition_node(
            zero_child,
            theta,
            remaining_bits - 1,
            extended_precision,
            memo,
        )
        result = _combine_children(transitioned_zero, transitioned_zero)
        memo[cache_key] = result
        return result

    complement = 1.0 - theta
    mixed_zero = _linear_combination_nodes(
        zero_child,
        complement,
        one_child,
        theta,
        remaining_bits - 1,
        extended_precision,
        memo,
    )
    mixed_one = _linear_combination_nodes(
        zero_child,
        theta,
        one_child,
        complement,
        remaining_bits - 1,
        extended_precision,
        memo,
    )
    result = _combine_children(
        _transition_node(
            mixed_zero,
            theta,
            remaining_bits - 1,
            extended_precision,
            memo,
        ),
        _transition_node(
            mixed_one,
            theta,
            remaining_bits - 1,
            extended_precision,
            memo,
        ),
    )
    memo[cache_key] = result
    return result


def _validate_counting_chains(
    bit_count: int,
    chains: Sequence[UntypedChain],
) -> tuple[UntypedChain, ...]:
    """Validate disjoint, ordered chain coordinates for one tree."""

    normalized_chains = tuple(chains)
    used_bit_indices: set[int] = set()
    for chain in normalized_chains:
        if tuple(sorted(chain.selector_bit_indices)) != (
            chain.selector_bit_indices
        ):
            raise ValueError("Untyped-chain selector bits must be ordered.")
        if any(
            bit_index < 0 or bit_index >= bit_count
            for bit_index in chain.selector_bit_indices
        ):
            raise ValueError(
                "Untyped-chain selector bit is outside the inheritance tree."
            )
        overlap = used_bit_indices.intersection(chain.selector_bit_indices)
        if overlap:
            raise ValueError("Untyped-chain selector bits must be disjoint.")
        used_bit_indices.update(chain.selector_bit_indices)
    return normalized_chains


def _to_decimal_tree_node(node: TreeNode) -> _DecimalTreeNode:
    """Copy a float tree into temporary high-precision leaves."""

    if isinstance(node, ZeroNode):
        return node
    if isinstance(node, LeafNode):
        return _DecimalLeafNode(Decimal.from_float(node.value))
    if isinstance(node, SharedNode):
        return _DecimalSharedNode(_to_decimal_tree_node(node.child))
    return _DecimalSplitNode(
        zero_child=_to_decimal_tree_node(node.zero_child),
        one_child=_to_decimal_tree_node(node.one_child),
    )


def _to_float_tree_node(node: _DecimalTreeNode) -> TreeNode:
    """Round temporary high-precision leaves once to float64."""

    if isinstance(node, ZeroNode):
        return node
    if isinstance(node, _DecimalLeafNode):
        value = float(node.value)
        return ZeroNode() if value == 0.0 else LeafNode(value)
    if isinstance(node, _DecimalSharedNode):
        child = _to_float_tree_node(node.child)
        return _combine_children(child, child)
    return _combine_children(
        _to_float_tree_node(node.zero_child),
        _to_float_tree_node(node.one_child),
    )


def _transition_decimal_node_excluding_bits(
    node: _DecimalTreeNode,
    theta: Decimal,
    bit_index: int,
    remaining_bits: int,
    excluded_bit_indices: frozenset[int],
) -> _DecimalTreeNode:
    """Transition ordinary bits without rounding before count reduction."""

    if isinstance(node, (ZeroNode, _DecimalLeafNode)):
        return node

    zero_child, one_child = _decimal_branch_children(node)
    if bit_index in excluded_bit_indices:
        transitioned_zero = _transition_decimal_node_excluding_bits(
            zero_child,
            theta,
            bit_index + 1,
            remaining_bits - 1,
            excluded_bit_indices,
        )
        if isinstance(node, _DecimalSharedNode):
            return _combine_decimal_children(
                transitioned_zero,
                transitioned_zero,
            )
        transitioned_one = _transition_decimal_node_excluding_bits(
            one_child,
            theta,
            bit_index + 1,
            remaining_bits - 1,
            excluded_bit_indices,
        )
        return _combine_decimal_children(
            transitioned_zero,
            transitioned_one,
        )

    if isinstance(node, _DecimalSharedNode):
        transitioned_child = _transition_decimal_node_excluding_bits(
            zero_child,
            theta,
            bit_index + 1,
            remaining_bits - 1,
            excluded_bit_indices,
        )
        return _combine_decimal_children(
            transitioned_child,
            transitioned_child,
        )

    complement = Decimal(1) - theta
    mixed_zero = _linear_combination_decimal_nodes(
        zero_child,
        complement,
        one_child,
        theta,
        remaining_bits - 1,
    )
    mixed_one = _linear_combination_decimal_nodes(
        zero_child,
        theta,
        one_child,
        complement,
        remaining_bits - 1,
    )
    return _combine_decimal_children(
        _transition_decimal_node_excluding_bits(
            mixed_zero,
            theta,
            bit_index + 1,
            remaining_bits - 1,
            excluded_bit_indices,
        ),
        _transition_decimal_node_excluding_bits(
            mixed_one,
            theta,
            bit_index + 1,
            remaining_bits - 1,
            excluded_bit_indices,
        ),
    )


def _linear_combination_decimal_nodes(
    first: _DecimalTreeNode,
    first_scale: Decimal,
    second: _DecimalTreeNode,
    second_scale: Decimal,
    remaining_bits: int,
) -> _DecimalTreeNode:
    """Combine two high-precision trees without intermediate float rounding."""

    if first_scale == 0 or isinstance(first, ZeroNode):
        return _scale_decimal_node(second, second_scale)
    if second_scale == 0 or isinstance(second, ZeroNode):
        return _scale_decimal_node(first, first_scale)
    if first == second:
        combined_scale = first_scale + second_scale
        if combined_scale == 1:
            return first
        return _scale_decimal_node(first, combined_scale)
    if isinstance(first, _DecimalLeafNode) and isinstance(
        second,
        _DecimalLeafNode,
    ):
        value = first_scale * first.value + second_scale * second.value
        return ZeroNode() if value == 0 else _DecimalLeafNode(value)
    if remaining_bits == 0:
        raise ValueError("Decimal trees exceed the declared bit depth.")

    first_zero, first_one = _decimal_children_or_constant(first)
    second_zero, second_one = _decimal_children_or_constant(second)
    return _combine_decimal_children(
        _linear_combination_decimal_nodes(
            first_zero,
            first_scale,
            second_zero,
            second_scale,
            remaining_bits - 1,
        ),
        _linear_combination_decimal_nodes(
            first_one,
            first_scale,
            second_one,
            second_scale,
            remaining_bits - 1,
        ),
    )


def _scale_decimal_node(
    node: _DecimalTreeNode,
    scale: Decimal,
) -> _DecimalTreeNode:
    """Scale one temporary high-precision tree."""

    if isinstance(node, ZeroNode):
        return node
    if isinstance(node, _DecimalLeafNode):
        value = node.value * scale
        return ZeroNode() if value == 0 else _DecimalLeafNode(value)
    if isinstance(node, _DecimalSharedNode):
        return _DecimalSharedNode(_scale_decimal_node(node.child, scale))
    return _DecimalSplitNode(
        zero_child=_scale_decimal_node(node.zero_child, scale),
        one_child=_scale_decimal_node(node.one_child, scale),
    )


def _transition_decimal_counting_chain(
    root: _DecimalTreeNode,
    bit_count: int,
    chain: UntypedChain,
    theta: float,
) -> _DecimalTreeNode:
    """Apply one count-class transition without intermediate float rounding."""

    partition = CountingPartition.from_chain(chain)
    class_nodes = tuple(
        _restrict_decimal_tree_bits(
            root,
            bit_index=0,
            bit_count=bit_count,
            assignments=dict(
                zip(
                    chain.selector_bit_indices,
                    representative,
                )
            ),
        )
        for representative in partition.representative_vectors
    )
    transition_matrix = partition.decimal_transition_matrix(theta)
    transitioned_class_nodes = tuple(
        _linear_combination_many_decimal_nodes(
            class_nodes,
            transition_matrix[next_on_count],
            remaining_bits=bit_count,
        )
        for next_on_count in range(partition.class_count)
    )
    return _restore_decimal_counting_class_nodes(
        transitioned_class_nodes,
        bit_index=0,
        bit_count=bit_count,
        on_count=0,
        on_value_by_bit_index=dict(
            zip(chain.selector_bit_indices, chain.on_values)
        ),
    )


def _restrict_decimal_tree_bits(
    node: _DecimalTreeNode,
    bit_index: int,
    bit_count: int,
    assignments: dict[int, int],
) -> _DecimalTreeNode:
    """Fix selected bits in a temporary high-precision tree."""

    if isinstance(node, (ZeroNode, _DecimalLeafNode)):
        return node
    if bit_index >= bit_count:
        raise ValueError("Inheritance-tree branches exceed the declared depth.")

    zero_child, one_child = _decimal_branch_children(node)
    assigned_value = assignments.get(bit_index)
    if assigned_value is not None:
        selected_child = zero_child if assigned_value == 0 else one_child
        restricted_child = _restrict_decimal_tree_bits(
            selected_child,
            bit_index + 1,
            bit_count,
            assignments,
        )
        return _combine_decimal_children(restricted_child, restricted_child)

    return _combine_decimal_children(
        _restrict_decimal_tree_bits(
            zero_child,
            bit_index + 1,
            bit_count,
            assignments,
        ),
        _restrict_decimal_tree_bits(
            one_child,
            bit_index + 1,
            bit_count,
            assignments,
        ),
    )


def _linear_combination_many_decimal_nodes(
    nodes: tuple[_DecimalTreeNode, ...],
    scales: tuple[Decimal, ...],
    remaining_bits: int,
) -> _DecimalTreeNode:
    """Combine count classes while retaining high-precision leaves."""

    if nodes and all(node == nodes[0] for node in nodes[1:]):
        # A Markov transition preserves a function that is invariant across
        # every source class. Avoid traversing its unrelated context branches.
        return nodes[0]

    active_terms = tuple(
        (node, scale)
        for node, scale in zip(nodes, scales)
        if scale != 0 and not isinstance(node, ZeroNode)
    )
    if not active_terms:
        return ZeroNode()

    active_nodes = tuple(node for node, _ in active_terms)
    active_scales = tuple(scale for _, scale in active_terms)
    if all(isinstance(node, _DecimalLeafNode) for node in active_nodes):
        values = tuple(
            node.value
            for node in active_nodes
            if isinstance(node, _DecimalLeafNode)
        )
        value = sum(
            (
                scale * value
                for scale, value in zip(active_scales, values)
            ),
            start=Decimal(0),
        )
        return ZeroNode() if value == 0 else _DecimalLeafNode(value)
    if remaining_bits == 0:
        raise ValueError("Count-class trees exceed the declared bit depth.")

    zero_nodes = []
    one_nodes = []
    for node in active_nodes:
        zero_child, one_child = _decimal_children_or_constant(node)
        zero_nodes.append(zero_child)
        one_nodes.append(one_child)
    return _combine_decimal_children(
        _linear_combination_many_decimal_nodes(
            tuple(zero_nodes),
            active_scales,
            remaining_bits - 1,
        ),
        _linear_combination_many_decimal_nodes(
            tuple(one_nodes),
            active_scales,
            remaining_bits - 1,
        ),
    )


def _restore_decimal_counting_class_nodes(
    class_nodes: tuple[_DecimalTreeNode, ...],
    bit_index: int,
    bit_count: int,
    on_count: int,
    on_value_by_bit_index: dict[int, int],
) -> _DecimalTreeNode:
    """Reconstruct binary coordinates while values remain high precision."""

    if class_nodes and all(
        node == class_nodes[0] for node in class_nodes[1:]
    ):
        return class_nodes[0]
    if bit_index == bit_count:
        return class_nodes[on_count]

    on_value = on_value_by_bit_index.get(bit_index)
    if on_value is not None:
        advanced_nodes = tuple(
            _advance_invariant_decimal_tree_level(node)
            for node in class_nodes
        )
        return _combine_decimal_children(
            _restore_decimal_counting_class_nodes(
                advanced_nodes,
                bit_index + 1,
                bit_count,
                on_count + int(on_value == 0),
                on_value_by_bit_index,
            ),
            _restore_decimal_counting_class_nodes(
                advanced_nodes,
                bit_index + 1,
                bit_count,
                on_count + int(on_value == 1),
                on_value_by_bit_index,
            ),
        )

    zero_nodes = []
    one_nodes = []
    for node in class_nodes:
        zero_child, one_child = _decimal_children_or_constant(node)
        zero_nodes.append(zero_child)
        one_nodes.append(one_child)
    return _combine_decimal_children(
        _restore_decimal_counting_class_nodes(
            tuple(zero_nodes),
            bit_index + 1,
            bit_count,
            on_count,
            on_value_by_bit_index,
        ),
        _restore_decimal_counting_class_nodes(
            tuple(one_nodes),
            bit_index + 1,
            bit_count,
            on_count,
            on_value_by_bit_index,
        ),
    )


def _advance_invariant_decimal_tree_level(
    node: _DecimalTreeNode,
) -> _DecimalTreeNode:
    """Consume one invariant selector in a high-precision class tree."""

    if isinstance(node, (ZeroNode, _DecimalLeafNode)):
        return node
    zero_child, one_child = _decimal_branch_children(node)
    if zero_child != one_child:
        raise ValueError(
            "Untyped-chain likelihood is not constant within a count class."
        )
    return zero_child


def _combine_decimal_children(
    zero_child: _DecimalTreeNode,
    one_child: _DecimalTreeNode,
) -> _DecimalTreeNode:
    """Compress one temporary high-precision branch."""

    if zero_child != one_child:
        return _DecimalSplitNode(zero_child, one_child)
    if isinstance(zero_child, (ZeroNode, _DecimalLeafNode)):
        return zero_child
    return _DecimalSharedNode(zero_child)


def _decimal_branch_children(
    node: _DecimalTreeNode,
) -> tuple[_DecimalTreeNode, _DecimalTreeNode]:
    """Return children from a temporary high-precision branch."""

    if isinstance(node, _DecimalSharedNode):
        return node.child, node.child
    if isinstance(node, _DecimalSplitNode):
        return node.zero_child, node.one_child
    raise ValueError("A terminal decimal tree node has no branches.")


def _decimal_children_or_constant(
    node: _DecimalTreeNode,
) -> tuple[_DecimalTreeNode, _DecimalTreeNode]:
    """Return branch children or repeat one temporary terminal."""

    if isinstance(node, (ZeroNode, _DecimalLeafNode)):
        return node, node
    return _decimal_branch_children(node)


def _restore_counting_class_nodes(
    class_nodes: tuple[TreeNode, ...],
    bit_index: int,
    bit_count: int,
    on_count: int,
    on_value_by_bit_index: dict[int, int],
) -> TreeNode:
    """Reconstruct binary coordinates from exact count-class functions."""

    if class_nodes and all(
        node == class_nodes[0] for node in class_nodes[1:]
    ):
        return class_nodes[0]
    if bit_index == bit_count:
        return class_nodes[on_count]

    on_value = on_value_by_bit_index.get(bit_index)
    if on_value is not None:
        advanced_nodes = tuple(
            _advance_invariant_tree_level(node) for node in class_nodes
        )
        return _combine_children(
            _restore_counting_class_nodes(
                advanced_nodes,
                bit_index + 1,
                bit_count,
                on_count + int(on_value == 0),
                on_value_by_bit_index,
            ),
            _restore_counting_class_nodes(
                advanced_nodes,
                bit_index + 1,
                bit_count,
                on_count + int(on_value == 1),
                on_value_by_bit_index,
            ),
        )

    zero_nodes = []
    one_nodes = []
    for node in class_nodes:
        zero_child, one_child = _children_or_constant(node)
        zero_nodes.append(zero_child)
        one_nodes.append(one_child)
    return _combine_children(
        _restore_counting_class_nodes(
            tuple(zero_nodes),
            bit_index + 1,
            bit_count,
            on_count,
            on_value_by_bit_index,
        ),
        _restore_counting_class_nodes(
            tuple(one_nodes),
            bit_index + 1,
            bit_count,
            on_count,
            on_value_by_bit_index,
        ),
    )


def _advance_invariant_tree_level(node: TreeNode) -> TreeNode:
    """Consume one selector level known to be constant within a class."""

    if _is_terminal_node(node):
        return node
    zero_child, one_child = _children_or_constant(node)
    if zero_child != one_child:
        raise ValueError(
            "Untyped-chain likelihood is not constant within a count class."
        )
    return zero_child


def _linear_combination_nodes(
    first: TreeNode,
    first_scale: float,
    second: TreeNode,
    second_scale: float,
    remaining_bits: int,
    extended_precision: bool,
    memo: dict[tuple[object, ...], TreeNode],
) -> TreeNode:
    cache_key = (
        "linear_combination",
        id(first),
        first_scale,
        id(second),
        second_scale,
        remaining_bits,
        extended_precision,
    )
    cached_node = memo.get(cache_key)
    if cached_node is not None:
        return cached_node

    if first_scale == 0.0 or isinstance(first, ZeroNode):
        result = _trim_node(
            _scale_node(second, second_scale, memo),
            memo,
        )
    elif second_scale == 0.0 or isinstance(second, ZeroNode):
        result = _trim_node(
            _scale_node(first, first_scale, memo),
            memo,
        )
    elif first is second or first == second:
        combined_scale = math.fsum((first_scale, second_scale))
        result = _trim_node(
            _scale_node(first, combined_scale, memo),
            memo,
        )
    elif isinstance(first, LeafNode) and isinstance(second, LeafNode):
        if extended_precision:
            # Retaining extended precision for scaled terms avoids accumulating
            # product-rounding error across successive meiosis mixes.
            value = _accurate_two_term_sumprod(
                (first_scale, second_scale),
                (first.value, second.value),
            )
        else:
            value = math.fsum(
                (
                    first_scale * first.value,
                    second_scale * second.value,
                )
            )
        result = _finite_terminal_node(value)
    else:
        first_zero, first_one = _children_or_constant(first)
        second_zero, second_one = _children_or_constant(second)
        result = _combine_children(
            _linear_combination_nodes(
                first_zero,
                first_scale,
                second_zero,
                second_scale,
                remaining_bits - 1,
                extended_precision,
                memo,
            ),
            _linear_combination_nodes(
                first_one,
                first_scale,
                second_one,
                second_scale,
                remaining_bits - 1,
                extended_precision,
                memo,
            ),
        )
    memo[cache_key] = result
    return result


def _accurate_two_term_sumprod(
    scales: tuple[float, float],
    values: tuple[float, float],
) -> float:
    """Reduce two scaled values with one final float64 rounding."""

    if hasattr(math, "sumprod"):
        return math.sumprod(scales, values)

    # Forty decimal digits exceed the precision required for the exact sum of
    # two binary64 products. This preserves accuracy on Python 3.10 and 3.11.
    with localcontext() as decimal_context:
        decimal_context.prec = 40
        return float(
            Decimal.from_float(scales[0])
            * Decimal.from_float(values[0])
            + Decimal.from_float(scales[1])
            * Decimal.from_float(values[1])
        )


def _branch_children(node: TreeNode) -> tuple[TreeNode, TreeNode]:
    if isinstance(node, SharedNode):
        return node.child, node.child
    if isinstance(node, SplitNode):
        return node.zero_child, node.one_child
    raise ValueError("A terminal inheritance-tree node has no branches.")


def _children_or_constant(node: TreeNode) -> tuple[TreeNode, TreeNode]:
    if isinstance(node, (ZeroNode, LeafNode)):
        return node, node
    if isinstance(node, _ScaledNode):
        zero_child, one_child = _children_or_constant(node.child)
        return (
            _scaled_node(zero_child, node.factor),
            _scaled_node(one_child, node.factor),
        )
    return _branch_children(node)


def _scale_node(
    node: TreeNode,
    scale: float,
    memo: dict[tuple[object, ...], TreeNode],
) -> TreeNode:
    cache_key = ("scale", id(node), scale)
    cached_node = memo.get(cache_key)
    if cached_node is not None:
        return cached_node
    if isinstance(node, ZeroNode):
        return node
    if isinstance(node, LeafNode):
        result: TreeNode = _finite_terminal_node(node.value * scale)
    elif isinstance(node, SharedNode):
        child = _scale_node(node.child, scale, memo)
        result = _combine_children(child, child)
    else:
        result = _combine_children(
            _scale_node(node.zero_child, scale, memo),
            _scale_node(node.one_child, scale, memo),
        )
    memo[cache_key] = result
    return result


def _ldexp_node(
    node: TreeNode,
    exponent: int,
    memo: dict[tuple[object, ...], TreeNode],
) -> TreeNode:
    cache_key = ("ldexp", id(node), exponent)
    cached_node = memo.get(cache_key)
    if cached_node is not None:
        return cached_node
    if isinstance(node, ZeroNode):
        return node
    if isinstance(node, LeafNode):
        result: TreeNode = _finite_terminal_node(
            math.ldexp(node.value, exponent)
        )
    elif isinstance(node, SharedNode):
        child = _ldexp_node(node.child, exponent, memo)
        result = _combine_children(child, child)
    else:
        result = _combine_children(
            _ldexp_node(node.zero_child, exponent, memo),
            _ldexp_node(node.one_child, exponent, memo),
        )
    memo[cache_key] = result
    return result


def _map_node_values(
    node: TreeNode,
    transform: Callable[[float], float],
    memo: dict[int, TreeNode],
) -> TreeNode:
    cached_node = memo.get(id(node))
    if cached_node is not None:
        return cached_node
    if isinstance(node, ZeroNode):
        transformed_value = float(transform(0.0))
        result: TreeNode = _finite_terminal_node(transformed_value)
    elif isinstance(node, LeafNode):
        transformed_value = float(transform(node.value))
        result = _finite_terminal_node(transformed_value)
    elif isinstance(node, SharedNode):
        child = _map_node_values(node.child, transform, memo)
        result = _combine_children(child, child)
    else:
        result = _combine_children(
            _map_node_values(node.zero_child, transform, memo),
            _map_node_values(node.one_child, transform, memo),
        )
    memo[id(node)] = result
    return result


def _accumulate_value_probabilities(
    node: TreeNode,
    probability: float,
    probability_terms: dict[float, list[float]],
) -> None:
    if isinstance(node, ZeroNode):
        probability_terms.setdefault(0.0, []).append(probability)
        return
    if isinstance(node, LeafNode):
        probability_terms.setdefault(node.value, []).append(probability)
        return
    if isinstance(node, SharedNode):
        _accumulate_value_probabilities(
            node.child,
            probability,
            probability_terms,
        )
        return

    branch_probability = probability * 0.5
    _accumulate_value_probabilities(
        node.zero_child,
        branch_probability,
        probability_terms,
    )
    _accumulate_value_probabilities(
        node.one_child,
        branch_probability,
        probability_terms,
    )


def _trim_node(
    node: TreeNode,
    memo: dict[tuple[object, ...], TreeNode],
) -> TreeNode:
    cache_key = ("trim", id(node))
    cached_node = memo.get(cache_key)
    if cached_node is not None:
        return cached_node
    if isinstance(node, ZeroNode):
        return node
    if isinstance(node, LeafNode):
        result: TreeNode = ZeroNode() if node.value == 0.0 else node
    elif isinstance(node, SharedNode):
        child = _trim_node(node.child, memo)
        if isinstance(child, (ZeroNode, LeafNode)):
            result = child
        else:
            result = _combine_children(child, child)
    else:
        zero_child = _trim_node(node.zero_child, memo)
        one_child = _trim_node(node.one_child, memo)
        result = _combine_children(zero_child, one_child)
    memo[cache_key] = result
    return result


def _validate_node_depth(
    node: TreeNode,
    remaining_bits: int,
    validated_states: set[tuple[int, int]],
) -> None:
    validation_key = (id(node), remaining_bits)
    if validation_key in validated_states:
        return
    if isinstance(node, LeafNode):
        if not math.isfinite(node.value):
            raise ValueError("Inheritance-tree leaf values must be finite.")
        validated_states.add(validation_key)
        return
    if isinstance(node, ZeroNode):
        validated_states.add(validation_key)
        return
    if remaining_bits == 0:
        raise ValueError("Inheritance-tree branches exceed the declared bit count.")
    if isinstance(node, SharedNode):
        _validate_node_depth(
            node.child,
            remaining_bits - 1,
            validated_states,
        )
        validated_states.add(validation_key)
        return
    _validate_node_depth(
        node.zero_child,
        remaining_bits - 1,
        validated_states,
    )
    _validate_node_depth(
        node.one_child,
        remaining_bits - 1,
        validated_states,
    )
    validated_states.add(validation_key)


def _node_count(node: TreeNode) -> int:
    if isinstance(node, (ZeroNode, LeafNode)):
        return 1
    if isinstance(node, SharedNode):
        return 1 + _node_count(node.child)
    return 1 + _node_count(node.zero_child) + _node_count(node.one_child)


def _unique_node_count(node: TreeNode, seen_node_ids: set[int]) -> int:
    node_id = id(node)
    if node_id in seen_node_ids:
        return 0
    seen_node_ids.add(node_id)
    if isinstance(node, (ZeroNode, LeafNode)):
        return 1
    if isinstance(node, SharedNode):
        return 1 + _unique_node_count(node.child, seen_node_ids)
    return (
        1
        + _unique_node_count(node.zero_child, seen_node_ids)
        + _unique_node_count(node.one_child, seen_node_ids)
    )
