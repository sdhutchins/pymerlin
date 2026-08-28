"""Exact quotient coordinates for arbitrary founder-allele orientations.

An inheritance vector assigns allele zero or one to every parental
transmission. The two allele labels of a founder are arbitrary, so jointly
flipping every transmission from that founder describes the same unlabeled
inheritance state. This module fixes one transmission per transmitting founder
to zero and retains the other transmissions as relative orientation bits.

The quotient changes transition coordinates, not the underlying likelihood
model. Relative bits from the same founder therefore share one exact
transition factor and must not be treated as independent binary meioses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .inheritance_tree import (
    InheritanceTree,
    LeafNode,
    SharedNode,
    SplitNode,
    TreeNode,
    ZeroNode,
    _combine_children,
    _inheritance_recursion_budget,
    _scaled_node,
    _ScaledNode,
)
from .models import Family


@dataclass(frozen=True)
class FounderOrientationGroup:
    """One founder's full transmissions and retained relative coordinates."""

    founder_id: str
    representative_full_bit_index: int
    member_full_bit_indices: tuple[int, ...]
    reduced_member_bit_indices: tuple[int, ...]


@dataclass(frozen=True)
class FounderOrientationQuotient:
    """Coordinate map after fixing one orientation bit per founder."""

    full_bit_count: int
    reduced_bit_count: int
    reduced_bit_index_by_full_bit: tuple[int | None, ...]
    groups: tuple[FounderOrientationGroup, ...]

    @property
    def transition_interaction_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return relative-bit scopes coupled by exact founder transitions."""

        return tuple(
            group.reduced_member_bit_indices
            for group in self.groups
            if len(group.reduced_member_bit_indices) > 1
        )

    def expand_canonical(
        self,
        reduced_bits: Sequence[int],
    ) -> tuple[int, ...]:
        """Expand quotient bits with every representative orientation at zero."""

        validated_reduced_bits = _validate_binary_bits(
            reduced_bits,
            self.reduced_bit_count,
            "Reduced inheritance vector",
        )
        full_bits = [0] * self.full_bit_count
        for full_bit_index, reduced_bit_index in enumerate(
            self.reduced_bit_index_by_full_bit
        ):
            if reduced_bit_index is not None:
                full_bits[full_bit_index] = validated_reduced_bits[reduced_bit_index]
        return tuple(full_bits)

    def project(self, full_bits: Sequence[int]) -> tuple[int, ...]:
        """Project a full vector to founder-relative quotient coordinates."""

        validated_full_bits = _validate_binary_bits(
            full_bits,
            self.full_bit_count,
            "Full inheritance vector",
        )
        reduced_bits = [0] * self.reduced_bit_count
        representative_by_member = {
            member_bit_index: group.representative_full_bit_index
            for group in self.groups
            for member_bit_index in group.member_full_bit_indices[1:]
        }
        for full_bit_index, reduced_bit_index in enumerate(
            self.reduced_bit_index_by_full_bit
        ):
            if reduced_bit_index is None:
                continue
            representative_bit_index = representative_by_member.get(full_bit_index)
            reduced_bits[reduced_bit_index] = validated_full_bits[full_bit_index]
            if representative_bit_index is not None:
                reduced_bits[reduced_bit_index] ^= validated_full_bits[
                    representative_bit_index
                ]
        return tuple(reduced_bits)


def build_founder_orientation_quotient(
    family: Family,
) -> FounderOrientationQuotient:
    """Build the exact founder-label quotient for one ordered meiosis list."""

    founder_ids = {founder.individual_id for founder in family.founders}
    member_indices_by_founder: dict[str, list[int]] = {}
    for full_bit_index, meiosis in enumerate(family.meioses):
        if meiosis.parent_id in founder_ids:
            member_indices_by_founder.setdefault(
                meiosis.parent_id,
                [],
            ).append(full_bit_index)

    representative_indices = {
        member_indices[0] for member_indices in member_indices_by_founder.values()
    }
    reduced_index_by_full_bit: list[int | None] = []
    next_reduced_bit_index = 0
    for full_bit_index in range(len(family.meioses)):
        if full_bit_index in representative_indices:
            reduced_index_by_full_bit.append(None)
            continue
        reduced_index_by_full_bit.append(next_reduced_bit_index)
        next_reduced_bit_index += 1

    groups = tuple(
        FounderOrientationGroup(
            founder_id=founder_id,
            representative_full_bit_index=member_indices[0],
            member_full_bit_indices=tuple(member_indices),
            reduced_member_bit_indices=tuple(
                reduced_index_by_full_bit[full_bit_index]
                for full_bit_index in member_indices[1:]
                if reduced_index_by_full_bit[full_bit_index] is not None
            ),
        )
        for founder_id, member_indices in member_indices_by_founder.items()
    )
    return FounderOrientationQuotient(
        full_bit_count=len(family.meioses),
        reduced_bit_count=next_reduced_bit_index,
        reduced_bit_index_by_full_bit=tuple(reduced_index_by_full_bit),
        groups=groups,
    )


def reduce_founder_orientation_tree(
    tree: InheritanceTree,
    quotient: FounderOrientationQuotient,
) -> InheritanceTree:
    """Select the canonical founder orientation and remove its fixed bits.

    The input function must be invariant under jointly flipping all
    transmissions from each founder. Marker likelihood trees satisfy this
    condition because founder allele labels are integrated symmetrically.
    """

    if tree.bit_count != quotient.full_bit_count:
        raise ValueError(
            "Inheritance tree and founder quotient must use the same full "
            "inheritance bits."
        )

    hidden_bit_indices = frozenset(
        group.representative_full_bit_index for group in quotient.groups
    )
    memo: dict[tuple[int, int], TreeNode] = {}

    def select_canonical(node: TreeNode, full_bit_index: int) -> TreeNode:
        cache_key = (id(node), full_bit_index)
        cached_node = memo.get(cache_key)
        if cached_node is not None:
            return cached_node

        if isinstance(node, (ZeroNode, LeafNode)):
            reduced_node = node
        elif isinstance(node, _ScaledNode):
            reduced_node = _scaled_node(
                select_canonical(node.child, full_bit_index),
                node.factor,
            )
        elif isinstance(node, SharedNode):
            reduced_child = select_canonical(
                node.child,
                full_bit_index + 1,
            )
            if full_bit_index in hidden_bit_indices:
                reduced_node = reduced_child
            else:
                reduced_node = _combine_children(
                    reduced_child,
                    reduced_child,
                )
        elif isinstance(node, SplitNode):
            reduced_zero_child = select_canonical(
                node.zero_child,
                full_bit_index + 1,
            )
            if full_bit_index in hidden_bit_indices:
                reduced_node = reduced_zero_child
            else:
                reduced_node = _combine_children(
                    reduced_zero_child,
                    select_canonical(
                        node.one_child,
                        full_bit_index + 1,
                    ),
                )
        else:
            raise TypeError(f"Unsupported inheritance-tree node: {type(node)!r}")

        memo[cache_key] = reduced_node
        return reduced_node

    with _inheritance_recursion_budget(tree.bit_count):
        reduced_root = select_canonical(tree.root, 0)
    return InheritanceTree(
        bit_count=quotient.reduced_bit_count,
        root=reduced_root,
    )


def founder_group_transition_probability(
    previous_relative_bits: Sequence[int],
    next_relative_bits: Sequence[int],
    recombination_fraction: float,
) -> float:
    """Return one founder group's exact quotient transition probability.

    The source uses canonical orientation zero. The result sums transitions to
    both full target orientations represented by the requested relative bits.
    """

    previous_bits = tuple(previous_relative_bits)
    next_bits = tuple(next_relative_bits)
    if len(previous_bits) != len(next_bits):
        raise ValueError(
            "Founder transition vectors must have the same number of bits."
        )
    previous_bits = _validate_binary_bits(
        previous_bits,
        len(previous_bits),
        "Previous founder-relative vector",
    )
    next_bits = _validate_binary_bits(
        next_bits,
        len(next_bits),
        "Next founder-relative vector",
    )
    theta = float(recombination_fraction)
    if not math.isfinite(theta) or not 0.0 <= theta <= 0.5:
        raise ValueError("Recombination fraction must be finite and between 0 and 0.5.")

    canonical_source = (0, *previous_bits)
    canonical_target = (0, *next_bits)
    complementary_target = (1, *(1 - bit for bit in next_bits))
    return _independent_transition_probability(
        canonical_source,
        canonical_target,
        theta,
    ) + _independent_transition_probability(
        canonical_source,
        complementary_target,
        theta,
    )


def _independent_transition_probability(
    previous_bits: tuple[int, ...],
    next_bits: tuple[int, ...],
    theta: float,
) -> float:
    """Return the product kernel for labeled full inheritance bits."""

    return math.prod(
        1.0 - theta if previous == next_value else theta
        for previous, next_value in zip(previous_bits, next_bits)
    )


def _validate_binary_bits(
    bits: Sequence[int],
    expected_length: int,
    label: str,
) -> tuple[int, ...]:
    """Validate and freeze one binary inheritance-vector input."""

    validated_bits = tuple(bits)
    if len(validated_bits) != expected_length:
        raise ValueError(f"{label} has the wrong number of bits.")
    if any(bit not in (0, 1) for bit in validated_bits):
        raise ValueError(f"{label} must contain only zero and one.")
    return validated_bits
