"""Exact quotient coordinates for exchangeable founder couples.

The marker likelihood is unchanged when two effectively identical founders
are exchanged and the transmissions from their shared children are
complemented. The founder-orientation quotient is applied first. This module
projects the remaining founder-couple involution into those relative
coordinates, fixes one complemented representative bit, and retains the exact
target-orbit mixture required by recombination.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .founder_orientation_quotient import (
    FounderOrientationQuotient,
    founder_group_transition_probability,
)
from .founder_symmetry import (
    FounderCoupleSymmetry,
    build_founder_couple_symmetry_plan,
)
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
class FounderCoupleQuotientGroup:
    """One founder-couple involution in orientation-reduced coordinates."""

    founder_ids: tuple[str, str]
    representative_input_bit_index: int
    affected_input_bit_indices: tuple[int, ...]
    input_bit_index_by_output_bit: tuple[int, ...]
    xor_offset_by_output_bit: tuple[int, ...]

    def transform(self, input_bits: Sequence[int]) -> tuple[int, ...]:
        """Apply this exact affine involution to one input vector."""

        validated_bits = _validate_binary_bits(
            input_bits,
            len(self.input_bit_index_by_output_bit),
            "Founder-couple input vector",
        )
        return tuple(
            validated_bits[source_bit_index] ^ xor_offset
            for source_bit_index, xor_offset in zip(
                self.input_bit_index_by_output_bit,
                self.xor_offset_by_output_bit,
            )
        )


@dataclass(frozen=True)
class FounderCoupleQuotient:
    """Coordinate map after fixing one bit per founder-couple orbit."""

    input_bit_count: int
    reduced_bit_count: int
    reduced_bit_index_by_input_bit: tuple[int | None, ...]
    groups: tuple[FounderCoupleQuotientGroup, ...]

    @property
    def transition_interaction_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return reduced scopes coupled by founder-couple target mixtures."""

        return tuple(
            tuple(
                reduced_bit_index
                for input_bit_index in group.affected_input_bit_indices
                if (
                    reduced_bit_index := self.reduced_bit_index_by_input_bit[
                        input_bit_index
                    ]
                )
                is not None
            )
            for group in self.groups
        )

    def remap_interaction_groups(
        self,
        input_groups: Sequence[Sequence[int]],
    ) -> tuple[tuple[int, ...], ...]:
        """Map pre-quotient transition scopes into retained coordinates."""

        remapped_groups = []
        for input_group in input_groups:
            remapped_group = tuple(
                reduced_bit_index
                for input_bit_index in input_group
                if (
                    reduced_bit_index := self.reduced_bit_index_by_input_bit[
                        input_bit_index
                    ]
                )
                is not None
            )
            if remapped_group:
                remapped_groups.append(remapped_group)
        return tuple(remapped_groups)

    def expand_canonical(
        self,
        reduced_bits: Sequence[int],
    ) -> tuple[int, ...]:
        """Expand retained bits with every couple representative fixed to zero."""

        validated_bits = _validate_binary_bits(
            reduced_bits,
            self.reduced_bit_count,
            "Founder-couple reduced vector",
        )
        input_bits = [0] * self.input_bit_count
        for input_bit_index, reduced_bit_index in enumerate(
            self.reduced_bit_index_by_input_bit
        ):
            if reduced_bit_index is not None:
                input_bits[input_bit_index] = validated_bits[reduced_bit_index]
        return tuple(input_bits)

    def canonicalize(self, input_bits: Sequence[int]) -> tuple[int, ...]:
        """Choose the representative with every hidden couple bit equal to zero."""

        canonical_bits = _validate_binary_bits(
            input_bits,
            self.input_bit_count,
            "Founder-couple input vector",
        )
        for group in self.groups:
            if canonical_bits[group.representative_input_bit_index] == 1:
                canonical_bits = group.transform(canonical_bits)
        return canonical_bits

    def project(self, input_bits: Sequence[int]) -> tuple[int, ...]:
        """Project one orientation-reduced vector to the couple quotient."""

        canonical_bits = self.canonicalize(input_bits)
        return tuple(
            canonical_bits[input_bit_index]
            for input_bit_index, reduced_bit_index in enumerate(
                self.reduced_bit_index_by_input_bit
            )
            if reduced_bit_index is not None
        )

    def target_orbit(self, input_bits: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        """Enumerate the distinct founder-couple orientations of one target."""

        orbit = {
            _validate_binary_bits(
                input_bits,
                self.input_bit_count,
                "Founder-couple input vector",
            )
        }
        for group in self.groups:
            orbit.update(group.transform(bits) for bits in tuple(orbit))
        return tuple(sorted(orbit))


def build_founder_couple_quotient(
    family: Family,
    founder_orientation_quotient: FounderOrientationQuotient,
    relevant_full_bit_indices: frozenset[int] | None = None,
) -> FounderCoupleQuotient:
    """Project exact founder-couple involutions after orientation reduction."""

    if founder_orientation_quotient.full_bit_count != len(family.meioses):
        raise ValueError(
            "Founder orientation quotient and family must use the same full bits."
        )
    if relevant_full_bit_indices is None:
        relevant_full_bit_indices = frozenset(range(len(family.meioses)))

    symmetry_plan = build_founder_couple_symmetry_plan(
        family,
        relevant_full_bit_indices,
    )
    groups = tuple(
        _project_symmetry_to_orientation_coordinates(
            symmetry,
            founder_orientation_quotient,
        )
        for symmetry in symmetry_plan.symmetries
    )
    _validate_disjoint_groups(groups)

    hidden_input_indices = {
        group.representative_input_bit_index for group in groups
    }
    reduced_index_by_input_bit: list[int | None] = []
    next_reduced_bit_index = 0
    for input_bit_index in range(founder_orientation_quotient.reduced_bit_count):
        if input_bit_index in hidden_input_indices:
            reduced_index_by_input_bit.append(None)
        else:
            reduced_index_by_input_bit.append(next_reduced_bit_index)
            next_reduced_bit_index += 1

    return FounderCoupleQuotient(
        input_bit_count=founder_orientation_quotient.reduced_bit_count,
        reduced_bit_count=next_reduced_bit_index,
        reduced_bit_index_by_input_bit=tuple(reduced_index_by_input_bit),
        groups=groups,
    )


def reduce_founder_couple_tree(
    tree: InheritanceTree,
    quotient: FounderCoupleQuotient,
) -> InheritanceTree:
    """Select canonical couple branches and remove their representative bits."""

    if tree.bit_count != quotient.input_bit_count:
        raise ValueError(
            "Inheritance tree and founder-couple quotient must use the same "
            "input bits."
        )

    hidden_bit_indices = frozenset(
        group.representative_input_bit_index for group in quotient.groups
    )
    memo: dict[tuple[int, int], TreeNode] = {}

    def select_canonical(node: TreeNode, input_bit_index: int) -> TreeNode:
        cache_key = (id(node), input_bit_index)
        cached_node = memo.get(cache_key)
        if cached_node is not None:
            return cached_node

        if isinstance(node, (ZeroNode, LeafNode)):
            reduced_node = node
        elif isinstance(node, _ScaledNode):
            reduced_node = _scaled_node(
                select_canonical(node.child, input_bit_index),
                node.factor,
            )
        elif isinstance(node, SharedNode):
            reduced_child = select_canonical(
                node.child,
                input_bit_index + 1,
            )
            if input_bit_index in hidden_bit_indices:
                reduced_node = reduced_child
            else:
                reduced_node = _combine_children(reduced_child, reduced_child)
        elif isinstance(node, SplitNode):
            reduced_zero_child = select_canonical(
                node.zero_child,
                input_bit_index + 1,
            )
            if input_bit_index in hidden_bit_indices:
                reduced_node = reduced_zero_child
            else:
                reduced_node = _combine_children(
                    reduced_zero_child,
                    select_canonical(
                        node.one_child,
                        input_bit_index + 1,
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


def founder_couple_transition_probability(
    previous_reduced_bits: Sequence[int],
    next_reduced_bits: Sequence[int],
    recombination_fraction: float,
    founder_orientation_quotient: FounderOrientationQuotient,
    founder_couple_quotient: FounderCoupleQuotient,
) -> float:
    """Return the exact compound-quotient transition probability."""

    if (
        founder_couple_quotient.input_bit_count
        != founder_orientation_quotient.reduced_bit_count
    ):
        raise ValueError("Founder quotients use incompatible coordinate counts.")
    previous_input_bits = founder_couple_quotient.expand_canonical(
        previous_reduced_bits
    )
    canonical_next_bits = founder_couple_quotient.expand_canonical(
        next_reduced_bits
    )
    return math.fsum(
        _founder_orientation_transition_probability(
            previous_input_bits,
            target_bits,
            recombination_fraction,
            founder_orientation_quotient,
        )
        for target_bits in founder_couple_quotient.target_orbit(
            canonical_next_bits
        )
    )


def _founder_orientation_transition_probability(
    previous_bits: tuple[int, ...],
    next_bits: tuple[int, ...],
    recombination_fraction: float,
    quotient: FounderOrientationQuotient,
) -> float:
    """Return the exact transition within founder-relative coordinates."""

    theta = float(recombination_fraction)
    if not math.isfinite(theta) or not 0.0 <= theta <= 0.5:
        raise ValueError("Recombination fraction must be between 0 and 0.5.")
    grouped_indices = {
        bit_index
        for group in quotient.groups
        for bit_index in group.reduced_member_bit_indices
    }
    founder_factors = (
        founder_group_transition_probability(
            tuple(previous_bits[index] for index in group.reduced_member_bit_indices),
            tuple(next_bits[index] for index in group.reduced_member_bit_indices),
            theta,
        )
        for group in quotient.groups
        if group.reduced_member_bit_indices
    )
    independent_factors = (
        1.0 - theta if previous_bits[index] == next_bits[index] else theta
        for index in range(quotient.reduced_bit_count)
        if index not in grouped_indices
    )
    return math.prod((*founder_factors, *independent_factors))


def _project_symmetry_to_orientation_coordinates(
    symmetry: FounderCoupleSymmetry,
    quotient: FounderOrientationQuotient,
) -> FounderCoupleQuotientGroup:
    """Derive one affine involution after founder orientations are hidden."""

    source_full_bit_by_output_bit = list(range(quotient.full_bit_count))
    xor_offset_by_full_output_bit = [0] * quotient.full_bit_count
    for first_bit_index, second_bit_index in symmetry.swapped_bit_pairs:
        source_full_bit_by_output_bit[first_bit_index] = second_bit_index
        source_full_bit_by_output_bit[second_bit_index] = first_bit_index
    for bit_index in symmetry.toggled_bit_indices:
        xor_offset_by_full_output_bit[bit_index] ^= 1

    representative_by_member = {
        member_full_bit_index: group.representative_full_bit_index
        for group in quotient.groups
        for member_full_bit_index in group.member_full_bit_indices[1:]
    }
    input_index_by_output_index = [0] * quotient.reduced_bit_count
    xor_offset_by_output_index = [0] * quotient.reduced_bit_count
    for full_output_bit_index, output_index in enumerate(
        quotient.reduced_bit_index_by_full_bit
    ):
        if output_index is None:
            continue
        coefficient_bits, xor_offset = _canonical_full_expression(
            full_output_bit_index,
            source_full_bit_by_output_bit,
            xor_offset_by_full_output_bit,
            quotient,
        )
        representative_full_bit_index = representative_by_member.get(
            full_output_bit_index
        )
        if representative_full_bit_index is not None:
            representative_coefficients, representative_offset = (
                _canonical_full_expression(
                    representative_full_bit_index,
                    source_full_bit_by_output_bit,
                    xor_offset_by_full_output_bit,
                    quotient,
                )
            )
            coefficient_bits ^= representative_coefficients
            xor_offset ^= representative_offset
        if coefficient_bits.bit_count() != 1:
            raise ValueError(
                "Founder-couple symmetry is not a coordinate involution after "
                "founder-orientation reduction."
            )
        input_index_by_output_index[output_index] = (
            coefficient_bits.bit_length() - 1
        )
        xor_offset_by_output_index[output_index] = xor_offset

    _validate_affine_involution(
        tuple(input_index_by_output_index),
        tuple(xor_offset_by_output_index),
    )
    representative_input_bit_index = quotient.reduced_bit_index_by_full_bit[
        symmetry.representative_bit_index
    ]
    if representative_input_bit_index is None:
        raise ValueError("Founder-couple representative was hidden unexpectedly.")
    if (
        input_index_by_output_index[representative_input_bit_index]
        != representative_input_bit_index
        or xor_offset_by_output_index[representative_input_bit_index] != 1
    ):
        raise ValueError(
            "Founder-couple involution must complement its representative bit."
        )

    affected_indices = tuple(
        output_index
        for output_index, (input_index, xor_offset) in enumerate(
            zip(input_index_by_output_index, xor_offset_by_output_index)
        )
        if input_index != output_index or xor_offset != 0
    )
    return FounderCoupleQuotientGroup(
        founder_ids=symmetry.founder_ids,
        representative_input_bit_index=representative_input_bit_index,
        affected_input_bit_indices=affected_indices,
        input_bit_index_by_output_bit=tuple(input_index_by_output_index),
        xor_offset_by_output_bit=tuple(xor_offset_by_output_index),
    )


def _canonical_full_expression(
    full_output_bit_index: int,
    source_full_bit_by_output_bit: Sequence[int],
    xor_offset_by_full_output_bit: Sequence[int],
    quotient: FounderOrientationQuotient,
) -> tuple[int, int]:
    """Express one transformed full output in canonical relative inputs."""

    source_full_bit_index = source_full_bit_by_output_bit[full_output_bit_index]
    source_input_bit_index = quotient.reduced_bit_index_by_full_bit[
        source_full_bit_index
    ]
    coefficient_bits = (
        0 if source_input_bit_index is None else 1 << source_input_bit_index
    )
    return coefficient_bits, xor_offset_by_full_output_bit[full_output_bit_index]


def _validate_affine_involution(
    input_index_by_output_index: tuple[int, ...],
    xor_offset_by_output_index: tuple[int, ...],
) -> None:
    """Require a bijective affine map whose square is the identity."""

    bit_count = len(input_index_by_output_index)
    if sorted(input_index_by_output_index) != list(range(bit_count)):
        raise ValueError("Founder-couple coordinate map must be a permutation.")
    for output_index, input_index in enumerate(input_index_by_output_index):
        second_input_index = input_index_by_output_index[input_index]
        second_offset = (
            xor_offset_by_output_index[output_index]
            ^ xor_offset_by_output_index[input_index]
        )
        if second_input_index != output_index or second_offset != 0:
            raise ValueError("Founder-couple coordinate map must be an involution.")


def _validate_disjoint_groups(
    groups: tuple[FounderCoupleQuotientGroup, ...],
) -> None:
    """Reject overlapping symmetries that need a larger joint group action."""

    used_indices: set[int] = set()
    for group in groups:
        affected_indices = set(group.affected_input_bit_indices)
        if used_indices.intersection(affected_indices):
            raise ValueError(
                "Founder-couple quotient groups must affect disjoint coordinates."
            )
        used_indices.update(affected_indices)


def _validate_binary_bits(
    bits: Sequence[int],
    expected_length: int,
    label: str,
) -> tuple[int, ...]:
    """Validate and freeze one binary coordinate vector."""

    validated_bits = tuple(bits)
    if len(validated_bits) != expected_length:
        raise ValueError(f"{label} has the wrong number of bits.")
    if any(bit not in (0, 1) for bit in validated_bits):
        raise ValueError(f"{label} must contain only zero and one.")
    return validated_bits
