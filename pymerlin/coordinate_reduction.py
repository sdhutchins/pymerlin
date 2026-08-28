"""Exact marker-specific reductions of inheritance-vector coordinates.

Marker likelihoods can force an inheritance bit to one value or force two
bits to be equal or opposite. Removing those redundant coordinates reduces
the state space without changing likelihood values or recombination weights.
This module provides a bounded Python reference implementation. It is intended
to establish correctness before the same operations are moved to a native
backend for large pedigrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

from .inheritance_tree import (
    InheritanceTree,
    LeafNode,
    SharedNode,
    TreeNode,
    ZeroNode,
    _inheritance_recursion_budget,
)


class CoordinateTransitionBudgetExceeded(RuntimeError):
    """Raised before a dense reference calculation exceeds its state budget."""


@dataclass(frozen=True)
class MarkerCoordinateMap:
    """Map reduced marker coordinates to complete inheritance vectors.

    ``coordinate_by_bit`` is ``None`` for a fixed full-space bit. Otherwise,
    the full bit equals its reduced coordinate XOR ``xor_offset_by_bit``.
    Multiple full bits may refer to one reduced coordinate when their values
    are constrained to be equal or opposite on the marker's nonzero support.
    """

    full_bit_count: int
    coordinate_by_bit: tuple[int | None, ...]
    xor_offset_by_bit: tuple[int, ...]
    representative_bit_by_coordinate: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.full_bit_count < 0:
            raise ValueError("Full inheritance-vector bit count cannot be negative.")
        if len(self.coordinate_by_bit) != self.full_bit_count:
            raise ValueError("Coordinate mapping must cover every full-space bit.")
        if len(self.xor_offset_by_bit) != self.full_bit_count:
            raise ValueError("XOR offsets must cover every full-space bit.")
        if any(offset not in (0, 1) for offset in self.xor_offset_by_bit):
            raise ValueError("Coordinate XOR offsets must be zero or one.")

        expected_coordinates = set(range(self.reduced_bit_count))
        observed_coordinates = {
            coordinate
            for coordinate in self.coordinate_by_bit
            if coordinate is not None
        }
        if observed_coordinates != expected_coordinates:
            raise ValueError("Reduced coordinate indices must be contiguous.")
        if len(self.representative_bit_by_coordinate) != self.reduced_bit_count:
            raise ValueError("Every reduced coordinate requires one representative.")

        for coordinate, bit_index in enumerate(self.representative_bit_by_coordinate):
            if not 0 <= bit_index < self.full_bit_count:
                raise ValueError("Representative bit index is outside the full space.")
            if self.coordinate_by_bit[bit_index] != coordinate:
                raise ValueError("Representative bit uses the wrong coordinate.")
            if self.xor_offset_by_bit[bit_index] != 0:
                raise ValueError("Representative bits must have zero XOR offsets.")

    @property
    def reduced_bit_count(self) -> int:
        """Return the number of variable marker coordinates."""

        coordinates = (
            coordinate
            for coordinate in self.coordinate_by_bit
            if coordinate is not None
        )
        return max(coordinates, default=-1) + 1

    @property
    def fixed_bit_count(self) -> int:
        """Return the number of coordinates fixed on the marker support."""

        return sum(coordinate is None for coordinate in self.coordinate_by_bit)

    @property
    def parity_reduced_bit_count(self) -> int:
        """Return variable full bits represented by another coordinate."""

        return self.full_bit_count - self.fixed_bit_count - self.reduced_bit_count

    def expand(self, reduced_bits: tuple[int, ...]) -> tuple[int, ...]:
        """Expand one reduced vector into its unique compatible full vector."""

        if len(reduced_bits) != self.reduced_bit_count:
            raise ValueError("Reduced-vector length does not match the coordinate map.")
        if any(bit not in (0, 1) for bit in reduced_bits):
            raise ValueError("Reduced inheritance bits must be zero or one.")

        return tuple(
            offset if coordinate is None else reduced_bits[coordinate] ^ offset
            for coordinate, offset in zip(
                self.coordinate_by_bit,
                self.xor_offset_by_bit,
            )
        )

    def project(self, full_bits: tuple[int, ...]) -> tuple[int, ...]:
        """Project one compatible full vector to marker coordinates."""

        if not self.is_compatible(full_bits):
            raise ValueError("Full inheritance vector violates the coordinate map.")
        return tuple(
            full_bits[bit_index] for bit_index in self.representative_bit_by_coordinate
        )

    def is_compatible(self, full_bits: tuple[int, ...]) -> bool:
        """Return whether a full vector satisfies every mapped restriction."""

        if len(full_bits) != self.full_bit_count:
            raise ValueError("Full-vector length does not match the coordinate map.")
        if any(bit not in (0, 1) for bit in full_bits):
            raise ValueError("Full inheritance bits must be zero or one.")

        for bit_index, (coordinate, offset) in enumerate(
            zip(self.coordinate_by_bit, self.xor_offset_by_bit)
        ):
            if coordinate is None:
                expected_bit = offset
            else:
                representative = self.representative_bit_by_coordinate[coordinate]
                expected_bit = full_bits[representative] ^ offset
            if full_bits[bit_index] != expected_bit:
                return False
        return True


@dataclass(frozen=True)
class ReducedInheritanceMessage:
    """An inheritance message stored in marker-specific coordinates."""

    coordinate_map: MarkerCoordinateMap
    tree: InheritanceTree

    def __post_init__(self) -> None:
        if self.tree.bit_count != self.coordinate_map.reduced_bit_count:
            raise ValueError(
                "Reduced tree depth does not match its marker coordinate map."
            )


def marker_coordinate_map(emission_tree: InheritanceTree) -> MarkerCoordinateMap:
    """Find exact fixed and pair-parity constraints on nonzero support.

    The compressed tree is traversed once per reachable node and depth. Support
    differences are accumulated as Python integer bitsets and row-reduced over
    GF(2). Equal basis-column signatures identify bits whose XOR is constant.
    General constraints involving three or more distinct signatures are not
    eliminated. Their incompatible reduced states retain zero likelihood.
    """

    with _inheritance_recursion_budget(emission_tree.bit_count):
        representatives = _support_representatives(emission_tree.root)
        root_key = (id(emission_tree.root), 0)
        representative = representatives[root_key]
        if representative is None:
            return _empty_support_map(emission_tree.bit_count)
        basis_rows = _support_difference_basis(
            emission_tree.root,
            emission_tree.bit_count,
            representatives,
        )

    signatures = _column_signatures(basis_rows, emission_tree.bit_count)
    coordinate_by_bit: list[int | None] = []
    xor_offset_by_bit: list[int] = []
    representative_by_signature: dict[int, tuple[int, int]] = {}
    representative_bits: list[int] = []

    for bit_index, signature in enumerate(signatures):
        representative_value = (representative >> bit_index) & 1
        if signature == 0:
            coordinate_by_bit.append(None)
            xor_offset_by_bit.append(representative_value)
            continue

        existing = representative_by_signature.get(signature)
        if existing is None:
            coordinate = len(representative_bits)
            representative_by_signature[signature] = (
                coordinate,
                representative_value,
            )
            representative_bits.append(bit_index)
            coordinate_by_bit.append(coordinate)
            xor_offset_by_bit.append(0)
            continue

        coordinate, coordinate_representative_value = existing
        coordinate_by_bit.append(coordinate)
        xor_offset_by_bit.append(representative_value ^ coordinate_representative_value)

    return MarkerCoordinateMap(
        full_bit_count=emission_tree.bit_count,
        coordinate_by_bit=tuple(coordinate_by_bit),
        xor_offset_by_bit=tuple(xor_offset_by_bit),
        representative_bit_by_coordinate=tuple(representative_bits),
    )


def reduce_inheritance_tree(
    tree: InheritanceTree,
    coordinate_map: MarkerCoordinateMap,
    *,
    state_limit: int = 1_000_000,
) -> ReducedInheritanceMessage:
    """Materialize a bounded dense reference tree in reduced coordinates."""

    if tree.bit_count != coordinate_map.full_bit_count:
        raise ValueError("Tree depth does not match the full coordinate space.")
    _validate_positive_state_limit(state_limit)
    state_count = 1 << coordinate_map.reduced_bit_count
    if state_count > state_limit:
        raise CoordinateTransitionBudgetExceeded(
            "Reduced tree requires "
            f"{state_count:,} states, above the {state_limit:,} state limit."
        )

    values = tuple(
        tree.value_at(coordinate_map.expand(reduced_bits))
        for reduced_bits in product(
            (0, 1),
            repeat=coordinate_map.reduced_bit_count,
        )
    )
    return ReducedInheritanceMessage(
        coordinate_map=coordinate_map,
        tree=InheritanceTree.from_dense(
            values,
            bit_count=coordinate_map.reduced_bit_count,
        ),
    )


def reference_state_pair_count(
    source_map: MarkerCoordinateMap,
    target_map: MarkerCoordinateMap,
) -> int:
    """Return the dense source-target work for the reference transition."""

    if source_map.full_bit_count != target_map.full_bit_count:
        raise ValueError("Source and target maps require the same full bit count.")
    return 1 << (source_map.reduced_bit_count + target_map.reduced_bit_count)


def exact_partial_transition(
    current_message: ReducedInheritanceMessage,
    next_emission: InheritanceTree,
    next_coordinate_map: MarkerCoordinateMap,
    recombination_fraction: float,
    *,
    state_pair_limit: int = 1_000_000,
) -> ReducedInheritanceMessage:
    """Transition and condition exactly using bounded reduced-state sums.

    This deliberately direct implementation sums the full independent-meiosis
    recombination kernel between compatible source and target vectors. It is a
    test oracle for a later partial-transform backend, not the scalable backend
    itself.
    """

    theta = float(recombination_fraction)
    if not math.isfinite(theta) or not 0.0 <= theta <= 0.5:
        raise ValueError("Recombination fraction must be finite and between 0 and 0.5.")
    _validate_positive_state_limit(state_pair_limit)

    source_map = current_message.coordinate_map
    if next_emission.bit_count != source_map.full_bit_count:
        raise ValueError("Next emission depth does not match the source space.")
    if next_coordinate_map.full_bit_count != source_map.full_bit_count:
        raise ValueError("Next coordinate map does not match the source space.")

    state_pair_count = reference_state_pair_count(
        source_map,
        next_coordinate_map,
    )
    if state_pair_count > state_pair_limit:
        raise CoordinateTransitionBudgetExceeded(
            "Exact reference transition requires "
            f"{state_pair_count:,} source-target pairs, above the "
            f"{state_pair_limit:,} pair limit."
        )

    source_states = tuple(
        (
            source_map.expand(reduced_bits),
            current_message.tree.value_at(reduced_bits),
        )
        for reduced_bits in product(
            (0, 1),
            repeat=source_map.reduced_bit_count,
        )
    )
    target_values: list[float] = []
    complement = 1.0 - theta

    for reduced_target in product(
        (0, 1),
        repeat=next_coordinate_map.reduced_bit_count,
    ):
        full_target = next_coordinate_map.expand(reduced_target)
        emission_value = next_emission.value_at(full_target)
        if emission_value == 0.0:
            target_values.append(0.0)
            continue

        transition_terms = []
        for full_source, source_value in source_states:
            if source_value == 0.0:
                continue
            changed_bit_count = sum(
                source_bit != target_bit
                for source_bit, target_bit in zip(
                    full_source,
                    full_target,
                )
            )
            unchanged_bit_count = source_map.full_bit_count - changed_bit_count
            transition_weight = (
                theta**changed_bit_count * complement**unchanged_bit_count
            )
            transition_terms.append(source_value * transition_weight)

        target_values.append(emission_value * math.fsum(transition_terms))

    return ReducedInheritanceMessage(
        coordinate_map=next_coordinate_map,
        tree=InheritanceTree.from_dense(
            target_values,
            bit_count=next_coordinate_map.reduced_bit_count,
        ),
    )


def _empty_support_map(bit_count: int) -> MarkerCoordinateMap:
    """Return a deterministic all-fixed map for an empty marker support."""

    return MarkerCoordinateMap(
        full_bit_count=bit_count,
        coordinate_by_bit=(None,) * bit_count,
        xor_offset_by_bit=(0,) * bit_count,
        representative_bit_by_coordinate=(),
    )


def _support_representatives(
    root: TreeNode,
) -> dict[tuple[int, int], int | None]:
    """Store one nonzero-support suffix for every reachable node and depth."""

    representatives: dict[tuple[int, int], int | None] = {}

    def visit(node: TreeNode, bit_index: int) -> int | None:
        key = (id(node), bit_index)
        if key in representatives:
            return representatives[key]

        if isinstance(node, ZeroNode):
            representative = None
        elif isinstance(node, LeafNode):
            representative = 0 if node.value != 0.0 else None
        elif isinstance(node, SharedNode):
            representative = visit(node.child, bit_index + 1)
        else:
            zero_representative = visit(node.zero_child, bit_index + 1)
            one_representative = visit(node.one_child, bit_index + 1)
            if zero_representative is not None:
                representative = zero_representative
            else:
                representative = (
                    None
                    if one_representative is None
                    else one_representative | (1 << bit_index)
                )

        representatives[key] = representative
        return representative

    visit(root, 0)
    return representatives


def _support_difference_basis(
    root: TreeNode,
    bit_count: int,
    representatives: dict[tuple[int, int], int | None],
) -> tuple[int, ...]:
    """Return an independent GF(2) basis for support-vector differences."""

    pivot_rows: dict[int, int] = {}
    visited: set[tuple[int, int]] = set()

    def add_generator(vector: int) -> None:
        while vector:
            pivot = vector.bit_length() - 1
            existing = pivot_rows.get(pivot)
            if existing is None:
                pivot_rows[pivot] = vector
                return
            vector ^= existing

    def visit(node: TreeNode, bit_index: int) -> None:
        key = (id(node), bit_index)
        if key in visited or representatives[key] is None:
            return
        visited.add(key)

        if isinstance(node, ZeroNode):
            return
        if isinstance(node, LeafNode):
            for suffix_bit_index in range(bit_index, bit_count):
                add_generator(1 << suffix_bit_index)
            return
        if isinstance(node, SharedNode):
            add_generator(1 << bit_index)
            visit(node.child, bit_index + 1)
            return

        zero_key = (id(node.zero_child), bit_index + 1)
        one_key = (id(node.one_child), bit_index + 1)
        zero_representative = representatives[zero_key]
        one_representative = representatives[one_key]
        if zero_representative is not None and one_representative is not None:
            add_generator(zero_representative ^ one_representative ^ (1 << bit_index))
        visit(node.zero_child, bit_index + 1)
        visit(node.one_child, bit_index + 1)

    visit(root, 0)
    return tuple(pivot_rows[pivot] for pivot in sorted(pivot_rows))


def _column_signatures(
    basis_rows: tuple[int, ...],
    bit_count: int,
) -> tuple[int, ...]:
    """Encode each full bit's values across the support-difference basis."""

    return tuple(
        sum(
            ((row >> bit_index) & 1) << row_index
            for row_index, row in enumerate(basis_rows)
        )
        for bit_index in range(bit_count)
    )


def _validate_positive_state_limit(state_limit: int) -> None:
    """Reject invalid limits before any state-space allocation."""

    if isinstance(state_limit, bool) or state_limit < 1:
        raise ValueError("State limit must be a positive integer.")
