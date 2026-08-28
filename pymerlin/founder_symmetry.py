"""Exact founder-allele orientation reductions for inheritance trees.

Founder allele labels are arbitrary. After fixing one transmission from a
founder to allele zero, the complementary orientation is obtained by flipping
every later transmission from that founder. MERLIN uses this symmetry to hide
one inheritance bit per informative founder and records the required flips.

PyMerlin currently retains the full public inheritance-vector coordinates.
The helpers below therefore rebuild the complementary branch from the
canonical branch instead of changing the multipoint transition state space.
"""

from __future__ import annotations

from dataclasses import dataclass

from .inheritance_tree import (
    LeafNode,
    SharedNode,
    TreeNode,
    ZeroNode,
    _ScaledNode,
    _combine_children,
    _scaled_node,
)
from .models import Family, Individual


@dataclass(frozen=True)
class FounderOrientationSymmetryPlan:
    """Founder flips indexed by the representative inheritance bit.

    ``None`` identifies an ordinary bit. A frozenset, including an empty one,
    identifies the first relevant transmission from one founder and lists the
    later transmissions that must be complemented for its allele-swapped
    branch.
    """

    descendant_flip_indices_by_bit: tuple[frozenset[int] | None, ...]

    def descendant_flip_indices(
        self,
        bit_index: int,
    ) -> frozenset[int] | None:
        """Return the founder flips for a representative bit, if present."""

        return self.descendant_flip_indices_by_bit[bit_index]


@dataclass(frozen=True)
class FounderCoupleSymmetry:
    """One exchangeable founder couple and its full-bit involution."""

    founder_ids: tuple[str, str]
    shared_child_ids: tuple[str, ...]
    representative_bit_index: int
    swapped_bit_pairs: tuple[tuple[int, int], ...]
    toggled_bit_indices: frozenset[int]


@dataclass(frozen=True)
class FounderCoupleSymmetryPlan:
    """Exact founder-couple symmetries that apply to one tree function."""

    symmetries: tuple[FounderCoupleSymmetry, ...]

    @property
    def representative_bit_indices(self) -> frozenset[int]:
        """Return grandchild bits fixed while canonical branches are scored."""

        return frozenset(
            symmetry.representative_bit_index
            for symmetry in self.symmetries
        )

    def for_ibd_pair(
        self,
        first_id: str,
        second_id: str,
    ) -> FounderCoupleSymmetryPlan:
        """Retain only symmetries that preserve one labeled IBD query."""

        pair_ids = {first_id, second_id}
        return FounderCoupleSymmetryPlan(
            symmetries=tuple(
                symmetry
                for symmetry in self.symmetries
                if len(pair_ids.intersection(symmetry.founder_ids)) != 1
            )
        )

    def for_affected_ids(
        self,
        affected_ids: tuple[str, ...],
    ) -> FounderCoupleSymmetryPlan:
        """Retain symmetries that preserve the labeled affected-person set."""

        affected_id_set = set(affected_ids)
        return FounderCoupleSymmetryPlan(
            symmetries=tuple(
                symmetry
                for symmetry in self.symmetries
                if (
                    (symmetry.founder_ids[0] in affected_id_set)
                    == (symmetry.founder_ids[1] in affected_id_set)
                )
            )
        )


def build_founder_orientation_symmetry_plan(
    family: Family,
    relevant_meiosis_indices: frozenset[int],
) -> FounderOrientationSymmetryPlan:
    """Choose one canonical transmission per relevant transmitting founder."""

    bit_count = len(family.meioses)
    invalid_indices = {
        bit_index
        for bit_index in relevant_meiosis_indices
        if not 0 <= bit_index < bit_count
    }
    if invalid_indices:
        raise ValueError(
            "Founder symmetry received inheritance-bit indices outside the "
            f"family: {sorted(invalid_indices)}."
        )

    founder_ids = {
        founder.individual_id for founder in family.founders
    }
    relevant_bits_by_founder: dict[str, list[int]] = {}
    for bit_index, meiosis in enumerate(family.meioses):
        if (
            bit_index in relevant_meiosis_indices
            and meiosis.parent_id in founder_ids
        ):
            relevant_bits_by_founder.setdefault(
                meiosis.parent_id,
                [],
            ).append(bit_index)

    flip_indices_by_bit: list[frozenset[int] | None] = [None] * bit_count
    for founder_bit_indices in relevant_bits_by_founder.values():
        representative_bit_index = founder_bit_indices[0]
        flip_indices_by_bit[representative_bit_index] = frozenset(
            founder_bit_indices[1:]
        )

    return FounderOrientationSymmetryPlan(
        descendant_flip_indices_by_bit=tuple(flip_indices_by_bit),
    )


def build_founder_couple_symmetry_plan(
    family: Family,
    relevant_meiosis_indices: frozenset[int],
) -> FounderCoupleSymmetryPlan:
    """Find MERLIN-compatible founder couples with relevant grandchildren."""

    founder_ids = {
        founder.individual_id for founder in family.founders
    }
    partners_by_founder: dict[str, set[str | None]] = {
        founder_id: set() for founder_id in founder_ids
    }
    for person in family.individuals:
        if person.is_founder:
            continue
        father_id = person.father_id
        mother_id = person.mother_id
        if father_id in founder_ids:
            partners_by_founder[father_id].add(
                mother_id if mother_id in founder_ids else None
            )
        if mother_id in founder_ids:
            partners_by_founder[mother_id].add(
                father_id if father_id in founder_ids else None
            )

    people_by_id = family.by_id
    meiosis_index_by_parent_child = {
        (meiosis.parent_id, meiosis.child_id): bit_index
        for bit_index, meiosis in enumerate(family.meioses)
    }
    seen_founder_ids: set[str] = set()
    symmetries: list[FounderCoupleSymmetry] = []
    for first_founder in family.founders:
        first_id = first_founder.individual_id
        if first_id in seen_founder_ids:
            continue
        partner_ids = partners_by_founder[first_id]
        if len(partner_ids) != 1:
            continue
        second_id = next(iter(partner_ids))
        if second_id is None or second_id in seen_founder_ids:
            continue
        if partners_by_founder.get(second_id) != {first_id}:
            continue

        second_founder = people_by_id[second_id]
        if not _founders_are_effectively_identical(
            first_founder,
            second_founder,
        ):
            continue

        shared_child_ids = tuple(
            person.individual_id
            for person in family.individuals
            if not person.is_founder
            and {person.father_id, person.mother_id} == {first_id, second_id}
        )
        swapped_bit_pairs = tuple(
            (
                meiosis_index_by_parent_child[(first_id, child_id)],
                meiosis_index_by_parent_child[(second_id, child_id)],
            )
            for child_id in shared_child_ids
        )
        toggled_bit_indices = frozenset(
            bit_index
            for bit_index, meiosis in enumerate(family.meioses)
            if meiosis.parent_id in shared_child_ids
        )
        relevant_toggled_indices = (
            toggled_bit_indices.intersection(relevant_meiosis_indices)
        )
        if not relevant_toggled_indices:
            continue

        symmetries.append(
            FounderCoupleSymmetry(
                founder_ids=(first_id, second_id),
                shared_child_ids=shared_child_ids,
                representative_bit_index=min(relevant_toggled_indices),
                swapped_bit_pairs=swapped_bit_pairs,
                toggled_bit_indices=toggled_bit_indices,
            )
        )
        seen_founder_ids.update((first_id, second_id))

    return FounderCoupleSymmetryPlan(symmetries=tuple(symmetries))


def restore_founder_orientation_branch(
    canonical_zero_child: TreeNode,
    representative_bit_index: int,
    descendant_flip_indices: frozenset[int],
) -> TreeNode:
    """Restore a founder's allele-swapped branch in full bit coordinates."""

    if any(
        bit_index <= representative_bit_index
        for bit_index in descendant_flip_indices
    ):
        raise ValueError(
            "Founder-orientation flips must follow the representative bit."
        )

    if not descendant_flip_indices:
        return _combine_children(
            canonical_zero_child,
            canonical_zero_child,
        )

    allele_swapped_child = _flip_descendant_branches(
        canonical_zero_child,
        bit_index=representative_bit_index + 1,
        descendant_flip_indices=descendant_flip_indices,
        final_flip_index=max(descendant_flip_indices),
        memo={},
    )
    return _combine_children(
        canonical_zero_child,
        allele_swapped_child,
    )


def _flip_descendant_branches(
    node: TreeNode,
    bit_index: int,
    descendant_flip_indices: frozenset[int],
    final_flip_index: int,
    memo: dict[tuple[int, int], TreeNode],
) -> TreeNode:
    """Copy one subtree while complementing selected absolute bit levels."""

    cache_key = (id(node), bit_index)
    cached_node = memo.get(cache_key)
    if cached_node is not None:
        return cached_node

    if (
        bit_index > final_flip_index
        or isinstance(node, (ZeroNode, LeafNode))
    ):
        return node

    if isinstance(node, _ScaledNode):
        result = _scaled_node(
            _flip_descendant_branches(
                node.child,
                bit_index,
                descendant_flip_indices,
                final_flip_index,
                memo,
            ),
            node.factor,
        )
        memo[cache_key] = result
        return result

    if isinstance(node, SharedNode):
        transformed_child = _flip_descendant_branches(
            node.child,
            bit_index + 1,
            descendant_flip_indices,
            final_flip_index,
            memo,
        )
        result = _combine_children(transformed_child, transformed_child)
        memo[cache_key] = result
        return result

    transformed_zero_child = _flip_descendant_branches(
        node.zero_child,
        bit_index + 1,
        descendant_flip_indices,
        final_flip_index,
        memo,
    )
    transformed_one_child = _flip_descendant_branches(
        node.one_child,
        bit_index + 1,
        descendant_flip_indices,
        final_flip_index,
        memo,
    )
    if bit_index in descendant_flip_indices:
        transformed_zero_child, transformed_one_child = (
            transformed_one_child,
            transformed_zero_child,
        )
    result = _combine_children(
        transformed_zero_child,
        transformed_one_child,
    )
    memo[cache_key] = result
    return result


def restore_founder_couple_symmetry_branches(
    canonical_root: TreeNode,
    symmetry_plan: FounderCoupleSymmetryPlan,
) -> TreeNode:
    """Restore grandchild branches omitted for exchangeable founder couples."""

    root = canonical_root
    for symmetry in symmetry_plan.symmetries:
        exchanged_root = root
        for first_bit_index, second_bit_index in symmetry.swapped_bit_pairs:
            exchanged_root = _swap_tree_bit_inputs(
                exchanged_root,
                first_bit_index,
                second_bit_index,
            )
        exchanged_root = _toggle_tree_bit_inputs(
            exchanged_root,
            symmetry.toggled_bit_indices,
        )
        root = _select_tree_by_bit(
            root,
            exchanged_root,
            symmetry.representative_bit_index,
        )
    return root


def _founders_are_effectively_identical(
    first_founder: Individual,
    second_founder: Individual,
) -> bool:
    """Apply a conservative equivalent of MERLIN's founder-couple check."""

    if first_founder.phenotypes != second_founder.phenotypes:
        return False

    marker_names = set(first_founder.genotypes).union(
        second_founder.genotypes
    )
    return all(
        _unordered_genotype(
            first_founder.genotypes.get(marker_name, (None, None))
        )
        == _unordered_genotype(
            second_founder.genotypes.get(marker_name, (None, None))
        )
        for marker_name in marker_names
    )


def _unordered_genotype(
    genotype: tuple[str | None, str | None],
) -> tuple[str | None, str | None]:
    """Return a deterministic representation of one unphased genotype."""

    return tuple(
        sorted(
            genotype,
            key=lambda allele: (allele is not None, allele or ""),
        )
    )


def _swap_tree_bit_inputs(
    node: TreeNode,
    first_bit_index: int,
    second_bit_index: int,
) -> TreeNode:
    """Exchange two inheritance-bit inputs without expanding dense values."""

    if first_bit_index == second_bit_index:
        return node
    first_index, second_index = sorted(
        (first_bit_index, second_bit_index)
    )
    transformed_node = node
    for bit_index in range(first_index, second_index):
        transformed_node = _swap_adjacent_tree_bit_inputs(
            transformed_node,
            target_bit_index=bit_index,
            bit_index=0,
            memo={},
        )
    for bit_index in range(second_index - 2, first_index - 1, -1):
        transformed_node = _swap_adjacent_tree_bit_inputs(
            transformed_node,
            target_bit_index=bit_index,
            bit_index=0,
            memo={},
        )
    return transformed_node


def _swap_adjacent_tree_bit_inputs(
    node: TreeNode,
    target_bit_index: int,
    bit_index: int,
    memo: dict[tuple[int, int], TreeNode],
) -> TreeNode:
    """Exchange adjacent decision variables at one absolute tree depth."""

    cache_key = (id(node), bit_index)
    cached_node = memo.get(cache_key)
    if cached_node is not None:
        return cached_node

    if isinstance(node, (ZeroNode, LeafNode)):
        return node
    if isinstance(node, _ScaledNode):
        result = _scaled_node(
            _swap_adjacent_tree_bit_inputs(
                node.child,
                target_bit_index,
                bit_index,
                memo,
            ),
            node.factor,
        )
        memo[cache_key] = result
        return result
    if bit_index < target_bit_index:
        zero_child, one_child = _children_or_constant(node)
        result = _combine_children(
            _swap_adjacent_tree_bit_inputs(
                zero_child,
                target_bit_index,
                bit_index + 1,
                memo,
            ),
            _swap_adjacent_tree_bit_inputs(
                one_child,
                target_bit_index,
                bit_index + 1,
                memo,
            ),
        )
        memo[cache_key] = result
        return result

    first_zero, first_one = _children_or_constant(node)
    zero_zero, zero_one = _children_or_constant(first_zero)
    one_zero, one_one = _children_or_constant(first_one)
    result = _combine_children(
        _combine_children(zero_zero, one_zero),
        _combine_children(zero_one, one_one),
    )
    memo[cache_key] = result
    return result


def _toggle_tree_bit_inputs(
    node: TreeNode,
    target_bit_indices: frozenset[int],
    bit_index: int = 0,
    memo: dict[tuple[int, int], TreeNode] | None = None,
) -> TreeNode:
    """Complement multiple inheritance inputs in one memoized DAG pass."""

    if not target_bit_indices:
        return node
    if memo is None:
        memo = {}
    cache_key = (id(node), bit_index)
    cached_node = memo.get(cache_key)
    if cached_node is not None:
        return cached_node
    if isinstance(node, (ZeroNode, LeafNode)):
        return node
    if isinstance(node, _ScaledNode):
        result = _scaled_node(
            _toggle_tree_bit_inputs(
                node.child,
                target_bit_indices,
                bit_index,
                memo,
            ),
            node.factor,
        )
        memo[cache_key] = result
        return result

    zero_child, one_child = _children_or_constant(node)
    transformed_zero = _toggle_tree_bit_inputs(
        zero_child,
        target_bit_indices,
        bit_index + 1,
        memo,
    )
    transformed_one = _toggle_tree_bit_inputs(
        one_child,
        target_bit_indices,
        bit_index + 1,
        memo,
    )
    if bit_index in target_bit_indices:
        transformed_zero, transformed_one = (
            transformed_one,
            transformed_zero,
        )
    result = _combine_children(transformed_zero, transformed_one)
    memo[cache_key] = result
    return result


def _toggle_tree_bit_input(
    node: TreeNode,
    target_bit_index: int,
    bit_index: int = 0,
) -> TreeNode:
    """Complement one inheritance-bit input without changing tree depth."""

    return _toggle_tree_bit_inputs(
        node,
        frozenset((target_bit_index,)),
        bit_index,
    )


def _select_tree_by_bit(
    zero_tree: TreeNode,
    one_tree: TreeNode,
    target_bit_index: int,
    bit_index: int = 0,
    memo: dict[tuple[int, int, int], TreeNode] | None = None,
) -> TreeNode:
    """Use separate full-tree functions for one bit's zero and one states."""

    if memo is None:
        memo = {}
    cache_key = (id(zero_tree), id(one_tree), bit_index)
    cached_node = memo.get(cache_key)
    if cached_node is not None:
        return cached_node

    zero_zero, zero_one = _children_or_constant(zero_tree)
    one_zero, one_one = _children_or_constant(one_tree)
    if bit_index == target_bit_index:
        result = _combine_children(zero_zero, one_one)
    else:
        result = _combine_children(
            _select_tree_by_bit(
                zero_zero,
                one_zero,
                target_bit_index,
                bit_index + 1,
                memo,
            ),
            _select_tree_by_bit(
                zero_one,
                one_one,
                target_bit_index,
                bit_index + 1,
                memo,
            ),
        )
    memo[cache_key] = result
    return result


def _children_or_constant(node: TreeNode) -> tuple[TreeNode, TreeNode]:
    """Return both branches, repeating terminal and shared values as needed."""

    if isinstance(node, (ZeroNode, LeafNode)):
        return node, node
    if isinstance(node, _ScaledNode):
        zero_child, one_child = _children_or_constant(node.child)
        return (
            _scaled_node(zero_child, node.factor),
            _scaled_node(one_child, node.factor),
        )
    if isinstance(node, SharedNode):
        return node.child, node.child
    return node.zero_child, node.one_child
