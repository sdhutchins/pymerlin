"""Bounded state audit for an exact fused transition-and-condition DAG.

The ordinary tree engine first transitions one inheritance tree and then
multiplies it by the next marker emission. That materialized intermediate can
be much larger than either input. A fused evaluator can instead memoize pairs
of source and emission DAG nodes at each inheritance-bit depth.

This module counts those prospective memoization keys without calculating
likelihoods. It also retains binary founder-orientation contexts across the
relative bits coupled by the exact founder quotient. A completed audit is an
implementation-specific work estimate. An audit that reaches its state budget
is only a lower bound and must not be presented as evidence of feasibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .founder_couple_quotient import (
    FounderCoupleQuotient,
    reduce_founder_couple_tree,
)
from .founder_orientation_quotient import FounderOrientationQuotient
from .inheritance_tree import (
    InheritanceTree,
    LeafNode,
    SplitNode,
    TreeNode,
    ZeroNode,
    _children_or_constant,
    _inheritance_recursion_budget,
    _ScaledNode,
)

_FounderContext = tuple[tuple[int, int], ...]
_AuditState = tuple[int, int, _FounderContext]


@dataclass(frozen=True)
class FounderCoupleKeyGroupAudit:
    """Structural key effect for one exact founder-couple quotient group."""

    founder_ids: tuple[str, str]
    representative_input_bit_index: int
    representative_is_active: bool
    active_input_bit_indices: tuple[int, ...]
    active_reduced_bit_indices: tuple[int, ...]
    context_first_bit_index: int | None
    context_last_bit_index: int | None
    requires_persistent_context: bool


@dataclass(frozen=True)
class FounderCoupleKeyAudit:
    """Compare DAG branching removed by a quotient with context it adds."""

    input_bit_count: int
    reduced_bit_count: int
    input_active_bit_count: int
    reduced_active_bit_count: int
    removed_active_bit_count: int
    persistent_context_group_count: int
    maximum_open_context_count: int
    maximum_context_lane_count: int
    groups: tuple[FounderCoupleKeyGroupAudit, ...]

    @property
    def removed_bit_count(self) -> int:
        """Return the number of coordinates hidden by the quotient."""

        return self.input_bit_count - self.reduced_bit_count


@dataclass(frozen=True)
class PairedDagTransitionAudit:
    """Bounded structural workload for one fused exact interval."""

    bit_count: int
    active_bit_count: int
    recombination_fraction: float
    founder_context_group_count: int
    founder_couple_context_group_count: int
    maximum_open_founder_contexts: int
    examined_unique_subproblem_count: int
    transition_arc_count: int
    maximum_frontier_state_count: int
    deepest_bit_index_reached: int
    maximum_unique_subproblems: int
    complete: bool

    @property
    def count_is_lower_bound(self) -> bool:
        """Return whether the configured budget truncated the state count."""

        return not self.complete


def audit_founder_couple_key_effect(
    current_tree: InheritanceTree,
    next_emission_tree: InheritanceTree,
    recombination_fraction: float,
    quotient: FounderCoupleQuotient,
) -> FounderCoupleKeyAudit:
    """Compare removed DAG splits with the exact orbit context requirement.

    The input trees must already use founder-orientation coordinates. The
    quotient trees are built internally so both active-bit sets come from the
    actual compressed DAGs. A group needs persistent binary lanes only when a
    nontrivial recombination transition couples at least two active reduced
    coordinates. One active coordinate can be summed locally without adding a
    value carried across bit levels.
    """

    _validate_tree_depths(current_tree, next_emission_tree)
    theta = _validate_recombination_fraction(recombination_fraction)
    if current_tree.bit_count != quotient.input_bit_count:
        raise ValueError(
            "Founder-couple quotient and input trees must use the same bits."
        )

    reduced_current_tree = reduce_founder_couple_tree(current_tree, quotient)
    reduced_next_tree = reduce_founder_couple_tree(next_emission_tree, quotient)
    input_active_indices = _active_split_bit_indices(
        current_tree,
        next_emission_tree,
    )
    reduced_active_indices = _active_split_bit_indices(
        reduced_current_tree,
        reduced_next_tree,
    )

    group_audits = tuple(
        _audit_founder_couple_key_group(
            group.founder_ids,
            group.representative_input_bit_index,
            group.affected_input_bit_indices,
            input_active_indices,
            reduced_active_indices,
            quotient.reduced_bit_index_by_input_bit,
            theta,
        )
        for group in quotient.groups
    )
    context_intervals = tuple(
        (group.context_first_bit_index, group.context_last_bit_index)
        for group in group_audits
        if group.requires_persistent_context
        and group.context_first_bit_index is not None
        and group.context_last_bit_index is not None
    )
    maximum_open_context_count = _maximum_overlapping_intervals(
        context_intervals
    )

    return FounderCoupleKeyAudit(
        input_bit_count=current_tree.bit_count,
        reduced_bit_count=quotient.reduced_bit_count,
        input_active_bit_count=len(input_active_indices),
        reduced_active_bit_count=len(reduced_active_indices),
        removed_active_bit_count=len(input_active_indices)
        - len(reduced_active_indices),
        persistent_context_group_count=len(context_intervals),
        maximum_open_context_count=maximum_open_context_count,
        maximum_context_lane_count=2**maximum_open_context_count,
        groups=group_audits,
    )


def audit_paired_dag_transition(
    current_tree: InheritanceTree,
    next_emission_tree: InheritanceTree,
    recombination_fraction: float,
    *,
    founder_quotient: FounderOrientationQuotient | None = None,
    founder_couple_quotient: FounderCoupleQuotient | None = None,
    maximum_unique_subproblems: int = 1_000_000,
) -> PairedDagTransitionAudit:
    """Count memoized node-pair states without evaluating likelihood values.

    For a nonzero recombination fraction below one half, a split source and a
    split target can require all four source-target child pairs. Identical
    child pairs merge in the next frontier. Founder-relative transition
    factors are mixtures over one binary target orientation. Their contexts
    remain in the memoization key from the first through the last active
    relative bit for that founder.

    Terminal pairs stop early because a transition preserves constant source
    functions and zero annihilates a conditioned result. The audit counts a
    specific top-down fused recursion. It does not prove that no better exact
    elimination order or algebraic representation exists.
    """

    _validate_tree_depths(current_tree, next_emission_tree)
    theta = _validate_recombination_fraction(recombination_fraction)
    _validate_positive_integer(
        maximum_unique_subproblems,
        "Maximum unique subproblems",
    )
    founder_interaction_groups, founder_couple_interaction_groups = (
        _transition_interaction_groups(
            founder_quotient,
            founder_couple_quotient,
            current_tree.bit_count,
        )
    )

    active_bit_indices = _active_split_bit_indices(
        current_tree,
        next_emission_tree,
    )
    founder_context_intervals = _active_context_intervals(
        founder_interaction_groups,
        active_bit_indices,
    )
    founder_couple_context_intervals = _active_context_intervals(
        founder_couple_interaction_groups,
        active_bit_indices,
    )
    context_intervals = (
        founder_context_intervals + founder_couple_context_intervals
    )
    opening_groups_by_bit = _group_indices_by_boundary(
        context_intervals,
        boundary_index=0,
    )
    closing_groups_by_bit = _group_indices_by_boundary(
        context_intervals,
        boundary_index=1,
    )

    source_nodes: dict[int, TreeNode] = {id(current_tree.root): current_tree.root}
    emission_nodes: dict[int, TreeNode] = {
        id(next_emission_tree.root): next_emission_tree.root
    }
    source_child_cache: dict[int, tuple[int, int]] = {}
    emission_child_cache: dict[int, tuple[int, int]] = {}
    frontier: set[_AuditState] = {
        (id(current_tree.root), id(next_emission_tree.root), ())
    }
    examined_state_count = 0
    transition_arc_count = 0
    maximum_frontier_state_count = 0
    maximum_open_context_count = 0
    bit_index = 0

    with _inheritance_recursion_budget(current_tree.bit_count):
        while frontier:
            terminal_frontier = {
                state
                for state in frontier
                if _state_is_terminal(
                    source_nodes[state[0]],
                    emission_nodes[state[1]],
                )
            }
            continuing_frontier = frontier - terminal_frontier
            opened_continuing_frontier = _open_founder_contexts(
                continuing_frontier,
                opening_groups_by_bit.get(bit_index, ()),
                theta,
                (
                    maximum_unique_subproblems
                    - examined_state_count
                    - len(terminal_frontier)
                ),
            )
            if opened_continuing_frontier is None:
                return _truncated_audit(
                    current_tree.bit_count,
                    len(active_bit_indices),
                    theta,
                    len(founder_context_intervals),
                    len(founder_couple_context_intervals),
                    maximum_open_context_count,
                    transition_arc_count,
                    maximum_frontier_state_count,
                    bit_index,
                    maximum_unique_subproblems,
                )
            opened_frontier = terminal_frontier | opened_continuing_frontier

            maximum_frontier_state_count = max(
                maximum_frontier_state_count,
                len(opened_frontier),
            )
            maximum_open_context_count = max(
                maximum_open_context_count,
                max(
                    (len(context) for _, _, context in opened_frontier),
                    default=0,
                ),
            )
            if examined_state_count + len(opened_frontier) > maximum_unique_subproblems:
                return _truncated_audit(
                    current_tree.bit_count,
                    len(active_bit_indices),
                    theta,
                    len(founder_context_intervals),
                    len(founder_couple_context_intervals),
                    maximum_open_context_count,
                    transition_arc_count,
                    maximum_frontier_state_count,
                    bit_index,
                    maximum_unique_subproblems,
                )
            examined_state_count += len(opened_frontier)

            next_frontier: set[_AuditState] = set()
            for source_node_id, emission_node_id, context in opened_frontier:
                source_node = source_nodes[source_node_id]
                emission_node = emission_nodes[emission_node_id]
                if _state_is_terminal(source_node, emission_node):
                    continue
                if bit_index >= current_tree.bit_count:
                    raise ValueError(
                        "Paired DAG contains a nonterminal below its declared "
                        "inheritance-bit depth."
                    )

                source_children = _child_node_ids(
                    source_node,
                    source_nodes,
                    source_child_cache,
                )
                emission_children = _child_node_ids(
                    emission_node,
                    emission_nodes,
                    emission_child_cache,
                )
                closed_context = _close_founder_contexts(
                    context,
                    closing_groups_by_bit.get(bit_index, ()),
                )
                child_pairs = _nonzero_transition_child_pairs(
                    source_children,
                    emission_children,
                    theta,
                )
                for child_source_id, child_emission_id in child_pairs:
                    if _is_zero(source_nodes[child_source_id]) or _is_zero(
                        emission_nodes[child_emission_id]
                    ):
                        continue
                    transition_arc_count += 1
                    next_frontier.add(
                        (
                            child_source_id,
                            child_emission_id,
                            closed_context,
                        )
                    )
                    if (
                        examined_state_count + len(next_frontier)
                        > maximum_unique_subproblems
                    ):
                        return _truncated_audit(
                            current_tree.bit_count,
                            len(active_bit_indices),
                            theta,
                            len(founder_context_intervals),
                            len(founder_couple_context_intervals),
                            maximum_open_context_count,
                            transition_arc_count,
                            max(
                                maximum_frontier_state_count,
                                len(next_frontier),
                            ),
                            bit_index + 1,
                            maximum_unique_subproblems,
                        )

            frontier = next_frontier
            bit_index += 1

    return PairedDagTransitionAudit(
        bit_count=current_tree.bit_count,
        active_bit_count=len(active_bit_indices),
        recombination_fraction=theta,
        founder_context_group_count=len(founder_context_intervals),
        founder_couple_context_group_count=len(
            founder_couple_context_intervals
        ),
        maximum_open_founder_contexts=maximum_open_context_count,
        examined_unique_subproblem_count=examined_state_count,
        transition_arc_count=transition_arc_count,
        maximum_frontier_state_count=maximum_frontier_state_count,
        deepest_bit_index_reached=max(0, bit_index - 1),
        maximum_unique_subproblems=maximum_unique_subproblems,
        complete=True,
    )


def format_paired_dag_transition_audit(
    audit: PairedDagTransitionAudit,
) -> str:
    """Format a deterministic machine-readable paired-DAG report."""

    return "\n".join(
        (
            f"bit_count\t{audit.bit_count}",
            f"active_bits\t{audit.active_bit_count}",
            f"recombination_fraction\t{audit.recombination_fraction:.17g}",
            (f"founder_context_groups\t{audit.founder_context_group_count}"),
            (
                "founder_couple_context_groups\t"
                f"{audit.founder_couple_context_group_count}"
            ),
            (f"maximum_open_founder_contexts\t{audit.maximum_open_founder_contexts}"),
            (f"examined_unique_subproblems\t{audit.examined_unique_subproblem_count}"),
            f"transition_arcs\t{audit.transition_arc_count}",
            (f"maximum_frontier_states\t{audit.maximum_frontier_state_count}"),
            f"deepest_bit_index\t{audit.deepest_bit_index_reached}",
            f"state_budget\t{audit.maximum_unique_subproblems}",
            f"complete\t{str(audit.complete).lower()}",
            f"count_is_lower_bound\t{str(audit.count_is_lower_bound).lower()}",
        )
    )


def format_founder_couple_key_audit(audit: FounderCoupleKeyAudit) -> str:
    """Format a deterministic structural comparison for benchmark output."""

    context_spans = ";".join(
        (
            f"{','.join(group.founder_ids)}:"
            f"{group.context_first_bit_index}-{group.context_last_bit_index}"
        )
        for group in audit.groups
        if group.requires_persistent_context
    )
    representative_activity = ";".join(
        (
            f"{','.join(group.founder_ids)}:"
            f"{str(group.representative_is_active).lower()}"
        )
        for group in audit.groups
    )
    return "\n".join(
        (
            f"founder_couple_input_bits\t{audit.input_bit_count}",
            f"founder_couple_reduced_bits\t{audit.reduced_bit_count}",
            f"founder_couple_removed_bits\t{audit.removed_bit_count}",
            (
                "founder_couple_input_active_bits\t"
                f"{audit.input_active_bit_count}"
            ),
            (
                "founder_couple_reduced_active_bits\t"
                f"{audit.reduced_active_bit_count}"
            ),
            (
                "founder_couple_removed_active_bits\t"
                f"{audit.removed_active_bit_count}"
            ),
            (
                "founder_couple_persistent_context_groups\t"
                f"{audit.persistent_context_group_count}"
            ),
            (
                "founder_couple_maximum_open_contexts\t"
                f"{audit.maximum_open_context_count}"
            ),
            (
                "founder_couple_maximum_context_lanes\t"
                f"{audit.maximum_context_lane_count}"
            ),
            (
                "founder_couple_representatives_active\t"
                f"{representative_activity or 'none'}"
            ),
            f"founder_couple_context_spans\t{context_spans or 'none'}",
        )
    )


def _active_split_bit_indices(
    first_tree: InheritanceTree,
    second_tree: InheritanceTree,
) -> frozenset[int]:
    """Return bit levels where either compressed DAG has a split node."""

    return _tree_split_bit_indices(first_tree).union(
        _tree_split_bit_indices(second_tree)
    )


def _tree_split_bit_indices(tree: InheritanceTree) -> frozenset[int]:
    """Return split levels from one tree with identity-aware DAG traversal."""

    active_indices: set[int] = set()
    visited: set[tuple[int, int]] = set()

    def visit(node: TreeNode, bit_index: int) -> None:
        cache_key = (id(node), bit_index)
        if cache_key in visited:
            return
        visited.add(cache_key)
        if isinstance(node, (ZeroNode, LeafNode)):
            return
        if isinstance(node, _ScaledNode):
            visit(node.child, bit_index)
            return
        if isinstance(node, SplitNode):
            active_indices.add(bit_index)
        zero_child, one_child = _children_or_constant(node)
        visit(zero_child, bit_index + 1)
        if one_child is not zero_child:
            visit(one_child, bit_index + 1)

    with _inheritance_recursion_budget(tree.bit_count):
        visit(tree.root, 0)
    return frozenset(active_indices)


def _audit_founder_couple_key_group(
    founder_ids: tuple[str, str],
    representative_input_bit_index: int,
    affected_input_bit_indices: tuple[int, ...],
    input_active_indices: frozenset[int],
    reduced_active_indices: frozenset[int],
    reduced_bit_index_by_input_bit: tuple[int | None, ...],
    theta: float,
) -> FounderCoupleKeyGroupAudit:
    """Measure branching and context for one projected involution."""

    active_input_bit_indices = tuple(
        bit_index
        for bit_index in affected_input_bit_indices
        if bit_index in input_active_indices
    )
    active_reduced_bit_indices = tuple(
        reduced_bit_index
        for input_bit_index in affected_input_bit_indices
        if (
            reduced_bit_index := reduced_bit_index_by_input_bit[
                input_bit_index
            ]
        )
        is not None
        and reduced_bit_index in reduced_active_indices
    )
    requires_persistent_context = (
        0.0 < theta < 0.5 and len(active_reduced_bit_indices) >= 2
    )
    context_first_bit_index = (
        min(active_reduced_bit_indices)
        if requires_persistent_context
        else None
    )
    context_last_bit_index = (
        max(active_reduced_bit_indices)
        if requires_persistent_context
        else None
    )
    return FounderCoupleKeyGroupAudit(
        founder_ids=founder_ids,
        representative_input_bit_index=representative_input_bit_index,
        representative_is_active=(
            representative_input_bit_index in input_active_indices
        ),
        active_input_bit_indices=active_input_bit_indices,
        active_reduced_bit_indices=active_reduced_bit_indices,
        context_first_bit_index=context_first_bit_index,
        context_last_bit_index=context_last_bit_index,
        requires_persistent_context=requires_persistent_context,
    )


def _maximum_overlapping_intervals(
    intervals: tuple[tuple[int, int], ...],
) -> int:
    """Return the largest number of inclusive intervals open at one bit."""

    if not intervals:
        return 0
    boundaries = sorted(
        {
            bit_index
            for first_bit_index, last_bit_index in intervals
            for bit_index in (first_bit_index, last_bit_index)
        }
    )
    return max(
        sum(
            first_bit_index <= boundary <= last_bit_index
            for first_bit_index, last_bit_index in intervals
        )
        for boundary in boundaries
    )


def _transition_interaction_groups(
    founder_quotient: FounderOrientationQuotient | None,
    founder_couple_quotient: FounderCoupleQuotient | None,
    tree_bit_count: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Validate quotient composition and return both transition group types."""

    if founder_couple_quotient is not None:
        if founder_quotient is None:
            raise ValueError(
                "Founder-couple quotient requires a founder-orientation quotient."
            )
        if (
            founder_couple_quotient.input_bit_count
            != founder_quotient.reduced_bit_count
        ):
            raise ValueError("Founder quotients use incompatible coordinate counts.")
        if founder_couple_quotient.reduced_bit_count != tree_bit_count:
            raise ValueError(
                "Founder-couple quotient and inheritance trees must use the "
                "same reduced inheritance bits."
            )
        founder_groups = founder_couple_quotient.remap_interaction_groups(
            tuple(
                group.reduced_member_bit_indices
                for group in founder_quotient.groups
                if group.reduced_member_bit_indices
            )
        )
        return (
            founder_groups,
            founder_couple_quotient.transition_interaction_groups,
        )

    if (
        founder_quotient is not None
        and founder_quotient.reduced_bit_count != tree_bit_count
    ):
        raise ValueError(
            "Founder quotient and inheritance trees must use the same "
            "reduced inheritance bits."
        )
    founder_groups = (
        tuple(
            group.reduced_member_bit_indices
            for group in founder_quotient.groups
            if group.reduced_member_bit_indices
        )
        if founder_quotient is not None
        else ()
    )
    return founder_groups, ()


def _active_context_intervals(
    interaction_groups: tuple[tuple[int, ...], ...],
    active_bit_indices: frozenset[int],
) -> tuple[tuple[int, int], ...]:
    """Return first and last active coordinates for each transition group."""

    intervals = []
    for interaction_group in interaction_groups:
        active_group_indices = tuple(
            bit_index
            for bit_index in interaction_group
            if bit_index in active_bit_indices
        )
        if active_group_indices:
            intervals.append((min(active_group_indices), max(active_group_indices)))
    return tuple(intervals)


def _group_indices_by_boundary(
    intervals: tuple[tuple[int, int], ...],
    boundary_index: int,
) -> dict[int, tuple[int, ...]]:
    """Index founder groups by either interval boundary."""

    groups_by_bit: dict[int, list[int]] = {}
    for group_index, interval in enumerate(intervals):
        groups_by_bit.setdefault(interval[boundary_index], []).append(group_index)
    return {
        bit_index: tuple(group_indices)
        for bit_index, group_indices in groups_by_bit.items()
    }


def _open_founder_contexts(
    frontier: set[_AuditState],
    opening_group_indices: tuple[int, ...],
    theta: float,
    remaining_state_budget: int,
) -> set[_AuditState] | None:
    """Expand nonzero latent founder orientations at one bit boundary."""

    if not opening_group_indices:
        return frontier
    orientation_values = (0,) if theta == 0.0 else (0, 1)
    opened_frontier = frontier
    for group_index in opening_group_indices:
        expanded_frontier: set[_AuditState] = set()
        for source_node_id, emission_node_id, context in opened_frontier:
            for orientation_value in orientation_values:
                expanded_frontier.add(
                    (
                        source_node_id,
                        emission_node_id,
                        (*context, (group_index, orientation_value)),
                    )
                )
                if len(expanded_frontier) > remaining_state_budget:
                    return None
        opened_frontier = expanded_frontier
    return opened_frontier


def _close_founder_contexts(
    context: _FounderContext,
    closing_group_indices: tuple[int, ...],
) -> _FounderContext:
    """Remove orientations after their last active relative coordinate."""

    if not closing_group_indices:
        return context
    closing_groups = frozenset(closing_group_indices)
    return tuple(
        context_entry
        for context_entry in context
        if context_entry[0] not in closing_groups
    )


def _child_node_ids(
    node: TreeNode,
    nodes_by_id: dict[int, TreeNode],
    child_cache: dict[int, tuple[int, int]],
) -> tuple[int, int]:
    """Return stable child identities, treating terminal nodes as constants."""

    node_id = id(node)
    cached_children = child_cache.get(node_id)
    if cached_children is not None:
        return cached_children
    zero_child, one_child = _children_or_constant(node)
    nodes_by_id[id(zero_child)] = zero_child
    nodes_by_id[id(one_child)] = one_child
    child_ids = (id(zero_child), id(one_child))
    child_cache[node_id] = child_ids
    return child_ids


def _nonzero_transition_child_pairs(
    source_children: tuple[int, int],
    emission_children: tuple[int, int],
    theta: float,
) -> tuple[tuple[int, int], ...]:
    """Return unique child pairs with nonzero current-bit transition weight."""

    if theta == 0.0:
        candidate_pairs = (
            (source_children[0], emission_children[0]),
            (source_children[1], emission_children[1]),
        )
    else:
        candidate_pairs = tuple(
            (source_children[source_value], emission_children[target_value])
            for target_value in (0, 1)
            for source_value in (0, 1)
        )
    return tuple(dict.fromkeys(candidate_pairs))


def _state_is_terminal(
    source_node: TreeNode,
    emission_node: TreeNode,
) -> bool:
    """Return whether a paired recursion can stop without another bit."""

    if _is_zero(source_node) or _is_zero(emission_node):
        return True
    return _is_terminal(source_node) and _is_terminal(emission_node)


def _is_zero(node: TreeNode) -> bool:
    """Return whether a possibly scaled node is identically zero."""

    while isinstance(node, _ScaledNode):
        node = node.child
    return isinstance(node, ZeroNode)


def _is_terminal(node: TreeNode) -> bool:
    """Return whether a possibly scaled node is constant over every suffix."""

    while isinstance(node, _ScaledNode):
        node = node.child
    return isinstance(node, (ZeroNode, LeafNode))


def _truncated_audit(
    bit_count: int,
    active_bit_count: int,
    theta: float,
    founder_context_group_count: int,
    founder_couple_context_group_count: int,
    maximum_open_founder_contexts: int,
    transition_arc_count: int,
    maximum_frontier_state_count: int,
    deepest_bit_index_reached: int,
    maximum_unique_subproblems: int,
) -> PairedDagTransitionAudit:
    """Return a deterministic lower-bound report at the configured cap."""

    return PairedDagTransitionAudit(
        bit_count=bit_count,
        active_bit_count=active_bit_count,
        recombination_fraction=theta,
        founder_context_group_count=founder_context_group_count,
        founder_couple_context_group_count=(
            founder_couple_context_group_count
        ),
        maximum_open_founder_contexts=maximum_open_founder_contexts,
        examined_unique_subproblem_count=maximum_unique_subproblems,
        transition_arc_count=transition_arc_count,
        maximum_frontier_state_count=maximum_frontier_state_count,
        deepest_bit_index_reached=deepest_bit_index_reached,
        maximum_unique_subproblems=maximum_unique_subproblems,
        complete=False,
    )


def _validate_tree_depths(
    current_tree: InheritanceTree,
    next_emission_tree: InheritanceTree,
) -> None:
    """Require paired trees over the same ordered inheritance coordinates."""

    if current_tree.bit_count != next_emission_tree.bit_count:
        raise ValueError("Paired DAG trees must use the same ordered inheritance bits.")


def _validate_recombination_fraction(value: float) -> float:
    """Validate a finite autosomal recombination fraction."""

    theta = float(value)
    if not math.isfinite(theta) or not 0.0 <= theta <= 0.5:
        raise ValueError("Recombination fraction must be finite and between 0 and 0.5.")
    return theta


def _validate_positive_integer(value: int, label: str) -> None:
    """Reject booleans and nonpositive state budgets."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
