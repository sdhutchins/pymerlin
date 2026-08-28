"""Conservative planning diagnostics for exact sparse transitions.

This module does not calculate multipoint probabilities. It inspects two
compressed inheritance trees and estimates whether conditioning can divide an
exact transition into bounded deterministic tasks. Estimates intentionally
prefer false declarations of infeasibility over unsupported declarations that
a large pedigree is tractable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .coordinate_reduction import MarkerCoordinateMap, marker_coordinate_map
from .inheritance_tree import (
    InheritanceTree,
    LeafNode,
    SharedNode,
    SplitNode,
    TreeNode,
    ZeroNode,
    _inheritance_recursion_budget,
    _ScaledNode,
)


@dataclass(frozen=True)
class SparseTransitionPlan:
    """Resource estimate for one exact transition-and-condition interval."""

    full_bit_count: int
    active_bit_indices: tuple[int, ...]
    initial_component_bit_indices: tuple[tuple[int, ...], ...]
    separator_bit_indices: tuple[int, ...]
    remaining_component_bit_indices: tuple[tuple[int, ...], ...]
    task_count_upper_bound: int
    task_count_is_exact: bool
    per_task_state_count_upper_bound: int
    per_task_node_work_upper_bound: int
    total_node_work_upper_bound: int
    peak_memory_bytes_per_task: int
    maximum_component_bits: int
    maximum_task_count: int
    maximum_total_node_work: int
    maximum_peak_memory_bytes: int

    @property
    def active_bit_count(self) -> int:
        """Return the number of bits that affect at least one input tree."""

        return len(self.active_bit_indices)

    @property
    def separator_width(self) -> int:
        """Return the number of inheritance bits assigned across tasks."""

        return len(self.separator_bit_indices)

    @property
    def largest_initial_component_bits(self) -> int:
        """Return the largest structural component before conditioning."""

        return max(
            map(len, self.initial_component_bit_indices),
            default=0,
        )

    @property
    def largest_remaining_component_bits(self) -> int:
        """Return the largest component after conditioning."""

        return max(
            map(len, self.remaining_component_bit_indices),
            default=0,
        )

    @property
    def is_feasible(self) -> bool:
        """Return whether every configured planning limit is satisfied."""

        return (
            self.largest_remaining_component_bits <= self.maximum_component_bits
            and self.task_count_upper_bound <= self.maximum_task_count
            and self.total_node_work_upper_bound <= self.maximum_total_node_work
            and self.peak_memory_bytes_per_task <= self.maximum_peak_memory_bytes
        )


def plan_sparse_transition(
    current_tree: InheritanceTree,
    next_emission_tree: InheritanceTree,
    *,
    maximum_component_bits: int = 24,
    maximum_task_count: int = 10_000,
    maximum_total_node_work: int = 1_000_000_000_000,
    maximum_peak_memory_bytes: int = 8 * 1024**3,
    bytes_per_value: int = 8,
    workspace_array_count: int = 2,
    support_projection_state_limit: int = 1_000_000,
    additional_interaction_groups: tuple[tuple[int, ...], ...] = (),
) -> SparseTransitionPlan:
    """Plan conservative conditioning without evaluating a transition.

    The structural graph joins each split bit to downstream split bits that
    occur in the same compressed subtree. This graph can retain edges that an
    algebraic factorization might remove. Such overconnection is intentional
    because the planner must not assume an unproved likelihood factorization.

    Separator bits are selected greedily until every remaining connected
    component fits ``maximum_component_bits``. Task count uses the next
    marker's exact fixed-bit and pair-parity restrictions. Higher-order support
    restrictions are ignored, making the task count an upper bound.

    Per-task work assumes an exact dense binary transform within each remaining
    component. Peak memory assumes sequential component processing with the
    requested number of float work arrays. These are planning estimates, not
    observed allocator measurements.
    """

    _validate_matching_tree_depths(current_tree, next_emission_tree)
    _validate_positive_limit(maximum_component_bits, "Maximum component bits")
    _validate_positive_limit(maximum_task_count, "Maximum task count")
    _validate_positive_limit(
        maximum_total_node_work,
        "Maximum total node work",
    )
    _validate_positive_limit(
        maximum_peak_memory_bytes,
        "Maximum peak memory",
    )
    _validate_positive_limit(bytes_per_value, "Bytes per value")
    _validate_positive_limit(workspace_array_count, "Workspace array count")
    _validate_positive_limit(
        support_projection_state_limit,
        "Support projection state limit",
    )

    adjacency_masks, active_mask = _combined_interaction_graph(
        current_tree,
        next_emission_tree,
    )
    adjacency_masks = _add_interaction_groups(
        adjacency_masks,
        current_tree.bit_count,
        additional_interaction_groups,
    )
    next_coordinate_map = marker_coordinate_map(next_emission_tree)
    initial_components = _connected_component_masks(
        adjacency_masks,
        active_mask,
    )
    separator_mask = _greedy_separator_mask(
        adjacency_masks,
        active_mask,
        maximum_component_bits,
        next_coordinate_map,
    )
    remaining_mask = active_mask & ~separator_mask
    remaining_components = _connected_component_masks(
        adjacency_masks,
        remaining_mask,
    )

    separator_bit_indices = _mask_bit_indices(separator_mask)
    task_count, task_count_is_exact = _separator_task_count(
        next_emission_tree,
        separator_bit_indices,
        next_coordinate_map,
        support_projection_state_limit,
    )
    component_sizes = tuple(component.bit_count() for component in remaining_components)
    per_task_state_count = max(
        1,
        sum(1 << component_size for component_size in component_sizes),
    )
    per_task_node_work = max(
        1,
        sum(
            component_size * (1 << component_size) for component_size in component_sizes
        ),
    )
    total_node_work = task_count * per_task_node_work
    largest_component_size = max(component_sizes, default=0)
    peak_memory_bytes = (
        workspace_array_count * bytes_per_value * (1 << largest_component_size)
    )

    return SparseTransitionPlan(
        full_bit_count=current_tree.bit_count,
        active_bit_indices=_mask_bit_indices(active_mask),
        initial_component_bit_indices=tuple(
            _mask_bit_indices(component) for component in initial_components
        ),
        separator_bit_indices=separator_bit_indices,
        remaining_component_bit_indices=tuple(
            _mask_bit_indices(component) for component in remaining_components
        ),
        task_count_upper_bound=task_count,
        task_count_is_exact=task_count_is_exact,
        per_task_state_count_upper_bound=per_task_state_count,
        per_task_node_work_upper_bound=per_task_node_work,
        total_node_work_upper_bound=total_node_work,
        peak_memory_bytes_per_task=peak_memory_bytes,
        maximum_component_bits=maximum_component_bits,
        maximum_task_count=maximum_task_count,
        maximum_total_node_work=maximum_total_node_work,
        maximum_peak_memory_bytes=maximum_peak_memory_bytes,
    )


def format_sparse_transition_plan(plan: SparseTransitionPlan) -> str:
    """Format one deterministic tabular diagnostic report."""

    initial_sizes = (
        ",".join(
            str(len(component)) for component in plan.initial_component_bit_indices
        )
        or "0"
    )
    remaining_sizes = (
        ",".join(
            str(len(component)) for component in plan.remaining_component_bit_indices
        )
        or "0"
    )
    separator_bits = ",".join(map(str, plan.separator_bit_indices)) or "none"
    return "\n".join(
        (
            f"full_bits\t{plan.full_bit_count}",
            f"active_bits\t{plan.active_bit_count}",
            f"initial_component_bits\t{initial_sizes}",
            f"separator_width\t{plan.separator_width}",
            f"separator_bits\t{separator_bits}",
            f"remaining_component_bits\t{remaining_sizes}",
            f"task_count_upper_bound\t{plan.task_count_upper_bound}",
            f"task_count_exact\t{str(plan.task_count_is_exact).lower()}",
            (
                "per_task_state_count_upper_bound\t"
                f"{plan.per_task_state_count_upper_bound}"
            ),
            (f"per_task_node_work_upper_bound\t{plan.per_task_node_work_upper_bound}"),
            f"total_node_work_upper_bound\t{plan.total_node_work_upper_bound}",
            (f"peak_memory_bytes_per_task\t{plan.peak_memory_bytes_per_task}"),
            f"feasible\t{str(plan.is_feasible).lower()}",
        )
    )


def _combined_interaction_graph(
    first_tree: InheritanceTree,
    second_tree: InheritanceTree,
) -> tuple[tuple[int, ...], int]:
    """Return the union of conservative tree-structure interactions."""

    first_adjacency, first_active = _tree_interaction_graph(first_tree)
    second_adjacency, second_active = _tree_interaction_graph(second_tree)
    return (
        tuple(
            first_neighbors | second_neighbors
            for first_neighbors, second_neighbors in zip(
                first_adjacency,
                second_adjacency,
            )
        ),
        first_active | second_active,
    )


def _add_interaction_groups(
    adjacency_masks: tuple[int, ...],
    bit_count: int,
    interaction_groups: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Add proved transition-factor scopes to a structural interaction graph."""

    updated_adjacency = list(adjacency_masks)
    for group in interaction_groups:
        if len(set(group)) != len(group):
            raise ValueError("Transition interaction groups cannot repeat bits.")
        if any(bit_index < 0 or bit_index >= bit_count for bit_index in group):
            raise ValueError(
                "Transition interaction group contains a bit outside the tree."
            )
        group_mask = sum(1 << bit_index for bit_index in group)
        for bit_index in group:
            updated_adjacency[bit_index] |= group_mask & ~(1 << bit_index)
    return tuple(updated_adjacency)


def _tree_interaction_graph(
    tree: InheritanceTree,
) -> tuple[tuple[int, ...], int]:
    """Build split-bit interactions from one compressed inheritance DAG."""

    adjacency_masks = [0] * tree.bit_count
    descendant_masks: dict[tuple[int, int], int] = {}

    def descendants(node: TreeNode, bit_index: int) -> int:
        key = (id(node), bit_index)
        if key in descendant_masks:
            return descendant_masks[key]

        if isinstance(node, (ZeroNode, LeafNode)):
            active_mask = 0
        elif isinstance(node, _ScaledNode):
            active_mask = descendants(node.child, bit_index)
        elif isinstance(node, SharedNode):
            active_mask = descendants(node.child, bit_index + 1)
        else:
            if not isinstance(node, SplitNode):
                raise TypeError(f"Unsupported inheritance-tree node: {type(node)!r}")
            zero_mask = descendants(node.zero_child, bit_index + 1)
            one_mask = descendants(node.one_child, bit_index + 1)
            downstream_mask = zero_mask | one_mask
            current_bit_mask = 1 << bit_index
            adjacency_masks[bit_index] |= downstream_mask
            for downstream_bit in _mask_bit_indices(downstream_mask):
                adjacency_masks[downstream_bit] |= current_bit_mask
            active_mask = current_bit_mask | downstream_mask

        descendant_masks[key] = active_mask
        return active_mask

    with _inheritance_recursion_budget(tree.bit_count):
        active_mask = descendants(tree.root, 0)
    return tuple(adjacency_masks), active_mask


def _greedy_separator_mask(
    adjacency_masks: tuple[int, ...],
    active_mask: int,
    maximum_component_bits: int,
    next_coordinate_map: MarkerCoordinateMap,
) -> int:
    """Choose a deterministic structural separator under a component limit."""

    separator_mask = 0
    remaining_mask = active_mask
    while True:
        components = _connected_component_masks(
            adjacency_masks,
            remaining_mask,
        )
        if not components:
            return separator_mask
        largest_size = components[0].bit_count()
        if largest_size <= maximum_component_bits:
            return separator_mask

        largest_components_mask = 0
        for component in components:
            if component.bit_count() != largest_size:
                break
            largest_components_mask |= component

        best_bit_index: int | None = None
        selected_coordinates = {
            next_coordinate_map.coordinate_by_bit[bit_index]
            for bit_index in _mask_bit_indices(separator_mask)
            if next_coordinate_map.coordinate_by_bit[bit_index] is not None
        }
        best_score: tuple[int, int, int, int] | None = None
        for bit_index in _mask_bit_indices(largest_components_mask):
            candidate_remaining = remaining_mask & ~(1 << bit_index)
            candidate_components = _connected_component_masks(
                adjacency_masks,
                candidate_remaining,
            )
            candidate_sizes = tuple(
                component.bit_count() for component in candidate_components
            )
            score = (
                max(candidate_sizes, default=0),
                sum(size * size for size in candidate_sizes),
                _coordinate_task_cost(
                    next_coordinate_map,
                    selected_coordinates,
                    bit_index,
                ),
                bit_index,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_bit_index = bit_index

        if best_bit_index is None:
            raise RuntimeError("Could not select a structural separator bit.")
        selected_bit_mask = 1 << best_bit_index
        separator_mask |= selected_bit_mask
        remaining_mask &= ~selected_bit_mask


def _connected_component_masks(
    adjacency_masks: tuple[int, ...],
    included_mask: int,
) -> tuple[int, ...]:
    """Return deterministic connected components of an integer-bitset graph."""

    components: list[int] = []
    unvisited_mask = included_mask
    while unvisited_mask:
        first_bit_mask = unvisited_mask & -unvisited_mask
        component_mask = 0
        frontier_mask = first_bit_mask
        while frontier_mask:
            bit_mask = frontier_mask & -frontier_mask
            frontier_mask &= ~bit_mask
            if component_mask & bit_mask:
                continue
            bit_index = bit_mask.bit_length() - 1
            component_mask |= bit_mask
            neighbors = adjacency_masks[bit_index] & included_mask
            frontier_mask |= neighbors & ~component_mask
        components.append(component_mask)
        unvisited_mask &= ~component_mask

    return tuple(
        sorted(
            components,
            key=lambda component: (
                -component.bit_count(),
                (component & -component).bit_length() - 1,
            ),
        )
    )


def _separator_task_count(
    next_emission_tree: InheritanceTree,
    separator_bit_indices: tuple[int, ...],
    coordinate_map: MarkerCoordinateMap,
    support_projection_state_limit: int,
) -> tuple[int, bool]:
    """Count compatible tasks exactly or return a proved affine upper bound."""

    if not _tree_has_nonzero_support(next_emission_tree):
        return 0, True
    try:
        exact_task_count = _exact_projected_support_count(
            next_emission_tree,
            separator_bit_indices,
            support_projection_state_limit,
        )
    except _SupportProjectionBudgetExceeded:
        exact_task_count = None
    if exact_task_count is not None:
        return exact_task_count, True

    separator_coordinates = {
        coordinate_map.coordinate_by_bit[bit_index]
        for bit_index in separator_bit_indices
        if coordinate_map.coordinate_by_bit[bit_index] is not None
    }
    return 1 << len(separator_coordinates), False


def _coordinate_task_cost(
    coordinate_map: MarkerCoordinateMap,
    selected_coordinates: set[int],
    bit_index: int,
) -> int:
    """Return whether one separator bit introduces a new task coordinate."""

    coordinate = coordinate_map.coordinate_by_bit[bit_index]
    return int(coordinate is not None and coordinate not in selected_coordinates)


class _SupportProjectionBudgetExceeded(RuntimeError):
    """Raised internally when exact Boolean projection exceeds its state cap."""


class _ProjectedSupportBuilder:
    """Build a reduced Boolean decision diagram over separator bits."""

    _FALSE_NODE = 0
    _TRUE_NODE = 1

    def __init__(
        self,
        separator_bit_indices: tuple[int, ...],
        state_limit: int,
    ) -> None:
        self._separator_level_by_bit = {
            bit_index: level for level, bit_index in enumerate(separator_bit_indices)
        }
        self._separator_bit_count = len(separator_bit_indices)
        self._state_limit = state_limit
        self._next_node_id = 2
        self._record_by_node_id: dict[int, tuple[int, int, int]] = {}
        self._node_id_by_record: dict[tuple[int, int, int], int] = {}
        self._projection_cache: dict[tuple[int, int], int] = {}
        self._or_cache: dict[tuple[int, int], int] = {}

    def projected_support_count(self, tree: InheritanceTree) -> int:
        """Return exact distinct supported assignments on separator bits."""

        with _inheritance_recursion_budget(tree.bit_count):
            root_node_id = self._project(tree.root, 0)
        return self._count_assignments(root_node_id, 0, {})

    def _project(self, node: TreeNode, bit_index: int) -> int:
        cache_key = (id(node), bit_index)
        cached_node_id = self._projection_cache.get(cache_key)
        if cached_node_id is not None:
            return cached_node_id
        self._check_state_budget()

        if isinstance(node, ZeroNode):
            projected_node_id = self._FALSE_NODE
        elif isinstance(node, LeafNode):
            projected_node_id = (
                self._TRUE_NODE if node.value != 0.0 else self._FALSE_NODE
            )
        elif isinstance(node, _ScaledNode):
            projected_node_id = self._project(node.child, bit_index)
        elif isinstance(node, SharedNode):
            projected_node_id = self._project(node.child, bit_index + 1)
        else:
            if not isinstance(node, SplitNode):
                raise TypeError(f"Unsupported inheritance-tree node: {type(node)!r}")
            zero_node_id = self._project(node.zero_child, bit_index + 1)
            one_node_id = self._project(node.one_child, bit_index + 1)
            separator_level = self._separator_level_by_bit.get(bit_index)
            if separator_level is None:
                projected_node_id = self._or_nodes(
                    zero_node_id,
                    one_node_id,
                )
            else:
                projected_node_id = self._make_node(
                    separator_level,
                    zero_node_id,
                    one_node_id,
                )

        self._projection_cache[cache_key] = projected_node_id
        return projected_node_id

    def _or_nodes(self, first_node_id: int, second_node_id: int) -> int:
        if first_node_id == self._TRUE_NODE or second_node_id == self._TRUE_NODE:
            return self._TRUE_NODE
        if first_node_id == self._FALSE_NODE:
            return second_node_id
        if second_node_id == self._FALSE_NODE:
            return first_node_id
        if first_node_id == second_node_id:
            return first_node_id

        cache_key = tuple(sorted((first_node_id, second_node_id)))
        cached_node_id = self._or_cache.get(cache_key)
        if cached_node_id is not None:
            return cached_node_id
        self._check_state_budget()

        first_level = self._record_by_node_id[first_node_id][0]
        second_level = self._record_by_node_id[second_node_id][0]
        current_level = min(first_level, second_level)
        first_zero, first_one = self._cofactors(first_node_id, current_level)
        second_zero, second_one = self._cofactors(second_node_id, current_level)
        result_node_id = self._make_node(
            current_level,
            self._or_nodes(first_zero, second_zero),
            self._or_nodes(first_one, second_one),
        )
        self._or_cache[cache_key] = result_node_id
        return result_node_id

    def _cofactors(self, node_id: int, level: int) -> tuple[int, int]:
        node_level, zero_node_id, one_node_id = self._record_by_node_id[node_id]
        if node_level == level:
            return zero_node_id, one_node_id
        return node_id, node_id

    def _make_node(
        self,
        level: int,
        zero_node_id: int,
        one_node_id: int,
    ) -> int:
        if zero_node_id == one_node_id:
            return zero_node_id
        record = (level, zero_node_id, one_node_id)
        existing_node_id = self._node_id_by_record.get(record)
        if existing_node_id is not None:
            return existing_node_id
        self._check_state_budget()
        node_id = self._next_node_id
        self._next_node_id += 1
        self._record_by_node_id[node_id] = record
        self._node_id_by_record[record] = node_id
        return node_id

    def _count_assignments(
        self,
        node_id: int,
        next_level: int,
        cache: dict[tuple[int, int], int],
    ) -> int:
        if node_id == self._FALSE_NODE:
            return 0
        if node_id == self._TRUE_NODE:
            return 1 << (self._separator_bit_count - next_level)
        cache_key = (node_id, next_level)
        cached_count = cache.get(cache_key)
        if cached_count is not None:
            return cached_count

        level, zero_node_id, one_node_id = self._record_by_node_id[node_id]
        skipped_assignment_count = 1 << (level - next_level)
        assignment_count = skipped_assignment_count * (
            self._count_assignments(zero_node_id, level + 1, cache)
            + self._count_assignments(one_node_id, level + 1, cache)
        )
        cache[cache_key] = assignment_count
        return assignment_count

    def _check_state_budget(self) -> None:
        state_count = (
            len(self._projection_cache)
            + len(self._or_cache)
            + len(self._record_by_node_id)
        )
        if state_count >= self._state_limit:
            raise _SupportProjectionBudgetExceeded


def _exact_projected_support_count(
    tree: InheritanceTree,
    separator_bit_indices: tuple[int, ...],
    state_limit: int,
) -> int:
    """Project nonzero support onto separator bits with a bounded ROBDD."""

    builder = _ProjectedSupportBuilder(separator_bit_indices, state_limit)
    return builder.projected_support_count(tree)


def _tree_has_nonzero_support(tree: InheritanceTree) -> bool:
    """Return whether a tree contains at least one nonzero terminal."""

    pending_nodes = [tree.root]
    visited_node_ids: set[int] = set()
    while pending_nodes:
        node = pending_nodes.pop()
        node_id = id(node)
        if node_id in visited_node_ids:
            continue
        visited_node_ids.add(node_id)
        if isinstance(node, ZeroNode):
            continue
        if isinstance(node, LeafNode):
            if node.value != 0.0:
                return True
            continue
        if isinstance(node, (_ScaledNode, SharedNode)):
            pending_nodes.append(node.child)
            continue
        if not isinstance(node, SplitNode):
            raise TypeError(f"Unsupported inheritance-tree node: {type(node)!r}")
        pending_nodes.extend((node.zero_child, node.one_child))
    return False


def _mask_bit_indices(mask: int) -> tuple[int, ...]:
    """Expand one nonnegative integer bitset in ascending order."""

    bit_indices: list[int] = []
    while mask:
        bit_mask = mask & -mask
        bit_indices.append(bit_mask.bit_length() - 1)
        mask &= ~bit_mask
    return tuple(bit_indices)


def _validate_matching_tree_depths(
    first_tree: InheritanceTree,
    second_tree: InheritanceTree,
) -> None:
    """Require one ordered inheritance-bit space for an interval."""

    if first_tree.bit_count != second_tree.bit_count:
        raise ValueError("Transition trees must use the same inheritance bits.")


def _validate_positive_limit(value: int, label: str) -> None:
    """Reject booleans and nonpositive resource-planning limits."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
