"""Tests for conservative exact-transition resource planning."""

from itertools import product

import pytest

from pymerlin.inheritance_tree import InheritanceTree
from pymerlin.transition_planner import (
    format_sparse_transition_plan,
    plan_sparse_transition,
)

THREE_BIT_VECTORS = tuple(product((0, 1), repeat=3))


def test_exhaustive_three_bit_support_plans_are_conservative() -> None:
    """Check all binary supports without assuming a favorable factorization."""

    for support_mask in range(1 << len(THREE_BIT_VECTORS)):
        values = tuple(
            float((support_mask >> state_index) & 1)
            for state_index in range(len(THREE_BIT_VECTORS))
        )
        tree = InheritanceTree.from_dense(values)
        original_values = tree.dense_values()

        plan = plan_sparse_transition(
            tree,
            tree,
            maximum_component_bits=1,
        )

        assert plan.active_bit_indices == _dependent_bit_indices(values)
        assert plan.largest_remaining_component_bits <= 1
        assert plan.total_node_work_upper_bound == (
            plan.task_count_upper_bound * plan.per_task_node_work_upper_bound
        )
        assert tree.dense_values() == original_values

        compatible_assignments = {
            tuple(bits[index] for index in plan.separator_bit_indices)
            for bits, value in zip(THREE_BIT_VECTORS, values)
            if value != 0.0
        }
        assert plan.task_count_is_exact
        assert len(compatible_assignments) == plan.task_count_upper_bound


def test_planner_selects_deterministic_lowest_tie_separator() -> None:
    values = tuple(float(1 + bits[0] + 2 * bits[2]) for bits in THREE_BIT_VECTORS)
    tree = InheritanceTree.from_dense(values)

    plan = plan_sparse_transition(
        tree,
        tree,
        maximum_component_bits=1,
    )

    assert plan.active_bit_indices == (0, 2)
    assert plan.initial_component_bit_indices == ((0, 2),)
    assert plan.separator_bit_indices == (0,)
    assert plan.remaining_component_bit_indices == ((2,),)
    assert plan.task_count_upper_bound == 2
    assert plan.task_count_is_exact
    assert plan.per_task_state_count_upper_bound == 2
    assert plan.per_task_node_work_upper_bound == 2
    assert plan.total_node_work_upper_bound == 4
    assert plan.peak_memory_bytes_per_task == 32
    assert plan.is_feasible


def test_fixed_target_separator_requires_one_compatible_task() -> None:
    current_values = tuple(
        float(1 + bits[0] + 2 * bits[2]) for bits in THREE_BIT_VECTORS
    )
    next_values = tuple(1.0 if bits[2] == 0 else 0.0 for bits in THREE_BIT_VECTORS)
    current_tree = InheritanceTree.from_dense(current_values)
    next_tree = InheritanceTree.from_dense(next_values)

    plan = plan_sparse_transition(
        current_tree,
        next_tree,
        maximum_component_bits=1,
    )

    assert plan.separator_bit_indices == (2,)
    assert plan.task_count_upper_bound == 1
    assert plan.task_count_is_exact


def test_separate_tree_dependencies_remain_separate_components() -> None:
    current_tree = InheritanceTree.from_dense(
        tuple(float(1 + bits[0]) for bits in THREE_BIT_VECTORS)
    )
    next_tree = InheritanceTree.from_dense(
        tuple(float(1 + bits[2]) for bits in THREE_BIT_VECTORS)
    )

    plan = plan_sparse_transition(
        current_tree,
        next_tree,
        maximum_component_bits=1,
    )

    assert plan.initial_component_bit_indices == ((0,), (2,))
    assert plan.separator_bit_indices == ()
    assert plan.remaining_component_bit_indices == ((0,), (2,))
    assert plan.task_count_upper_bound == 1
    assert plan.task_count_is_exact
    assert plan.per_task_state_count_upper_bound == 4
    assert plan.per_task_node_work_upper_bound == 4


def test_empty_next_marker_support_produces_no_tasks() -> None:
    current_tree = InheritanceTree.from_dense(tuple(range(1, 9)))
    empty_tree = InheritanceTree.from_dense((0.0,) * 8)

    plan = plan_sparse_transition(
        current_tree,
        empty_tree,
        maximum_component_bits=1,
    )

    assert plan.task_count_upper_bound == 0
    assert plan.task_count_is_exact
    assert plan.total_node_work_upper_bound == 0
    assert plan.is_feasible


def test_plan_format_is_deterministic_and_machine_readable() -> None:
    values = tuple(float(1 + bits[0] + 2 * bits[2]) for bits in THREE_BIT_VECTORS)
    tree = InheritanceTree.from_dense(values)
    plan = plan_sparse_transition(
        tree,
        tree,
        maximum_component_bits=1,
    )

    expected_report = (
        "full_bits\t3\n"
        "active_bits\t2\n"
        "initial_component_bits\t2\n"
        "separator_width\t1\n"
        "separator_bits\t0\n"
        "remaining_component_bits\t1\n"
        "task_count_upper_bound\t2\n"
        "task_count_exact\ttrue\n"
        "per_task_state_count_upper_bound\t2\n"
        "per_task_node_work_upper_bound\t2\n"
        "total_node_work_upper_bound\t4\n"
        "peak_memory_bytes_per_task\t32\n"
        "feasible\ttrue"
    )
    assert format_sparse_transition_plan(plan) == expected_report


@pytest.mark.parametrize(
    ("keyword", "value", "label"),
    [
        ("maximum_component_bits", 0, "Maximum component bits"),
        ("maximum_task_count", True, "Maximum task count"),
        ("maximum_total_node_work", 1.5, "Maximum total node work"),
        ("maximum_peak_memory_bytes", -1, "Maximum peak memory"),
        ("bytes_per_value", 0, "Bytes per value"),
        ("workspace_array_count", 0, "Workspace array count"),
        (
            "support_projection_state_limit",
            0,
            "Support projection state limit",
        ),
    ],
)
def test_planner_rejects_invalid_limits(
    keyword: str,
    value: object,
    label: str,
) -> None:
    tree = InheritanceTree.from_dense((1.0, 2.0))

    with pytest.raises(ValueError, match=label):
        plan_sparse_transition(tree, tree, **{keyword: value})


def test_planner_rejects_mismatched_tree_depths() -> None:
    one_bit_tree = InheritanceTree.from_dense((1.0, 2.0))
    two_bit_tree = InheritanceTree.from_dense((1.0, 2.0, 3.0, 4.0))

    with pytest.raises(ValueError, match="same inheritance bits"):
        plan_sparse_transition(one_bit_tree, two_bit_tree)


def test_additional_transition_group_connects_active_coordinates() -> None:
    first_tree = InheritanceTree.from_dense(
        tuple(float(1 + bits[0]) for bits in THREE_BIT_VECTORS)
    )
    second_tree = InheritanceTree.from_dense(
        tuple(float(1 + bits[2]) for bits in THREE_BIT_VECTORS)
    )

    plan = plan_sparse_transition(
        first_tree,
        second_tree,
        maximum_component_bits=1,
        additional_interaction_groups=((0, 2),),
    )

    assert plan.initial_component_bit_indices == ((0, 2),)
    assert plan.separator_width == 1


@pytest.mark.parametrize(
    "interaction_group",
    [((0, 0),), ((0, 3),)],
)
def test_planner_rejects_invalid_transition_groups(
    interaction_group: tuple[tuple[int, ...], ...],
) -> None:
    tree = InheritanceTree.from_dense(tuple(range(1, 9)))

    with pytest.raises(ValueError, match="interaction group"):
        plan_sparse_transition(
            tree,
            tree,
            additional_interaction_groups=interaction_group,
        )


def test_projection_budget_falls_back_to_affine_task_upper_bound() -> None:
    values = tuple(
        1.0 if bits[2] == bits[0] ^ bits[1] else 0.0 for bits in THREE_BIT_VECTORS
    )
    tree = InheritanceTree.from_dense(values)

    plan = plan_sparse_transition(
        tree,
        tree,
        maximum_component_bits=1,
        support_projection_state_limit=1,
    )

    assert not plan.task_count_is_exact
    assert plan.task_count_upper_bound >= 1


def _dependent_bit_indices(values: tuple[float, ...]) -> tuple[int, ...]:
    """Return exact bit dependence by exhaustive paired-state comparison."""

    value_by_bits = dict(zip(THREE_BIT_VECTORS, values))
    dependent_indices = []
    for bit_index in range(3):
        if any(
            value_by_bits[bits]
            != value_by_bits[
                bits[:bit_index] + (1 - bits[bit_index],) + bits[bit_index + 1 :]
            ]
            for bits in THREE_BIT_VECTORS
        ):
            dependent_indices.append(bit_index)
    return tuple(dependent_indices)
