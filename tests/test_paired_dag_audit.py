"""Tests for the bounded exact paired-DAG transition audit."""

from pymerlin.founder_orientation_quotient import (
    FounderOrientationGroup,
    FounderOrientationQuotient,
)
from pymerlin.inheritance_tree import InheritanceTree
from pymerlin.paired_dag_audit import (
    audit_paired_dag_transition,
    format_paired_dag_transition_audit,
)


def test_constant_pair_stops_at_one_subproblem() -> None:
    first_tree = InheritanceTree.from_dense((1.0,) * 8)
    second_tree = InheritanceTree.from_dense((2.0,) * 8)

    audit = audit_paired_dag_transition(first_tree, second_tree, 0.1)

    assert audit.complete
    assert audit.examined_unique_subproblem_count == 1
    assert audit.transition_arc_count == 0
    assert audit.deepest_bit_index_reached == 0


def test_one_split_bit_visits_four_unique_terminal_pairs() -> None:
    first_tree = InheritanceTree.from_dense((1.0, 2.0))
    second_tree = InheritanceTree.from_dense((3.0, 4.0))

    audit = audit_paired_dag_transition(first_tree, second_tree, 0.1)

    assert audit.complete
    assert audit.active_bit_count == 1
    assert audit.examined_unique_subproblem_count == 5
    assert audit.transition_arc_count == 4
    assert audit.maximum_frontier_state_count == 4
    assert audit.deepest_bit_index_reached == 1


def test_zero_recombination_omits_cross_branch_pairs() -> None:
    first_tree = InheritanceTree.from_dense((1.0, 2.0))
    second_tree = InheritanceTree.from_dense((3.0, 4.0))

    audit = audit_paired_dag_transition(first_tree, second_tree, 0.0)

    assert audit.complete
    assert audit.examined_unique_subproblem_count == 3
    assert audit.transition_arc_count == 2


def test_zero_child_contributions_are_not_enqueued() -> None:
    first_tree = InheritanceTree.from_dense((0.0, 2.0))
    second_tree = InheritanceTree.from_dense((3.0, 4.0))

    audit = audit_paired_dag_transition(first_tree, second_tree, 0.1)

    assert audit.complete
    assert audit.examined_unique_subproblem_count == 3
    assert audit.transition_arc_count == 2


def test_founder_context_remains_open_across_active_relative_bits() -> None:
    first_tree = InheritanceTree.from_dense((1.0, 2.0, 3.0, 4.0))
    second_tree = InheritanceTree.from_dense((5.0, 6.0, 7.0, 8.0))
    quotient = FounderOrientationQuotient(
        full_bit_count=3,
        reduced_bit_count=2,
        reduced_bit_index_by_full_bit=(None, 0, 1),
        groups=(
            FounderOrientationGroup(
                founder_id="founder",
                representative_full_bit_index=0,
                member_full_bit_indices=(0, 1, 2),
                reduced_member_bit_indices=(0, 1),
            ),
        ),
    )

    audit = audit_paired_dag_transition(
        first_tree,
        second_tree,
        0.1,
        founder_quotient=quotient,
    )

    assert audit.complete
    assert audit.founder_context_group_count == 1
    assert audit.maximum_open_founder_contexts == 1
    assert audit.examined_unique_subproblem_count == 26


def test_state_budget_returns_an_explicit_lower_bound() -> None:
    first_tree = InheritanceTree.from_dense((1.0, 2.0))
    second_tree = InheritanceTree.from_dense((3.0, 4.0))

    audit = audit_paired_dag_transition(
        first_tree,
        second_tree,
        0.1,
        maximum_unique_subproblems=3,
    )

    assert not audit.complete
    assert audit.count_is_lower_bound
    assert audit.examined_unique_subproblem_count == 3
    assert audit.deepest_bit_index_reached == 1


def test_audit_format_distinguishes_complete_count_from_lower_bound() -> None:
    tree = InheritanceTree.from_dense((1.0, 2.0))
    audit = audit_paired_dag_transition(
        tree,
        tree,
        0.125,
        maximum_unique_subproblems=3,
    )

    expected_report = (
        "bit_count\t1\n"
        "active_bits\t1\n"
        "recombination_fraction\t0.125\n"
        "founder_context_groups\t0\n"
        "maximum_open_founder_contexts\t0\n"
        "examined_unique_subproblems\t3\n"
        "transition_arcs\t3\n"
        "maximum_frontier_states\t3\n"
        "deepest_bit_index\t1\n"
        "state_budget\t3\n"
        "complete\tfalse\n"
        "count_is_lower_bound\ttrue"
    )
    assert format_paired_dag_transition_audit(audit) == expected_report


def test_audit_rejects_mismatched_depths_and_quotient() -> None:
    one_bit_tree = InheritanceTree.from_dense((1.0, 2.0))
    two_bit_tree = InheritanceTree.from_dense((1.0, 2.0, 3.0, 4.0))
    wrong_quotient = FounderOrientationQuotient(
        full_bit_count=3,
        reduced_bit_count=2,
        reduced_bit_index_by_full_bit=(None, 0, 1),
        groups=(),
    )

    try:
        audit_paired_dag_transition(one_bit_tree, two_bit_tree, 0.1)
    except ValueError as error:
        assert "same ordered inheritance bits" in str(error)
    else:
        raise AssertionError("Mismatched trees should fail.")

    try:
        audit_paired_dag_transition(
            one_bit_tree,
            one_bit_tree,
            0.1,
            founder_quotient=wrong_quotient,
        )
    except ValueError as error:
        assert "same reduced inheritance bits" in str(error)
    else:
        raise AssertionError("Mismatched founder quotient should fail.")
