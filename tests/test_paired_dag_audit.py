"""Tests for the bounded exact paired-DAG transition audit."""

from pymerlin.founder_couple_quotient import (
    FounderCoupleQuotient,
    FounderCoupleQuotientGroup,
)
from pymerlin.founder_orientation_quotient import (
    FounderOrientationGroup,
    FounderOrientationQuotient,
)
from pymerlin.inheritance_tree import InheritanceTree
from pymerlin.paired_dag_audit import (
    audit_founder_couple_key_effect,
    audit_paired_dag_transition,
    format_founder_couple_key_audit,
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


def test_compound_quotient_tracks_both_exact_transition_contexts() -> None:
    first_tree = InheritanceTree.from_dense((1.0, 2.0))
    second_tree = InheritanceTree.from_dense((3.0, 4.0))
    orientation_quotient = FounderOrientationQuotient(
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
    couple_quotient = FounderCoupleQuotient(
        input_bit_count=2,
        reduced_bit_count=1,
        reduced_bit_index_by_input_bit=(None, 0),
        groups=(
            FounderCoupleQuotientGroup(
                founder_ids=("founder", "partner"),
                representative_input_bit_index=0,
                affected_input_bit_indices=(0, 1),
                input_bit_index_by_output_bit=(0, 1),
                xor_offset_by_output_bit=(1, 1),
            ),
        ),
    )

    audit = audit_paired_dag_transition(
        first_tree,
        second_tree,
        0.1,
        founder_quotient=orientation_quotient,
        founder_couple_quotient=couple_quotient,
    )

    assert audit.complete
    assert audit.founder_context_group_count == 1
    assert audit.founder_couple_context_group_count == 1
    assert audit.maximum_open_founder_contexts == 2


def test_founder_couple_key_audit_detects_context_without_removed_split() -> None:
    input_tree = InheritanceTree.from_dense(
        (1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0)
    )
    quotient = FounderCoupleQuotient(
        input_bit_count=3,
        reduced_bit_count=2,
        reduced_bit_index_by_input_bit=(None, 0, 1),
        groups=(
            FounderCoupleQuotientGroup(
                founder_ids=("founder", "partner"),
                representative_input_bit_index=0,
                affected_input_bit_indices=(0, 1, 2),
                input_bit_index_by_output_bit=(0, 1, 2),
                xor_offset_by_output_bit=(1, 1, 1),
            ),
        ),
    )

    audit = audit_founder_couple_key_effect(
        input_tree,
        input_tree,
        0.1,
        quotient,
    )

    assert audit.input_bit_count == 3
    assert audit.reduced_bit_count == 2
    assert audit.removed_bit_count == 1
    assert audit.input_active_bit_count == 2
    assert audit.reduced_active_bit_count == 2
    assert audit.removed_active_bit_count == 0
    assert audit.persistent_context_group_count == 1
    assert audit.maximum_open_context_count == 1
    assert audit.maximum_context_lane_count == 2
    assert not audit.groups[0].representative_is_active
    assert audit.groups[0].active_reduced_bit_indices == (0, 1)
    assert audit.groups[0].context_first_bit_index == 0
    assert audit.groups[0].context_last_bit_index == 1

    expected_report = (
        "founder_couple_input_bits\t3\n"
        "founder_couple_reduced_bits\t2\n"
        "founder_couple_removed_bits\t1\n"
        "founder_couple_input_active_bits\t2\n"
        "founder_couple_reduced_active_bits\t2\n"
        "founder_couple_removed_active_bits\t0\n"
        "founder_couple_persistent_context_groups\t1\n"
        "founder_couple_maximum_open_contexts\t1\n"
        "founder_couple_maximum_context_lanes\t2\n"
        "founder_couple_representatives_active\tfounder,partner:false\n"
        "founder_couple_context_spans\tfounder,partner:0-1"
    )
    assert format_founder_couple_key_audit(audit) == expected_report


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
        "founder_couple_context_groups\t0\n"
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
