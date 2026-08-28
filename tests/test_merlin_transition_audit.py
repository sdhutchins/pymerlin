"""Tests for reproducible MERLIN transition-route diagnostics."""

import math
from itertools import product

import pytest

from pymerlin.information import _tree_family_information
from pymerlin.inheritance_tree import InheritanceTree, LeafNode, SharedNode
from pymerlin.merlin_transition_audit import (
    audit_merlin_transition,
    format_merlin_transition_audit,
    marker_tree_information,
)
from pymerlin.models import Dataset, Family, Individual, Marker, Meiosis


def test_marker_information_matches_small_tree_reference() -> None:
    values = (0.0, 1.0, 2.0, 2.0, 0.0, 3.0, 3.0, 4.0)
    tree = InheritanceTree.from_dense(values)

    information = marker_tree_information(tree, merlin_effective_bit_count=3)

    assert information.information == pytest.approx(
        _tree_family_information(tree, full_bit_count=3, bit_count=3),
        rel=1e-14,
    )
    assert information.support_fraction == pytest.approx(0.75)
    assert information.minimum_information == pytest.approx(-math.log2(0.75) / 3)


def test_audit_detects_sparse_route_and_founder_orientation_quotient() -> None:
    dataset, family = _three_sibling_dataset()
    vectors = tuple(product((0, 1), repeat=6))
    values = tuple(
        1.0 if bits[0] == bits[2] == bits[4] == 0 else 0.0 for bits in vectors
    )
    tree = InheritanceTree.from_dense(values)

    audit = audit_merlin_transition(dataset, family, tree, tree)

    assert audit.merlin_effective_bit_count == 4
    assert audit.active_bit_count == 3
    assert audit.active_founder_orientation_group_sizes == (3,)
    assert audit.active_founder_orientation_group_count == 1
    assert audit.known_quotient_hidden_bit_indices == (0,)
    assert audit.active_bits_after_known_symmetry_quotients == 2
    assert audit.active_founder_couple_group_count == 0
    assert audit.active_untyped_chain_count == 0
    assert audit.current_marker_information.information == pytest.approx(0.75)
    assert audit.next_marker_information.information == pytest.approx(0.75)
    assert audit.sparse_conditioning_bound_sum == pytest.approx(1.5)
    assert audit.merlin_would_use_sparse_conditioning


def test_uninformative_markers_do_not_select_sparse_conditioning() -> None:
    dataset, family = _three_sibling_dataset()
    tree = InheritanceTree.from_dense((1.0,) * 64)

    audit = audit_merlin_transition(dataset, family, tree, tree)

    assert audit.current_marker_information.information == 0.0
    assert audit.next_marker_information.information == 0.0
    assert audit.sparse_conditioning_bound_sum == 0.0
    assert not audit.merlin_would_use_sparse_conditioning


def test_audit_format_is_deterministic() -> None:
    dataset, family = _three_sibling_dataset()
    vectors = tuple(product((0, 1), repeat=6))
    tree = InheritanceTree.from_dense(
        tuple(1.0 if bits[0] == bits[2] == bits[4] == 0 else 0.0 for bits in vectors)
    )

    report = format_merlin_transition_audit(
        audit_merlin_transition(dataset, family, tree, tree)
    )

    assert "merlin_effective_bits\t4" in report
    assert "active_founder_orientation_group_sizes\t3" in report
    assert "active_bits_after_known_symmetry_quotients\t2" in report
    assert "sparse_conditioning_bound_sum\t1.50000000" in report
    assert "merlin_would_use_sparse_conditioning\ttrue" in report


def test_marker_information_rejects_invalid_effective_bit_counts() -> None:
    tree = InheritanceTree.from_dense((1.0, 2.0))

    with pytest.raises(ValueError, match="cannot be negative"):
        marker_tree_information(tree, -1)
    with pytest.raises(ValueError, match="cannot exceed"):
        marker_tree_information(tree, 2)


def test_marker_information_handles_deep_shared_tree() -> None:
    root = LeafNode(2.0)
    for _bit_index in range(1_100):
        root = SharedNode(root)
    tree = InheritanceTree(bit_count=1_100, root=root)

    information = marker_tree_information(tree, merlin_effective_bit_count=900)

    assert information.information == 0.0
    assert information.minimum_information == 0.0
    assert information.support_fraction == 1.0


def test_audit_rejects_tree_depth_mismatch() -> None:
    dataset, family = _three_sibling_dataset()
    one_bit_tree = InheritanceTree.from_dense((1.0, 2.0))
    family_tree = InheritanceTree.from_dense((1.0,) * 64)

    with pytest.raises(ValueError, match="Current tree"):
        audit_merlin_transition(dataset, family, one_bit_tree, family_tree)
    with pytest.raises(ValueError, match="Next emission tree"):
        audit_merlin_transition(dataset, family, family_tree, one_bit_tree)


def _three_sibling_dataset() -> tuple[Dataset, Family]:
    """Return a pedigree with four MERLIN-effective inheritance bits."""

    individuals = (
        _individual("FATHER", None, None, "1"),
        _individual("MOTHER", None, None, "2"),
        _individual("CHILD1", "FATHER", "MOTHER", "1"),
        _individual("CHILD2", "FATHER", "MOTHER", "2"),
        _individual("CHILD3", "FATHER", "MOTHER", "1"),
    )
    meioses = tuple(
        meiosis
        for child_id in ("CHILD1", "CHILD2", "CHILD3")
        for meiosis in (
            Meiosis("FATHER", child_id, "1"),
            Meiosis("MOTHER", child_id, "2"),
        )
    )
    family = Family(
        family_id="F1",
        individuals=individuals,
        meioses=meioses,
    )
    marker = Marker(
        name="M1",
        chromosome="1",
        position_cm=0.0,
        allele_frequencies={"1": 0.5, "2": 0.5},
    )
    return Dataset((marker,), (family,), ("AFFECTED",)), family


def _individual(
    individual_id: str,
    father_id: str | None,
    mother_id: str | None,
    sex: str,
) -> Individual:
    """Build one untyped, unaffected person for the test pedigree."""

    return Individual(
        family_id="F1",
        individual_id=individual_id,
        father_id=father_id,
        mother_id=mother_id,
        sex=sex,
        phenotypes={"AFFECTED": "1"},
        genotypes={"M1": (None, None)},
    )
