"""Tests for the exact founder-couple quotient after orientation reduction."""

from __future__ import annotations

from itertools import product

import pytest

from pymerlin import (
    Family,
    Individual,
    Marker,
    Meiosis,
    build_founder_couple_quotient,
    build_founder_orientation_quotient,
    family_marker_likelihood_tree,
    founder_couple_transition_probability,
    reduce_founder_couple_tree,
    reduce_founder_orientation_tree,
)


def test_couple_quotient_projects_the_full_involution() -> None:
    _, family = _founder_couple_family()
    orientation_quotient = build_founder_orientation_quotient(family)

    couple_quotient = build_founder_couple_quotient(
        family,
        orientation_quotient,
    )

    assert orientation_quotient.reduced_bit_count == 5
    assert couple_quotient.reduced_bit_count == 4
    assert couple_quotient.reduced_bit_index_by_input_bit == (0, 1, None, 2, 3)
    assert len(couple_quotient.groups) == 1
    group = couple_quotient.groups[0]
    assert group.founder_ids == ("1", "2")
    assert group.representative_input_bit_index == 2
    assert group.affected_input_bit_indices == (0, 1, 2, 3)
    assert group.transform((0, 1, 0, 1, 1)) == (1, 0, 1, 0, 1)
    assert couple_quotient.transition_interaction_groups == ((0, 1, 2),)

    input_bits = (0, 1, 1, 1, 0)
    transformed_bits = group.transform(input_bits)
    assert couple_quotient.project(input_bits) == couple_quotient.project(
        transformed_bits
    )
    assert couple_quotient.expand_canonical(
        couple_quotient.project(input_bits)
    )[group.representative_input_bit_index] == 0


def test_couple_tree_reduction_preserves_every_marker_likelihood() -> None:
    marker, family = _founder_couple_family()
    orientation_quotient = build_founder_orientation_quotient(family)
    couple_quotient = build_founder_couple_quotient(
        family,
        orientation_quotient,
    )
    full_tree = family_marker_likelihood_tree(family, marker)
    orientation_tree = reduce_founder_orientation_tree(
        full_tree,
        orientation_quotient,
    )

    couple_tree = reduce_founder_couple_tree(
        orientation_tree,
        couple_quotient,
    )

    for full_bits in product((0, 1), repeat=len(family.meioses)):
        reduced_bits = couple_quotient.project(
            orientation_quotient.project(full_bits)
        )
        assert couple_tree.value_at(reduced_bits) == full_tree.value_at(full_bits)


def test_couple_transition_matches_exhaustive_full_target_orbit() -> None:
    _, family = _founder_couple_family()
    orientation_quotient = build_founder_orientation_quotient(family)
    couple_quotient = build_founder_couple_quotient(
        family,
        orientation_quotient,
    )
    theta = 0.13
    reduced_vectors = tuple(
        product((0, 1), repeat=couple_quotient.reduced_bit_count)
    )
    full_vectors = tuple(product((0, 1), repeat=len(family.meioses)))

    for previous_reduced_bits in reduced_vectors:
        previous_full_bits = orientation_quotient.expand_canonical(
            couple_quotient.expand_canonical(previous_reduced_bits)
        )
        row_sum = 0.0
        for next_reduced_bits in reduced_vectors:
            observed = founder_couple_transition_probability(
                previous_reduced_bits,
                next_reduced_bits,
                theta,
                orientation_quotient,
                couple_quotient,
            )
            expected = sum(
                _full_transition_probability(
                    previous_full_bits,
                    next_full_bits,
                    theta,
                )
                for next_full_bits in full_vectors
                if couple_quotient.project(
                    orientation_quotient.project(next_full_bits)
                )
                == next_reduced_bits
            )
            assert observed == pytest.approx(expected)
            row_sum += observed
        assert row_sum == pytest.approx(1.0)


def _founder_couple_family() -> tuple[Marker, Family]:
    marker = Marker(
        name="founder_couple_marker",
        chromosome="1",
        position_cm=0.0,
        allele_frequencies={"1": 0.4, "2": 0.6},
    )
    observed_genotypes = {
        "3": ("1", "1"),
        "4": ("1", "2"),
        "6": ("2", "2"),
    }

    def person(
        individual_id: str,
        father_id: str | None,
        mother_id: str | None,
        sex: str,
    ) -> Individual:
        return Individual(
            family_id="1",
            individual_id=individual_id,
            father_id=father_id,
            mother_id=mother_id,
            sex=sex,
            phenotypes={},
            genotypes={
                marker.name: observed_genotypes.get(
                    individual_id,
                    (None, None),
                )
            },
        )

    family = Family(
        family_id="1",
        individuals=(
            person("1", None, None, "1"),
            person("2", None, None, "2"),
            person("3", "1", "2", "1"),
            person("4", "1", "2", "2"),
            person("5", "3", "4", "1"),
            person("7", None, None, "2"),
            person("6", "5", "7", "1"),
        ),
        meioses=(
            Meiosis(parent_id="1", child_id="3", parent_sex="1"),
            Meiosis(parent_id="2", child_id="3", parent_sex="2"),
            Meiosis(parent_id="1", child_id="4", parent_sex="1"),
            Meiosis(parent_id="2", child_id="4", parent_sex="2"),
            Meiosis(parent_id="3", child_id="5", parent_sex="1"),
            Meiosis(parent_id="4", child_id="5", parent_sex="2"),
            Meiosis(parent_id="5", child_id="6", parent_sex="1"),
            Meiosis(parent_id="7", child_id="6", parent_sex="2"),
        ),
    )
    return marker, family


def _full_transition_probability(
    previous_bits: tuple[int, ...],
    next_bits: tuple[int, ...],
    theta: float,
) -> float:
    probability = 1.0
    for previous_bit, next_bit in zip(previous_bits, next_bits):
        probability *= 1.0 - theta if previous_bit == next_bit else theta
    return probability
