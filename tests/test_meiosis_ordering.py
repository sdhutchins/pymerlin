"""Tests for exact inheritance-vector coordinate orderings."""

from fractions import Fraction
from itertools import product

from pymerlin import Family, Individual, Marker, Meiosis
from pymerlin.likelihood import family_marker_likelihood_tree
from pymerlin.meiosis_ordering import order_family_meioses


def test_orderings_preserve_meioses_and_return_inverse_permutations() -> None:
    marker, family = _ordering_family()

    identifier_order = order_family_meioses(
        family,
        "individual_identifier",
    )
    topological_order = order_family_meioses(
        family,
        "parent_before_child",
    )

    assert identifier_order.source_index_by_ordered_index == (2, 3, 0, 1)
    assert topological_order.source_index_by_ordered_index == (0, 1, 2, 3)
    assert set(identifier_order.family.meioses) == set(family.meioses)
    assert identifier_order.family.individuals == family.individuals

    for source_bits in product((0, 1), repeat=len(family.meioses)):
        ordered_bits = identifier_order.to_ordered_bits(source_bits)
        assert identifier_order.to_source_bits(ordered_bits) == source_bits

    source_tree = family_marker_likelihood_tree(family, marker)
    ordered_tree = family_marker_likelihood_tree(
        identifier_order.family,
        marker,
    )
    for source_bits in product((0, 1), repeat=len(family.meioses)):
        ordered_bits = identifier_order.to_ordered_bits(source_bits)
        assert source_tree.value_at(source_bits) == ordered_tree.value_at(
            ordered_bits
        )


def test_recombination_probability_is_invariant_to_coordinate_order() -> None:
    _, family = _ordering_family()
    ordering = order_family_meioses(family, "individual_identifier")
    theta = Fraction(17, 100)

    for source_bits in product((0, 1), repeat=len(family.meioses)):
        for target_bits in product((0, 1), repeat=len(family.meioses)):
            ordered_source_bits = ordering.to_ordered_bits(source_bits)
            ordered_target_bits = ordering.to_ordered_bits(target_bits)
            assert _transition_probability(
                source_bits,
                target_bits,
                theta,
            ) == _transition_probability(
                ordered_source_bits,
                ordered_target_bits,
                theta,
            )


def _ordering_family() -> tuple[Marker, Family]:
    marker = Marker(
        name="marker",
        chromosome="1",
        position_cm=0.0,
        allele_frequencies={"1": 0.5, "2": 0.5},
    )

    def person(
        individual_id: str,
        father_id: str | None,
        mother_id: str | None,
        sex: str,
        genotype: tuple[str | None, str | None],
    ) -> Individual:
        return Individual(
            family_id="1",
            individual_id=individual_id,
            father_id=father_id,
            mother_id=mother_id,
            sex=sex,
            phenotypes={},
            genotypes={marker.name: genotype},
        )

    family = Family(
        family_id="1",
        individuals=(
            person("grandchild", "child", "other", "2", ("1", "1")),
            person("child", "parent", "spouse", "1", ("1", "2")),
            person("other", None, None, "1", (None, None)),
            person("parent", None, None, "1", (None, None)),
            person("spouse", None, None, "2", (None, None)),
        ),
        meioses=(
            Meiosis(parent_id="parent", child_id="child", parent_sex="1"),
            Meiosis(parent_id="spouse", child_id="child", parent_sex="2"),
            Meiosis(parent_id="child", child_id="grandchild", parent_sex="1"),
            Meiosis(parent_id="other", child_id="grandchild", parent_sex="1"),
        ),
    )
    return marker, family


def _transition_probability(
    source_bits: tuple[int, ...],
    target_bits: tuple[int, ...],
    theta: Fraction,
) -> Fraction:
    probability = Fraction(1)
    for source_bit, target_bit in zip(source_bits, target_bits, strict=True):
        probability *= 1 - theta if source_bit == target_bit else theta
    return probability
