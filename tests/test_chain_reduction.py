import math
from decimal import Decimal, localcontext
from itertools import product

import pytest
from gmpy2 import mpfr

from pymerlin import (
    Family,
    Individual,
    Marker,
    Meiosis,
    family_marker_likelihood_tree,
)
from pymerlin.chain_reduction import (
    CountingPartition,
    UntypedChain,
    dense_count_transition_probability,
    detect_untyped_chains,
)
from pymerlin.inheritance_tree import InheritanceTree
from pymerlin.likelihood import _score_family_marker
from pymerlin.multipoint import _tree_forward_backward_trees
from tests.oracles.mpfr_multipoint import mpfr_marker_state_likelihoods


def test_detected_untyped_chain_has_r_plus_one_exact_classes() -> None:
    marker, family = _chain_family_marker()
    del marker

    chains = detect_untyped_chains(family)

    assert len(chains) == 1
    chain = chains[0]
    assert chain.entry_person_id == "A"
    assert chain.internal_person_ids == ("U0", "U1", "U2")
    assert chain.endpoint_person_id == "E"
    assert chain.retained_entry_bit_index == 0
    assert chain.selector_bit_indices == (2, 4, 6)
    assert chain.on_values == (0, 0, 0)
    assert chain.class_count == 4

    partition = CountingPartition.from_chain(chain)
    assert partition.multiplicities == (1, 3, 3, 1)
    assert sum(partition.multiplicities) == 2**chain.selector_count
    assert math.fsum(partition.prior_probabilities) == pytest.approx(1.0)
    assert tuple(
        partition.class_index(representative)
        for representative in partition.representative_vectors
    ) == (0, 1, 2, 3)


def test_count_transition_matches_every_dense_state_in_each_class() -> None:
    partition = CountingPartition(on_values=(0, 1, 0, 1))
    theta = 0.073
    transition_matrix = partition.transition_matrix(theta)

    for previous_bits in product((0, 1), repeat=partition.selector_count):
        previous_class = partition.class_index(previous_bits)
        for next_class in range(partition.class_count):
            dense_probability = dense_count_transition_probability(
                previous_bits,
                next_class,
                partition.on_values,
                theta,
            )
            assert transition_matrix[previous_class][next_class] == (
                pytest.approx(dense_probability, abs=math.ulp(dense_probability))
            )

    assert all(
        math.fsum(row) == pytest.approx(1.0)
        for row in transition_matrix
    )


def test_marker_likelihood_is_float64_constant_within_chain_classes() -> None:
    marker, family = _chain_family_marker()
    chain = detect_untyped_chains(family)[0]
    oracle_likelihood_by_bits = mpfr_marker_state_likelihoods(family, marker)
    likelihoods_by_context_and_class: dict[
        tuple[tuple[int, ...], int],
        set[float],
    ] = {}
    selector_index_set = set(chain.selector_bit_indices)

    for inheritance_bits in product(
        (0, 1),
        repeat=len(family.meioses),
    ):
        context_bits = tuple(
            bit
            for bit_index, bit in enumerate(inheritance_bits)
            if bit_index not in selector_index_set
        )
        class_index = chain.on_count(inheritance_bits)
        likelihoods_by_context_and_class.setdefault(
            (context_bits, class_index),
            set(),
        ).add(
            float(
                oracle_likelihood_by_bits.get(
                    inheritance_bits,
                    mpfr(0),
                )
            )
        )

    assert likelihoods_by_context_and_class
    assert all(
        len(likelihoods) == 1
        for likelihoods in likelihoods_by_context_and_class.values()
    )


def test_count_reduced_marker_tree_matches_every_dense_state() -> None:
    marker, family = _chain_family_marker()
    oracle_likelihood_by_bits = mpfr_marker_state_likelihoods(family, marker)
    dense_likelihood_by_bits = {
        state.bits: state.likelihood
        for state in _score_family_marker(family, marker)
    }

    tree = family_marker_likelihood_tree(family, marker)

    failures: list[str] = []
    for inheritance_bits in product(
        (0, 1),
        repeat=len(family.meioses),
    ):
        dense_likelihood = dense_likelihood_by_bits.get(
            inheritance_bits,
            0.0,
        )
        tree_likelihood = tree.value_at(inheritance_bits)
        oracle_likelihood = oracle_likelihood_by_bits.get(
            inheritance_bits,
            mpfr(0),
        )
        tree_error = abs(mpfr(tree_likelihood) - oracle_likelihood)
        dense_error = abs(mpfr(dense_likelihood) - oracle_likelihood)
        representation_unit = mpfr(math.ulp(float(oracle_likelihood)))
        if tree_error > max(dense_error, representation_unit):
            failures.append(
                f"Count-reduced marker tree is less accurate for "
                f"{inheritance_bits=}: {tree_error=} > {dense_error=}, "
                f"{representation_unit=}"
            )

    assert not failures, "\n".join(failures)


def test_count_reduced_transition_is_float64_accurate_for_every_state() -> None:
    chain = UntypedChain(
        entry_person_id="entry",
        internal_person_ids=("u0", "u1", "u2"),
        endpoint_person_id="endpoint",
        retained_entry_bit_index=0,
        selector_bit_indices=(1, 3, 5),
        on_values=(0, 1, 0),
    )
    bit_count = 6
    inheritance_states = tuple(product((0, 1), repeat=bit_count))
    source_values = tuple(
        float(
            1
            + 3 * bits[0]
            + 5 * bits[2]
            + 7 * bits[4]
            + chain.on_count(bits)
        )
        for bits in inheritance_states
    )
    source_tree = InheritanceTree.from_dense(source_values)
    theta = 0.073

    transitioned_tree = source_tree.transition_counting_chains(
        theta,
        (chain,),
        extended_precision=True,
    )
    binary_transition_tree = source_tree.transition(
        theta,
        extended_precision=True,
    )

    with localcontext() as decimal_context:
        decimal_context.prec = 80
        decimal_theta = Decimal.from_float(theta)
        decimal_complement = Decimal(1) - decimal_theta
        failures: list[str] = []
        for next_bits in inheritance_states:
            exact_probability = sum(
                (
                    Decimal.from_float(source_value)
                    * decimal_theta**sum(
                        previous_bit != next_bit
                        for previous_bit, next_bit in zip(
                            previous_bits,
                            next_bits,
                        )
                    )
                    * decimal_complement**sum(
                        previous_bit == next_bit
                        for previous_bit, next_bit in zip(
                            previous_bits,
                            next_bits,
                        )
                    )
                    for previous_bits, source_value in zip(
                        inheritance_states,
                        source_values,
                    )
                ),
                start=Decimal(0),
            )
            tree_value = transitioned_tree.value_at(next_bits)
            representation_unit = Decimal.from_float(
                math.ulp(float(exact_probability))
            )
            tree_error = abs(
                Decimal.from_float(tree_value) - exact_probability
            )
            binary_transition_error = abs(
                Decimal.from_float(
                    binary_transition_tree.value_at(next_bits)
                )
                - exact_probability
            )
            if tree_error > max(
                binary_transition_error,
                representation_unit,
            ):
                failures.append(
                    f"Count transition is inaccurate for {next_bits=}: "
                    f"{tree_error=} > {binary_transition_error=}, "
                    f"{representation_unit=}"
                )

    assert not failures, "\n".join(failures)


def test_tree_forward_backward_routes_detected_counting_chains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker, family = _chain_family_marker()
    right_marker = Marker(
        name="right_marker",
        chromosome="1",
        position_cm=1.0,
        allele_frequencies=marker.allele_frequencies,
    )
    emission_tree = family_marker_likelihood_tree(family, marker)
    original_transition = InheritanceTree.transition_counting_chains
    routed_chains: list[tuple[UntypedChain, ...]] = []

    def counted_transition(
        tree: InheritanceTree,
        recombination_fraction: float,
        chains: tuple[UntypedChain, ...],
        *,
        extended_precision: bool = False,
    ) -> InheritanceTree:
        routed_chains.append(chains)
        return original_transition(
            tree,
            recombination_fraction,
            chains,
            extended_precision=extended_precision,
        )

    monkeypatch.setattr(
        InheritanceTree,
        "transition_counting_chains",
        counted_transition,
    )

    _tree_forward_backward_trees(
        family,
        (marker, right_marker),
        (0.01,),
        emission_trees=(emission_tree, emission_tree),
    )

    assert len(routed_chains) == 2
    assert all(chains == detect_untyped_chains(family) for chains in routed_chains)


def test_parallel_forward_backward_matches_serial_trees() -> None:
    """Require two directional workers to preserve every tree value."""

    marker, family = _chain_family_marker()
    markers = (
        marker,
        Marker(
            name="right_marker",
            chromosome="1",
            position_cm=1.0,
            allele_frequencies=marker.allele_frequencies,
        ),
    )
    emission_trees = tuple(
        family_marker_likelihood_tree(family, current_marker)
        for current_marker in markers
    )

    serial_messages = _tree_forward_backward_trees(
        family,
        markers,
        (0.01,),
        emission_trees=emission_trees,
        workers=1,
    )
    parallel_messages = _tree_forward_backward_trees(
        family,
        markers,
        (0.01,),
        emission_trees=emission_trees,
        workers=2,
    )

    assert tuple(
        tree.dense_values()
        for direction in parallel_messages
        for tree in direction
    ) == tuple(
        tree.dense_values()
        for direction in serial_messages
        for tree in direction
    )


def _chain_family_marker() -> tuple[Marker, Family]:
    marker = Marker(
        name="chain_marker",
        chromosome="1",
        position_cm=0.0,
        allele_frequencies={"1": 0.45, "2": 0.55},
    )
    observed_genotypes = {
        "E": ("1", "1"),
        "T": ("1", "2"),
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
            phenotypes={"trait": "0"},
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
            person("A", None, None, "1"),
            person("S0", None, None, "2"),
            person("U0", "A", "S0", "1"),
            person("S1", None, None, "2"),
            person("U1", "U0", "S1", "1"),
            person("S2", None, None, "2"),
            person("U2", "U1", "S2", "1"),
            person("S3", None, None, "2"),
            person("E", "U2", "S3", "1"),
            person("ST", None, None, "2"),
            person("T", "A", "ST", "1"),
        ),
        meioses=(
            Meiosis(parent_id="A", child_id="U0", parent_sex="1"),
            Meiosis(parent_id="S0", child_id="U0", parent_sex="2"),
            Meiosis(parent_id="U0", child_id="U1", parent_sex="1"),
            Meiosis(parent_id="S1", child_id="U1", parent_sex="2"),
            Meiosis(parent_id="U1", child_id="U2", parent_sex="1"),
            Meiosis(parent_id="S2", child_id="U2", parent_sex="2"),
            Meiosis(parent_id="U2", child_id="E", parent_sex="1"),
            Meiosis(parent_id="S3", child_id="E", parent_sex="2"),
            Meiosis(parent_id="A", child_id="T", parent_sex="1"),
            Meiosis(parent_id="ST", child_id="T", parent_sex="2"),
        ),
    )
    return marker, family
