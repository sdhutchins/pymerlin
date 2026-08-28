import math
from decimal import Decimal
from itertools import product

import pytest
from gmpy2 import mpfr

import pymerlin.likelihood as likelihood_module
from pymerlin import (
    Dataset,
    Family,
    Individual,
    LeafNode,
    Marker,
    Meiosis,
    family_marker_likelihood_tree,
    inheritance_origins,
    load_merlin_inputs,
    peeled_state_likelihood,
    single_marker_likelihood,
)
from tests.oracles.mpfr_multipoint import mpfr_marker_state_likelihoods


@pytest.mark.parametrize("marker_name", ["some_marker", "another_marker"])
def test_marker_tree_matches_every_explicit_inheritance_state(
    marker_name: str,
) -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    family = dataset.families[0]
    marker = dataset.marker_by_name[marker_name]

    explicit_result = single_marker_likelihood(dataset, marker_name)
    tree = family_marker_likelihood_tree(family, marker)
    oracle_likelihoods = mpfr_marker_state_likelihoods(family, marker)
    explicit_likelihoods = {
        state.bits: state.likelihood
        for state in explicit_result.states
        if state.family_id == family.family_id
    }
    expected_values = tuple(
        explicit_likelihoods.get(bits, 0.0)
        for bits in product((0, 1), repeat=len(family.meioses))
    )

    failures = []
    for bits, tree_value, explicit_value in zip(
        product((0, 1), repeat=len(family.meioses)),
        tree.dense_values(),
        expected_values,
    ):
        oracle_value = oracle_likelihoods.get(tuple(bits), mpfr(0))
        tree_error = abs(mpfr(tree_value) - oracle_value)
        explicit_error = abs(mpfr(explicit_value) - oracle_value)
        representation_unit = mpfr(math.ulp(float(oracle_value)))
        if tree_error > max(explicit_error, representation_unit):
            failures.append(
                f"Tree likelihood is less accurate for {marker_name=}, "
                f"{bits=}: {tree_error=} > {explicit_error=}, "
                f"{representation_unit=}"
            )

    assert not failures, "\n".join(failures)
    assert tree.weighted_sum() * (2 ** tree.bit_count) == pytest.approx(
        explicit_result.likelihood
    )


def test_marker_tree_uses_merlin_uninformative_family_fallback() -> None:
    marker, family = _incompatible_family_marker()
    dataset = Dataset(markers=(marker,), families=(family,))

    explicit_result = single_marker_likelihood(dataset, marker.name)
    tree = family_marker_likelihood_tree(family, marker)

    assert tree.root == LeafNode(1.0)
    assert tree.node_count() == 1
    assert tree.dense_values() == tuple(
        state.likelihood for state in explicit_result.states
    )
    assert tree.build_statistics is not None
    assert tree.build_statistics.zero_peeled_factor_count > 0


def test_canonical_suffix_cache_preserves_every_likelihood_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare canonical-state reuse with the unreduced recursion."""

    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    family = dataset.families[0]
    marker = dataset.markers[0]
    cached_tree = family_marker_likelihood_tree(family, marker)

    monkeypatch.setattr(
        likelihood_module._MarkerTraversalState,
        "cached_suffix_tree",
        lambda self, bit_index: None,
    )
    monkeypatch.setattr(
        likelihood_module._MarkerTraversalState,
        "cache_suffix_tree",
        lambda self, bit_index, node: None,
    )
    uncached_tree = family_marker_likelihood_tree(family, marker)

    assert cached_tree.dense_values() == uncached_tree.dense_values()


def test_suffix_cache_distinguishes_pending_parental_transmissions() -> None:
    """Keep selected alleles live until both child inputs are available."""

    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    family = dataset.families[0]
    marker = dataset.markers[0]
    inheritance_bits = [0] * len(family.meioses)
    relevant_individual_ids = (
        likelihood_module._marker_relevant_individual_ids(
            family,
            marker.name,
        )
    )
    traversal_state = likelihood_module._MarkerTraversalState(
        family,
        marker.name,
        inheritance_bits,
        relevant_individual_ids,
    )
    initial_checkpoint = traversal_state.checkpoint()
    pending_child_bit_count = 5
    cached_node = LeafNode(0.25)

    inheritance_bits[4] = 0
    traversal_state.advance_to(pending_child_bit_count)
    traversal_state.cache_suffix_tree(
        pending_child_bit_count,
        cached_node,
    )
    assert (
        traversal_state.cached_suffix_tree(pending_child_bit_count)
        == cached_node
    )

    traversal_state.rollback(initial_checkpoint)
    inheritance_bits[4] = 1
    traversal_state.advance_to(pending_child_bit_count)

    assert traversal_state.cached_suffix_tree(pending_child_bit_count) is None


def test_closed_component_peeling_restores_constraints_on_rollback() -> None:
    """Keep completed factors outside a canonical future-state cache key."""

    marker, family = _closed_component_cache_family_marker()
    inheritance_bits = [0] * len(family.meioses)
    relevant_individual_ids = (
        likelihood_module._marker_relevant_individual_ids(
            family,
            marker.name,
        )
    )
    traversal_state = likelihood_module._MarkerTraversalState(
        family,
        marker.name,
        inheritance_bits,
        relevant_individual_ids,
    )
    allele_frequencies = likelihood_module._decimal_allele_frequencies(
        family,
        marker,
    )
    initial_checkpoint = traversal_state.checkpoint()
    closed_component_bit_count = 6

    traversal_state.advance_to(closed_component_bit_count)
    before_peeling_checkpoint = traversal_state.checkpoint()
    assert len(traversal_state.constraints()) == 1

    identical_origin_factor = traversal_state.peel_closed_components(
        closed_component_bit_count,
        allele_frequencies,
    )
    identical_origin_key = traversal_state.canonical_future_key(
        closed_component_bit_count
    )
    normalized_suffix = LeafNode(0.75)
    traversal_state.cache_suffix_tree(
        closed_component_bit_count,
        normalized_suffix,
    )

    assert identical_origin_factor == Decimal("0.25")
    assert traversal_state.constraints() == ()

    traversal_state.rollback(before_peeling_checkpoint)
    assert len(traversal_state.constraints()) == 1
    traversal_state.rollback(initial_checkpoint)

    # The second history makes the observed homozygote depend on two distinct
    # founder alleles. Its completed factor differs, but the unrelated future
    # pedigree is unchanged and must therefore share the normalized suffix.
    inheritance_bits[5] = 1
    traversal_state.advance_to(closed_component_bit_count)
    distinct_origin_factor = traversal_state.peel_closed_components(
        closed_component_bit_count,
        allele_frequencies,
    )
    distinct_origin_key = traversal_state.canonical_future_key(
        closed_component_bit_count
    )

    assert distinct_origin_factor == Decimal("0.0625")
    assert distinct_origin_key == identical_origin_key
    assert (
        traversal_state.cached_suffix_tree(closed_component_bit_count)
        == normalized_suffix
    )
    assert traversal_state.peeled_component_count == 2
    assert traversal_state.peeled_constraint_count == 2
    assert traversal_state.normalized_cache_reuse_count == 1


def test_closed_component_peeling_preserves_every_likelihood_value() -> None:
    """Compare incremental component factors with the explicit state oracle."""

    marker, family = _closed_component_cache_family_marker()
    dataset = Dataset(markers=(marker,), families=(family,))
    explicit_result = single_marker_likelihood(dataset, marker.name)
    explicit_values = tuple(
        state.likelihood for state in explicit_result.states
    )

    tree = family_marker_likelihood_tree(family, marker)

    assert tree.dense_values() == explicit_values
    assert tree.build_statistics is not None
    assert tree.build_statistics.peeled_component_count > 0
    assert tree.build_statistics.peeled_constraint_count > 0


def test_boundary_potentials_detect_exact_proportional_histories() -> None:
    """Share suffixes when extra internal founders add only a scale."""

    allele_frequencies = {"1": Decimal("0.5"), "2": Decimal("0.5")}
    one_internal_origin = ((0, 1, "1", "2"),)
    two_internal_origins = (
        (0, 1, "1", "2"),
        (1, 2, "1", "2"),
    )

    first_signature, first_scale = (
        likelihood_module._normalized_component_potential(
            one_internal_origin,
            {0},
            ("1", "2"),
            allele_frequencies,
        )
    )
    second_signature, second_scale = (
        likelihood_module._normalized_component_potential(
            two_internal_origins,
            {0},
            ("1", "2"),
            allele_frequencies,
        )
    )

    assert first_signature == second_signature
    assert first_scale == Decimal("0.5")
    assert second_scale == Decimal("0.25")


def test_peeling_integrates_unobserved_founders_without_enumerating_them() -> None:
    marker = Marker(
        name="marker",
        chromosome="1",
        position_cm=0.0,
        allele_frequencies={"1": 0.5, "2": 0.5},
    )
    individuals = tuple(
        Individual(
            family_id="1",
            individual_id=str(index),
            father_id=None,
            mother_id=None,
            sex="1",
            phenotypes={},
            genotypes={
                marker.name: (("1", "2") if index == 1 else (None, None))
            },
        )
        for index in range(1, 13)
    )
    family = Family(
        family_id="1",
        individuals=individuals,
        meioses=(),
    )
    origins = inheritance_origins(family, ())

    likelihood = peeled_state_likelihood(family, marker, origins)

    assert likelihood == 0.5


def test_marker_tree_prunes_impossible_partial_inheritance_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker, family = _partially_incompatible_family_marker()
    dataset = Dataset(markers=(marker,), families=(family,))
    explicit_result = single_marker_likelihood(dataset, marker.name)
    explicit_values = {
        state.bits: state.likelihood for state in explicit_result.states
    }
    expected_values = tuple(
        explicit_values.get(bits, 0.0)
        for bits in product((0, 1), repeat=len(family.meioses))
    )
    original_evaluator = likelihood_module._peeled_constraints_likelihood
    original_partial_origins = likelihood_module._partial_inheritance_origins
    evaluated_leaf_count = 0
    partial_origin_call_count = 0

    def counted_evaluator(
        constraints: tuple[
            likelihood_module._FounderGenotypeConstraint,
            ...,
        ],
        allele_frequencies: dict[str, Decimal],
    ) -> float:
        nonlocal evaluated_leaf_count
        evaluated_leaf_count += 1
        return original_evaluator(
            constraints,
            allele_frequencies,
        )

    def counted_partial_origins(
        evaluated_family: Family,
        inheritance_bits: list[int],
        assigned_bit_count: int,
    ) -> dict[str, tuple[tuple[str, int], tuple[str, int]]]:
        nonlocal partial_origin_call_count
        partial_origin_call_count += 1
        return original_partial_origins(
            evaluated_family,
            inheritance_bits,
            assigned_bit_count,
        )

    monkeypatch.setattr(
        likelihood_module,
        "_peeled_constraints_likelihood",
        counted_evaluator,
    )
    monkeypatch.setattr(
        likelihood_module,
        "_partial_inheritance_origins",
        counted_partial_origins,
    )

    tree = family_marker_likelihood_tree(family, marker)

    assert tree.dense_values() == expected_values
    # Eight observed-genotype prefixes are compatible. The remaining four
    # transmissions lead only to ungenotyped children and are shared.
    assert 0 < evaluated_leaf_count <= 8
    assert evaluated_leaf_count < 2 ** len(family.meioses)
    assert partial_origin_call_count == 0


def test_marker_tree_shares_branches_without_genotyped_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker, family = _unobserved_descendant_family_marker()
    dataset = Dataset(markers=(marker,), families=(family,))
    explicit_result = single_marker_likelihood(dataset, marker.name)
    expected_values = tuple(
        state.likelihood for state in explicit_result.states
    )
    original_evaluator = likelihood_module.peeled_state_likelihood
    original_constraint_evaluator = (
        likelihood_module._peeled_constraints_likelihood
    )
    original_partial_origins = likelihood_module._partial_inheritance_origins
    evaluated_leaf_count = 0
    evaluated_constraint_count = 0
    partial_origin_call_count = 0

    def counted_evaluator(
        evaluated_family: Family,
        evaluated_marker: Marker,
        allele_origins: dict[
            str,
            tuple[tuple[str, int], tuple[str, int]],
        ],
    ) -> float:
        nonlocal evaluated_leaf_count
        evaluated_leaf_count += 1
        return original_evaluator(
            evaluated_family,
            evaluated_marker,
            allele_origins,
        )

    def counted_constraint_evaluator(
        constraints: tuple[
            likelihood_module._FounderGenotypeConstraint,
            ...,
        ],
        allele_frequencies: dict[str, Decimal],
    ) -> float:
        nonlocal evaluated_constraint_count
        evaluated_constraint_count += 1
        return original_constraint_evaluator(
            constraints,
            allele_frequencies,
        )

    def counted_partial_origins(
        evaluated_family: Family,
        inheritance_bits: list[int],
        assigned_bit_count: int,
    ) -> dict[str, tuple[tuple[str, int], tuple[str, int]]]:
        nonlocal partial_origin_call_count
        partial_origin_call_count += 1
        return original_partial_origins(
            evaluated_family,
            inheritance_bits,
            assigned_bit_count,
        )

    monkeypatch.setattr(
        likelihood_module,
        "peeled_state_likelihood",
        counted_evaluator,
    )
    monkeypatch.setattr(
        likelihood_module,
        "_peeled_constraints_likelihood",
        counted_constraint_evaluator,
    )
    monkeypatch.setattr(
        likelihood_module,
        "_partial_inheritance_origins",
        counted_partial_origins,
    )

    tree = family_marker_likelihood_tree(family, marker)

    assert tree.dense_values() == expected_values
    assert evaluated_leaf_count == 0
    assert evaluated_constraint_count == 1
    assert partial_origin_call_count == 0


def test_marker_tree_incrementally_resolves_non_topological_meiosis_order(
) -> None:
    marker = Marker(
        name="deep_marker",
        chromosome="1",
        position_cm=0.0,
        allele_frequencies={"1": 0.5, "2": 0.5},
    )
    ordered_family = _inbred_parent_family()
    family = Family(
        family_id=ordered_family.family_id,
        individuals=tuple(
            Individual(
                family_id=person.family_id,
                individual_id=person.individual_id,
                father_id=person.father_id,
                mother_id=person.mother_id,
                sex=person.sex,
                phenotypes=person.phenotypes,
                genotypes={
                    marker.name: (
                        ("1", "2")
                        if person.individual_id == "6"
                        else (None, None)
                    )
                },
            )
            for person in ordered_family.individuals
        ),
        meioses=tuple(reversed(ordered_family.meioses)),
    )
    dataset = Dataset(markers=(marker,), families=(family,))
    explicit_result = single_marker_likelihood(dataset, marker.name)
    explicit_values = {
        state.bits: state.likelihood for state in explicit_result.states
    }
    expected_values = tuple(
        explicit_values.get(bits, 0.0)
        for bits in product((0, 1), repeat=len(family.meioses))
    )

    tree = family_marker_likelihood_tree(family, marker)

    assert tree.dense_values() == expected_values


def test_marker_tree_shares_relevant_branch_for_identical_parental_origins(
) -> None:
    family = _inbred_parent_family()
    identical_origin_bits = [0] * len(family.meioses)
    identical_origins = likelihood_module._partial_inheritance_origins(
        family,
        identical_origin_bits,
        assigned_bit_count=6,
    )

    assert likelihood_module._meiosis_is_likelihood_invariant(
        family,
        identical_origins,
        bit_index=6,
        marker_relevant_individual_ids=frozenset({"6"}),
    )

    distinct_origin_bits = identical_origin_bits.copy()
    distinct_origin_bits[5] = 1
    distinct_origins = likelihood_module._partial_inheritance_origins(
        family,
        distinct_origin_bits,
        assigned_bit_count=6,
    )

    assert not likelihood_module._meiosis_is_likelihood_invariant(
        family,
        distinct_origins,
        bit_index=6,
        marker_relevant_individual_ids=frozenset({"6"}),
    )


def _incompatible_family_marker() -> tuple[Marker, Family]:
    marker = Marker(
        name="inconsistent_marker",
        chromosome="1",
        position_cm=0.0,
        allele_frequencies={"1": 0.5, "2": 0.5},
    )
    family = Family(
        family_id="1",
        individuals=(
            Individual(
                family_id="1",
                individual_id="1",
                father_id=None,
                mother_id=None,
                sex="1",
                phenotypes={},
                genotypes={marker.name: (None, None)},
            ),
            Individual(
                family_id="1",
                individual_id="2",
                father_id=None,
                mother_id=None,
                sex="2",
                phenotypes={},
                genotypes={marker.name: ("1", "1")},
            ),
            Individual(
                family_id="1",
                individual_id="3",
                father_id="1",
                mother_id="2",
                sex="1",
                phenotypes={},
                genotypes={marker.name: ("2", "2")},
            ),
        ),
        meioses=(
            Meiosis(parent_id="1", child_id="3", parent_sex="1"),
            Meiosis(parent_id="2", child_id="3", parent_sex="2"),
        ),
    )
    return marker, family


def _partially_incompatible_family_marker() -> tuple[Marker, Family]:
    marker = Marker(
        name="branch_marker",
        chromosome="1",
        position_cm=0.0,
        allele_frequencies={"1": 0.5, "2": 0.5},
    )

    def person(
        individual_id: str,
        father_id: str | None,
        mother_id: str | None,
        genotype: tuple[str | None, str | None],
    ) -> Individual:
        return Individual(
            family_id="1",
            individual_id=individual_id,
            father_id=father_id,
            mother_id=mother_id,
            sex="1",
            phenotypes={},
            genotypes={marker.name: genotype},
        )

    family = Family(
        family_id="1",
        individuals=(
            person("1", None, None, ("1", "2")),
            person("2", None, None, ("1", "1")),
            person("3", "1", "2", ("1", "1")),
            person("4", "1", "2", ("1", "2")),
            person("5", "1", "2", (None, None)),
            person("6", "1", "2", (None, None)),
        ),
        meioses=tuple(
            Meiosis(
                parent_id=parent_id,
                child_id=child_id,
                parent_sex=parent_sex,
            )
            for child_id in ("3", "4", "5", "6")
            for parent_id, parent_sex in (("1", "1"), ("2", "2"))
        ),
    )
    return marker, family


def _unobserved_descendant_family_marker() -> tuple[Marker, Family]:
    marker = Marker(
        name="missing_descendant_marker",
        chromosome="1",
        position_cm=0.0,
        allele_frequencies={"1": 0.5, "2": 0.5},
    )

    def person(
        individual_id: str,
        father_id: str | None,
        mother_id: str | None,
        genotype: tuple[str | None, str | None],
    ) -> Individual:
        return Individual(
            family_id="1",
            individual_id=individual_id,
            father_id=father_id,
            mother_id=mother_id,
            sex="1",
            phenotypes={},
            genotypes={marker.name: genotype},
        )

    family = Family(
        family_id="1",
        individuals=(
            person("1", None, None, ("1", "2")),
            person("2", None, None, ("1", "1")),
            person("3", "1", "2", (None, None)),
            person("4", "1", "2", (None, None)),
        ),
        meioses=tuple(
            Meiosis(
                parent_id=parent_id,
                child_id=child_id,
                parent_sex=parent_sex,
            )
            for child_id in ("3", "4")
            for parent_id, parent_sex in (("1", "1"), ("2", "2"))
        ),
    )
    return marker, family


def _closed_component_cache_family_marker() -> tuple[Marker, Family]:
    """Create one completed inbred component before an unrelated component."""

    marker = Marker(
        name="closed_component_marker",
        chromosome="1",
        position_cm=0.0,
        allele_frequencies={"1": 0.25, "2": 0.75},
    )

    def person(
        individual_id: str,
        father_id: str | None,
        mother_id: str | None,
        genotype: tuple[str | None, str | None],
    ) -> Individual:
        return Individual(
            family_id="1",
            individual_id=individual_id,
            father_id=father_id,
            mother_id=mother_id,
            sex="1",
            phenotypes={},
            genotypes={marker.name: genotype},
        )

    family = Family(
        family_id="1",
        individuals=(
            person("1", None, None, (None, None)),
            person("2", None, None, (None, None)),
            person("3", "1", "2", (None, None)),
            person("4", "1", "2", (None, None)),
            person("5", "3", "4", ("1", "1")),
            person("6", None, None, (None, None)),
            person("7", None, None, (None, None)),
            person("8", "6", "7", ("1", "2")),
        ),
        meioses=(
            Meiosis(parent_id="1", child_id="3", parent_sex="1"),
            Meiosis(parent_id="2", child_id="3", parent_sex="2"),
            Meiosis(parent_id="1", child_id="4", parent_sex="1"),
            Meiosis(parent_id="2", child_id="4", parent_sex="2"),
            Meiosis(parent_id="3", child_id="5", parent_sex="1"),
            Meiosis(parent_id="4", child_id="5", parent_sex="2"),
            Meiosis(parent_id="6", child_id="8", parent_sex="1"),
            Meiosis(parent_id="7", child_id="8", parent_sex="2"),
        ),
    )
    return marker, family


def _inbred_parent_family() -> Family:
    def person(
        individual_id: str,
        father_id: str | None,
        mother_id: str | None,
    ) -> Individual:
        return Individual(
            family_id="1",
            individual_id=individual_id,
            father_id=father_id,
            mother_id=mother_id,
            sex="1",
            phenotypes={},
            genotypes={},
        )

    return Family(
        family_id="1",
        individuals=(
            person("1", None, None),
            person("2", None, None),
            person("3", "1", "2"),
            person("4", "1", "2"),
            person("5", "3", "4"),
            person("7", None, None),
            person("6", "5", "7"),
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
