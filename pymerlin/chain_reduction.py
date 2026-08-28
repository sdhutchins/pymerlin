"""Exact counting partitions for untyped pedigree chains.

Geiger, Meek, and Wexler showed that selected meioses along an untyped chain
can be partitioned by the number of selectors that transmit the chain allele.
For ``r`` selectors, the quotient has ``r + 1`` states instead of ``2 ** r``.
This module defines the quotient and conservatively detects pedigree paths for
which PyMerlin's marker-likelihood and affected-only analyses cannot
distinguish the internal people.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from itertools import product

from .models import Family, Individual


_DECIMAL_PRECISION = 80


@dataclass(frozen=True)
class UntypedChain:
    """One exact counting cluster embedded in a pedigree."""

    entry_person_id: str
    internal_person_ids: tuple[str, ...]
    endpoint_person_id: str
    retained_entry_bit_index: int
    selector_bit_indices: tuple[int, ...]
    on_values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.selector_bit_indices) != len(self.on_values):
            raise ValueError("One on value is required per chain selector.")
        if len(self.selector_bit_indices) < 2:
            raise ValueError(
                "An untyped-chain reduction requires at least two selectors."
            )
        if len(set(self.selector_bit_indices)) != len(
            self.selector_bit_indices
        ):
            raise ValueError("Untyped-chain selector bits must be distinct.")
        if any(on_value not in (0, 1) for on_value in self.on_values):
            raise ValueError("Untyped-chain on values must be zero or one.")

    @property
    def selector_count(self) -> int:
        """Return the number of binary selectors represented by this chain."""

        return len(self.selector_bit_indices)

    @property
    def class_count(self) -> int:
        """Return the number of exact counting classes."""

        return self.selector_count + 1

    def on_count(self, inheritance_bits: tuple[int, ...]) -> int:
        """Map one full inheritance vector to this chain's counting class."""

        if any(
            bit_index >= len(inheritance_bits)
            for bit_index in self.selector_bit_indices
        ):
            raise ValueError(
                "Inheritance vector is shorter than an untyped-chain bit."
            )
        return sum(
            inheritance_bits[bit_index] == on_value
            for bit_index, on_value in zip(
                self.selector_bit_indices,
                self.on_values,
            )
        )


@dataclass(frozen=True)
class CountingPartition:
    """The exact ``r + 1`` counting quotient for homogeneous selectors."""

    on_values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.on_values:
            raise ValueError("A counting partition requires at least one selector.")
        if any(on_value not in (0, 1) for on_value in self.on_values):
            raise ValueError("Counting-partition on values must be zero or one.")

    @classmethod
    def from_chain(cls, chain: UntypedChain) -> CountingPartition:
        """Build the counting quotient associated with one detected chain."""

        return cls(on_values=chain.on_values)

    @property
    def selector_count(self) -> int:
        """Return the number of original binary selectors."""

        return len(self.on_values)

    @property
    def class_count(self) -> int:
        """Return the number of quotient states."""

        return self.selector_count + 1

    @property
    def multiplicities(self) -> tuple[int, ...]:
        """Return the number of binary vectors in each counting class."""

        return tuple(
            math.comb(self.selector_count, on_count)
            for on_count in range(self.class_count)
        )

    @property
    def prior_probabilities(self) -> tuple[float, ...]:
        """Aggregate the uniform inheritance prior by counting class."""

        return tuple(
            math.ldexp(multiplicity, -self.selector_count)
            for multiplicity in self.multiplicities
        )

    @property
    def representative_vectors(self) -> tuple[tuple[int, ...], ...]:
        """Return one deterministic selector vector for every class."""

        representatives = []
        for on_count in range(self.class_count):
            representatives.append(
                tuple(
                    on_value if selector_index < on_count else 1 - on_value
                    for selector_index, on_value in enumerate(self.on_values)
                )
            )
        return tuple(representatives)

    def class_index(self, selector_bits: tuple[int, ...]) -> int:
        """Return the on-count class for one chain selector vector."""

        if len(selector_bits) != self.selector_count:
            raise ValueError(
                "Selector-vector length does not match the counting partition."
            )
        if any(bit not in (0, 1) for bit in selector_bits):
            raise ValueError("Selector values must be zero or one.")
        return sum(
            bit == on_value
            for bit, on_value in zip(selector_bits, self.on_values)
        )

    def transition_matrix(
        self,
        recombination_fraction: float,
    ) -> tuple[tuple[float, ...], ...]:
        """Aggregate independent selector transitions between count classes."""

        return tuple(
            tuple(float(probability) for probability in row)
            for row in self.decimal_transition_matrix(recombination_fraction)
        )

    def decimal_transition_matrix(
        self,
        recombination_fraction: float,
    ) -> tuple[tuple[Decimal, ...], ...]:
        """Return count transitions before their final float64 rounding."""

        theta = float(recombination_fraction)
        if not math.isfinite(theta) or not 0.0 <= theta <= 0.5:
            raise ValueError(
                "Recombination fraction must be finite and between 0 and 0.5."
            )

        with localcontext() as decimal_context:
            decimal_context.prec = _DECIMAL_PRECISION
            decimal_theta = Decimal.from_float(theta)
            decimal_complement = Decimal(1) - decimal_theta
            return tuple(
                tuple(
                    _count_transition_probability(
                        self.selector_count,
                        previous_on_count,
                        next_on_count,
                        decimal_theta,
                        decimal_complement,
                    )
                    for next_on_count in range(self.class_count)
                )
                for previous_on_count in range(self.class_count)
            )


def detect_untyped_chains(family: Family) -> tuple[UntypedChain, ...]:
    """Detect disjoint chains satisfying a conservative exactness criterion.

    Each internal person must have no observed marker allele, must not be
    affected, and must have exactly one child. Every side parent must be an
    unobserved, unaffected founder whose only child is the chain person. These
    restrictions ensure that an off selector introduces an exchangeable random
    founder allele and cannot create a distinguishable side-path correlation.
    """

    people_by_id = family.by_id
    founder_ids = {
        founder.individual_id for founder in family.founders
    }
    children_by_parent_id: dict[str, list[str]] = {
        person.individual_id: [] for person in family.individuals
    }
    for person in family.individuals:
        for parent_id in (person.father_id, person.mother_id):
            if parent_id in children_by_parent_id:
                children_by_parent_id[parent_id].append(person.individual_id)

    meiosis_index_by_parent_child = {
        (meiosis.parent_id, meiosis.child_id): bit_index
        for bit_index, meiosis in enumerate(family.meioses)
    }
    used_selector_indices: set[int] = set()
    chains: list[UntypedChain] = []

    for first_internal_person in family.individuals:
        first_internal_id = first_internal_person.individual_id
        if not _is_chain_internal_person(
            first_internal_person,
            children_by_parent_id,
        ):
            continue

        entry_candidates = []
        parent_ids = (
            first_internal_person.father_id,
            first_internal_person.mother_id,
        )
        for entry_person_id, side_parent_id in (
            (parent_ids[0], parent_ids[1]),
            (parent_ids[1], parent_ids[0]),
        ):
            if entry_person_id not in people_by_id:
                continue
            if not _is_exchangeable_side_parent(
                side_parent_id,
                first_internal_id,
                people_by_id,
                founder_ids,
                children_by_parent_id,
            ):
                continue
            entry_candidates.append(
                (
                    meiosis_index_by_parent_child[
                        (entry_person_id, first_internal_id)
                    ],
                    entry_person_id,
                )
            )
        if not entry_candidates:
            continue

        retained_entry_bit_index, entry_person_id = min(entry_candidates)
        chain = _trace_untyped_chain(
            family,
            first_internal_id,
            entry_person_id,
            retained_entry_bit_index,
            people_by_id,
            founder_ids,
            children_by_parent_id,
            meiosis_index_by_parent_child,
        )
        if chain is None:
            continue
        if tuple(sorted(chain.selector_bit_indices)) != (
            chain.selector_bit_indices
        ):
            # The current incremental scorer assigns a prefix in meiosis
            # order. A genealogical chain that runs backward in that order is
            # still exact, but cannot use this reduction without reordering
            # the family's public inheritance coordinates.
            continue
        if used_selector_indices.intersection(chain.selector_bit_indices):
            continue
        used_selector_indices.update(chain.selector_bit_indices)
        chains.append(chain)

    return tuple(
        sorted(chains, key=lambda chain: chain.selector_bit_indices[0])
    )


def _trace_untyped_chain(
    family: Family,
    first_internal_id: str,
    entry_person_id: str,
    retained_entry_bit_index: int,
    people_by_id: dict[str, Individual],
    founder_ids: set[str],
    children_by_parent_id: dict[str, list[str]],
    meiosis_index_by_parent_child: dict[tuple[str, str], int],
) -> UntypedChain | None:
    """Follow one candidate path until it reaches a distinguishing boundary."""

    del family
    previous_person_id = entry_person_id
    current_person_id = first_internal_id
    internal_person_ids: list[str] = []
    selector_bit_indices: list[int] = []
    on_values: list[int] = []

    while True:
        current_person = people_by_id[current_person_id]
        child_ids = children_by_parent_id[current_person_id]
        if len(child_ids) != 1:
            return None
        next_person_id = child_ids[0]
        selector_bit_indices.append(
            meiosis_index_by_parent_child[(current_person_id, next_person_id)]
        )
        on_values.append(
            0 if current_person.father_id == previous_person_id else 1
        )
        internal_person_ids.append(current_person_id)

        next_person = people_by_id[next_person_id]
        if not _is_chain_internal_person(
            next_person,
            children_by_parent_id,
        ):
            endpoint_person_id = next_person_id
            break

        next_parent_ids = (next_person.father_id, next_person.mother_id)
        if current_person_id == next_parent_ids[0]:
            side_parent_id = next_parent_ids[1]
        elif current_person_id == next_parent_ids[1]:
            side_parent_id = next_parent_ids[0]
        else:
            return None
        if not _is_exchangeable_side_parent(
            side_parent_id,
            next_person_id,
            people_by_id,
            founder_ids,
            children_by_parent_id,
        ):
            return None

        previous_person_id = current_person_id
        current_person_id = next_person_id

    if len(selector_bit_indices) < 2:
        return None
    return UntypedChain(
        entry_person_id=entry_person_id,
        internal_person_ids=tuple(internal_person_ids),
        endpoint_person_id=endpoint_person_id,
        retained_entry_bit_index=retained_entry_bit_index,
        selector_bit_indices=tuple(selector_bit_indices),
        on_values=tuple(on_values),
    )


def _is_chain_internal_person(
    person: Individual,
    children_by_parent_id: dict[str, list[str]],
) -> bool:
    """Return whether a person can be hidden inside an exact chain."""

    return (
        _is_unobserved_and_not_affected(person)
        and len(children_by_parent_id[person.individual_id]) == 1
    )


def _is_exchangeable_side_parent(
    person_id: str | None,
    child_id: str,
    people_by_id: dict[str, Individual],
    founder_ids: set[str],
    children_by_parent_id: dict[str, list[str]],
) -> bool:
    """Return whether a side parent contributes an independent random allele."""

    if person_id not in founder_ids:
        return False
    person = people_by_id[person_id]
    return (
        _is_unobserved_and_not_affected(person)
        and children_by_parent_id[person_id] == [child_id]
    )


def _is_unobserved_and_not_affected(person: Individual) -> bool:
    """Return whether current scientific outputs cannot distinguish a person."""

    has_observed_allele = any(
        allele is not None
        for genotype in person.genotypes.values()
        for allele in genotype
    )
    is_affected = any(value == "2" for value in person.phenotypes.values())
    return not has_observed_allele and not is_affected


def _count_transition_probability(
    selector_count: int,
    previous_on_count: int,
    next_on_count: int,
    theta: Decimal,
    complement: Decimal,
) -> Decimal:
    """Return one exact counting-class transition before float64 rounding."""

    minimum_surviving_on = max(
        0,
        previous_on_count + next_on_count - selector_count,
    )
    maximum_surviving_on = min(previous_on_count, next_on_count)
    terms = []
    for surviving_on_count in range(
        minimum_surviving_on,
        maximum_surviving_on + 1,
    ):
        newly_on_count = next_on_count - surviving_on_count
        recombination_count = (
            previous_on_count
            + next_on_count
            - 2 * surviving_on_count
        )
        nonrecombination_count = selector_count - recombination_count
        path_count = math.comb(
            previous_on_count,
            surviving_on_count,
        ) * math.comb(
            selector_count - previous_on_count,
            newly_on_count,
        )
        terms.append(
            Decimal(path_count)
            * theta**recombination_count
            * complement**nonrecombination_count
        )
    return sum(terms, start=Decimal(0))


def dense_count_transition_probability(
    previous_selector_bits: tuple[int, ...],
    next_on_count: int,
    on_values: tuple[int, ...],
    recombination_fraction: float,
) -> float:
    """Enumerate one count transition as a small validation oracle."""

    if len(previous_selector_bits) != len(on_values):
        raise ValueError("Dense count-transition vectors must have equal length.")
    theta = float(recombination_fraction)
    probabilities = []
    for next_selector_bits in product(
        (0, 1),
        repeat=len(previous_selector_bits),
    ):
        observed_on_count = sum(
            bit == on_value
            for bit, on_value in zip(next_selector_bits, on_values)
        )
        if observed_on_count != next_on_count:
            continue
        recombination_count = sum(
            previous_bit != next_bit
            for previous_bit, next_bit in zip(
                previous_selector_bits,
                next_selector_bits,
            )
        )
        probabilities.append(
            theta**recombination_count
            * (1.0 - theta)
            ** (len(previous_selector_bits) - recombination_count)
        )
    return math.fsum(probabilities)
