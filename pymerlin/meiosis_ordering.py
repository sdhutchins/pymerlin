"""Exact coordinate permutations for inheritance-vector ordering audits.

Changing meiosis order changes only the coordinate layout of an inheritance
vector. This module keeps the pedigree and every meiosis unchanged while
returning explicit forward and inverse permutations. It supports bounded
comparisons of decision-DAG variable order without changing scientific output.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Literal

from .models import Family, Meiosis

MeiosisOrderingName = Literal[
    "current",
    "individual_identifier",
    "parent_before_child",
]


@dataclass(frozen=True)
class OrderedFamilyMeioses:
    """A family with one explicit permutation of its meiosis coordinates."""

    ordering_name: MeiosisOrderingName
    family: Family
    source_index_by_ordered_index: tuple[int, ...]
    ordered_index_by_source_index: tuple[int, ...]

    def to_ordered_bits(self, source_bits: tuple[int, ...]) -> tuple[int, ...]:
        """Permute source inheritance bits into this ordering."""

        _validate_bits(source_bits, len(self.source_index_by_ordered_index))
        return tuple(
            source_bits[source_index]
            for source_index in self.source_index_by_ordered_index
        )

    def to_source_bits(self, ordered_bits: tuple[int, ...]) -> tuple[int, ...]:
        """Restore ordered inheritance bits to the source family order."""

        _validate_bits(ordered_bits, len(self.ordered_index_by_source_index))
        return tuple(
            ordered_bits[ordered_index]
            for ordered_index in self.ordered_index_by_source_index
        )


def order_family_meioses(
    family: Family,
    ordering_name: MeiosisOrderingName,
) -> OrderedFamilyMeioses:
    """Return a coordinate-permuted family without changing its meioses."""

    if ordering_name == "current":
        source_indices = tuple(range(len(family.meioses)))
    elif ordering_name == "individual_identifier":
        source_indices = _individual_identifier_order(family)
    elif ordering_name == "parent_before_child":
        source_indices = _parent_before_child_order(family)
    else:
        raise ValueError(f"Unsupported meiosis ordering: {ordering_name!r}.")

    _validate_source_indices(source_indices, len(family.meioses))
    ordered_index_by_source_index = [0] * len(source_indices)
    for ordered_index, source_index in enumerate(source_indices):
        ordered_index_by_source_index[source_index] = ordered_index
    ordered_family = Family(
        family_id=family.family_id,
        individuals=family.individuals,
        meioses=tuple(family.meioses[index] for index in source_indices),
    )
    return OrderedFamilyMeioses(
        ordering_name=ordering_name,
        family=ordered_family,
        source_index_by_ordered_index=source_indices,
        ordered_index_by_source_index=tuple(ordered_index_by_source_index),
    )


def _individual_identifier_order(family: Family) -> tuple[int, ...]:
    """Group transmissions by the family's normalized individual order."""

    person_index_by_id = {
        person.individual_id: person_index
        for person_index, person in enumerate(family.individuals)
    }
    return tuple(
        source_index
        for source_index, _ in sorted(
            enumerate(family.meioses),
            key=lambda indexed_meiosis: _meiosis_sort_key(
                indexed_meiosis,
                person_index_by_id,
            ),
        )
    )


def _parent_before_child_order(family: Family) -> tuple[int, ...]:
    """Use stable topological people order before grouping transmissions."""

    person_index_by_id = {
        person.individual_id: person_index
        for person_index, person in enumerate(family.individuals)
    }
    missing_child_ids = sorted(
        {
            meiosis.child_id
            for meiosis in family.meioses
            if meiosis.child_id not in person_index_by_id
        }
    )
    if missing_child_ids:
        raise ValueError(
            "Meiosis children are absent from family members: "
            f"{missing_child_ids!r}."
        )
    child_ids_by_parent_id: dict[str, set[str]] = {}
    in_family_parent_ids_by_child_id: dict[str, set[str]] = {
        person.individual_id: set() for person in family.individuals
    }
    for meiosis in family.meioses:
        if meiosis.parent_id not in person_index_by_id:
            continue
        in_family_parent_ids_by_child_id.setdefault(
            meiosis.child_id,
            set(),
        ).add(meiosis.parent_id)
        child_ids_by_parent_id.setdefault(meiosis.parent_id, set()).add(
            meiosis.child_id
        )

    unresolved_parent_count_by_id = {
        person_id: len(parent_ids)
        for person_id, parent_ids in in_family_parent_ids_by_child_id.items()
    }
    ready_people: list[tuple[int, str]] = []
    for person_id, unresolved_parent_count in (
        unresolved_parent_count_by_id.items()
    ):
        if unresolved_parent_count == 0:
            heappush(
                ready_people,
                (person_index_by_id.get(person_id, len(family.individuals)), person_id),
            )

    topological_index_by_id: dict[str, int] = {}
    while ready_people:
        _, person_id = heappop(ready_people)
        if person_id in topological_index_by_id:
            continue
        topological_index_by_id[person_id] = len(topological_index_by_id)
        for child_id in child_ids_by_parent_id.get(person_id, ()):
            unresolved_parent_count_by_id[child_id] -= 1
            if unresolved_parent_count_by_id[child_id] == 0:
                heappush(
                    ready_people,
                    (
                        person_index_by_id.get(
                            child_id,
                            len(family.individuals),
                        ),
                        child_id,
                    ),
                )

    if len(topological_index_by_id) != len(unresolved_parent_count_by_id):
        unresolved_ids = sorted(
            set(unresolved_parent_count_by_id).difference(
                topological_index_by_id
            )
        )
        raise ValueError(
            "Could not topologically order family meioses: "
            f"{unresolved_ids!r}."
        )

    return tuple(
        source_index
        for source_index, _ in sorted(
            enumerate(family.meioses),
            key=lambda indexed_meiosis: _meiosis_sort_key(
                indexed_meiosis,
                topological_index_by_id,
            ),
        )
    )


def _meiosis_sort_key(
    indexed_meiosis: tuple[int, Meiosis],
    child_index_by_id: dict[str, int],
) -> tuple[int, int, int]:
    """Return deterministic child, parental-sex, and source tie breakers."""

    source_index, meiosis = indexed_meiosis
    child_index = child_index_by_id.get(meiosis.child_id)
    if child_index is None:
        raise ValueError(
            f"Meiosis child is absent from family members: {meiosis.child_id!r}."
        )
    parental_sex_order = 0 if meiosis.parent_sex == "1" else 1
    return child_index, parental_sex_order, source_index


def _validate_source_indices(
    source_indices: tuple[int, ...],
    expected_count: int,
) -> None:
    """Require an exact permutation of every source coordinate."""

    if tuple(sorted(source_indices)) != tuple(range(expected_count)):
        raise ValueError("Meiosis ordering must retain every source index once.")


def _validate_bits(bits: tuple[int, ...], expected_count: int) -> None:
    """Validate one binary inheritance vector before coordinate permutation."""

    if len(bits) != expected_count:
        raise ValueError("Inheritance vector has the wrong number of bits.")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Inheritance vector must contain only zero and one.")
