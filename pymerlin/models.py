"""Core data models for pedigree likelihood calculations."""

from __future__ import annotations

from dataclasses import dataclass


MissingAllele = "0"


@dataclass(frozen=True)
class Marker:
    """A genetic marker from a MERLIN/QTDT data file."""

    name: str
    chromosome: str | None = None
    position_cm: float | None = None
    allele_frequencies: dict[str, float] | None = None


@dataclass(frozen=True)
class Individual:
    """A pedigree member with marker genotypes keyed by marker name."""

    family_id: str
    individual_id: str
    father_id: str | None
    mother_id: str | None
    sex: str
    phenotypes: dict[str, str]
    genotypes: dict[str, tuple[str | None, str | None]]

    @property
    def key(self) -> tuple[str, str]:
        return (self.family_id, self.individual_id)

    @property
    def is_founder(self) -> bool:
        return self.father_id is None and self.mother_id is None


@dataclass(frozen=True)
class Meiosis:
    """One parental allele transmission into a child."""

    parent_id: str
    child_id: str
    parent_sex: str


@dataclass(frozen=True)
class Family:
    """A single family/pedigree, normalized from the input rows."""

    family_id: str
    individuals: tuple[Individual, ...]
    meioses: tuple[Meiosis, ...]

    @property
    def by_id(self) -> dict[str, Individual]:
        return {person.individual_id: person for person in self.individuals}

    @property
    def founders(self) -> tuple[Individual, ...]:
        return tuple(person for person in self.individuals if person.is_founder)


@dataclass(frozen=True)
class Dataset:
    """All parsed input needed for the first likelihood milestone."""

    markers: tuple[Marker, ...]
    families: tuple[Family, ...]
    affection_names: tuple[str, ...] = ()

    @property
    def marker_by_name(self) -> dict[str, Marker]:
        return {marker.name: marker for marker in self.markers}
