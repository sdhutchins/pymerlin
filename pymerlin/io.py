"""MERLIN-style input readers.

The parser intentionally accepts the small set of classic MERLIN/QTDT files
needed for the reference engine: `.dat`, `.ped`, `.map`, and optional `.freq`.
"""

from __future__ import annotations

from collections import defaultdict
from functools import cmp_to_key
from heapq import heappop, heappush
from pathlib import Path

from .models import Dataset, Family, Individual, Marker, Meiosis


MISSING_TOKENS = {"0", "x", "X", "?", "-9", "-99", "-99.999"}


def load_merlin_inputs(
    ped_path: str | Path,
    dat_path: str | Path,
    map_path: str | Path | None = None,
    freq_path: str | Path | None = None,
    frequency_mode: str = "all",
) -> Dataset:
    """Load MERLIN/QTDT-style input files into normalized Python objects."""

    declared_marker_names, field_specs = _read_dat(dat_path)
    marker_metadata = _read_map(map_path) if map_path is not None else {}
    if map_path is None:
        marker_names = declared_marker_names
        retained_marker_names: set[str] | None = None
    else:
        marker_names = [
            marker_name
            for marker_name in declared_marker_names
            if marker_name in marker_metadata
        ]
        if not marker_names:
            raise ValueError(
                "No markers from the map file were declared in the data file."
            )
        retained_marker_names = set(marker_names)

    frequencies = _read_freq(freq_path) if freq_path is not None else {}
    individuals = sorted(
        _read_ped(
            ped_path,
            field_specs,
            retained_marker_names=retained_marker_names,
        ),
        key=cmp_to_key(_compare_individuals),
    )
    if freq_path is None:
        frequencies = _estimate_marker_frequencies(marker_names, individuals, frequency_mode)

    families: list[Family] = []
    for family_id, family_members in _group_by_family(individuals).items():
        by_id = {person.individual_id: person for person in family_members}
        meioses: list[Meiosis] = []
        for person in _merlin_meiosis_person_order(family_members):
            if person.father_id is not None and person.father_id in by_id:
                meioses.append(
                    Meiosis(
                        person.father_id,
                        person.individual_id,
                        by_id[person.father_id].sex,
                    )
                )
            if person.mother_id is not None and person.mother_id in by_id:
                meioses.append(
                    Meiosis(
                        person.mother_id,
                        person.individual_id,
                        by_id[person.mother_id].sex,
                    )
                )
        families.append(
            Family(
                family_id,
                tuple(family_members),
                tuple(meioses),
            )
        )

    markers = []
    for marker_name in marker_names:
        chromosome, position = marker_metadata.get(marker_name, (None, None))
        markers.append(
            Marker(
                name=marker_name,
                chromosome=chromosome,
                position_cm=position,
                allele_frequencies=frequencies.get(marker_name),
            )
        )
    affection_names = tuple(
        name for field_type, name in field_specs if field_type == "A"
    )
    return Dataset(tuple(markers), tuple(families), affection_names)


def _merlin_meiosis_person_order(
    family_members: list[Individual],
) -> tuple[Individual, ...]:
    """Order transmissions with parents before informative descendants.

    MERLIN separates pedigree display order from its internal traversal path.
    Its traversal keeps parents before children and, among currently eligible
    descendants, prefers people with more genotypes and then affected people.
    Among equally informative people, processing the branch with fewer
    informative descendants first closes its marker constraints sooner. This
    limits the active founder-origin graph without changing individual or
    output ordering.
    """

    people_by_id = {
        person.individual_id: person for person in family_members
    }
    original_index_by_id = {
        person.individual_id: person_index
        for person_index, person in enumerate(family_members)
    }
    in_family_parent_ids_by_person_id = {
        person.individual_id: {
            parent_id
            for parent_id in (person.father_id, person.mother_id)
            if parent_id in people_by_id
        }
        for person in family_members
    }
    child_ids_by_parent_id: dict[str, list[str]] = defaultdict(list)
    for person_id, parent_ids in in_family_parent_ids_by_person_id.items():
        for parent_id in parent_ids:
            child_ids_by_parent_id[parent_id].append(person_id)
    descendant_informativeness_by_person_id = (
        _descendant_informativeness_by_person_id(
            people_by_id,
            in_family_parent_ids_by_person_id,
            child_ids_by_parent_id,
            original_index_by_id,
        )
    )

    resolved_ids = {
        person_id
        for person_id, parent_ids in (
            in_family_parent_ids_by_person_id.items()
        )
        if not parent_ids
    }
    unresolved_parent_count_by_person_id = {
        person_id: len(parent_ids)
        for person_id, parent_ids in (
            in_family_parent_ids_by_person_id.items()
        )
        if parent_ids
    }
    ready_people: list[tuple[int, int, int, str]] = []
    queued_ids: set[str] = set()

    def queue_if_ready(person_id: str) -> None:
        if (
            person_id in queued_ids
            or person_id not in unresolved_parent_count_by_person_id
            or unresolved_parent_count_by_person_id[person_id] != 0
        ):
            return
        person = people_by_id[person_id]
        heappush(
            ready_people,
            (
                -_merlin_person_informativeness(person),
                descendant_informativeness_by_person_id[person_id],
                original_index_by_id[person_id],
                person_id,
            ),
        )
        queued_ids.add(person_id)

    for parent_id in resolved_ids:
        for child_id in child_ids_by_parent_id[parent_id]:
            unresolved_parent_count_by_person_id[child_id] -= 1
            queue_if_ready(child_id)

    ordered_people: list[Individual] = []
    while ready_people:
        _, _, _, person_id = heappop(ready_people)
        queued_ids.remove(person_id)
        if person_id in resolved_ids:
            continue

        person = people_by_id[person_id]
        ordered_people.append(person)
        resolved_ids.add(person_id)
        for child_id in child_ids_by_parent_id[person_id]:
            unresolved_parent_count_by_person_id[child_id] -= 1
            queue_if_ready(child_id)

    unresolved_ids = set(people_by_id).difference(resolved_ids)
    if unresolved_ids:
        raise ValueError(
            "Could not topologically order family members for meiosis "
            f"construction: {sorted(unresolved_ids)!r}."
        )

    return tuple(ordered_people)


def _merlin_person_informativeness(person: Individual) -> int:
    """Return MERLIN's genotype-first traversal priority score."""

    complete_genotype_count = sum(
        all(allele is not None for allele in genotype)
        for genotype in person.genotypes.values()
    )
    first_affection = next(iter(person.phenotypes.values()), None)
    return 2 * complete_genotype_count + int(first_affection == "2")


def _descendant_informativeness_by_person_id(
    people_by_id: dict[str, Individual],
    parent_ids_by_person_id: dict[str, set[str]],
    child_ids_by_parent_id: dict[str, list[str]],
    original_index_by_id: dict[str, int],
) -> dict[str, int]:
    """Count informative descendant paths for traversal tie-breaking.

    A descendant reached through two pedigree paths contributes to both paths,
    matching the allele-copy graph that drives marker-tree complexity. The
    calculation uses a separate topological pass so the priority itself never
    changes which descendants are counted.
    """

    unresolved_parent_counts = {
        person_id: len(parent_ids)
        for person_id, parent_ids in parent_ids_by_person_id.items()
    }
    ready_ids: list[tuple[int, str]] = []
    for person_id, parent_count in unresolved_parent_counts.items():
        if parent_count == 0:
            heappush(
                ready_ids,
                (original_index_by_id[person_id], person_id),
            )
    topological_ids: list[str] = []
    while ready_ids:
        _, person_id = heappop(ready_ids)
        topological_ids.append(person_id)
        for child_id in child_ids_by_parent_id.get(person_id, ()):
            unresolved_parent_counts[child_id] -= 1
            if unresolved_parent_counts[child_id] == 0:
                heappush(
                    ready_ids,
                    (original_index_by_id[child_id], child_id)
                )

    if len(topological_ids) != len(people_by_id):
        unresolved_ids = set(people_by_id).difference(topological_ids)
        raise ValueError(
            "Could not topologically order family members for meiosis "
            f"construction: {sorted(unresolved_ids)!r}."
        )

    descendant_scores: dict[str, int] = {}
    for person_id in reversed(topological_ids):
        descendant_scores[person_id] = sum(
            _merlin_person_informativeness(people_by_id[child_id])
            + descendant_scores[child_id]
            for child_id in child_ids_by_parent_id.get(person_id, ())
        )
    return descendant_scores


def _read_dat(path: str | Path) -> tuple[list[str], list[tuple[str, str]]]:
    marker_names: list[str] = []
    field_specs: list[tuple[str, str]] = []
    marker_name_set: set[str] = set()
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.lower() == "end"
            ):
                continue
            tokens = stripped.split()
            field_type = tokens[0].upper()
            if field_type == "E":
                continue
            if len(tokens) < 2:
                raise ValueError(f"Malformed data-file line: {line!r}")
            name = tokens[1]
            field_specs.append((field_type, name))
            if field_type == "M":
                if name in marker_name_set:
                    raise ValueError(
                        f"Duplicate marker {name!r} in data file."
                    )
                marker_names.append(name)
                marker_name_set.add(name)
    return marker_names, field_specs


def _read_map(path: str | Path) -> dict[str, tuple[str, float]]:
    marker_metadata: dict[str, tuple[str, float]] = {}
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.lower() == "end"
            ):
                continue
            tokens = stripped.split()
            if tokens[0].upper() in {"CHROMOSOME", "CHR"}:
                continue
            if len(tokens) < 3:
                raise ValueError(f"Malformed map-file line: {line!r}")
            marker_name = tokens[1]
            if marker_name in marker_metadata:
                raise ValueError(
                    f"Duplicate marker {marker_name!r} in map file."
                )
            marker_metadata[marker_name] = (tokens[0], float(tokens[2]))
    return marker_metadata


def _read_freq(path: str | Path) -> dict[str, dict[str, float]]:
    frequencies: dict[str, dict[str, float]] = {}
    current_marker: str | None = None
    next_allele = 1
    with Path(path).open() as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            record_type = tokens[0].upper()
            if record_type == "M":
                current_marker = tokens[1]
                frequencies[current_marker] = {}
                next_allele = 1
            elif record_type == "F":
                if current_marker is None:
                    raise ValueError(
                        "Frequency line encountered before a marker line."
                    )
                frequencies[current_marker][str(next_allele)] = float(tokens[1])
                next_allele += 1
    return frequencies


def _read_ped(
    path: str | Path,
    field_specs: list[tuple[str, str]],
    *,
    retained_marker_names: set[str] | None,
) -> list[Individual]:
    individuals: list[Individual] = []
    with Path(path).open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.lower() == "end"
            ):
                continue
            tokens = _split_ped_line(stripped)
            if len(tokens) < 5:
                raise ValueError(
                    f"Malformed pedigree record at line {line_number}."
                )

            family_id, individual_id, father_id, mother_id, sex = tokens[:5]
            cursor = 5
            phenotypes: dict[str, str] = {}
            genotypes: dict[str, tuple[str | None, str | None]] = {}

            for field_type, name in field_specs:
                if field_type == "M":
                    if cursor + 1 >= len(tokens):
                        raise ValueError(
                            "Missing genotype columns for marker "
                            f"{name!r} at pedigree line {line_number}."
                        )
                    if (
                        retained_marker_names is None
                        or name in retained_marker_names
                    ):
                        genotypes[name] = (
                            _normalize_allele(tokens[cursor]),
                            _normalize_allele(tokens[cursor + 1]),
                        )
                    cursor += 2
                else:
                    if cursor >= len(tokens):
                        raise ValueError(
                            "Missing phenotype column "
                            f"{name!r} at pedigree line {line_number}."
                        )
                    phenotypes[name] = tokens[cursor]
                    cursor += 1

            individuals.append(
                Individual(
                    family_id=family_id,
                    individual_id=individual_id,
                    father_id=_normalize_parent_id(father_id),
                    mother_id=_normalize_parent_id(mother_id),
                    sex=sex,
                    phenotypes=phenotypes,
                    genotypes=genotypes,
                )
            )
    return individuals


def _split_ped_line(line: str) -> list[str]:
    return [token.replace("/", "") for token in line.split()]


def _normalize_allele(value: str) -> str | None:
    return None if value in MISSING_TOKENS else value


def _normalize_parent_id(value: str) -> str | None:
    """Normalize the two missing-parent identifiers accepted by MERLIN."""

    return None if value in {"0", "."} else value


def _group_by_family(individuals: list[Individual]) -> dict[str, list[Individual]]:
    grouped: dict[str, list[Individual]] = defaultdict(list)
    for individual in individuals:
        grouped[individual.family_id].append(individual)
    return dict(grouped)


def _compare_individuals(first: Individual, second: Individual) -> int:
    """Order pedigree rows by family and person as MERLIN does."""

    family_comparison = _compare_merlin_identifiers(
        first.family_id,
        second.family_id,
    )
    if family_comparison != 0:
        return family_comparison
    return _compare_merlin_identifiers(
        first.individual_id,
        second.individual_id,
    )


def _compare_merlin_identifiers(first: str, second: str) -> int:
    """Reproduce MERLIN's case-insensitive natural string comparison."""

    first_upper = first.upper()
    second_upper = second.upper()
    index = 0

    while True:
        first_character = _character_at(first_upper, index)
        second_character = _character_at(second_upper, index)
        if first_character == second_character:
            if first_character == "\0":
                return 0
            index += 1
            continue

        digit_index = index
        while _is_ascii_digit(
            _character_at(first_upper, digit_index)
        ) and _is_ascii_digit(_character_at(second_upper, digit_index)):
            digit_index += 1

        if _is_ascii_digit(_character_at(first_upper, digit_index)):
            return 1
        if _is_ascii_digit(_character_at(second_upper, digit_index)):
            return -1
        return ord(first_character) - ord(second_character)


def _character_at(value: str, index: int) -> str:
    return value[index] if index < len(value) else "\0"


def _is_ascii_digit(value: str) -> bool:
    # MERLIN applies the C isdigit function to classic pedigree identifiers.
    return "0" <= value <= "9"


def _estimate_marker_frequencies(
    marker_names: list[str],
    individuals: list[Individual],
    mode: str,
) -> dict[str, dict[str, float]]:
    normalized_mode = mode.lower()
    if normalized_mode in {"all", "a"}:
        return {
            marker_name: _frequency_from_counts(_allele_counts(individuals, marker_name))
            for marker_name in marker_names
        }
    if normalized_mode in {"founders", "f"}:
        return {
            marker_name: _founder_frequency(individuals, marker_name)
            for marker_name in marker_names
        }
    if normalized_mode in {"equal", "e"}:
        return {
            marker_name: _equal_frequency(_allele_counts(individuals, marker_name))
            for marker_name in marker_names
        }
    if normalized_mode in {"ml", "m"}:
        raise NotImplementedError("Maximum-likelihood allele frequency estimation (-fm) is not implemented yet.")
    raise ValueError(f"Unknown allele frequency mode: {mode!r}")


def _founder_frequency(individuals: list[Individual], marker_name: str) -> dict[str, float]:
    all_counts = _allele_counts(individuals, marker_name)
    founder_counts = _allele_counts(
        [individual for individual in individuals if individual.is_founder],
        marker_name,
    )
    if sum(founder_counts.values()) == 0:
        return _frequency_from_counts(all_counts)
    for allele, count in all_counts.items():
        if count > 0 and founder_counts.get(allele, 0) == 0:
            founder_counts[allele] = 1
    return _frequency_from_counts(founder_counts)


def _allele_counts(individuals: list[Individual], marker_name: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for individual in individuals:
        for allele in individual.genotypes.get(marker_name, (None, None)):
            if allele is not None:
                counts[allele] += 1
    return dict(counts)


def _frequency_from_counts(counts: dict[str, int]) -> dict[str, float]:
    if not counts:
        return {"1": 0.99999, "2": 0.00001}
    total = sum(counts.values())
    return {allele: count / total for allele, count in sorted(counts.items())}


def _equal_frequency(counts: dict[str, int]) -> dict[str, float]:
    if not counts:
        return {"1": 0.99999, "2": 0.00001}
    alleles = sorted(counts)
    frequency = 1.0 / len(alleles)
    return {allele: frequency for allele in alleles}
