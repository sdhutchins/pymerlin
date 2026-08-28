"""Report computationally bounded ancestral PAH subpedigrees.

The report is diagnostic. It does not modify a pedigree, run linkage, or imply
that overlapping affected-pair subpedigrees are statistically independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx

from .information import _merlin_bit_count
from .models import Dataset, Family, Individual, Marker

DEFAULT_MERLIN_BIT_LIMIT = 24


@dataclass(frozen=True)
class AffectedPairSubpedigree:
    """Complexity and retained observations for one ancestral closure."""

    affected_pair: tuple[str, str]
    individual_ids: tuple[str, ...]
    meiosis_count: int
    merlin_bit_count: int
    marker_relevant_meiosis_count: int
    typed_person_count: int
    affected_person_count: int
    relationship_distance: int | None
    connected_component_count: int
    within_bit_limit: bool
    review_candidate: bool


@dataclass(frozen=True)
class PedigreeReductionReport:
    """Full-pedigree context and affected-pair candidate subpedigrees."""

    family_id: str
    affection_name: str
    marker_names: tuple[str, ...]
    bit_limit: int
    full_individual_count: int
    full_meiosis_count: int
    full_merlin_bit_count: int
    full_marker_relevant_meiosis_count: int
    full_typed_person_count: int
    full_affected_person_count: int
    candidates: tuple[AffectedPairSubpedigree, ...]

    @property
    def review_candidates(self) -> tuple[AffectedPairSubpedigree, ...]:
        """Return bounded connected candidates retaining multiple typed people."""

        return tuple(
            candidate for candidate in self.candidates if candidate.review_candidate
        )


def analyze_pedigree_reduction(
    dataset: Dataset,
    family_id: str,
    *,
    affection_name: str | None = None,
    marker_names: tuple[str, ...] | None = None,
    bit_limit: int = DEFAULT_MERLIN_BIT_LIMIT,
) -> PedigreeReductionReport:
    """Enumerate exact affected-pair ancestral closures for manual review.

    A candidate contains each affected pair and every represented ancestor of
    either person. This preserves the parentage inside that candidate. It does
    not preserve marker information or affected people outside the closure.
    """

    if bit_limit < 1:
        raise ValueError("bit_limit must be at least 1.")

    family = _family_by_id(dataset, family_id)
    selected_affection_name = _resolve_affection_name(
        dataset,
        affection_name,
    )
    selected_markers = _resolve_markers(dataset, marker_names)
    selected_marker_names = tuple(marker.name for marker in selected_markers)
    selected_dataset = Dataset(
        markers=selected_markers,
        families=(family,),
        affection_names=dataset.affection_names,
    )

    affected_ids = frozenset(
        person.individual_id
        for person in family.individuals
        if person.phenotypes.get(selected_affection_name) == "2"
    )
    if len(affected_ids) < 2:
        raise ValueError(
            "A pedigree-reduction report requires at least two affected "
            f"people in family {family_id!r}."
        )

    typed_ids = _typed_individual_ids(family, selected_marker_names)
    relationship_graph = _relationship_graph(family)
    full_marker_relevant_meiosis_count = _marker_relevant_meiosis_count(
        family,
        typed_ids,
    )

    candidates = tuple(
        _affected_pair_candidate(
            selected_dataset,
            family,
            affected_pair,
            affected_ids,
            typed_ids,
            relationship_graph,
            bit_limit,
        )
        for affected_pair in combinations(sorted(affected_ids), 2)
    )

    return PedigreeReductionReport(
        family_id=family.family_id,
        affection_name=selected_affection_name,
        marker_names=selected_marker_names,
        bit_limit=bit_limit,
        full_individual_count=len(family.individuals),
        full_meiosis_count=len(family.meioses),
        full_merlin_bit_count=_merlin_bit_count(selected_dataset, family),
        full_marker_relevant_meiosis_count=(full_marker_relevant_meiosis_count),
        full_typed_person_count=len(typed_ids),
        full_affected_person_count=len(affected_ids),
        candidates=candidates,
    )


def format_pedigree_reduction_report(
    report: PedigreeReductionReport,
) -> str:
    """Format a deterministic TSV report with explicit full-pedigree totals."""

    lines = [
        f"# family_id\t{report.family_id}",
        f"# affection_name\t{report.affection_name}",
        f"# marker_count\t{len(report.marker_names)}",
        f"# bit_limit\t{report.bit_limit}",
        f"# full_individuals\t{report.full_individual_count}",
        f"# full_meioses\t{report.full_meiosis_count}",
        f"# full_merlin_bits\t{report.full_merlin_bit_count}",
        (
            "# full_marker_relevant_meioses\t"
            f"{report.full_marker_relevant_meiosis_count}"
        ),
        f"# full_typed_people\t{report.full_typed_person_count}",
        f"# full_affected_people\t{report.full_affected_person_count}",
        f"# review_candidates\t{len(report.review_candidates)}",
        (
            "affected_1\taffected_2\tindividuals\tmeioses\tmerlin_bits\t"
            "marker_relevant_meioses\ttyped_retained\taffected_retained\t"
            "relationship_distance\tconnected_components\twithin_bit_limit\t"
            "review_candidate\tindividual_ids"
        ),
    ]
    for candidate in report.candidates:
        distance = (
            "NA"
            if candidate.relationship_distance is None
            else str(candidate.relationship_distance)
        )
        lines.append(
            "\t".join(
                (
                    candidate.affected_pair[0],
                    candidate.affected_pair[1],
                    str(len(candidate.individual_ids)),
                    str(candidate.meiosis_count),
                    str(candidate.merlin_bit_count),
                    str(candidate.marker_relevant_meiosis_count),
                    str(candidate.typed_person_count),
                    str(candidate.affected_person_count),
                    distance,
                    str(candidate.connected_component_count),
                    _format_boolean(candidate.within_bit_limit),
                    _format_boolean(candidate.review_candidate),
                    ",".join(candidate.individual_ids),
                )
            )
        )
    return "\n".join(lines) + "\n"


def _affected_pair_candidate(
    dataset: Dataset,
    family: Family,
    affected_pair: tuple[str, str],
    affected_ids: frozenset[str],
    typed_ids: frozenset[str],
    relationship_graph: nx.Graph,
    bit_limit: int,
) -> AffectedPairSubpedigree:
    """Build one candidate without treating it as an independent family."""

    selected_ids = _ancestral_individual_ids(family, affected_pair)
    subfamily = _subfamily_from_ancestral_ids(family, selected_ids)
    candidate_dataset = Dataset(
        markers=dataset.markers,
        families=(subfamily,),
        affection_names=dataset.affection_names,
    )
    candidate_typed_ids = typed_ids.intersection(selected_ids)
    merlin_bit_count = _merlin_bit_count(candidate_dataset, subfamily)
    connected_component_count = nx.number_connected_components(
        relationship_graph.subgraph(selected_ids)
    )
    within_bit_limit = merlin_bit_count <= bit_limit
    review_candidate = (
        connected_component_count == 1
        and within_bit_limit
        and len(candidate_typed_ids) >= 2
    )

    return AffectedPairSubpedigree(
        affected_pair=affected_pair,
        individual_ids=tuple(
            person.individual_id
            for person in family.individuals
            if person.individual_id in selected_ids
        ),
        meiosis_count=len(subfamily.meioses),
        merlin_bit_count=merlin_bit_count,
        marker_relevant_meiosis_count=_marker_relevant_meiosis_count(
            subfamily,
            candidate_typed_ids,
        ),
        typed_person_count=len(candidate_typed_ids),
        affected_person_count=len(affected_ids.intersection(selected_ids)),
        relationship_distance=_relationship_distance(
            relationship_graph,
            affected_pair,
        ),
        connected_component_count=connected_component_count,
        within_bit_limit=within_bit_limit,
        review_candidate=review_candidate,
    )


def _family_by_id(dataset: Dataset, family_id: str) -> Family:
    matching_families = tuple(
        family for family in dataset.families if family.family_id == family_id
    )
    if len(matching_families) != 1:
        raise ValueError(f"Dataset does not contain one family {family_id!r}.")
    return matching_families[0]


def _resolve_affection_name(
    dataset: Dataset,
    affection_name: str | None,
) -> str:
    if affection_name is None:
        if len(dataset.affection_names) != 1:
            raise ValueError(
                "affection_name is required unless the dataset defines "
                "exactly one affection phenotype."
            )
        return dataset.affection_names[0]
    if affection_name not in dataset.affection_names:
        raise ValueError(f"Unknown affection phenotype {affection_name!r}.")
    return affection_name


def _resolve_markers(
    dataset: Dataset,
    marker_names: tuple[str, ...] | None,
) -> tuple[Marker, ...]:
    if marker_names is None:
        if not dataset.markers:
            raise ValueError(
                "A pedigree-reduction report requires at least one marker."
            )
        return dataset.markers
    if not marker_names:
        raise ValueError("marker_names must contain at least one marker.")
    if len(set(marker_names)) != len(marker_names):
        raise ValueError("marker_names contains a duplicate marker.")

    markers_by_name = dataset.marker_by_name
    missing_marker_names = set(marker_names).difference(markers_by_name)
    if missing_marker_names:
        raise ValueError(f"Unknown marker(s): {sorted(missing_marker_names)!r}.")
    return tuple(markers_by_name[marker_name] for marker_name in marker_names)


def _typed_individual_ids(
    family: Family,
    marker_names: tuple[str, ...],
) -> frozenset[str]:
    """Return people with a complete genotype at any selected marker."""

    return frozenset(
        person.individual_id
        for person in family.individuals
        if any(
            _has_complete_genotype(person, marker_name) for marker_name in marker_names
        )
    )


def _has_complete_genotype(person: Individual, marker_name: str) -> bool:
    return all(
        allele is not None for allele in person.genotypes.get(marker_name, (None, None))
    )


def _relationship_graph(family: Family) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(person.individual_id for person in family.individuals)
    graph.add_edges_from(
        (meiosis.parent_id, meiosis.child_id) for meiosis in family.meioses
    )
    return graph


def _relationship_distance(
    relationship_graph: nx.Graph,
    affected_pair: tuple[str, str],
) -> int | None:
    try:
        return nx.shortest_path_length(
            relationship_graph,
            source=affected_pair[0],
            target=affected_pair[1],
        )
    except nx.NetworkXNoPath:
        return None


def _ancestral_individual_ids(
    family: Family,
    seed_ids: tuple[str, ...],
) -> frozenset[str]:
    people_by_id = family.by_id
    selected_ids = set(seed_ids)
    pending_ids = list(seed_ids)
    while pending_ids:
        person = people_by_id[pending_ids.pop()]
        for parent_id in (person.father_id, person.mother_id):
            if parent_id not in people_by_id or parent_id in selected_ids:
                continue
            selected_ids.add(parent_id)
            pending_ids.append(parent_id)
    return frozenset(selected_ids)


def _subfamily_from_ancestral_ids(
    family: Family,
    selected_ids: frozenset[str],
) -> Family:
    """Retain complete ancestral parentage in the family's existing order."""

    return Family(
        family_id=family.family_id,
        individuals=tuple(
            person
            for person in family.individuals
            if person.individual_id in selected_ids
        ),
        meioses=tuple(
            meiosis for meiosis in family.meioses if meiosis.child_id in selected_ids
        ),
    )


def _marker_relevant_meiosis_count(
    family: Family,
    typed_ids: frozenset[str],
) -> int:
    if not typed_ids:
        return 0
    relevant_ids = _ancestral_individual_ids(family, tuple(typed_ids))
    return sum(meiosis.child_id in relevant_ids for meiosis in family.meioses)


def _format_boolean(value: bool) -> str:
    return "yes" if value else "no"
