"""IBD and kinship summaries from posterior inheritance states."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import fsum

from .founder_symmetry import (
    FounderOrientationSymmetryPlan,
    build_founder_couple_symmetry_plan,
    build_founder_orientation_symmetry_plan,
    restore_founder_couple_symmetry_branches,
    restore_founder_orientation_branch,
)
from .inheritance_tree import (
    InheritanceTree,
    LeafNode,
    TreeNode,
    ZeroNode,
    _combine_children,
    _inheritance_recursion_budget,
)
from .likelihood import inheritance_origins, single_marker_likelihood
from .models import Dataset, Family
from .selection import select_markers


@dataclass(frozen=True)
class IbdResult:
    """Pairwise IBD summary for one marker."""

    marker_name: str
    rows: tuple[dict[str, float | str], ...]


def estimate_ibd(
    dataset: Dataset,
    marker_id: str,
    backend: str = "numpy",
    workers: int = 1,
) -> IbdResult:
    """Estimate pairwise IBD state probabilities and kinship for one marker."""

    likelihood = single_marker_likelihood(
        dataset,
        marker_id,
        backend=backend,
        workers=workers,
    )
    family_people = {
        family.family_id: tuple(person.individual_id for person in family.individuals)
        for family in dataset.families
    }
    accumulators: dict[
        tuple[str, str, str],
        dict[str, list[float]],
    ] = defaultdict(
        lambda: {"z0": [], "z1": [], "z2": []}
    )

    for state in likelihood.states:
        people = family_people[state.family_id]
        for index, first_id in enumerate(people):
            for second_id in people[index + 1 :]:
                shared = _shared_allele_count(
                    state.allele_origins[first_id],
                    state.allele_origins[second_id],
                )
                pair_key = (state.family_id, first_id, second_id)
                accumulators[pair_key][f"z{shared}"].append(
                    state.posterior_weight
                )

    rows = []
    for (family_id, first_id, second_id), values in sorted(accumulators.items()):
        z0 = fsum(values["z0"])
        z1 = fsum(values["z1"])
        z2 = fsum(values["z2"])
        total = fsum((z0, z1, z2))
        if total <= 0.0:
            raise ValueError(
                "Single-point IBD probabilities have a non-positive total for "
                f"family {family_id!r}, pair ({first_id!r}, {second_id!r})."
            )

        # The three IBD states partition the posterior probability space.
        z0, z1, z2 = (value / total for value in (z0, z1, z2))
        pi_hat = 0.5 * z1 + z2
        rows.append(
            {
                "family_id": family_id,
                "id1": first_id,
                "id2": second_id,
                "z0": z0,
                "z1": z1,
                "z2": z2,
                "pi_hat": pi_hat,
                "kinship": 0.5 * pi_hat,
            }
        )
    return IbdResult(marker_name=marker_id, rows=tuple(rows))


def estimate_ibd_for_markers(
    dataset: Dataset,
    marker_names: list[str] | None = None,
    chromosome: str | None = None,
    start_cm: float | None = None,
    end_cm: float | None = None,
    backend: str = "numpy",
    workers: int = 1,
) -> tuple[IbdResult, ...]:
    """Estimate IBD for explicit markers or a genetic-map interval."""

    markers = select_markers(dataset, marker_names, chromosome, start_cm, end_cm)
    return tuple(
        estimate_ibd(
            dataset,
            marker.name,
            backend=backend,
            workers=workers,
        )
        for marker in markers
    )


def _shared_allele_count(
    first: tuple[tuple[str, int], tuple[str, int]],
    second: tuple[tuple[str, int], tuple[str, int]],
) -> int:
    return len(set(first).intersection(second))


def _pair_ibd_indicator_trees(
    family: Family,
    first_id: str,
    second_id: str,
) -> tuple[InheritanceTree, InheritanceTree, InheritanceTree]:
    """Build compressed indicators for a pair's three IBD states."""

    people_by_id = family.by_id
    missing_ids = {
        individual_id
        for individual_id in (first_id, second_id)
        if individual_id not in people_by_id
    }
    if missing_ids:
        missing = ", ".join(sorted(missing_ids))
        raise ValueError(
            f"Unknown individual(s) in family {family.family_id!r}: {missing}"
        )
    if first_id == second_id:
        raise ValueError("Pairwise IBD indicators require two distinct people.")

    relevant_meiosis_indices = _pair_ancestral_meiosis_indices(
        family,
        first_id,
        second_id,
    )
    inheritance_bits = [0] * len(family.meioses)
    founder_symmetry_plan = build_founder_orientation_symmetry_plan(
        family,
        relevant_meiosis_indices,
    )
    founder_couple_symmetry_plan = build_founder_couple_symmetry_plan(
        family,
        relevant_meiosis_indices,
    ).for_ibd_pair(first_id, second_id)
    founder_couple_representative_indices = (
        founder_couple_symmetry_plan.representative_bit_indices
    )
    with _inheritance_recursion_budget(len(family.meioses)):
        roots = _pair_ibd_indicator_nodes(
            family,
            first_id,
            second_id,
            relevant_meiosis_indices,
            inheritance_bits,
            bit_index=0,
            founder_symmetry_plan=founder_symmetry_plan,
            founder_couple_representative_indices=(
                founder_couple_representative_indices
            ),
        )
        roots = tuple(
            restore_founder_couple_symmetry_branches(
                root,
                founder_couple_symmetry_plan,
            )
            for root in roots
        )
    return tuple(
        InheritanceTree(
            bit_count=len(family.meioses),
            root=root,
        )
        for root in roots
    )


def _pair_ancestral_meiosis_indices(
    family: Family,
    first_id: str,
    second_id: str,
) -> frozenset[int]:
    """Return transmissions that can affect either member of one pair."""

    return _ancestral_meiosis_indices(family, (first_id, second_id))


def _ancestral_meiosis_indices(
    family: Family,
    individual_ids: tuple[str, ...],
) -> frozenset[int]:
    """Return transmissions that can affect the requested individuals."""

    people_by_id = family.by_id
    missing_ids = set(individual_ids).difference(people_by_id)
    if missing_ids:
        missing = ", ".join(sorted(missing_ids))
        raise ValueError(
            f"Unknown individual(s) in family {family.family_id!r}: {missing}"
        )

    ancestral_ids = set(individual_ids)
    unresolved_ancestors = list(individual_ids)
    while unresolved_ancestors:
        person = people_by_id[unresolved_ancestors.pop()]
        for parent_id in (person.father_id, person.mother_id):
            if parent_id not in people_by_id or parent_id in ancestral_ids:
                continue
            ancestral_ids.add(parent_id)
            unresolved_ancestors.append(parent_id)

    return frozenset(
        index
        for index, meiosis in enumerate(family.meioses)
        if meiosis.child_id in ancestral_ids
    )


def _pair_ibd_indicator_nodes(
    family: Family,
    first_id: str,
    second_id: str,
    relevant_meiosis_indices: frozenset[int],
    inheritance_bits: list[int],
    bit_index: int,
    founder_symmetry_plan: FounderOrientationSymmetryPlan,
    founder_couple_representative_indices: frozenset[int],
) -> tuple[TreeNode, TreeNode, TreeNode]:
    """Score all three IBD indicators in one inheritance-tree traversal."""

    if bit_index == len(inheritance_bits):
        origins = inheritance_origins(family, tuple(inheritance_bits))
        shared_count = _shared_allele_count(
            origins[first_id],
            origins[second_id],
        )
        return tuple(
            LeafNode(1.0) if ibd_state == shared_count else ZeroNode()
            for ibd_state in range(3)
        )

    inheritance_bits[bit_index] = 0
    zero_children = _pair_ibd_indicator_nodes(
        family,
        first_id,
        second_id,
        relevant_meiosis_indices,
        inheritance_bits,
        bit_index + 1,
        founder_symmetry_plan,
        founder_couple_representative_indices,
    )
    if bit_index not in relevant_meiosis_indices:
        return tuple(
            _combine_children(child, child) for child in zero_children
        )

    if (
        bit_index
        in founder_couple_representative_indices
    ):
        return tuple(
            _combine_children(child, child) for child in zero_children
        )

    founder_flip_indices = founder_symmetry_plan.descendant_flip_indices(
        bit_index
    )
    if founder_flip_indices is not None:
        return tuple(
            restore_founder_orientation_branch(
                child,
                bit_index,
                founder_flip_indices,
            )
            for child in zero_children
        )

    inheritance_bits[bit_index] = 1
    one_children = _pair_ibd_indicator_nodes(
        family,
        first_id,
        second_id,
        relevant_meiosis_indices,
        inheritance_bits,
        bit_index + 1,
        founder_symmetry_plan,
        founder_couple_representative_indices,
    )
    return tuple(
        _combine_children(zero_child, one_child)
        for zero_child, one_child in zip(zero_children, one_children)
    )
