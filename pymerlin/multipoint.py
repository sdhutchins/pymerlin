"""Small-pedigree multipoint IBD reference implementation.

The reference engine uses explicit inheritance states and a forward-backward
calculation. Its purpose is to establish correct results for small pedigrees
before introducing MERLIN's sparse gene-flow tree optimizations.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, localcontext
from itertools import product, repeat
from math import fsum
from typing import TypeVar

from .backends import MultipointEngine, validate_multipoint_engine
from .chain_reduction import UntypedChain, detect_untyped_chains
from .ibd import _pair_ibd_indicator_trees
from .inheritance_tree import (
    InheritanceTree,
    LeafNode,
    _inheritance_recursion_budget,
    _required_inheritance_recursion_limit,
)
from .likelihood import (
    AlleleOrigin,
    InheritanceState,
    _score_family_markers,
    family_marker_likelihood_tree,
    inheritance_origins,
)
from .map import haldane_recombination_fraction, map_distance_cm
from .models import Dataset, Family, Marker
from .parallel import validate_workers
from .positions import AnalysisPosition


_TreeFamilyResult = TypeVar("_TreeFamilyResult")
_MARKER_WORKER_FAMILY: Family | None = None


@dataclass(frozen=True)
class MultipointIbdResult:
    """Pairwise multipoint IBD probabilities at one marker."""

    marker_name: str
    rows: tuple[dict[str, float | str], ...]


@dataclass(frozen=True)
class PositionIbdResult:
    """Pairwise multipoint IBD probabilities at an analysis position."""

    position_cm: float
    label: str
    marker_name: str | None
    rows: tuple[dict[str, float | str], ...]


@dataclass(frozen=True)
class PosteriorInheritanceState:
    """One inheritance vector and its multipoint posterior probability."""

    bits: tuple[int, ...]
    probability: float
    allele_origins: dict[str, tuple[AlleleOrigin, AlleleOrigin]]


@dataclass(frozen=True)
class FamilyStatePosterior:
    """Posterior inheritance states for one family at one position."""

    family_id: str
    states: tuple[PosteriorInheritanceState, ...]


@dataclass(frozen=True)
class PositionStatePosterior:
    """Family-level inheritance-state posteriors at one analysis position."""

    position_cm: float
    label: str
    marker_name: str | None
    families: tuple[FamilyStatePosterior, ...]


@dataclass(frozen=True)
class FamilyTreePosteriors:
    """Compressed posterior trees for one family across analysis positions."""

    family_id: str
    trees: tuple[InheritanceTree, ...]


@dataclass(frozen=True)
class TreePositionPosteriors:
    """Reusable compressed posteriors for one marker and position grid."""

    positions: tuple[AnalysisPosition, ...]
    marker_names: tuple[str, ...]
    families: tuple[FamilyTreePosteriors, ...]


@dataclass(frozen=True)
class _FamilyStateFactors:
    """Unnormalized factors shared by IBD and posterior-state consumers."""

    family_id: str
    people: tuple[str, ...]
    states: tuple[InheritanceState, ...]
    left_factors: tuple[float, ...]
    right_factors: tuple[float, ...]


def multipoint_ibd(
    dataset: Dataset,
    marker_names: list[str] | None = None,
    workers: int = 1,
    engine: MultipointEngine = "dense",
) -> tuple[MultipointIbdResult, ...]:
    """Compute exact multipoint IBD across an ordered marker sequence.

    Markers are ordered by genetic-map position. All selected markers must
    belong to one chromosome and have centimorgan positions. Each pedigree is
    analyzed and normalized independently because pedigrees are independent
    likelihood units.
    """

    markers = _ordered_markers(dataset, marker_names)
    analysis_positions = tuple(
        AnalysisPosition(
            position_cm=float(marker.position_cm),
            label=marker.name,
            marker_name=marker.name,
        )
        for marker in markers
    )
    position_results = multipoint_ibd_at_positions(
        dataset,
        analysis_positions,
        marker_names=[marker.name for marker in markers],
        workers=workers,
        engine=engine,
    )
    return tuple(
        MultipointIbdResult(
            marker_name=result.marker_name or result.label,
            rows=result.rows,
        )
        for result in position_results
    )


def multipoint_ibd_at_positions(
    dataset: Dataset,
    analysis_positions: tuple[AnalysisPosition, ...],
    marker_names: list[str] | None = None,
    workers: int = 1,
    engine: MultipointEngine = "dense",
    tree_posteriors: TreePositionPosteriors | None = None,
) -> tuple[PositionIbdResult, ...]:
    """Compute exact multipoint IBD at marker or intermarker positions."""

    selected_engine = validate_multipoint_engine(engine)
    if selected_engine == "tree":
        return _tree_multipoint_ibd_at_positions(
            dataset,
            analysis_positions,
            marker_names,
            workers,
            tree_posteriors,
        )
    if tree_posteriors is not None:
        raise ValueError("tree_posteriors requires engine='tree'.")

    accumulators = tuple(_empty_accumulator() for _ in analysis_positions)
    factors_by_position = _multipoint_state_factors_at_positions(
        dataset,
        analysis_positions,
        marker_names,
        workers,
    )

    for accumulator, family_factors in zip(
        accumulators,
        factors_by_position,
    ):
        for factors in family_factors:
            _accumulate_state_factors(
                accumulator,
                factors.people,
                factors.states,
                factors.left_factors,
                factors.right_factors,
            )

    return tuple(
        PositionIbdResult(
            position_cm=analysis_position.position_cm,
            label=analysis_position.label,
            marker_name=analysis_position.marker_name,
            rows=_rows_from_accumulator(accumulator),
        )
        for analysis_position, accumulator in zip(
            analysis_positions,
            accumulators,
        )
    )


def multipoint_state_posteriors_at_positions(
    dataset: Dataset,
    analysis_positions: tuple[AnalysisPosition, ...],
    marker_names: list[str] | None = None,
    workers: int = 1,
) -> tuple[PositionStatePosterior, ...]:
    """Return normalized family inheritance states at analysis positions.

    This is the common scientific input for statistics such as affected-pairs
    NPL. Each family is normalized independently because pedigrees are
    independent likelihood units.
    """

    factors_by_position = _multipoint_state_factors_at_positions(
        dataset,
        analysis_positions,
        marker_names,
        workers,
    )
    results: list[PositionStatePosterior] = []

    for analysis_position, family_factors in zip(
        analysis_positions,
        factors_by_position,
    ):
        family_posteriors = []
        for factors in family_factors:
            total = _accurate_sumprod(
                factors.left_factors,
                factors.right_factors,
            )
            if total <= 0.0:
                raise ValueError(
                    "Multipoint likelihood is zero at analysis position "
                    f"{analysis_position.position_cm:g} cM for family "
                    f"{factors.family_id!r}."
                )
            family_posteriors.append(
                FamilyStatePosterior(
                    family_id=factors.family_id,
                    states=tuple(
                        PosteriorInheritanceState(
                            bits=state.bits,
                            probability=(left_factor * right_factor) / total,
                            allele_origins=state.allele_origins,
                        )
                        for state, left_factor, right_factor in zip(
                            factors.states,
                            factors.left_factors,
                            factors.right_factors,
                        )
                    ),
                )
            )
        results.append(
            PositionStatePosterior(
                position_cm=analysis_position.position_cm,
                label=analysis_position.label,
                marker_name=analysis_position.marker_name,
                families=tuple(family_posteriors),
            )
        )

    return tuple(results)


def multipoint_tree_posteriors_at_positions(
    dataset: Dataset,
    analysis_positions: tuple[AnalysisPosition, ...],
    marker_names: list[str] | None = None,
    workers: int = 1,
) -> TreePositionPosteriors:
    """Compute reusable compressed family posteriors at requested positions."""

    markers, recombination_fractions = _tree_analysis_inputs(
        dataset,
        marker_names,
    )
    family_posterior_trees = _run_tree_family_tasks(
        _tree_posteriors_at_positions,
        dataset.families,
        markers,
        analysis_positions,
        recombination_fractions,
        workers,
    )
    return TreePositionPosteriors(
        positions=analysis_positions,
        marker_names=tuple(marker.name for marker in markers),
        families=tuple(
            FamilyTreePosteriors(
                family_id=family.family_id,
                trees=posterior_trees,
            )
            for family, posterior_trees in zip(
                dataset.families,
                family_posterior_trees,
            )
        ),
    )


def two_marker_multipoint_ibd(
    dataset: Dataset,
    left_marker_name: str,
    right_marker_name: str,
    workers: int = 1,
    engine: MultipointEngine = "dense",
) -> tuple[MultipointIbdResult, MultipointIbdResult]:
    """Compute exact two-marker IBD while preserving the original API order."""

    results = {
        result.marker_name: result
        for result in multipoint_ibd(
            dataset,
            marker_names=[left_marker_name, right_marker_name],
            workers=workers,
            engine=engine,
        )
    }
    return results[left_marker_name], results[right_marker_name]


def _multipoint_state_factors_at_positions(
    dataset: Dataset,
    analysis_positions: tuple[AnalysisPosition, ...],
    marker_names: list[str] | None,
    workers: int,
) -> tuple[tuple[_FamilyStateFactors, ...], ...]:
    """Calculate reusable, unnormalized state factors for each family."""

    markers = _ordered_markers(dataset, marker_names)
    states_by_marker = _states_by_family(dataset, markers, workers)
    recombination_fractions = tuple(
        haldane_recombination_fraction(
            map_distance_cm(
                float(left.position_cm),
                float(right.position_cm),
            )
        )
        for left, right in zip(markers, markers[1:])
    )
    factors_by_position: list[list[_FamilyStateFactors]] = [
        [] for _ in analysis_positions
    ]

    for family in dataset.families:
        family_states = tuple(
            states_by_marker[marker.name][family.family_id] for marker in markers
        )
        forward_weights, backward_weights = _forward_backward_weights(
            family_states,
            recombination_fractions,
        )
        people = tuple(person.individual_id for person in family.individuals)

        for position_index, analysis_position in enumerate(analysis_positions):
            (
                position_states,
                left_conditionals,
                right_conditionals,
            ) = _state_factors_at_position(
                family,
                markers,
                family_states,
                forward_weights,
                backward_weights,
                analysis_position.position_cm,
            )
            factors_by_position[position_index].append(
                _FamilyStateFactors(
                    family_id=family.family_id,
                    people=people,
                    states=position_states,
                    left_factors=left_conditionals,
                    right_factors=right_conditionals,
                )
            )

    return tuple(
        tuple(position_factors) for position_factors in factors_by_position
    )


def _ordered_markers(
    dataset: Dataset,
    marker_names: list[str] | None,
) -> tuple[Marker, ...]:
    if marker_names is None:
        markers = list(dataset.markers)
    else:
        if len(marker_names) != len(set(marker_names)):
            raise ValueError("Multipoint marker names must be unique.")
        missing = set(marker_names).difference(dataset.marker_by_name)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"Unknown marker(s): {missing_names}")
        markers = [dataset.marker_by_name[name] for name in marker_names]

    if not markers:
        raise ValueError("Multipoint IBD requires at least one marker.")
    if any(marker.position_cm is None for marker in markers):
        raise ValueError("Multipoint IBD requires a map position for every marker.")

    chromosomes = {marker.chromosome for marker in markers}
    if len(chromosomes) != 1:
        raise ValueError("Multipoint IBD requires markers from one chromosome.")

    return tuple(
        sorted(
            markers,
            key=lambda marker: (float(marker.position_cm), marker.name),
        )
    )


def _states_by_family(
    dataset: Dataset,
    markers: tuple[Marker, ...],
    workers: int = 1,
) -> dict[str, dict[str, tuple[InheritanceState, ...]]]:
    workers = validate_workers(workers)
    if workers == 1:
        family_state_groups = tuple(
            _score_family_markers(family, markers)
            for family in dataset.families
        )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            family_state_groups = tuple(
                executor.map(
                    _score_family_markers,
                    dataset.families,
                    repeat(markers),
                )
            )

    states_by_marker: dict[str, dict[str, tuple[InheritanceState, ...]]] = {
        marker.name: {} for marker in markers
    }
    for family, states_for_family in zip(
        dataset.families,
        family_state_groups,
    ):
        for marker, states in zip(markers, states_for_family):
            if not states:
                raise ValueError(
                    "Multipoint likelihood is zero for "
                    f"family {family.family_id!r} at marker {marker.name!r}."
                )
            states_by_marker[marker.name][family.family_id] = states

    return states_by_marker


def _posterior_state_weights(
    states_by_marker: tuple[tuple[InheritanceState, ...], ...],
    recombination_fractions: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    forward_weights, backward_weights = _forward_backward_weights(
        states_by_marker,
        recombination_fractions,
    )
    return tuple(
        _normalize_weights(
            tuple(
                forward_weight * backward_weight
                for forward_weight, backward_weight in zip(forward, backward)
            ),
            context=f"marker {marker_index + 1} posterior",
        )
        for marker_index, (forward, backward) in enumerate(
            zip(forward_weights, backward_weights)
        )
    )


def _tree_multipoint_ibd_at_positions(
    dataset: Dataset,
    analysis_positions: tuple[AnalysisPosition, ...],
    marker_names: list[str] | None,
    workers: int,
    tree_posteriors: TreePositionPosteriors | None,
) -> tuple[PositionIbdResult, ...]:
    """Compute public position IBD results through compressed family trees."""

    markers, recombination_fractions = _tree_analysis_inputs(
        dataset,
        marker_names,
    )
    if tree_posteriors is None:
        family_results = _run_tree_family_tasks(
            _tree_pairwise_ibd_at_positions,
            dataset.families,
            markers,
            analysis_positions,
            recombination_fractions,
            workers,
        )
    else:
        posterior_by_family = _validate_tree_position_posteriors(
            dataset,
            analysis_positions,
            marker_names,
            tree_posteriors,
        )
        family_results = tuple(
            _tree_pairwise_ibd_from_posteriors(
                family,
                posterior_by_family[family.family_id],
            )
            for family in dataset.families
        )

    rows_by_position: list[list[dict[str, float | str]]] = [
        [] for _ in analysis_positions
    ]
    for family, results_at_positions in zip(
        dataset.families,
        family_results,
    ):
        for position_index, pair_probabilities in enumerate(
            results_at_positions
        ):
            for (first_id, second_id), probabilities in (
                pair_probabilities.items()
            ):
                z0, z1, z2 = probabilities
                pi_hat = 0.5 * z1 + z2
                rows_by_position[position_index].append(
                    {
                        "family_id": family.family_id,
                        "id1": first_id,
                        "id2": second_id,
                        "z0": z0,
                        "z1": z1,
                        "z2": z2,
                        "pi_hat": pi_hat,
                        "kinship": 0.5 * pi_hat,
                    }
                )

    return tuple(
        PositionIbdResult(
            position_cm=analysis_position.position_cm,
            label=analysis_position.label,
            marker_name=analysis_position.marker_name,
            rows=tuple(
                sorted(
                    rows,
                    key=lambda row: (
                        str(row["family_id"]),
                        str(row["id1"]),
                        str(row["id2"]),
                    ),
                )
            ),
        )
        for analysis_position, rows in zip(
            analysis_positions,
            rows_by_position,
        )
    )


def _tree_analysis_inputs(
    dataset: Dataset,
    marker_names: list[str] | None,
) -> tuple[tuple[Marker, ...], tuple[float, ...]]:
    """Return ordered markers and transition fractions for a tree analysis."""

    markers = _ordered_markers(dataset, marker_names)
    recombination_fractions = tuple(
        haldane_recombination_fraction(
            map_distance_cm(
                float(left_marker.position_cm),
                float(right_marker.position_cm),
            )
        )
        for left_marker, right_marker in zip(markers, markers[1:])
    )
    return markers, recombination_fractions


def _validate_tree_position_posteriors(
    dataset: Dataset,
    analysis_positions: tuple[AnalysisPosition, ...],
    marker_names: list[str] | None,
    tree_posteriors: TreePositionPosteriors,
) -> dict[str, tuple[InheritanceTree, ...]]:
    """Validate a reusable tree result against one requested analysis."""

    markers = _ordered_markers(dataset, marker_names)
    expected_marker_names = tuple(marker.name for marker in markers)
    if tree_posteriors.marker_names != expected_marker_names:
        raise ValueError(
            "Compressed posteriors do not match the selected marker sequence."
        )
    if tree_posteriors.positions != analysis_positions:
        raise ValueError(
            "Compressed posteriors do not match the analysis-position grid."
        )

    family_by_id = {family.family_id: family for family in dataset.families}
    posterior_by_family = {
        family_posteriors.family_id: family_posteriors.trees
        for family_posteriors in tree_posteriors.families
    }
    if len(posterior_by_family) != len(tree_posteriors.families):
        raise ValueError("Compressed posteriors contain duplicate family IDs.")
    if posterior_by_family.keys() != family_by_id.keys():
        raise ValueError(
            "Compressed posterior families do not match the dataset families."
        )

    for family_id, posterior_trees in posterior_by_family.items():
        family = family_by_id[family_id]
        if len(posterior_trees) != len(analysis_positions):
            raise ValueError(
                "Every compressed family posterior requires one tree per "
                "analysis position."
            )
        if any(
            tree.bit_count != len(family.meioses) for tree in posterior_trees
        ):
            raise ValueError(
                "Compressed posterior meiosis bits do not match family "
                f"{family_id!r}."
            )

    return posterior_by_family


def _run_tree_family_tasks(
    task: Callable[
        [
            Family,
            tuple[Marker, ...],
            tuple[AnalysisPosition, ...],
            tuple[float, ...],
            int,
        ],
        _TreeFamilyResult,
    ],
    families: tuple[Family, ...],
    markers: tuple[Marker, ...],
    analysis_positions: tuple[AnalysisPosition, ...],
    recombination_fractions: tuple[float, ...],
    workers: int,
) -> tuple[_TreeFamilyResult, ...]:
    """Parallelize families, or markers when only one family is available."""

    selected_workers = validate_workers(workers)
    if not families:
        return ()
    if selected_workers == 1:
        return tuple(
            task(
                family,
                markers,
                analysis_positions,
                recombination_fractions,
                1,
            )
            for family in families
        )

    if len(families) == 1:
        return (
            task(
                families[0],
                markers,
                analysis_positions,
                recombination_fractions,
                selected_workers,
            ),
        )

    maximum_bit_count = max(len(family.meioses) for family in families)
    with _inheritance_recursion_budget(maximum_bit_count):
        with ProcessPoolExecutor(
            max_workers=selected_workers,
            initializer=_initialize_tree_worker,
            initargs=(maximum_bit_count,),
        ) as executor:
            return tuple(
                executor.map(
                    task,
                    families,
                    repeat(markers),
                    repeat(analysis_positions),
                    repeat(recombination_fractions),
                    repeat(1),
                )
            )


def _tree_marker_posteriors(
    family: Family,
    markers: tuple[Marker, ...],
    recombination_fractions: tuple[float, ...],
    marker_workers: int = 1,
) -> tuple[InheritanceTree, ...]:
    """Return normalized marker posteriors through compressed tree messages."""

    emission_trees = _family_marker_likelihood_trees(
        family,
        markers,
        marker_workers,
    )
    forward_trees, backward_trees = _tree_forward_backward_trees(
        family,
        markers,
        recombination_fractions,
        emission_trees=emission_trees,
        workers=marker_workers,
    )
    return tuple(
        forward_tree.pointwise_multiply(backward_tree).normalize()
        for forward_tree, backward_tree in zip(
            forward_trees,
            backward_trees,
        )
    )


def _tree_posteriors_at_positions(
    family: Family,
    markers: tuple[Marker, ...],
    analysis_positions: tuple[AnalysisPosition, ...],
    recombination_fractions: tuple[float, ...],
    marker_workers: int = 1,
) -> tuple[InheritanceTree, ...]:
    """Return compressed posteriors at marker or intermarker positions."""

    emission_trees = _family_marker_likelihood_trees(
        family,
        markers,
        marker_workers,
    )
    counting_chains = detect_untyped_chains(family)
    forward_trees, backward_trees = _tree_forward_backward_trees(
        family,
        markers,
        recombination_fractions,
        emission_trees=emission_trees,
        workers=marker_workers,
    )
    marker_positions = tuple(float(marker.position_cm) for marker in markers)
    has_intermarker_positions = any(
        analysis_position.position_cm not in marker_positions
        for analysis_position in analysis_positions
    )
    if has_intermarker_positions:
        intermarker_forward_trees, intermarker_backward_trees = (
            _tree_forward_backward_trees(
                family,
                markers,
                recombination_fractions,
                extended_precision=True,
                emission_trees=emission_trees,
                workers=marker_workers,
            )
        )
    else:
        intermarker_forward_trees = forward_trees
        intermarker_backward_trees = backward_trees
    unconditioned_tree = InheritanceTree(
        bit_count=len(family.meioses),
        root=LeafNode(1.0),
    )
    posterior_trees = []

    for analysis_position in analysis_positions:
        position_cm = analysis_position.position_cm
        exact_marker_indices = tuple(
            marker_index
            for marker_index, marker_position in enumerate(marker_positions)
            if marker_position == position_cm
        )
        if exact_marker_indices:
            marker_index = exact_marker_indices[-1]
            posterior_trees.append(
                forward_trees[marker_index]
                .pointwise_multiply(backward_trees[marker_index])
                .normalize()
            )
            continue

        left_marker_index = next(
            (
                marker_index
                for marker_index in range(len(markers) - 1, -1, -1)
                if marker_positions[marker_index] < position_cm
            ),
            None,
        )
        right_marker_index = next(
            (
                marker_index
                for marker_index, marker_position in enumerate(marker_positions)
                if marker_position > position_cm
            ),
            None,
        )

        if left_marker_index is None:
            left_tree = unconditioned_tree
        else:
            left_theta = haldane_recombination_fraction(
                map_distance_cm(
                    marker_positions[left_marker_index],
                    position_cm,
                )
            )
            left_tree = intermarker_forward_trees[
                left_marker_index
            ].transition_counting_chains(
                left_theta,
                counting_chains,
                extended_precision=True,
            ).binary_rescale()

        if right_marker_index is None:
            right_tree = unconditioned_tree
        else:
            right_theta = haldane_recombination_fraction(
                map_distance_cm(
                    position_cm,
                    marker_positions[right_marker_index],
                )
            )
            right_conditioned_tree = emission_trees[
                right_marker_index
            ].pointwise_multiply(
                intermarker_backward_trees[right_marker_index]
            )
            right_tree = (
                right_conditioned_tree.binary_rescale()
                .transition_counting_chains(
                    right_theta,
                    counting_chains,
                    extended_precision=True,
                )
                .binary_rescale()
            )

        posterior_trees.append(
            left_tree.pointwise_multiply(right_tree).normalize()
        )

    return tuple(posterior_trees)


def _tree_pairwise_ibd_probabilities(
    family: Family,
    markers: tuple[Marker, ...],
    recombination_fractions: tuple[float, ...],
) -> tuple[
    dict[tuple[str, str], tuple[float, float, float]],
    ...,
]:
    """Aggregate marker IBD probabilities from compressed posterior trees."""

    posterior_trees = _tree_marker_posteriors(
        family,
        markers,
        recombination_fractions,
    )
    return _tree_pairwise_ibd_from_posteriors(family, posterior_trees)


def _tree_pairwise_ibd_at_positions(
    family: Family,
    markers: tuple[Marker, ...],
    analysis_positions: tuple[AnalysisPosition, ...],
    recombination_fractions: tuple[float, ...],
    marker_workers: int = 1,
) -> tuple[
    dict[tuple[str, str], tuple[float, float, float]],
    ...,
]:
    """Aggregate compressed IBD probabilities at requested map positions."""

    posterior_trees = _tree_posteriors_at_positions(
        family,
        markers,
        analysis_positions,
        recombination_fractions,
        marker_workers,
    )
    return _tree_pairwise_ibd_from_posteriors(family, posterior_trees)


def _tree_pairwise_ibd_from_posteriors(
    family: Family,
    posterior_trees: tuple[InheritanceTree, ...],
) -> tuple[
    dict[tuple[str, str], tuple[float, float, float]],
    ...,
]:
    """Aggregate pairwise IBD states from normalized posterior trees."""

    people = tuple(person.individual_id for person in family.individuals)
    pair_indicators = {
        (first_id, second_id): _pair_ibd_indicator_trees(
            family,
            first_id,
            second_id,
        )
        for index, first_id in enumerate(people)
        for second_id in people[index + 1 :]
    }
    results = []
    for posterior_tree in posterior_trees:
        marker_probabilities = {}
        for pair, indicator_trees in pair_indicators.items():
            probabilities = tuple(
                posterior_tree.mean_product(indicator_tree)
                for indicator_tree in indicator_trees
            )
            total = fsum(probabilities)
            if total <= 0.0:
                raise ValueError(
                    "Tree IBD probabilities have a non-positive total for "
                    f"family {family.family_id!r}, pair {pair!r}."
                )
            marker_probabilities[pair] = tuple(
                probability / total for probability in probabilities
            )
        results.append(marker_probabilities)

    return tuple(results)


def _tree_forward_backward_trees(
    family: Family,
    markers: tuple[Marker, ...],
    recombination_fractions: tuple[float, ...],
    *,
    extended_precision: bool = False,
    emission_trees: tuple[InheritanceTree, ...] | None = None,
    workers: int = 1,
) -> tuple[
    tuple[InheritanceTree, ...],
    tuple[InheritanceTree, ...],
]:
    """Calculate normalized multipoint messages without dense state vectors."""

    if not markers:
        raise ValueError("Tree forward-backward calculation requires a marker.")
    if len(recombination_fractions) != len(markers) - 1:
        raise ValueError("One recombination fraction is required per marker interval.")

    selected_workers = validate_workers(workers)
    if emission_trees is None:
        emission_trees = _family_marker_likelihood_trees(
            family,
            markers,
            workers=selected_workers,
        )
    elif len(emission_trees) != len(markers):
        raise ValueError("One emission tree is required per marker.")
    counting_chains = detect_untyped_chains(family)
    if selected_workers >= 2 and len(markers) > 1:
        with _inheritance_recursion_budget(len(family.meioses)):
            with ProcessPoolExecutor(
                max_workers=2,
                initializer=_initialize_tree_worker,
                initargs=(len(family.meioses),),
            ) as executor:
                forward_future = executor.submit(
                    _tree_forward_pass,
                    emission_trees,
                    recombination_fractions,
                    counting_chains,
                    extended_precision,
                )
                backward_future = executor.submit(
                    _tree_backward_pass,
                    len(family.meioses),
                    emission_trees,
                    recombination_fractions,
                    counting_chains,
                    extended_precision,
                )
                return forward_future.result(), backward_future.result()

    return (
        _tree_forward_pass(
            emission_trees,
            recombination_fractions,
            counting_chains,
            extended_precision,
        ),
        _tree_backward_pass(
            len(family.meioses),
            emission_trees,
            recombination_fractions,
            counting_chains,
            extended_precision,
        ),
    )


def _tree_forward_pass(
    emission_trees: tuple[InheritanceTree, ...],
    recombination_fractions: tuple[float, ...],
    counting_chains: tuple[UntypedChain, ...],
    extended_precision: bool,
) -> tuple[InheritanceTree, ...]:
    """Calculate ordered left-to-right tree messages."""

    forward_trees = [emission_trees[0].binary_rescale()]
    for marker_index, theta in enumerate(recombination_fractions, start=1):
        transitioned = forward_trees[-1].transition_counting_chains(
            theta,
            counting_chains,
            extended_precision=extended_precision,
        )
        conditioned = transitioned.pointwise_multiply(
            emission_trees[marker_index]
        )
        forward_trees.append(conditioned.binary_rescale())
    return tuple(forward_trees)


def _tree_backward_pass(
    bit_count: int,
    emission_trees: tuple[InheritanceTree, ...],
    recombination_fractions: tuple[float, ...],
    counting_chains: tuple[UntypedChain, ...],
    extended_precision: bool,
) -> tuple[InheritanceTree, ...]:
    """Calculate ordered right-to-left tree messages."""

    backward_trees = [
        InheritanceTree(
            bit_count=bit_count,
            root=LeafNode(1.0),
        )
        for _ in emission_trees
    ]
    for marker_index in range(len(emission_trees) - 2, -1, -1):
        conditioned = emission_trees[marker_index + 1].pointwise_multiply(
            backward_trees[marker_index + 1]
        )
        backward_trees[marker_index] = (
            conditioned.transition_counting_chains(
                recombination_fractions[marker_index],
                counting_chains,
                extended_precision=extended_precision,
            ).binary_rescale()
        )

    return tuple(backward_trees)


def _family_marker_likelihood_trees(
    family: Family,
    markers: tuple[Marker, ...],
    workers: int,
    progress: Callable[[int, int], None] | None = None,
    diagnostic_progress: Callable[[str], None] | None = None,
    heartbeat_node_interval: int | None = None,
    emission_node_limit: int | None = None,
    emission_time_limit_seconds: float | None = None,
) -> tuple[InheritanceTree, ...]:
    """Score independent markers serially or in ordered worker processes."""

    selected_workers = validate_workers(workers)
    if selected_workers == 1 or len(markers) <= 1:
        trees = []
        for marker_index, marker in enumerate(markers, start=1):
            trees.append(
                family_marker_likelihood_tree(
                    family,
                    marker,
                    progress=diagnostic_progress,
                    heartbeat_node_interval=heartbeat_node_interval,
                    node_limit=emission_node_limit,
                    time_limit_seconds=emission_time_limit_seconds,
                )
            )
            if progress is not None:
                progress(marker_index, len(markers))
        return tuple(trees)

    process_count = min(selected_workers, len(markers))
    bit_count = len(family.meioses)
    with _inheritance_recursion_budget(bit_count):
        with ProcessPoolExecutor(
            max_workers=process_count,
            initializer=_initialize_marker_worker,
            initargs=(family,),
        ) as executor:
            futures_by_marker_index = {
                executor.submit(
                    _score_marker_in_worker,
                    marker,
                    heartbeat_node_interval,
                    emission_node_limit,
                    emission_time_limit_seconds,
                    diagnostic_progress is not None,
                ): marker_index
                for marker_index, marker in enumerate(markers)
            }
            ordered_trees: list[InheritanceTree | None] = [None] * len(
                markers
            )
            completed_count = 0
            for future in as_completed(futures_by_marker_index):
                marker_index = futures_by_marker_index[future]
                ordered_trees[marker_index] = future.result()
                completed_count += 1
                if progress is not None:
                    progress(completed_count, len(markers))

            if any(tree is None for tree in ordered_trees):
                raise RuntimeError("A marker worker returned no tree.")
            return tuple(
                tree
                for tree in ordered_trees
                if tree is not None
            )


def _initialize_tree_worker(bit_count: int) -> None:
    """Keep a private worker recursion budget active through result pickling."""

    required_limit = _required_inheritance_recursion_limit(bit_count)
    if required_limit > sys.getrecursionlimit():
        sys.setrecursionlimit(required_limit)


def _initialize_marker_worker(family: Family) -> None:
    """Install one family once per marker-scoring worker process."""

    global _MARKER_WORKER_FAMILY
    _initialize_tree_worker(len(family.meioses))
    _MARKER_WORKER_FAMILY = family


def _score_marker_in_worker(
    marker: Marker,
    heartbeat_node_interval: int | None = None,
    node_limit: int | None = None,
    time_limit_seconds: float | None = None,
    emit_progress: bool = False,
) -> InheritanceTree:
    """Score one marker using the family installed in this worker."""

    if _MARKER_WORKER_FAMILY is None:
        raise RuntimeError("Marker worker was not initialized with a family.")
    return family_marker_likelihood_tree(
        _MARKER_WORKER_FAMILY,
        marker,
        progress=_print_marker_worker_progress if emit_progress else None,
        heartbeat_node_interval=heartbeat_node_interval,
        node_limit=node_limit,
        time_limit_seconds=time_limit_seconds,
    )


def _print_marker_worker_progress(message: str) -> None:
    """Write benchmark heartbeats directly from a marker worker process."""

    print(message, file=sys.stderr, flush=True)


def _forward_backward_weights(
    states_by_marker: tuple[tuple[InheritanceState, ...], ...],
    recombination_fractions: tuple[float, ...],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
]:
    if len(recombination_fractions) != len(states_by_marker) - 1:
        raise ValueError("One recombination fraction is required per marker interval.")

    forward_weights: list[tuple[float, ...]] = [
        _normalize_weights(
            tuple(state.likelihood for state in states_by_marker[0]),
            context="first marker forward pass",
        )
    ]
    for marker_index, theta in enumerate(recombination_fractions, start=1):
        previous_states = states_by_marker[marker_index - 1]
        current_states = states_by_marker[marker_index]
        previous_weights = forward_weights[-1]
        current_weights = tuple(
            current_state.likelihood
            * fsum(
                previous_weight
                * _transition_weight(previous_state.bits, current_state.bits, theta)
                for previous_state, previous_weight in zip(
                    previous_states,
                    previous_weights,
                )
            )
            for current_state in current_states
        )
        forward_weights.append(
            _normalize_weights(
                current_weights,
                context=f"marker {marker_index + 1} forward pass",
            )
        )

    backward_weights: list[tuple[float, ...]] = [tuple()] * len(states_by_marker)
    backward_weights[-1] = tuple(1.0 for _ in states_by_marker[-1])
    for marker_index in range(len(states_by_marker) - 2, -1, -1):
        current_states = states_by_marker[marker_index]
        next_states = states_by_marker[marker_index + 1]
        next_weights = backward_weights[marker_index + 1]
        theta = recombination_fractions[marker_index]
        current_weights = tuple(
            fsum(
                _transition_weight(current_state.bits, next_state.bits, theta)
                * next_state.likelihood
                * next_weight
                for next_state, next_weight in zip(next_states, next_weights)
            )
            for current_state in current_states
        )
        backward_weights[marker_index] = _normalize_weights(
            current_weights,
            context=f"marker {marker_index + 1} backward pass",
        )

    return (
        tuple(forward_weights),
        tuple(backward_weights),
    )


def _state_factors_at_position(
    family: Family,
    markers: tuple[Marker, ...],
    states_by_marker: tuple[tuple[InheritanceState, ...], ...],
    forward_weights: tuple[tuple[float, ...], ...],
    backward_weights: tuple[tuple[float, ...], ...],
    position_cm: float,
) -> tuple[
    tuple[InheritanceState, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    marker_positions = tuple(float(marker.position_cm) for marker in markers)
    exact_marker_indices = tuple(
        marker_index
        for marker_index, marker_position in enumerate(marker_positions)
        if marker_position == position_cm
    )
    if exact_marker_indices:
        marker_index = exact_marker_indices[-1]
        left_conditionals = forward_weights[marker_index]
        right_conditionals = backward_weights[marker_index]
        _require_positive_dot_product(
            left_conditionals,
            right_conditionals,
            context=f"analysis position {position_cm:g} cM",
        )
        return (
            states_by_marker[marker_index],
            left_conditionals,
            right_conditionals,
        )

    # A position without observed genotypes has no emission filter. Therefore,
    # every inheritance vector must be restored before propagating the two
    # marker conditionals to this location.
    query_states = tuple(
        InheritanceState(
            family_id=family.family_id,
            bits=tuple(bits),
            likelihood=1.0,
            posterior_weight=0.0,
            allele_origins=inheritance_origins(family, tuple(bits)),
        )
        for bits in product((0, 1), repeat=len(family.meioses))
    )
    left_marker_index = next(
        (
            marker_index
            for marker_index in range(len(markers) - 1, -1, -1)
            if marker_positions[marker_index] < position_cm
        ),
        None,
    )
    right_marker_index = next(
        (
            marker_index
            for marker_index, marker_position in enumerate(marker_positions)
            if marker_position > position_cm
        ),
        None,
    )

    left_conditionals = _left_conditionals_at_position(
        query_states,
        states_by_marker,
        forward_weights,
        marker_positions,
        left_marker_index,
        position_cm,
    )
    right_conditionals = _right_conditionals_at_position(
        query_states,
        states_by_marker,
        backward_weights,
        marker_positions,
        right_marker_index,
        position_cm,
    )
    _require_positive_dot_product(
        left_conditionals,
        right_conditionals,
        context=f"analysis position {position_cm:g} cM",
    )
    return query_states, left_conditionals, right_conditionals


def _left_conditionals_at_position(
    query_states: tuple[InheritanceState, ...],
    states_by_marker: tuple[tuple[InheritanceState, ...], ...],
    forward_weights: tuple[tuple[float, ...], ...],
    marker_positions: tuple[float, ...],
    left_marker_index: int | None,
    position_cm: float,
) -> tuple[float, ...]:
    if left_marker_index is None:
        return tuple(1.0 for _ in query_states)

    theta = haldane_recombination_fraction(
        map_distance_cm(
            marker_positions[left_marker_index],
            position_cm,
        )
    )
    return tuple(
        fsum(
            marker_weight
            * _transition_weight(marker_state.bits, query_state.bits, theta)
            for marker_state, marker_weight in zip(
                states_by_marker[left_marker_index],
                forward_weights[left_marker_index],
            )
        )
        for query_state in query_states
    )


def _right_conditionals_at_position(
    query_states: tuple[InheritanceState, ...],
    states_by_marker: tuple[tuple[InheritanceState, ...], ...],
    backward_weights: tuple[tuple[float, ...], ...],
    marker_positions: tuple[float, ...],
    right_marker_index: int | None,
    position_cm: float,
) -> tuple[float, ...]:
    if right_marker_index is None:
        return tuple(1.0 for _ in query_states)

    theta = haldane_recombination_fraction(
        map_distance_cm(
            position_cm,
            marker_positions[right_marker_index],
        )
    )
    return tuple(
        fsum(
            _transition_weight(query_state.bits, marker_state.bits, theta)
            * marker_state.likelihood
            * backward_weight
            for marker_state, backward_weight in zip(
                states_by_marker[right_marker_index],
                backward_weights[right_marker_index],
            )
        )
        for query_state in query_states
    )


def _normalize_weights(
    weights: tuple[float, ...],
    context: str,
) -> tuple[float, ...]:
    total = fsum(weights)
    if total <= 0.0:
        raise ValueError(f"Multipoint likelihood is zero during {context}.")
    return tuple(weight / total for weight in weights)


def _transition_weight(
    left_bits: tuple[int, ...],
    right_bits: tuple[int, ...],
    theta: float,
) -> float:
    if len(left_bits) != len(right_bits):
        raise ValueError("Inheritance states must describe the same meioses.")

    weight = 1.0
    for left_bit, right_bit in zip(left_bits, right_bits):
        weight *= theta if left_bit != right_bit else 1.0 - theta
    return weight


def _empty_accumulator() -> defaultdict[
    tuple[str, str, str],
    dict[str, list[float]],
]:
    return defaultdict(lambda: {"z0": [], "z1": [], "z2": []})


def _accumulate_state_factors(
    accumulator: defaultdict[
        tuple[str, str, str],
        dict[str, list[float]],
    ],
    people: tuple[str, ...],
    states: tuple[InheritanceState, ...],
    left_factors: tuple[float, ...],
    right_factors: tuple[float, ...],
) -> None:
    if not len(states) == len(left_factors) == len(right_factors):
        raise ValueError("Each inheritance state requires two weight factors.")

    for index, first_id in enumerate(people):
        for second_id in people[index + 1 :]:
            left_by_ibd_state: list[list[float]] = [[], [], []]
            right_by_ibd_state: list[list[float]] = [[], [], []]
            for state, left_factor, right_factor in zip(
                states,
                left_factors,
                right_factors,
            ):
                shared = len(
                    set(state.allele_origins[first_id]).intersection(
                        state.allele_origins[second_id]
                    )
                )
                left_by_ibd_state[shared].append(left_factor)
                right_by_ibd_state[shared].append(right_factor)

            family_id = states[0].family_id
            pair_values = accumulator[(family_id, first_id, second_id)]
            for shared in range(3):
                pair_values[f"z{shared}"].append(
                    _accurate_sumprod(
                        tuple(left_by_ibd_state[shared]),
                        tuple(right_by_ibd_state[shared]),
                    )
                )


def _require_positive_dot_product(
    left_values: tuple[float, ...],
    right_values: tuple[float, ...],
    context: str,
) -> None:
    if _accurate_sumprod(left_values, right_values) <= 0.0:
        raise ValueError(f"Multipoint likelihood is zero during {context}.")


def _accurate_sumprod(
    left_values: tuple[float, ...],
    right_values: tuple[float, ...],
) -> float:
    """Accurately reduce products with a deterministic legacy fallback."""

    if len(left_values) != len(right_values):
        raise ValueError("Product vectors must have equal lengths.")

    sumprod = getattr(math, "sumprod", None)
    if sumprod is not None:
        return float(sumprod(left_values, right_values))

    # Python 3.10 and 3.11 do not provide math.sumprod. Decimal.from_float
    # retains each binary64 input exactly, so this slower path preserves the
    # reference engine's accuracy contract without adding a dependency.
    with localcontext() as decimal_context:
        decimal_context.prec = 80
        total = Decimal(0)
        for left_value, right_value in zip(left_values, right_values):
            total += Decimal.from_float(left_value) * Decimal.from_float(
                right_value
            )
    return float(total)


def _rows_from_accumulator(
    accumulator: defaultdict[
        tuple[str, str, str],
        dict[str, list[float]],
    ],
) -> tuple[dict[str, float | str], ...]:
    rows = []
    for (family_id, first_id, second_id), values in sorted(accumulator.items()):
        z0 = fsum(values["z0"])
        z1 = fsum(values["z1"])
        z2 = fsum(values["z2"])
        total = fsum((z0, z1, z2))
        if total <= 0.0:
            raise ValueError(
                "Multipoint IBD probabilities have a non-positive total for "
                f"family {family_id!r}, pair ({first_id!r}, {second_id!r})."
            )

        # IBD states are exhaustive and mutually exclusive. Renormalizing the
        # accurately reduced bins removes drift away from the probability simplex.
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
    return tuple(rows)
