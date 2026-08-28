"""Affected-pairs nonparametric linkage scoring.

The implementation follows MERLIN's NPL_Pairs statistic. It operates on the
exact multipoint inheritance-state posteriors produced by the reference engine.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import fsum

from .backends import MultipointEngine, validate_multipoint_engine
from .founder_symmetry import (
    FounderOrientationSymmetryPlan,
    build_founder_couple_symmetry_plan,
    build_founder_orientation_symmetry_plan,
    restore_founder_couple_symmetry_branches,
    restore_founder_orientation_branch,
)
from .ibd import _ancestral_meiosis_indices
from .inheritance_tree import (
    InheritanceTree,
    LeafNode,
    TreeNode,
    _combine_children,
    _inheritance_recursion_budget,
)
from .likelihood import AlleleOrigin, inheritance_origins
from .models import Dataset, Family
from .multipoint import (
    FamilyStatePosterior,
    PositionStatePosterior,
    TreePositionPosteriors,
    _validate_tree_position_posteriors,
    multipoint_state_posteriors_at_positions,
    multipoint_tree_posteriors_at_positions,
)
from .positions import AnalysisPosition


@dataclass(frozen=True)
class FamilyNplPairsResult:
    """Standardized affected-pairs scores for one family."""

    family_id: str
    null_mean: float
    null_variance: float
    z_min: float
    z_max: float
    z_scores: tuple[float, ...]
    standardized_score_values: tuple[float, ...]
    null_probabilities: tuple[float, ...]
    posterior_probabilities: tuple[tuple[float, ...], ...]

    @property
    def informative(self) -> bool:
        """Report whether the family statistic varies under the null."""

        return self.null_variance > 0.0


@dataclass(frozen=True)
class AffectionNplPairsResult:
    """Affected-pairs results for one affection phenotype."""

    affection_name: str
    families: tuple[FamilyNplPairsResult, ...]


@dataclass(frozen=True)
class NplPairsResult:
    """Affected-pairs results for one chromosome and position grid."""

    chromosome: str
    positions: tuple[AnalysisPosition, ...]
    analyses: tuple[AffectionNplPairsResult, ...]


@dataclass(frozen=True)
class _NullDistribution:
    mean: float
    variance: float
    minimum: float
    maximum: float
    raw_score_by_bits: dict[tuple[int, ...], float]
    raw_score_values: tuple[float, ...]
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class _TreeNullDistribution:
    """Compressed NPL score function and its Mendelian null distribution."""

    score_tree: InheritanceTree
    mean: float
    variance: float
    minimum: float
    maximum: float
    raw_score_values: tuple[float, ...]
    probabilities: tuple[float, ...]


def multipoint_npl_pairs(
    dataset: Dataset,
    analysis_positions: tuple[AnalysisPosition, ...],
    affection_names: tuple[str, ...] | None = None,
    marker_names: list[str] | None = None,
    position_posteriors: tuple[PositionStatePosterior, ...] | None = None,
    workers: int = 1,
    engine: MultipointEngine = "dense",
    tree_posteriors: TreePositionPosteriors | None = None,
) -> NplPairsResult:
    """Calculate MERLIN-compatible affected-pairs family Z scores.

    The raw statistic is standardized with its exact Mendelian null mean and
    variance separately for each family. Posterior expectations then use all
    inheritance states supported by the marker data at each position.
    """

    selected_engine = validate_multipoint_engine(engine)
    if selected_engine == "tree":
        if position_posteriors is not None:
            raise ValueError(
                "The tree engine computes compressed posteriors internally; "
                "position_posteriors must be omitted."
            )
        return _tree_multipoint_npl_pairs(
            dataset,
            analysis_positions,
            affection_names,
            marker_names,
            workers,
            tree_posteriors,
        )
    if tree_posteriors is not None:
        raise ValueError("tree_posteriors requires engine='tree'.")

    selected_affection_names = (
        dataset.affection_names if affection_names is None else affection_names
    )
    if not selected_affection_names:
        raise ValueError("Affected-pairs analysis requires an affection phenotype.")

    unknown_affection_names = set(selected_affection_names).difference(
        dataset.affection_names
    )
    if unknown_affection_names:
        unknown = ", ".join(sorted(unknown_affection_names))
        raise ValueError(f"Unknown affection phenotype(s): {unknown}")

    chromosomes = {marker.chromosome for marker in dataset.markers}
    if len(chromosomes) != 1 or None in chromosomes:
        raise ValueError(
            "Affected-pairs analysis requires mapped markers from one chromosome."
        )
    chromosome = next(iter(chromosomes))
    if chromosome is None:
        raise ValueError("Affected-pairs analysis requires a chromosome label.")

    posteriors = (
        multipoint_state_posteriors_at_positions(
            dataset,
            analysis_positions,
            marker_names,
            workers=workers,
        )
        if position_posteriors is None
        else position_posteriors
    )
    if len(posteriors) != len(analysis_positions):
        raise ValueError(
            "Every affected-pairs analysis position requires posterior states."
        )
    for position, posterior in zip(analysis_positions, posteriors):
        if (
            posterior.label != position.label
            or posterior.position_cm != position.position_cm
        ):
            raise ValueError(
                "Affected-pairs positions do not match the posterior-state grid."
            )
    posterior_lookup = tuple(
        {
            family_posterior.family_id: family_posterior
            for family_posterior in position_posterior.families
        }
        for position_posterior in posteriors
    )

    analyses = tuple(
        _score_affection(
            dataset,
            posterior_lookup,
            affection_name,
        )
        for affection_name in selected_affection_names
    )
    return NplPairsResult(
        chromosome=chromosome,
        positions=analysis_positions,
        analyses=analyses,
    )


def _tree_multipoint_npl_pairs(
    dataset: Dataset,
    analysis_positions: tuple[AnalysisPosition, ...],
    affection_names: tuple[str, ...] | None = None,
    marker_names: list[str] | None = None,
    workers: int = 1,
    tree_posteriors: TreePositionPosteriors | None = None,
) -> NplPairsResult:
    """Calculate affected-pairs results from compressed posterior trees."""

    selected_affection_names = (
        dataset.affection_names if affection_names is None else affection_names
    )
    if not selected_affection_names:
        raise ValueError("Affected-pairs analysis requires an affection phenotype.")

    unknown_affection_names = set(selected_affection_names).difference(
        dataset.affection_names
    )
    if unknown_affection_names:
        unknown = ", ".join(sorted(unknown_affection_names))
        raise ValueError(f"Unknown affection phenotype(s): {unknown}")

    chromosomes = {marker.chromosome for marker in dataset.markers}
    if len(chromosomes) != 1 or None in chromosomes:
        raise ValueError(
            "Affected-pairs analysis requires mapped markers from one chromosome."
        )
    chromosome = next(iter(chromosomes))
    if chromosome is None:
        raise ValueError("Affected-pairs analysis requires a chromosome label.")

    if tree_posteriors is None:
        tree_posteriors = multipoint_tree_posteriors_at_positions(
            dataset,
            analysis_positions,
            marker_names,
            workers,
        )
    posterior_trees_by_family = _validate_tree_position_posteriors(
        dataset,
        analysis_positions,
        marker_names,
        tree_posteriors,
    )
    analyses = tuple(
        _tree_score_affection(
            dataset,
            posterior_trees_by_family,
            affection_name,
        )
        for affection_name in selected_affection_names
    )
    return NplPairsResult(
        chromosome=chromosome,
        positions=analysis_positions,
        analyses=analyses,
    )


def npl_pairs_score(
    allele_origins: dict[
        str,
        tuple[AlleleOrigin, AlleleOrigin],
    ],
    affected_ids: tuple[str, ...],
) -> float:
    """Score founder-allele sharing using MERLIN's NPL_Pairs definition."""

    if not affected_ids:
        # MERLIN represents an affection with no affected relatives as a
        # constant leaf. Its value is irrelevant after zero-variance filtering.
        return 1.0

    score = 0
    for affected_index, affected_id in enumerate(affected_ids):
        first_origin, second_origin = allele_origins[affected_id]

        for comparison_id in affected_ids[: affected_index + 1]:
            comparison_origins = allele_origins[comparison_id]
            score += int(comparison_origins[0] == first_origin)
            score += int(comparison_origins[0] == second_origin)
            score += int(comparison_origins[1] == first_origin)
            score += int(comparison_origins[1] == second_origin)

    return float(score)


def _npl_pairs_score_tree(
    family: Family,
    affected_ids: tuple[str, ...],
) -> InheritanceTree:
    """Build MERLIN's affected-pairs score as a compressed tree."""

    relevant_meiosis_indices = _ancestral_meiosis_indices(
        family,
        affected_ids,
    )
    inheritance_bits = [0] * len(family.meioses)
    founder_symmetry_plan = build_founder_orientation_symmetry_plan(
        family,
        relevant_meiosis_indices,
    )
    founder_couple_symmetry_plan = build_founder_couple_symmetry_plan(
        family,
        relevant_meiosis_indices,
    ).for_affected_ids(affected_ids)
    founder_couple_representative_indices = (
        founder_couple_symmetry_plan.representative_bit_indices
    )
    with _inheritance_recursion_budget(len(family.meioses)):
        root = _npl_pairs_score_node(
            family,
            affected_ids,
            relevant_meiosis_indices,
            inheritance_bits,
            bit_index=0,
            founder_symmetry_plan=founder_symmetry_plan,
            founder_couple_representative_indices=(
                founder_couple_representative_indices
            ),
        )
        root = restore_founder_couple_symmetry_branches(
            root,
            founder_couple_symmetry_plan,
        )
        return InheritanceTree(
            bit_count=len(family.meioses),
            root=root,
        )


def _npl_pairs_score_node(
    family: Family,
    affected_ids: tuple[str, ...],
    relevant_meiosis_indices: frozenset[int],
    inheritance_bits: list[int],
    bit_index: int,
    founder_symmetry_plan: FounderOrientationSymmetryPlan,
    founder_couple_representative_indices: frozenset[int],
) -> TreeNode:
    """Recursively score only transmissions ancestral to affected people."""

    if bit_index == len(inheritance_bits):
        score = npl_pairs_score(
            inheritance_origins(family, tuple(inheritance_bits)),
            affected_ids,
        )
        return LeafNode(score)

    inheritance_bits[bit_index] = 0
    zero_child = _npl_pairs_score_node(
        family,
        affected_ids,
        relevant_meiosis_indices,
        inheritance_bits,
        bit_index + 1,
        founder_symmetry_plan,
        founder_couple_representative_indices,
    )
    if bit_index not in relevant_meiosis_indices:
        return _combine_children(zero_child, zero_child)

    if (
        bit_index
        in founder_couple_representative_indices
    ):
        return _combine_children(zero_child, zero_child)

    founder_flip_indices = founder_symmetry_plan.descendant_flip_indices(
        bit_index
    )
    if founder_flip_indices is not None:
        return restore_founder_orientation_branch(
            zero_child,
            bit_index,
            founder_flip_indices,
        )

    inheritance_bits[bit_index] = 1
    one_child = _npl_pairs_score_node(
        family,
        affected_ids,
        relevant_meiosis_indices,
        inheritance_bits,
        bit_index + 1,
        founder_symmetry_plan,
        founder_couple_representative_indices,
    )
    return _combine_children(zero_child, one_child)


def format_merlin_npl_zscores(
    chromosome_results: tuple[NplPairsResult, ...],
) -> str:
    """Format raw family Z scores using MERLIN's `.zscore` layout."""

    lines: list[str] = []
    for result in chromosome_results:
        try:
            chromosome_number = int(result.chromosome)
        except ValueError as error:
            raise ValueError(
                "MERLIN autosomal Z-score output requires a numeric chromosome."
            ) from error

        lines.append(f"CHROMOSOME {chromosome_number}")
        position_labels = " ".join(position.label for position in result.positions)
        lines.append(f"POSITIONS {position_labels} ")

        for analysis in result.analyses:
            lines.append(f"ANALYSIS {analysis.affection_name}:pairs")
            for family_result in analysis.families:
                if not family_result.informative:
                    continue
                score_text = " ".join(
                    _format_zscore(score) for score in family_result.z_scores
                )
                score_suffix = f"{score_text} " if score_text else ""
                lines.append(
                    f"FAMILY {family_result.family_id} "
                    f"ZMIN {_format_zscore(family_result.z_min)} "
                    f"ZMAX {_format_zscore(family_result.z_max)} "
                    f"SCORES {score_suffix}"
                )

    return "\n".join(lines) + "\n"


def _score_affection(
    dataset: Dataset,
    posterior_lookup: tuple[dict[str, FamilyStatePosterior], ...],
    affection_name: str,
) -> AffectionNplPairsResult:
    family_results = []

    for family in dataset.families:
        affected_ids = tuple(
            person.individual_id
            for person in family.individuals
            if person.phenotypes.get(affection_name) == "2"
        )
        null_distribution = _npl_pairs_null_distribution(
            family,
            affected_ids,
        )

        if null_distribution.variance > 0.0:
            scale = 1.0 / math.sqrt(null_distribution.variance)
            z_min = (
                null_distribution.minimum - null_distribution.mean
            ) * scale
            z_max = (
                null_distribution.maximum - null_distribution.mean
            ) * scale
            z_scores = tuple(
                (
                    _posterior_expected_score(
                        posterior_by_family[family.family_id],
                        null_distribution.raw_score_by_bits,
                    )
                    - null_distribution.mean
                )
                * scale
                for posterior_by_family in posterior_lookup
            )
            standardized_score_values = tuple(
                (raw_score - null_distribution.mean) * scale
                for raw_score in null_distribution.raw_score_values
            )
        else:
            z_min = 0.0
            z_max = 0.0
            z_scores = tuple(0.0 for _ in posterior_lookup)
            standardized_score_values = tuple(
                0.0 for _ in null_distribution.raw_score_values
            )

        posterior_probabilities = tuple(
            _posterior_score_probabilities(
                posterior_by_family[family.family_id],
                null_distribution.raw_score_values,
                null_distribution.raw_score_by_bits,
            )
            for posterior_by_family in posterior_lookup
        )

        family_results.append(
            FamilyNplPairsResult(
                family_id=family.family_id,
                null_mean=null_distribution.mean,
                null_variance=null_distribution.variance,
                z_min=z_min,
                z_max=z_max,
                z_scores=z_scores,
                standardized_score_values=standardized_score_values,
                null_probabilities=null_distribution.probabilities,
                posterior_probabilities=posterior_probabilities,
            )
        )

    return AffectionNplPairsResult(
        affection_name=affection_name,
        families=tuple(family_results),
    )


def _tree_score_affection(
    dataset: Dataset,
    posterior_trees_by_family: dict[str, tuple[InheritanceTree, ...]],
    affection_name: str,
) -> AffectionNplPairsResult:
    """Score one affection using compressed NPL and posterior trees."""

    family_results = []
    for family in dataset.families:
        affected_ids = tuple(
            person.individual_id
            for person in family.individuals
            if person.phenotypes.get(affection_name) == "2"
        )
        null_distribution = _tree_npl_pairs_null_distribution(
            family,
            affected_ids,
        )
        posterior_trees = posterior_trees_by_family[family.family_id]
        if null_distribution.variance > 0.0:
            scale = 1.0 / math.sqrt(null_distribution.variance)
            z_min = (
                null_distribution.minimum - null_distribution.mean
            ) * scale
            z_max = (
                null_distribution.maximum - null_distribution.mean
            ) * scale
            z_scores = tuple(
                (
                    _tree_posterior_expected_score(
                        posterior_tree,
                        null_distribution.score_tree,
                    )
                    - null_distribution.mean
                )
                * scale
                for posterior_tree in posterior_trees
            )
            standardized_score_values = tuple(
                (raw_score - null_distribution.mean) * scale
                for raw_score in null_distribution.raw_score_values
            )
        else:
            z_min = 0.0
            z_max = 0.0
            z_scores = tuple(0.0 for _ in posterior_trees)
            standardized_score_values = tuple(
                0.0 for _ in null_distribution.raw_score_values
            )

        score_indicator_trees = tuple(
            null_distribution.score_tree.map_values(
                lambda score, raw_score=raw_score: float(score == raw_score)
            )
            for raw_score in null_distribution.raw_score_values
        )
        posterior_probabilities = tuple(
            _tree_posterior_score_probabilities(
                posterior_tree,
                score_indicator_trees,
                family.family_id,
            )
            for posterior_tree in posterior_trees
        )
        family_results.append(
            FamilyNplPairsResult(
                family_id=family.family_id,
                null_mean=null_distribution.mean,
                null_variance=null_distribution.variance,
                z_min=z_min,
                z_max=z_max,
                z_scores=z_scores,
                standardized_score_values=standardized_score_values,
                null_probabilities=null_distribution.probabilities,
                posterior_probabilities=posterior_probabilities,
            )
        )

    return AffectionNplPairsResult(
        affection_name=affection_name,
        families=tuple(family_results),
    )


def _npl_pairs_null_distribution(
    family: Family,
    affected_ids: tuple[str, ...],
) -> _NullDistribution:
    raw_score_by_bits = {
        tuple(bits): npl_pairs_score(
            inheritance_origins(family, tuple(bits)),
            affected_ids,
        )
        for bits in product((0, 1), repeat=len(family.meioses))
    }
    raw_scores = tuple(raw_score_by_bits.values())
    mean = fsum(raw_scores) / len(raw_scores)
    second_moment = fsum(score * score for score in raw_scores) / len(raw_scores)
    variance = max(0.0, second_moment - mean * mean)
    score_counts = Counter(raw_scores)
    raw_score_values = tuple(sorted(score_counts))
    probabilities = tuple(
        score_counts[raw_score] / len(raw_scores)
        for raw_score in raw_score_values
    )

    return _NullDistribution(
        mean=mean,
        variance=variance,
        minimum=min(raw_scores),
        maximum=max(raw_scores),
        raw_score_by_bits=raw_score_by_bits,
        raw_score_values=raw_score_values,
        probabilities=probabilities,
    )


def _tree_npl_pairs_null_distribution(
    family: Family,
    affected_ids: tuple[str, ...],
) -> _TreeNullDistribution:
    """Calculate the Mendelian NPL null from compressed score leaves."""

    score_tree = _npl_pairs_score_tree(family, affected_ids)
    probabilities_by_score = score_tree.value_probabilities()
    raw_score_values = tuple(probabilities_by_score)
    mean = score_tree.weighted_sum()
    second_moment = score_tree.mean_product(score_tree)
    variance = max(0.0, second_moment - mean * mean)
    return _TreeNullDistribution(
        score_tree=score_tree,
        mean=mean,
        variance=variance,
        minimum=raw_score_values[0],
        maximum=raw_score_values[-1],
        raw_score_values=raw_score_values,
        probabilities=tuple(
            probabilities_by_score[raw_score]
            for raw_score in raw_score_values
        ),
    )


def _tree_posterior_expected_score(
    posterior_tree: InheritanceTree,
    score_tree: InheritanceTree,
) -> float:
    """Return a normalized posterior expectation of a tree-valued score."""

    probability_total = posterior_tree.weighted_sum()
    if probability_total <= 0.0:
        raise ValueError(
            "Posterior inheritance-tree probabilities have a non-positive total."
        )
    return posterior_tree.mean_product(score_tree) / probability_total


def _tree_posterior_score_probabilities(
    posterior_tree: InheritanceTree,
    score_indicator_trees: tuple[InheritanceTree, ...],
    family_id: str,
) -> tuple[float, ...]:
    """Aggregate posterior mass for each compressed NPL score value."""

    probabilities = tuple(
        posterior_tree.mean_product(indicator_tree)
        for indicator_tree in score_indicator_trees
    )
    total = fsum(probabilities)
    if total <= 0.0:
        raise ValueError(
            "Posterior NPL score probabilities have a non-positive total for "
            f"family {family_id!r}."
        )
    return tuple(probability / total for probability in probabilities)


def _posterior_expected_score(
    posterior: FamilyStatePosterior,
    raw_score_by_bits: dict[tuple[int, ...], float],
) -> float:
    probabilities = tuple(state.probability for state in posterior.states)
    raw_scores = tuple(
        raw_score_by_bits[state.bits] for state in posterior.states
    )
    probability_total = fsum(probabilities)
    if probability_total <= 0.0:
        raise ValueError(
            "Posterior inheritance-state probabilities have a non-positive "
            f"total for family {posterior.family_id!r}."
        )

    sumprod = getattr(math, "sumprod", None)
    if sumprod is not None:
        weighted_score = float(sumprod(probabilities, raw_scores))
    else:
        weighted_score = fsum(
            probability * raw_score
            for probability, raw_score in zip(probabilities, raw_scores)
        )
    return weighted_score / probability_total


def _posterior_score_probabilities(
    posterior: FamilyStatePosterior,
    raw_score_values: tuple[float, ...],
    raw_score_by_bits: dict[tuple[int, ...], float],
) -> tuple[float, ...]:
    probabilities_by_score: dict[float, list[float]] = {
        raw_score: [] for raw_score in raw_score_values
    }
    for state in posterior.states:
        raw_score = raw_score_by_bits[state.bits]
        probabilities_by_score[raw_score].append(state.probability)

    probabilities = tuple(
        fsum(probabilities_by_score[raw_score])
        for raw_score in raw_score_values
    )
    total = fsum(probabilities)
    if total <= 0.0:
        raise ValueError(
            "Posterior NPL score probabilities have a non-positive total for "
            f"family {posterior.family_id!r}."
        )
    return tuple(probability / total for probability in probabilities)


def _format_zscore(value: float) -> str:
    formatted = f"{value:.6f}"
    return "0.000000" if formatted == "-0.000000" else formatted
