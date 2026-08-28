"""MERLIN-compatible multipoint inheritance information content."""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import fsum

from .backends import MultipointEngine, validate_multipoint_engine
from .inheritance_tree import InheritanceTree
from .io import MISSING_TOKENS
from .models import Dataset, Family, Individual
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
class FamilyInformationContentResult:
    """Information content for one family across analysis positions."""

    family_id: str
    bit_count: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class InformationContentResult:
    """Bit-weighted inheritance information for one chromosome."""

    chromosome: str
    positions: tuple[AnalysisPosition, ...]
    families: tuple[FamilyInformationContentResult, ...]
    values: tuple[float, ...]


def multipoint_information_content(
    dataset: Dataset,
    analysis_positions: tuple[AnalysisPosition, ...],
    marker_names: list[str] | None = None,
    position_posteriors: tuple[PositionStatePosterior, ...] | None = None,
    workers: int = 1,
    engine: MultipointEngine = "dense",
    tree_posteriors: TreePositionPosteriors | None = None,
) -> InformationContentResult:
    """Calculate MERLIN's entropy-based inheritance information content."""

    selected_engine = validate_multipoint_engine(engine)
    if selected_engine == "tree":
        if position_posteriors is not None:
            raise ValueError(
                "The tree engine computes compressed posteriors internally; "
                "position_posteriors must be omitted."
            )
        return _tree_multipoint_information_content(
            dataset,
            analysis_positions,
            marker_names,
            workers,
            tree_posteriors,
        )
    if tree_posteriors is not None:
        raise ValueError("tree_posteriors requires engine='tree'.")

    chromosome = _single_chromosome(dataset)
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
    _validate_position_posteriors(analysis_positions, posteriors)

    family_by_id = {family.family_id: family for family in dataset.families}
    expected_family_ids = set(family_by_id)
    for position_posterior in posteriors:
        posterior_family_ids = {
            family.family_id for family in position_posterior.families
        }
        if posterior_family_ids != expected_family_ids:
            raise ValueError(
                "Information posterior states do not match the dataset families."
            )
    posterior_by_position = tuple(
        {
            family_posterior.family_id: family_posterior
            for family_posterior in position_posterior.families
        }
        for position_posterior in posteriors
    )
    family_results = []
    for family_id, family in family_by_id.items():
        bit_count = _merlin_bit_count(dataset, family)
        family_results.append(
            FamilyInformationContentResult(
                family_id=family_id,
                bit_count=bit_count,
                values=tuple(
                    _family_information(
                        position_families[family_id],
                        len(family.meioses),
                        bit_count,
                    )
                    for position_families in posterior_by_position
                ),
            )
        )

    return _aggregate_information_content(
        chromosome=chromosome,
        analysis_positions=analysis_positions,
        family_results=tuple(family_results),
    )


def _tree_multipoint_information_content(
    dataset: Dataset,
    analysis_positions: tuple[AnalysisPosition, ...],
    marker_names: list[str] | None,
    workers: int,
    tree_posteriors: TreePositionPosteriors | None,
) -> InformationContentResult:
    """Calculate information content from compressed posterior trees."""

    chromosome = _single_chromosome(dataset)
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
    family_results = []
    for family in dataset.families:
        posterior_trees = posterior_trees_by_family[family.family_id]
        bit_count = _merlin_bit_count(dataset, family)
        family_results.append(
            FamilyInformationContentResult(
                family_id=family.family_id,
                bit_count=bit_count,
                values=tuple(
                    _tree_family_information(
                        posterior_tree,
                        len(family.meioses),
                        bit_count,
                    )
                    for posterior_tree in posterior_trees
                ),
            )
        )
    return _aggregate_information_content(
        chromosome=chromosome,
        analysis_positions=analysis_positions,
        family_results=tuple(family_results),
    )


def _aggregate_information_content(
    chromosome: str,
    analysis_positions: tuple[AnalysisPosition, ...],
    family_results: tuple[FamilyInformationContentResult, ...],
) -> InformationContentResult:
    """Combine family information using MERLIN's effective-bit weights."""

    total_bits = sum(result.bit_count for result in family_results)
    values = tuple(
        (
            fsum(
                result.values[position_index] * result.bit_count
                for result in family_results
            )
            / total_bits
            if total_bits > 0
            else 0.0
        )
        for position_index in range(len(analysis_positions))
    )
    return InformationContentResult(
        chromosome=chromosome,
        positions=analysis_positions,
        families=family_results,
        values=values,
    )


def format_merlin_information_console(
    chromosome_results: tuple[InformationContentResult, ...],
) -> str:
    """Format the information summary printed by MERLIN."""

    lines: list[str] = []
    for result in chromosome_results:
        lines.append(f"{'Position':>15} {'Info':>6}")
        lines.extend(
            f"{position.label:>15} {value:6.4f}"
            for position, value in zip(result.positions, result.values)
        )
        lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


def format_merlin_information_table(
    chromosome_results: tuple[InformationContentResult, ...],
) -> str:
    """Format MERLIN's tab-delimited information-content output."""

    lines = ["CHR\tPOS\tLABEL\tINFO"]
    for result in chromosome_results:
        try:
            chromosome_number = int(result.chromosome)
        except ValueError as error:
            raise ValueError(
                "MERLIN information output requires a numeric chromosome."
            ) from error
        lines.extend(
            (
                f"{chromosome_number}\t{position.position_cm:.4f}\t"
                f"{position.label}\t{value:.5f}"
            )
            for position, value in zip(result.positions, result.values)
        )
    return "\n".join(lines) + "\n"


def _family_information(
    posterior: FamilyStatePosterior,
    full_bit_count: int,
    bit_count: int,
) -> float:
    if bit_count == 0:
        return 0.0

    probability_total = fsum(
        state.probability for state in posterior.states
    )
    if probability_total <= 0.0:
        raise ValueError(
            "Posterior inheritance-state probabilities have a non-positive "
            f"total for family {posterior.family_id!r}."
        )
    normalized_probabilities = tuple(
        state.probability / probability_total
        for state in posterior.states
        if state.probability > 0.0
    )
    full_posterior_entropy = -fsum(
        probability * math.log(probability)
        for probability in normalized_probabilities
    )
    log_two = math.log(2.0)
    hidden_entropy = (full_bit_count - bit_count) * log_two
    posterior_entropy = max(full_posterior_entropy - hidden_entropy, 0.0)
    prior_entropy = bit_count * log_two

    # MERLIN only clamps negative information caused by roundoff. It does not
    # clamp the upper endpoint, so preserve that behavior for parity.
    return max(1.0 - posterior_entropy / prior_entropy, 0.0)


def _tree_family_information(
    posterior_tree: InheritanceTree,
    full_bit_count: int,
    bit_count: int,
) -> float:
    """Calculate MERLIN information from compressed posterior densities."""

    if bit_count == 0:
        return 0.0

    probability_total = posterior_tree.weighted_sum()
    if probability_total <= 0.0:
        raise ValueError(
            "Posterior inheritance-tree probabilities have a non-positive total."
        )

    entropy_terms = []
    log_two = math.log(2.0)
    for density, state_fraction in posterior_tree.value_probabilities().items():
        if density < 0.0:
            raise ValueError("Posterior inheritance-tree density is negative.")
        normalized_density = density / probability_total
        if normalized_density == 0.0:
            continue
        state_probability = math.ldexp(
            normalized_density,
            -full_bit_count,
        )
        if state_probability > 0.0:
            log_state_probability = math.log(state_probability)
        else:
            # Keep very large compressed pedigrees analyzable even when one
            # inheritance state's probability is below binary64 range.
            log_state_probability = (
                math.log(normalized_density) - full_bit_count * log_two
            )
        entropy_terms.append(
            -state_fraction * normalized_density * log_state_probability
        )

    full_posterior_entropy = fsum(entropy_terms)
    hidden_entropy = (full_bit_count - bit_count) * log_two
    posterior_entropy = max(full_posterior_entropy - hidden_entropy, 0.0)
    prior_entropy = bit_count * log_two
    return max(1.0 - posterior_entropy / prior_entropy, 0.0)


def _merlin_bit_count(dataset: Dataset, family: Family) -> int:
    """Count inheritance bits after MERLIN's founder symmetry reductions."""

    founder_ids = {founder.individual_id for founder in family.founders}
    transmitting_founders = {
        meiosis.parent_id
        for meiosis in family.meioses
        if meiosis.parent_id in founder_ids
    }
    bit_count = len(family.meioses) - len(transmitting_founders)
    bit_count -= _symmetric_founder_couple_count(dataset, family)
    if bit_count < 0:
        raise ValueError(
            f"MERLIN inheritance-bit count is negative for {family.family_id!r}."
        )
    return bit_count


def _symmetric_founder_couple_count(
    dataset: Dataset,
    family: Family,
) -> int:
    """Count MERLIN founder-couple symmetries that hide one more bit."""

    founders = {founder.individual_id: founder for founder in family.founders}
    mates: dict[str, set[str]] = {founder_id: set() for founder_id in founders}
    children_by_couple: dict[frozenset[str], list[str]] = {}
    for person in family.individuals:
        if person.father_id is None or person.mother_id is None:
            continue
        if person.father_id in mates:
            mates[person.father_id].add(person.mother_id)
        if person.mother_id in mates:
            mates[person.mother_id].add(person.father_id)
        if person.father_id in founders and person.mother_id in founders:
            couple = frozenset((person.father_id, person.mother_id))
            children_by_couple.setdefault(couple, []).append(person.individual_id)

    parent_ids = {meiosis.parent_id for meiosis in family.meioses}
    symmetric_count = 0
    for couple, child_ids in children_by_couple.items():
        first_id, second_id = tuple(couple)
        if mates[first_id] != {second_id} or mates[second_id] != {first_id}:
            continue
        if not any(child_id in parent_ids for child_id in child_ids):
            continue
        if _effectively_identical_founders(
            dataset,
            founders[first_id],
            founders[second_id],
        ):
            symmetric_count += 1
    return symmetric_count


def _effectively_identical_founders(
    dataset: Dataset,
    first: Individual,
    second: Individual,
) -> bool:
    """Apply the genotype and phenotype checks used by MERLIN's Mantra."""

    if any(
        _unordered_genotype(first.genotypes.get(marker.name))
        != _unordered_genotype(second.genotypes.get(marker.name))
        for marker in dataset.markers
    ):
        return False

    phenotype_names = set(first.phenotypes) | set(second.phenotypes)
    for phenotype_name in phenotype_names:
        first_value = first.phenotypes.get(phenotype_name)
        second_value = second.phenotypes.get(phenotype_name)
        if phenotype_name in dataset.affection_names:
            if (first_value == "2") != (second_value == "2"):
                return False
        elif _normalized_phenotype(first_value) != _normalized_phenotype(
            second_value
        ):
            return False
    return True


def _unordered_genotype(
    genotype: tuple[str | None, str | None] | None,
) -> tuple[str | None, str | None]:
    if genotype is None:
        return (None, None)
    first, second = genotype
    ordered = sorted(
        (first, second),
        key=lambda allele: (allele is not None, allele or ""),
    )
    return (ordered[0], ordered[1])


def _normalized_phenotype(value: str | None) -> str | None:
    return None if value in MISSING_TOKENS else value


def _single_chromosome(dataset: Dataset) -> str:
    chromosomes = {marker.chromosome for marker in dataset.markers}
    if len(chromosomes) != 1 or None in chromosomes:
        raise ValueError(
            "Information analysis requires mapped markers from one chromosome."
        )
    chromosome = next(iter(chromosomes))
    if chromosome is None:
        raise ValueError("Information analysis requires a chromosome label.")
    return chromosome


def _validate_position_posteriors(
    analysis_positions: tuple[AnalysisPosition, ...],
    position_posteriors: tuple[PositionStatePosterior, ...],
) -> None:
    if len(position_posteriors) != len(analysis_positions):
        raise ValueError(
            "Every information analysis position requires posterior states."
        )
    for position, posterior in zip(analysis_positions, position_posteriors):
        if (
            posterior.label != position.label
            or posterior.position_cm != position.position_cm
        ):
            raise ValueError(
                "Information positions do not match the posterior-state grid."
            )
