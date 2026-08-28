"""MERLIN-compatible multipoint genotype error detection."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from math import fsum

from .likelihood import (
    InheritanceState,
    _build_family_state_space,
    _build_marker_assignment_space,
    _score_family_marker,
)
from .map import haldane_recombination_fraction, map_distance_cm
from .models import Dataset, Family, Marker
from .multipoint import (
    _forward_backward_weights,
    _ordered_markers,
    _states_by_family,
    _transition_weight,
)
from .parallel import validate_workers


MERLIN_ERROR_RATIO_THRESHOLD = 0.025


@dataclass(frozen=True)
class GenotypeError:
    """A genotype that is unlikely given the surrounding marker data."""

    family_id: str
    person_id: str
    marker_name: str
    likelihood_ratio: float


def detect_unlikely_genotypes(
    dataset: Dataset,
    threshold: float = MERLIN_ERROR_RATIO_THRESHOLD,
    workers: int = 1,
) -> tuple[GenotypeError, ...]:
    """Identify genotypes below MERLIN's conditional likelihood threshold.

    The reported ratio compares the multipoint likelihood with and without
    one genotype, then removes the genotype's single-point contribution. This
    isolates evidence supplied by the surrounding markers.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError("The genotype-error threshold must be between 0 and 1.")
    workers = validate_workers(workers)

    markers = _ordered_markers(dataset, marker_names=None)
    states_by_marker = _states_by_family(dataset, markers, workers)
    recombination_fractions = tuple(
        haldane_recombination_fraction(
            map_distance_cm(
                float(left_marker.position_cm),
                float(right_marker.position_cm),
            )
        )
        for left_marker, right_marker in zip(markers, markers[1:])
    )
    baseline_states_by_family = tuple(
        tuple(
            states_by_marker[marker.name][family.family_id]
            for marker in markers
        )
        for family in dataset.families
    )
    if workers == 1:
        family_error_groups = tuple(
            _detect_family_errors(
                family,
                markers,
                baseline_states,
                recombination_fractions,
                threshold,
            )
            for family, baseline_states in zip(
                dataset.families,
                baseline_states_by_family,
            )
        )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            family_error_groups = tuple(
                executor.map(
                    _detect_family_errors,
                    dataset.families,
                    repeat(markers),
                    baseline_states_by_family,
                    repeat(recombination_fractions),
                    repeat(threshold),
                )
            )

    return tuple(
        genotype_error
        for family_errors in family_error_groups
        for genotype_error in family_errors
    )


def _detect_family_errors(
    family: Family,
    markers: tuple[Marker, ...],
    baseline_states: tuple[tuple[InheritanceState, ...], ...],
    recombination_fractions: tuple[float, ...],
    threshold: float,
) -> tuple[GenotypeError, ...]:
    """Evaluate one complete family so process results remain independent."""

    informative_marker_indices = {
        marker_index
        for marker_index, marker_states in enumerate(baseline_states)
        if _is_informative_emission(marker_states, len(family.meioses))
    }
    if len(informative_marker_indices) <= 1:
        return ()

    family_state_space = _build_family_state_space(family)
    forward_weights, backward_weights = _forward_backward_weights(
        baseline_states,
        recombination_fractions,
    )
    errors: list[GenotypeError] = []
    for marker_index in sorted(informative_marker_indices):
        marker = markers[marker_index]
        marker_states = baseline_states[marker_index]
        observed_singlepoint = fsum(
            state.likelihood for state in marker_states
        )
        observed_likelihood_by_bits = {
            state.bits: state.likelihood for state in marker_states
        }
        marker_assignment_space = _build_marker_assignment_space(
            family,
            marker,
        )

        for person in family.individuals:
            genotype = person.genotypes.get(marker.name, (None, None))
            if genotype[0] is None or genotype[1] is None:
                continue

            alternative_states = tuple(
                _score_family_marker(
                    family,
                    marker,
                    ignored_individual_id=person.individual_id,
                    use_uninformative_fallback=False,
                    family_state_space=family_state_space,
                    marker_assignment_space=marker_assignment_space,
                )
            )
            if not alternative_states:
                continue
            alternative_singlepoint = fsum(
                state.likelihood for state in alternative_states
            )
            if alternative_singlepoint == observed_singlepoint:
                continue

            left_factors = _left_outside_factors(
                alternative_states,
                baseline_states,
                forward_weights,
                recombination_fractions,
                marker_index,
            )
            right_factors = _right_outside_factors(
                alternative_states,
                baseline_states,
                backward_weights,
                recombination_fractions,
                marker_index,
            )
            observed_multipoint = fsum(
                left_factor
                * right_factor
                * observed_likelihood_by_bits.get(state.bits, 0.0)
                for state, left_factor, right_factor in zip(
                    alternative_states,
                    left_factors,
                    right_factors,
                )
            )
            alternative_multipoint = fsum(
                left_factor * right_factor * state.likelihood
                for state, left_factor, right_factor in zip(
                    alternative_states,
                    left_factors,
                    right_factors,
                )
            )
            if observed_multipoint <= 0.0 or alternative_multipoint <= 0.0:
                continue

            likelihood_ratio = (
                observed_multipoint * alternative_singlepoint
            ) / (alternative_multipoint * observed_singlepoint)
            if likelihood_ratio < threshold:
                errors.append(
                    GenotypeError(
                        family_id=family.family_id,
                        person_id=person.individual_id,
                        marker_name=marker.name,
                        likelihood_ratio=likelihood_ratio,
                    )
                )

    return tuple(errors)


def format_merlin_error_file(errors: tuple[GenotypeError, ...]) -> str:
    """Format unlikely genotypes as the file consumed by MERLIN pedwipe."""

    lines = [
        f"{'FAMILY':>10} {'PERSON':>10} {'MARKER':>10} {'RATIO':>10}"
    ]
    lines.extend(
        (
            f"{error.family_id:>10} {error.person_id:>10} "
            f"{error.marker_name:>10} {error.likelihood_ratio:#10.3g}"
        )
        for error in errors
    )
    return "\n".join(lines) + "\n"


def _is_informative_emission(
    states: tuple[InheritanceState, ...],
    bit_count: int,
) -> bool:
    """Exclude markers whose likelihood is constant over every state."""

    if len(states) != 2**bit_count:
        return True
    first_likelihood = states[0].likelihood
    return any(state.likelihood != first_likelihood for state in states[1:])


def _left_outside_factors(
    query_states: tuple[InheritanceState, ...],
    states_by_marker: tuple[tuple[InheritanceState, ...], ...],
    forward_weights: tuple[tuple[float, ...], ...],
    recombination_fractions: tuple[float, ...],
    marker_index: int,
) -> tuple[float, ...]:
    """Propagate all evidence left of a marker into its candidate states."""

    if marker_index == 0:
        return tuple(1.0 for _ in query_states)

    previous_states = states_by_marker[marker_index - 1]
    previous_weights = forward_weights[marker_index - 1]
    theta = recombination_fractions[marker_index - 1]
    return tuple(
        fsum(
            previous_weight
            * _transition_weight(previous_state.bits, query_state.bits, theta)
            for previous_state, previous_weight in zip(
                previous_states,
                previous_weights,
            )
        )
        for query_state in query_states
    )


def _right_outside_factors(
    query_states: tuple[InheritanceState, ...],
    states_by_marker: tuple[tuple[InheritanceState, ...], ...],
    backward_weights: tuple[tuple[float, ...], ...],
    recombination_fractions: tuple[float, ...],
    marker_index: int,
) -> tuple[float, ...]:
    """Propagate all evidence right of a marker into its candidate states."""

    if marker_index == len(states_by_marker) - 1:
        return tuple(1.0 for _ in query_states)

    next_states = states_by_marker[marker_index + 1]
    next_weights = backward_weights[marker_index + 1]
    theta = recombination_fractions[marker_index]
    return tuple(
        fsum(
            _transition_weight(query_state.bits, next_state.bits, theta)
            * next_state.likelihood
            * next_weight
            for next_state, next_weight in zip(next_states, next_weights)
        )
        for query_state in query_states
    )
