"""High-precision multipoint IBD oracle based on MPFR arithmetic.

This implementation intentionally repeats the numerical algorithm instead of
calling PyMerlin's floating-point helpers. It shares only the parsed data model
so that a defect in float64 normalization or reduction is visible to tests.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product

from gmpy2 import context, exp, get_context, mpfr

from pymerlin.models import Dataset, Family, Marker
from pymerlin.positions import AnalysisPosition


AlleleOrigin = tuple[str, int]
OracleIbdKey = tuple[str, str, str, str]
OracleIbdProbabilities = tuple[mpfr, mpfr, mpfr]


@dataclass(frozen=True)
class OracleState:
    """One non-zero inheritance state evaluated with MPFR."""

    bits: tuple[int, ...]
    likelihood: mpfr
    allele_origins: dict[str, tuple[AlleleOrigin, AlleleOrigin]]


def mpfr_multipoint_ibd(
    dataset: Dataset,
    marker_names: list[str] | None = None,
    precision_bits: int = 256,
) -> dict[OracleIbdKey, OracleIbdProbabilities]:
    """Calculate multipoint IBD probabilities with high-precision arithmetic.

    The returned key is ``(marker, family, id1, id2)``. Self-pairs are omitted
    because PyMerlin's public multipoint result also omits them.
    """

    if precision_bits < 128:
        raise ValueError("The MPFR oracle requires at least 128 bits.")

    with context(get_context(), precision=precision_bits):
        markers = _ordered_markers(dataset, marker_names)
        recombination_fractions = tuple(
            _haldane_recombination_fraction(
                mpfr(str(right.position_cm)) - mpfr(str(left.position_cm))
            )
            for left, right in zip(markers, markers[1:])
        )
        results: dict[OracleIbdKey, OracleIbdProbabilities] = {}

        for family in dataset.families:
            states_by_marker = tuple(
                tuple(_score_family_marker(family, marker)) for marker in markers
            )
            posterior_weights = _posterior_state_weights(
                states_by_marker,
                recombination_fractions,
            )
            people = tuple(person.individual_id for person in family.individuals)

            for marker, states, weights in zip(
                markers,
                states_by_marker,
                posterior_weights,
                strict=True,
            ):
                accumulator = _accumulate_ibd(people, states, weights)
                for (first_id, second_id), probabilities in accumulator.items():
                    results[
                        (marker.name, family.family_id, first_id, second_id)
                    ] = probabilities

        return results


def mpfr_multipoint_ibd_at_positions(
    dataset: Dataset,
    analysis_positions: tuple[AnalysisPosition, ...],
    marker_names: list[str] | None = None,
    precision_bits: int = 256,
) -> dict[OracleIbdKey, OracleIbdProbabilities]:
    """Calculate high-precision IBD at marker or intermarker positions."""

    if precision_bits < 128:
        raise ValueError("The MPFR oracle requires at least 128 bits.")

    with context(get_context(), precision=precision_bits):
        markers = _ordered_markers(dataset, marker_names)
        recombination_fractions = tuple(
            _haldane_recombination_fraction(
                mpfr(str(right.position_cm)) - mpfr(str(left.position_cm))
            )
            for left, right in zip(markers, markers[1:])
        )
        results: dict[OracleIbdKey, OracleIbdProbabilities] = {}

        for family in dataset.families:
            states_by_marker = tuple(
                tuple(_score_family_marker(family, marker))
                for marker in markers
            )
            forward_weights, backward_weights = _forward_backward_weights(
                states_by_marker,
                recombination_fractions,
            )
            people = tuple(
                person.individual_id for person in family.individuals
            )

            for analysis_position in analysis_positions:
                states, weights = _state_weights_at_position(
                    family,
                    markers,
                    states_by_marker,
                    forward_weights,
                    backward_weights,
                    mpfr(str(analysis_position.position_cm)),
                )
                accumulator = _accumulate_ibd(people, states, weights)
                for (first_id, second_id), probabilities in accumulator.items():
                    results[
                        (
                            analysis_position.label,
                            family.family_id,
                            first_id,
                            second_id,
                        )
                    ] = probabilities

        return results


def mpfr_marker_state_likelihoods(
    family: Family,
    marker: Marker,
    precision_bits: int = 256,
) -> dict[tuple[int, ...], mpfr]:
    """Return independent high-precision likelihoods by inheritance vector."""

    if precision_bits < 128:
        raise ValueError("The MPFR oracle requires at least 128 bits.")

    with context(get_context(), precision=precision_bits):
        return {
            state.bits: state.likelihood
            for state in _score_family_marker(family, marker)
        }


def _ordered_markers(
    dataset: Dataset,
    marker_names: list[str] | None,
) -> tuple[Marker, ...]:
    if marker_names is None:
        markers = list(dataset.markers)
    else:
        markers = [dataset.marker_by_name[name] for name in marker_names]

    if not markers:
        raise ValueError("The MPFR oracle requires at least one marker.")
    if any(marker.position_cm is None for marker in markers):
        raise ValueError("The MPFR oracle requires every map position.")
    if len({marker.chromosome for marker in markers}) != 1:
        raise ValueError("The MPFR oracle requires markers from one chromosome.")

    return tuple(
        sorted(
            markers,
            key=lambda marker: (float(marker.position_cm), marker.name),
        )
    )


def _score_family_marker(family: Family, marker: Marker) -> list[OracleState]:
    frequencies = _normalized_frequencies(family, marker)
    founder_slots = tuple(
        (founder.individual_id, copy_index)
        for founder in family.founders
        for copy_index in (0, 1)
    )
    states: list[OracleState] = []

    for bits in product((0, 1), repeat=len(family.meioses)):
        allele_origins = _inheritance_origins(family, tuple(bits))
        likelihood = mpfr(0)

        for assignment_values in product(frequencies, repeat=len(founder_slots)):
            founder_assignment = dict(zip(founder_slots, assignment_values))
            assigned_alleles = {
                person_id: (
                    founder_assignment[origins[0]],
                    founder_assignment[origins[1]],
                )
                for person_id, origins in allele_origins.items()
            }
            if not _family_genotypes_match(
                family,
                marker.name,
                assigned_alleles,
            ):
                continue

            assignment_probability = mpfr(1)
            for allele in assignment_values:
                assignment_probability *= frequencies[allele]
            likelihood += assignment_probability

        if likelihood > 0:
            states.append(
                OracleState(
                    bits=tuple(bits),
                    likelihood=likelihood,
                    allele_origins=allele_origins,
                )
            )

    if not states:
        raise ValueError(
            "The MPFR multipoint likelihood is zero for "
            f"family {family.family_id!r} at marker {marker.name!r}."
        )
    return states


def _normalized_frequencies(
    family: Family,
    marker: Marker,
) -> dict[str, mpfr]:
    if marker.allele_frequencies:
        frequencies = {
            allele: mpfr(str(frequency))
            for allele, frequency in marker.allele_frequencies.items()
        }
    else:
        observed = sorted(
            {
                allele
                for person in family.individuals
                for allele in person.genotypes.get(marker.name, (None, None))
                if allele is not None
            }
        )
        if not observed:
            raise ValueError(
                f"Marker {marker.name!r} has no alleles for the MPFR oracle."
            )
        frequencies = {
            allele: mpfr(1) / len(observed) for allele in observed
        }

    total = _sum_mpfr(tuple(frequencies.values()))
    if total <= 0:
        raise ValueError(
            f"Allele frequencies for marker {marker.name!r} sum to zero."
        )
    return {allele: frequency / total for allele, frequency in frequencies.items()}


def _inheritance_origins(
    family: Family,
    bits: tuple[int, ...],
) -> dict[str, tuple[AlleleOrigin, AlleleOrigin]]:
    transmissions = {
        (meiosis.parent_id, meiosis.child_id): bit
        for meiosis, bit in zip(family.meioses, bits, strict=True)
    }
    origins = {
        founder.individual_id: (
            (founder.individual_id, 0),
            (founder.individual_id, 1),
        )
        for founder in family.founders
    }
    remaining = {
        person.individual_id
        for person in family.individuals
        if not person.is_founder
    }

    while remaining:
        progressed = False
        for person_id in tuple(remaining):
            person = family.by_id[person_id]
            if person.father_id not in origins or person.mother_id not in origins:
                continue

            paternal_bit = transmissions[(person.father_id, person_id)]
            maternal_bit = transmissions[(person.mother_id, person_id)]
            origins[person_id] = (
                origins[person.father_id][paternal_bit],
                origins[person.mother_id][maternal_bit],
            )
            remaining.remove(person_id)
            progressed = True

        if not progressed:
            raise ValueError(
                f"Could not resolve family {family.family_id!r} in the MPFR oracle."
            )

    return origins


def _family_genotypes_match(
    family: Family,
    marker_name: str,
    assigned_alleles: dict[str, tuple[str, str]],
) -> bool:
    for person in family.individuals:
        observed = person.genotypes.get(marker_name, (None, None))
        if observed[0] is None or observed[1] is None:
            continue
        if sorted(observed) != sorted(assigned_alleles[person.individual_id]):
            return False
    return True


def _haldane_recombination_fraction(distance_cm: mpfr) -> mpfr:
    if distance_cm < 0:
        raise ValueError("Map distance cannot be negative.")
    return (mpfr(1) - exp(-mpfr(2) * distance_cm / mpfr(100))) / mpfr(2)


def _posterior_state_weights(
    states_by_marker: tuple[tuple[OracleState, ...], ...],
    recombination_fractions: tuple[mpfr, ...],
) -> tuple[tuple[mpfr, ...], ...]:
    forward_weights, backward_weights = _forward_backward_weights(
        states_by_marker,
        recombination_fractions,
    )
    return tuple(
        _normalize(
            tuple(
                forward_weight * backward_weight
                for forward_weight, backward_weight in zip(
                    forward,
                    backward,
                    strict=True,
                )
            )
        )
        for forward, backward in zip(
            forward_weights,
            backward_weights,
            strict=True,
        )
    )


def _forward_backward_weights(
    states_by_marker: tuple[tuple[OracleState, ...], ...],
    recombination_fractions: tuple[mpfr, ...],
) -> tuple[
    tuple[tuple[mpfr, ...], ...],
    tuple[tuple[mpfr, ...], ...],
]:
    forward_weights: list[tuple[mpfr, ...]] = [
        _normalize(tuple(state.likelihood for state in states_by_marker[0]))
    ]

    for marker_index, theta in enumerate(recombination_fractions, start=1):
        previous_states = states_by_marker[marker_index - 1]
        current_states = states_by_marker[marker_index]
        previous_weights = forward_weights[-1]
        current_weights = tuple(
            current_state.likelihood
            * _sum_mpfr(
                tuple(
                    previous_weight
                    * _transition_weight(
                        previous_state.bits,
                        current_state.bits,
                        theta,
                    )
                    for previous_state, previous_weight in zip(
                        previous_states,
                        previous_weights,
                        strict=True,
                    )
                )
            )
            for current_state in current_states
        )
        forward_weights.append(_normalize(current_weights))

    backward_weights: list[tuple[mpfr, ...]] = [tuple()] * len(
        states_by_marker
    )
    backward_weights[-1] = tuple(mpfr(1) for _ in states_by_marker[-1])

    for marker_index in range(len(states_by_marker) - 2, -1, -1):
        current_states = states_by_marker[marker_index]
        next_states = states_by_marker[marker_index + 1]
        next_weights = backward_weights[marker_index + 1]
        theta = recombination_fractions[marker_index]
        current_weights = tuple(
            _sum_mpfr(
                tuple(
                    _transition_weight(
                        current_state.bits,
                        next_state.bits,
                        theta,
                    )
                    * next_state.likelihood
                    * next_weight
                    for next_state, next_weight in zip(
                        next_states,
                        next_weights,
                        strict=True,
                    )
                )
            )
            for current_state in current_states
        )
        backward_weights[marker_index] = _normalize(current_weights)

    return (
        tuple(forward_weights),
        tuple(backward_weights),
    )


def _state_weights_at_position(
    family: Family,
    markers: tuple[Marker, ...],
    states_by_marker: tuple[tuple[OracleState, ...], ...],
    forward_weights: tuple[tuple[mpfr, ...], ...],
    backward_weights: tuple[tuple[mpfr, ...], ...],
    position_cm: mpfr,
) -> tuple[tuple[OracleState, ...], tuple[mpfr, ...]]:
    marker_positions = tuple(
        mpfr(str(marker.position_cm)) for marker in markers
    )
    exact_indices = tuple(
        marker_index
        for marker_index, marker_position in enumerate(marker_positions)
        if marker_position == position_cm
    )
    if exact_indices:
        marker_index = exact_indices[-1]
        weights = _normalize(
            tuple(
                forward_weight * backward_weight
                for forward_weight, backward_weight in zip(
                    forward_weights[marker_index],
                    backward_weights[marker_index],
                    strict=True,
                )
            )
        )
        return states_by_marker[marker_index], weights

    # Intermarker locations have no genotype emission, so zero-likelihood
    # states omitted at markers can regain posterior mass after recombination.
    query_states = tuple(
        OracleState(
            bits=tuple(bits),
            likelihood=mpfr(1),
            allele_origins=_inheritance_origins(family, tuple(bits)),
        )
        for bits in product((0, 1), repeat=len(family.meioses))
    )
    left_index = next(
        (
            marker_index
            for marker_index in range(len(markers) - 1, -1, -1)
            if marker_positions[marker_index] < position_cm
        ),
        None,
    )
    right_index = next(
        (
            marker_index
            for marker_index, marker_position in enumerate(marker_positions)
            if marker_position > position_cm
        ),
        None,
    )

    if left_index is None:
        left_conditionals = tuple(mpfr(1) for _ in query_states)
    else:
        theta = _haldane_recombination_fraction(
            position_cm - marker_positions[left_index]
        )
        left_conditionals = tuple(
            _sum_mpfr(
                tuple(
                    marker_weight
                    * _transition_weight(
                        marker_state.bits,
                        query_state.bits,
                        theta,
                    )
                    for marker_state, marker_weight in zip(
                        states_by_marker[left_index],
                        forward_weights[left_index],
                        strict=True,
                    )
                )
            )
            for query_state in query_states
        )

    if right_index is None:
        right_conditionals = tuple(mpfr(1) for _ in query_states)
    else:
        theta = _haldane_recombination_fraction(
            marker_positions[right_index] - position_cm
        )
        right_conditionals = tuple(
            _sum_mpfr(
                tuple(
                    _transition_weight(
                        query_state.bits,
                        marker_state.bits,
                        theta,
                    )
                    * marker_state.likelihood
                    * backward_weight
                    for marker_state, backward_weight in zip(
                        states_by_marker[right_index],
                        backward_weights[right_index],
                        strict=True,
                    )
                )
            )
            for query_state in query_states
        )

    weights = _normalize(
        tuple(
            left_weight * right_weight
            for left_weight, right_weight in zip(
                left_conditionals,
                right_conditionals,
                strict=True,
            )
        )
    )
    return query_states, weights


def _transition_weight(
    left_bits: tuple[int, ...],
    right_bits: tuple[int, ...],
    theta: mpfr,
) -> mpfr:
    weight = mpfr(1)
    for left_bit, right_bit in zip(left_bits, right_bits, strict=True):
        weight *= theta if left_bit != right_bit else mpfr(1) - theta
    return weight


def _normalize(values: tuple[mpfr, ...]) -> tuple[mpfr, ...]:
    total = _sum_mpfr(values)
    if total <= 0:
        raise ValueError("The MPFR oracle cannot normalize a zero likelihood.")
    return tuple(value / total for value in values)


def _sum_mpfr(values: tuple[mpfr, ...]) -> mpfr:
    total = mpfr(0)
    for value in values:
        total += value
    return total


def _accumulate_ibd(
    people: tuple[str, ...],
    states: tuple[OracleState, ...],
    weights: tuple[mpfr, ...],
) -> dict[tuple[str, str], OracleIbdProbabilities]:
    accumulator: defaultdict[tuple[str, str], list[mpfr]] = defaultdict(
        lambda: [mpfr(0), mpfr(0), mpfr(0)]
    )

    for state, weight in zip(states, weights, strict=True):
        for index, first_id in enumerate(people):
            for second_id in people[index + 1 :]:
                shared = len(
                    set(state.allele_origins[first_id]).intersection(
                        state.allele_origins[second_id]
                    )
                )
                accumulator[(first_id, second_id)][shared] += weight

    return {
        pair: (probabilities[0], probabilities[1], probabilities[2])
        for pair, probabilities in accumulator.items()
    }
