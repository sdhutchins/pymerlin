"""Linear and exponential Kong-Cox aggregation of family NPL scores."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from math import fsum

from .npl import (
    AffectionNplPairsResult,
    FamilyNplPairsResult,
    NplPairsResult,
)
from .positions import AnalysisPosition


_GOLD = 0.61803399
_COMPLEMENTARY_GOLD = 0.38196601
_BRENT_TOLERANCE = 1.0e-6
_BRENT_ZERO_TOLERANCE = 3.0e-10
_BRENT_MAX_ITERATIONS = 200
_LOD_SCALE = 0.2171472409516


@dataclass(frozen=True)
class LinearKongCoxRow:
    """One linear Kong-Cox summary row."""

    label: str
    position_cm: float | None
    z_score: float
    z_p_value: float
    delta: float
    lod_score: float
    lod_p_value: float


@dataclass(frozen=True)
class LinearKongCoxAnalysisResult:
    """Linear Kong-Cox results for one affection phenotype."""

    affection_name: str
    informative_family_count: int
    minimum: LinearKongCoxRow | None
    maximum: LinearKongCoxRow | None
    rows: tuple[LinearKongCoxRow, ...]


@dataclass(frozen=True)
class LinearKongCoxResult:
    """Linear Kong-Cox results for one chromosome."""

    chromosome: str
    positions: tuple[AnalysisPosition, ...]
    analyses: tuple[LinearKongCoxAnalysisResult, ...]


@dataclass(frozen=True)
class ExponentialKongCoxRow:
    """One exponential Kong-Cox likelihood summary."""

    label: str
    position_cm: float | None
    delta: float
    lod_score: float
    lod_p_value: float


@dataclass(frozen=True)
class ExponentialKongCoxAnalysisResult:
    """Exponential Kong-Cox results for one affection phenotype."""

    affection_name: str
    informative_family_count: int
    minimum: ExponentialKongCoxRow | None
    maximum: ExponentialKongCoxRow | None
    rows: tuple[ExponentialKongCoxRow, ...]


@dataclass(frozen=True)
class ExponentialKongCoxResult:
    """Exponential Kong-Cox results for one chromosome."""

    chromosome: str
    positions: tuple[AnalysisPosition, ...]
    analyses: tuple[ExponentialKongCoxAnalysisResult, ...]


def linear_kong_cox(npl_result: NplPairsResult) -> LinearKongCoxResult:
    """Aggregate family NPL-pairs Z scores using MERLIN's linear model."""

    analyses = tuple(
        _aggregate_affection(analysis, npl_result.positions)
        for analysis in npl_result.analyses
    )
    return LinearKongCoxResult(
        chromosome=npl_result.chromosome,
        positions=npl_result.positions,
        analyses=analyses,
    )


def exponential_kong_cox(
    npl_result: NplPairsResult,
) -> ExponentialKongCoxResult:
    """Aggregate full family NPL distributions with MERLIN's exp model."""

    analyses = tuple(
        _aggregate_exponential_affection(analysis, npl_result.positions)
        for analysis in npl_result.analyses
    )
    return ExponentialKongCoxResult(
        chromosome=npl_result.chromosome,
        positions=npl_result.positions,
        analyses=analyses,
    )


def format_merlin_linear_kong_cox_table(
    chromosome_results: tuple[LinearKongCoxResult, ...],
) -> str:
    """Format linear results using MERLIN's nonparametric table layout."""

    return format_merlin_kong_cox_table(chromosome_results)


def format_merlin_kong_cox_table(
    linear_results: tuple[LinearKongCoxResult, ...],
    exponential_results: tuple[ExponentialKongCoxResult, ...] | None = None,
) -> str:
    """Format linear and optional exponential results as MERLIN does."""

    header = "CHR\tPOS\tLABEL\tANALYSIS\tZSCORE\tDELTA\tLOD\tPVALUE"
    if exponential_results is not None:
        header += "\tExDELTA\tExLOD\tPVALUE"
    lines = [header]
    exponential_lookup = _exponential_analysis_lookup(exponential_results)

    for chromosome_result in linear_results:
        try:
            chromosome_number = int(chromosome_result.chromosome)
        except ValueError as error:
            raise ValueError(
                "MERLIN autosomal linkage output requires a numeric chromosome."
            ) from error

        for analysis in chromosome_result.analyses:
            if analysis.minimum is None or analysis.maximum is None:
                continue
            analysis_label = f"{analysis.affection_name} [Pairs]"
            exponential_analysis = exponential_lookup.get(
                (chromosome_result.chromosome, analysis.affection_name)
            )
            if exponential_results is not None and exponential_analysis is None:
                raise ValueError(
                    "Missing exponential result for chromosome "
                    f"{chromosome_result.chromosome!r} and affection "
                    f"{analysis.affection_name!r}."
                )
            if exponential_analysis is not None and (
                exponential_analysis.minimum is None
                or exponential_analysis.maximum is None
            ):
                raise ValueError(
                    "Exponential bounds are missing for an informative analysis."
                )
            lines.append(
                _format_table_row(
                    "na",
                    "na",
                    analysis.minimum,
                    analysis_label,
                    (
                        exponential_analysis.minimum
                        if exponential_analysis is not None
                        else None
                    ),
                )
            )
            lines.append(
                _format_table_row(
                    "na",
                    "na",
                    analysis.maximum,
                    analysis_label,
                    (
                        exponential_analysis.maximum
                        if exponential_analysis is not None
                        else None
                    ),
                )
            )
            exponential_rows = (
                {row.label: row for row in exponential_analysis.rows}
                if exponential_analysis is not None
                else {}
            )
            for row in analysis.rows:
                if row.position_cm is None:
                    raise ValueError("A position row requires a map coordinate.")
                exponential_row = exponential_rows.get(row.label)
                if exponential_analysis is not None and exponential_row is None:
                    raise ValueError(
                        f"Missing exponential result for position {row.label!r}."
                    )
                lines.append(
                    _format_table_row(
                        str(chromosome_number),
                        _format_fixed(row.position_cm, 3),
                        row,
                        analysis_label,
                        exponential_row,
                    )
                )

    return "\n".join(lines) + "\n"


def format_merlin_linear_kong_cox_console(
    chromosome_results: tuple[LinearKongCoxResult, ...],
) -> str:
    """Format the scientific linear-model table shown by MERLIN."""

    return format_merlin_kong_cox_console(chromosome_results)


def format_merlin_kong_cox_console(
    linear_results: tuple[LinearKongCoxResult, ...],
    exponential_results: tuple[ExponentialKongCoxResult, ...] | None = None,
) -> str:
    """Format linear and optional exponential console linkage results."""

    lines: list[str] = []
    exponential_lookup = _exponential_analysis_lookup(exponential_results)
    for chromosome_result in linear_results:
        for analysis in chromosome_result.analyses:
            analysis_label = f"{analysis.affection_name} [Pairs]"
            if analysis.minimum is None or analysis.maximum is None:
                lines.extend(
                    [f"No informative families for {analysis_label}", ""]
                )
                continue

            exponential_analysis = exponential_lookup.get(
                (chromosome_result.chromosome, analysis.affection_name)
            )
            if exponential_results is not None and exponential_analysis is None:
                raise ValueError(
                    "Missing exponential result for console linkage output."
                )
            if exponential_analysis is not None and (
                exponential_analysis.minimum is None
                or exponential_analysis.maximum is None
            ):
                raise ValueError(
                    "Exponential console bounds are missing."
                )

            family_word = (
                "family"
                if analysis.informative_family_count == 1
                else "families"
            )
            lines.extend(
                [
                    f"Phenotype: {analysis_label} "
                    f"({analysis.informative_family_count} {family_word})",
                    (
                        "======================================================"
                        "========================="
                        if exponential_analysis is not None
                        else "======================================================"
                    ),
                    (
                        "            Pos   Zmean  pvalue linDelta    LOD  pvalue "
                        "expDelta    LOD  pvalue"
                        if exponential_analysis is not None
                        else "            Pos   Zmean  pvalue    delta    LOD  pvalue"
                    ),
                    _format_combined_console_row(
                        analysis.minimum,
                        (
                            exponential_analysis.minimum
                            if exponential_analysis is not None
                            else None
                        ),
                    ),
                    _format_combined_console_row(
                        analysis.maximum,
                        (
                            exponential_analysis.maximum
                            if exponential_analysis is not None
                            else None
                        ),
                    ),
                ]
            )
            exponential_rows = (
                {row.label: row for row in exponential_analysis.rows}
                if exponential_analysis is not None
                else {}
            )
            for row in analysis.rows:
                exponential_row = exponential_rows.get(row.label)
                if exponential_analysis is not None and exponential_row is None:
                    raise ValueError(
                        "Missing exponential console result for position "
                        f"{row.label!r}."
                    )
                lines.append(
                    _format_combined_console_row(row, exponential_row)
                )
            lines.append("")

    return "\n".join(lines) + ("\n" if lines else "")


def _aggregate_affection(
    analysis: AffectionNplPairsResult,
    positions: tuple[AnalysisPosition, ...],
) -> LinearKongCoxAnalysisResult:
    informative_families = tuple(
        family for family in analysis.families if family.informative
    )
    if not informative_families:
        return LinearKongCoxAnalysisResult(
            affection_name=analysis.affection_name,
            informative_family_count=0,
            minimum=None,
            maximum=None,
            rows=(),
        )

    lower_score_bound = min(
        0.0,
        *(family.z_min for family in informative_families),
    )
    upper_score_bound = max(
        0.0,
        *(family.z_max for family in informative_families),
    )
    if lower_score_bound == upper_score_bound:
        return LinearKongCoxAnalysisResult(
            affection_name=analysis.affection_name,
            informative_family_count=0,
            minimum=None,
            maximum=None,
            rows=(),
        )

    minimum_delta = -1.0 / upper_score_bound
    maximum_delta = -1.0 / lower_score_bound
    aggregate_scale = 1.0 / math.sqrt(len(informative_families))

    minimum = _fixed_delta_row(
        label="min",
        family_scores=tuple(
            family.z_min for family in informative_families
        ),
        aggregate_scale=aggregate_scale,
        delta=minimum_delta,
    )
    maximum = _fixed_delta_row(
        label="max",
        family_scores=tuple(
            family.z_max for family in informative_families
        ),
        aggregate_scale=aggregate_scale,
        delta=maximum_delta,
    )

    rows = []
    for position_index, position in enumerate(positions):
        family_scores = tuple(
            family.z_scores[position_index]
            for family in informative_families
        )
        z_score = fsum(family_scores) * aggregate_scale
        delta, chi_square = _maximize_linear_likelihood(
            family_scores,
            minimum_delta,
            maximum_delta,
        )
        lod_score, lod_p_value = _lod_summary(delta, chi_square)
        rows.append(
            LinearKongCoxRow(
                label=position.label,
                position_cm=position.position_cm,
                z_score=z_score,
                z_p_value=_normal_survival(z_score),
                delta=delta,
                lod_score=lod_score,
                lod_p_value=lod_p_value,
            )
        )

    return LinearKongCoxAnalysisResult(
        affection_name=analysis.affection_name,
        informative_family_count=len(informative_families),
        minimum=minimum,
        maximum=maximum,
        rows=tuple(rows),
    )


def _aggregate_exponential_affection(
    analysis: AffectionNplPairsResult,
    positions: tuple[AnalysisPosition, ...],
) -> ExponentialKongCoxAnalysisResult:
    informative_families = tuple(
        family
        for family in analysis.families
        if len(family.standardized_score_values) > 1
    )
    if not informative_families:
        return ExponentialKongCoxAnalysisResult(
            affection_name=analysis.affection_name,
            informative_family_count=0,
            minimum=None,
            maximum=None,
            rows=(),
        )

    lower_score = min(
        0.0,
        *(family.standardized_score_values[0] for family in informative_families),
    )
    upper_score = max(
        0.0,
        *(family.standardized_score_values[-1] for family in informative_families),
    )
    # Preserve MERLIN's published implementation exactly, including the
    # positive extreme lower bound when a standardized score is below -10.
    minimum_delta = 100.0 / abs(lower_score) if lower_score < -10.0 else -9.999
    maximum_delta = 100.0 / abs(upper_score) if upper_score > 10.0 else 9.999

    minimum = _fixed_exponential_row(
        "min",
        informative_families,
        minimum_delta,
    )
    maximum = _fixed_exponential_row(
        "max",
        informative_families,
        maximum_delta,
    )
    rows = tuple(
        _fit_exponential_position(
            position,
            position_index,
            informative_families,
            minimum_delta,
            maximum_delta,
        )
        for position_index, position in enumerate(positions)
    )
    return ExponentialKongCoxAnalysisResult(
        affection_name=analysis.affection_name,
        informative_family_count=len(informative_families),
        minimum=minimum,
        maximum=maximum,
        rows=rows,
    )


def _fixed_exponential_row(
    label: str,
    families: tuple[FamilyNplPairsResult, ...],
    delta: float,
) -> ExponentialKongCoxRow:
    log_likelihood_terms = []
    for family in families:
        scores = family.standardized_score_values
        log_squared_factor_sum = _log_exponential_sum(2.0 * delta, scores)
        log_factor_sum = _log_exponential_sum(delta, scores)
        log_null_scale = _log_weighted_exponential_sum(
            delta,
            scores,
            family.null_probabilities,
        )
        log_likelihood_terms.append(
            log_squared_factor_sum - log_factor_sum - log_null_scale
        )

    chi_square = 2.0 * fsum(log_likelihood_terms)
    lod_score, lod_p_value = _lod_summary(delta, chi_square)
    return ExponentialKongCoxRow(
        label=label,
        position_cm=None,
        delta=delta,
        lod_score=lod_score,
        lod_p_value=lod_p_value,
    )


def _fit_exponential_position(
    position: AnalysisPosition,
    position_index: int,
    families: tuple[FamilyNplPairsResult, ...],
    minimum_delta: float,
    maximum_delta: float,
) -> ExponentialKongCoxRow:
    objective = lambda delta: _exponential_objective(
        delta,
        families,
        position_index,
    )
    delta, minimum_objective = _merlin_brent_minimize(
        objective,
        minimum_delta,
        maximum_delta,
    )
    chi_square = -minimum_objective
    lod_score, lod_p_value = _lod_summary(delta, chi_square)
    return ExponentialKongCoxRow(
        label=position.label,
        position_cm=position.position_cm,
        delta=delta,
        lod_score=lod_score,
        lod_p_value=lod_p_value,
    )


def _exponential_objective(
    delta: float,
    families: tuple[FamilyNplPairsResult, ...],
    position_index: int,
) -> float:
    log_likelihood_terms = tuple(
        _log_weighted_exponential_sum(
            delta,
            family.standardized_score_values,
            family.posterior_probabilities[position_index],
        )
        - _log_weighted_exponential_sum(
            delta,
            family.standardized_score_values,
            family.null_probabilities,
        )
        for family in families
    )
    return -2.0 * fsum(log_likelihood_terms)


def _log_weighted_exponential_sum(
    delta: float,
    scores: tuple[float, ...],
    probabilities: tuple[float, ...],
) -> float:
    """Evaluate a weighted exponential sum without avoidable overflow."""

    if len(scores) != len(probabilities):
        raise ValueError("Every NPL score requires a probability.")

    exponents = tuple(delta * score for score in scores)
    maximum_exponent = max(exponents)
    scaled_sum = fsum(
        probability * math.exp(exponent - maximum_exponent)
        for probability, exponent in zip(probabilities, exponents)
    )
    if scaled_sum <= 0.0:
        raise ValueError("Exponential NPL probability mass is zero.")
    return maximum_exponent + math.log(scaled_sum)


def _log_exponential_sum(
    delta: float,
    scores: tuple[float, ...],
) -> float:
    """Evaluate an unweighted exponential sum on the log scale."""

    exponents = tuple(delta * score for score in scores)
    maximum_exponent = max(exponents)
    return maximum_exponent + math.log(
        fsum(math.exp(exponent - maximum_exponent) for exponent in exponents)
    )


def _fixed_delta_row(
    label: str,
    family_scores: tuple[float, ...],
    aggregate_scale: float,
    delta: float,
) -> LinearKongCoxRow:
    z_score = fsum(family_scores) * aggregate_scale
    chi_square = -_linear_objective(delta, family_scores)
    lod_score, lod_p_value = _lod_summary(delta, chi_square)
    return LinearKongCoxRow(
        label=label,
        position_cm=None,
        z_score=z_score,
        z_p_value=_normal_survival(z_score),
        delta=delta,
        lod_score=lod_score,
        lod_p_value=lod_p_value,
    )


def _maximize_linear_likelihood(
    family_scores: tuple[float, ...],
    minimum_delta: float,
    maximum_delta: float,
) -> tuple[float, float]:
    objective = lambda delta: _linear_objective(delta, family_scores)
    delta, minimum_objective = _merlin_brent_minimize(
        objective,
        minimum_delta,
        maximum_delta,
    )
    return delta, -minimum_objective


def _linear_objective(
    delta: float,
    family_scores: tuple[float, ...],
) -> float:
    terms = []
    for score in family_scores:
        likelihood_factor = 1.0 + delta * score
        if likelihood_factor <= 0.0:
            return math.inf
        terms.append(2.0 * math.log(likelihood_factor))
    return -fsum(terms)


def _merlin_brent_minimize(
    objective: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    """Port MERLIN's bounded ScalarMinimizer::Brent implementation."""

    a = lower_bound
    c = upper_bound
    if a > c:
        a, c = c, a

    minimum = a + (c - a) * _GOLD
    minimum_value = objective(minimum)
    w = minimum
    v = minimum
    w_value = minimum_value
    v_value = minimum_value
    previous_step = 0.0
    proposed_step = 0.0

    for _ in range(_BRENT_MAX_ITERATIONS):
        middle = 0.5 * (a + c)
        tolerance = (
            _BRENT_TOLERANCE * abs(minimum) + _BRENT_ZERO_TOLERANCE
        )
        doubled_tolerance = 2.0 * tolerance
        if abs(minimum - middle) <= (
            doubled_tolerance - 0.5 * (c - a)
        ):
            return minimum, minimum_value

        if abs(previous_step) > tolerance:
            r = (minimum - w) * (minimum_value - v_value)
            q = (minimum - v) * (minimum_value - w_value)
            p = (minimum - v) * q - (minimum - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            older_step = previous_step
            previous_step = proposed_step

            if (
                abs(p) >= abs(0.5 * q * older_step)
                or p <= q * (a - minimum)
                or p >= q * (c - minimum)
            ):
                previous_step = a - minimum if minimum >= middle else c - minimum
                proposed_step = _COMPLEMENTARY_GOLD * previous_step
            else:
                proposed_step = p / q
                candidate = minimum + proposed_step
                if (
                    candidate - a < doubled_tolerance
                    or c - candidate < doubled_tolerance
                ):
                    proposed_step = _signed_magnitude(
                        tolerance,
                        middle - minimum,
                    )
        else:
            previous_step = a - minimum if minimum >= middle else c - minimum
            proposed_step = _COMPLEMENTARY_GOLD * previous_step

        candidate = (
            minimum + proposed_step
            if abs(proposed_step) >= tolerance
            else minimum + _signed_magnitude(tolerance, proposed_step)
        )
        candidate_value = objective(candidate)

        if candidate_value <= minimum_value:
            if candidate >= minimum:
                a = minimum
            else:
                c = minimum
            v, w, minimum = w, minimum, candidate
            v_value, w_value, minimum_value = (
                w_value,
                minimum_value,
                candidate_value,
            )
        else:
            if candidate < minimum:
                a = candidate
            else:
                c = candidate
            if candidate_value <= w_value or w == minimum:
                v, w = w, candidate
                v_value, w_value = w_value, candidate_value
            elif candidate_value <= v_value or v == minimum or v == w:
                v = candidate
                v_value = candidate_value

    raise RuntimeError("Kong-Cox optimization did not converge.")


def _lod_summary(delta: float, chi_square: float) -> tuple[float, float]:
    lod_score = chi_square * _LOD_SCALE
    lod_p_value = (
        0.5 * math.erfc(math.sqrt(0.5 * chi_square))
        if chi_square > 0.0
        else 0.5
    )
    if delta < 0.0:
        lod_score = -lod_score
        lod_p_value = 1.0 - lod_p_value
    return lod_score, lod_p_value


def _normal_survival(z_score: float) -> float:
    return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def _signed_magnitude(magnitude: float, direction: float) -> float:
    """Match MERLIN's sign helper, including its treatment of negative zero."""

    return abs(magnitude) if direction >= 0.0 else -abs(magnitude)


def _format_table_row(
    chromosome: str,
    position: str,
    row: LinearKongCoxRow,
    analysis_label: str,
    exponential_row: ExponentialKongCoxRow | None = None,
) -> str:
    values = [
        chromosome,
        position,
        row.label,
        analysis_label,
        _format_fixed(row.z_score, 3),
        _format_fixed(row.delta, 3),
        _format_fixed(row.lod_score, 3),
        f"{row.lod_p_value:.4g}",
    ]
    if exponential_row is not None:
        values.extend(
            [
                _format_fixed(exponential_row.delta, 3),
                _format_fixed(exponential_row.lod_score, 3),
                f"{exponential_row.lod_p_value:.4g}",
            ]
        )
    return "\t".join(values)


def _format_console_row(row: LinearKongCoxRow) -> str:
    z_p_digits = _p_value_digits(row.z_p_value)
    lod_p_digits = _p_value_digits(row.lod_p_value)
    lod_digits = (
        2
        if abs(row.lod_score) < 100.0
        else 1 if abs(row.lod_score) < 1000.0 else 0
    )
    return (
        f"{row.label:>15} {_format_fixed(row.z_score, 2):>7} "
        f"{row.z_p_value:7.{z_p_digits}f} "
        f"{_format_fixed(row.delta, 3):>8} "
        f"{_format_fixed(row.lod_score, lod_digits):>6} "
        f"{row.lod_p_value:7.{lod_p_digits}f} "
    )


def _format_combined_console_row(
    linear_row: LinearKongCoxRow,
    exponential_row: ExponentialKongCoxRow | None,
) -> str:
    formatted = _format_console_row(linear_row)
    if exponential_row is None:
        return formatted

    p_value_digits = _p_value_digits(exponential_row.lod_p_value)
    lod_digits = (
        2
        if abs(exponential_row.lod_score) < 100.0
        else 1 if abs(exponential_row.lod_score) < 1000.0 else 0
    )
    return (
        formatted
        + f"{_format_fixed(exponential_row.delta, 3):>8} "
        + f"{_format_fixed(exponential_row.lod_score, lod_digits):>6} "
        + f"{exponential_row.lod_p_value:7.{p_value_digits}f} "
    )


def _exponential_analysis_lookup(
    results: tuple[ExponentialKongCoxResult, ...] | None,
) -> dict[tuple[str, str], ExponentialKongCoxAnalysisResult]:
    if results is None:
        return {}
    return {
        (result.chromosome, analysis.affection_name): analysis
        for result in results
        for analysis in result.analyses
    }


def _p_value_digits(p_value: float) -> int:
    if p_value < 1.5e-4:
        return 5
    return int(math.log(p_value * 0.06666666) * -0.4343)


def _format_fixed(value: float, digits: int) -> str:
    formatted = f"{value:.{digits}f}"
    negative_zero = (
        "-0" if digits == 0 else f"-0.{''.join('0' for _ in range(digits))}"
    )
    return formatted.removeprefix("-") if formatted == negative_zero else formatted
