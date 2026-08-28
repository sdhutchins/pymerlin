"""Accuracy tests against MERLIN output and an independent MPFR oracle."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from gmpy2 import mpfr

from pymerlin import (
    AnalysisPosition,
    estimate_ibd,
    load_merlin_inputs,
    multipoint_ibd,
    multipoint_ibd_at_positions,
)
from pymerlin.models import Dataset
from tests.oracles.mpfr_multipoint import (
    mpfr_multipoint_ibd,
    mpfr_multipoint_ibd_at_positions,
)


IbdKey = tuple[str, str, str, str]
MerlinProbabilities = tuple[str, str, str]
AnalysisMode = Literal["singlepoint", "multipoint"]


def test_basic2_default_singlepoint_ibd_accuracy() -> None:
    """Validate default-frequency single-point IBD against MERLIN and MPFR."""

    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
    )

    _assert_ibd_accuracy(
        dataset,
        Path("tests/fixtures/merlin/basic2_default_single.ibd"),
        {},
        analysis_mode="singlepoint",
    )


def test_basic2_explicit_frequency_singlepoint_ibd_accuracy() -> None:
    """Validate explicit-frequency single-point IBD against MERLIN and MPFR."""

    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    _assert_ibd_accuracy(
        dataset,
        Path("tests/fixtures/merlin/basic2_freq_single.ibd"),
        {},
        analysis_mode="singlepoint",
    )


def test_basic2_two_marker_multipoint_ibd_accuracy() -> None:
    """Validate explicit-frequency two-marker IBD against MERLIN and MPFR."""

    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    _assert_ibd_accuracy(
        dataset,
        Path("tests/fixtures/merlin/basic2_multipoint_freq.ibd"),
        {"123.400": "some_marker", "136.200": "another_marker"},
        analysis_mode="multipoint",
    )


def test_sibs_three_marker_multipoint_ibd_accuracy() -> None:
    """Validate all three-marker IBD values against MERLIN and MPFR."""

    dataset = load_merlin_inputs(
        "examples/sibs.ped",
        "examples/sibs.dat",
        "examples/sibs.map",
    )
    family = next(
        family for family in dataset.families if family.family_id == "1"
    )
    family_dataset = Dataset(
        dataset.markers,
        (family,),
        dataset.affection_names,
    )

    _assert_ibd_accuracy(
        family_dataset,
        Path("tests/fixtures/merlin/sibs_three_marker_multipoint.ibd"),
        {"0.000": "SNP_1", "0.100": "SNP_2", "0.200": "SNP_3"},
        analysis_mode="multipoint",
    )


def test_basic2_intermarker_ibd_is_float64_accurate() -> None:
    """Validate an intermarker posterior against the 256-bit MPFR oracle."""

    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    analysis_positions = (
        AnalysisPosition(position_cm=129.8, label="129.800"),
    )
    result = multipoint_ibd_at_positions(dataset, analysis_positions)[0]
    oracle_values = mpfr_multipoint_ibd_at_positions(
        dataset,
        analysis_positions,
        precision_bits=256,
    )

    pymerlin_values = {
        _canonical_key(
            result.label,
            str(row["family_id"]),
            str(row["id1"]),
            str(row["id2"]),
        ): (
            float(row["z0"]),
            float(row["z1"]),
            float(row["z2"]),
        )
        for row in result.rows
    }
    canonical_oracle_values = _canonicalize_oracle_keys(oracle_values)
    assert pymerlin_values.keys() == canonical_oracle_values.keys()

    failures: list[str] = []
    for key, pymerlin_probabilities in pymerlin_values.items():
        oracle_probabilities = canonical_oracle_values[key]
        for state_index, (pymerlin_value, oracle_value) in enumerate(
            zip(pymerlin_probabilities, oracle_probabilities)
        ):
            error = abs(mpfr(pymerlin_value) - oracle_value)
            representation_unit = mpfr(math.ulp(float(oracle_value)))
            if error > representation_unit:
                failures.append(
                    f"Intermarker accuracy failure for {key=}, P{state_index}: "
                    f"{error=} > {representation_unit=}"
                )

    assert not failures, "\n".join(failures)


def _assert_ibd_accuracy(
    dataset: Dataset,
    fixture_path: Path,
    position_to_marker: dict[str, str],
    analysis_mode: AnalysisMode,
) -> None:
    merlin_values = _read_merlin_probabilities(
        fixture_path,
        position_to_marker,
    )
    fixture_marker_names = {key[0] for key in merlin_values}
    marker_names = tuple(
        marker.name
        for marker in dataset.markers
        if marker.name in fixture_marker_names
    )
    pymerlin_values = _pymerlin_probabilities(
        dataset,
        marker_names,
        analysis_mode,
    )
    oracle_values = _oracle_probabilities(
        dataset,
        marker_names,
        analysis_mode,
    )

    assert pymerlin_values.keys() == merlin_values.keys()
    assert oracle_values.keys() == merlin_values.keys()

    failures: list[str] = []
    for key, merlin_probabilities in merlin_values.items():
        pymerlin_probabilities = pymerlin_values[key]
        high_precision_probabilities = oracle_values[key]

        for state_index, merlin_text in enumerate(merlin_probabilities):
            pymerlin_value = pymerlin_probabilities[state_index]
            oracle_value = high_precision_probabilities[state_index]

            pymerlin_display = _format_probability(pymerlin_value)
            merlin_display = _format_probability(float(merlin_text))
            if pymerlin_display != merlin_display:
                failures.append(
                    f"Display mismatch for {key=}, state P{state_index}: "
                    f"{pymerlin_display=} != {merlin_display=}"
                )

            pymerlin_error = abs(mpfr(pymerlin_value) - oracle_value)
            merlin_error = abs(mpfr(merlin_text) - oracle_value)
            representation_unit = mpfr(math.ulp(float(oracle_value)))

            if pymerlin_error > max(merlin_error, representation_unit):
                failures.append(
                    f"PyMerlin is less accurate than MERLIN for {key=}, "
                    f"state P{state_index}: {pymerlin_error=} > "
                    f"{merlin_error=}, {representation_unit=}"
                )

    assert not failures, "\n".join(failures)


def _canonical_key(
    marker_name: str,
    family_id: str,
    first_id: str,
    second_id: str,
) -> IbdKey:
    lower_id, upper_id = sorted((first_id, second_id))
    return (marker_name, family_id, lower_id, upper_id)


def _pymerlin_probabilities(
    dataset: Dataset,
    marker_names: tuple[str, ...],
    analysis_mode: AnalysisMode,
) -> dict[IbdKey, tuple[float, float, float]]:
    probabilities = {}
    if analysis_mode == "singlepoint":
        results = tuple(
            estimate_ibd(dataset, marker_name) for marker_name in marker_names
        )
    else:
        results = multipoint_ibd(dataset, marker_names=list(marker_names))

    for result in results:
        for row in result.rows:
            key = _canonical_key(
                result.marker_name,
                str(row["family_id"]),
                str(row["id1"]),
                str(row["id2"]),
            )
            probabilities[key] = (
                float(row["z0"]),
                float(row["z1"]),
                float(row["z2"]),
            )
    return probabilities


def _canonicalize_oracle_keys(
    values: dict[IbdKey, tuple[mpfr, mpfr, mpfr]],
) -> dict[IbdKey, tuple[mpfr, mpfr, mpfr]]:
    return {
        _canonical_key(marker_name, family_id, first_id, second_id): probabilities
        for (
            marker_name,
            family_id,
            first_id,
            second_id,
        ), probabilities in values.items()
    }


def _oracle_probabilities(
    dataset: Dataset,
    marker_names: tuple[str, ...],
    analysis_mode: AnalysisMode,
) -> dict[IbdKey, tuple[mpfr, mpfr, mpfr]]:
    if analysis_mode == "multipoint":
        values = mpfr_multipoint_ibd(
            dataset,
            marker_names=list(marker_names),
            precision_bits=256,
        )
        return _canonicalize_oracle_keys(values)

    values = {}
    for marker_name in marker_names:
        marker_values = mpfr_multipoint_ibd(
            dataset,
            marker_names=[marker_name],
            precision_bits=256,
        )
        values.update(marker_values)
    return _canonicalize_oracle_keys(values)


def _read_merlin_probabilities(
    path: Path,
    position_to_marker: dict[str, str],
) -> dict[IbdKey, MerlinProbabilities]:
    probabilities = {}
    with path.open() as handle:
        for line in handle:
            if line.startswith("FAMILY"):
                continue
            family_id, first_id, second_id, position, z0, z1, z2 = line.split()
            if first_id == second_id:
                continue
            marker_name = position_to_marker.get(position, position)
            key = _canonical_key(
                marker_name,
                family_id,
                first_id,
                second_id,
            )
            probabilities[key] = (z0, z1, z2)
    return probabilities


def _format_probability(value: float) -> str:
    formatted = f"{value:.5f}"
    # A mathematically exact zero is preferable to MERLIN's occasional -0.00000.
    return "0.00000" if formatted == "-0.00000" else formatted
