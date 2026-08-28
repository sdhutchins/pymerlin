"""MERLIN-compatible planning of multipoint analysis positions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Dataset, Marker


@dataclass(frozen=True)
class AnalysisPosition:
    """One requested multipoint location and its user-visible label."""

    position_cm: float
    label: str
    marker_name: str | None = None


def merlin_analysis_positions(
    dataset: Dataset,
    *,
    steps_per_interval: int = 0,
    max_step_cm: float | None = None,
    min_step_cm: float | None = None,
    grid_cm: float | None = None,
    start_cm: float | None = None,
    stop_cm: float | None = None,
    position_list: str | None = None,
    use_marker_names: bool = False,
) -> tuple[AnalysisPosition, ...]:
    """Plan locations using the precedence and labels from MERLIN SetupMap."""

    markers = _ordered_mapped_markers(dataset)
    _validate_finite_options(
        max_step_cm=max_step_cm,
        min_step_cm=min_step_cm,
        grid_cm=grid_cm,
        start_cm=start_cm,
        stop_cm=stop_cm,
    )

    if position_list:
        return _explicit_positions(
            dataset,
            markers,
            position_list,
            start_cm,
            stop_cm,
        )
    if grid_cm is not None:
        return _grid_positions(
            markers,
            grid_cm,
            start_cm,
            stop_cm,
        )
    return _marker_and_interval_positions(
        markers,
        max(0, steps_per_interval),
        max_step_cm,
        min_step_cm,
        start_cm,
        stop_cm,
        use_marker_names,
    )


def _ordered_mapped_markers(dataset: Dataset) -> tuple[Marker, ...]:
    if not dataset.markers:
        raise ValueError("Multipoint analysis requires at least one marker.")
    if any(marker.position_cm is None for marker in dataset.markers):
        raise ValueError("Multipoint analysis requires a position for every marker.")
    if len({marker.chromosome for marker in dataset.markers}) != 1:
        raise ValueError("Multipoint analysis currently requires one chromosome.")
    return tuple(
        sorted(
            dataset.markers,
            key=lambda marker: (float(marker.position_cm), marker.name),
        )
    )


def _validate_finite_options(**options: float | None) -> None:
    for name, value in options.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")


def _explicit_positions(
    dataset: Dataset,
    markers: tuple[Marker, ...],
    position_list: str,
    start_cm: float | None,
    stop_cm: float | None,
) -> tuple[AnalysisPosition, ...]:
    marker_names = {marker.name for marker in markers}
    planned: list[AnalysisPosition] = []

    for raw_token in position_list.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token in marker_names:
            marker = dataset.marker_by_name[token]
            position_cm = float(marker.position_cm)
            marker_name: str | None = marker.name
        else:
            if not token[0].isdigit():
                continue
            try:
                position_cm = float(token)
            except ValueError:
                continue
            marker_name = None

        if not math.isfinite(position_cm):
            continue
        if start_cm is not None and position_cm < start_cm:
            continue
        if stop_cm is not None and position_cm > stop_cm:
            continue
        planned.append(
            AnalysisPosition(
                position_cm=position_cm,
                label=token,
                marker_name=marker_name,
            )
        )

    return tuple(sorted(planned, key=lambda position: position.position_cm))


def _grid_positions(
    markers: tuple[Marker, ...],
    grid_cm: float,
    start_cm: float | None,
    stop_cm: float | None,
) -> tuple[AnalysisPosition, ...]:
    if grid_cm <= 0.0:
        raise ValueError("--grid must be greater than zero.")

    position_cm = (
        start_cm if start_cm is not None else float(markers[0].position_cm)
    )
    map_end_cm = (
        stop_cm if stop_cm is not None else float(markers[-1].position_cm)
    )
    planned: list[AnalysisPosition] = []

    while True:
        planned.append(
            AnalysisPosition(
                position_cm=position_cm,
                label=_format_position(position_cm),
            )
        )
        if map_end_cm - position_cm < 0.001:
            break
        position_cm += grid_cm

    return tuple(planned)


def _marker_and_interval_positions(
    markers: tuple[Marker, ...],
    steps_per_interval: int,
    max_step_cm: float | None,
    min_step_cm: float | None,
    start_cm: float | None,
    stop_cm: float | None,
    use_marker_names: bool,
) -> tuple[AnalysisPosition, ...]:
    if max_step_cm is not None and max_step_cm <= 0.0:
        raise ValueError("--maxStep must be greater than zero.")
    if min_step_cm is not None and min_step_cm <= 0.0:
        raise ValueError("--minStep must be greater than zero.")

    planned: list[AnalysisPosition] = []
    last_position_cm = float(markers[0].position_cm)

    for marker_index, marker in enumerate(markers):
        marker_position_cm = float(marker.position_cm)
        if marker_position_cm > last_position_cm:
            interval_cm = marker_position_cm - last_position_cm
            interval_steps = steps_per_interval
            step_cm = interval_cm / (interval_steps + 1)

            if max_step_cm is not None and max_step_cm < step_cm:
                interval_steps = int(
                    (interval_cm / 100.0) / (max_step_cm / 100.0 + 1e-6)
                )
                step_cm = interval_cm / (interval_steps + 1)

            if min_step_cm is not None and min_step_cm > step_cm:
                # MERLIN 1.1.2 compares in cM but uses its stored command-line
                # value in this calculation. Mirroring that behavior preserves
                # position compatibility with the reference executable.
                interval_steps = int(
                    (interval_cm / 100.0) / (min_step_cm + 1e-6) - 0.999
                )

            interval_steps = max(0, interval_steps)
            for step_index in range(1, interval_steps + 1):
                position_cm = (
                    last_position_cm
                    + interval_cm * step_index / (interval_steps + 1)
                )
                planned.append(
                    AnalysisPosition(
                        position_cm=position_cm,
                        label=_format_position(position_cm),
                    )
                )

        if marker_position_cm > last_position_cm or marker_index == 0:
            planned.append(
                AnalysisPosition(
                    position_cm=marker_position_cm,
                    label=(
                        marker.name
                        if use_marker_names
                        else _format_position(marker_position_cm)
                    ),
                    marker_name=marker.name,
                )
            )
            last_position_cm = marker_position_cm

    return tuple(
        position
        for position in planned
        if (start_cm is None or position.position_cm >= start_cm)
        and (stop_cm is None or position.position_cm <= stop_cm)
    )


def _format_position(position_cm: float) -> str:
    return f"{position_cm:.3f}"
