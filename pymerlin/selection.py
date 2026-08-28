"""Marker selection helpers for CLI and batch APIs."""

from __future__ import annotations

from collections import defaultdict

from .models import Dataset, Marker


def select_markers(
    dataset: Dataset,
    marker_names: list[str] | None = None,
    chromosome: str | None = None,
    start_cm: float | None = None,
    end_cm: float | None = None,
) -> tuple[Marker, ...]:
    """Select markers by explicit names and/or a map interval."""

    selected = list(dataset.markers)
    if marker_names:
        requested = set(marker_names)
        missing = requested.difference(dataset.marker_by_name)
        if missing:
            raise ValueError(f"Unknown marker(s): {', '.join(sorted(missing))}")
        selected = [dataset.marker_by_name[name] for name in marker_names]

    if chromosome is not None:
        selected = [marker for marker in selected if marker.chromosome == chromosome]
    if start_cm is not None:
        selected = [
            marker
            for marker in selected
            if marker.position_cm is not None and marker.position_cm >= start_cm
        ]
    if end_cm is not None:
        selected = [
            marker
            for marker in selected
            if marker.position_cm is not None and marker.position_cm <= end_cm
        ]

    if not selected:
        raise ValueError("No markers matched the requested selection.")
    return tuple(selected)


def partition_dataset_by_chromosome(
    dataset: Dataset,
) -> tuple[Dataset, ...]:
    """Create one position-ordered dataset per mapped chromosome."""

    markers_by_chromosome: dict[str, list[Marker]] = defaultdict(list)
    for marker in dataset.markers:
        if marker.chromosome is None or marker.position_cm is None:
            raise ValueError(
                "Chromosome partitioning requires a mapped position for "
                f"marker {marker.name!r}."
            )
        markers_by_chromosome[marker.chromosome].append(marker)

    if not markers_by_chromosome:
        raise ValueError("Chromosome partitioning requires at least one marker.")

    chromosome_datasets = []
    for chromosome in sorted(
        markers_by_chromosome,
        key=_merlin_chromosome_sort_key,
    ):
        # Python's stable sort preserves DAT order when positions are tied,
        # matching MERLIN's marker serial-number tie break.
        markers = tuple(
            sorted(
                markers_by_chromosome[chromosome],
                key=lambda marker: float(marker.position_cm),
            )
        )
        chromosome_datasets.append(
            Dataset(
                markers=markers,
                families=dataset.families,
                affection_names=dataset.affection_names,
            )
        )

    return tuple(chromosome_datasets)


def _merlin_chromosome_sort_key(chromosome: str) -> tuple[int, int | str]:
    if chromosome.upper() == "X":
        return (0, 999)
    try:
        return (0, int(chromosome))
    except ValueError:
        return (1, chromosome.upper())
