"""Comparison helpers for validating PyMerlin against MERLIN output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from .ibd import estimate_ibd_for_markers
from .io import load_merlin_inputs
from .merlin_runner import run_merlin_singlepoint_ibd
from .models import Dataset


@dataclass(frozen=True)
class IbdMismatch:
    """A single pairwise IBD discrepancy against MERLIN."""

    marker: str
    family_id: str
    id1: str
    id2: str
    merlin: tuple[float, float, float]
    pymerlin: tuple[float, float, float]
    max_abs_diff: float


def compare_singlepoint_ibd_to_merlin(
    dataset: Dataset,
    merlin_ibd_path: str | Path,
    marker_names: list[str] | None = None,
    tolerance: float = 1e-4,
) -> tuple[IbdMismatch, ...]:
    """Compare PyMerlin single-point IBD probabilities to a MERLIN `.ibd` file."""

    marker_lookup = _marker_token_lookup(dataset)
    merlin_rows = _read_merlin_ibd(merlin_ibd_path, marker_lookup)
    selected_markers = marker_names or sorted({key[0] for key in merlin_rows})
    pymerlin_rows = _pymerlin_ibd_rows(dataset, selected_markers)

    mismatches: list[IbdMismatch] = []
    for key, merlin_values in merlin_rows.items():
        marker, family_id, id1, id2 = key
        if marker not in selected_markers:
            continue
        pymerlin_values = pymerlin_rows[key]
        max_abs_diff = max(abs(merlin_values[index] - pymerlin_values[index]) for index in range(3))
        if max_abs_diff > tolerance:
            mismatches.append(
                IbdMismatch(
                    marker=marker,
                    family_id=family_id,
                    id1=id1,
                    id2=id2,
                    merlin=merlin_values,
                    pymerlin=pymerlin_values,
                    max_abs_diff=max_abs_diff,
                )
            )
    return tuple(mismatches)


def compare_singlepoint_ibd_to_merlin_executable(
    merlin_executable: str | Path,
    ped_path: str | Path,
    dat_path: str | Path,
    map_path: str | Path | None = None,
    freq_path: str | Path | None = None,
    tolerance: float = 1e-4,
) -> tuple[IbdMismatch, ...]:
    """Run MERLIN on the same files and compare its single-point IBD output."""

    merlin_output = run_merlin_singlepoint_ibd(
        merlin_executable,
        ped_path,
        dat_path,
        map_path,
        freq_path,
    )
    dataset = load_merlin_inputs(ped_path, dat_path, map_path, freq_path)
    with NamedTemporaryFile("w", suffix=".ibd") as merlin_ibd_file:
        merlin_ibd_file.write(merlin_output)
        merlin_ibd_file.flush()
        return compare_singlepoint_ibd_to_merlin(
            dataset,
            merlin_ibd_file.name,
            tolerance=tolerance,
        )


def _read_merlin_ibd(
    path: str | Path,
    marker_lookup: dict[str, str],
) -> dict[tuple[str, str, str, str], tuple[float, float, float]]:
    rows: dict[tuple[str, str, str, str], tuple[float, float, float]] = {}
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("FAMILY"):
            continue
        family_id, id1, id2, marker_token, p0, p1, p2 = stripped.split()
        if id1 == id2:
            continue
        first_id, second_id = sorted((id1, id2), key=_natural_id_key)
        marker_name = marker_lookup.get(marker_token, marker_token)
        rows[(marker_name, family_id, first_id, second_id)] = (float(p0), float(p1), float(p2))
    return rows


def _pymerlin_ibd_rows(
    dataset: Dataset,
    marker_names: list[str],
) -> dict[tuple[str, str, str, str], tuple[float, float, float]]:
    rows: dict[tuple[str, str, str, str], tuple[float, float, float]] = {}
    for result in estimate_ibd_for_markers(dataset, marker_names=marker_names):
        for row in result.rows:
            key = (
                result.marker_name,
                str(row["family_id"]),
                str(row["id1"]),
                str(row["id2"]),
            )
            rows[key] = (float(row["z0"]), float(row["z1"]), float(row["z2"]))
    return rows


def _marker_token_lookup(dataset: Dataset) -> dict[str, str]:
    lookup = {marker.name: marker.name for marker in dataset.markers}
    for marker in dataset.markers:
        if marker.position_cm is not None:
            lookup[f"{marker.position_cm:.3f}"] = marker.name
    return lookup


def _natural_id_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):020d}") if value.isdigit() else (1, value)
