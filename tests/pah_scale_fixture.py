"""Build synthetic marker data on the anonymized PAH pedigree topology."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pah_scale"


@dataclass(frozen=True)
class PahScaleInputPaths:
    """Paths to one generated, MERLIN-compatible PAH-scale dataset."""

    ped: Path
    dat: Path
    map: Path
    freq: Path


@dataclass(frozen=True)
class _PedigreeRow:
    family_id: str
    individual_id: str
    father_id: str
    mother_id: str
    sex: str
    affection: str

    @property
    def is_founder(self) -> bool:
        return self.father_id == "0" and self.mother_id == "0"


MARKERS = (
    ("PAH_SIM_1", "10", 0.0),
    ("PAH_SIM_2", "10", 1.0),
    ("PAH_SIM_3", "10", 2.0),
)

PAH_PARITY_INDIVIDUAL_IDS = frozenset(
    {
        "P0785",
        "P0792",
        "P0793",
        "P0794",
        "P0803",
        "P0806",
        "P0807",
        "P0902",
        "P0907",
    }
)


def build_pah_scale_inputs(output_dir: Path) -> PahScaleInputPaths:
    """Create deterministic synthetic genotypes on the PAH pedigree graph.

    The copied fixture contains only anonymized relationships, sex, and
    affection status. Founder alleles and transmissions are deterministically
    gene-dropped through the pedigree. Genotypes are then masked for everyone
    except the 23 originally sequenced cohort members.
    """

    pedigree_rows = _read_pedigree_rows(FIXTURE_DIR / "pedigree.ped")
    genotyped_ids = frozenset(
        (FIXTURE_DIR / "genotyped_ids.txt").read_text(encoding="utf-8").splitlines()
    )
    pedigree_ids = {row.individual_id for row in pedigree_rows}
    missing_genotyped_ids = genotyped_ids - pedigree_ids
    if missing_genotyped_ids:
        raise ValueError(
            "Genotyped fixture IDs are absent from the pedigree: "
            f"{sorted(missing_genotyped_ids)!r}"
        )

    return _build_synthetic_inputs(
        output_dir,
        pedigree_rows,
        genotyped_ids,
        file_stem="pah_scale",
        markers=MARKERS,
    )


def build_pah_marker_benchmark_inputs(
    output_dir: Path,
    marker_count: int = 25,
) -> PahScaleInputPaths:
    """Create a PAH-scale dataset for measuring marker-level parallelism."""

    if marker_count < 1:
        raise ValueError("A marker benchmark requires at least one marker.")
    pedigree_rows = _read_pedigree_rows(FIXTURE_DIR / "pedigree.ped")
    genotyped_ids = frozenset(
        (FIXTURE_DIR / "genotyped_ids.txt").read_text(encoding="utf-8").splitlines()
    )
    markers = tuple(
        (f"PAH_SIM_{marker_index + 1}", "10", float(marker_index))
        for marker_index in range(marker_count)
    )
    return _build_synthetic_inputs(
        output_dir,
        pedigree_rows,
        genotyped_ids,
        file_stem=f"pah_marker_benchmark_{marker_count}",
        markers=markers,
    )


def build_pah_parity_inputs(output_dir: Path) -> PahScaleInputPaths:
    """Create a MERLIN-tractable branch derived from the PAH pedigree.

    The branch retains three generations, two affected siblings, one affected
    child, five genotyped people, and missing genotypes. People whose parents
    fall outside the selected branch become boundary founders.
    """

    pedigree_rows = _read_pedigree_rows(FIXTURE_DIR / "pedigree.ped")
    genotyped_ids = frozenset(
        (FIXTURE_DIR / "genotyped_ids.txt").read_text(encoding="utf-8").splitlines()
    )
    selected_rows = _select_pedigree_branch(
        pedigree_rows,
        PAH_PARITY_INDIVIDUAL_IDS,
    )
    # Type the second affected sibling only in this synthetic fixture so the
    # parity test compares identifiable linkage estimates, not optimizer ties.
    parity_genotyped_ids = (genotyped_ids & PAH_PARITY_INDIVIDUAL_IDS) | {"P0803"}
    return _build_synthetic_inputs(
        output_dir,
        selected_rows,
        frozenset(parity_genotyped_ids),
        file_stem="pah_parity",
        markers=MARKERS,
    )


def _build_synthetic_inputs(
    output_dir: Path,
    pedigree_rows: tuple[_PedigreeRow, ...],
    genotyped_ids: frozenset[str],
    *,
    file_stem: str,
    markers: tuple[tuple[str, str, float], ...],
) -> PahScaleInputPaths:
    """Write one deterministic synthetic dataset for a selected pedigree."""

    ordered_rows = _topological_pedigree_order(pedigree_rows)
    inherited_genotypes = _gene_drop_genotypes(ordered_rows, markers)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = PahScaleInputPaths(
        ped=output_dir / f"{file_stem}.ped",
        dat=output_dir / f"{file_stem}.dat",
        map=output_dir / f"{file_stem}.map",
        freq=output_dir / f"{file_stem}.freq",
    )
    _write_dat(paths.dat, markers)
    _write_map(paths.map, markers)
    _write_freq(paths.freq, markers)
    _write_ped(
        paths.ped,
        pedigree_rows,
        inherited_genotypes,
        genotyped_ids,
        markers,
    )
    return paths


def _select_pedigree_branch(
    pedigree_rows: tuple[_PedigreeRow, ...],
    selected_ids: frozenset[str],
) -> tuple[_PedigreeRow, ...]:
    """Select one branch and promote outside-parent boundaries to founders."""

    rows_by_id = {row.individual_id: row for row in pedigree_rows}
    missing_ids = selected_ids - rows_by_id.keys()
    if missing_ids:
        raise ValueError(
            "Selected PAH parity IDs are absent from the fixture: "
            f"{sorted(missing_ids)!r}"
        )

    selected_rows: list[_PedigreeRow] = []
    for row in pedigree_rows:
        if row.individual_id not in selected_ids:
            continue
        father_id = row.father_id if row.father_id in selected_ids else "0"
        mother_id = row.mother_id if row.mother_id in selected_ids else "0"
        if (father_id == "0") != (mother_id == "0"):
            raise ValueError(
                "A bounded parity branch cannot retain only one parent for "
                f"{row.individual_id!r}."
            )
        selected_rows.append(
            _PedigreeRow(
                family_id=row.family_id,
                individual_id=row.individual_id,
                father_id=father_id,
                mother_id=mother_id,
                sex=row.sex,
                affection=row.affection,
            )
        )
    return tuple(selected_rows)


def _read_pedigree_rows(path: Path) -> tuple[_PedigreeRow, ...]:
    rows: list[_PedigreeRow] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        tokens = raw_line.split()
        if len(tokens) != 6:
            raise ValueError(f"Expected six pedigree fields at {path}:{line_number}.")
        rows.append(_PedigreeRow(*tokens))
    return tuple(rows)


def _topological_pedigree_order(
    pedigree_rows: tuple[_PedigreeRow, ...],
) -> tuple[_PedigreeRow, ...]:
    rows_by_id = {row.individual_id: row for row in pedigree_rows}
    if len(rows_by_id) != len(pedigree_rows):
        raise ValueError("The PAH-scale fixture contains duplicate IDs.")

    missing_parent_ids = {
        parent_id
        for row in pedigree_rows
        for parent_id in (row.father_id, row.mother_id)
        if parent_id != "0" and parent_id not in rows_by_id
    }
    if missing_parent_ids:
        raise ValueError(
            f"The PAH-scale fixture has missing parents: {sorted(missing_parent_ids)!r}"
        )

    ordered_rows: list[_PedigreeRow] = []
    resolved_ids: set[str] = set()
    unresolved_rows = list(pedigree_rows)
    while unresolved_rows:
        next_unresolved_rows: list[_PedigreeRow] = []
        for row in unresolved_rows:
            parent_ids = {row.father_id, row.mother_id} - {"0"}
            if parent_ids <= resolved_ids:
                ordered_rows.append(row)
                resolved_ids.add(row.individual_id)
            else:
                next_unresolved_rows.append(row)
        if len(next_unresolved_rows) == len(unresolved_rows):
            unresolved_ids = sorted(row.individual_id for row in next_unresolved_rows)
            raise ValueError(
                "Could not topologically resolve the PAH-scale pedigree: "
                f"{unresolved_ids!r}"
            )
        unresolved_rows = next_unresolved_rows

    return tuple(ordered_rows)


def _gene_drop_genotypes(
    ordered_rows: tuple[_PedigreeRow, ...],
    markers: tuple[tuple[str, str, float], ...],
) -> dict[str, tuple[tuple[str, str], ...]]:
    genotypes: dict[str, tuple[tuple[str, str], ...]] = {}
    for row in ordered_rows:
        marker_genotypes: list[tuple[str, str]] = []
        for marker_name, _, _ in markers:
            if row.is_founder:
                marker_genotypes.append(
                    (
                        _deterministic_allele(
                            "founder", row.individual_id, marker_name, "0"
                        ),
                        _deterministic_allele(
                            "founder", row.individual_id, marker_name, "1"
                        ),
                    )
                )
                continue

            if row.father_id == "0" or row.mother_id == "0":
                raise ValueError(
                    "Every nonfounder requires two parents for gene dropping: "
                    f"{row.individual_id!r}"
                )
            father_genotype = genotypes[row.father_id][len(marker_genotypes)]
            mother_genotype = genotypes[row.mother_id][len(marker_genotypes)]
            paternal_index = _deterministic_bit(
                "paternal", row.individual_id, marker_name
            )
            maternal_index = _deterministic_bit(
                "maternal", row.individual_id, marker_name
            )
            marker_genotypes.append(
                (
                    father_genotype[paternal_index],
                    mother_genotype[maternal_index],
                )
            )
        genotypes[row.individual_id] = tuple(marker_genotypes)
    return genotypes


def _deterministic_allele(*parts: str) -> str:
    return str(_deterministic_bit(*parts) + 1)


def _deterministic_bit(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return digest[0] & 1


def _write_dat(
    path: Path,
    markers: tuple[tuple[str, str, float], ...],
) -> None:
    lines = ["A HPAH", *(f"M {name}" for name, _, _ in markers)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_map(
    path: Path,
    markers: tuple[tuple[str, str, float], ...],
) -> None:
    lines = [
        f"{chromosome} {name} {position_cm:.1f}"
        for name, chromosome, position_cm in markers
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_freq(
    path: Path,
    markers: tuple[tuple[str, str, float], ...],
) -> None:
    lines: list[str] = []
    for marker_name, _, _ in markers:
        lines.extend((f"M {marker_name}", "F 0.5", "F 0.5"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ped(
    path: Path,
    pedigree_rows: tuple[_PedigreeRow, ...],
    inherited_genotypes: dict[str, tuple[tuple[str, str], ...]],
    genotyped_ids: frozenset[str],
    markers: tuple[tuple[str, str, float], ...],
) -> None:
    lines: list[str] = []
    for row in pedigree_rows:
        genotype_fields = (
            tuple(
                allele
                for genotype in inherited_genotypes[row.individual_id]
                for allele in genotype
            )
            if row.individual_id in genotyped_ids
            else ("0",) * (2 * len(markers))
        )
        lines.append(
            " ".join(
                (
                    row.family_id,
                    row.individual_id,
                    row.father_id,
                    row.mother_id,
                    row.sex,
                    row.affection,
                    *genotype_fields,
                )
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
