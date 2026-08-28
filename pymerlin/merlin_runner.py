"""Helpers for running a local MERLIN executable during parity checks."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(frozen=True)
class MerlinCombinedMultipointOutputs:
    """User-visible outputs from one combined MERLIN linkage run."""

    zscores: str
    kong_cox_table: str
    information_table: str


def run_merlin_combined_multipoint_outputs(
    merlin_executable: str | Path,
    ped_path: str | Path,
    dat_path: str | Path,
    map_path: str | Path,
    freq_path: str | Path | None = None,
    compatibility_arguments: tuple[str, ...] = (),
) -> MerlinCombinedMultipointOutputs:
    """Run paired NPL, Kong-Cox, and information analyses together."""

    merlin_path = Path(merlin_executable).resolve()
    with TemporaryDirectory(prefix="pymerlin_merlin_") as temp_dir:
        prefix = Path(temp_dir) / "merlin_compare"
        command = [
            str(merlin_path),
            "-d",
            str(Path(dat_path).resolve()),
            "-p",
            str(Path(ped_path).resolve()),
            "-m",
            str(Path(map_path).resolve()),
            *compatibility_arguments,
            "--pairs",
            "--zscores",
            "--exp",
            "--information",
            "--tabulate",
            f"--prefix:{prefix}",
            "--quiet",
        ]
        if freq_path is not None:
            command.extend(["-f", str(Path(freq_path).resolve())])

        subprocess.run(command, check=True, capture_output=True, text=True)
        return MerlinCombinedMultipointOutputs(
            zscores=prefix.with_suffix(".zscore").read_text(),
            kong_cox_table=Path(
                f"{prefix}-nonparametric.tbl"
            ).read_text(),
            information_table=Path(f"{prefix}-info.tbl").read_text(),
        )


def run_merlin_error_file(
    merlin_executable: str | Path,
    ped_path: str | Path,
    dat_path: str | Path,
    map_path: str | Path,
    freq_path: str | Path | None = None,
) -> str:
    """Run MERLIN genotype error detection and return the `.err` text."""

    merlin_path = Path(merlin_executable).resolve()
    with TemporaryDirectory(prefix="pymerlin_merlin_") as temp_dir:
        prefix = Path(temp_dir) / "merlin_compare"
        command = [
            str(merlin_path),
            "-d",
            str(Path(dat_path).resolve()),
            "-p",
            str(Path(ped_path).resolve()),
            "-m",
            str(Path(map_path).resolve()),
            "--error",
            f"--prefix:{prefix}",
            "--quiet",
        ]
        if freq_path is not None:
            command.extend(["-f", str(Path(freq_path).resolve())])

        subprocess.run(command, check=True, capture_output=True, text=True)
        return prefix.with_suffix(".err").read_text()


def run_merlin_singlepoint_ibd(
    merlin_executable: str | Path,
    ped_path: str | Path,
    dat_path: str | Path,
    map_path: str | Path | None = None,
    freq_path: str | Path | None = None,
) -> str:
    """Run MERLIN `--ibd --singlepoint` and return the generated `.ibd` text."""

    merlin_path = Path(merlin_executable).resolve()
    with TemporaryDirectory(prefix="pymerlin_merlin_") as temp_dir:
        prefix = Path(temp_dir) / "merlin_compare"
        command = [
            str(merlin_path),
            "-d",
            str(Path(dat_path).resolve()),
            "-p",
            str(Path(ped_path).resolve()),
            "--ibd",
            "--singlepoint",
            f"--prefix:{prefix}",
            "--quiet",
        ]
        if map_path is not None:
            command.extend(["-m", str(Path(map_path).resolve())])
        if freq_path is not None:
            command.extend(["-f", str(Path(freq_path).resolve())])

        subprocess.run(command, check=True, capture_output=True, text=True)
        return prefix.with_suffix(".ibd").read_text()


def run_merlin_multipoint_ibd(
    merlin_executable: str | Path,
    ped_path: str | Path,
    dat_path: str | Path,
    map_path: str | Path,
    freq_path: str | Path | None = None,
    position_arguments: tuple[str, ...] = (),
) -> str:
    """Run MERLIN multipoint IBD at requested analysis positions."""

    merlin_path = Path(merlin_executable).resolve()
    with TemporaryDirectory(prefix="pymerlin_merlin_") as temp_dir:
        prefix = Path(temp_dir) / "merlin_compare"
        command = [
            str(merlin_path),
            "-d",
            str(Path(dat_path).resolve()),
            "-p",
            str(Path(ped_path).resolve()),
            "-m",
            str(Path(map_path).resolve()),
            "--ibd",
            *position_arguments,
            f"--prefix:{prefix}",
            "--quiet",
        ]
        if freq_path is not None:
            command.extend(["-f", str(Path(freq_path).resolve())])

        subprocess.run(command, check=True, capture_output=True, text=True)
        return prefix.with_suffix(".ibd").read_text()


def run_merlin_npl_pairs_zscores(
    merlin_executable: str | Path,
    ped_path: str | Path,
    dat_path: str | Path,
    map_path: str | Path,
    freq_path: str | Path | None = None,
    position_arguments: tuple[str, ...] = (),
) -> str:
    """Run MERLIN affected-pairs NPL and return its raw `.zscore` text."""

    merlin_path = Path(merlin_executable).resolve()
    with TemporaryDirectory(prefix="pymerlin_merlin_") as temp_dir:
        prefix = Path(temp_dir) / "merlin_compare"
        command = [
            str(merlin_path),
            "-d",
            str(Path(dat_path).resolve()),
            "-p",
            str(Path(ped_path).resolve()),
            "-m",
            str(Path(map_path).resolve()),
            "--pairs",
            "--zscores",
            *position_arguments,
            f"--prefix:{prefix}",
            "--quiet",
        ]
        if freq_path is not None:
            command.extend(["-f", str(Path(freq_path).resolve())])

        subprocess.run(command, check=True, capture_output=True, text=True)
        return prefix.with_suffix(".zscore").read_text()


def run_merlin_linear_kong_cox_table(
    merlin_executable: str | Path,
    ped_path: str | Path,
    dat_path: str | Path,
    map_path: str | Path,
    freq_path: str | Path | None = None,
    position_arguments: tuple[str, ...] = (),
) -> str:
    """Run MERLIN's linear affected-pairs model and return its table."""

    merlin_path = Path(merlin_executable).resolve()
    with TemporaryDirectory(prefix="pymerlin_merlin_") as temp_dir:
        prefix = Path(temp_dir) / "merlin_compare"
        command = [
            str(merlin_path),
            "-d",
            str(Path(dat_path).resolve()),
            "-p",
            str(Path(ped_path).resolve()),
            "-m",
            str(Path(map_path).resolve()),
            "--pairs",
            "--tabulate",
            *position_arguments,
            f"--prefix:{prefix}",
            "--quiet",
        ]
        if freq_path is not None:
            command.extend(["-f", str(Path(freq_path).resolve())])

        subprocess.run(command, check=True, capture_output=True, text=True)
        table_path = Path(f"{prefix}-nonparametric.tbl")
        return table_path.read_text()


def run_merlin_exponential_kong_cox_table(
    merlin_executable: str | Path,
    ped_path: str | Path,
    dat_path: str | Path,
    map_path: str | Path,
    freq_path: str | Path | None = None,
    position_arguments: tuple[str, ...] = (),
) -> str:
    """Run MERLIN's exponential affected-pairs model and return its table."""

    merlin_path = Path(merlin_executable).resolve()
    with TemporaryDirectory(prefix="pymerlin_merlin_") as temp_dir:
        prefix = Path(temp_dir) / "merlin_compare"
        command = [
            str(merlin_path),
            "-d",
            str(Path(dat_path).resolve()),
            "-p",
            str(Path(ped_path).resolve()),
            "-m",
            str(Path(map_path).resolve()),
            "--pairs",
            "--exp",
            "--tabulate",
            *position_arguments,
            f"--prefix:{prefix}",
            "--quiet",
        ]
        if freq_path is not None:
            command.extend(["-f", str(Path(freq_path).resolve())])

        subprocess.run(command, check=True, capture_output=True, text=True)
        table_path = Path(f"{prefix}-nonparametric.tbl")
        return table_path.read_text()


def run_merlin_information_table(
    merlin_executable: str | Path,
    ped_path: str | Path,
    dat_path: str | Path,
    map_path: str | Path,
    freq_path: str | Path | None = None,
    position_arguments: tuple[str, ...] = (),
) -> str:
    """Run MERLIN's information analysis and return its table."""

    merlin_path = Path(merlin_executable).resolve()
    with TemporaryDirectory(prefix="pymerlin_merlin_") as temp_dir:
        prefix = Path(temp_dir) / "merlin_compare"
        command = [
            str(merlin_path),
            "-d",
            str(Path(dat_path).resolve()),
            "-p",
            str(Path(ped_path).resolve()),
            "-m",
            str(Path(map_path).resolve()),
            "--information",
            "--tabulate",
            *position_arguments,
            f"--prefix:{prefix}",
            "--quiet",
        ]
        if freq_path is not None:
            command.extend(["-f", str(Path(freq_path).resolve())])

        subprocess.run(command, check=True, capture_output=True, text=True)
        table_path = Path(f"{prefix}-info.tbl")
        return table_path.read_text()
