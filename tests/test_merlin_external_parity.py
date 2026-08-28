import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from pymerlin import compare_singlepoint_ibd_to_merlin_executable
from pymerlin.merlin_runner import (
    run_merlin_combined_multipoint_outputs,
    run_merlin_error_file,
    run_merlin_exponential_kong_cox_table,
    run_merlin_information_table,
    run_merlin_linear_kong_cox_table,
    run_merlin_multipoint_ibd,
    run_merlin_npl_pairs_zscores,
)
from tests.pah_scale_fixture import build_pah_parity_inputs

MERLIN_BIN = os.environ.get("PYMERLIN_MERLIN_BIN")


pytestmark = pytest.mark.skipif(
    not MERLIN_BIN,
    reason="Set PYMERLIN_MERLIN_BIN to run external MERLIN parity tests.",
)


@pytest.mark.parametrize(
    ("ped_path", "dat_path", "map_path", "freq_path"),
    [
        (
            "examples/basic2.ped",
            "examples/basic2.dat",
            "examples/basic2.map",
            "examples/basic2.freq",
        ),
        (
            "examples/error.ped",
            "examples/error.dat",
            "examples/error.map",
            None,
        ),
    ],
    ids=["no-errors", "unlikely-genotypes"],
)
def test_error_file_matches_external_merlin(
    tmp_path: Path,
    ped_path: str,
    dat_path: str,
    map_path: str,
    freq_path: str | None,
) -> None:
    """Require byte-identical genotype error output on example datasets."""

    assert MERLIN_BIN is not None
    merlin_output = run_merlin_error_file(
        MERLIN_BIN,
        ped_path,
        dat_path,
        map_path,
        freq_path,
    )
    prefix = tmp_path / "pymerlin_error_compare"
    command = [
        sys.executable,
        "-m",
        "pymerlin.cli",
        "-d",
        dat_path,
        "-p",
        ped_path,
        "-m",
        map_path,
        "--error",
        "--cpus",
        "2",
        "--quiet",
        f"--prefix:{prefix}",
    ]
    if freq_path is not None:
        command.extend(["-f", freq_path])

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert prefix.with_suffix(".err").read_text() == merlin_output


@pytest.mark.parametrize(
    ("ped_path", "dat_path", "map_path", "freq_path"),
    [
        ("examples/basic2.ped", "examples/basic2.dat", "examples/basic2.map", None),
        (
            "examples/basic2.ped",
            "examples/basic2.dat",
            "examples/basic2.map",
            "examples/basic2.freq",
        ),
        ("examples/haplo.ped", "examples/haplo.dat", "examples/haplo.map", None),
        ("examples/sibs.ped", "examples/sibs.dat", "examples/sibs.map", None),
    ],
)
def test_singlepoint_ibd_matches_external_merlin(
    ped_path: str,
    dat_path: str,
    map_path: str,
    freq_path: str | None,
) -> None:
    assert MERLIN_BIN is not None
    mismatches = compare_singlepoint_ibd_to_merlin_executable(
        MERLIN_BIN,
        ped_path,
        dat_path,
        map_path,
        freq_path,
    )

    assert mismatches == ()


@pytest.mark.parametrize(
    ("ped_path", "dat_path", "map_path", "freq_path", "position_arguments"),
    [
        (
            "examples/basic2.ped",
            "examples/basic2.dat",
            "examples/basic2.map",
            None,
            (),
        ),
        (
            "examples/basic2.ped",
            "examples/basic2.dat",
            "examples/basic2.map",
            "examples/basic2.freq",
            (),
        ),
        (
            "examples/basic2.ped",
            "examples/basic2.dat",
            "examples/basic2.map",
            "examples/basic2.freq",
            ("--steps", "1"),
        ),
        (
            "examples/basic2.ped",
            "examples/basic2.dat",
            "examples/basic2.map",
            "examples/basic2.freq",
            ("--steps", "2"),
        ),
        (
            "examples/basic2.ped",
            "examples/basic2.dat",
            "examples/basic2.map",
            "examples/basic2.freq",
            ("--maxStep", "5"),
        ),
        (
            "examples/basic2.ped",
            "examples/basic2.dat",
            "examples/basic2.map",
            "examples/basic2.freq",
            ("--steps", "4", "--minStep", "5"),
        ),
        (
            "examples/basic2.ped",
            "examples/basic2.dat",
            "examples/basic2.map",
            "examples/basic2.freq",
            ("--grid", "4", "--start", "124", "--stop", "136"),
        ),
        (
            "examples/basic2.ped",
            "examples/basic2.dat",
            "examples/basic2.map",
            "examples/basic2.freq",
            ("--positions", "some_marker,125.0,another_marker"),
        ),
        (
            "examples/basic2.ped",
            "examples/basic2.dat",
            "examples/basic2.map",
            "examples/basic2.freq",
            ("--steps", "1", "--markerNames"),
        ),
        (
            "examples/sibs.ped",
            "examples/sibs.dat",
            "examples/sibs.map",
            None,
            (),
        ),
    ],
    ids=[
        "basic2-estimated-marker-positions",
        "basic2-explicit-marker-positions",
        "basic2-one-step",
        "basic2-two-steps",
        "basic2-max-step",
        "basic2-min-step",
        "basic2-grid-bounds",
        "basic2-explicit-positions",
        "basic2-marker-names",
        "sibs-estimated-marker-positions",
    ],
)
def test_multipoint_ibd_matches_external_merlin(
    tmp_path: Path,
    ped_path: str,
    dat_path: str,
    map_path: str,
    freq_path: str | None,
    position_arguments: tuple[str, ...],
) -> None:
    """Require byte-identical multipoint IBD output across position modes."""

    assert MERLIN_BIN is not None
    merlin_output = run_merlin_multipoint_ibd(
        MERLIN_BIN,
        ped_path,
        dat_path,
        map_path,
        freq_path,
        position_arguments=position_arguments,
    )
    prefix = tmp_path / "pymerlin_compare"
    command = [
        sys.executable,
        "-m",
        "pymerlin.cli",
        "-d",
        dat_path,
        "-p",
        ped_path,
        "-m",
        map_path,
        "--ibd",
        *position_arguments,
        "--quiet",
        f"--prefix:{prefix}",
    ]
    if freq_path is not None:
        command.extend(["-f", freq_path])

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert prefix.with_suffix(".ibd").read_text() == _canonicalize_merlin_signed_zero(
        merlin_output
    )


def test_affected_pairs_zscores_match_external_merlin(tmp_path: Path) -> None:
    """Require exact raw family NPL-pairs output for two affected siblings."""

    assert MERLIN_BIN is not None
    merlin_output = run_merlin_npl_pairs_zscores(
        MERLIN_BIN,
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    prefix = tmp_path / "pymerlin_pairs_compare"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-d",
            "examples/basic2.dat",
            "-p",
            "examples/basic2.ped",
            "-m",
            "examples/basic2.map",
            "-f",
            "examples/basic2.freq",
            "--pairs",
            "--zscores",
            "--cpus",
            "2",
            "--quiet",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert prefix.with_suffix(
        ".zscore"
    ).read_text() == _canonicalize_merlin_signed_zero(merlin_output)


def test_linear_kong_cox_table_matches_external_merlin(tmp_path: Path) -> None:
    """Require exact aggregate linear affected-pairs output."""

    assert MERLIN_BIN is not None
    merlin_output = run_merlin_linear_kong_cox_table(
        MERLIN_BIN,
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    prefix = tmp_path / "pymerlin_linear_compare"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-d",
            "examples/basic2.dat",
            "-p",
            "examples/basic2.ped",
            "-m",
            "examples/basic2.map",
            "-f",
            "examples/basic2.freq",
            "--pairs",
            "--tabulate",
            "--quiet",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(
        f"{prefix}-nonparametric.tbl"
    ).read_text() == _canonicalize_merlin_signed_zero(merlin_output)


def test_exponential_kong_cox_table_matches_external_merlin(
    tmp_path: Path,
) -> None:
    """Require exact aggregate exponential affected-pairs output."""

    assert MERLIN_BIN is not None
    merlin_output = run_merlin_exponential_kong_cox_table(
        MERLIN_BIN,
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    prefix = tmp_path / "pymerlin_exponential_compare"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-d",
            "examples/basic2.dat",
            "-p",
            "examples/basic2.ped",
            "-m",
            "examples/basic2.map",
            "-f",
            "examples/basic2.freq",
            "--pairs",
            "--exp",
            "--tabulate",
            "--quiet",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(
        f"{prefix}-nonparametric.tbl"
    ).read_text() == _canonicalize_merlin_signed_zero(merlin_output)


@pytest.mark.parametrize(
    ("ped_path", "dat_path", "map_path", "freq_path"),
    [
        (
            "examples/basic2.ped",
            "examples/basic2.dat",
            "examples/basic2.map",
            "examples/basic2.freq",
        ),
        (
            "examples/sibs.ped",
            "examples/sibs.dat",
            "examples/sibs.map",
            None,
        ),
    ],
    ids=["basic2", "sibs"],
)
def test_information_table_matches_external_merlin(
    tmp_path: Path,
    ped_path: str,
    dat_path: str,
    map_path: str,
    freq_path: str | None,
) -> None:
    """Require exact information output for supported example datasets."""

    assert MERLIN_BIN is not None
    merlin_output = run_merlin_information_table(
        MERLIN_BIN,
        ped_path,
        dat_path,
        map_path,
        freq_path,
    )
    prefix = tmp_path / "pymerlin_information_compare"
    command = [
        sys.executable,
        "-m",
        "pymerlin.cli",
        "-d",
        dat_path,
        "-p",
        ped_path,
        "-m",
        map_path,
        "--information",
        "--tabulate",
        "--quiet",
        f"--prefix:{prefix}",
    ]
    if freq_path is not None:
        command.extend(["-f", freq_path])

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(f"{prefix}-info.tbl").read_text() == merlin_output


def test_tree_engine_combined_outputs_match_external_merlin(
    tmp_path: Path,
) -> None:
    """Require direct MERLIN parity for the reusable tree CLI path."""

    assert MERLIN_BIN is not None
    ped_path = "examples/basic2.ped"
    dat_path = "examples/basic2.dat"
    map_path = "examples/basic2.map"
    freq_path = "examples/basic2.freq"
    merlin_ibd = run_merlin_multipoint_ibd(
        MERLIN_BIN,
        ped_path,
        dat_path,
        map_path,
        freq_path,
    )
    merlin_zscores = run_merlin_npl_pairs_zscores(
        MERLIN_BIN,
        ped_path,
        dat_path,
        map_path,
        freq_path,
    )
    merlin_kong_cox = run_merlin_exponential_kong_cox_table(
        MERLIN_BIN,
        ped_path,
        dat_path,
        map_path,
        freq_path,
    )
    merlin_information = run_merlin_information_table(
        MERLIN_BIN,
        ped_path,
        dat_path,
        map_path,
        freq_path,
    )
    prefix = tmp_path / "pymerlin_tree_combined_compare"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-d",
            dat_path,
            "-p",
            ped_path,
            "-m",
            map_path,
            "-f",
            freq_path,
            "--ibd",
            "--pairs",
            "--zscores",
            "--exp",
            "--information",
            "--tabulate",
            "--engine",
            "tree",
            "--cpus",
            "2",
            "--quiet",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert prefix.with_suffix(".ibd").read_text() == (
        _canonicalize_merlin_signed_zero(merlin_ibd)
    )
    assert prefix.with_suffix(".zscore").read_text() == (
        _canonicalize_merlin_signed_zero(merlin_zscores)
    )
    assert Path(f"{prefix}-nonparametric.tbl").read_text() == (
        _canonicalize_merlin_signed_zero(merlin_kong_cox)
    )
    assert Path(f"{prefix}-info.tbl").read_text() == merlin_information


@pytest.mark.pah_scale
def test_pah_derived_tree_matches_external_merlin(tmp_path: Path) -> None:
    """Require exact parity on a bounded branch of the PAH pedigree."""

    assert MERLIN_BIN is not None
    input_paths = build_pah_parity_inputs(tmp_path / "inputs")
    merlin_outputs = run_merlin_combined_multipoint_outputs(
        MERLIN_BIN,
        input_paths.ped,
        input_paths.dat,
        input_paths.map,
        input_paths.freq,
    )
    prefix = tmp_path / "pymerlin_pah_derived_compare"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-d",
            str(input_paths.dat),
            "-p",
            str(input_paths.ped),
            "-m",
            str(input_paths.map),
            "-f",
            str(input_paths.freq),
            "--pairs",
            "--zscores",
            "--exp",
            "--information",
            "--tabulate",
            "--engine:tree",
            "--cpus:4",
            "--quiet",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert prefix.with_suffix(".zscore").read_text() == (
        _canonicalize_merlin_signed_zero(merlin_outputs.zscores)
    )
    assert Path(f"{prefix}-nonparametric.tbl").read_text() == (
        _canonicalize_merlin_signed_zero(merlin_outputs.kong_cox_table)
    )
    assert Path(f"{prefix}-info.tbl").read_text() == (merlin_outputs.information_table)


def _canonicalize_merlin_signed_zero(output: str) -> str:
    """Replace MERLIN's rounded negative zero with exact display zero."""

    return re.sub(
        r"(?<!\S)-0\.(0{3}|0{5}|0{6})(?=\s|$)",
        lambda match: match.group(0).removeprefix("-"),
        output,
    )
