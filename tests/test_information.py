import subprocess
import sys
from pathlib import Path

import pytest

from pymerlin import (
    Dataset,
    format_merlin_information_table,
    load_merlin_inputs,
    multipoint_information_content,
)
from pymerlin.positions import merlin_analysis_positions


def test_basic2_information_uses_merlin_effective_bit_count() -> None:
    dataset = _load_basic2()
    positions = merlin_analysis_positions(dataset)

    result = multipoint_information_content(dataset, positions)

    assert result.chromosome == "24"
    assert result.families[0].bit_count == 3
    assert len(result.values) == len(positions)
    assert all(0.0 <= value <= 1.0 for value in result.values)
    assert result.values == pytest.approx(result.families[0].values)


def test_merlin_compatible_cli_writes_information_table(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "basic2_information"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-dexamples/basic2.dat",
            "-pexamples/basic2.ped",
            "-mexamples/basic2.map",
            "-fexamples/basic2.freq",
            "--information",
            "--tabulate",
            "--quiet",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    dataset = _load_basic2()
    positions = merlin_analysis_positions(dataset)
    expected = format_merlin_information_table(
        (multipoint_information_content(dataset, positions),)
    )
    assert Path(f"{prefix}-info.tbl").read_text() == expected


def test_linkage_options_write_linkage_and_information_tables(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "basic2_linkage_options"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-dexamples/basic2.dat",
            "-pexamples/basic2.ped",
            "-mexamples/basic2.map",
            "-fexamples/basic2.freq",
            "--pairs",
            "--exp",
            "--information",
            "--tabulate",
            "--quiet",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(f"{prefix}-nonparametric.tbl").is_file()
    assert Path(f"{prefix}-info.tbl").is_file()


def _load_basic2() -> Dataset:
    return load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
