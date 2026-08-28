import math
import subprocess
import sys
from pathlib import Path

import pytest

from pymerlin import (
    LinearKongCoxResult,
    NplPairsResult,
    exponential_kong_cox,
    format_merlin_kong_cox_table,
    format_merlin_linear_kong_cox_table,
    linear_kong_cox,
    load_merlin_inputs,
    multipoint_npl_pairs,
)
from pymerlin.positions import merlin_analysis_positions


def test_linear_kong_cox_aggregates_informative_family_scores() -> None:
    npl_result, linear_result = _basic2_results()
    family_result = npl_result.analyses[0].families[0]
    analysis = linear_result.analyses[0]

    assert analysis.informative_family_count == 1
    assert tuple(row.z_score for row in analysis.rows) == pytest.approx(
        family_result.z_scores
    )
    assert analysis.minimum is not None
    assert analysis.maximum is not None
    assert analysis.minimum.delta == pytest.approx(-1.0 / math.sqrt(2.0))
    assert analysis.maximum.delta == pytest.approx(1.0 / math.sqrt(2.0))
    assert analysis.minimum.lod_score == pytest.approx(-math.log10(2.0))
    assert analysis.maximum.lod_score == pytest.approx(math.log10(2.0))


def test_linear_kong_cox_table_has_merlin_columns_and_position_labels() -> None:
    _, linear_result = _basic2_results()

    output = format_merlin_linear_kong_cox_table((linear_result,))
    lines = output.splitlines()

    assert lines[0] == (
        "CHR\tPOS\tLABEL\tANALYSIS\tZSCORE\tDELTA\tLOD\tPVALUE"
    )
    assert [line.split("\t")[2] for line in lines[1:]] == [
        "min",
        "max",
        "123.400",
        "136.200",
    ]
    assert all(
        line.split("\t")[3] == "some_disease [Pairs]"
        for line in lines[1:]
    )


def test_merlin_compatible_cli_writes_linear_kong_cox_table(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "basic2_linear_pairs"
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
            "--tabulate",
            "--quiet",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    _, linear_result = _basic2_results()
    assert Path(f"{prefix}-nonparametric.tbl").read_text() == (
        format_merlin_linear_kong_cox_table((linear_result,))
    )


def test_exponential_kong_cox_uses_full_family_score_distributions() -> None:
    npl_result, _ = _basic2_results()

    result = exponential_kong_cox(npl_result)
    analysis = result.analyses[0]

    assert analysis.informative_family_count == 1
    assert analysis.minimum is not None
    assert analysis.maximum is not None
    assert analysis.minimum.delta == -9.999
    assert analysis.maximum.delta == 9.999
    assert tuple(row.label for row in analysis.rows) == (
        "123.400",
        "136.200",
    )
    assert all(math.isfinite(row.lod_score) for row in analysis.rows)


def test_merlin_compatible_cli_writes_exponential_kong_cox_table(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "basic2_exponential_pairs"
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
            "--tabulate",
            "--quiet",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    npl_result, linear_result = _basic2_results()
    exponential_result = exponential_kong_cox(npl_result)
    expected = format_merlin_kong_cox_table(
        (linear_result,),
        (exponential_result,),
    )
    output = Path(f"{prefix}-nonparametric.tbl").read_text()

    assert output == expected
    assert output.splitlines()[0].endswith("ExDELTA\tExLOD\tPVALUE")
    assert all(len(line.split("\t")) == 11 for line in output.splitlines())


def _basic2_results() -> tuple[NplPairsResult, LinearKongCoxResult]:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    positions = merlin_analysis_positions(dataset)
    npl_result = multipoint_npl_pairs(dataset, positions)
    return npl_result, linear_kong_cox(npl_result)
