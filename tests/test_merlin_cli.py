import subprocess
import sys
from pathlib import Path

from pymerlin import load_merlin_inputs, multipoint_ibd
from pymerlin.benchmark import repeat_families
from pymerlin.merlin_cli import format_merlin_ibd


def test_merlin_compatible_cli_writes_exact_singlepoint_ibd(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "basic2_pymerlin"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-dexamples/basic2.dat",
            "-pexamples/basic2.ped",
            "-mexamples/basic2.map",
            "-fexamples/basic2.freq",
            "--ibd",
            "--singlepoint",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    output_path = tmp_path / "basic2_pymerlin.ibd"
    expected_output = Path(
        "tests/fixtures/merlin/basic2_freq_single.ibd"
    ).read_text()
    assert output_path.read_text() == expected_output


def test_merlin_compatible_cli_writes_exact_multipoint_ibd(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "basic2_multipoint_pymerlin"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-dexamples/basic2.dat",
            "-pexamples/basic2.ped",
            "-mexamples/basic2.map",
            "-fexamples/basic2.freq",
            "--ibd",
            "--cpus:2",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    output_path = tmp_path / "basic2_multipoint_pymerlin.ibd"
    expected_output = Path(
        "tests/fixtures/merlin/basic2_multipoint_freq.ibd"
    ).read_text()
    assert output_path.read_text() == expected_output


def test_merlin_compatible_cli_marker_names_match_merlin_labels(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "basic2_marker_names_pymerlin"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-dexamples/basic2.dat",
            "-pexamples/basic2.ped",
            "-mexamples/basic2.map",
            "-fexamples/basic2.freq",
            "--ibd",
            "--markerNames",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    output_path = tmp_path / "basic2_marker_names_pymerlin.ibd"
    expected_output = (
        Path("tests/fixtures/merlin/basic2_multipoint_freq.ibd")
        .read_text()
        .replace("123.400", "some_marker")
        .replace("136.200", "another_marker")
    )
    assert output_path.read_text() == expected_output


def test_merlin_compatible_cli_steps_add_intermarker_output(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "basic2_steps_pymerlin"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-dexamples/basic2.dat",
            "-pexamples/basic2.ped",
            "-mexamples/basic2.map",
            "-fexamples/basic2.freq",
            "--ibd",
            "--steps:1",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    output_path = tmp_path / "basic2_steps_pymerlin.ibd"
    labels = []
    for line in output_path.read_text().splitlines()[1:]:
        label = line.split()[3]
        if not labels or labels[-1] != label:
            labels.append(label)
    assert labels == ["123.400", "129.800", "136.200"]


def test_merlin_formatter_groups_all_positions_by_family() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    repeated_dataset = repeat_families(dataset, copies=2)
    results = multipoint_ibd(repeated_dataset)

    output = format_merlin_ibd(
        repeated_dataset,
        results,
        use_marker_names=False,
    )
    family_position_blocks: list[tuple[str, str]] = []
    for line in output.splitlines()[1:]:
        family_id, _, _, position, *_ = line.split()
        block = (family_id, position)
        if not family_position_blocks or family_position_blocks[-1] != block:
            family_position_blocks.append(block)

    assert family_position_blocks == [
        ("1_copy1", "123.400"),
        ("1_copy1", "136.200"),
        ("1_copy2", "123.400"),
        ("1_copy2", "136.200"),
    ]


def test_merlin_compatible_cli_multipoint_requires_map(tmp_path: Path) -> None:
    prefix = tmp_path / "basic2_pymerlin"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-dexamples/basic2.dat",
            "-pexamples/basic2.ped",
            "-fexamples/basic2.freq",
            "--ibd",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "requires a map file" in completed.stderr


def test_merlin_compatible_cli_default_frequencies_match_merlin_fixture(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "basic2_default_pymerlin"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-dexamples/basic2.dat",
            "-pexamples/basic2.ped",
            "-mexamples/basic2.map",
            "--ibd",
            "--singlepoint",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    output_path = tmp_path / "basic2_default_pymerlin.ibd"
    expected_output = Path(
        "tests/fixtures/merlin/basic2_default_single.ibd"
    ).read_text()
    assert output_path.read_text() == expected_output


def test_merlin_compatible_cli_rejects_ml_frequencies_cleanly(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "basic2_ml_pymerlin"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-dexamples/basic2.dat",
            "-pexamples/basic2.ped",
            "-mexamples/basic2.map",
            "-fm",
            "--ibd",
            "--singlepoint",
            f"--prefix:{prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Maximum-likelihood allele frequency estimation" in completed.stderr
    assert "Traceback" not in completed.stderr
