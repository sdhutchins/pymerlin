"""Regression tests for Cheaha Slurm script path handling."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
SLURM_SCRIPT_NAMES = (
    "cheaha_setup_pymerlin.sh",
    "cheaha_pah_paired_dag_100m.sh",
    "cheaha_pah_ordering.sh",
)


@pytest.mark.parametrize("script_name", SLURM_SCRIPT_NAMES)
def test_slurm_scripts_reject_spool_directory_as_repository(
    script_name: str,
    tmp_path: Path,
) -> None:
    """A spooled script path must not become the repository root."""

    environment = os.environ.copy()
    environment.update(
        {
            "SLURM_JOB_ID": "12345",
            "SLURM_SUBMIT_DIR": str(tmp_path),
        }
    )
    completed = subprocess.run(
        ["bash", str(REPOSITORY_ROOT / "benchmarks" / script_name)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Submit this job from the PyMerlin repository root" in completed.stderr
    assert str(tmp_path) in completed.stderr
    assert "EnvironmentFileNotFound" not in completed.stderr
