import subprocess
import sys
from pathlib import Path

import pytest

from pymerlin import (
    AnalysisPosition,
    Dataset,
    NplPairsResult,
    exponential_kong_cox,
    format_merlin_information_table,
    format_merlin_kong_cox_table,
    linear_kong_cox,
    load_merlin_inputs,
    multipoint_ibd,
    multipoint_ibd_at_positions,
    multipoint_information_content,
    multipoint_npl_pairs,
    multipoint_state_posteriors_at_positions,
    multipoint_tree_posteriors_at_positions,
    two_marker_multipoint_ibd,
)
from pymerlin.benchmark import repeat_families
from pymerlin.merlin_cli import format_merlin_ibd


def test_public_tree_ibd_preserves_dense_display_and_worker_determinism() -> None:
    dataset = repeat_families(_load_basic2(), 2)
    analysis_positions = _analysis_positions()

    dense_results = multipoint_ibd_at_positions(
        dataset,
        analysis_positions,
        engine="dense",
    )
    serial_tree_results = multipoint_ibd_at_positions(
        dataset,
        analysis_positions,
        workers=1,
        engine="tree",
    )
    parallel_tree_results = multipoint_ibd_at_positions(
        dataset,
        analysis_positions,
        workers=2,
        engine="tree",
    )

    assert parallel_tree_results == serial_tree_results
    assert format_merlin_ibd(
        dataset,
        serial_tree_results,
        use_marker_names=False,
    ) == format_merlin_ibd(
        dataset,
        dense_results,
        use_marker_names=False,
    )


def test_marker_ibd_apis_route_the_tree_engine() -> None:
    dataset = _load_basic2()

    dense_results = multipoint_ibd(dataset, engine="dense")
    tree_results = multipoint_ibd(dataset, engine="tree")
    tree_two_marker_results = two_marker_multipoint_ibd(
        dataset,
        "some_marker",
        "another_marker",
        engine="tree",
    )

    assert tree_two_marker_results == tree_results
    assert format_merlin_ibd(
        dataset,
        tree_results,
        use_marker_names=True,
    ) == format_merlin_ibd(
        dataset,
        dense_results,
        use_marker_names=True,
    )


def test_public_tree_npl_preserves_kong_cox_and_worker_determinism() -> None:
    dataset = repeat_families(_load_basic2(), 2)
    analysis_positions = _analysis_positions()

    dense_result = multipoint_npl_pairs(
        dataset,
        analysis_positions,
        engine="dense",
    )
    serial_tree_result = multipoint_npl_pairs(
        dataset,
        analysis_positions,
        workers=1,
        engine="tree",
    )
    parallel_tree_result = multipoint_npl_pairs(
        dataset,
        analysis_positions,
        workers=2,
        engine="tree",
    )

    assert parallel_tree_result == serial_tree_result
    assert _format_kong_cox(serial_tree_result) == _format_kong_cox(
        dense_result
    )


def test_public_tree_information_preserves_display_and_workers() -> None:
    dataset = repeat_families(_load_basic2(), 2)
    analysis_positions = _analysis_positions()

    dense_result = multipoint_information_content(
        dataset,
        analysis_positions,
        engine="dense",
    )
    serial_tree_result = multipoint_information_content(
        dataset,
        analysis_positions,
        workers=1,
        engine="tree",
    )
    parallel_tree_result = multipoint_information_content(
        dataset,
        analysis_positions,
        workers=2,
        engine="tree",
    )

    assert parallel_tree_result == serial_tree_result
    assert format_merlin_information_table(
        (serial_tree_result,)
    ) == format_merlin_information_table((dense_result,))


def test_reusable_tree_posteriors_feed_all_multipoint_analyses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _load_basic2()
    analysis_positions = _analysis_positions()
    expected_ibd = multipoint_ibd_at_positions(
        dataset,
        analysis_positions,
        engine="tree",
    )
    expected_npl = multipoint_npl_pairs(
        dataset,
        analysis_positions,
        engine="tree",
    )
    expected_information = multipoint_information_content(
        dataset,
        analysis_positions,
        engine="tree",
    )
    tree_posteriors = multipoint_tree_posteriors_at_positions(
        dataset,
        analysis_positions,
    )

    def fail_if_recomputed(*args: object, **kwargs: object) -> None:
        raise AssertionError("Compressed posteriors were recomputed.")

    monkeypatch.setattr(
        "pymerlin.multipoint._tree_posteriors_at_positions",
        fail_if_recomputed,
    )

    cached_ibd = multipoint_ibd_at_positions(
        dataset,
        analysis_positions,
        engine="tree",
        tree_posteriors=tree_posteriors,
    )
    cached_npl = multipoint_npl_pairs(
        dataset,
        analysis_positions,
        engine="tree",
        tree_posteriors=tree_posteriors,
    )
    cached_information = multipoint_information_content(
        dataset,
        analysis_positions,
        engine="tree",
        tree_posteriors=tree_posteriors,
    )

    assert cached_ibd == expected_ibd
    assert cached_npl == expected_npl
    assert cached_information == expected_information


def test_single_family_marker_workers_preserve_tree_posteriors() -> None:
    """Require ordered marker parallelism to be bitwise deterministic."""

    dataset = _load_basic2()
    analysis_positions = _analysis_positions()

    serial_posteriors = multipoint_tree_posteriors_at_positions(
        dataset,
        analysis_positions,
        workers=1,
    )
    parallel_posteriors = multipoint_tree_posteriors_at_positions(
        dataset,
        analysis_positions,
        workers=2,
    )

    assert parallel_posteriors == serial_posteriors


def test_reusable_tree_posteriors_validate_the_position_grid() -> None:
    dataset = _load_basic2()
    analysis_positions = _analysis_positions()
    tree_posteriors = multipoint_tree_posteriors_at_positions(
        dataset,
        analysis_positions,
    )
    changed_positions = analysis_positions[:-1]

    with pytest.raises(ValueError, match="analysis-position grid"):
        multipoint_information_content(
            dataset,
            changed_positions,
            engine="tree",
            tree_posteriors=tree_posteriors,
        )


def test_tree_cli_preserves_combined_dense_outputs(tmp_path: Path) -> None:
    dense_prefix = tmp_path / "dense_combined"
    tree_prefix = tmp_path / "tree_combined"
    common_arguments = [
        sys.executable,
        "-m",
        "pymerlin.cli",
        "-dexamples/basic2.dat",
        "-pexamples/basic2.ped",
        "-mexamples/basic2.map",
        "-fexamples/basic2.freq",
        "--ibd",
        "--pairs",
        "--zscores",
        "--exp",
        "--information",
        "--tabulate",
        "--steps:1",
        "--quiet",
    ]
    dense_completed = subprocess.run(
        [*common_arguments, f"--prefix:{dense_prefix}"],
        check=False,
        capture_output=True,
        text=True,
    )
    tree_completed = subprocess.run(
        [
            *common_arguments,
            "--engine:tree",
            "--cpus:2",
            f"--prefix:{tree_prefix}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert dense_completed.returncode == 0, dense_completed.stderr
    assert tree_completed.returncode == 0, tree_completed.stderr
    for suffix in (
        ".ibd",
        ".zscore",
        "-nonparametric.tbl",
        "-info.tbl",
    ):
        assert Path(f"{tree_prefix}{suffix}").read_text() == Path(
            f"{dense_prefix}{suffix}"
        ).read_text()


def test_tree_cli_rejects_singlepoint_mode() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pymerlin.cli",
            "-dexamples/basic2.dat",
            "-pexamples/basic2.ped",
            "--ibd",
            "--singlepoint",
            "--engine:tree",
            "--quiet",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--engine tree requires multipoint mode" in completed.stderr


@pytest.mark.parametrize("engine", ["unknown", "numpy", "gpu"])
def test_public_multipoint_apis_reject_unknown_engines(engine: str) -> None:
    dataset = _load_basic2()
    analysis_positions = _analysis_positions()

    with pytest.raises(ValueError, match="Unknown multipoint engine"):
        multipoint_ibd_at_positions(
            dataset,
            analysis_positions,
            engine=engine,
        )
    with pytest.raises(ValueError, match="Unknown multipoint engine"):
        multipoint_npl_pairs(
            dataset,
            analysis_positions,
            engine=engine,
        )
    with pytest.raises(ValueError, match="Unknown multipoint engine"):
        multipoint_information_content(
            dataset,
            analysis_positions,
            engine=engine,
        )


def test_tree_analyses_reject_dense_position_posteriors() -> None:
    dataset = _load_basic2()
    analysis_positions = _analysis_positions()
    dense_posteriors = multipoint_state_posteriors_at_positions(
        dataset,
        analysis_positions,
    )

    with pytest.raises(ValueError, match="must be omitted"):
        multipoint_npl_pairs(
            dataset,
            analysis_positions,
            position_posteriors=dense_posteriors,
            engine="tree",
        )
    with pytest.raises(ValueError, match="must be omitted"):
        multipoint_information_content(
            dataset,
            analysis_positions,
            position_posteriors=dense_posteriors,
            engine="tree",
        )


def _format_kong_cox(npl_result: NplPairsResult) -> str:
    return format_merlin_kong_cox_table(
        (linear_kong_cox(npl_result),),
        (exponential_kong_cox(npl_result),),
    )


def _analysis_positions() -> tuple[AnalysisPosition, ...]:
    return (
        AnalysisPosition(
            position_cm=123.4,
            label="some_marker",
            marker_name="some_marker",
        ),
        AnalysisPosition(position_cm=129.8, label="129.800"),
        AnalysisPosition(
            position_cm=136.2,
            label="another_marker",
            marker_name="another_marker",
        ),
    )


def _load_basic2() -> Dataset:
    return load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
