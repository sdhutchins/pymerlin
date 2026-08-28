"""Tests for MERLIN-compatible analysis position planning."""

from pymerlin import load_merlin_inputs, merlin_analysis_positions


def test_steps_add_equally_spaced_intermarker_positions() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    positions = merlin_analysis_positions(
        dataset,
        steps_per_interval=1,
    )

    assert [position.position_cm for position in positions] == [
        123.4,
        129.8,
        136.2,
    ]
    assert [position.label for position in positions] == [
        "123.400",
        "129.800",
        "136.200",
    ]


def test_marker_names_apply_only_at_marker_positions() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    positions = merlin_analysis_positions(
        dataset,
        steps_per_interval=1,
        use_marker_names=True,
    )

    assert [position.label for position in positions] == [
        "some_marker",
        "129.800",
        "another_marker",
    ]


def test_explicit_positions_override_grid_and_steps() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    positions = merlin_analysis_positions(
        dataset,
        steps_per_interval=4,
        grid_cm=1.0,
        position_list="another_marker,125.0,some_marker",
    )

    assert [position.label for position in positions] == [
        "some_marker",
        "125.0",
        "another_marker",
    ]


def test_grid_respects_start_and_stop_positions() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    positions = merlin_analysis_positions(
        dataset,
        grid_cm=1.0,
        start_cm=124.0,
        stop_cm=126.0,
    )

    assert [position.label for position in positions] == [
        "124.000",
        "125.000",
        "126.000",
    ]


def test_max_step_adds_enough_intermarker_positions() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    positions = merlin_analysis_positions(
        dataset,
        max_step_cm=5.0,
    )

    assert [position.label for position in positions] == [
        "123.400",
        "127.667",
        "131.933",
        "136.200",
    ]


def test_min_step_removes_positions_using_merlin_112_behavior() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    positions = merlin_analysis_positions(
        dataset,
        steps_per_interval=4,
        min_step_cm=5.0,
    )

    assert [position.label for position in positions] == [
        "123.400",
        "136.200",
    ]


def test_start_and_stop_filter_default_analysis_positions() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    positions = merlin_analysis_positions(
        dataset,
        steps_per_interval=3,
        start_cm=127.0,
        stop_cm=134.0,
    )

    assert [position.label for position in positions] == [
        "129.800",
        "133.000",
    ]
