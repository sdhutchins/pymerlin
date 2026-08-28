from pymerlin import compare_singlepoint_ibd_to_merlin, load_merlin_inputs


def test_basic2_singlepoint_ibd_matches_merlin_fixture():
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    mismatches = compare_singlepoint_ibd_to_merlin(
        dataset,
        "tests/fixtures/merlin/basic2_freq_single.ibd",
    )

    assert mismatches == ()


def test_basic2_default_frequency_ibd_matches_merlin_fixture():
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
    )

    mismatches = compare_singlepoint_ibd_to_merlin(
        dataset,
        "tests/fixtures/merlin/basic2_default_single.ibd",
    )

    assert mismatches == ()
