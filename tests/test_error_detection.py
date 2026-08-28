from pymerlin import (
    GenotypeError,
    detect_unlikely_genotypes,
    format_merlin_error_file,
    load_merlin_inputs,
)


def test_merlin_error_file_format_is_pedwipe_compatible() -> None:
    errors = (
        GenotypeError(
            family_id="2",
            person_id="3",
            marker_name="MRK11",
            likelihood_ratio=0.00385,
        ),
    )

    output = format_merlin_error_file(errors)

    assert output == (
        "    FAMILY     PERSON     MARKER      RATIO\n"
        "         2          3      MRK11    0.00385\n"
    )


def test_empty_merlin_error_file_contains_the_required_header() -> None:
    assert format_merlin_error_file(()) == (
        "    FAMILY     PERSON     MARKER      RATIO\n"
    )


def test_parallel_error_detection_is_identical_to_serial() -> None:
    dataset = load_merlin_inputs(
        "examples/error.ped",
        "examples/error.dat",
        "examples/error.map",
    )

    serial_errors = detect_unlikely_genotypes(dataset, workers=1)
    parallel_errors = detect_unlikely_genotypes(dataset, workers=2)

    assert parallel_errors == serial_errors
