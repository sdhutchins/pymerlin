import math

from gmpy2 import context, get_context, log, mpfr

from pymerlin import (
    AnalysisPosition,
    Dataset,
    format_merlin_information_table,
    load_merlin_inputs,
    multipoint_information_content,
)
from pymerlin.information import _merlin_bit_count
from tests.oracles import mpfr_multipoint as mpfr_oracle


def test_tree_information_is_not_less_accurate_than_dense_basic2() -> None:
    dataset = _load_basic2()
    analysis_positions = _analysis_positions()
    dense_result = multipoint_information_content(
        dataset,
        analysis_positions,
        engine="dense",
    )
    tree_result = multipoint_information_content(
        dataset,
        analysis_positions,
        engine="tree",
    )
    oracle_values = _oracle_information_at_positions(
        dataset,
        analysis_positions,
    )
    failures: list[str] = []

    assert tree_result.chromosome == dense_result.chromosome
    assert tree_result.positions == dense_result.positions
    assert tuple(family.bit_count for family in tree_result.families) == (3,)

    for analysis_position, tree_value, dense_value, oracle_value in zip(
        analysis_positions,
        tree_result.values,
        dense_result.values,
        oracle_values,
    ):
        with context(get_context(), precision=256):
            tree_error = abs(mpfr(tree_value) - oracle_value)
            dense_error = abs(mpfr(dense_value) - oracle_value)
            representation_unit = mpfr(math.ulp(float(oracle_value)))
        if tree_error > max(dense_error, representation_unit):
            failures.append(
                "Tree information is less accurate for "
                f"{analysis_position.label=}: {tree_error=} > "
                f"{dense_error=}, {representation_unit=}"
            )

    assert not failures, "\n".join(failures)


def test_tree_information_preserves_dense_display_output() -> None:
    dataset = _load_basic2()
    analysis_positions = _analysis_positions()
    dense_result = multipoint_information_content(
        dataset,
        analysis_positions,
        engine="dense",
    )
    tree_result = multipoint_information_content(
        dataset,
        analysis_positions,
        engine="tree",
    )

    assert format_merlin_information_table(
        (tree_result,)
    ) == format_merlin_information_table((dense_result,))


def _oracle_information_at_positions(
    dataset: Dataset,
    analysis_positions: tuple[AnalysisPosition, ...],
) -> tuple[mpfr, ...]:
    family = dataset.families[0]
    bit_count = _merlin_bit_count(dataset, family)
    with context(get_context(), precision=256):
        markers = mpfr_oracle._ordered_markers(dataset, None)
        recombination_fractions = tuple(
            mpfr_oracle._haldane_recombination_fraction(
                mpfr(str(right_marker.position_cm))
                - mpfr(str(left_marker.position_cm))
            )
            for left_marker, right_marker in zip(markers, markers[1:])
        )
        states_by_marker = tuple(
            tuple(mpfr_oracle._score_family_marker(family, marker))
            for marker in markers
        )
        forward_weights, backward_weights = (
            mpfr_oracle._forward_backward_weights(
                states_by_marker,
                recombination_fractions,
            )
        )
        log_two = log(mpfr(2))
        results = []
        for analysis_position in analysis_positions:
            _, weights = mpfr_oracle._state_weights_at_position(
                family,
                markers,
                states_by_marker,
                forward_weights,
                backward_weights,
                mpfr(str(analysis_position.position_cm)),
            )
            full_posterior_entropy = -sum(
                (
                    probability * log(probability)
                    for probability in weights
                    if probability > 0
                ),
                mpfr(0),
            )
            hidden_entropy = (
                len(family.meioses) - bit_count
            ) * log_two
            posterior_entropy = max(
                full_posterior_entropy - hidden_entropy,
                mpfr(0),
            )
            prior_entropy = bit_count * log_two
            results.append(
                max(
                    mpfr(1) - posterior_entropy / prior_entropy,
                    mpfr(0),
                )
            )

    return tuple(results)


def _analysis_positions() -> tuple[AnalysisPosition, ...]:
    return (
        AnalysisPosition(position_cm=120.0, label="120.000"),
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
        AnalysisPosition(position_cm=140.0, label="140.000"),
    )


def _load_basic2() -> Dataset:
    return load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
