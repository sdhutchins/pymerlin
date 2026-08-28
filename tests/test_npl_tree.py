import math
from collections import defaultdict
from itertools import product

from gmpy2 import context, get_context, mpfr, sqrt

from pymerlin import (
    AnalysisPosition,
    Dataset,
    exponential_kong_cox,
    format_merlin_kong_cox_table,
    linear_kong_cox,
    load_merlin_inputs,
    multipoint_npl_pairs,
    npl_pairs_score,
)
from pymerlin.likelihood import inheritance_origins
from pymerlin.npl import (
    _npl_pairs_score_tree,
    _tree_multipoint_npl_pairs,
    _tree_npl_pairs_null_distribution,
)
from tests.oracles import mpfr_multipoint as mpfr_oracle


def test_npl_pairs_score_tree_matches_every_basic2_state() -> None:
    dataset = _load_basic2()
    family = dataset.families[0]
    affected_ids = ("5", "6")
    score_tree = _npl_pairs_score_tree(family, affected_ids)
    full_tree_node_count = 2 ** (len(family.meioses) + 1) - 1

    for bits in product((0, 1), repeat=len(family.meioses)):
        assert score_tree.value_at(bits) == npl_pairs_score(
            inheritance_origins(family, bits),
            affected_ids,
        )

    null_distribution = _tree_npl_pairs_null_distribution(
        family,
        affected_ids,
    )
    assert score_tree.node_count() < full_tree_node_count
    assert null_distribution.mean == 5.0
    assert null_distribution.variance == 0.5
    assert null_distribution.minimum == 4.0
    assert null_distribution.maximum == 6.0
    assert null_distribution.raw_score_values == (4.0, 5.0, 6.0)
    assert null_distribution.probabilities == (0.25, 0.5, 0.25)


def test_tree_npl_is_not_less_accurate_than_dense_basic2() -> None:
    dataset = _load_basic2()
    family = dataset.families[0]
    affected_ids = ("5", "6")
    analysis_positions = _analysis_positions()
    dense_result = multipoint_npl_pairs(dataset, analysis_positions)
    tree_result = _tree_multipoint_npl_pairs(dataset, analysis_positions)
    dense_family = dense_result.analyses[0].families[0]
    tree_family = tree_result.analyses[0].families[0]
    oracle_scores, oracle_probabilities = _oracle_npl_at_positions(
        dataset,
        analysis_positions,
        affected_ids,
    )
    failures: list[str] = []

    assert tree_result.chromosome == dense_result.chromosome
    assert tree_result.positions == dense_result.positions
    assert tree_family.null_mean == dense_family.null_mean
    assert tree_family.null_variance == dense_family.null_variance
    assert tree_family.standardized_score_values == (
        dense_family.standardized_score_values
    )
    assert tree_family.null_probabilities == dense_family.null_probabilities

    for position_index, analysis_position in enumerate(analysis_positions):
        tree_z_score = tree_family.z_scores[position_index]
        dense_z_score = dense_family.z_scores[position_index]
        oracle_z_score = oracle_scores[position_index]
        with context(get_context(), precision=256):
            tree_error = abs(mpfr(tree_z_score) - oracle_z_score)
            dense_error = abs(mpfr(dense_z_score) - oracle_z_score)
            representation_unit = mpfr(math.ulp(float(oracle_z_score)))
        if tree_error > max(dense_error, representation_unit):
            failures.append(
                f"Tree NPL is less accurate for {analysis_position.label=}: "
                f"{tree_error=} > {dense_error=}, {representation_unit=}"
            )

        for score_index, (
            tree_probability,
            dense_probability,
            oracle_probability,
        ) in enumerate(
            zip(
                tree_family.posterior_probabilities[position_index],
                dense_family.posterior_probabilities[position_index],
                oracle_probabilities[position_index],
            )
        ):
            with context(get_context(), precision=256):
                tree_error = abs(mpfr(tree_probability) - oracle_probability)
                dense_error = abs(mpfr(dense_probability) - oracle_probability)
                representation_unit = mpfr(
                    math.ulp(float(oracle_probability))
                )
            if tree_error > max(dense_error, representation_unit):
                failures.append(
                    "Tree NPL probability is less accurate for "
                    f"{analysis_position.label=}, {score_index=}: "
                    f"{tree_error=} > {dense_error=}, "
                    f"{representation_unit=}"
                )

    assert not failures, "\n".join(failures)


def test_tree_npl_preserves_kong_cox_display_output() -> None:
    dataset = _load_basic2()
    analysis_positions = _analysis_positions()
    dense_result = multipoint_npl_pairs(dataset, analysis_positions)
    tree_result = _tree_multipoint_npl_pairs(dataset, analysis_positions)

    dense_output = format_merlin_kong_cox_table(
        (linear_kong_cox(dense_result),),
        (exponential_kong_cox(dense_result),),
    )
    tree_output = format_merlin_kong_cox_table(
        (linear_kong_cox(tree_result),),
        (exponential_kong_cox(tree_result),),
    )

    assert tree_output == dense_output


def _oracle_npl_at_positions(
    dataset: Dataset,
    analysis_positions: tuple[AnalysisPosition, ...],
    affected_ids: tuple[str, ...],
) -> tuple[tuple[mpfr, ...], tuple[tuple[mpfr, ...], ...]]:
    family = dataset.families[0]
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
        z_scores = []
        probabilities_by_position = []
        for analysis_position in analysis_positions:
            states, weights = mpfr_oracle._state_weights_at_position(
                family,
                markers,
                states_by_marker,
                forward_weights,
                backward_weights,
                mpfr(str(analysis_position.position_cm)),
            )
            probabilities_by_score: defaultdict[float, mpfr] = defaultdict(
                lambda: mpfr(0)
            )
            expected_score = mpfr(0)
            for state, weight in zip(states, weights):
                raw_score = npl_pairs_score(
                    state.allele_origins,
                    affected_ids,
                )
                expected_score += weight * mpfr(raw_score)
                probabilities_by_score[raw_score] += weight
            z_scores.append(
                (expected_score - mpfr(5)) / sqrt(mpfr("0.5"))
            )
            probabilities_by_position.append(
                tuple(
                    probabilities_by_score[raw_score]
                    for raw_score in (4.0, 5.0, 6.0)
                )
            )

    return tuple(z_scores), tuple(probabilities_by_position)


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
