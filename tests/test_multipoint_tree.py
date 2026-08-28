import math
from itertools import product

import pytest
from gmpy2 import context, get_context, mpfr

from pymerlin import (
    AnalysisPosition,
    load_merlin_inputs,
    multipoint_ibd,
    multipoint_ibd_at_positions,
)
from pymerlin.likelihood import _score_family_markers
from pymerlin.map import haldane_recombination_fraction, map_distance_cm
from pymerlin.multipoint import (
    _posterior_state_weights,
    _tree_marker_posteriors,
    _tree_pairwise_ibd_at_positions,
    _tree_pairwise_ibd_probabilities,
    _tree_posteriors_at_positions,
)
from tests.oracles import mpfr_multipoint as mpfr_oracle


def test_tree_marker_posteriors_match_every_dense_basic2_state() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    family = dataset.families[0]
    markers = tuple(
        sorted(
            dataset.markers,
            key=lambda marker: (float(marker.position_cm), marker.name),
        )
    )
    recombination_fractions = tuple(
        haldane_recombination_fraction(
            map_distance_cm(
                float(left_marker.position_cm),
                float(right_marker.position_cm),
            )
        )
        for left_marker, right_marker in zip(markers, markers[1:])
    )
    dense_states = _score_family_markers(family, markers)
    dense_posteriors = _posterior_state_weights(
        dense_states,
        recombination_fractions,
    )
    with context(get_context(), precision=256):
        oracle_recombination_fractions = tuple(
            mpfr_oracle._haldane_recombination_fraction(
                mpfr(str(right_marker.position_cm))
                - mpfr(str(left_marker.position_cm))
            )
            for left_marker, right_marker in zip(markers, markers[1:])
        )
        oracle_states = tuple(
            tuple(mpfr_oracle._score_family_marker(family, marker))
            for marker in markers
        )
        oracle_posteriors = mpfr_oracle._posterior_state_weights(
            oracle_states,
            oracle_recombination_fractions,
        )

    posterior_trees = _tree_marker_posteriors(
        family,
        markers,
        recombination_fractions,
    )

    state_count = 2 ** len(family.meioses)
    full_tree_node_count = 2 ** (len(family.meioses) + 1) - 1
    all_bits = tuple(product((0, 1), repeat=len(family.meioses)))
    failures: list[str] = []
    for (
        marker,
        states,
        dense_weights,
        oracle_marker_states,
        oracle_weights,
        posterior_tree,
    ) in zip(
        markers,
        dense_states,
        dense_posteriors,
        oracle_states,
        oracle_posteriors,
        posterior_trees,
    ):
        dense_by_bits = {
            state.bits: weight
            for state, weight in zip(states, dense_weights)
        }
        oracle_by_bits = {
            state.bits: weight
            for state, weight in zip(
                oracle_marker_states,
                oracle_weights,
            )
        }

        assert posterior_tree.weighted_sum() == pytest.approx(1.0)
        assert posterior_tree.node_count() <= full_tree_node_count
        for bits in all_bits:
            tree_probability = posterior_tree.value_at(bits) / state_count
            dense_probability = dense_by_bits.get(bits, 0.0)
            oracle_probability = oracle_by_bits.get(bits, mpfr(0))
            with context(get_context(), precision=256):
                tree_error = abs(mpfr(tree_probability) - oracle_probability)
                dense_error = abs(
                    mpfr(dense_probability) - oracle_probability
                )
                representation_unit = mpfr(
                    math.ulp(float(oracle_probability))
                )
            if tree_error > max(dense_error, representation_unit):
                failures.append(
                    f"Tree posterior is less accurate for {marker.name=}, "
                    f"{bits=}: {tree_error=} > {dense_error=}, "
                    f"{representation_unit=}"
                )

    assert any(
        tree.node_count() < full_tree_node_count for tree in posterior_trees
    )
    assert not failures, "\n".join(failures)


def test_tree_forward_backward_requires_one_fraction_per_interval() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    with pytest.raises(ValueError, match="One recombination fraction"):
        _tree_marker_posteriors(
            dataset.families[0],
            dataset.markers,
            (),
        )


def test_tree_pairwise_ibd_is_not_less_accurate_than_dense_basic2() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    family = dataset.families[0]
    markers = tuple(
        sorted(
            dataset.markers,
            key=lambda marker: (float(marker.position_cm), marker.name),
        )
    )
    recombination_fractions = tuple(
        haldane_recombination_fraction(
            map_distance_cm(
                float(left_marker.position_cm),
                float(right_marker.position_cm),
            )
        )
        for left_marker, right_marker in zip(markers, markers[1:])
    )
    tree_results = _tree_pairwise_ibd_probabilities(
        family,
        markers,
        recombination_fractions,
    )
    dense_results = multipoint_ibd(dataset)
    dense_by_key = {
        (
            result.marker_name,
            str(row["id1"]),
            str(row["id2"]),
        ): (
            float(row["z0"]),
            float(row["z1"]),
            float(row["z2"]),
        )
        for result in dense_results
        for row in result.rows
    }
    oracle_results = mpfr_oracle.mpfr_multipoint_ibd(dataset)
    failures: list[str] = []

    for marker, marker_results in zip(markers, tree_results):
        for pair, tree_probabilities in marker_results.items():
            dense_probabilities = dense_by_key[(marker.name, *pair)]
            oracle_probabilities = oracle_results[
                (marker.name, family.family_id, *pair)
            ]
            assert math.fsum(tree_probabilities) == pytest.approx(1.0)

            for ibd_state, (
                tree_probability,
                dense_probability,
                oracle_probability,
            ) in enumerate(
                zip(
                    tree_probabilities,
                    dense_probabilities,
                    oracle_probabilities,
                )
            ):
                with context(get_context(), precision=256):
                    tree_error = abs(
                        mpfr(tree_probability) - oracle_probability
                    )
                    dense_error = abs(
                        mpfr(dense_probability) - oracle_probability
                    )
                    representation_unit = mpfr(
                        math.ulp(float(oracle_probability))
                    )
                if tree_error > max(dense_error, representation_unit):
                    failures.append(
                        f"Tree IBD is less accurate for {marker.name=}, "
                        f"{pair=}, P{ibd_state}: {tree_error=} > "
                        f"{dense_error=}, {representation_unit=}"
                    )

    assert not failures, "\n".join(failures)


def test_tree_intermarker_ibd_is_not_less_accurate_than_dense_basic2() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    family = dataset.families[0]
    markers = tuple(
        sorted(
            dataset.markers,
            key=lambda marker: (float(marker.position_cm), marker.name),
        )
    )
    analysis_positions = (
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
    recombination_fractions = tuple(
        haldane_recombination_fraction(
            map_distance_cm(
                float(left_marker.position_cm),
                float(right_marker.position_cm),
            )
        )
        for left_marker, right_marker in zip(markers, markers[1:])
    )
    posterior_trees = _tree_posteriors_at_positions(
        family,
        markers,
        analysis_positions,
        recombination_fractions,
    )
    tree_results = _tree_pairwise_ibd_at_positions(
        family,
        markers,
        analysis_positions,
        recombination_fractions,
    )
    dense_results = multipoint_ibd_at_positions(
        dataset,
        analysis_positions,
    )
    dense_by_key = {
        (
            result.label,
            str(row["id1"]),
            str(row["id2"]),
        ): (
            float(row["z0"]),
            float(row["z1"]),
            float(row["z2"]),
        )
        for result in dense_results
        for row in result.rows
    }
    oracle_results = mpfr_oracle.mpfr_multipoint_ibd_at_positions(
        dataset,
        analysis_positions,
    )
    full_tree_node_count = 2 ** (len(family.meioses) + 1) - 1
    failures: list[str] = []

    assert len(posterior_trees) == len(analysis_positions)
    assert all(
        tree.weighted_sum() == pytest.approx(1.0)
        for tree in posterior_trees
    )
    assert any(
        tree.node_count() < full_tree_node_count for tree in posterior_trees
    )

    for analysis_position, position_results in zip(
        analysis_positions,
        tree_results,
    ):
        for pair, tree_probabilities in position_results.items():
            dense_probabilities = dense_by_key[
                (analysis_position.label, *pair)
            ]
            oracle_probabilities = oracle_results[
                (
                    analysis_position.label,
                    family.family_id,
                    *pair,
                )
            ]
            assert math.fsum(tree_probabilities) == pytest.approx(1.0)

            for ibd_state, (
                tree_probability,
                dense_probability,
                oracle_probability,
            ) in enumerate(
                zip(
                    tree_probabilities,
                    dense_probabilities,
                    oracle_probabilities,
                )
            ):
                with context(get_context(), precision=256):
                    tree_error = abs(
                        mpfr(tree_probability) - oracle_probability
                    )
                    dense_error = abs(
                        mpfr(dense_probability) - oracle_probability
                    )
                    representation_unit = mpfr(
                        math.ulp(float(oracle_probability))
                    )
                if tree_error > max(dense_error, representation_unit):
                    failures.append(
                        "Tree position IBD is less accurate for "
                        f"{analysis_position.label=}, {pair=}, "
                        f"P{ibd_state}: {tree_error=} > {dense_error=}, "
                        f"{representation_unit=}"
                    )

    assert not failures, "\n".join(failures)
