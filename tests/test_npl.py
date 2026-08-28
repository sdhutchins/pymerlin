import math
from pathlib import Path

import pytest

from pymerlin import (
    Dataset,
    load_merlin_inputs,
    multipoint_npl_pairs,
    multipoint_state_posteriors_at_positions,
    npl_pairs_score,
)
from pymerlin.likelihood import inheritance_origins
from pymerlin.positions import merlin_analysis_positions


def test_npl_pairs_score_counts_affected_sibling_allele_sharing() -> None:
    dataset = _load_basic2()
    family = dataset.families[0]
    affected_ids = ("5", "6")

    share_both_origins = inheritance_origins(family, (0, 0, 0, 0, 0, 0))
    share_no_origins = inheritance_origins(family, (0, 0, 0, 0, 1, 1))

    assert npl_pairs_score(share_both_origins, affected_ids) == 6.0
    assert npl_pairs_score(share_no_origins, affected_ids) == 4.0


def test_multipoint_state_posteriors_normalize_each_family() -> None:
    dataset = _load_basic2()
    positions = merlin_analysis_positions(dataset)

    position_posteriors = multipoint_state_posteriors_at_positions(
        dataset,
        positions,
    )

    assert tuple(result.label for result in position_posteriors) == (
        "123.400",
        "136.200",
    )
    for position_posterior in position_posteriors:
        assert tuple(
            family.family_id for family in position_posterior.families
        ) == ("1",)
        probability_total = math.fsum(
            state.probability
            for state in position_posterior.families[0].states
        )
        assert probability_total == pytest.approx(1.0)


def test_basic2_affected_sibling_null_distribution_matches_merlin() -> None:
    dataset = _load_basic2()
    positions = merlin_analysis_positions(dataset)

    result = multipoint_npl_pairs(dataset, positions)
    family_result = result.analyses[0].families[0]

    assert result.chromosome == "24"
    assert result.analyses[0].affection_name == "some_disease"
    assert family_result.null_mean == 5.0
    assert family_result.null_variance == 0.5
    assert family_result.z_min == pytest.approx(-math.sqrt(2.0))
    assert family_result.z_max == pytest.approx(math.sqrt(2.0))
    assert len(family_result.z_scores) == len(positions)
    assert family_result.standardized_score_values == pytest.approx(
        (-math.sqrt(2.0), 0.0, math.sqrt(2.0))
    )
    assert family_result.null_probabilities == pytest.approx(
        (0.25, 0.5, 0.25)
    )
    assert all(
        math.fsum(probabilities) == pytest.approx(1.0)
        for probabilities in family_result.posterior_probabilities
    )


def _load_basic2() -> Dataset:
    return load_merlin_inputs(
        Path("examples/basic2.ped"),
        Path("examples/basic2.dat"),
        Path("examples/basic2.map"),
        Path("examples/basic2.freq"),
    )
