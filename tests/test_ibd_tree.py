from itertools import product

import pytest

from pymerlin import load_merlin_inputs
from pymerlin.ibd import _pair_ibd_indicator_trees, _shared_allele_count
from pymerlin.likelihood import inheritance_origins


def test_pair_ibd_indicator_trees_match_every_basic2_inheritance_state() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    family = dataset.families[0]
    people = tuple(person.individual_id for person in family.individuals)
    full_tree_node_count = 2 ** (len(family.meioses) + 1) - 1
    compressed_tree_found = False

    for index, first_id in enumerate(people):
        for second_id in people[index + 1 :]:
            indicator_trees = _pair_ibd_indicator_trees(
                family,
                first_id,
                second_id,
            )
            compressed_tree_found |= any(
                tree.node_count() < full_tree_node_count
                for tree in indicator_trees
            )

            for bits in product((0, 1), repeat=len(family.meioses)):
                origins = inheritance_origins(family, tuple(bits))
                expected_state = _shared_allele_count(
                    origins[first_id],
                    origins[second_id],
                )
                observed_indicators = tuple(
                    tree.value_at(bits) for tree in indicator_trees
                )

                assert observed_indicators[expected_state] == 1.0
                assert sum(observed_indicators) == 1.0

    assert compressed_tree_found


def test_pair_ibd_indicator_trees_reject_invalid_pairs() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    family = dataset.families[0]

    with pytest.raises(ValueError, match="distinct"):
        _pair_ibd_indicator_trees(family, "1", "1")
    with pytest.raises(ValueError, match="Unknown individual"):
        _pair_ibd_indicator_trees(family, "1", "missing")
