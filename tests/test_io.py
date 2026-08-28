from pathlib import Path

import pytest

from pymerlin import (
    Dataset,
    Family,
    Individual,
    Meiosis,
    estimate_ibd,
    load_merlin_inputs,
)
from pymerlin.io import _merlin_meiosis_person_order


def test_loads_basic_merlin_files():
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    assert [marker.name for marker in dataset.markers] == ["some_marker", "another_marker"]
    assert dataset.marker_by_name["some_marker"].chromosome == "24"
    assert dataset.marker_by_name["some_marker"].allele_frequencies == {
        "1": 0.1,
        "2": 0.2,
        "3": 0.3,
        "4": 0.4,
    }
    assert dataset.affection_names == ("some_disease",)
    assert len(dataset.families) == 1
    assert len(dataset.families[0].meioses) == 6


def test_estimates_default_frequencies_from_all_genotypes():
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
    )

    assert dataset.marker_by_name["another_marker"].allele_frequencies == {
        "2": 1.0,
    }


def test_slash_genotypes_are_parsed():
    dataset = load_merlin_inputs("examples/haplo.ped", "examples/haplo.dat", "examples/haplo.map")

    first_family = dataset.families[0]
    first_person = first_family.individuals[0]

    assert first_person.genotypes["SNP-1"] == ("2", "1")
    assert first_person.genotypes["SNP-3"] == ("2", "1")


def test_sorts_pedigree_identifiers_like_merlin(tmp_path: Path) -> None:
    dat_path = tmp_path / "ordering.dat"
    ped_path = tmp_path / "ordering.ped"
    dat_path.write_text("M marker\n")
    ped_path.write_text(
        """\
10 10 0 0 1 1 1
1 10 0 0 1 1 1
2 10 0 0 1 1 1
10 2 0 0 1 1 1
1 2 0 0 1 1 1
2 2 0 0 1 1 1
10 1 0 0 1 1 1
1 1 0 0 1 1 1
2 1 0 0 1 1 1
"""
    )

    dataset = load_merlin_inputs(ped_path, dat_path)

    assert [family.family_id for family in dataset.families] == [
        "1",
        "2",
        "10",
    ]
    for family in dataset.families:
        assert [person.individual_id for person in family.individuals] == [
            "1",
            "2",
            "10",
        ]


def test_dot_parent_ids_identify_founders(tmp_path: Path) -> None:
    dat_path = tmp_path / "dot_parents.dat"
    ped_path = tmp_path / "dot_parents.ped"
    dat_path.write_text("M marker\n")
    ped_path.write_text(
        """\
1 1 . . 1 1 1
1 2 . . 2 1 1
1 3 1 2 1 1 1
"""
    )

    dataset = load_merlin_inputs(ped_path, dat_path)

    family = dataset.families[0]
    assert [founder.individual_id for founder in family.founders] == ["1", "2"]
    assert family.by_id["1"].father_id is None
    assert family.by_id["1"].mother_id is None
    assert len(family.meioses) == 2


def test_meioses_follow_merlin_parent_before_child_traversal(
    tmp_path: Path,
) -> None:
    dat_path = tmp_path / "traversal.dat"
    ped_path = tmp_path / "traversal.ped"
    dat_path.write_text("A disease\nM marker\n")
    ped_path.write_text(
        """\
1 1 2 3 1 2 1 2
1 2 8 9 1 1 1 1
1 3 0 0 2 1 0 0
1 8 0 0 1 1 0 0
1 9 0 0 2 1 0 0
"""
    )

    dataset = load_merlin_inputs(ped_path, dat_path)

    family = dataset.families[0]
    assert [person.individual_id for person in family.individuals] == [
        "1",
        "2",
        "3",
        "8",
        "9",
    ]
    assert family.meioses == (
        Meiosis(parent_id="8", child_id="2", parent_sex="1"),
        Meiosis(parent_id="9", child_id="2", parent_sex="2"),
        Meiosis(parent_id="2", child_id="1", parent_sex="1"),
        Meiosis(parent_id="3", child_id="1", parent_sex="2"),
    )

    legacy_family = Family(
        family_id=family.family_id,
        individuals=family.individuals,
        meioses=(
            Meiosis(parent_id="2", child_id="1", parent_sex="1"),
            Meiosis(parent_id="3", child_id="1", parent_sex="2"),
            Meiosis(parent_id="8", child_id="2", parent_sex="1"),
            Meiosis(parent_id="9", child_id="2", parent_sex="2"),
        ),
    )
    legacy_dataset = Dataset(
        markers=dataset.markers,
        families=(legacy_family,),
        affection_names=dataset.affection_names,
    )

    assert estimate_ibd(dataset, "marker") == estimate_ibd(
        legacy_dataset,
        "marker",
    )


def test_meiosis_order_closes_equally_typed_leaf_branches_first() -> None:
    """Keep informative descendant histories out of the active frontier."""

    def person(
        individual_id: str,
        father_id: str | None,
        mother_id: str | None,
        genotype: tuple[str | None, str | None],
    ) -> Individual:
        return Individual(
            family_id="1",
            individual_id=individual_id,
            father_id=father_id,
            mother_id=mother_id,
            sex="1",
            phenotypes={},
            genotypes={"marker": genotype},
        )

    ordered_people = _merlin_meiosis_person_order(
        [
            person("father", None, None, (None, None)),
            person("mother", None, None, (None, None)),
            person("spouse", None, None, (None, None)),
            person("branch", "father", "mother", ("1", "2")),
            person("leaf", "father", "mother", ("1", "2")),
            person("grandchild", "branch", "spouse", ("1", "2")),
        ]
    )

    assert [person.individual_id for person in ordered_people] == [
        "leaf",
        "branch",
        "grandchild",
    ]


def test_map_filters_genotypes_without_shifting_ped_columns(
    tmp_path: Path,
) -> None:
    dat_path = tmp_path / "filtered.dat"
    map_path = tmp_path / "filtered.map"
    ped_path = tmp_path / "filtered.ped"
    dat_path.write_text(
        """\
A trait
M unmapped
M chr2_right
M chr1_marker
M chr2_left
"""
    )
    map_path.write_text(
        """\
CHROMOSOME MARKER POSITION
2 chr2_right 20.0
1 chr1_marker 10.0
2 chr2_left 5.0
"""
    )
    ped_path.write_text("1 1 0 0 1 2 9 9 1 2 2 2 1 1\n")

    dataset = load_merlin_inputs(ped_path, dat_path, map_path)

    assert [marker.name for marker in dataset.markers] == [
        "chr2_right",
        "chr1_marker",
        "chr2_left",
    ]
    person = dataset.families[0].individuals[0]
    assert person.phenotypes == {"trait": "2"}
    assert person.genotypes == {
        "chr2_right": ("1", "2"),
        "chr1_marker": ("2", "2"),
        "chr2_left": ("1", "1"),
    }


def test_rejects_duplicate_map_markers(tmp_path: Path) -> None:
    dat_path = tmp_path / "duplicate.dat"
    map_path = tmp_path / "duplicate.map"
    ped_path = tmp_path / "duplicate.ped"
    dat_path.write_text("M marker\n")
    map_path.write_text("1 marker 1.0\n1 marker 2.0\n")
    ped_path.write_text("1 1 0 0 1 1 1\n")

    with pytest.raises(ValueError, match="Duplicate marker 'marker' in map"):
        load_merlin_inputs(ped_path, dat_path, map_path)
