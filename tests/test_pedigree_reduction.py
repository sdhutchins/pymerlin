from pathlib import Path

import pytest

from pymerlin import (
    analyze_pedigree_reduction,
    format_pedigree_reduction_report,
    load_merlin_inputs,
)


def test_reports_affected_pair_ancestral_complexity() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    report = analyze_pedigree_reduction(dataset, "1")

    assert report.family_id == "1"
    assert report.affection_name == "some_disease"
    assert report.full_individual_count == 6
    assert report.full_meiosis_count == 6
    assert report.full_merlin_bit_count == 3
    assert report.full_marker_relevant_meiosis_count == 6
    assert report.full_typed_person_count == 6
    assert report.full_affected_person_count == 2
    assert len(report.candidates) == 1

    candidate = report.candidates[0]
    assert candidate.affected_pair == ("5", "6")
    assert candidate.individual_ids == ("1", "2", "3", "4", "5", "6")
    assert candidate.meiosis_count == 6
    assert candidate.merlin_bit_count == 3
    assert candidate.marker_relevant_meiosis_count == 6
    assert candidate.typed_person_count == 6
    assert candidate.affected_person_count == 2
    assert candidate.relationship_distance == 2
    assert candidate.connected_component_count == 1
    assert candidate.within_bit_limit
    assert candidate.review_candidate
    assert report.review_candidates == (candidate,)


def test_formats_explicit_denominators_and_candidate_ids() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )
    report = analyze_pedigree_reduction(dataset, "1", bit_limit=2)

    formatted_report = format_pedigree_reduction_report(report)

    assert "# full_typed_people\t6\n" in formatted_report
    assert "# full_affected_people\t2\n" in formatted_report
    assert "# review_candidates\t0\n" in formatted_report
    assert "5\t6\t6\t6\t3\t6\t6\t2\t2\t1\tno\tno\t" in formatted_report
    assert formatted_report.endswith("1,2,3,4,5,6\n")


@pytest.mark.parametrize("bit_limit", (0, -1))
def test_rejects_nonpositive_bit_limits(bit_limit: int) -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    with pytest.raises(ValueError, match="bit_limit must be at least 1"):
        analyze_pedigree_reduction(dataset, "1", bit_limit=bit_limit)


def test_requires_known_markers() -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    with pytest.raises(ValueError, match="Unknown marker"):
        analyze_pedigree_reduction(
            dataset,
            "1",
            marker_names=("absent_marker",),
        )


def test_writes_no_files_while_constructing_report(tmp_path: Path) -> None:
    dataset = load_merlin_inputs(
        "examples/basic2.ped",
        "examples/basic2.dat",
        "examples/basic2.map",
        "examples/basic2.freq",
    )

    analyze_pedigree_reduction(dataset, "1")

    assert tuple(tmp_path.iterdir()) == ()
