"""MERLIN-compatible command line entrypoint behavior."""

from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path

from .backends import executable_multipoint_engines
from .error_detection import (
    detect_unlikely_genotypes,
    format_merlin_error_file,
)
from .ibd import IbdResult, estimate_ibd_for_markers
from .information import (
    format_merlin_information_console,
    format_merlin_information_table,
    multipoint_information_content,
)
from .io import load_merlin_inputs
from .kong_cox import (
    exponential_kong_cox,
    format_merlin_kong_cox_console,
    format_merlin_kong_cox_table,
    linear_kong_cox,
)
from .models import Dataset, Family, Individual
from .multipoint import (
    MultipointIbdResult,
    PositionIbdResult,
    TreePositionPosteriors,
    multipoint_ibd_at_positions,
    multipoint_state_posteriors_at_positions,
    multipoint_tree_posteriors_at_positions,
)
from .npl import format_merlin_npl_zscores, multipoint_npl_pairs
from .positions import AnalysisPosition, merlin_analysis_positions
from .selection import partition_dataset_by_chromosome


class _InvariantIbd(Enum):
    ZERO = "zero"
    HALF = "half"
    ONE = "one"
    UNKNOWN = "unknown"


def run_merlin_compatible(argv: list[str]) -> int:
    """Run the supported MERLIN-compatible CLI surface."""

    parser = argparse.ArgumentParser(prog="pymerlin")
    parser.add_argument(
        "-d",
        dest="dat",
        required=True,
        help="Data description file.",
    )
    parser.add_argument("-p", dest="ped", required=True, help="Pedigree file.")
    parser.add_argument("-m", dest="map", default=None, help="Map file.")
    parser.add_argument(
        "-f",
        dest="freq",
        default=None,
        help="Allele frequency mode or file.",
    )
    parser.add_argument(
        "-x",
        dest="missing",
        default="-99.999",
        help="Missing value code.",
    )
    parser.add_argument("-r", dest="seed", default="123456", help="Random seed.")
    parser.add_argument(
        "--cpus",
        type=_positive_int,
        default=1,
        help="Independent family worker processes.",
    )
    parser.add_argument(
        "--engine",
        choices=executable_multipoint_engines(),
        default="dense",
        help="Multipoint inheritance-state engine.",
    )
    parser.add_argument(
        "--error",
        action="store_true",
        help="Identify genotypes that are unlikely given flanking markers.",
    )
    parser.add_argument(
        "--ibd",
        action="store_true",
        help="Estimate IBD probabilities.",
    )
    parser.add_argument(
        "--pairs",
        action="store_true",
        help="Calculate the affected-pairs NPL statistic.",
    )
    parser.add_argument(
        "--zscores",
        action="store_true",
        help="Write raw family nonparametric Z scores.",
    )
    parser.add_argument(
        "--tabulate",
        action="store_true",
        help="Write a machine-readable nonparametric linkage table.",
    )
    parser.add_argument(
        "--exp",
        action="store_true",
        help="Calculate the exponential Kong-Cox model.",
    )
    parser.add_argument(
        "--information",
        action="store_true",
        help="Calculate multipoint inheritance information content.",
    )
    parser.add_argument(
        "--singlepoint",
        action="store_true",
        help="Run single-point analysis.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Positions per marker interval.",
    )
    parser.add_argument("--maxStep", dest="max_step", type=float, default=None)
    parser.add_argument("--minStep", dest="min_step", type=float, default=None)
    parser.add_argument(
        "--grid",
        type=float,
        default=None,
        help="Analysis grid spacing in cM.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="First analysis position in cM.",
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=None,
        help="Last analysis position in cM.",
    )
    parser.add_argument(
        "--positions",
        default=None,
        help="Comma-separated positions or markers.",
    )
    parser.add_argument("--prefix", default="merlin", help="Output file prefix.")
    parser.add_argument(
        "--markerNames",
        dest="marker_names",
        action="store_true",
        help="Label multipoint output with marker names.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Accepted for MERLIN compatibility.",
    )

    args = parser.parse_args(_normalize_merlin_tokens(argv))
    if not args.error and not args.ibd and not args.pairs and not args.information:
        parser.error(
            "PyMerlin currently requires --error, --ibd, --pairs, or "
            "--information."
        )
    if args.zscores and not args.pairs:
        parser.error("--zscores currently requires --pairs.")
    if args.tabulate and not args.pairs and not args.information:
        parser.error("--tabulate requires --pairs or --information.")
    if args.exp and not args.pairs:
        parser.error("--exp requires --pairs.")
    if (args.pairs or args.information) and args.singlepoint:
        parser.error(
            "Affected-pairs and information analyses require multipoint mode."
        )
    if args.engine == "tree" and args.singlepoint:
        parser.error("--engine tree requires multipoint mode.")

    freq_path, frequency_mode = _frequency_args(args.freq)
    try:
        dataset = load_merlin_inputs(
            args.ped,
            args.dat,
            args.map,
            freq_path,
            frequency_mode,
        )
    except NotImplementedError as error:
        parser.error(str(error))

    tree_chromosome_datasets: tuple[Dataset, ...] = ()
    tree_analysis_positions: tuple[tuple[AnalysisPosition, ...], ...] = ()
    tree_posteriors: tuple[TreePositionPosteriors, ...] = ()
    needs_tree_multipoint = args.engine == "tree" and (
        (args.ibd and not args.singlepoint) or args.pairs or args.information
    )

    if args.error:
        if args.map is None:
            parser.error("MERLIN-compatible error detection requires a map file.")
        try:
            genotype_errors = tuple(
                genotype_error
                for chromosome_dataset in partition_dataset_by_chromosome(dataset)
                for genotype_error in detect_unlikely_genotypes(
                    chromosome_dataset,
                    workers=args.cpus,
                )
            )
        except ValueError as error:
            parser.error(str(error))

        error_output_path = Path(f"{args.prefix}.err")
        error_output_path.write_text(
            format_merlin_error_file(genotype_errors)
        )
        if not args.quiet:
            print(
                "Unlikely genotypes listed in file "
                f"[{error_output_path}]"
            )

    if needs_tree_multipoint:
        if args.map is None:
            parser.error("The tree engine requires a map file.")
        try:
            tree_chromosome_datasets = partition_dataset_by_chromosome(dataset)
            tree_analysis_positions = tuple(
                _analysis_positions(chromosome_dataset, args)
                for chromosome_dataset in tree_chromosome_datasets
            )
            tree_posteriors = tuple(
                multipoint_tree_posteriors_at_positions(
                    chromosome_dataset,
                    analysis_positions,
                    workers=args.cpus,
                )
                for chromosome_dataset, analysis_positions in zip(
                    tree_chromosome_datasets,
                    tree_analysis_positions,
                )
            )
        except ValueError as error:
            parser.error(str(error))

    if args.ibd:
        if args.singlepoint:
            ibd_results: tuple[
                IbdResult | MultipointIbdResult | PositionIbdResult,
                ...,
            ] = estimate_ibd_for_markers(dataset, workers=args.cpus)
            use_marker_names = True
        else:
            if args.map is None:
                parser.error(
                    "MERLIN-compatible multipoint IBD requires a map file."
                )
            try:
                if args.engine == "tree":
                    ibd_results = tuple(
                        result
                        for (
                            chromosome_dataset,
                            analysis_positions,
                            chromosome_posteriors,
                        ) in zip(
                            tree_chromosome_datasets,
                            tree_analysis_positions,
                            tree_posteriors,
                        )
                        for result in multipoint_ibd_at_positions(
                            chromosome_dataset,
                            analysis_positions,
                            workers=args.cpus,
                            engine="tree",
                            tree_posteriors=chromosome_posteriors,
                        )
                    )
                else:
                    analysis_positions = _analysis_positions(dataset, args)
                    ibd_results = multipoint_ibd_at_positions(
                        dataset,
                        analysis_positions,
                        workers=args.cpus,
                    )
            except ValueError as error:
                parser.error(str(error))
            use_marker_names = False

        ibd_output_path = Path(f"{args.prefix}.ibd")
        ibd_output_path.write_text(
            format_merlin_ibd(
                dataset,
                ibd_results,
                use_marker_names=use_marker_names,
            )
        )
        if not args.quiet:
            print(f"IBD probabilities stored in file [{ibd_output_path}]")

    if args.pairs or args.information:
        if args.map is None:
            parser.error(
                "MERLIN-compatible multipoint analysis requires a map file."
            )
        try:
            if args.engine == "tree":
                chromosome_datasets = tree_chromosome_datasets
                analysis_positions_by_chromosome = tree_analysis_positions
                posteriors_by_chromosome = tree_posteriors
            else:
                chromosome_datasets = partition_dataset_by_chromosome(dataset)
                analysis_positions_by_chromosome = tuple(
                    _analysis_positions(chromosome_dataset, args)
                    for chromosome_dataset in chromosome_datasets
                )
                posteriors_by_chromosome = tuple(
                    multipoint_state_posteriors_at_positions(
                        chromosome_dataset,
                        analysis_positions,
                        workers=args.cpus,
                    )
                    for chromosome_dataset, analysis_positions in zip(
                        chromosome_datasets,
                        analysis_positions_by_chromosome,
                    )
                )
            if args.pairs and args.engine == "tree":
                npl_results = tuple(
                    multipoint_npl_pairs(
                        chromosome_dataset,
                        analysis_positions,
                        workers=args.cpus,
                        engine="tree",
                        tree_posteriors=position_posteriors,
                    )
                    for (
                        chromosome_dataset,
                        analysis_positions,
                        position_posteriors,
                    ) in zip(
                        chromosome_datasets,
                        analysis_positions_by_chromosome,
                        posteriors_by_chromosome,
                    )
                )
            elif args.pairs:
                npl_results = tuple(
                    multipoint_npl_pairs(
                        chromosome_dataset,
                        analysis_positions,
                        position_posteriors=position_posteriors,
                        workers=args.cpus,
                    )
                    for (
                        chromosome_dataset,
                        analysis_positions,
                        position_posteriors,
                    ) in zip(
                        chromosome_datasets,
                        analysis_positions_by_chromosome,
                        posteriors_by_chromosome,
                    )
                )
            else:
                npl_results = ()
            linear_results = tuple(linear_kong_cox(result) for result in npl_results)
            exponential_results = (
                tuple(exponential_kong_cox(result) for result in npl_results)
                if args.exp
                else None
            )
            if args.information and args.engine == "tree":
                information_results = tuple(
                    multipoint_information_content(
                        chromosome_dataset,
                        analysis_positions,
                        workers=args.cpus,
                        engine="tree",
                        tree_posteriors=position_posteriors,
                    )
                    for (
                        chromosome_dataset,
                        analysis_positions,
                        position_posteriors,
                    ) in zip(
                        chromosome_datasets,
                        analysis_positions_by_chromosome,
                        posteriors_by_chromosome,
                    )
                )
            elif args.information:
                information_results = tuple(
                    multipoint_information_content(
                        chromosome_dataset,
                        analysis_positions,
                        position_posteriors=position_posteriors,
                        workers=args.cpus,
                    )
                    for (
                        chromosome_dataset,
                        analysis_positions,
                        position_posteriors,
                    ) in zip(
                        chromosome_datasets,
                        analysis_positions_by_chromosome,
                        posteriors_by_chromosome,
                    )
                )
            else:
                information_results = ()
        except ValueError as error:
            parser.error(str(error))

        if args.pairs and not args.quiet:
            print(
                format_merlin_kong_cox_console(
                    linear_results,
                    exponential_results,
                ),
                end="",
            )

        if args.pairs and args.tabulate:
            table_output_path = Path(f"{args.prefix}-nonparametric.tbl")
            table_output_path.write_text(
                format_merlin_kong_cox_table(
                    linear_results,
                    exponential_results,
                )
            )
            if not args.quiet:
                print(f"NPL scores tabulated in [{table_output_path}]")

        if args.pairs and args.zscores:
            zscore_output_path = Path(f"{args.prefix}.zscore")
            zscore_output_path.write_text(
                format_merlin_npl_zscores(npl_results)
            )
            if not args.quiet:
                print(
                    "Nonparametric Z-scores for individual families stored in "
                    f"[{zscore_output_path}]"
                )

        if args.information:
            if not args.quiet:
                print(
                    format_merlin_information_console(information_results),
                    end="",
                )
            if args.tabulate:
                information_output_path = Path(f"{args.prefix}-info.tbl")
                information_output_path.write_text(
                    format_merlin_information_table(information_results)
                )
                if not args.quiet:
                    print(
                        "Information content tabulated in file "
                        f"[{information_output_path}]"
                    )

    return 0


def _analysis_positions(
    dataset: Dataset,
    args: argparse.Namespace,
) -> tuple[AnalysisPosition, ...]:
    """Plan analysis positions from the shared MERLIN-compatible options."""

    return merlin_analysis_positions(
        dataset,
        steps_per_interval=args.steps,
        max_step_cm=args.max_step,
        min_step_cm=args.min_step,
        grid_cm=args.grid,
        start_cm=args.start,
        stop_cm=args.stop,
        position_list=args.positions,
        use_marker_names=args.marker_names,
    )


def format_merlin_ibd(
    dataset: Dataset,
    results: tuple[
        IbdResult | MultipointIbdResult | PositionIbdResult,
        ...,
    ],
    use_marker_names: bool,
) -> str:
    """Format IBD results using MERLIN's autosomal output conventions."""

    lines = ["FAMILY ID1 ID2 MARKER P0 P1 P2"]
    family_orders = {
        family.family_id: _merlin_family_order(dataset, family)
        for family in dataset.families
    }
    invariant_relationships = {
        family.family_id: _invariant_ibd_relationships(
            family.family_id,
            family_orders[family.family_id],
        )
        for family in dataset.families
    }

    formatted_results = []
    for result in results:
        if isinstance(result, PositionIbdResult):
            marker_label = result.label
        else:
            marker = dataset.marker_by_name[result.marker_name]
            if use_marker_names:
                marker_label = marker.name
            else:
                if marker.position_cm is None:
                    raise ValueError(
                        "MERLIN-compatible multipoint output requires map positions."
                    )
                marker_label = f"{marker.position_cm:.3f}"
        row_lookup = {
            (str(row["family_id"]), str(row["id1"]), str(row["id2"])): row
            for row in result.rows
        }
        formatted_results.append((marker_label, row_lookup))

    # MERLIN analyzes every requested position for one family before moving
    # to the next family, so its output is family-major and position-minor.
    for family in dataset.families:
        people = [
            person.individual_id
            for person in family_orders[family.family_id]
        ]
        family_relationships = invariant_relationships[family.family_id]
        for marker_label, row_lookup in formatted_results:
            for index, id1 in enumerate(people):
                for id2 in people[: index + 1]:
                    relationship = family_relationships[(id1, id2)]
                    if relationship is not _InvariantIbd.UNKNOWN:
                        probabilities = _format_invariant_probabilities(relationship)
                    else:
                        row = row_lookup.get((family.family_id, id2, id1))
                        if row is None:
                            row = row_lookup[(family.family_id, id1, id2)]
                        probabilities = " ".join(
                            _format_computed_probability(float(row[state]))
                            for state in ("z0", "z1", "z2")
                        )
                    lines.append(
                        f"{family.family_id} {id1} {id2} {marker_label}  "
                        f"{probabilities}"
                    )
    return "\n".join(lines) + "\n"


def format_merlin_singlepoint_ibd(
    dataset: Dataset,
    results: tuple[IbdResult, ...],
) -> str:
    """Preserve the original public formatter for single-point callers."""

    return format_merlin_ibd(dataset, results, use_marker_names=True)


def _invariant_ibd_relationships(
    family_id: str,
    ordered_people: tuple[Individual, ...],
) -> dict[tuple[str, str], _InvariantIbd]:
    """Classify autosomal relationships using MERLIN's PrepareIBD rules."""

    people = tuple(person.individual_id for person in ordered_people)
    index_by_id = {
        individual_id: index
        for index, individual_id in enumerate(people)
    }
    relationships: dict[tuple[int, int], _InvariantIbd] = {}

    for index, person in enumerate(ordered_people):
        if person.is_founder:
            relationships[(index, index)] = _InvariantIbd.ONE
            for previous_index in range(index):
                _set_symmetric_relationship(
                    relationships,
                    index,
                    previous_index,
                    _InvariantIbd.ZERO,
                )
            continue

        if person.father_id not in index_by_id or person.mother_id not in index_by_id:
            raise ValueError(
                "MERLIN-compatible invariant IBD formatting requires both "
                f"parents of {person.individual_id!r} in family {family_id!r}."
            )

        father_index = index_by_id[person.father_id]
        mother_index = index_by_id[person.mother_id]
        if father_index >= index or mother_index >= index:
            raise ValueError(
                "MERLIN-compatible invariant IBD formatting requires parents "
                "to precede their children in pedigree order."
            )

        inbred = (
            relationships[(mother_index, father_index)] is not _InvariantIbd.ZERO
        )
        relationships[(index, index)] = (
            _InvariantIbd.UNKNOWN if inbred else _InvariantIbd.ONE
        )

        for previous_index in range(index):
            mother_relationship = relationships[(mother_index, previous_index)]
            father_relationship = relationships[(father_index, previous_index)]
            if (
                mother_relationship is _InvariantIbd.ZERO
                and father_relationship is _InvariantIbd.ZERO
            ):
                relationship = _InvariantIbd.ZERO
            elif inbred or previous_index not in {mother_index, father_index}:
                relationship = _InvariantIbd.UNKNOWN
            elif (
                relationships[(previous_index, previous_index)]
                is _InvariantIbd.ONE
            ):
                relationship = _InvariantIbd.HALF
            else:
                relationship = _InvariantIbd.UNKNOWN

            _set_symmetric_relationship(
                relationships,
                index,
                previous_index,
                relationship,
            )

    return {
        (people[first_index], people[second_index]): relationship
        for (first_index, second_index), relationship in relationships.items()
    }


def _merlin_family_order(
    dataset: Dataset,
    family: Family,
) -> tuple[Individual, ...]:
    """Reproduce MERLIN's ancestor-safe, informativeness-based family path."""

    founders = [person for person in family.individuals if person.is_founder]
    pending = [person for person in family.individuals if not person.is_founder]
    ordered_people = founders.copy()
    ordered_ids = {person.individual_id for person in founders}

    while pending:
        newly_ordered = [
            person
            for person in pending
            if person.father_id in ordered_ids and person.mother_id in ordered_ids
        ]
        if not newly_ordered:
            unresolved_ids = ", ".join(
                person.individual_id for person in pending
            )
            raise ValueError(
                "MERLIN-compatible family ordering could not place "
                f"individuals {unresolved_ids} in family {family.family_id!r}."
            )

        ordered_people.extend(newly_ordered)
        ordered_ids.update(person.individual_id for person in newly_ordered)
        newly_ordered_ids = {
            person.individual_id for person in newly_ordered
        }
        pending = [
            person
            for person in pending
            if person.individual_id not in newly_ordered_ids
        ]

    scores = [
        _merlin_informativeness_score(dataset, person)
        for person in ordered_people
    ]
    founder_count = len(founders)
    for current_index in range(founder_count + 1, len(ordered_people)):
        person = ordered_people[current_index]
        positions = {
            ordered_person.individual_id: index
            for index, ordered_person in enumerate(ordered_people)
        }
        if person.father_id not in positions or person.mother_id not in positions:
            raise ValueError(
                "MERLIN-compatible family ordering requires both parents of "
                f"{person.individual_id!r} in family {family.family_id!r}."
            )

        new_index = max(
            founder_count,
            positions[person.father_id] + 1,
            positions[person.mother_id] + 1,
        )
        while (
            new_index < current_index
            and scores[new_index] > scores[current_index]
        ):
            new_index += 1

        if new_index != current_index:
            ordered_people.insert(new_index, ordered_people.pop(current_index))
            scores.insert(new_index, scores.pop(current_index))

    return tuple(ordered_people)


def _merlin_informativeness_score(
    dataset: Dataset,
    person: Individual,
) -> int:
    """Score a person as MERLIN does before optimizing its family path."""

    genotyped_markers = sum(
        all(allele is not None for allele in person.genotypes[marker.name])
        for marker in dataset.markers
    )
    affected = (
        bool(dataset.affection_names)
        and person.phenotypes.get(dataset.affection_names[0]) == "2"
    )
    return genotyped_markers * 2 + int(affected)


def _set_symmetric_relationship(
    relationships: dict[tuple[int, int], _InvariantIbd],
    first_index: int,
    second_index: int,
    relationship: _InvariantIbd,
) -> None:
    relationships[(first_index, second_index)] = relationship
    relationships[(second_index, first_index)] = relationship


def _format_invariant_probabilities(relationship: _InvariantIbd) -> str:
    if relationship is _InvariantIbd.ZERO:
        return "1.0 0.0 0.0"
    if relationship is _InvariantIbd.HALF:
        return "0.0 1.0 0.0"
    if relationship is _InvariantIbd.ONE:
        return "0.0 0.0 1.0"
    raise ValueError("Unknown IBD relationships require computed probabilities.")


def _format_computed_probability(value: float) -> str:
    formatted = f"{value:.5f}"
    # IBD probabilities cannot be negative. Canonicalizing rounded negative
    # zero prevents floating-point residue from becoming user-visible output.
    return "0.00000" if formatted == "-0.00000" else formatted


def _normalize_merlin_tokens(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    for token in argv:
        if token.startswith("--prefix:"):
            normalized.extend(["--prefix", token.split(":", 1)[1]])
        elif token.startswith("--") and ":" in token:
            option, value = token.split(":", 1)
            normalized.extend([option, value])
        else:
            normalized.append(token)
    return normalized


def _frequency_args(freq_arg: str | None) -> tuple[str | None, str]:
    if freq_arg is None:
        return None, "all"
    if freq_arg in {"a", "A"}:
        return None, "all"
    if freq_arg in {"e", "E"}:
        return None, "equal"
    if freq_arg in {"f", "F"}:
        return None, "founders"
    if freq_arg in {"m", "M"}:
        return None, "ml"
    return freq_arg, "all"


def _positive_int(value: str) -> int:
    """Parse a process count while producing an argparse-compatible error."""

    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("--cpus must be a positive integer.")
    return parsed_value
