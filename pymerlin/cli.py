"""Command line interface for PyMerlin."""

from __future__ import annotations

import argparse
import sys

from .backends import list_backend_status
from .benchmark import (
    benchmark_error_detection,
    benchmark_marker,
    benchmark_tree_multipoint,
)
from .compare import compare_singlepoint_ibd_to_merlin
from .ibd import estimate_ibd_for_markers
from .io import load_merlin_inputs
from .merlin_cli import run_merlin_compatible


SUBCOMMANDS = {"ibd", "benchmark", "compare-merlin-ibd", "backends"}


def _positive_int(value: str) -> int:
    """Parse a process count while producing an argparse-compatible error."""

    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("--cpus must be a positive integer.")
    return parsed_value


def _positive_float(value: str) -> float:
    """Parse a positive duration with an argparse-compatible error."""

    parsed_value = float(value)
    if parsed_value <= 0.0:
        raise argparse.ArgumentTypeError(
            "The diagnostic duration must be positive."
        )
    return parsed_value


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] not in SUBCOMMANDS:
        return run_merlin_compatible(argv)

    parser = argparse.ArgumentParser(prog="pymerlin")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ibd_parser = subparsers.add_parser(
        "ibd",
        help="Estimate pairwise IBD/kinship for one marker.",
    )
    ibd_parser.add_argument(
        "-p",
        "--ped",
        required=True,
        help="MERLIN/QTDT pedigree file.",
    )
    ibd_parser.add_argument(
        "-d",
        "--dat",
        required=True,
        help="MERLIN/QTDT data description file.",
    )
    ibd_parser.add_argument("-m", "--map", default=None, help="Optional marker map file.")
    ibd_parser.add_argument("-f", "--freq", default=None, help="Optional allele frequency file.")
    ibd_parser.add_argument(
        "--marker",
        action="append",
        help="Marker name to analyze. Repeat for multiple markers.",
    )
    ibd_parser.add_argument(
        "--chromosome",
        default=None,
        help="Restrict analysis to a chromosome in the map file.",
    )
    ibd_parser.add_argument(
        "--start-cm",
        type=float,
        default=None,
        help="Restrict analysis to markers at or after this cM.",
    )
    ibd_parser.add_argument(
        "--end-cm",
        type=float,
        default=None,
        help="Restrict analysis to markers at or before this cM.",
    )
    ibd_parser.add_argument(
        "--backend",
        default="numpy",
        choices=["numpy"],
        help="Compute backend.",
    )
    ibd_parser.add_argument(
        "--cpus",
        type=_positive_int,
        default=1,
        help="Independent family worker processes.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Time likelihood, multipoint-tree, or error-detection phases.",
    )
    benchmark_parser.add_argument("-p", "--ped", required=True, help="MERLIN/QTDT pedigree file.")
    benchmark_parser.add_argument("-d", "--dat", required=True, help="MERLIN/QTDT data description file.")
    benchmark_parser.add_argument("-m", "--map", default=None, help="Optional marker map file.")
    benchmark_parser.add_argument("-f", "--freq", default=None, help="Optional allele frequency file.")
    benchmark_parser.add_argument(
        "--workload",
        choices=("singlepoint", "error", "tree-multipoint"),
        default="singlepoint",
        help="Compute path to benchmark.",
    )
    benchmark_parser.add_argument("--marker", help="Marker name for a singlepoint benchmark.")
    benchmark_parser.add_argument("--repeats", type=int, default=3, help="Number of timed repeats.")
    benchmark_parser.add_argument(
        "--cpus",
        type=_positive_int,
        default=1,
        help="Independent family worker processes.",
    )
    benchmark_parser.add_argument(
        "--family-copies",
        type=int,
        default=1,
        help="Duplicate families for synthetic scaling.",
    )
    benchmark_parser.add_argument(
        "--marker-limit",
        type=_positive_int,
        default=5,
        help="Ordered marker count for a tree-multipoint diagnostic.",
    )
    benchmark_parser.add_argument(
        "--heartbeat-nodes",
        type=_positive_int,
        default=10_000,
        help="Report tree traversal counters after this many expanded nodes.",
    )
    benchmark_parser.add_argument(
        "--max-emission-nodes",
        type=_positive_int,
        default=None,
        help="Stop each diagnostic marker after this many expanded nodes.",
    )
    benchmark_parser.add_argument(
        "--max-emission-seconds",
        type=_positive_float,
        default=None,
        help="Stop each diagnostic marker after this many seconds.",
    )

    compare_parser = subparsers.add_parser("compare-merlin-ibd", help="Compare to a MERLIN singlepoint .ibd file.")
    compare_parser.add_argument("-p", "--ped", required=True, help="MERLIN/QTDT pedigree file.")
    compare_parser.add_argument("-d", "--dat", required=True, help="MERLIN/QTDT data description file.")
    compare_parser.add_argument("-m", "--map", default=None, help="Optional marker map file.")
    compare_parser.add_argument("-f", "--freq", default=None, help="Optional allele frequency file.")
    compare_parser.add_argument("--merlin-ibd", required=True, help="MERLIN .ibd output file.")
    compare_parser.add_argument(
        "--marker",
        action="append",
        help="Marker name to compare. Repeat for multiple markers.",
    )
    compare_parser.add_argument("--tolerance", type=float, default=1e-4, help="Maximum allowed absolute difference.")

    subparsers.add_parser("backends", help="Report accelerator backend availability.")

    args = parser.parse_args(argv)
    if args.command == "ibd":
        dataset = load_merlin_inputs(args.ped, args.dat, args.map, args.freq)
        results = estimate_ibd_for_markers(
            dataset,
            marker_names=args.marker,
            chromosome=args.chromosome,
            start_cm=args.start_cm,
            end_cm=args.end_cm,
            backend=args.backend,
            workers=args.cpus,
        )
        print("marker\tfamily_id\tid1\tid2\tz0\tz1\tz2\tpi_hat\tkinship")
        for result in results:
            for row in result.rows:
                print(
                    (
                        "{marker}\t{family_id}\t{id1}\t{id2}\t{z0:.6g}\t{z1:.6g}\t"
                        "{z2:.6g}\t{pi_hat:.6g}\t{kinship:.6g}"
                    ).format(
                        marker=result.marker_name,
                        **row,
                    )
                )
    elif args.command == "benchmark":
        if args.workload == "singlepoint":
            if args.marker is None:
                parser.error("--marker is required for a singlepoint benchmark.")
            summary = benchmark_marker(
                args.ped,
                args.dat,
                args.marker,
                args.map,
                args.freq,
                repeats=args.repeats,
                family_copies=args.family_copies,
                workers=args.cpus,
            )
        elif args.workload == "error":
            if args.map is None:
                parser.error("--map is required for an error benchmark.")
            summary = benchmark_error_detection(
                args.ped,
                args.dat,
                args.map,
                args.freq,
                repeats=args.repeats,
                family_copies=args.family_copies,
                workers=args.cpus,
            )
        else:
            if args.map is None:
                parser.error(
                    "--map is required for a tree-multipoint benchmark."
                )
            summary = benchmark_tree_multipoint(
                args.ped,
                args.dat,
                args.map,
                args.freq,
                marker_limit=args.marker_limit,
                workers=args.cpus,
                heartbeat_node_interval=args.heartbeat_nodes,
                emission_node_limit=args.max_emission_nodes,
                emission_time_limit_seconds=args.max_emission_seconds,
                progress=lambda message: print(
                    message,
                    file=sys.stderr,
                    flush=True,
                ),
            )
        for key, value in summary.items():
            print(f"{key}\t{value}")
    elif args.command == "compare-merlin-ibd":
        dataset = load_merlin_inputs(args.ped, args.dat, args.map, args.freq)
        mismatches = compare_singlepoint_ibd_to_merlin(
            dataset,
            args.merlin_ibd,
            marker_names=args.marker,
            tolerance=args.tolerance,
        )
        print("marker\tfamily_id\tid1\tid2\tmax_abs_diff\tmerlin_p0p1p2\tpymerlin_p0p1p2")
        for mismatch in mismatches:
            print(
                (
                    "{marker}\t{family_id}\t{id1}\t{id2}\t{max_abs_diff:.6g}\t"
                    "{merlin}\t{pymerlin}"
                ).format(
                    marker=mismatch.marker,
                    family_id=mismatch.family_id,
                    id1=mismatch.id1,
                    id2=mismatch.id2,
                    max_abs_diff=mismatch.max_abs_diff,
                    merlin=",".join(f"{value:.6g}" for value in mismatch.merlin),
                    pymerlin=",".join(f"{value:.6g}" for value in mismatch.pymerlin),
                )
            )
        return 1 if mismatches else 0
    elif args.command == "backends":
        print("backend\tavailable\trole")
        for backend in list_backend_status():
            print(f"{backend.name}\t{str(backend.available).lower()}\t{backend.role}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
