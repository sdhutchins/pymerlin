#!/bin/bash

set -euo pipefail

if [[ "$#" -ne 1 ]] || [[ ! "$1" =~ ^[0-9]+$ ]]; then
    printf 'Usage: %s SLURM_JOB_ID\n' "${0##*/}" >&2
    exit 2
fi

for required_command in awk date mkdir mv sacct seff; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        printf 'Required command is unavailable: %s\n' "${required_command}" >&2
        exit 2
    fi
done

readonly job_id="$1"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly script_directory
readonly default_results_directory="${script_directory}/results"
readonly results_directory="${PYMERLIN_BENCHMARK_RESULTS_DIR:-${default_results_directory}}"
readonly sacct_result_path="${results_directory}/pah_paired_dag_${job_id}.sacct.tsv"
readonly seff_result_path="${results_directory}/pah_paired_dag_${job_id}.seff.txt"
readonly partial_sacct_path="${sacct_result_path}.partial"
readonly partial_seff_path="${seff_result_path}.partial"

job_state="$(
    sacct -j "${job_id}" -X --noheader --format=State \
        | awk 'NF {print $1; exit}'
)"
if [[ -z "${job_state}" ]]; then
    printf 'No Slurm accounting record is available for job %s.\n' \
        "${job_id}" >&2
    exit 1
fi
case "${job_state}" in
    PENDING*|RUNNING*|COMPLETING*|CONFIGURING*)
        printf 'Job %s is not terminal: %s\n' "${job_id}" "${job_state}" >&2
        exit 1
        ;;
esac

mkdir -p "${results_directory}"
sacct \
    -j "${job_id}" \
    --parsable2 \
    --format=JobID,JobName,State,Partition,ReqCPUS,AllocCPUS,ReqMem,Elapsed,TotalCPU,MaxRSS,ExitCode \
    > "${partial_sacct_path}"
seff "${job_id}" > "${partial_seff_path}"
mv "${partial_sacct_path}" "${sacct_result_path}"
mv "${partial_seff_path}" "${seff_result_path}"

printf 'reviewed_at_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'job_state\t%s\n' "${job_state}"
printf 'sacct_result\t%s\n' "${sacct_result_path}"
printf 'seff_result\t%s\n' "${seff_result_path}"
