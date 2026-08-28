#!/bin/bash

#SBATCH --job-name=pymerlin-pah-dag-100m
#SBATCH --nodes=1
#SBATCH --ntasks=1
# The paired-DAG audit is serial, so additional allocated CPUs would be idle.
#SBATCH --cpus-per-task=1
# The 25-million-state run used 1.05 GiB. Sixteen GiB allows fourfold state
# growth plus margin for the observed nonlinear frontier expansion.
#SBATCH --mem=16G
# The measured 25-million run took 129 seconds end to end. Thirty minutes
# allows fourfold state growth plus a scheduler and hardware safety margin.
#SBATCH --time=00:30:00
# The documented two-hour express limit fits this bounded 30-minute CPU job.
#SBATCH --partition=express
#SBATCH --output=pymerlin-pah-dag-100m-%j.out
#SBATCH --error=pymerlin-pah-dag-100m-%j.err

set -euo pipefail

readonly state_limit=100000000
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly script_directory
repository_root="$(cd "${script_directory}/.." && pwd)"
readonly repository_root
readonly results_directory="${repository_root}/benchmarks/results"
readonly final_result_path="${results_directory}/pah_paired_dag_100m.tsv"
readonly conda_module="${PYMERLIN_CONDA_MODULE:-minforge/conda}"
readonly conda_environment_prefix="${PYMERLIN_CONDA_ENV_PREFIX:-${repository_root}/.conda/pymerlin}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' "This benchmark must run inside a Slurm allocation." >&2
    exit 2
fi

for required_command in awk date git hostname mkdir mv sha256sum; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        printf 'Required command is unavailable: %s\n' "${required_command}" >&2
        exit 2
    fi
done

cd "${repository_root}"
if ! command -v module >/dev/null 2>&1; then
    printf '%s\n' "The Cheaha module command is unavailable." >&2
    exit 2
fi
# The Conda module lives in a personal module tree inherited through MODULEPATH.
module load "${conda_module}"
if ! command -v conda >/dev/null 2>&1; then
    printf 'Conda is unavailable after loading module: %s\n' \
        "${conda_module}" >&2
    exit 2
fi
conda_base="$(conda info --base)"
readonly conda_base
# Cheaha manages Conda through modules, so activate without running conda init.
# shellcheck source=/dev/null
source "${conda_base}/etc/profile.d/conda.sh"
if [[ ! -x "${conda_environment_prefix}/bin/python" ]]; then
    printf 'PyMerlin Conda environment is unavailable: %s\n' \
        "${conda_environment_prefix}" >&2
    printf '%s\n' \
        "Run benchmarks/cheaha_setup_pymerlin.sh through sbatch first." >&2
    exit 2
fi
conda activate "${conda_environment_prefix}"
if ! python -c "import pymerlin"; then
    printf '%s\n' "The activated Conda environment cannot import pymerlin." >&2
    exit 2
fi

mkdir -p "${results_directory}"
source_signature="$({
    sha256sum \
        benchmarks/pah_paired_dag_benchmark.py \
        environment.yml \
        pymerlin/founder_orientation_quotient.py \
        pymerlin/inheritance_tree.py \
        pymerlin/likelihood.py \
        pymerlin/paired_dag_audit.py \
        tests/pah_scale_fixture.py \
        tests/fixtures/pah_scale/genotyped_ids.txt \
        tests/fixtures/pah_scale/pedigree.ped
} | sha256sum | awk '{print $1}')"
readonly source_signature

if [[ -f "${final_result_path}" ]] \
    && awk -F '\t' -v signature="${source_signature}" '
        $1 == "benchmark_completed" && $2 == "true" {completed = 1}
        $1 == "source_signature" && $2 == signature {matched = 1}
        END {exit !(completed && matched)}
    ' "${final_result_path}"; then
    printf 'Matching completed result already exists: %s\n' \
        "${final_result_path}"
    exit 0
fi

readonly partial_result_path="${final_result_path}.partial.${SLURM_JOB_ID}"
git_revision="$(git rev-parse --verify HEAD 2>/dev/null || printf 'unavailable')"
readonly git_revision

printf 'benchmark_job_id\t%s\n' "${SLURM_JOB_ID}"
printf 'benchmark_host\t%s\n' "$(hostname)"
printf 'benchmark_started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'git_revision\t%s\n' "${git_revision}"
printf 'source_signature\t%s\n' "${source_signature}"

# Limit implicit numerical libraries to the one CPU requested from Slurm.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONHASHSEED=0
export PYMERLIN_BENCHMARK_RESULT_PATH="${partial_result_path}"
export PYMERLIN_BENCHMARK_SOURCE_SIGNATURE="${source_signature}"
export PYMERLIN_PAIRED_DAG_STATE_LIMIT="${state_limit}"

python -m benchmarks.pah_paired_dag_benchmark
mv "${partial_result_path}" "${final_result_path}"

printf 'benchmark_finished_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'benchmark_result\t%s\n' "${final_result_path}"
while IFS= read -r result_line; do
    printf '%s\n' "${result_line}"
done < "${final_result_path}"
printf 'review_command\tbenchmarks/review_cheaha_pah_paired_dag_job.sh %s\n' \
    "${SLURM_JOB_ID}"
