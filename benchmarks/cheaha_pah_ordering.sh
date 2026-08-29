#!/bin/bash

#SBATCH --job-name=pymerlin-pah-ordering
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Each marker-tree build and paired-DAG audit is serial.
#SBATCH --cpus-per-task=1
# The larger paired-DAG experiment used 3.07 GB. Four GB bounds this audit,
# which also stops any candidate after 100,000 recursive marker nodes.
#SBATCH --mem=4G
# The bounded local smoke took 67 seconds. Ten minutes also covers both
# 120-second fallback time limits plus scheduler and node variation.
#SBATCH --time=00:10:00
# The documented two-hour express limit fits this bounded serial comparison.
#SBATCH --partition=express
#SBATCH --output=pymerlin-pah-ordering-%j.out
#SBATCH --error=pymerlin-pah-ordering-%j.err

set -euo pipefail

readonly state_limit=1000000
readonly marker_node_limit=100000
readonly marker_time_limit_seconds=120
readonly conda_module="${PYMERLIN_CONDA_MODULE:-miniforge/conda}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' "This benchmark must run inside a Slurm allocation." >&2
    exit 2
fi
if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    printf '%s\n' "SLURM_SUBMIT_DIR is unavailable." >&2
    exit 2
fi

readonly repository_root="${SLURM_SUBMIT_DIR}"
readonly results_directory="${repository_root}/benchmarks/results"
readonly final_result_path="${results_directory}/pah_meiosis_ordering.tsv"
readonly conda_environment_prefix="${PYMERLIN_CONDA_ENV_PREFIX:-${repository_root}/.conda/pymerlin}"
if [[ ! -f "${repository_root}/environment.yml" ]] \
    || [[ ! -f "${repository_root}/pyproject.toml" ]]; then
    printf 'Submit this job from the PyMerlin repository root: %s\n' \
        "${repository_root}" >&2
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
        benchmarks/pah_ordering_benchmark.py \
        environment.yml \
        pymerlin/__init__.py \
        pymerlin/founder_orientation_quotient.py \
        pymerlin/inheritance_tree.py \
        pymerlin/likelihood.py \
        pymerlin/meiosis_ordering.py \
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONHASHSEED=0
export PYMERLIN_ORDERING_RESULT_PATH="${partial_result_path}"
export PYMERLIN_ORDERING_SOURCE_SIGNATURE="${source_signature}"
export PYMERLIN_ORDERING_STATE_LIMIT="${state_limit}"
export PYMERLIN_ORDERING_MARKER_NODE_LIMIT="${marker_node_limit}"
export PYMERLIN_ORDERING_MARKER_TIME_LIMIT="${marker_time_limit_seconds}"

python -m benchmarks.pah_ordering_benchmark
mv "${partial_result_path}" "${final_result_path}"

printf 'benchmark_finished_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'benchmark_result\t%s\n' "${final_result_path}"
while IFS= read -r result_line; do
    printf '%s\n' "${result_line}"
done < "${final_result_path}"
printf 'review_command\tbenchmarks/review_cheaha_pah_ordering_job.sh %s\n' \
    "${SLURM_JOB_ID}"
