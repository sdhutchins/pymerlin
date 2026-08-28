#!/bin/bash

#SBATCH --job-name=pymerlin-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Conda solving and package installation are serial in this workflow.
#SBATCH --cpus-per-task=1
# Environment creation is a low-compute task. Eight GiB leaves margin for the
# Conda solver and compiled dependency metadata without making a large request.
#SBATCH --mem=8G
# Thirty minutes allows dependency solving and installation from a cold cache.
#SBATCH --time=00:30:00
# The documented two-hour express limit fits this bounded setup job.
#SBATCH --partition=express
#SBATCH --output=pymerlin-setup-%j.out
#SBATCH --error=pymerlin-setup-%j.err

set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly script_directory
repository_root="$(cd "${script_directory}/.." && pwd)"
readonly repository_root
readonly conda_module="${PYMERLIN_CONDA_MODULE:-minforge/conda}"
readonly conda_environment_prefix="${PYMERLIN_CONDA_ENV_PREFIX:-${repository_root}/.conda/pymerlin}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' "Environment setup must run inside a Slurm allocation." >&2
    exit 2
fi
if ! command -v module >/dev/null 2>&1; then
    printf '%s\n' "The Cheaha module command is unavailable." >&2
    exit 2
fi

cd "${repository_root}"
module reset
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

if [[ -x "${conda_environment_prefix}/bin/python" ]]; then
    conda env update \
        --prefix "${conda_environment_prefix}" \
        --file environment.yml \
        --prune
else
    conda env create \
        --prefix "${conda_environment_prefix}" \
        --file environment.yml
fi

conda activate "${conda_environment_prefix}"
python -m pip install --no-deps --editable ".[test]"
python -c "import gmpy2, networkx, numpy, pymerlin"
python -m pytest \
    tests/test_paired_dag_audit.py \
    tests/test_founder_orientation_quotient.py \
    tests/test_transition_planner.py \
    -q

printf 'setup_job_id\t%s\n' "${SLURM_JOB_ID}"
printf 'conda_module\t%s\n' "${conda_module}"
printf 'conda_environment\t%s\n' "${conda_environment_prefix}"
printf 'python_executable\t%s\n' "$(command -v python)"
