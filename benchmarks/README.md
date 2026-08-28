# Benchmark datasets

The benchmark registry separates scientific relevance from permission to use
or redistribute data. A paper citing MERLIN is evidence that a dataset is a
useful workload. It is not evidence that the underlying individual-level data
are public.

`datasets.json` records four states:

- `active`: usable now from files already in the repository;
- `candidate`: scientifically suitable, but acquisition or conversion remains;
- `blocked`: access approval or an authoritative data source is required; and
- `excluded`: useful context that cannot become a reproducible project
  benchmark under its current access conditions.

Before changing a dataset to `active`, record:

1. the authoritative download location and accession;
2. access and redistribution terms;
3. exact filenames, sizes, and SHA-256 checksums;
4. the conversion from source data to MERLIN input files;
5. the MERLIN version and complete reference command; and
6. expected output files and checksums.

Do not commit controlled individual-level genotype data. Locally acquired data
should live outside the repository or in an ignored directory documented by
the future acquisition workflow.

## Initial candidates

The first real-data target is Platinum Genomes CEPH pedigree 1463. The study
used MERLIN to calculate genome-wide inheritance vectors. The complete
17-member pedigree is represented by dbGaP accession `phs001224.v1.p1`; a
six-sample sequencing subset is publicly listed under ENA project
`PRJEB3381`. Dataset activation requires choosing a subset that contains the
pedigree members and marker representation needed for a MERLIN comparison.

GAW14 is a strong simulated stress test because a published analysis ran
MERLIN on 50 extended pedigrees across microsatellite and multiple SNP-density
maps. Historical distribution required a data request. It remains blocked
until current permission and redistribution terms are confirmed.

## Cheaha paired-DAG scaling diagnostic

`cheaha_pah_paired_dag_100m.sh` runs one bounded 100-million-state structural
audit on the synthetic PAH-scale fixture. It does not calculate linkage
statistics and does not submit itself.

The workflow uses the user-observed Cheaha command
`module load miniforge/conda`. The public UAB module page does not currently
list that exact module name. Verify it directly on Cheaha before setup:

```bash
module load miniforge/conda
```

The module is provided through a personal module tree. Preserve the submitting
shell's `MODULEPATH`; the setup and benchmark scripts do not call `module reset`.

Do not run `conda init` on Cheaha. The setup and benchmark scripts source the
module-managed Conda activation script directly.

Clone or update the repository from GitHub, then submit environment creation
from the repository root. Environment creation runs as a compute job because
UAB directs users not to create Conda environments on login nodes:

```bash
git clone https://github.com/sdhutchins/pymerlin.git
cd pymerlin
sbatch benchmarks/cheaha_setup_pymerlin.sh
```

After the setup job passes its focused tests, submit the diagnostic:

```bash
sbatch benchmarks/cheaha_pah_paired_dag_100m.sh
```

Both scripts use `.conda/pymerlin` by default. Override the observed module or
environment prefix only when Cheaha reports a different live configuration:

```bash
export PYMERLIN_CONDA_MODULE="miniforge/conda"
export PYMERLIN_CONDA_ENV_PREFIX="/path/to/pymerlin-conda"
```

The Slurm request is based on the completed local 25-million-state benchmark:

- one CPU because the audit is serial;
- 16 GB because the local process used 1.05 GiB and the larger frontier grew
  nonlinearly;
- 30 minutes because the local end-to-end run took 129 seconds; and
- `express` because UAB documents a two-hour limit for that CPU partition.

The script writes a partial result during execution and atomically promotes it
after success. A matching completed result is reused when its source signature
matches. An interrupted job restarts from the beginning because the in-memory
DAG frontier is not checkpointed. Partial files are retained for diagnosis.

After the Slurm job reaches a terminal state, collect final accounting data:

```bash
benchmarks/review_cheaha_pah_paired_dag_job.sh SLURM_JOB_ID
```

The review helper writes both `sacct` and `seff` output under
`benchmarks/results/`. Use completed jobs for resource-efficiency estimates.
Use failed or cancelled jobs to diagnose the command or environment instead.

UAB Research Computing documentation checked August 28, 2026:

- [Cheaha hardware and partition limits](https://docs.rc.uab.edu/cheaha/hardware/)
- [Pre-installed modules](https://docs.rc.uab.edu/cheaha/software/modules/)
- [Self-installed software and Conda restrictions](https://docs.rc.uab.edu/cheaha/software/software/)
- [Submitting Slurm jobs](https://docs.rc.uab.edu/cheaha/slurm/submitting_jobs/)
- [Reviewing jobs with sacct and seff](https://docs.rc.uab.edu/cheaha/slurm/job_management/)
- [Known issues](https://docs.rc.uab.edu/news/category/known-issues/)
- [Maintenance notices](https://docs.rc.uab.edu/news/category/maintenance/)
