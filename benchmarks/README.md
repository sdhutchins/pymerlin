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
