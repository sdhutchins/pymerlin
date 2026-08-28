# Numerical validation contract

PyMerlin aims to reproduce MERLIN's scientific results while supporting CPU
and GPU computation. A backend is not accepted merely because its values fall
within a fixed tolerance. It must preserve MERLIN-compatible output and be at
least as accurate as MERLIN when both are compared with an independent
high-precision reference.

## Three separate requirements

| Requirement | Question | Acceptance evidence |
| --- | --- | --- |
| Compatibility | Does the user receive the same scientific result as MERLIN? | The same rows, identifiers, ordering, and values after MERLIN-style formatting. |
| Reproducibility | Does the same computation produce the same result again? | Repeated runs use a documented reduction order and produce identical output for a backend. |
| Accuracy | Is the computed value close to the mathematical result? | PyMerlin and MERLIN are compared independently with a 256-bit MPFR oracle. |

Passing one requirement does not imply that the other two pass.

## Current acceptance rule

For every marker, family, non-self individual pair, and IBD state:

1. PyMerlin must match MERLIN after formatting each probability to five
   decimal places.
2. Signed zero is canonicalized to `0.00000`. This is an intentional accuracy
   correction for MERLIN output such as `-0.00000`.
3. The absolute PyMerlin error relative to the MPFR oracle must not exceed the
   larger of:

   - MERLIN's absolute error relative to the same oracle.
   - One float64 representation unit at the oracle value.

The representation allowance prevents a correctly rounded float64 value from
failing when MERLIN happens to print an exact decimal. It is not a general
numerical tolerance.

The current marker-position tests cover all 315 probabilities from 105
non-self pair-marker rows across every available MERLIN IBD fixture:

- default-frequency single-point IBD;
- explicit-frequency single-point IBD;
- explicit-frequency two-marker multipoint IBD; and
- three-marker multipoint IBD with estimated frequencies.

Intermarker validation adds a 256-bit MPFR comparison for all 45 probabilities
from the midpoint of the `basic2` map. An opt-in external test also runs MERLIN
and PyMerlin with `--steps 1` and requires byte-identical `.ibd` output. Set
`PYMERLIN_MERLIN_BIN` to the MERLIN executable to enable that test.

New algorithms and backends must add larger and more difficult pedigrees
rather than weakening this rule.

## Oracle boundary

The MPFR implementation independently repeats the numerical path:

- founder allele-assignment probabilities;
- inheritance-state likelihood reductions;
- Haldane recombination fractions;
- forward and backward recursions;
- propagation of both conditionals to intermarker analysis positions;
- posterior normalization; and
- pairwise IBD accumulation.

It shares the parsed `Dataset` model with PyMerlin. Therefore, the current
oracle validates computation after parsing. A later fixture layer must retain
decimal input tokens so that parsing and frequency estimation can also be
checked against exact decimal or rational inputs.

PyMerlin subtracts map coordinates through their shortest decimal
representations and evaluates Haldane fractions with `expm1`. Intermarker IBD
bins are accumulated with `math.sumprod` before normalization. Python 3.10 and
3.11 use an exact-input `Decimal` fallback because they do not provide
`math.sumprod`.

Single-point markers must be evaluated independently by both PyMerlin and the
MPFR oracle. Passing multiple markers to the oracle creates a multipoint model
and is not a valid single-point reference.

## CPU and GPU policy

The reference implementation uses float64. An accelerated backend must:

- use float64 by default for compatibility runs;
- avoid reassociation and `fast-math` transformations in compatibility mode;
- use a deterministic reduction strategy;
- report its maximum absolute, relative, and ULP error against MPFR; and
- identify every value that is worse than MERLIN against MPFR.

A faster backend is rejected if any value is less accurate than MERLIN beyond
the float64 representation allowance. A backend that is more accurate may
differ internally, but it must retain compatible formatted output unless the
MERLIN display itself exposes an inaccurate artifact such as signed zero.

For large reductions, reproducible or correctly rounded accumulation should be
evaluated before accepting a performance-only reduction. ExBLAS-style long
accumulators and floating-point expansions are established approaches for
bit-reproducible parallel reductions. Monte Carlo arithmetic is useful as an
additional instability detector, but it does not replace the MPFR oracle.

## Backend package policy

- NumPy is the reference CPU array package. Compatibility code must request
  `numpy.float64` explicitly when constructing numerical arrays.
- gmpy2 is a test-only dependency that provides the MPFR accuracy oracle.
- CuPy is the primary CUDA array prototype. It must use explicit float64 data
  and a validated deterministic reduction instead of assuming that a stock GPU
  reduction reproduces CPU ordering.
- JAX is experimental because it disables 64-bit values by default. Any JAX
  backend must enable `jax_enable_x64` before creating arrays and must still
  pass the MPFR contract.
- Numba-CUDA-MLIR is the candidate for custom deterministic CUDA kernels.
  NVIDIA identifies it as the new-development successor to maintenance-mode
  Numba-CUDA. It requires an NVIDIA GPU, a compatible driver, and the CUDA 12
  or CUDA 13 installation extra selected for the execution host.

Package choices follow the current official documentation for
[NumPy](https://numpy.org/doc/stable/user/basics.types.html),
[gmpy2](https://gmpy2.readthedocs.io/en/stable/overview.html),
[CuPy](https://docs.cupy.dev/en/stable/),
[JAX](https://docs.jax.dev/en/latest/default_dtypes.html), and
[NVIDIA Numba-CUDA-MLIR](https://nvidia.github.io/numba-cuda-mlir/).

## CPU process policy

`--cpus` distributes independent families across worker processes. For one
large family, it distributes independent marker emissions and then computes
the forward and backward message directions concurrently with at most two
processes. Each direction retains the original marker, inheritance-state, and
floating-point reduction order. Trees cross process boundaries through a flat
DAG encoding that records each shared node once and preserves float64 leaves
exactly.

The default is `--cpus 1`. Higher values should be selected from representative
benchmarks because process startup and data serialization can outweigh useful
parallel work for small pedigrees or short marker panels.

Use the phase diagnostic before a long pedigree run:

```bash
.venv/bin/python -m pymerlin.cli benchmark \
    -p path/to/input.ped \
    -d path/to/input.dat \
    -m path/to/input.map \
    -f path/to/input.freq \
    --workload tree-multipoint \
    --marker-limit 5 \
    --cpus 4
```

The diagnostic reports emission, forward-backward, and posterior times plus
the relevant-meiosis range, detected counting selectors, and maximum unique
DAG node counts. Canonical suffix cache hits and misses quantify the broader
founder-renaming equivalence reduction even when no strict untyped chain is
detected. These fields identify the phase that needs further reduction without
weakening output parity.

## Validation sequence

1. Run the small MPFR accuracy test on the NumPy reference backend.
2. Add a benchmark dataset only after its provenance and access terms are
   recorded in `benchmarks/datasets.json`.
3. Generate reference MERLIN output with the recorded executable version and
   command.
4. Compare MERLIN and each PyMerlin backend with the same MPFR oracle on a
   tractable subset.
5. Run full-dataset compatibility and performance benchmarks only after the
   subset passes.

## Method references

- Radford M. Neal. "Fast exact summation using small and large
  superaccumulators." arXiv:1505.05571 (2015).
- Roman Iakymchuk et al. "Reproducibility of Parallel Preconditioned Conjugate
  Gradient in Hybrid Programming Environments." arXiv:2005.07282 (2020).
- Christophe Denis, Pablo de Oliveira Castro, and Eric Petit. "Verificarlo:
  Checking Floating Point Accuracy through Monte Carlo Arithmetic."
  doi:10.1109/ARITH.2016.31 (2016).
