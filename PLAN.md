# PyMerlin Implementation Plan

## Objective

PyMerlin must reproduce MERLIN's scientific results and user-visible output
while supporting deterministic CPU and GPU computation. Performance changes
are accepted only after they satisfy the numerical contract in
`docs/numerical-validation.md`.

The immediate target is an exact multipoint linkage analysis on the synthetic
PAH-scale pedigree. PDF reporting is intentionally deferred. Text and tabular
scientific outputs are the acceptance artifacts.

## Non-negotiable acceptance contract

Every implementation phase must preserve all of the following:

1. MERLIN-compatible rows, identifiers, ordering, labels, and formatted values.
2. PyMerlin error against the MPFR oracle no greater than the larger of
   MERLIN's error and one float64 representation unit.
3. Deterministic output for repeated runs with the same backend and worker
   count.
4. No silent fallback from float64 to lower precision.
5. No performance optimization accepted only because a loose tolerance passes.
6. No weakening of existing external-MERLIN or numerical-accuracy tests.

## Current measured baseline

The synthetic PAH-scale fixture contains:

- 912 people;
- 1,336 total meioses;
- 23 genotyped people;
- 66 marker-relevant people; and
- 82 marker-relevant meioses.

The original bounded one-marker diagnostic expanded 100,000 recursive nodes
without finishing. Exact closed-component peeling subsequently allowed one
synthetic PAH marker emission to complete in approximately 14.5 seconds. Five
marker emissions completed in approximately 18.3 seconds with four workers.
The forward-backward transition then exceeded five minutes and approximately
2.8 GB of memory.

| Internal ordering | Seconds | Cache hits | Cache misses | Prunes |
| --- | ---: | ---: | ---: | ---: |
| Identifier-derived order | 7.417 | 27,902 | 72,097 | 5,152 |
| Parent-before-child order | 9.072 | 19,969 | 80,030 | 9,923 |

The traversal executes nodes quickly enough for a Python reference engine.
The immediate problem is now transition-state growth. A one-marker factor
diagnostic had induced width 11, but filtering across one recombination
interval produced a connected 68-bit message. Outer marker workers cannot
correct exponential work inside that transition.

The parent-before-child ordering remains provisional because it follows the
structure used by MERLIN, but it did not improve performance by itself. It
has not been reevaluated because transition-state growth is the higher-priority
blocker.

## Phase 1: Implement exact closed-component peeling (completed)

### Rationale

`FuzzyInheritanceTree::ScoreRecursive()` in `merlin/Houdini.cpp` gradually
peels founder-origin graph components when they have no remaining phenotyped
descendants. PyMerlin currently recognizes disconnected constraint components
but retains their full signatures in every later suffix-cache key. Completed
history therefore prevents otherwise equivalent future states from sharing a
subtree.

### Implementation

- Partition resolved genotype constraints into open and closed components.
- Treat a component as closed only when none of its founder origins is present
  in an active parent or pending parental transmission.
- Evaluate each newly closed component once with the existing Decimal founder
  component likelihood code.
- Journal peeled constraints so recursive rollback restores the exact parent
  state.
- Exclude peeled components from canonical future-state keys.
- Cache a normalized future subtree that does not contain the current peeled
  factor.
- Apply the local Decimal factor when restoring the branch.
- Convert to float64 only at a controlled terminal boundary.
- Add counters for peeled components, peeled constraints, zero factors, and
  normalized-cache reuse.

### Accuracy tests

- Compare every inheritance-state likelihood with the MPFR oracle.
- Compare peeled and non-peeled trees exactly on small pedigrees.
- Include branches where components close at different recursion depths.
- Include two histories with different peeled factors but the same open state.
- Require no loss of accuracy after factor application.
- Retain the pending-parental-transmission regression.

### Performance gate

- The PAH bounded diagnostic must report at least one peeled component.
- Cache-miss growth must slow materially relative to the current baseline.
- A one-marker PAH emission must complete before attempting five markers.
- If one marker still exceeds five minutes, stop and inspect the new counters.

### Completion status

- Focused peeling and rollback tests pass.
- One synthetic PAH marker emission completes within the five-minute gate.
- Five marker emissions complete with four workers.
- Multipoint transition growth remains unresolved and is now the active
  computational blocker.

## Phase 1B: Report bounded PAH subpedigrees (implemented)

The pedigree-reduction report enumerates every affected-pair ancestral closure
without modifying the input pedigree. For each closure it records:

- retained individual identifiers;
- raw meioses and MERLIN-effective inheritance bits;
- marker-relevant meioses;
- retained typed and affected people;
- relationship distance and connected-component count; and
- whether the closure is below MERLIN's default 24-bit limit.

The synthetic PAH fixture contains 666 affected-pair closures. All are below
the 24-bit computational threshold, but only 153 retain at least two typed
people. No candidate retains more than 6 of the 23 typed people. These are
information-retention counts, not linkage-power estimates.

Candidate subpedigrees overlap and can share founders, meioses, affected
people, and genotype observations. They must not be analyzed and combined as
though they were independent families. Selecting a subpedigree also changes
the scientific analysis relative to the complete 912-person pedigree.

## Phase 1C: Test exact marker-coordinate reduction (reference implemented)

The coordinate-reduction reference finds restrictions that hold across every
nonzero state of one marker-emission tree. It removes fixed inheritance bits
and groups bit pairs whose XOR is constant. Higher-order affine restrictions
remain explicit as zero-valued states, so the reference does not enlarge the
nonzero support or change recombination weights.

The implementation includes:

- a deterministic map between full and reduced inheritance vectors;
- compressed-tree support analysis using GF(2) integer bitsets;
- a bounded dense tree reducer;
- a bounded exact source-target recombination oracle; and
- regression tests against the existing full-tree transition at recombination
  fractions 0, 0.2, and 0.5.

The two-marker PAH diagnostic showed that pairwise coordinate constraints are
too weak for this pedigree:

| Marker | Full bits | Fixed bits | Pair-parity removals | Reduced bits |
| --- | ---: | ---: | ---: | ---: |
| `PAH_SIM_1` | 1,336 | 0 | 2 | 1,334 |
| `PAH_SIM_2` | 1,336 | 1 | 4 | 1,331 |

The corresponding dense reference calculation would require
`2 ** (1,334 + 1,331)` source-target pairs. The explicit one-million-pair
guard correctly prevents that calculation. Marker-coordinate constraints are
therefore a correctness primitive, not a PAH-scale transition solution.

The next exact performance experiment must operate on sparse tree structure.
It should condition the transition on the next marker while preserving shared
subtrees and expose separator assignments as independent deterministic tasks.
Before implementation, measure separator width and total conditioned work on
the 68-bit connected PAH message. HPC distribution is useful only if that
measurement shows bounded per-task memory and feasible total work.

### Sparse conditioning planner result

The diagnostic planner now constructs a conservative interaction graph from
the current marker tree and the next marker-emission tree. It connects a split
bit to downstream split bits in the same compressed subtree. The planner then
selects separator bits until every remaining component is at most 24 bits. It
prefers fixed or pair-parity-linked marker coordinates when structural choices
are otherwise equal.

On the first synthetic PAH interval, the planner measured:

- 65 functionally active inheritance bits in one structural component;
- 41 separator bits required to leave one 24-bit component;
- exactly 12,272,533,504 marker-compatible assignments for the selected
  separator;
- 16,777,216 component states per task;
- 402,653,184 estimated transform-node operations per task;
- 4,941,574,691,132,276,736 estimated total node operations; and
- 268,435,456 bytes, or 256 MiB, of float workspace per task.

The memory and node-work estimates assume two float64 workspace arrays and
sequential component processing. The structural graph can overestimate
interactions because it does not assume an unproved algebraic factorization.
The task count is exact for this selected separator. It is calculated by a
bounded reduced Boolean decision diagram that projects the next marker's
nonzero support onto the separator bits.

This selected conditioning plan is not HPC-feasible. Per-task memory is
modest, but more than 12 billion separator assignments are not a responsible
Slurm array. The approximately `4.94e18` total-operation figure is a
conservative upper estimate, not a lower-bound proof against every possible
exact factorization. Do not implement or submit a Slurm array for this plan.

### Exact founder-orientation quotient result

The founder-orientation quotient fixes one arbitrary allele-label orientation
per transmitting founder and retains the other transmissions as relative XOR
coordinates. Relative bits from the same founder use one coupled exact
transition factor. They are not treated as independent meioses by the sparse
planner.

On the first synthetic PAH interval, the quotient measured:

- 1,336 full inheritance bits;
- 1,092 exact quotient bits after removing 244 founder orientations;
- 60 active quotient bits, compared with 65 active full-coordinate bits;
- 36 separator bits required to leave one 24-bit component;
- exactly 788,529,152 compatible separator assignments; and
- approximately `3.18e17` estimated transform-node operations.

This is approximately a 15.6-fold reduction in the selected separator task
count. It remains too large for a Slurm array and does not make the existing
materialized transition implementation PAH-ready.

### Bounded paired-DAG transition audit

The paired-DAG audit models a fused exact transition and next-marker
conditioning recursion. Its memoization key contains the current DAG node,
the next-marker DAG node, inheritance-bit depth, and any open latent founder
orientation contexts. It prunes zero contributions and stops terminal node
pairs before opening new founder contexts.

For the 1 cM interval from `PAH_SIM_1` to `PAH_SIM_2`, using Haldane's map
function and the founder quotient, the 10-million-state audit measured:

- recombination fraction `0.0099006633466223494`;
- 9 active founder context groups;
- at most 3 simultaneously open founder contexts;
- a maximum frontier of 322,944 unique states;
- 10,272,067 explored nonzero transition arcs; and
- the state cap reached at reduced bit 960 of 1,092 after 29.556 seconds of
  audit time, excluding marker-tree construction.

The audit did not finish. Ten million unique subproblems is therefore a lower
bound, not the total interval workload. The result demonstrates much greater
reuse than independent separator tasks, but it does not yet justify a
production fused evaluator or a large Cheaha allocation.

A subsequent 25-million-state run also reached its cap. It measured:

- reduced bit 988 of 1,092 reached;
- a maximum frontier of 2,358,768 unique states;
- 29,124,967 explored nonzero transition arcs;
- 110.987 seconds of audit time;
- 128.96 seconds for fixture construction, marker trees, quotient reduction,
  and the audit together; and
- 1,124,925,440 bytes, approximately 1.05 GiB, of peak resident memory for the
  complete process.

The larger frontier shows that state growth is not linear in bit depth. Do not
extrapolate a complete count directly from the 10-million or 25-million runs.
These measurements are sufficient to size another bounded diagnostic, not a
production likelihood evaluator.

Cheaha job `39871380` completed the bounded 100-million-state benchmark at Git
revision `bc526007ac8885b4d3527e4ce20e3e48ef270c9a`. It reached the cap rather
than completing the structural state space. The baseline measured:

- reduced bit 1,030 of 1,092 reached;
- a maximum frontier of 4,342,800 unique states;
- 113,385,847 explored nonzero transition arcs;
- 536.675 seconds of audit time and 564.163 seconds total;
- 94.32% CPU efficiency on one CPU; and
- 1.56 GiB peak resident memory from the batch step.

The measured resource request for another run at the same cap is one serial
CPU, 4 GB of memory, 20 minutes, and the `express` partition. The memory and
walltime each retain more than a twofold margin over job `39871380`. The
wrapper refuses to run outside Slurm, promotes results atomically, and reuses
only a completed result with a matching source signature.

### Exact founder-couple quotient result

The remaining exchangeable-founder-couple symmetry is now projected exactly
through the founder-orientation quotient. The implementation fixes one
complemented representative coordinate per eligible couple. Recombination
still sums over the complete target orbit, so the hidden coordinate is not
treated as an independent or zero-recombination bit.

On an exhaustive bounded pedigree, every reduced marker likelihood matched its
full inheritance-state likelihood. Every compound-quotient transition entry
matched the sum over its full target states, and every transition row summed to
one. These tests establish exactness for the tested topology. They do not by
themselves establish tractability on PAH.

The synthetic PAH fixture contains one eligible founder couple. Cheaha job
`39873064` tested the compound quotient at the same 100-million-state cap as
the founder-orientation baseline. It measured:

- 1,091 compound-quotient bits, one fewer than the founder-orientation result;
- 60 active bits;
- 9 active founder-orientation context groups; and
- 1 active founder-couple context group;
- a maximum frontier of 8,685,600 states, exactly twice the baseline;
- 126,771,693 transition arcs, 11.8% more than the baseline;
- 711.960 seconds of audit time, 32.7% more than the baseline; and
- 3.07 GB peak memory, compared with 1.56 GB for the baseline.

The job completed successfully with 97.38% CPU efficiency on one CPU. It
reached the state cap at reduced bit 1,009, which was 21 bits earlier than the
baseline. The exact quotient is therefore a net loss in the current PAH
paired-DAG representation.

A direct key-structure audit explains the result. The quotient representative
is orientation-reduced bit 0, but that coordinate is already shared rather
than split in both marker DAGs. The quotient therefore removes zero active
branches. Its target-orbit choice affects 10 active coordinates spanning
reduced bits 93 through 1,045. Those alternatives must remain distinguishable
between the first and last affected coordinates for an exact nontrivial
recombination transition.

A node-only canonicalization has no branch at the inactive representative to
remove and cannot discard the distinct transition weights. Packing both
orientations into a two-lane numeric value could reduce Python key overhead,
but it would retain the doubled scalar work across almost the whole pedigree.
The PAH benchmark therefore keeps the exact founder-couple implementation for
validation but audits the faster founder-orientation representation. It also
reports the structural founder-couple key assessment so this decision remains
visible and testable.

## Phase 2: Add MERLIN-style logging and message parity

### Logging architecture

- Create module loggers with `logging.getLogger(__name__)`.
- Library modules must never configure the root logger or install handlers.
- The CLI configures one stderr handler and owns verbosity policy.
- Scientific result files remain separate from logs.
- Normal analysis must not log inside the recursive hot loop.
- Diagnostic heartbeats remain opt-in and interval-bounded.
- Worker processes return structured progress events to the parent process.
- The parent orders user-visible events deterministically before emitting them.
- Include family, chromosome, marker, analysis phase, and worker context when it
  helps identify a long-running operation.

### Log-level policy

| Level | Intended content |
| --- | --- |
| `ERROR` | Invalid inputs, impossible execution state, and failed output writes |
| `WARNING` | Mendelian incompatibility, uninformative-family fallback, ignored options, and scientifically important data limitations |
| `INFO` | Input summaries, chromosome and family analysis lifecycle, progress, completed phases, and written output paths |
| `DEBUG` | Cache statistics, pruning, symmetry plans, component peeling, worker scheduling, and numerical diagnostics |

### MERLIN message inventory

The first logging pass will inventory and map messages from these source
locations:

| MERLIN source | Message category | PyMerlin destination |
| --- | --- | --- |
| `merlin/Merlin.cpp` | Version, chromosome lifecycle, simulation and frequency-file announcements | CLI startup and completion logger |
| `merlin/MerlinCore.cpp::PrintMessage` | Analysis messages and marker issues | Analysis logger at `INFO` or `WARNING` |
| `merlin/MerlinCore.cpp::ProgressMessage` | Family and marker progress | Parent-process progress reporter |
| `merlin/MerlinCore.cpp::ShowFamily` | Founder, descendant, and bit summaries | Family-selection logger at `INFO` |
| `merlin/MerlinParameters.cpp` | Disabled or incompatible options | CLI validation logger and parser errors |
| MERLIN output modules | Written-file announcements | Output writer logger at `INFO` |

### Quiet and interactive behavior

- `--quiet` suppresses informational lifecycle and progress messages.
- Errors remain visible.
- Warning behavior will follow the corresponding MERLIN source branch and be
  covered by tests rather than assumed.
- Interactive terminals may use carriage-return progress updates.
- Redirected output and test capture use complete newline-terminated records.
- Multiprocessing logs must not interleave partial lines.

### Logging tests

- Use `caplog` to verify levels and message fields.
- Require `--quiet` to suppress informational messages.
- Require failures to remain visible under `--quiet`.
- Compare selected normalized PyMerlin messages with external MERLIN output.
- Require identical scientific output files with logging enabled and disabled.
- Verify that parallel marker completion produces deterministic parent messages.
- Measure logging-disabled overhead and require it to remain negligible.

## Phase 3: Reevaluate traversal ordering and exact equivalence classes

### Traversal-order decision

- Rerun the bounded PAH diagnostic after component peeling.
- Compare identifier-derived, parent-before-child, and MERLIN-compatible
  traversal paths using the same marker and node budget.
- Record maximum open founder-origin frontier width and maximum pending
  transmission count.
- Keep the parent-before-child order only if it improves the complete peeled
  algorithm or is required for exact MERLIN behavior.
- Otherwise restore the faster order and keep the internal permutation
  explicit and tested.

### Remaining exact reductions

- Extend untyped-chain detection only when an equivalence proof covers the
  additional topology.
- Distinguish marker-emission equivalence from NPL and IBD equivalence.
- Do not exclude or include affected people without considering the downstream
  score tree.
- Compare founder-couple handling directly with `merlin/Mantra.cpp`.
- Implement additional founder-couple or grandchild symmetry only on pedigrees
  that satisfy MERLIN's `EffectivelyIdentical()` conditions.
- Add state-by-state MPFR tests for every new quotient.

### Completion gate

- One marker on the PAH fixture completes deterministically.
- Every reported reduction has a nonzero counter on a fixture designed to
  exercise it.
- No reduction is accepted based only on a smaller node count.

## Phase 4: Complete PAH scientific output parity

The following outputs are required before the PAH workflow is considered
ready:

- multipoint affected-pairs NPL Z scores;
- linear and exponential Kong-Cox tables;
- information-content tables;
- pairwise IBD at markers and requested intermarker positions;
- unlikely-genotype error output; and
- MERLIN-compatible text and tabular formatting.

### Validation sequence

1. Validate a bounded PAH-derived branch against external MERLIN.
2. Validate every tractable state against MPFR.
3. Run one full synthetic PAH marker.
4. Run five markers with four CPU workers.
5. Run 25 markers and confirm approximately linear marker scaling.
6. Run the refined real PAH inputs only after their metadata, genotypes, map,
   and allele-frequency files pass input validation.

### PAH acceptance gate

- All requested analyses complete without dense inheritance-state expansion.
- Results are deterministic across repeated runs.
- One-worker and four-worker formatted outputs are identical.
- Peak memory and elapsed time are recorded.
- The exact input file checksums and command are recorded with each benchmark.
- Real-data conclusions are not drawn from the synthetic fixture.

## Phase 5: Refine CPU parallel execution

CPU work begins only after one marker completes efficiently.

- Keep independent-family parallelism for datasets with multiple families.
- Keep independent-marker parallelism for a single large family.
- Run forward and backward directions concurrently only when workers remain.
- Prevent nested process pools and CPU oversubscription.
- Pass worker counts explicitly through every analysis layer.
- Preserve marker ordering after completion-order scheduling.
- Use flat DAG serialization and verify shared-node identity after unpickling.
- Benchmark one, two, and four workers on the same workload.
- Report parent and child CPU time separately when the platform permits it.

### CPU acceptance gate

- Parallel output is byte-identical to single-worker output.
- Four workers improve representative walltime without excessive serialization.
- Resource recommendations are based on measured elapsed time and peak memory.

## Phase 6: Add an accelerated backend

GPU work starts only after the reference CPU tree algorithm is correct and
tractable.

### Backend sequence

1. Define a backend-neutral array and reduction interface.
2. Keep NumPy float64 as the reference implementation.
3. Prototype CuPy float64 kernels for independent, regular operations.
4. Evaluate custom CUDA kernels only where deterministic reduction order is
   required.
5. Keep JAX experimental and require `jax_enable_x64`.

### GPU numerical rules

- Disable fast-math and unsafe reassociation in compatibility mode.
- Use deterministic reduction trees.
- Compare every accelerated result independently with MPFR and MERLIN.
- Reject a faster result if it is less accurate than MERLIN beyond one float64
  representation unit.
- Permit different internal values only when they are more accurate and retain
  the required user-visible output.

### GPU acceptance gate

- CPU and GPU formatted outputs match MERLIN.
- Repeated GPU runs are deterministic in compatibility mode.
- Maximum absolute, relative, and ULP errors are reported.
- Transfer and kernel timings are reported separately.
- GPU acceleration improves a representative workload after transfer costs.

## Phase 7: Broader MERLIN completeness

After PAH readiness, inventory remaining MERLIN options and classify them as:

- implemented with exact external parity;
- implemented without an external fixture;
- partially implemented;
- intentionally deferred; or
- out of scope.

Likely later areas include chromosome-X behavior, parametric linkage models,
haplotyping, marker clusters, simulation, quantitative traits, association,
and additional report formats. Each area requires its own scientific contract
and fixtures. Feature percentages must be based on this inventory, not a
subjective estimate.

## Validation commands

Run focused correctness tests after each implementation slice:

```bash
.venv/bin/python -m pytest \
  tests/test_io.py \
  tests/test_likelihood.py \
  tests/test_marker_likelihood_tree.py \
  tests/test_founder_symmetry.py \
  tests/test_chain_reduction.py \
  tests/test_multipoint_tree.py \
  tests/test_npl.py \
  tests/test_benchmark.py \
  -q
```

Run numerical validation before accepting a scientific change:

```bash
.venv/bin/python -m pytest tests/test_numerical_accuracy.py -q
```

Run external MERLIN parity when the executable is available:

```bash
export PYMERLIN_MERLIN_BIN="$PWD/executables/merlin-macos-arm64"
.venv/bin/python -m pytest tests/test_merlin_external_parity.py -q
```

Do not run the five-marker PAH workload until the bounded one-marker diagnostic
shows that component peeling has constrained state growth.

## Documentation deliverables

- Keep `docs/numerical-validation.md` synchronized with accepted algorithms.
- Record benchmark dataset provenance and checksums.
- Document every CLI message whose wording intentionally differs from MERLIN.
- Document backend precision and determinism modes.
- Document known limitations without presenting unvalidated features as
  complete.
- Add user-facing PAH workflow instructions only after the full acceptance gate
  passes.

## Immediate next action

Do not apply the founder-couple quotient to the PAH paired-DAG evaluator. The
next exact performance investigation should compare inheritance-bit orderings
using the same bounded PAH interval and state cap. An ordering is useful only
if it reduces active marker-DAG nodes or the paired frontier without adding a
latent state dimension. Do not submit the separator-conditioning plan to
Slurm.
