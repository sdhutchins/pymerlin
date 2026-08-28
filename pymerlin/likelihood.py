"""Reference single-marker likelihood engine.

This module favors correctness and transparent biological assumptions over
speed. It enumerates inheritance states and founder allele assignments, which
makes it a stable oracle for later sparse and GPU implementations.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product, repeat
from math import fsum
from time import perf_counter

import numpy as np

from .chain_reduction import (
    CountingPartition,
    UntypedChain,
    detect_untyped_chains,
)
from .founder_symmetry import (
    FounderOrientationSymmetryPlan,
    build_founder_couple_symmetry_plan,
    build_founder_orientation_symmetry_plan,
    restore_founder_couple_symmetry_branches,
    restore_founder_orientation_branch,
)
from .inheritance_tree import (
    InheritanceTree,
    LeafNode,
    SharedNode,
    SplitNode,
    TreeBuildStatistics,
    TreeNode,
    ZeroNode,
    _combine_children,
    _inheritance_recursion_budget,
    _materialize_scaled_tree,
    _restore_counting_class_nodes,
    _scaled_node,
)
from .models import Dataset, Family, Individual, Marker
from .parallel import validate_workers


FounderAllele = tuple[str, int]
AlleleOrigin = FounderAllele
_DECIMAL_PRECISION = 80


class MarkerTreeBudgetExceeded(RuntimeError):
    """Raised when a diagnostic marker-tree traversal reaches its budget."""


@dataclass(frozen=True)
class InheritanceState:
    """An inheritance state for one family at a marker or analysis position."""

    family_id: str
    bits: tuple[int, ...]
    likelihood: float
    posterior_weight: float
    allele_origins: dict[str, tuple[AlleleOrigin, AlleleOrigin]]


@dataclass(frozen=True)
class LikelihoodResult:
    """Likelihood summary for one marker across all families."""

    marker: Marker
    states: tuple[InheritanceState, ...]
    likelihood: float


@dataclass(frozen=True)
class FamilyStateSpace:
    """Inheritance vectors and origins determined only by family topology."""

    bits: tuple[tuple[int, ...], ...]
    allele_origins: tuple[
        dict[str, tuple[AlleleOrigin, AlleleOrigin]],
        ...,
    ]


@dataclass(frozen=True)
class MarkerAssignmentSpace:
    """Founder assignments and float64 probabilities for one family-marker."""

    assignments: tuple[dict[FounderAllele, str], ...]
    probabilities: np.ndarray


@dataclass(frozen=True)
class _FounderGenotypeConstraint:
    """One observed unordered genotype over two founder-allele origins."""

    first_origin: FounderAllele
    second_origin: FounderAllele
    first_allele: str
    second_allele: str


class _MarkerTraversalState:
    """Incrementally propagate founder origins through one marker tree.

    Tree recursion assigns inheritance bits in prefix order. This state keeps
    only origins implied by the current prefix and journals newly resolved
    people so a completed branch can be rolled back without rescanning the
    pedigree.
    """

    def __init__(
        self,
        family: Family,
        marker_name: str,
        inheritance_bits: list[int],
        relevant_individual_ids: frozenset[str],
        allele_frequencies: dict[str, Decimal] | None = None,
        positive_alleles: tuple[str, ...] = (),
        enable_suffix_cache: bool = True,
        progress: Callable[[str], None] | None = None,
        heartbeat_node_interval: int | None = None,
        node_limit: int | None = None,
        time_limit_seconds: float | None = None,
    ) -> None:
        if heartbeat_node_interval is not None and heartbeat_node_interval < 1:
            raise ValueError(
                "A marker-tree heartbeat interval must be positive."
            )
        if node_limit is not None and node_limit < 1:
            raise ValueError("A marker-tree node limit must be positive.")
        if time_limit_seconds is not None and time_limit_seconds <= 0.0:
            raise ValueError("A marker-tree time limit must be positive.")

        self.family = family
        self.marker_name = marker_name
        self.inheritance_bits = inheritance_bits
        self.relevant_individual_ids = relevant_individual_ids
        self._allele_frequencies = allele_frequencies
        self._positive_alleles = positive_alleles
        self.origins: dict[
            str,
            tuple[AlleleOrigin, AlleleOrigin],
        ] = {
            founder.individual_id: (
                (founder.individual_id, 0),
                (founder.individual_id, 1),
            )
            for founder in family.founders
            if founder.individual_id in relevant_individual_ids
        }
        self._people_by_id = family.by_id
        self._person_index_by_id = {
            person.individual_id: person_index
            for person_index, person in enumerate(family.individuals)
        }
        self._relevant_person_ids_in_order = tuple(
            person.individual_id
            for person in family.individuals
            if person.individual_id in relevant_individual_ids
        )
        self._observed_person_indices = tuple(
            person_index
            for person_index, person in enumerate(family.individuals)
            if _has_complete_marker_genotype(person, marker_name)
        )
        self._observed_person_index_set = frozenset(
            self._observed_person_indices
        )
        self._incoming_meiosis_indices = (
            self._build_incoming_meiosis_indices()
        )
        self._children_by_parent_id = self._build_children_by_parent_id()
        self._outgoing_meiosis_indices_by_parent_id = (
            self._build_outgoing_meiosis_indices_by_parent_id()
        )
        self._last_outgoing_meiosis_index_by_parent_id = {
            parent_id: bit_indices[-1]
            for parent_id, bit_indices in (
                self._outgoing_meiosis_indices_by_parent_id.items()
            )
        }
        self._constraints_by_person_index: dict[
            int,
            _FounderGenotypeConstraint,
        ] = {}
        self._resolved_journal: list[tuple[str, int | None]] = []
        self._peeled_constraint_journal: list[
            tuple[int, _FounderGenotypeConstraint]
        ] = []
        self._assigned_bit_count = 0
        self._enable_suffix_cache = enable_suffix_cache
        self._suffix_cache_hits = 0
        self._suffix_cache_misses = 0
        self._recursive_node_count = 0
        self._maximum_recursion_depth = 0
        self._contradiction_prune_count = 0
        self._founder_orientation_reduction_count = 0
        self._founder_couple_reduction_count = 0
        self._counting_reduction_count = 0
        self._invariant_reduction_count = 0
        self._peeled_component_count = 0
        self._peeled_constraint_count = 0
        self._zero_peeled_factor_count = 0
        self._normalized_cache_reuse_count = 0
        self._peeled_factor_cache_hit_count = 0
        self._peeled_factor_cache_miss_count = 0
        self._scaled_tree_cache_hit_count = 0
        self._progress = progress
        self._heartbeat_node_interval = heartbeat_node_interval
        self._node_limit = node_limit
        self._time_limit_seconds = time_limit_seconds
        self._traversal_start = perf_counter()
        self._time_check_node_interval = min(
            heartbeat_node_interval or 1_000,
            1_000,
        )
        self._suffix_tree_by_canonical_state: dict[
            tuple[object, ...],
            tuple[TreeNode, Decimal],
        ] = {}
        self._normalized_state_by_raw_key: dict[
            tuple[object, ...],
            tuple[tuple[object, ...], Decimal],
        ] = {}
        self._shared_prefix_by_node_and_depth: dict[
            tuple[int, int],
            TreeNode,
        ] = {}
        self._closed_component_factor_by_signature: dict[
            tuple[tuple[object, ...], ...],
            Decimal,
        ] = {}
        self._scaled_tree_by_node_and_factor: dict[
            tuple[int, Decimal],
            tuple[TreeNode, TreeNode],
        ] = {}

        for person_index in self._observed_person_indices:
            person = family.individuals[person_index]
            if person.individual_id in self.origins:
                self._constraints_by_person_index[person_index] = (
                    self._constraint_for_person(person)
                )

    def checkpoint(self) -> tuple[int, int, int]:
        """Return the prefix and journal size needed to restore this state."""

        return (
            self._assigned_bit_count,
            len(self._resolved_journal),
            len(self._peeled_constraint_journal),
        )

    def advance_to(self, assigned_bit_count: int) -> None:
        """Resolve origins newly implied by a longer inheritance prefix."""

        if not self._assigned_bit_count <= assigned_bit_count <= len(
            self.family.meioses
        ):
            raise ValueError(
                "Incremental inheritance traversal cannot move backwards."
            )

        previous_bit_count = self._assigned_bit_count
        self._assigned_bit_count = assigned_bit_count
        candidate_ids = (
            self.family.meioses[bit_index].child_id
            for bit_index in range(previous_bit_count, assigned_bit_count)
            if self.family.meioses[bit_index].child_id
            in self.relevant_individual_ids
        )
        self._resolve_available_people(candidate_ids)

    def rollback(self, checkpoint: tuple[int, int, int]) -> None:
        """Undo origins and constraints added after a recursion checkpoint."""

        assigned_bit_count, resolved_journal_size, peeled_journal_size = (
            checkpoint
        )
        if resolved_journal_size > len(self._resolved_journal):
            raise ValueError("Traversal checkpoint is newer than current state.")
        if peeled_journal_size > len(self._peeled_constraint_journal):
            raise ValueError("Traversal checkpoint is newer than current state.")

        while len(self._peeled_constraint_journal) > peeled_journal_size:
            person_index, constraint = self._peeled_constraint_journal.pop()
            self._constraints_by_person_index[person_index] = constraint

        while len(self._resolved_journal) > resolved_journal_size:
            person_id, observed_person_index = self._resolved_journal.pop()
            self.origins.pop(person_id)
            if observed_person_index is not None:
                self._constraints_by_person_index.pop(observed_person_index)
        self._assigned_bit_count = assigned_bit_count

    def constraints(self) -> tuple[_FounderGenotypeConstraint, ...]:
        """Return resolved genotype constraints in stable pedigree order."""

        return tuple(
            self._constraints_by_person_index[person_index]
            for person_index in self._observed_person_indices
            if person_index in self._constraints_by_person_index
        )

    def peel_closed_components(
        self,
        bit_index: int,
        allele_frequencies: dict[str, Decimal],
    ) -> Decimal:
        """Remove completed constraint components and return their factor.

        A component is safe to integrate only after none of its founder
        origins can participate in a future transmission. Removing it keeps
        completed pedigree history out of the canonical suffix-cache key. The
        journal makes that removal branch-local and exactly reversible.
        """

        live_origins = self._live_founder_origins(bit_index)
        closed_components = tuple(
            (variables, constraints)
            for variables, constraints in _constraint_components(
                self.constraints()
            )
            if live_origins.isdisjoint(variables)
        )
        if not closed_components:
            return Decimal(1)

        closed_constraint_ids = {
            id(constraint)
            for _, constraints in closed_components
            for constraint in constraints
        }
        for person_index in self._observed_person_indices:
            constraint = self._constraints_by_person_index.get(person_index)
            if (
                constraint is None
                or id(constraint) not in closed_constraint_ids
            ):
                continue
            self._peeled_constraint_journal.append((person_index, constraint))
            self._constraints_by_person_index.pop(person_index)

        component_factors: list[Decimal] = []
        for variables, constraints in closed_components:
            component_signature = _canonical_constraint_signature(
                constraints
            )
            factor = self._closed_component_factor_by_signature.get(
                component_signature
            )
            if factor is None:
                factor = _founder_component_likelihood(
                    variables,
                    constraints,
                    allele_frequencies,
                )
                self._closed_component_factor_by_signature[
                    component_signature
                ] = factor
                self._peeled_factor_cache_miss_count += 1
            else:
                self._peeled_factor_cache_hit_count += 1
            component_factors.append(factor)
            if factor == 0:
                self._zero_peeled_factor_count += 1

        self._peeled_component_count += len(closed_components)
        self._peeled_constraint_count += sum(
            len(constraints) for _, constraints in closed_components
        )
        return _decimal_product(component_factors)

    def apply_peeled_factor(
        self,
        node: TreeNode,
        factor: Decimal,
    ) -> TreeNode:
        """Scale a normalized suffix while preserving shared DAG structure."""

        if factor == 1 or isinstance(node, ZeroNode):
            return node
        cache_key = (id(node), factor)
        cached_entry = self._scaled_tree_by_node_and_factor.get(cache_key)
        if cached_entry is not None and cached_entry[0] is node:
            self._scaled_tree_cache_hit_count += 1
            return cached_entry[1]

        scaled_node = _scale_tree_by_decimal(node, factor)
        # Retaining the source object prevents a recycled Python object ID from
        # returning a scaled tree for an unrelated suffix node.
        self._scaled_tree_by_node_and_factor[cache_key] = (node, scaled_node)
        return scaled_node

    def _live_founder_origins(self, bit_index: int) -> set[AlleleOrigin]:
        """Return founder origins still required by unresolved transmissions."""

        live_origins = {
            origin
            for person_id in self._relevant_person_ids_in_order
            if (
                person_id in self.origins
                and self._last_outgoing_meiosis_index_by_parent_id.get(
                    person_id,
                    -1,
                )
                >= bit_index
            )
            for origin in self.origins[person_id]
        }
        for person_id in self._relevant_person_ids_in_order:
            if person_id in self.origins:
                continue
            paternal_index, maternal_index = (
                self._incoming_meiosis_indices[person_id]
            )
            person = self._people_by_id[person_id]
            parental_inputs = (
                (person.father_id, paternal_index),
                (person.mother_id, maternal_index),
            )
            for parent_id, incoming_index in parental_inputs:
                if incoming_index >= bit_index or parent_id is None:
                    continue
                parent_origins = self.origins.get(parent_id)
                if parent_origins is None:
                    continue
                selector = self.inheritance_bits[incoming_index]
                live_origins.add(parent_origins[selector])
        return live_origins

    def record_recursive_node(self, bit_index: int) -> None:
        """Record one expanded node and enforce diagnostic-only budgets."""

        self._recursive_node_count += 1
        self._maximum_recursion_depth = max(
            self._maximum_recursion_depth,
            bit_index,
        )

        reached_node_limit = (
            self._node_limit is not None
            and self._recursive_node_count >= self._node_limit
        )
        heartbeat_due = (
            self._heartbeat_node_interval is not None
            and self._recursive_node_count
            % self._heartbeat_node_interval
            == 0
        )
        time_check_due = (
            self._time_limit_seconds is not None
            and self._recursive_node_count
            % self._time_check_node_interval
            == 0
        )
        elapsed_seconds: float | None = None
        if reached_node_limit or heartbeat_due or time_check_due:
            elapsed_seconds = perf_counter() - self._traversal_start

        reached_time_limit = (
            time_check_due
            and elapsed_seconds is not None
            and self._time_limit_seconds is not None
            and elapsed_seconds >= self._time_limit_seconds
        )
        if heartbeat_due or reached_node_limit or reached_time_limit:
            self._report_progress(elapsed_seconds)

        if reached_node_limit:
            raise MarkerTreeBudgetExceeded(
                "Marker-tree node budget reached for "
                f"{self.marker_name!r}: {self._recursive_node_count} nodes."
            )
        if reached_time_limit:
            raise MarkerTreeBudgetExceeded(
                "Marker-tree time budget reached for "
                f"{self.marker_name!r}: {elapsed_seconds:.3f} seconds."
            )

    def record_contradiction_prune(self) -> None:
        """Record a partial genotype contradiction."""

        self._contradiction_prune_count += 1

    def record_founder_orientation_reduction(self) -> None:
        """Record one founder-orientation quotient."""

        self._founder_orientation_reduction_count += 1

    def record_founder_couple_reduction(self) -> None:
        """Record one founder-couple quotient."""

        self._founder_couple_reduction_count += 1

    def record_counting_reduction(self) -> None:
        """Record one exact untyped-chain counting quotient."""

        self._counting_reduction_count += 1

    def record_invariant_reduction(self) -> None:
        """Record one likelihood-invariant meiosis quotient."""

        self._invariant_reduction_count += 1


    def _report_progress(self, elapsed_seconds: float | None) -> None:
        """Emit one compact, flushed-by-caller diagnostic heartbeat."""

        if self._progress is None:
            return
        if elapsed_seconds is None:
            elapsed_seconds = perf_counter() - self._traversal_start
        self._progress(
            f"marker tree heartbeat\tmarker={self.marker_name}\t"
            f"seconds={elapsed_seconds:.3f}\t"
            f"nodes={self._recursive_node_count}\t"
            f"depth={self._maximum_recursion_depth}\t"
            f"cache_hits={self._suffix_cache_hits}\t"
            f"cache_misses={self._suffix_cache_misses}\t"
            f"prunes={self._contradiction_prune_count}\t"
            "founder_orientation="
            f"{self._founder_orientation_reduction_count}\t"
            f"founder_couples={self._founder_couple_reduction_count}\t"
            f"counting={self._counting_reduction_count}\t"
            f"invariant={self._invariant_reduction_count}\t"
            f"peeled_components={self._peeled_component_count}\t"
            f"peeled_constraints={self._peeled_constraint_count}\t"
            f"zero_peeled_factors={self._zero_peeled_factor_count}\t"
            f"normalized_cache_reuses={self._normalized_cache_reuse_count}\t"
            f"peeled_factor_cache_hits={self._peeled_factor_cache_hit_count}\t"
            f"peeled_factor_cache_misses={self._peeled_factor_cache_miss_count}\t"
            f"scaled_tree_cache_hits={self._scaled_tree_cache_hit_count}"
        )

    def require_complete(self) -> None:
        """Require every marker-relevant person to have resolved origins."""

        unresolved_ids = self.relevant_individual_ids.difference(self.origins)
        if unresolved_ids:
            raise ValueError(
                f"Could not topologically resolve family "
                f"{self.family.family_id!r}."
            )

    def canonical_future_key(
        self,
        bit_index: int,
    ) -> tuple[object, ...]:
        """Describe the exact future state up to founder-variable renaming.

        Founder allele variables are independent draws from the same marker
        frequencies. Their literal pedigree IDs therefore do not affect a
        suffix likelihood. Equality relationships do affect it. Assigning
        compact labels by stable first occurrence preserves those relations
        while allowing exchangeable histories to share one computed subtree.
        """

        canonical_index_by_origin: dict[AlleleOrigin, int] = {}

        def canonical_origin(origin: AlleleOrigin) -> int:
            if origin not in canonical_index_by_origin:
                canonical_index_by_origin[origin] = len(
                    canonical_index_by_origin
                )
            return canonical_index_by_origin[origin]

        frontier = []
        live_origins: set[AlleleOrigin] = set()
        for person_id in self._relevant_person_ids_in_order:
            origins = self.origins.get(person_id)
            if (
                origins is None
                or self._last_outgoing_meiosis_index_by_parent_id.get(
                    person_id,
                    -1,
                )
                < bit_index
            ):
                continue
            live_origins.update(origins)
            frontier.append(
                (
                    person_id,
                    canonical_origin(origins[0]),
                    canonical_origin(origins[1]),
                )
            )

        pending_transmissions = self._pending_transmission_frontier(
            bit_index,
            canonical_origin,
            live_origins,
        )
        canonical_constraint_components = (
            self._canonical_constraint_components(
                canonical_origin,
                live_origins,
            )
        )
        return (
            bit_index,
            tuple(frontier),
            pending_transmissions,
            canonical_constraint_components,
        )

    def _pending_transmission_frontier(
        self,
        bit_index: int,
        canonical_origin: Callable[[AlleleOrigin], int],
        live_origins: set[AlleleOrigin],
    ) -> tuple[tuple[object, ...], ...]:
        """Describe assigned inputs of children awaiting their other parent.

        A parent's final outgoing meiosis can precede the other parental
        meiosis for the same child. The selected allele remains part of the
        dynamic state until that child resolves, even though the parent is no
        longer in the future-transmission frontier. Recording that allele
        prevents distinct inheritance prefixes from sharing an invalid suffix.

        A deferred record covers pedigrees whose meioses are not topologically
        ordered. In that case the assigned selector must remain live until the
        unresolved parent acquires its own founder origins.
        """

        pending_transmissions: list[tuple[object, ...]] = []
        for person_id in self._relevant_person_ids_in_order:
            if person_id in self.origins:
                continue

            person = self._people_by_id[person_id]
            paternal_index, maternal_index = (
                self._incoming_meiosis_indices[person_id]
            )
            parental_inputs = (
                ("paternal", person.father_id, paternal_index),
                ("maternal", person.mother_id, maternal_index),
            )
            for parental_side, parent_id, incoming_index in parental_inputs:
                if incoming_index >= bit_index:
                    continue
                if parent_id is None:
                    raise ValueError(
                        "A nonfounder marker-relevant person requires two "
                        "parents."
                    )

                selector = self.inheritance_bits[incoming_index]
                parent_origins = self.origins.get(parent_id)
                if parent_origins is None:
                    pending_transmissions.append(
                        (
                            person_id,
                            parental_side,
                            parent_id,
                            "deferred",
                            selector,
                        )
                    )
                    continue

                transmitted_origin = parent_origins[selector]
                live_origins.add(transmitted_origin)
                pending_transmissions.append(
                    (
                        person_id,
                        parental_side,
                        parent_id,
                        "resolved",
                        canonical_origin(transmitted_origin),
                    )
                )

        return tuple(pending_transmissions)

    def _canonical_constraint_components(
        self,
        canonical_frontier_origin: Callable[[AlleleOrigin], int],
        live_origins: set[AlleleOrigin],
    ) -> tuple[tuple[tuple[object, ...], ...], ...]:
        """Canonicalize independent observed-origin graph components."""

        component_signatures = []
        for variables, component_constraints in _constraint_components(
            self.constraints()
        ):
            if live_origins.isdisjoint(variables):
                local_index_by_origin: dict[AlleleOrigin, int] = {}

                def canonical_origin(origin: AlleleOrigin) -> int:
                    if origin not in local_index_by_origin:
                        local_index_by_origin[origin] = len(
                            local_index_by_origin
                        )
                    return local_index_by_origin[origin]

            else:
                canonical_origin = canonical_frontier_origin
            component_signatures.append(
                tuple(
                    (
                        canonical_origin(constraint.first_origin),
                        canonical_origin(constraint.second_origin),
                        constraint.first_allele,
                        constraint.second_allele,
                    )
                    for constraint in component_constraints
                )
            )
        return tuple(sorted(component_signatures))

    def cached_suffix_tree(self, bit_index: int) -> TreeNode | None:
        """Return a previously scored equivalent suffix, when enabled."""

        if not self._enable_suffix_cache:
            return None
        canonical_state, current_scale = self._normalized_future_state(
            bit_index
        )
        cached_entry = self._suffix_tree_by_canonical_state.get(
            canonical_state
        )
        if cached_entry is None or current_scale == 0:
            self._suffix_cache_misses += 1
            return None

        cached_node, cached_scale = cached_entry
        self._suffix_cache_hits += 1
        self._normalized_cache_reuse_count += 1
        return self.apply_peeled_factor(
            cached_node,
            current_scale / cached_scale,
        )

    def cache_suffix_tree(self, bit_index: int, node: TreeNode) -> None:
        """Record an exact suffix result under its canonical frontier state."""

        if not self._enable_suffix_cache:
            return
        canonical_state, current_scale = self._normalized_future_state(
            bit_index
        )
        if current_scale == 0:
            return
        self._suffix_tree_by_canonical_state[canonical_state] = (
            node,
            current_scale,
        )

    def _normalized_future_state(
        self,
        bit_index: int,
    ) -> tuple[tuple[object, ...], Decimal]:
        """Return an exact proportional-potential key and its scale.

        Resolved genotype constraints induce a likelihood potential over the
        founder origins that remain live. Constraint histories with
        proportional potentials have proportional suffix trees, even when
        their literal constraint graphs differ. The exact rational key records
        their relative boundary probabilities. The separate Decimal scale
        restores the original likelihood magnitude on reuse.
        """

        raw_key = self.canonical_future_key(bit_index)
        if self._allele_frequencies is None:
            return raw_key, Decimal(1)
        cached_state = self._normalized_state_by_raw_key.get(raw_key)
        if cached_state is not None:
            return cached_state

        _, frontier, pending_transmissions, components = raw_key
        live_canonical_origins = {
            origin
            for _, first_origin, second_origin in frontier
            for origin in (first_origin, second_origin)
        }
        live_canonical_origins.update(
            transmission[-1]
            for transmission in pending_transmissions
            if transmission[-2] == "resolved"
        )
        potential_signatures = []
        component_scales = []
        for component in components:
            signature, scale = _normalized_component_potential(
                component,
                live_canonical_origins,
                self._positive_alleles,
                self._allele_frequencies,
            )
            potential_signatures.append(signature)
            component_scales.append(scale)

        normalized_state = (
            bit_index,
            frontier,
            pending_transmissions,
            tuple(sorted(potential_signatures)),
        )
        state_and_scale = (
            normalized_state,
            _decimal_product(component_scales),
        )
        self._normalized_state_by_raw_key[raw_key] = state_and_scale
        return state_and_scale

    def restore_shared_prefix(
        self,
        node: TreeNode,
        level_count: int,
    ) -> TreeNode:
        """Restore omitted invariant levels while reusing wrapper chains."""

        if level_count == 0:
            return node
        cache_key = (id(node), level_count)
        cached_node = self._shared_prefix_by_node_and_depth.get(cache_key)
        if cached_node is not None:
            return cached_node

        restored_node = node
        for _ in range(level_count):
            restored_node = _combine_children(restored_node, restored_node)
        self._shared_prefix_by_node_and_depth[cache_key] = restored_node
        return restored_node

    @property
    def suffix_cache_hits(self) -> int:
        """Return the number of canonical suffix states reused."""

        return self._suffix_cache_hits

    @property
    def suffix_cache_misses(self) -> int:
        """Return the number of canonical suffix states first encountered."""

        return self._suffix_cache_misses

    @property
    def cached_suffix_count(self) -> int:
        """Return the number of stored canonical suffix states."""

        return len(self._suffix_tree_by_canonical_state)

    @property
    def recursive_node_count(self) -> int:
        """Return the number of recursive marker-tree nodes expanded."""

        return self._recursive_node_count

    @property
    def maximum_recursion_depth(self) -> int:
        """Return the deepest inheritance-bit index visited."""

        return self._maximum_recursion_depth

    @property
    def contradiction_prune_count(self) -> int:
        """Return the number of incompatible partial branches pruned."""

        return self._contradiction_prune_count

    @property
    def founder_orientation_reduction_count(self) -> int:
        """Return the number of founder-orientation quotients applied."""

        return self._founder_orientation_reduction_count

    @property
    def founder_couple_reduction_count(self) -> int:
        """Return the number of founder-couple quotients applied."""

        return self._founder_couple_reduction_count

    @property
    def counting_reduction_count(self) -> int:
        """Return the number of untyped-chain counting quotients applied."""

        return self._counting_reduction_count

    @property
    def invariant_reduction_count(self) -> int:
        """Return the number of invariant meiosis quotients applied."""

        return self._invariant_reduction_count


    @property
    def peeled_component_count(self) -> int:
        """Return the number of closed components integrated."""

        return self._peeled_component_count

    @property
    def peeled_constraint_count(self) -> int:
        """Return the number of constraints removed after integration."""

        return self._peeled_constraint_count

    @property
    def zero_peeled_factor_count(self) -> int:
        """Return the number of incompatible closed components."""

        return self._zero_peeled_factor_count

    @property
    def normalized_cache_reuse_count(self) -> int:
        """Return the number of normalized future subtrees reused."""

        return self._normalized_cache_reuse_count

    @property
    def peeled_factor_cache_hit_count(self) -> int:
        """Return the number of repeated closed-component factors reused."""

        return self._peeled_factor_cache_hit_count

    @property
    def peeled_factor_cache_miss_count(self) -> int:
        """Return the number of distinct closed-component factors evaluated."""

        return self._peeled_factor_cache_miss_count

    @property
    def scaled_tree_cache_hit_count(self) -> int:
        """Return the number of repeated normalized-tree scales reused."""

        return self._scaled_tree_cache_hit_count

    def _build_incoming_meiosis_indices(
        self,
    ) -> dict[str, tuple[int, int]]:
        indices_by_child_and_parent: dict[str, dict[str, int]] = {}
        for bit_index, meiosis in enumerate(self.family.meioses):
            if meiosis.child_id not in self.relevant_individual_ids:
                continue
            indices_by_child_and_parent.setdefault(
                meiosis.child_id,
                {},
            )[meiosis.parent_id] = bit_index

        incoming_indices: dict[str, tuple[int, int]] = {}
        for person_id in self.relevant_individual_ids:
            person = self._people_by_id[person_id]
            if person.is_founder:
                continue
            parent_indices = indices_by_child_and_parent.get(person_id, {})
            if (
                person.father_id not in parent_indices
                or person.mother_id not in parent_indices
            ):
                raise ValueError(
                    f"Could not topologically resolve family "
                    f"{self.family.family_id!r}."
                )
            incoming_indices[person_id] = (
                parent_indices[person.father_id],
                parent_indices[person.mother_id],
            )
        return incoming_indices


    def _build_children_by_parent_id(self) -> dict[str, tuple[str, ...]]:
        children_by_parent_id: dict[str, list[str]] = {}
        for person in self.family.individuals:
            if (
                person.individual_id not in self.relevant_individual_ids
                or person.is_founder
            ):
                continue
            for parent_id in (person.father_id, person.mother_id):
                if parent_id is not None:
                    children_by_parent_id.setdefault(parent_id, []).append(
                        person.individual_id
                    )
        return {
            parent_id: tuple(child_ids)
            for parent_id, child_ids in children_by_parent_id.items()
        }

    def _build_outgoing_meiosis_indices_by_parent_id(
        self,
    ) -> dict[str, tuple[int, ...]]:
        """Index future relevant transmissions for frontier memoization."""

        indices_by_parent_id: dict[str, list[int]] = {}
        for bit_index, meiosis in enumerate(self.family.meioses):
            if meiosis.child_id not in self.relevant_individual_ids:
                continue
            indices_by_parent_id.setdefault(meiosis.parent_id, []).append(
                bit_index
            )
        return {
            parent_id: tuple(bit_indices)
            for parent_id, bit_indices in indices_by_parent_id.items()
        }

    def _resolve_available_people(self, candidate_ids: Iterable[str]) -> None:
        unresolved_queue = deque(candidate_ids)
        queued_ids = set(unresolved_queue)

        while unresolved_queue:
            person_id = unresolved_queue.popleft()
            queued_ids.discard(person_id)
            if person_id in self.origins:
                continue

            person = self._people_by_id[person_id]
            paternal_index, maternal_index = (
                self._incoming_meiosis_indices[person_id]
            )
            if (
                paternal_index >= self._assigned_bit_count
                or maternal_index >= self._assigned_bit_count
                or person.father_id not in self.origins
                or person.mother_id not in self.origins
            ):
                continue

            paternal_origin = self.origins[person.father_id][
                self.inheritance_bits[paternal_index]
            ]
            maternal_origin = self.origins[person.mother_id][
                self.inheritance_bits[maternal_index]
            ]
            self.origins[person_id] = (paternal_origin, maternal_origin)

            person_index = self._person_index_by_id[person_id]
            observed_person_index: int | None = None
            if person_index in self._observed_person_index_set:
                self._constraints_by_person_index[person_index] = (
                    self._constraint_for_person(person)
                )
                observed_person_index = person_index
            self._resolved_journal.append((person_id, observed_person_index))

            for child_id in self._children_by_parent_id.get(person_id, ()):
                if child_id not in self.origins and child_id not in queued_ids:
                    unresolved_queue.append(child_id)
                    queued_ids.add(child_id)

    def _constraint_for_person(
        self,
        person: Individual,
    ) -> _FounderGenotypeConstraint:
        first_allele, second_allele = person.genotypes[self.marker_name]
        if first_allele is None or second_allele is None:
            raise ValueError("A marker constraint requires a complete genotype.")
        first_origin, second_origin = self.origins[person.individual_id]
        return _FounderGenotypeConstraint(
            first_origin=first_origin,
            second_origin=second_origin,
            first_allele=first_allele,
            second_allele=second_allele,
        )


def single_marker_likelihood(
    dataset: Dataset,
    marker_id: str,
    backend: str = "numpy",
    workers: int = 1,
) -> LikelihoodResult:
    """Compute posterior inheritance-state weights for a single marker."""

    if backend != "numpy":
        raise NotImplementedError("Only the NumPy reference backend is implemented.")
    workers = validate_workers(workers)

    marker = dataset.marker_by_name[marker_id]
    all_states: list[InheritanceState] = []
    total_likelihood = 1.0

    if workers == 1:
        family_state_groups = tuple(
            _score_family_marker(family, marker)
            for family in dataset.families
        )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            family_state_groups = tuple(
                executor.map(
                    _score_family_marker,
                    dataset.families,
                    repeat(marker),
                )
            )

    for family_states in family_state_groups:
        family_likelihood = fsum(state.likelihood for state in family_states)
        if family_likelihood == 0.0:
            total_likelihood = 0.0
            all_states.extend(family_states)
            continue
        total_likelihood *= family_likelihood
        all_states.extend(
            InheritanceState(
                family_id=state.family_id,
                bits=state.bits,
                likelihood=state.likelihood,
                posterior_weight=state.likelihood / family_likelihood,
                allele_origins=state.allele_origins,
            )
            for state in family_states
        )

    return LikelihoodResult(
        marker=marker,
        states=tuple(all_states),
        likelihood=total_likelihood,
    )


def family_marker_likelihood_tree(
    family: Family,
    marker: Marker,
    *,
    progress: Callable[[str], None] | None = None,
    heartbeat_node_interval: int | None = None,
    node_limit: int | None = None,
    time_limit_seconds: float | None = None,
) -> InheritanceTree:
    """Score one family-marker likelihood as a compressed inheritance tree.

    This opt-in path uses component-wise founder peeling while the explicit
    reference engine retains global assignment enumeration. It compresses
    branches while recursion unwinds, so it avoids retaining the complete
    dense inheritance-state table.
    """

    inheritance_bits = [0] * len(family.meioses)
    marker_relevant_individual_ids = _marker_relevant_individual_ids(
        family,
        marker.name,
    )
    marker_relevant_meiosis_indices = _marker_relevant_meiosis_indices(
        family,
        marker_relevant_individual_ids,
    )
    founder_symmetry_plan = build_founder_orientation_symmetry_plan(
        family,
        marker_relevant_meiosis_indices,
    )
    founder_couple_symmetry_plan = build_founder_couple_symmetry_plan(
        family,
        marker_relevant_meiosis_indices,
    )
    founder_couple_representative_indices = (
        founder_couple_symmetry_plan.representative_bit_indices
    )
    counting_chains = tuple(
        chain
        for chain in detect_untyped_chains(family)
        if set(chain.selector_bit_indices).issubset(
            marker_relevant_meiosis_indices
        )
    )
    counting_chain_by_first_selector = {
        chain.selector_bit_indices[0]: chain for chain in counting_chains
    }
    fixed_counting_selector_indices = frozenset(
        bit_index
        for chain in counting_chains
        for bit_index in chain.selector_bit_indices[1:]
    )
    satisfiability_cache: dict[
        tuple[tuple[object, ...], ...],
        bool,
    ] = {}
    likelihood_cache: dict[
        tuple[_FounderGenotypeConstraint, ...],
        float,
    ] = {}

    with localcontext() as decimal_context:
        decimal_context.prec = _DECIMAL_PRECISION
        allele_frequencies = _decimal_allele_frequencies(family, marker)
        positive_alleles = tuple(
            allele
            for allele, frequency in allele_frequencies.items()
            if frequency > 0
        )
        traversal_state = _MarkerTraversalState(
            family,
            marker.name,
            inheritance_bits,
            marker_relevant_individual_ids,
            allele_frequencies=allele_frequencies,
            positive_alleles=positive_alleles,
            # Count representatives preassign selector bits beyond the current
            # prefix. Those fixed future values need a richer cache key before
            # the general suffix cache can safely share their subtrees.
            enable_suffix_cache=not counting_chains,
            progress=progress,
            heartbeat_node_interval=heartbeat_node_interval,
            node_limit=node_limit,
            time_limit_seconds=time_limit_seconds,
        )
        with _inheritance_recursion_budget(len(family.meioses)):
            root = _score_family_marker_tree_node(
                traversal_state,
                bit_index=0,
                allele_frequencies=allele_frequencies,
                positive_alleles=positive_alleles,
                satisfiability_cache=satisfiability_cache,
                likelihood_cache=likelihood_cache,
                marker_relevant_meiosis_indices=marker_relevant_meiosis_indices,
                founder_symmetry_plan=founder_symmetry_plan,
                founder_couple_representative_indices=(
                    founder_couple_representative_indices
                ),
                counting_chain_by_first_selector=(
                    counting_chain_by_first_selector
                ),
                fixed_counting_selector_indices=(
                    fixed_counting_selector_indices
                ),
            )

    with _inheritance_recursion_budget(len(family.meioses)):
        root = restore_founder_couple_symmetry_branches(
            root,
            founder_couple_symmetry_plan,
        )
        root = _materialize_scaled_tree(root)

    if isinstance(root, ZeroNode):
        # MERLIN treats a marker-level Mendelian incompatibility as
        # uninformative for this family, matching _score_family_marker().
        root = LeafNode(1.0)

    return InheritanceTree(
        bit_count=len(family.meioses),
        root=root,
        build_statistics=TreeBuildStatistics(
            relevant_individual_count=len(marker_relevant_individual_ids),
            relevant_meiosis_count=len(marker_relevant_meiosis_indices),
            suffix_cache_hits=traversal_state.suffix_cache_hits,
            suffix_cache_misses=traversal_state.suffix_cache_misses,
            cached_suffix_count=traversal_state.cached_suffix_count,
            counting_chain_count=len(counting_chains),
            counted_selector_count=sum(
                chain.selector_count for chain in counting_chains
            ),
            recursive_node_count=traversal_state.recursive_node_count,
            maximum_recursion_depth=(
                traversal_state.maximum_recursion_depth
            ),
            contradiction_prune_count=(
                traversal_state.contradiction_prune_count
            ),
            founder_orientation_reduction_count=(
                traversal_state.founder_orientation_reduction_count
            ),
            founder_couple_reduction_count=(
                traversal_state.founder_couple_reduction_count
            ),
            counting_reduction_count=(
                traversal_state.counting_reduction_count
            ),
            invariant_reduction_count=(
                traversal_state.invariant_reduction_count
            ),
            peeled_component_count=(
                traversal_state.peeled_component_count
            ),
            peeled_constraint_count=(
                traversal_state.peeled_constraint_count
            ),
            zero_peeled_factor_count=(
                traversal_state.zero_peeled_factor_count
            ),
            normalized_cache_reuse_count=(
                traversal_state.normalized_cache_reuse_count
            ),
            peeled_factor_cache_hit_count=(
                traversal_state.peeled_factor_cache_hit_count
            ),
            peeled_factor_cache_miss_count=(
                traversal_state.peeled_factor_cache_miss_count
            ),
            scaled_tree_cache_hit_count=(
                traversal_state.scaled_tree_cache_hit_count
            ),
        ),
    )


def _score_family_marker_tree_node(
    traversal_state: _MarkerTraversalState,
    bit_index: int,
    allele_frequencies: dict[str, Decimal],
    positive_alleles: tuple[str, ...],
    satisfiability_cache: dict[
        tuple[tuple[object, ...], ...],
        bool,
    ],
    likelihood_cache: dict[
        tuple[_FounderGenotypeConstraint, ...],
        float,
    ],
    marker_relevant_meiosis_indices: frozenset[int],
    founder_symmetry_plan: FounderOrientationSymmetryPlan,
    founder_couple_representative_indices: frozenset[int],
    counting_chain_by_first_selector: dict[int, UntypedChain],
    fixed_counting_selector_indices: frozenset[int],
) -> TreeNode:
    """Recursively score, prune, and compress one inheritance-bit subtree."""

    family = traversal_state.family
    inheritance_bits = traversal_state.inheritance_bits
    checkpoint = traversal_state.checkpoint()
    skipped_bit_count = 0
    while (
        bit_index < len(inheritance_bits)
        and bit_index not in marker_relevant_meiosis_indices
    ):
        # Ancestor closure guarantees that this transmission cannot influence
        # any observed genotype. Fixing it to zero avoids rebuilding origins
        # for an irrelevant level. The invariant levels are restored below.
        inheritance_bits[bit_index] = 0
        bit_index += 1
        skipped_bit_count += 1

    traversal_state.record_recursive_node(bit_index)
    traversal_state.advance_to(bit_index)
    try:
        peeled_factor = traversal_state.peel_closed_components(
            bit_index,
            allele_frequencies,
        )
        partial_constraints = traversal_state.constraints()
        cached_node: TreeNode | None = None
        if peeled_factor == 0:
            # A completed incompatible component makes only this history zero.
            # Caching that zero under the open future state would poison other
            # histories whose completed factors differ.
            node: TreeNode = ZeroNode()
        elif (
            cached_node := traversal_state.cached_suffix_tree(bit_index)
        ) is not None:
            node = cached_node
        elif not _founder_constraints_are_satisfiable(
            partial_constraints,
            positive_alleles,
            satisfiability_cache,
        ):
            # A contradiction among already resolved people cannot be repaired
            # by assigning transmissions farther down this branch.
            traversal_state.record_contradiction_prune()
            node = ZeroNode()
        elif bit_index == len(inheritance_bits):
            traversal_state.require_complete()
            likelihood = _cached_peeled_constraints_likelihood(
                partial_constraints,
                allele_frequencies,
                likelihood_cache,
            )
            node = (
                ZeroNode()
                if likelihood == 0.0
                else LeafNode(likelihood)
            )
        elif counting_chain := counting_chain_by_first_selector.get(
            bit_index
        ):
            traversal_state.record_counting_reduction()
            partition = CountingPartition.from_chain(counting_chain)
            class_children = []
            for representative in partition.representative_vectors:
                for selector_bit_index, selector_value in zip(
                    counting_chain.selector_bit_indices,
                    representative,
                ):
                    inheritance_bits[selector_bit_index] = selector_value
                class_children.append(
                    _score_family_marker_tree_node(
                        traversal_state,
                        bit_index + 1,
                        allele_frequencies,
                        positive_alleles,
                        satisfiability_cache,
                        likelihood_cache,
                        marker_relevant_meiosis_indices,
                        founder_symmetry_plan,
                        founder_couple_representative_indices,
                        counting_chain_by_first_selector,
                        fixed_counting_selector_indices,
                    )
                )
            first_on_value = counting_chain.on_values[0]
            remaining_on_values = dict(
                zip(
                    counting_chain.selector_bit_indices[1:],
                    counting_chain.on_values[1:],
                )
            )
            zero_child = _restore_counting_class_nodes(
                tuple(class_children),
                bit_index + 1,
                len(inheritance_bits),
                int(first_on_value == 0),
                remaining_on_values,
            )
            one_child = _restore_counting_class_nodes(
                tuple(class_children),
                bit_index + 1,
                len(inheritance_bits),
                int(first_on_value == 1),
                remaining_on_values,
            )
            node = _combine_children(zero_child, one_child)
        elif bit_index in fixed_counting_selector_indices:
            child = _score_family_marker_tree_node(
                traversal_state,
                bit_index + 1,
                allele_frequencies,
                positive_alleles,
                satisfiability_cache,
                likelihood_cache,
                marker_relevant_meiosis_indices,
                founder_symmetry_plan,
                founder_couple_representative_indices,
                counting_chain_by_first_selector,
                fixed_counting_selector_indices,
            )
            node = _combine_children(child, child)
        elif (
            bit_index
            in founder_couple_representative_indices
        ):
            traversal_state.record_founder_couple_reduction()
            inheritance_bits[bit_index] = 0
            canonical_child = _score_family_marker_tree_node(
                traversal_state,
                bit_index + 1,
                allele_frequencies,
                positive_alleles,
                satisfiability_cache,
                likelihood_cache,
                marker_relevant_meiosis_indices,
                founder_symmetry_plan,
                founder_couple_representative_indices,
                counting_chain_by_first_selector,
                fixed_counting_selector_indices,
            )
            node = _combine_children(canonical_child, canonical_child)
        elif (
            founder_flip_indices := (
                founder_symmetry_plan.descendant_flip_indices(bit_index)
            )
        ) is not None:
            # Founder allele labels are exchangeable. Score one orientation,
            # then reconstruct the complementary full-bit branch by swapping
            # this founder's later transmissions.
            traversal_state.record_founder_orientation_reduction()
            inheritance_bits[bit_index] = 0
            canonical_child = _score_family_marker_tree_node(
                traversal_state,
                bit_index + 1,
                allele_frequencies,
                positive_alleles,
                satisfiability_cache,
                likelihood_cache,
                marker_relevant_meiosis_indices,
                founder_symmetry_plan,
                founder_couple_representative_indices,
                counting_chain_by_first_selector,
                fixed_counting_selector_indices,
            )
            node = restore_founder_orientation_branch(
                canonical_child,
                bit_index,
                founder_flip_indices,
            )
        elif _meiosis_is_likelihood_invariant(
            family,
            traversal_state.origins,
            bit_index,
            traversal_state.relevant_individual_ids,
        ):
            traversal_state.record_invariant_reduction()
            inheritance_bits[bit_index] = 0
            child = _score_family_marker_tree_node(
                traversal_state,
                bit_index + 1,
                allele_frequencies,
                positive_alleles,
                satisfiability_cache,
                likelihood_cache,
                marker_relevant_meiosis_indices,
                founder_symmetry_plan,
                founder_couple_representative_indices,
                counting_chain_by_first_selector,
                fixed_counting_selector_indices,
            )
            node = _combine_children(child, child)
        else:
            inheritance_bits[bit_index] = 0
            zero_child = _score_family_marker_tree_node(
                traversal_state,
                bit_index + 1,
                allele_frequencies,
                positive_alleles,
                satisfiability_cache,
                likelihood_cache,
                marker_relevant_meiosis_indices,
                founder_symmetry_plan,
                founder_couple_representative_indices,
                counting_chain_by_first_selector,
                fixed_counting_selector_indices,
            )
            inheritance_bits[bit_index] = 1
            one_child = _score_family_marker_tree_node(
                traversal_state,
                bit_index + 1,
                allele_frequencies,
                positive_alleles,
                satisfiability_cache,
                likelihood_cache,
                marker_relevant_meiosis_indices,
                founder_symmetry_plan,
                founder_couple_representative_indices,
                counting_chain_by_first_selector,
                fixed_counting_selector_indices,
            )
            node = _combine_children(zero_child, one_child)

        if cached_node is None:
            # Cache the suffix before restoring irrelevant prefix levels. The
            # canonical key describes only the open state at the advanced bit
            # index. A zero completed factor is history-specific and must not
            # be stored as though it were part of that normalized future.
            if peeled_factor != 0:
                traversal_state.cache_suffix_tree(bit_index, node)

        node = traversal_state.apply_peeled_factor(node, peeled_factor)

        return traversal_state.restore_shared_prefix(
            node,
            skipped_bit_count,
        )
    finally:
        traversal_state.rollback(checkpoint)


def _marker_relevant_individual_ids(
    family: Family,
    marker_name: str,
) -> frozenset[str]:
    """Return genotyped people and ancestors who can affect their alleles."""

    people_by_id = family.by_id
    relevant_ids = {
        person.individual_id
        for person in family.individuals
        if _has_complete_marker_genotype(person, marker_name)
    }
    unresolved_ancestors = list(relevant_ids)

    while unresolved_ancestors:
        person = people_by_id[unresolved_ancestors.pop()]
        for parent_id in (person.father_id, person.mother_id):
            if parent_id not in people_by_id or parent_id in relevant_ids:
                continue
            relevant_ids.add(parent_id)
            unresolved_ancestors.append(parent_id)

    return frozenset(relevant_ids)


def _has_complete_marker_genotype(
    person: Individual,
    marker_name: str,
) -> bool:
    """Return whether both marker alleles are observed for one person."""

    return all(
        allele is not None
        for allele in person.genotypes.get(marker_name, (None, None))
    )


def _marker_relevant_meiosis_indices(
    family: Family,
    marker_relevant_individual_ids: frozenset[str],
) -> frozenset[int]:
    """Return transmissions that can influence an observed marker genotype."""

    return frozenset(
        bit_index
        for bit_index, meiosis in enumerate(family.meioses)
        if meiosis.child_id in marker_relevant_individual_ids
    )


def _meiosis_is_likelihood_invariant(
    family: Family,
    partial_origins: dict[
        str,
        tuple[AlleleOrigin, AlleleOrigin],
    ],
    bit_index: int,
    marker_relevant_individual_ids: frozenset[str],
) -> bool:
    """Return whether both choices for one meiosis have identical effects."""

    meiosis = family.meioses[bit_index]
    if meiosis.child_id not in marker_relevant_individual_ids:
        return True

    parent_origins = partial_origins.get(meiosis.parent_id)
    return parent_origins is not None and parent_origins[0] == parent_origins[1]


def _partial_inheritance_origins(
    family: Family,
    inheritance_bits: list[int],
    assigned_bit_count: int,
) -> dict[str, tuple[AlleleOrigin, AlleleOrigin]]:
    """Propagate origins whose required inheritance bits are already assigned."""

    if not 0 <= assigned_bit_count <= len(family.meioses):
        raise ValueError("Assigned inheritance-bit count is outside the family.")

    transmitted = {
        (meiosis.parent_id, meiosis.child_id): inheritance_bits[index]
        for index, meiosis in enumerate(family.meioses[:assigned_bit_count])
    }
    origins = {
        founder.individual_id: (
            (founder.individual_id, 0),
            (founder.individual_id, 1),
        )
        for founder in family.founders
    }
    unresolved_ids = {
        person.individual_id
        for person in family.individuals
        if not person.is_founder
    }
    people_by_id = family.by_id

    progressed = True
    while progressed:
        progressed = False
        for person_id in tuple(unresolved_ids):
            person = people_by_id[person_id]
            paternal_key = (person.father_id, person_id)
            maternal_key = (person.mother_id, person_id)
            if (
                person.father_id not in origins
                or person.mother_id not in origins
                or paternal_key not in transmitted
                or maternal_key not in transmitted
            ):
                continue

            paternal_origin = origins[person.father_id][
                transmitted[paternal_key]
            ]
            maternal_origin = origins[person.mother_id][
                transmitted[maternal_key]
            ]
            origins[person_id] = (paternal_origin, maternal_origin)
            unresolved_ids.remove(person_id)
            progressed = True

    return origins


def _score_family_marker(
    family: Family,
    marker: Marker,
    *,
    ignored_individual_id: str | None = None,
    use_uninformative_fallback: bool = True,
    family_state_space: FamilyStateSpace | None = None,
    marker_assignment_space: MarkerAssignmentSpace | None = None,
) -> list[InheritanceState]:
    state_space = (
        _build_family_state_space(family)
        if family_state_space is None
        else family_state_space
    )
    assignment_space = (
        _build_marker_assignment_space(family, marker)
        if marker_assignment_space is None
        else marker_assignment_space
    )
    states: list[InheritanceState] = []

    for bits, origins in zip(state_space.bits, state_space.allele_origins):
        likelihood = _evaluate_state(
            family,
            marker,
            origins,
            assignment_space,
            ignored_individual_id=ignored_individual_id,
        )
        if likelihood > 0.0:
            states.append(
                InheritanceState(
                    family_id=family.family_id,
                    bits=tuple(bits),
                    likelihood=likelihood,
                    posterior_weight=0.0,
                    allele_origins=origins,
                )
            )

    if states or not use_uninformative_fallback:
        return states

    # MERLIN treats a family-marker Mendelian incompatibility as uninformative
    # for that family only. Keeping every inheritance vector with equal
    # likelihood preserves the remaining families and markers in the analysis.
    return _uninformative_family_states(family, state_space)


def _uninformative_family_states(
    family: Family,
    family_state_space: FamilyStateSpace | None = None,
) -> list[InheritanceState]:
    """Return the uniform inheritance distribution used by MERLIN's fallback."""

    state_space = (
        _build_family_state_space(family)
        if family_state_space is None
        else family_state_space
    )
    return [
        InheritanceState(
            family_id=family.family_id,
            bits=bits,
            likelihood=1.0,
            posterior_weight=0.0,
            allele_origins=origins,
        )
        for bits, origins in zip(
            state_space.bits,
            state_space.allele_origins,
        )
    ]


def _score_family_markers(
    family: Family,
    markers: tuple[Marker, ...],
) -> tuple[tuple[InheritanceState, ...], ...]:
    """Score all markers while reusing topology-derived inheritance origins."""

    family_state_space = _build_family_state_space(family)
    return tuple(
        tuple(
            _score_family_marker(
                family,
                marker,
                family_state_space=family_state_space,
            )
        )
        for marker in markers
    )


def _build_family_state_space(family: Family) -> FamilyStateSpace:
    """Build inheritance origins once for every vector in a family."""

    bits = tuple(product((0, 1), repeat=len(family.meioses)))
    return FamilyStateSpace(
        bits=bits,
        allele_origins=tuple(
            inheritance_origins(family, inheritance_bits)
            for inheritance_bits in bits
        ),
    )


def _build_marker_assignment_space(
    family: Family,
    marker: Marker,
) -> MarkerAssignmentSpace:
    """Build founder assignments once for reuse by all inheritance vectors."""

    allele_frequencies = _allele_frequencies(family, marker)
    founder_slots = tuple(
        (founder.individual_id, copy_index)
        for founder in family.founders
        for copy_index in (0, 1)
    )
    assignment_values = tuple(
        product(tuple(allele_frequencies), repeat=len(founder_slots))
    )
    assignments = tuple(
        dict(zip(founder_slots, values)) for values in assignment_values
    )
    probabilities = np.asarray(
        [
            _product(allele_frequencies[allele] for allele in values)
            for values in assignment_values
        ],
        dtype=np.float64,
    )
    probabilities.setflags(write=False)
    return MarkerAssignmentSpace(
        assignments=assignments,
        probabilities=probabilities,
    )


def _evaluate_state(
    family: Family,
    marker: Marker,
    origins: dict[str, tuple[AlleleOrigin, AlleleOrigin]],
    assignment_space: MarkerAssignmentSpace,
    *,
    ignored_individual_id: str | None = None,
) -> float:
    compatible_assignment_probabilities: list[float] = []

    for founder_assignment, assignment_probability in zip(
        assignment_space.assignments,
        assignment_space.probabilities,
    ):
        if assignment_probability == 0.0:
            continue
        assigned_alleles = _propagate_alleles(origins, founder_assignment)
        if _family_genotypes_match(
            family,
            marker.name,
            assigned_alleles,
            ignored_individual_id=ignored_individual_id,
        ):
            compatible_assignment_probabilities.append(float(assignment_probability))

    return fsum(compatible_assignment_probabilities)


def peeled_state_likelihood(
    family: Family,
    marker: Marker,
    allele_origins: dict[
        str,
        tuple[AlleleOrigin, AlleleOrigin],
    ],
) -> float:
    """Evaluate one inheritance vector by peeling founder constraints.

    Missing genotypes add no constraint, so their founder allele variables
    integrate to one and never enter the enumerated components. Observed
    genotypes connect only the origins they reference. Independent connected
    components can therefore be summed separately and multiplied afterward.
    """

    constraints = _founder_genotype_constraints(
        family,
        marker.name,
        allele_origins,
    )

    with localcontext() as decimal_context:
        decimal_context.prec = _DECIMAL_PRECISION
        allele_frequencies = _decimal_allele_frequencies(family, marker)
        return _peeled_constraints_likelihood(
            constraints,
            allele_frequencies,
        )


def _cached_peeled_constraints_likelihood(
    constraints: tuple[_FounderGenotypeConstraint, ...],
    allele_frequencies: dict[str, Decimal],
    cache: dict[tuple[_FounderGenotypeConstraint, ...], float],
) -> float:
    """Evaluate each canonical constraint state at most once per marker."""

    if constraints not in cache:
        cache[constraints] = _peeled_constraints_likelihood(
            constraints,
            allele_frequencies,
        )
    return cache[constraints]


def _peeled_constraints_likelihood(
    constraints: tuple[_FounderGenotypeConstraint, ...],
    allele_frequencies: dict[str, Decimal],
) -> float:
    """Integrate one stable set of founder-origin genotype constraints."""

    if not constraints:
        return 1.0

    component_likelihoods: list[Decimal] = []
    for variables, component_constraints in _constraint_components(
        constraints
    ):
        component_likelihood = _founder_component_likelihood(
            variables,
            component_constraints,
            allele_frequencies,
        )
        if component_likelihood == 0:
            return 0.0
        component_likelihoods.append(component_likelihood)

    # One final float conversion avoids compounding binary rounding between
    # independent components. Eighty decimal digits slightly exceed the
    # 256-bit precision used by the independent MPFR test oracle.
    return float(_decimal_product(component_likelihoods))


def _scale_tree_by_decimal(
    node: TreeNode,
    factor: Decimal,
    memo: dict[int, TreeNode] | None = None,
) -> TreeNode:
    """Attach one exact build-time scale without copying a shared subtree.

    Scales are materialized once after all emission-tree symmetries have been
    restored. Delaying that work prevents proportional suffix-cache hits from
    rebuilding the same shared DAG for every constraint-history scale.
    """

    return _scaled_node(node, factor)


def _founder_genotype_constraints(
    family: Family,
    marker_name: str,
    allele_origins: dict[
        str,
        tuple[AlleleOrigin, AlleleOrigin],
    ],
    *,
    allow_partial: bool = False,
) -> tuple[_FounderGenotypeConstraint, ...]:
    constraints = []
    for person in family.individuals:
        first_allele, second_allele = person.genotypes.get(
            marker_name,
            (None, None),
        )
        if first_allele is None or second_allele is None:
            continue
        if allow_partial and person.individual_id not in allele_origins:
            continue
        first_origin, second_origin = allele_origins[person.individual_id]
        constraints.append(
            _FounderGenotypeConstraint(
                first_origin=first_origin,
                second_origin=second_origin,
                first_allele=first_allele,
                second_allele=second_allele,
            )
        )
    return tuple(constraints)


def _founder_constraints_are_satisfiable(
    constraints: tuple[_FounderGenotypeConstraint, ...],
    positive_alleles: tuple[str, ...],
    cache: dict[tuple[tuple[object, ...], ...], bool],
) -> bool:
    """Return whether discrete founder alleles can satisfy the constraints."""

    canonical_signature = _canonical_constraint_signature(constraints)
    if canonical_signature in cache:
        return cache[canonical_signature]

    is_satisfiable = all(
        next(
            _compatible_founder_component_assignments(
                variables,
                component_constraints,
                positive_alleles,
            ),
            None,
        )
        is not None
        for variables, component_constraints in _constraint_components(
            constraints
        )
    )
    cache[canonical_signature] = is_satisfiable
    return is_satisfiable


def _canonical_constraint_signature(
    constraints: tuple[_FounderGenotypeConstraint, ...],
) -> tuple[tuple[object, ...], ...]:
    """Describe constraint equality patterns independent of founder labels."""

    canonical_index_by_origin: dict[FounderAllele, int] = {}

    def canonical_origin(origin: FounderAllele) -> int:
        if origin not in canonical_index_by_origin:
            canonical_index_by_origin[origin] = len(canonical_index_by_origin)
        return canonical_index_by_origin[origin]

    return tuple(
        (
            canonical_origin(constraint.first_origin),
            canonical_origin(constraint.second_origin),
            constraint.first_allele,
            constraint.second_allele,
        )
        for constraint in constraints
    )


def _normalized_component_potential(
    canonical_component: tuple[tuple[object, ...], ...],
    live_canonical_origins: set[int],
    positive_alleles: tuple[str, ...],
    allele_frequencies: dict[str, Decimal],
) -> tuple[tuple[object, ...], Decimal]:
    """Normalize one open component using exact rational probabilities.

    Decimal marker frequencies are exact rational numbers. Evaluating the
    boundary table as Fractions allows histories to share a cache key whenever
    their potentials are exactly proportional, including when several
    internal assignments contribute to one boundary state. The separate
    Decimal reference value supplies the likelihood scale restored on reuse.
    """

    canonical_constraints = tuple(
        _FounderGenotypeConstraint(
            first_origin=constraint[0],
            second_origin=constraint[1],
            first_allele=constraint[2],
            second_allele=constraint[3],
        )
        for constraint in canonical_component
    )
    variables = tuple(
        dict.fromkeys(
            origin
            for constraint in canonical_constraints
            for origin in (
                constraint.first_origin,
                constraint.second_origin,
            )
        )
    )
    boundary_variables = tuple(
        sorted(live_canonical_origins.intersection(variables))
    )
    if not boundary_variables:
        raise ValueError(
            "A normalized suffix-cache component must have a live boundary."
        )
    boundary_variable_set = set(boundary_variables)
    internal_variables = tuple(
        variable
        for variable in variables
        if variable not in boundary_variable_set
    )
    probability_by_boundary_assignment: dict[
        tuple[str, ...],
        Fraction,
    ] = {}
    for assignment in _compatible_founder_component_assignments(
        variables,
        canonical_constraints,
        positive_alleles,
    ):
        boundary_assignment = tuple(
            assignment[variable] for variable in boundary_variables
        )
        internal_probability = _fraction_product(
            Fraction(allele_frequencies[assignment[variable]])
            for variable in internal_variables
        )
        probability_by_boundary_assignment[boundary_assignment] = (
            probability_by_boundary_assignment.get(
                boundary_assignment,
                Fraction(0),
            )
            + internal_probability
        )

    ordered_probabilities = tuple(
        probability_by_boundary_assignment.get(
            boundary_assignment,
            Fraction(0),
        )
        for boundary_assignment in product(
            positive_alleles,
            repeat=len(boundary_variables),
        )
    )
    reference_probability = next(
        (
            probability
            for probability in ordered_probabilities
            if probability
        ),
        None,
    )
    if reference_probability is None:
        return (boundary_variables, ordered_probabilities), Decimal(0)

    normalized_probabilities = tuple(
        probability / reference_probability
        for probability in ordered_probabilities
    )
    reference_scale = (
        Decimal(reference_probability.numerator)
        / Decimal(reference_probability.denominator)
    )
    return (
        (boundary_variables, normalized_probabilities),
        reference_scale,
    )


def _constraint_components(
    constraints: tuple[_FounderGenotypeConstraint, ...],
) -> tuple[
    tuple[
        tuple[FounderAllele, ...],
        tuple[_FounderGenotypeConstraint, ...],
    ],
    ...,
]:
    """Group constraints by connected founder-allele origins."""

    parents: dict[FounderAllele, FounderAllele] = {}
    variable_order: list[FounderAllele] = []

    def add_variable(variable: FounderAllele) -> None:
        if variable not in parents:
            parents[variable] = variable
            variable_order.append(variable)

    def find_root(variable: FounderAllele) -> FounderAllele:
        root = variable
        while parents[root] != root:
            root = parents[root]
        while parents[variable] != variable:
            parent = parents[variable]
            parents[variable] = root
            variable = parent
        return root

    def join(first: FounderAllele, second: FounderAllele) -> None:
        first_root = find_root(first)
        second_root = find_root(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for constraint in constraints:
        add_variable(constraint.first_origin)
        add_variable(constraint.second_origin)
        join(constraint.first_origin, constraint.second_origin)

    variables_by_root: dict[FounderAllele, list[FounderAllele]] = {}
    for variable in variable_order:
        root = find_root(variable)
        variables_by_root.setdefault(root, []).append(variable)

    constraints_by_root: dict[
        FounderAllele,
        list[_FounderGenotypeConstraint],
    ] = {root: [] for root in variables_by_root}
    for constraint in constraints:
        root = find_root(constraint.first_origin)
        constraints_by_root[root].append(constraint)

    return tuple(
        (
            tuple(variables),
            tuple(constraints_by_root[root]),
        )
        for root, variables in variables_by_root.items()
    )


def _founder_component_likelihood(
    variables: tuple[FounderAllele, ...],
    constraints: tuple[_FounderGenotypeConstraint, ...],
    allele_frequencies: dict[str, Decimal],
) -> Decimal:
    alleles = tuple(allele_frequencies)
    return sum(
        (
            _decimal_product(
                allele_frequencies[assignment[variable]]
                for variable in variables
            )
            for assignment in _compatible_founder_component_assignments(
                variables,
                constraints,
                alleles,
            )
        ),
        start=Decimal(0),
    )


def _compatible_founder_component_assignments(
    variables: tuple[FounderAllele, ...],
    constraints: tuple[_FounderGenotypeConstraint, ...],
    candidate_alleles: tuple[str, ...],
) -> Iterator[dict[FounderAllele, str]]:
    """Propagate exact assignments through one connected constraint graph.

    An unordered diploid genotype permits at most one neighboring allele for
    a known allele. Choosing the first founder-origin allele therefore fixes
    every value in a connected component. Scanning root alleles is exact and
    avoids enumerating the Cartesian product of all founder-origin domains.
    """

    if not variables:
        yield {}
        return

    adjacent_constraints: dict[
        FounderAllele,
        list[tuple[FounderAllele, str, str]],
    ] = {variable: [] for variable in variables}
    for constraint in constraints:
        adjacent_constraints[constraint.first_origin].append(
            (
                constraint.second_origin,
                constraint.first_allele,
                constraint.second_allele,
            )
        )
        if constraint.first_origin != constraint.second_origin:
            adjacent_constraints[constraint.second_origin].append(
                (
                    constraint.first_origin,
                    constraint.second_allele,
                    constraint.first_allele,
                )
            )

    root = variables[0]
    for root_allele in candidate_alleles:
        assignment = {root: root_allele}
        pending_origins = deque((root,))
        is_compatible = True
        while pending_origins and is_compatible:
            origin = pending_origins.popleft()
            origin_allele = assignment[origin]
            for (
                neighbor,
                first_allele,
                second_allele,
            ) in adjacent_constraints[origin]:
                if origin_allele == first_allele:
                    neighbor_allele = second_allele
                elif origin_allele == second_allele:
                    neighbor_allele = first_allele
                else:
                    is_compatible = False
                    break

                assigned_neighbor_allele = assignment.get(neighbor)
                if assigned_neighbor_allele is None:
                    assignment[neighbor] = neighbor_allele
                    pending_origins.append(neighbor)
                elif assigned_neighbor_allele != neighbor_allele:
                    is_compatible = False
                    break

        if is_compatible and len(assignment) == len(variables):
            yield assignment


def _propagate_alleles(
    origins: dict[str, tuple[AlleleOrigin, AlleleOrigin]],
    founder_assignment: dict[FounderAllele, str],
) -> dict[str, tuple[str, str]]:
    return {
        person_id: (
            founder_assignment[person_origins[0]],
            founder_assignment[person_origins[1]],
        )
        for person_id, person_origins in origins.items()
    }


def inheritance_origins(
    family: Family,
    bits: tuple[int, ...],
) -> dict[str, tuple[AlleleOrigin, AlleleOrigin]]:
    """Propagate founder-allele origins for one inheritance vector."""

    if len(bits) != len(family.meioses):
        raise ValueError("One inheritance bit is required per meiosis.")

    transmitted = {
        (meiosis.parent_id, meiosis.child_id): bit
        for meiosis, bit in zip(family.meioses, bits)
    }
    origins = {
        founder.individual_id: (
            (founder.individual_id, 0),
            (founder.individual_id, 1),
        )
        for founder in family.founders
    }

    remaining = [
        person.individual_id
        for person in family.individuals
        if not person.is_founder
    ]
    while remaining:
        progressed = False
        for person_id in tuple(remaining):
            person = family.by_id[person_id]
            if person.father_id not in origins or person.mother_id not in origins:
                continue
            paternal_bit = transmitted[(person.father_id, person.individual_id)]
            maternal_bit = transmitted[(person.mother_id, person.individual_id)]
            paternal_origin = origins[person.father_id][paternal_bit]
            maternal_origin = origins[person.mother_id][maternal_bit]
            origins[person_id] = (paternal_origin, maternal_origin)
            remaining.remove(person_id)
            progressed = True
        if not progressed:
            raise ValueError(
                f"Could not topologically resolve family {family.family_id!r}."
            )

    return origins


def _family_genotypes_match(
    family: Family,
    marker_name: str,
    assigned_alleles: dict[str, tuple[str, str]],
    *,
    ignored_individual_id: str | None = None,
) -> bool:
    return all(
        (
            person.individual_id == ignored_individual_id
            or _genotype_matches(
                person.genotypes.get(marker_name, (None, None)),
                assigned_alleles[person.individual_id],
            )
        )
        for person in family.individuals
    )


def _genotype_matches(
    observed: tuple[str | None, str | None],
    assigned: tuple[str, str],
) -> bool:
    if observed[0] is None or observed[1] is None:
        return True
    return sorted(observed) == sorted(assigned)


def _allele_frequencies(family: Family, marker: Marker) -> dict[str, float]:
    if marker.allele_frequencies:
        total = fsum(marker.allele_frequencies.values())
        if total <= 0:
            raise ValueError(
                f"Allele frequencies for marker {marker.name!r} sum to zero."
            )
        return {
            allele: frequency / total
            for allele, frequency in marker.allele_frequencies.items()
        }

    observed = sorted(
        {
            allele
            for person in family.individuals
            for allele in person.genotypes.get(marker.name, (None, None))
            if allele is not None
        }
    )
    if not observed:
        raise ValueError(
            f"Marker {marker.name!r} has no observed alleles and no frequency file."
        )
    frequency = 1.0 / len(observed)
    return {allele: frequency for allele in observed}


def _decimal_allele_frequencies(
    family: Family,
    marker: Marker,
) -> dict[str, Decimal]:
    """Normalize frequencies without introducing intermediate float error."""

    if marker.allele_frequencies:
        frequencies = {
            allele: Decimal(str(frequency))
            for allele, frequency in marker.allele_frequencies.items()
        }
    else:
        observed = sorted(
            {
                allele
                for person in family.individuals
                for allele in person.genotypes.get(
                    marker.name,
                    (None, None),
                )
                if allele is not None
            }
        )
        if not observed:
            raise ValueError(
                f"Marker {marker.name!r} has no observed alleles and no "
                "frequency file."
            )
        frequency = Decimal(1) / Decimal(len(observed))
        frequencies = {allele: frequency for allele in observed}

    total = sum(frequencies.values(), start=Decimal(0))
    if total <= 0:
        raise ValueError(
            f"Allele frequencies for marker {marker.name!r} sum to zero."
        )
    return {
        allele: frequency / total
        for allele, frequency in frequencies.items()
    }


def _product(values) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


def _decimal_product(values: Iterable[Decimal]) -> Decimal:
    result = Decimal(1)
    for value in values:
        result *= value
    return result


def _fraction_product(values: Iterable[Fraction]) -> Fraction:
    result = Fraction(1)
    for value in values:
        result *= value
    return result
