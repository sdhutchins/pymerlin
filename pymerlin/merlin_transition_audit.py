"""Audit exact MERLIN transition reductions without changing inference.

The audit combines measured marker information, structural tree interactions,
and symmetry plans already proved elsewhere in PyMerlin. It reports which
MERLIN transition route would apply and how many active coordinates remain
after known founder quotients. It does not execute a transition or claim that
the quotient is sufficient for a complete multipoint analysis.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .chain_reduction import detect_untyped_chains
from .founder_symmetry import (
    build_founder_couple_symmetry_plan,
    build_founder_orientation_symmetry_plan,
)
from .information import _merlin_bit_count
from .inheritance_tree import (
    InheritanceTree,
    LeafNode,
    SharedNode,
    SplitNode,
    TreeNode,
    ZeroNode,
    _inheritance_recursion_budget,
    _materialize_scaled_tree,
)
from .models import Dataset, Family
from .transition_planner import SparseTransitionPlan, plan_sparse_transition

_MINIMUM_INFORMATION_SCORE_FOR_SPARSE_BOUND = 0.3
_SPARSE_CONDITIONING_BOUND_SUM = 1.0


@dataclass(frozen=True)
class MarkerTreeInformation:
    """Entropy and support information for one marker tree."""

    information: float
    minimum_information: float
    support_fraction: float


@dataclass(frozen=True)
class MerlinTransitionAudit:
    """Measured route and quotient diagnostics for one marker interval."""

    structural_plan: SparseTransitionPlan
    current_marker_information: MarkerTreeInformation
    next_marker_information: MarkerTreeInformation
    current_sparse_information_bound: float
    next_sparse_information_bound: float
    sparse_conditioning_bound_sum: float
    merlin_would_use_sparse_conditioning: bool
    merlin_effective_bit_count: int
    active_founder_orientation_group_sizes: tuple[int, ...]
    active_founder_couple_representative_indices: tuple[int, ...]
    active_untyped_chain_selector_sizes: tuple[int, ...]
    known_quotient_hidden_bit_indices: tuple[int, ...]

    @property
    def active_bit_count(self) -> int:
        """Return structurally active full-coordinate bits."""

        return self.structural_plan.active_bit_count

    @property
    def active_founder_orientation_group_count(self) -> int:
        """Return founder orientation groups affecting this interval."""

        return len(self.active_founder_orientation_group_sizes)

    @property
    def active_founder_couple_group_count(self) -> int:
        """Return founder-couple quotients affecting this interval."""

        return len(self.active_founder_couple_representative_indices)

    @property
    def active_untyped_chain_count(self) -> int:
        """Return detected counting-chain quotients touching active bits."""

        return len(self.active_untyped_chain_selector_sizes)

    @property
    def active_bits_after_known_symmetry_quotients(self) -> int:
        """Return active coordinates after one bit per known symmetry orbit."""

        return self.active_bit_count - len(self.known_quotient_hidden_bit_indices)


def audit_merlin_transition(
    dataset: Dataset,
    family: Family,
    current_tree: InheritanceTree,
    next_emission_tree: InheritanceTree,
    *,
    maximum_component_bits: int = 24,
) -> MerlinTransitionAudit:
    """Measure MERLIN route selection and known exact transition quotients.

    The sparse-route decision mirrors MERLIN's information gates for an
    ordinary exact autosomal multipoint interval. It assumes two-point mode,
    zero-recombination mode, and recombinant-count approximation are disabled.
    """

    if current_tree.bit_count != len(family.meioses):
        raise ValueError("Current tree does not match the family meiosis count.")
    if next_emission_tree.bit_count != len(family.meioses):
        raise ValueError("Next emission tree does not match the family meiosis count.")

    structural_plan = plan_sparse_transition(
        current_tree,
        next_emission_tree,
        maximum_component_bits=maximum_component_bits,
    )
    effective_bit_count = _merlin_bit_count(dataset, family)
    current_information = marker_tree_information(
        current_tree,
        effective_bit_count,
    )
    next_information = marker_tree_information(
        next_emission_tree,
        effective_bit_count,
    )

    founder_ids = {founder.individual_id for founder in family.founders}
    transmitting_founder_count = len(
        {
            meiosis.parent_id
            for meiosis in family.meioses
            if meiosis.parent_id in founder_ids
        }
    )
    founder_couple_count = (
        len(family.meioses) - transmitting_founder_count - effective_bit_count
    )
    if founder_couple_count < 0:
        raise ValueError("MERLIN effective bit count exceeds founder reductions.")
    couple_information_penalty = (
        founder_couple_count / effective_bit_count if effective_bit_count > 0 else 0.0
    )
    current_bound = _sparse_information_bound(
        current_information,
        couple_information_penalty,
    )
    next_bound = _sparse_information_bound(
        next_information,
        couple_information_penalty,
    )
    bound_sum = current_bound + next_bound

    active_indices = frozenset(structural_plan.active_bit_indices)
    orientation_plan = build_founder_orientation_symmetry_plan(
        family,
        active_indices,
    )
    orientation_groups = tuple(
        (representative_bit_index, descendant_flip_indices)
        for representative_bit_index, descendant_flip_indices in enumerate(
            orientation_plan.descendant_flip_indices_by_bit
        )
        if descendant_flip_indices is not None
    )
    founder_couple_plan = build_founder_couple_symmetry_plan(
        family,
        active_indices,
    )
    active_chains = tuple(
        chain
        for chain in detect_untyped_chains(family)
        if active_indices.intersection(chain.selector_bit_indices)
    )
    orientation_representatives = {
        representative_bit_index for representative_bit_index, _ in orientation_groups
    }
    couple_representatives = {
        symmetry.representative_bit_index for symmetry in founder_couple_plan.symmetries
    }
    known_hidden_indices = tuple(
        sorted(orientation_representatives | couple_representatives)
    )

    return MerlinTransitionAudit(
        structural_plan=structural_plan,
        current_marker_information=current_information,
        next_marker_information=next_information,
        current_sparse_information_bound=current_bound,
        next_sparse_information_bound=next_bound,
        sparse_conditioning_bound_sum=bound_sum,
        merlin_would_use_sparse_conditioning=(
            effective_bit_count >= 3 and bound_sum > _SPARSE_CONDITIONING_BOUND_SUM
        ),
        merlin_effective_bit_count=effective_bit_count,
        active_founder_orientation_group_sizes=tuple(
            1 + len(descendant_flip_indices)
            for _, descendant_flip_indices in orientation_groups
        ),
        active_founder_couple_representative_indices=tuple(
            sorted(couple_representatives)
        ),
        active_untyped_chain_selector_sizes=tuple(
            len(active_indices.intersection(chain.selector_bit_indices))
            for chain in active_chains
        ),
        known_quotient_hidden_bit_indices=known_hidden_indices,
    )


def marker_tree_information(
    tree: InheritanceTree,
    merlin_effective_bit_count: int,
) -> MarkerTreeInformation:
    """Calculate MERLIN-compatible entropy and support information by DAG.

    Incoming probability mass is merged by node identity at every tree depth.
    This avoids enumerating the exponentially many paths represented by a
    shared subtree.
    """

    if merlin_effective_bit_count < 0:
        raise ValueError("MERLIN effective bit count cannot be negative.")
    if merlin_effective_bit_count > tree.bit_count:
        raise ValueError("MERLIN effective bits cannot exceed full tree bits.")
    if merlin_effective_bit_count == 0:
        return MarkerTreeInformation(
            information=0.0,
            minimum_information=0.0,
            support_fraction=1.0,
        )

    with _inheritance_recursion_budget(tree.bit_count):
        materialized_root = _materialize_scaled_tree(tree.root)
    value_probability_terms = _dag_value_probability_terms(
        materialized_root,
        tree.bit_count,
    )
    probability_by_value = {
        value: math.fsum(probability_terms)
        for value, probability_terms in value_probability_terms.items()
    }
    likelihood_total = math.fsum(
        value * probability for value, probability in probability_by_value.items()
    )
    if likelihood_total <= 0.0:
        raise ValueError("Marker tree has non-positive total likelihood.")

    entropy_numerator = math.fsum(
        probability * value * math.log(value)
        for value, probability in probability_by_value.items()
        if value > 0.0
    )
    posterior_entropy_term = entropy_numerator / likelihood_total - math.log(
        likelihood_total
    )
    information = max(
        posterior_entropy_term / (merlin_effective_bit_count * math.log(2.0)),
        0.0,
    )
    support_fraction = math.fsum(
        probability
        for value, probability in probability_by_value.items()
        if value != 0.0
    )
    minimum_information = (
        -math.log2(support_fraction) / merlin_effective_bit_count
        if support_fraction > 0.0
        else 0.0
    )
    return MarkerTreeInformation(
        information=information,
        minimum_information=minimum_information,
        support_fraction=support_fraction,
    )


def format_merlin_transition_audit(audit: MerlinTransitionAudit) -> str:
    """Format one deterministic transition-audit report."""

    orientation_sizes = (
        ",".join(map(str, audit.active_founder_orientation_group_sizes)) or "none"
    )
    couple_representatives = (
        ",".join(map(str, audit.active_founder_couple_representative_indices)) or "none"
    )
    chain_sizes = (
        ",".join(map(str, audit.active_untyped_chain_selector_sizes)) or "none"
    )
    return "\n".join(
        (
            f"full_bits\t{audit.structural_plan.full_bit_count}",
            f"merlin_effective_bits\t{audit.merlin_effective_bit_count}",
            f"active_bits\t{audit.active_bit_count}",
            (
                "active_founder_orientation_groups\t"
                f"{audit.active_founder_orientation_group_count}"
            ),
            f"active_founder_orientation_group_sizes\t{orientation_sizes}",
            (
                "active_founder_couple_groups\t"
                f"{audit.active_founder_couple_group_count}"
            ),
            f"active_founder_couple_representatives\t{couple_representatives}",
            f"active_untyped_chains\t{audit.active_untyped_chain_count}",
            f"active_untyped_chain_selector_sizes\t{chain_sizes}",
            (
                "active_bits_after_known_symmetry_quotients\t"
                f"{audit.active_bits_after_known_symmetry_quotients}"
            ),
            (
                "current_information\t"
                f"{audit.current_marker_information.information:.8f}"
            ),
            (f"next_information\t{audit.next_marker_information.information:.8f}"),
            (
                "current_minimum_information\t"
                f"{audit.current_marker_information.minimum_information:.8f}"
            ),
            (
                "next_minimum_information\t"
                f"{audit.next_marker_information.minimum_information:.8f}"
            ),
            (
                "sparse_conditioning_bound_sum\t"
                f"{audit.sparse_conditioning_bound_sum:.8f}"
            ),
            (
                "merlin_would_use_sparse_conditioning\t"
                f"{str(audit.merlin_would_use_sparse_conditioning).lower()}"
            ),
        )
    )


def _sparse_information_bound(
    marker_information: MarkerTreeInformation,
    couple_information_penalty: float,
) -> float:
    """Apply MERLIN's conservative sparse-route information gate."""

    information_bound = (
        marker_information.minimum_information
        if marker_information.information > _MINIMUM_INFORMATION_SCORE_FOR_SPARSE_BOUND
        else 0.0
    )
    return information_bound - couple_information_penalty


def _dag_value_probability_terms(
    root: TreeNode,
    bit_count: int,
) -> dict[float, list[float]]:
    """Accumulate uniform path mass while merging shared DAG states."""

    current_nodes = {id(root): (root, 1.0)}
    value_probability_terms: dict[float, list[float]] = defaultdict(list)
    for _bit_index in range(bit_count + 1):
        next_probability_terms: dict[int, list[float]] = defaultdict(list)
        next_node_by_id: dict[int, TreeNode] = {}
        for node, probability in current_nodes.values():
            if isinstance(node, ZeroNode):
                value_probability_terms[0.0].append(probability)
            elif isinstance(node, LeafNode):
                value_probability_terms[node.value].append(probability)
            elif isinstance(node, SharedNode):
                next_node_by_id[id(node.child)] = node.child
                next_probability_terms[id(node.child)].append(probability)
            elif isinstance(node, SplitNode):
                for child in (node.zero_child, node.one_child):
                    next_node_by_id[id(child)] = child
                    next_probability_terms[id(child)].append(0.5 * probability)
            else:
                raise TypeError(f"Unsupported materialized tree node: {type(node)!r}")
        current_nodes = {
            node_id: (
                next_node_by_id[node_id],
                math.fsum(probability_terms),
            )
            for node_id, probability_terms in next_probability_terms.items()
        }
        if not current_nodes:
            return value_probability_terms

    raise ValueError("Inheritance tree exceeds its declared bit depth.")
