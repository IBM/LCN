# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Credal Cluster Tree Elimination (two-pass bucket tree) for LCNs
# Computes ALL marginals in a single run via collect + distribute passes.

import itertools
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
from pyomo.environ import (
    ConcreteModel, Var, Objective, ConstraintList,
    NonNegativeReals, minimize, maximize, SolverFactory, value
)

# Local
from lcn.core.model import LCN
from lcn.inference.marginal.cve import CredalVE, Potential
from lcn.inference.utils.common import check_consistency


class CredalCTE:
    """
    Credal Cluster Tree Elimination. Extends bucket elimination with a
    downward (distribute) pass so that all variable marginals are computed
    in a single two-pass execution over the bucket tree.

    References:
        - Dechter (1999). Bucket elimination: A unifying framework.
        - Kask, Dechter, Larrosa, Cozman (2001). Bucket-tree elimination.
        - Mauá, Cozman (2020). Thirty years of credal networks.
    """

    def __init__(self, cve: CredalVE):
        """
        Args:
            cve: A CredalVE instance with build() already called.
        """
        assert cve.extreme_points is not None, \
            "CredalVE must have build() called before passing to CTE."
        assert cve.bn_min is not None
        self.cve = cve
        self.marginals = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, evidence: dict = {},
            epsilon: float = None,
            n_clusters: int = 0,
            cluster_representative: str = "plub",
            verbosity: int = 1) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute lower and upper bounds on the marginal of EVERY variable
        using two-pass bucket tree elimination.

        Args:
            evidence: {variable_name: value} for observed variables.
            epsilon: None for exact pruning, >0 for epsilon-approximate.
            n_clusters: If >0, cluster functions into n_clusters groups
                using K-means with Manhattan distance before pruning.
            cluster_representative: How to compute cluster representatives.
                "plub" — Pareto Least Upper Bound (componentwise max).
                "mean" — cluster centroid (componentwise mean).
            verbosity: 0=silent, 1=summary, 2=detailed.

        Returns:
            Dict mapping variable name to (lower_bounds, upper_bounds)
            numpy arrays. Includes both compound and singleton marginals.
        """

        t_start = time.time()

        # Choose pruning function
        if epsilon is not None and epsilon > 0:
            def prune_fn(pot):
                return pot.epsilon_prune(epsilon)
        else:
            def prune_fn(pot):
                return pot.prune()

        # Chain clustering before pruning if requested
        if n_clusters > 0:
            _base_prune = prune_fn
            def prune_fn(pot, _nc=n_clusters, _cr=cluster_representative,
                         _bp=_base_prune):
                pot = pot.cluster_prune(_nc, representative=_cr)
                return _bp(pot)

        # Step 1: Build potentials from extreme points + evidence
        bn = self.cve.bn_min
        node_names = [bn.variable(n).name() for n in bn.nodes()]
        cards = {}
        for nid in bn.nodes():
            cards[bn.variable(nid).name()] = bn.variable(nid).domainSize()

        potentials = self._build_potentials(evidence)

        # Step 2: Compute elimination ordering (min-fill heuristic)
        scopes = [p.scope for p in potentials]
        elim_order = CredalVE._min_fill_order(scopes, exclude=set())

        if verbosity > 0:
            eps_str = f", epsilon={epsilon}" if epsilon else ""
            print(f"[CredalCTE] Computing all marginals (min-fill{eps_str})")
            print(f"[CredalCTE] Variables: {elim_order}")
            print(f"[CredalCTE] Evidence: {evidence}")
            total_funcs = sum(len(p.functions) for p in potentials)
            print(f"[CredalCTE] Initial potentials: {len(potentials)}, "
                  f"total functions: {total_funcs}")

        # Step 3: Build bucket tree
        buckets, parent, children, local_pots = \
            self._build_bucket_tree(potentials, elim_order)

        # Step 4: Upward pass (collect)
        up_msgs, combined, collect_sizes = self._collect(
            elim_order, buckets, parent, children, local_pots, prune_fn,
            verbosity)

        # Step 5: Downward pass (distribute)
        down_msgs, distribute_sizes = self._distribute(
            elim_order, parent, children, up_msgs, local_pots, prune_fn,
            verbosity)

        # Message size statistics
        all_msg_sizes = collect_sizes + distribute_sizes
        if all_msg_sizes and verbosity > 0:
            avg_sz = sum(all_msg_sizes) / len(all_msg_sizes)
            min_sz = min(all_msg_sizes)
            max_sz = max(all_msg_sizes)
            print(f"[CredalCTE] Message sizes: "
                  f"avg={avg_sz:.1f}, min={min_sz}, max={max_sz} "
                  f"(collect={len(collect_sizes)}, "
                  f"distribute={len(distribute_sizes)})")

        # Step 6: Extract all marginals (compound variables)
        self.marginals = self._extract_marginals(
            elim_order, cards, up_msgs, down_msgs, local_pots, children,
            prune_fn)

        # Step 7: Extract singleton marginals from compound variables
        self.singleton_marginals = self._extract_singleton_marginals(
            self.marginals, evidence)

        t_end = time.time()

        if verbosity > 0:
            print(f"[CredalCTE] Compound variable marginals:")
            for var in elim_order:
                lo, hi = self.marginals[var]
                for val in range(len(lo)):
                    print(f"  P({var}={val}): "
                          f"[{lo[val]:.6f}, {hi[val]:.6f}]")

            if self.singleton_marginals:
                print(f"[CredalCTE] Singleton variable marginals:")
                for atom in sorted(self.singleton_marginals):
                    lo, hi = self.singleton_marginals[atom]
                    print(f"  P({atom}=0): [{1.0 - hi:.6f}, {1.0 - lo:.6f}]")
                    print(f"  P({atom}=1): [{lo:.6f}, {hi:.6f}]")

            print(f"[CredalCTE] Time elapsed: {t_end - t_start:.4f} sec")

        # Return combined dict of all marginals
        all_marginals = dict(self.marginals)
        for atom, (lo, hi) in self.singleton_marginals.items():
            all_marginals[atom] = (
                np.array([1.0 - hi, lo]),
                np.array([1.0 - lo, hi])
            )
        return all_marginals

    # ------------------------------------------------------------------
    # Potential construction (same logic as CredalVE.run)
    # ------------------------------------------------------------------

    def _build_potentials(self, evidence: dict) -> List[Potential]:
        """Build initial potentials from extreme points and evidence."""
        bn = self.cve.bn_min
        node_names = [bn.variable(n).name() for n in bn.nodes()]
        cards = {}
        for nid in bn.nodes():
            cards[bn.variable(nid).name()] = bn.variable(nid).domainSize()

        potentials = []
        for node_name, configs in self.cve.extreme_points.items():
            nid = bn.idFromName(node_name)
            parent_ids = sorted(bn.parents(nid))
            parent_names = [bn.variable(pid).name() for pid in parent_ids]
            scope = [node_name] + parent_names
            shape = tuple(cards[v] for v in scope)

            config_map = {}
            for config_str, vertices in configs.items():
                if config_str == "<>":
                    parent_config = ()
                else:
                    inner = config_str[1:-1]
                    parts = inner.split("|")
                    pvals = {}
                    for p in parts:
                        pname, pval = p.split(":")
                        pvals[pname] = int(pval)
                    parent_config = tuple(pvals[pn] for pn in parent_names)
                config_map[parent_config] = vertices

            if len(parent_names) == 0:
                all_parent_configs = [()]
            else:
                all_parent_configs = list(itertools.product(
                    *[range(cards[pn]) for pn in parent_names]
                ))

            vertex_counts = [len(config_map.get(pc, [[]]))
                             for pc in all_parent_configs]
            vertex_combos = list(itertools.product(
                *[range(c) for c in vertex_counts]
            ))

            functions = []
            for combo in vertex_combos:
                arr = np.zeros(shape)
                for pc_idx, pc in enumerate(all_parent_configs):
                    v_idx = combo[pc_idx]
                    verts = config_map.get(
                        pc, [[1.0 / cards[node_name]] * cards[node_name]])
                    vertex = verts[v_idx]
                    for child_val in range(cards[node_name]):
                        idx = [slice(None)] * len(scope)
                        idx[0] = child_val
                        for pi, pn in enumerate(parent_names):
                            idx[1 + pi] = pc[pi]
                        arr[tuple(idx)] = vertex[child_val]
                functions.append(arr)

            potentials.append(Potential(scope, cards, functions))

        # Evidence indicator potentials
        for ev_var, ev_val in evidence.items():
            assert ev_var in node_names, \
                f"Evidence variable '{ev_var}' not found."
            indicator = np.zeros(cards[ev_var])
            indicator[ev_val] = 1.0
            potentials.append(Potential([ev_var], cards, [indicator]))

        return potentials

    # ------------------------------------------------------------------
    # Bucket tree construction
    # ------------------------------------------------------------------

    def _build_bucket_tree(self, potentials, elim_order):
        """
        Assign potentials to buckets and determine the tree structure.

        Returns:
            buckets: dict var -> list of Potential (local potentials in bucket)
            parent: dict var -> parent var (or None for root)
            children: dict var -> list of child vars
            local_pots: dict var -> list of original potentials assigned
        """
        # Initialize buckets
        buckets = {var: [] for var in elim_order}
        assigned = set()

        # Assign each potential to the first bucket in the ordering that
        # contains a variable in the potential's scope
        remaining_potentials = list(potentials)
        for var in elim_order:
            to_assign = []
            still_remaining = []
            for p in remaining_potentials:
                if var in p.scope:
                    to_assign.append(p)
                else:
                    still_remaining.append(p)
            buckets[var] = to_assign
            remaining_potentials = still_remaining

        # Any remaining potentials (shouldn't happen, but safety)
        if remaining_potentials:
            buckets[elim_order[-1]].extend(remaining_potentials)

        # Save local potentials for later use
        local_pots = {var: list(buckets[var]) for var in elim_order}

        # Determine parent-child relationships by simulating scope
        # propagation. When bucket Xi sends a message, its scope is the
        # union of all potentials in the bucket (local + received messages)
        # minus Xi. The parent is the next variable in the ordering that
        # appears in this message scope. The message is then placed in the
        # parent's bucket, expanding the parent's effective scope.
        parent = {var: None for var in elim_order}
        children = {var: [] for var in elim_order}
        # Track the effective scope of each bucket (local + received msgs)
        effective_scope = {var: set() for var in elim_order}
        for var in elim_order:
            for p in buckets[var]:
                effective_scope[var].update(p.scope)

        for i, var in enumerate(elim_order):
            # Message scope = effective bucket scope minus the eliminated var
            msg_scope = effective_scope[var] - {var}

            # Find the parent: next variable in ordering in msg_scope
            for j in range(i + 1, len(elim_order)):
                candidate = elim_order[j]
                if candidate in msg_scope:
                    parent[var] = candidate
                    children[candidate].append(var)
                    # The message goes to the parent's bucket, expanding
                    # the parent's effective scope
                    effective_scope[candidate].update(msg_scope)
                    break

        return buckets, parent, children, local_pots

    # ------------------------------------------------------------------
    # Upward pass (collect to root)
    # ------------------------------------------------------------------

    def _collect(self, elim_order, buckets, parent, children, local_pots,
                 prune_fn, verbosity):
        """
        Process buckets in elimination order (leaves to root).
        Returns upward messages, combined potentials, and message sizes.
        """
        up_msgs = {}     # var -> Potential (message sent upward to parent)
        combined = {}    # var -> Potential (combined before marginalization)
        msg_sizes = []   # number of functions in each message sent

        for var in elim_order:
            # Gather: local potentials + incoming upward messages from children
            bucket_pots = list(local_pots[var])
            for child in children[var]:
                if child in up_msgs:
                    bucket_pots.append(up_msgs[child])

            if not bucket_pots:
                # Empty bucket — create trivial potential
                cards = self.cve.bn_min
                nid = cards.idFromName(var)
                k = cards.variable(nid).domainSize()
                cards_dict = {}
                for n in self.cve.bn_min.nodes():
                    cards_dict[self.cve.bn_min.variable(n).name()] = \
                        self.cve.bn_min.variable(n).domainSize()
                trivial = Potential([var], cards_dict,
                                    [np.ones(k)])
                bucket_pots = [trivial]

            # Combine all potentials in the bucket
            comb = bucket_pots[0]
            for p in bucket_pots[1:]:
                comb = comb.combine(p)
            combined[var] = comb

            # Marginalize out var
            msg = comb.marginalize(var)

            # Prune
            msg = prune_fn(msg)

            up_msgs[var] = msg
            msg_sizes.append(len(msg.functions))

            if verbosity > 0:
                print(f"  [Collect] {var}: "
                      f"combined={len(comb.functions)} -> "
                      f"msg={len(msg.functions)} funcs, "
                      f"scope {msg.scope}")

        return up_msgs, combined, msg_sizes

    # ------------------------------------------------------------------
    # Downward pass (distribute from root)
    # ------------------------------------------------------------------

    def _distribute(self, elim_order, parent, children, up_msgs,
                    local_pots, prune_fn, verbosity):
        """
        Process buckets in reverse elimination order (root to leaves).
        Returns downward messages and message sizes.
        """
        down_msgs = {}  # var -> Potential (message received from parent)
        msg_sizes = []  # number of functions in each message sent

        # Process in reverse order (root first, then down to leaves)
        for var in reversed(elim_order):
            # For each child of var, compute a downward message
            for child in children[var]:
                # The downward message to `child` is computed by combining:
                # 1. The downward message that `var` received from its parent
                # 2. Local potentials in var's bucket
                # 3. Upward messages from all children of var EXCEPT `child`
                # Then marginalize out `var`.

                pots_to_combine = list(local_pots[var])

                # Add parent's downward message (if var is not root)
                if var in down_msgs:
                    pots_to_combine.append(down_msgs[var])

                # Add upward messages from siblings (children of var, except child)
                for sibling in children[var]:
                    if sibling != child and sibling in up_msgs:
                        pots_to_combine.append(up_msgs[sibling])

                if not pots_to_combine:
                    # Nothing to send — create a trivial potential
                    cards_dict = {}
                    for n in self.cve.bn_min.nodes():
                        cards_dict[self.cve.bn_min.variable(n).name()] = \
                            self.cve.bn_min.variable(n).domainSize()
                    nid = self.cve.bn_min.idFromName(var)
                    k = self.cve.bn_min.variable(nid).domainSize()
                    msg = Potential([var], cards_dict, [np.ones(k)])
                    msg = msg.marginalize(var)
                    comb_size = 1
                else:
                    comb = pots_to_combine[0]
                    for p in pots_to_combine[1:]:
                        comb = comb.combine(p)
                    comb_size = len(comb.functions)

                    # Marginalize out var
                    msg = comb.marginalize(var)

                # Prune
                msg = prune_fn(msg)

                down_msgs[child] = msg
                msg_sizes.append(len(msg.functions))

                if verbosity > 0:
                    print(f"  [Distribute] {var} -> {child}: "
                          f"combined={comb_size} -> "
                          f"msg={len(msg.functions)} funcs, "
                          f"scope {msg.scope}")

        return down_msgs, msg_sizes

    # ------------------------------------------------------------------
    # Extract all marginals
    # ------------------------------------------------------------------

    def _extract_marginals(self, elim_order, cards, up_msgs, down_msgs,
                           local_pots, children, prune_fn):
        """
        At each bucket, combine all available information and extract
        the marginal for that bucket's variable.

        Full information at bucket Xi =
            local_pots[Xi] + up_msgs from children + down_msg from parent
        """
        marginals = {}

        for var in elim_order:
            all_pots = list(local_pots[var])

            # Add upward messages from children of var
            for child in children[var]:
                if child in up_msgs:
                    all_pots.append(up_msgs[child])

            # Add downward message from parent (if exists)
            if var in down_msgs:
                all_pots.append(down_msgs[var])

            if not all_pots:
                k = cards[var]
                marginals[var] = (np.zeros(k), np.ones(k))
                continue

            # Combine all potentials
            comb = all_pots[0]
            for p in all_pots[1:]:
                comb = comb.combine(p)

            # Prune before marginalizing
            comb = prune_fn(comb)

            # Marginalize out everything except var
            for v in list(comb.scope):
                if v != var:
                    comb = comb.marginalize(v)

            # Extract bounds
            k = cards[var]
            lo = np.ones(k)
            hi = np.zeros(k)

            for f in comb.functions:
                total = np.sum(f)
                if total <= 0:
                    continue
                probs = f / total
                for val in range(k):
                    lo[val] = min(lo[val], probs[val])
                    hi[val] = max(hi[val], probs[val])

            marginals[var] = (lo, hi)

        return marginals

    # ------------------------------------------------------------------
    # Singleton marginals from compound variables
    # ------------------------------------------------------------------

    def _extract_singleton_marginals(
        self, marginals: Dict[str, Tuple[np.ndarray, np.ndarray]],
        evidence: dict
    ) -> Dict[str, Tuple[float, float]]:
        """
        For each compound variable (name contains '-'), identify the
        singleton atoms and compute their marginal bounds by solving
        LPs over the compound marginal polytope.

        Args:
            marginals: compound variable marginals from _extract_marginals.
            evidence: evidence dict (singleton atoms in evidence are skipped).

        Returns:
            Dict mapping singleton atom name to (lower_bound, upper_bound)
            for P(atom=1).
        """
        singleton_marginals = {}
        solver = SolverFactory('ipopt')

        # Suppress ipopt output
        ipopt_log = logging.getLogger('pyomo')
        ipopt_log.setLevel(logging.ERROR)

        for var_name, (lo, hi) in marginals.items():
            if '-' not in var_name:
                continue  # already a singleton

            atoms = var_name.split('-')
            n_atoms = len(atoms)
            k = 2 ** n_atoms  # number of compound states

            for atom_idx, atom in enumerate(atoms):
                if atom in evidence:
                    continue  # skip observed atoms

                # Identify which compound states have this atom = 1
                # Using big-endian bit encoding (same as cve.py)
                ones_states = []
                for s in range(k):
                    bit = (s >> (n_atoms - 1 - atom_idx)) & 1
                    if bit == 1:
                        ones_states.append(s)

                # Solve min LP: minimize P(atom=1)
                lower = self._solve_singleton_lp(
                    lo, hi, k, ones_states, minimize, solver)
                # Solve max LP: maximize P(atom=1)
                upper = self._solve_singleton_lp(
                    lo, hi, k, ones_states, maximize, solver)

                singleton_marginals[atom] = (lower, upper)

        return singleton_marginals

    @staticmethod
    def _solve_singleton_lp(lo, hi, k, target_states, sense, solver):
        """
        Solve a single LP to find the min or max of sum(p[s] for s in
        target_states) subject to the compound marginal bounds.

        Args:
            lo: lower bounds on compound variable states.
            hi: upper bounds on compound variable states.
            k: number of compound states.
            target_states: list of state indices where the atom = 1.
            sense: pyomo minimize or maximize.
            solver: pyomo SolverFactory instance.

        Returns:
            Optimal value of the objective.
        """
        model = ConcreteModel()
        model.S = range(k)
        model.p = Var(model.S, within=NonNegativeReals)
        model.constr = ConstraintList()

        # Probability distribution constraint
        model.constr.add(sum(model.p[s] for s in model.S) == 1.0)

        # Bound constraints from compound marginal
        for s in model.S:
            model.constr.add(model.p[s] >= float(lo[s]))
            model.constr.add(model.p[s] <= float(hi[s]))

        # Objective: sum of p[s] for states where atom=1
        model.obj = Objective(
            expr=sum(model.p[s] for s in target_states),
            sense=sense
        )

        results = solver.solve(model, tee=False)
        return value(model.obj)


if __name__ == "__main__":

    def print_singleton_marginals(results):
        """Print only singleton variable marginals from the results."""
        print("  Singleton variable marginals:")
        for var in sorted(results):
            if '-' not in var:
                lo, hi = results[var]
                for val in range(len(lo)):
                    print(f"    P({var}={val}): [{lo[val]:.6f}, {hi[val]:.6f}]")

    # Load the LCN
    # file_name = "examples/alarm.lcn"
    file_name = "benchmarks/polytree/polytree_n10_1.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Check consistency
    # ok = check_consistency(l)
    # if ok:
    #     print("CONSISTENT")
    # else:
    #     print("INCONSISTENT")

    # Build the CredalVE (needed for extreme points)
    cve = CredalVE(lcn=l)
    cve.build(verbosity=0)

    # Create the CTE solver
    cte = CredalCTE(cve=cve)

    # Exact all-marginals (no evidence)
    # print("\n=== All marginals (exact, no evidence) ===")
    # results = cte.run(evidence={}, verbosity=2)
    # print_singleton_marginals(results)

    # # Exact all-marginals (with evidence)
    # print("\n=== All marginals (exact, B=0, E=0) ===")
    # results = cte.run(evidence={"B": 0, "E": 0}, verbosity=1)
    # print_singleton_marginals(results)

    # Epsilon-approximate all-marginals
    print("\n=== All marginals (epsilon=0.1, no evidence) ===")
    results = cte.run(evidence={}, epsilon=0.1, verbosity=2)
    print_singleton_marginals(results)
