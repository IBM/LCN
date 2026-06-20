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

# Interval Iterative Join-Graph Propagation (IJGP) for Credal Networks

import itertools
import logging
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from pyomo.environ import (
    ConcreteModel, Var, Objective, ConstraintList,
    NonNegativeReals, minimize, maximize, SolverFactory, value
)

# Local
from lcn.core.model import LCN
from lcn.inference.marginal.cn.cve import CredalVE, Potential
from lcn.inference.utils.common import make_ipopt


class CredalIJGP:
    """
    Interval Iterative Join-Graph Propagation for credal networks.

    Constructs a join graph via mini-bucket partitioning from the credal
    network's factor graph, then iteratively passes Potential messages
    between clusters until convergence. The i_bound parameter controls
    the maximum cluster size (number of variables per mini-bucket).

    References:
        - Dechter, Kask, Mateescu (2002). Iterative join-graph
          propagation. UAI.
        - Mauá, Cozman (2020). Thirty years of credal networks.
    """

    def __init__(self, cve: CredalVE):
        """
        Args:
            cve: A CredalVE instance with build() already called.
        """
        assert cve.extreme_points is not None, \
            "CredalVE must have build() called before passing to IJGP."
        assert cve.bn_min is not None
        self.cve = cve
        self.marginals = None
        self.singleton_marginals = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, evidence: dict = {},
            i_bound: int = 4,
            n_iters: int = 100,
            threshold: float = 1e-6,
            epsilon: float = None,
            n_clusters: int = 0,
            cluster_representative: str = "plub",
            verbosity: int = 1) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run interval IJGP inference for credal networks. Computes marginal
        bounds for ALL non-evidence variables.

        Args:
            evidence: {variable_name: value} for observed variables.
            i_bound: Maximum number of variables per mini-bucket cluster.
            n_iters: Maximum number of forward+backward iterations.
            threshold: Convergence threshold on max message change.
            epsilon: If not None, use epsilon-approximate pruning.
            n_clusters: If >0, cluster functions into n_clusters groups
                using K-means with Manhattan distance before pruning.
            cluster_representative: How to compute cluster representatives.
                "plub" — Pareto Least Upper Bound (componentwise max).
                "mean" — cluster centroid (componentwise mean).
            verbosity: Verbosity level (0=silent).

        Returns:
            Dict mapping variable name to (lower_bounds, upper_bounds)
            numpy arrays. Includes both compound and singleton marginals.
        """
        t_start = time.time()

        # Pruning function
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

        # Build card dictionary
        bn = self.cve.bn_min
        node_names = [bn.variable(n).name() for n in bn.nodes()]
        cards = {}
        for nid in bn.nodes():
            cards[bn.variable(nid).name()] = bn.variable(nid).domainSize()

        # Step 1: Build potentials from extreme points + evidence
        potentials = self._build_potentials(evidence)

        # Step 2: Compute elimination ordering (min-fill)
        scopes = [p.scope for p in potentials]
        elim_order = CredalVE._min_fill_order(scopes, exclude=set())

        if verbosity > 0:
            eps_str = f", epsilon={epsilon}" if epsilon else ""
            print(f"[IJGP] Computing all marginals "
                  f"(i_bound={i_bound}{eps_str})")
            print(f"[IJGP] Variables: {elim_order}")
            print(f"[IJGP] Evidence: {evidence}")
            total_funcs = sum(len(p.functions) for p in potentials)
            print(f"[IJGP] Initial potentials: {len(potentials)}, "
                  f"total functions: {total_funcs}")

        # Step 3: Build join graph via mini-bucket partitioning
        clusters, edges, neighbors, cluster_order = \
            self._build_join_graph(potentials, elim_order, i_bound, cards,
                                   verbosity)

        if verbosity > 0:
            print(f"[IJGP] Join graph: {len(clusters)} clusters, "
                  f"{len(edges)} edges")
            max_scope = max(len(c['scope']) for c in clusters) \
                if clusters else 0
            print(f"[IJGP] Max cluster scope: {max_scope}")

        # Step 4: Initialize messages to unit potentials over separators
        messages = {}
        for (a, b), sep in edges.items():
            sep_list = sorted(sep)
            shape = tuple(cards[v] for v in sep_list)
            unit = Potential(sep_list, cards, [np.ones(shape)])
            messages[(a, b)] = unit
            messages[(b, a)] = unit

        # Step 5: Iterative message passing
        for iteration in range(n_iters):
            fwd_delta = self._forward_pass(
                cluster_order, clusters, edges, neighbors,
                messages, cards, prune_fn, verbosity)
            bwd_delta = self._backward_pass(
                cluster_order, clusters, edges, neighbors,
                messages, cards, prune_fn, verbosity)
            max_delta = max(fwd_delta, bwd_delta)

            if verbosity > 0:
                # Message size statistics for this iteration
                msg_sizes = [len(m.functions)
                             for m in messages.values()]
                avg_sz = sum(msg_sizes) / len(msg_sizes) \
                    if msg_sizes else 0
                max_sz = max(msg_sizes) if msg_sizes else 0
                print(f"  Iteration {iteration}: "
                      f"fwd_delta={fwd_delta:.6f}, "
                      f"bwd_delta={bwd_delta:.6f}, "
                      f"max_delta={max_delta:.6f}, "
                      f"msgs={len(msg_sizes)} "
                      f"(avg={avg_sz:.1f}, max={max_sz} funcs)")

            if max_delta < threshold:
                if verbosity > 0:
                    print(f"[IJGP] Converged after {iteration + 1} "
                          f"iterations (delta={max_delta:.2e})")
                break
        else:
            if verbosity > 0:
                print(f"[IJGP] Reached max iterations ({n_iters}), "
                      f"delta={max_delta:.2e}")

        # Step 6: Extract compound marginals
        self.marginals = self._extract_marginals(
            clusters, edges, neighbors, messages, cards, prune_fn)

        # Step 7: Extract singleton marginals
        self.singleton_marginals = self._extract_singleton_marginals(
            self.marginals, evidence)

        t_end = time.time()

        if verbosity > 0:
            print(f"[IJGP] Compound variable marginals:")
            for var in elim_order:
                if var in self.marginals:
                    lo, hi = self.marginals[var]
                    for val in range(len(lo)):
                        print(f"  P({var}={val}): "
                              f"[{lo[val]:.6f}, {hi[val]:.6f}]")

            if self.singleton_marginals:
                print(f"[IJGP] Singleton variable marginals:")
                for atom in sorted(self.singleton_marginals):
                    lo, hi = self.singleton_marginals[atom]
                    print(f"  P({atom}=0): "
                          f"[{1.0 - hi:.6f}, {1.0 - lo:.6f}]")
                    print(f"  P({atom}=1): [{lo:.6f}, {hi:.6f}]")

            print(f"[IJGP] Time elapsed: {t_end - t_start:.4f} sec")

        # Return combined dict of all marginals
        all_marginals = dict(self.marginals)
        for atom, (lo, hi) in self.singleton_marginals.items():
            all_marginals[atom] = (
                np.array([1.0 - hi, lo]),
                np.array([1.0 - lo, hi])
            )
        return all_marginals

    # ------------------------------------------------------------------
    # Potential construction (same as CredalCTE)
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
    # Join graph construction via mini-bucket partitioning
    # ------------------------------------------------------------------

    def _mini_bucket_partition(self, bucket_pots: List[Potential],
                               var: str,
                               i_bound: int) -> List[List[Potential]]:
        """
        Partition potentials into mini-buckets such that no mini-bucket's
        combined scope exceeds i_bound variables.

        Greedy: sort by scope size descending, place each potential into
        the existing mini-bucket that produces the smallest combined scope
        (if within i_bound), otherwise create a new mini-bucket.
        """
        sorted_pots = sorted(bucket_pots,
                             key=lambda p: len(p.scope), reverse=True)

        # Each entry: (list_of_potentials, combined_scope_set)
        mini_buckets: List[Tuple[List[Potential], Set[str]]] = []

        for pot in sorted_pots:
            pot_scope = set(pot.scope)
            best_idx = None
            best_size = float('inf')

            for mb_idx, (mb_pots, mb_scope) in enumerate(mini_buckets):
                new_scope = mb_scope | pot_scope
                if len(new_scope) <= i_bound and len(new_scope) < best_size:
                    best_size = len(new_scope)
                    best_idx = mb_idx

            if best_idx is not None:
                mb_pots, mb_scope = mini_buckets[best_idx]
                mb_pots.append(pot)
                mini_buckets[best_idx] = (mb_pots, mb_scope | pot_scope)
            else:
                mini_buckets.append(([pot], pot_scope))

        return [mb_pots for (mb_pots, _) in mini_buckets]

    def _build_join_graph(self, potentials: List[Potential],
                          elim_order: List[str],
                          i_bound: int,
                          cards: Dict[str, int],
                          verbosity: int = 0):
        """
        Build a join graph via mini-bucket partitioning along the
        elimination ordering.

        Returns:
            clusters: list of dicts with keys 'id', 'var', 'scope',
                      'potentials'
            edges: dict (min_id, max_id) -> set of separator variables
            neighbors: dict cluster_id -> set of neighbor cluster IDs
            cluster_order: list of cluster IDs in topological order
                          (leaves first)
        """
        clusters = []
        cluster_order = []
        edges_list = []
        neighbors = defaultdict(set)

        # Active potentials flowing through the elimination ordering.
        # Each entry: (Potential, source_cluster_id_or_None)
        active = [(p, None) for p in potentials]

        for var in elim_order:
            # Collect potentials whose scope contains var
            bucket = [(p, src) for (p, src) in active if var in p.scope]
            rest = [(p, src) for (p, src) in active if var not in p.scope]

            if not bucket:
                active = rest
                continue

            pots_only = [p for (p, src) in bucket]
            pot_to_source = {id(p): src for (p, src) in bucket}

            # Partition into mini-buckets
            mini_bucket_groups = self._mini_bucket_partition(
                pots_only, var, i_bound)

            if verbosity > 1:
                mb_scopes = []
                for mb in mini_bucket_groups:
                    s = set()
                    for p in mb:
                        s.update(p.scope)
                    mb_scopes.append(s)
                print(f"  [Build] Bucket {var}: "
                      f"{len(pots_only)} potentials -> "
                      f"{len(mini_bucket_groups)} mini-bucket(s), "
                      f"scopes {[sorted(s) for s in mb_scopes]}")

            mb_cluster_ids = []
            for mb in mini_bucket_groups:
                cid = len(clusters)
                scope = set()
                for p in mb:
                    scope.update(p.scope)

                # Connect to source clusters (from earlier buckets)
                for p in mb:
                    src_cid = pot_to_source.get(id(p))
                    if src_cid is not None:
                        sep = scope & clusters[src_cid]['scope']
                        if sep:
                            key = (min(src_cid, cid), max(src_cid, cid))
                            edges_list.append((key, frozenset(sep)))

                clusters.append({
                    'id': cid,
                    'var': var,
                    'scope': scope,
                    'potentials': list(mb),
                })
                cluster_order.append(cid)
                mb_cluster_ids.append(cid)

            # Add edges between mini-buckets of the same variable
            for i in range(len(mb_cluster_ids)):
                for j in range(i + 1, len(mb_cluster_ids)):
                    ci, cj = mb_cluster_ids[i], mb_cluster_ids[j]
                    sep = clusters[ci]['scope'] & clusters[cj]['scope']
                    if sep:
                        key = (min(ci, cj), max(ci, cj))
                        edges_list.append((key, frozenset(sep)))

            # Create message placeholders for the next buckets
            for cid in mb_cluster_ids:
                msg_scope = clusters[cid]['scope'] - {var}
                if msg_scope:
                    scope_list = sorted(msg_scope)
                    shape = tuple(cards[v] for v in scope_list)
                    placeholder = Potential(scope_list, cards,
                                           [np.ones(shape)])
                    rest.append((placeholder, cid))

            active = rest

        # Deduplicate edges and build final structures
        edges = {}
        for (key, sep) in edges_list:
            if key not in edges:
                edges[key] = set(sep)
            else:
                edges[key] |= set(sep)

        for (a, b) in edges:
            neighbors[a].add(b)
            neighbors[b].add(a)

        if verbosity > 1:
            for c in clusters:
                n_funcs = sum(len(p.functions) for p in c['potentials'])
                print(f"  [Build] Cluster {c['id']} (elim={c['var']}): "
                      f"scope={sorted(c['scope'])}, "
                      f"{len(c['potentials'])} pots, "
                      f"{n_funcs} funcs")
            for (a, b), sep in edges.items():
                print(f"  [Build] Edge ({a},{b}): "
                      f"sep={sorted(sep)}")

        return clusters, edges, neighbors, cluster_order

    # ------------------------------------------------------------------
    # Message passing
    # ------------------------------------------------------------------

    def _compute_message(self, src_id: int, dst_id: int,
                         clusters, edges, neighbors, messages,
                         cards, prune_fn) -> Potential:
        """
        Compute the message from cluster src to cluster dst.

        message(src -> dst) = prune(marginalize_{eliminator}(
            product(local_potentials, incoming_msgs_except_from_dst)
        ))

        where eliminator = scope(combined) - separator(src, dst).
        """
        src = clusters[src_id]

        # Get separator
        edge_key = (min(src_id, dst_id), max(src_id, dst_id))
        separator = edges[edge_key]

        # Gather: local potentials + incoming messages except from dst
        pots = list(src['potentials'])
        for nbr_id in neighbors[src_id]:
            if nbr_id != dst_id and (nbr_id, src_id) in messages:
                pots.append(messages[(nbr_id, src_id)])

        if not pots:
            # Return trivial potential over separator
            sep_list = sorted(separator)
            shape = tuple(cards[v] for v in sep_list)
            return Potential(sep_list, cards, [np.ones(shape)])

        # Combine all potentials
        combined = pots[0]
        for p in pots[1:]:
            combined = combined.combine(p)
            # Intermediate pruning to control growth
            if len(combined.functions) > 100:
                combined = prune_fn(combined)

        # Marginalize out eliminator variables
        eliminator = set(combined.scope) - separator
        result = combined
        for var in eliminator:
            result = result.marginalize(var)

        # Prune
        result = prune_fn(result)

        return result

    @staticmethod
    def _message_delta(old_msg: Potential, new_msg: Potential) -> float:
        """
        Compute the maximum change between two message Potentials by
        comparing their lower/upper envelopes.
        """
        if not old_msg.functions or not new_msg.functions:
            return float('inf')

        old_flat = [f.ravel() for f in old_msg.functions]
        new_flat = [f.ravel() for f in new_msg.functions]

        old_lo = np.min(old_flat, axis=0)
        old_hi = np.max(old_flat, axis=0)
        new_lo = np.min(new_flat, axis=0)
        new_hi = np.max(new_flat, axis=0)

        # Handle shape mismatches (shouldn't happen but be safe)
        if old_lo.shape != new_lo.shape:
            return float('inf')

        return max(np.max(np.abs(old_lo - new_lo)),
                   np.max(np.abs(old_hi - new_hi)))

    def _forward_pass(self, cluster_order, clusters, edges, neighbors,
                      messages, cards, prune_fn, verbosity):
        """
        Forward pass: process clusters in order (leaves to root).
        For each cluster, send messages to neighbors later in the ordering.
        """
        max_delta = 0.0
        order_index = {cid: idx for idx, cid in enumerate(cluster_order)}

        for src_id in cluster_order:
            for dst_id in neighbors.get(src_id, set()):
                if order_index.get(dst_id, -1) > order_index.get(src_id, -1):
                    new_msg = self._compute_message(
                        src_id, dst_id, clusters, edges, neighbors,
                        messages, cards, prune_fn)

                    old_msg = messages.get((src_id, dst_id))
                    if old_msg is not None:
                        delta = self._message_delta(old_msg, new_msg)
                        max_delta = max(max_delta, delta)
                    else:
                        delta = float('inf')

                    messages[(src_id, dst_id)] = new_msg

                    if verbosity > 2:
                        print(f"    [Fwd] {src_id}->{dst_id}: "
                              f"{len(new_msg.functions)} funcs, "
                              f"scope={new_msg.scope}, "
                              f"delta={delta:.6f}")

        return max_delta

    def _backward_pass(self, cluster_order, clusters, edges, neighbors,
                       messages, cards, prune_fn, verbosity):
        """
        Backward pass: process clusters in reverse order (root to leaves).
        For each cluster, send messages to neighbors earlier in the ordering.
        """
        max_delta = 0.0
        order_index = {cid: idx for idx, cid in enumerate(cluster_order)}

        for src_id in reversed(cluster_order):
            for dst_id in neighbors.get(src_id, set()):
                if order_index.get(dst_id, -1) < order_index.get(src_id, -1):
                    new_msg = self._compute_message(
                        src_id, dst_id, clusters, edges, neighbors,
                        messages, cards, prune_fn)

                    old_msg = messages.get((src_id, dst_id))
                    if old_msg is not None:
                        delta = self._message_delta(old_msg, new_msg)
                        max_delta = max(max_delta, delta)
                    else:
                        delta = float('inf')

                    messages[(src_id, dst_id)] = new_msg

                    if verbosity > 2:
                        print(f"    [Bwd] {src_id}->{dst_id}: "
                              f"{len(new_msg.functions)} funcs, "
                              f"scope={new_msg.scope}, "
                              f"delta={delta:.6f}")

        return max_delta

    # ------------------------------------------------------------------
    # Marginal extraction
    # ------------------------------------------------------------------

    def _extract_marginals(self, clusters, edges, neighbors, messages,
                           cards, prune_fn):
        """
        For each variable, find the smallest-scope cluster containing it,
        combine all potentials and incoming messages, marginalize to the
        variable, and extract lower/upper bounds.
        """
        # Collect all variables from cluster scopes
        all_vars = set()
        for c in clusters:
            all_vars.update(c['scope'])

        # Map each variable to the best cluster (smallest scope)
        var_to_cluster = {}
        for var in all_vars:
            best_cid = None
            best_size = float('inf')
            for c in clusters:
                if var in c['scope'] and len(c['scope']) < best_size:
                    best_size = len(c['scope'])
                    best_cid = c['id']
            var_to_cluster[var] = best_cid

        marginals = {}
        for var in all_vars:
            cid = var_to_cluster[var]
            if cid is None:
                k = cards.get(var, 2)
                marginals[var] = (np.zeros(k), np.ones(k))
                continue

            c = clusters[cid]

            # Gather: local potentials + ALL incoming messages
            pots = list(c['potentials'])
            for nbr_id in neighbors.get(cid, set()):
                if (nbr_id, cid) in messages:
                    pots.append(messages[(nbr_id, cid)])

            if not pots:
                k = cards.get(var, 2)
                marginals[var] = (np.zeros(k), np.ones(k))
                continue

            # Combine
            combined = pots[0]
            for p in pots[1:]:
                combined = combined.combine(p)
                if len(combined.functions) > 100:
                    combined = prune_fn(combined)

            combined = prune_fn(combined)

            # Marginalize out everything except var
            for v in list(combined.scope):
                if v != var:
                    combined = combined.marginalize(v)

            # Extract bounds
            k = cards.get(var, 2)
            lo = np.ones(k)
            hi = np.zeros(k)

            for f in combined.functions:
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
        """
        singleton_marginals = {}
        solver = make_ipopt()

        ipopt_log = logging.getLogger('pyomo')
        ipopt_log.setLevel(logging.ERROR)

        for var_name, (lo, hi) in marginals.items():
            if '-' not in var_name:
                continue

            atoms = var_name.split('-')
            n_atoms = len(atoms)
            k = 2 ** n_atoms

            for atom_idx, atom in enumerate(atoms):
                if atom in evidence:
                    continue

                ones_states = []
                for s in range(k):
                    bit = (s >> (n_atoms - 1 - atom_idx)) & 1
                    if bit == 1:
                        ones_states.append(s)

                lower = self._solve_singleton_lp(
                    lo, hi, k, ones_states, minimize, solver)
                upper = self._solve_singleton_lp(
                    lo, hi, k, ones_states, maximize, solver)

                singleton_marginals[atom] = (lower, upper)

        return singleton_marginals

    @staticmethod
    def _solve_singleton_lp(lo, hi, k, target_states, sense, solver):
        """
        Solve a single LP to find the min or max of sum(p[s] for s in
        target_states) subject to the compound marginal bounds.
        """
        model = ConcreteModel()
        model.S = range(k)
        model.p = Var(model.S, within=NonNegativeReals)
        model.constr = ConstraintList()

        model.constr.add(sum(model.p[s] for s in model.S) == 1.0)

        for s in model.S:
            model.constr.add(model.p[s] >= float(lo[s]))
            model.constr.add(model.p[s] <= float(hi[s]))

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
    file_name = "benchmarks/real/alarm.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Build the CredalVE (needed for extreme points)
    cve = CredalVE(lcn=l)
    cve.build(verbosity=1, factorization_method="nlp")

    # Create the IJGP solver
    ijgp = CredalIJGP(cve=cve)

    # Run IJGP with i_bound=2
    print("\n=== Interval IJGP (i_bound=2, no evidence) ===")
    results = ijgp.run(evidence={}, i_bound=2, verbosity=3)
    print_singleton_marginals(results)

    # Run IJGP with i_bound=4
    # print("\n=== Interval IJGP (i_bound=4, no evidence) ===")
    # results = ijgp.run(evidence={}, i_bound=2, verbosity=3, n_clusters=4, cluster_representative="plub")
    # print_singleton_marginals(results)

    # Run IJGP with evidence
    # print("\n=== Interval IJGP (i_bound=4, x0=0) ===")
    # results = ijgp.run(evidence={"x0": 0}, i_bound=4, verbosity=3, epsilon=0.1)
    # print_singleton_marginals(results)
