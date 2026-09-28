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

# Shared building blocks for credal-network inference algorithms:
# the `Potential` data structure (a finite set of functions over a scope,
# representing combinations of extreme points) and the `min_fill_order`
# elimination-ordering heuristic. Both are consumed by CredalVE, CCTE and
# IJGP and are intentionally free of any pyAgrum / LCN coupling.

from typing import Dict, List

import numpy as np


class Potential:
    """
    A potential over a set of discrete variables. Stores a finite set of
    non-negative real-valued functions (numpy arrays) over the joint domain.
    Used in credal variable elimination where each function corresponds to
    one combination of extreme points from the local credal sets.
    """

    def __init__(self, scope: List[str], cards: Dict[str, int],
                 functions: List[np.ndarray]):
        """
        Args:
            scope: list of variable names in this potential's domain.
            cards: dict mapping variable name to its cardinality.
            functions: list of numpy arrays, each with shape
                       tuple(cards[v] for v in scope).
        """
        self.scope = list(scope)
        self.cards = cards
        self.functions = list(functions)

    def combine(self, other: 'Potential') -> 'Potential':
        """
        Product of two potentials. The result scope is the union of both
        scopes. Each pair of functions (one from each potential) is
        broadcast-multiplied over the joint domain.
        """
        new_scope = list(self.scope)
        for v in other.scope:
            if v not in new_scope:
                new_scope.append(v)

        # Build shapes for broadcasting into the joint domain
        def _expand(func, src_scope, new_scope, cards):
            """Reshape and transpose a function to align with new_scope."""
            # Build a mapping: for each dim in new_scope, which dim in src_scope?
            shape = []
            for v in new_scope:
                if v in src_scope:
                    shape.append(cards[v])
                else:
                    shape.append(1)
            # Reorder: need to move dimensions of func to match new_scope order
            # func's axes correspond to src_scope in order
            src_idx = {v: i for i, v in enumerate(src_scope)}
            # First, expand func to have len(new_scope) dims
            arr = func
            # Insert size-1 axes for variables not in src_scope
            # Strategy: build target via transpose + reshape
            # Place src dims in correct positions, add size-1 for missing
            target_shape = []
            perm = []
            src_dim = 0
            for v in new_scope:
                if v in src_idx:
                    perm.append(src_idx[v])
                    target_shape.append(cards[v])
                else:
                    target_shape.append(1)

            # Transpose func to match the order of src vars as they appear in new_scope
            src_order = [v for v in new_scope if v in src_idx]
            transpose_perm = [src_idx[v] for v in src_order]
            arr = np.transpose(arr, transpose_perm)

            # Now insert size-1 axes for missing variables
            result_shape = []
            src_dim = 0
            for v in new_scope:
                if v in src_idx:
                    result_shape.append(cards[v])
                    src_dim += 1
                else:
                    result_shape.append(1)
            arr = arr.reshape(result_shape)
            return arr

        new_functions = []
        for f1 in self.functions:
            f1_exp = _expand(f1, self.scope, new_scope, self.cards)
            for f2 in other.functions:
                f2_exp = _expand(f2, other.scope, new_scope, other.cards)
                new_functions.append(f1_exp * f2_exp)

        merged_cards = dict(self.cards)
        merged_cards.update(other.cards)
        return Potential(new_scope, merged_cards, new_functions)

    def marginalize(self, var: str) -> 'Potential':
        """Sum-marginal: sum out a variable from each function."""
        if var not in self.scope:
            return self
        axis = self.scope.index(var)
        new_scope = [v for v in self.scope if v != var]
        new_functions = [np.sum(f, axis=axis) for f in self.functions]
        return Potential(new_scope, self.cards, new_functions)

    def prune(self) -> 'Potential':
        """
        Remove dominated functions. A function p is dominated if there
        exists another function q such that q(y) >= p(y) for all y
        (componentwise). This reduces the potential's cardinality.
        """
        if len(self.functions) <= 1:
            return self
        flat = [f.ravel() for f in self.functions]
        n = len(flat)
        dominated = [False] * n
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                # Check if j dominates i: flat[j] >= flat[i] everywhere
                if np.all(flat[j] >= flat[i]) and np.any(flat[j] > flat[i]):
                    dominated[i] = True
                    break
        new_functions = [self.functions[i] for i in range(n) if not dominated[i]]
        if not new_functions:
            new_functions = [self.functions[0]]
        return Potential(self.scope, self.cards, new_functions)

    def epsilon_prune(self, epsilon: float) -> 'Potential':
        """
        Epsilon-approximate pruning (Mauá et al.). Remove function f if
        there exists another function g such that g(y) >= f(y) - epsilon
        for all y (epsilon-domination). More aggressive than exact prune(),
        controls potential cardinality growth with bounded error.

        Args:
            epsilon: float
                The approximation tolerance. Larger values prune more
                aggressively. When epsilon=0 this is equivalent to exact
                prune().
        """
        if len(self.functions) <= 1:
            return self
        flat = [f.ravel() for f in self.functions]
        n = len(flat)
        dominated = [False] * n
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                # Check if j epsilon-dominates i:
                # flat[j](y) >= flat[i](y) - epsilon for all y
                if np.all(flat[j] >= flat[i] - epsilon):
                    dominated[i] = True
                    break
        new_functions = [self.functions[i] for i in range(n) if not dominated[i]]
        if not new_functions:
            new_functions = [self.functions[0]]
        return Potential(self.scope, self.cards, new_functions)

    def filter_infeasible(self, constraints, node_atoms) -> 'Potential':
        """
        Scheme D4: drop functions that violate a cross-family coupling
        constraint (a `coupling.CouplingConstraints`). Each surviving function
        is a candidate joint over this potential's scope; a constraint is only
        applied once the scope carries all of its atoms (the checkability gate
        inside ``constraints.is_feasible``), so calling this after any
        elimination step is sound -- it can only remove genuinely-infeasible
        functions, never widen the bound.

        Never returns an empty potential: if every function would be dropped
        (which would signal an inconsistent credal net rather than a valid
        prune), the first function is kept so elimination can proceed.
        """
        if constraints is None or len(constraints) == 0 or len(self.functions) <= 1:
            return self
        kept = [f for f in self.functions
                if constraints.is_feasible(f, self.scope, node_atoms, self.cards)]
        if not kept:
            kept = [self.functions[0]]
        if len(kept) == len(self.functions):
            return self
        return Potential(self.scope, self.cards, kept)

    def cluster_prune(self, n_clusters: int,
                      max_iters: int = 10,
                      representative: str = "plub") -> 'Potential':
        """
        Approximate the potential by clustering its functions using
        K-means with Manhattan (L1) distance, then replacing each
        cluster with a single representative function.

        This produces exactly n_clusters representative functions.

        Args:
            n_clusters: Target number of clusters (k).
            max_iters: Maximum K-means iterations (default 10).
            representative: How to compute the cluster representative.
                "plub" — Pareto Least Upper Bound (componentwise max).
                "mean" — cluster centroid (componentwise mean).
        """
        n = len(self.functions)
        if n <= n_clusters or n_clusters <= 0:
            return self

        shape = self.functions[0].shape
        flat = np.array([f.ravel() for f in self.functions])  # (n, d)

        # Initialize centroids: pick k distinct indices at random
        rng = np.random.RandomState(42)
        indices = rng.choice(n, size=n_clusters, replace=False)
        centroids = flat[indices].copy()  # (k, d)

        assignments = np.zeros(n, dtype=int)

        for _ in range(max_iters):
            # Assignment: each function -> nearest centroid (Manhattan).
            # Vectorized: |flat[:,None,:] - centroids[None,:,:]| summed over the
            # feature axis gives the (n, k) distance matrix in one shot; argmin
            # over the centroid axis ties to the lowest index, matching the
            # former per-point np.argmin loop bit-for-bit.
            dists = np.abs(flat[:, None, :] - centroids[None, :, :]).sum(axis=2)
            new_assignments = np.argmin(dists, axis=1)

            # Check convergence
            if np.array_equal(assignments, new_assignments):
                break
            assignments = new_assignments

            # Update centroids: mean of assigned functions. Empty clusters keep
            # their previous centroid (as in the former per-cluster loop, which
            # skipped the update when a cluster had no members).
            for c in range(n_clusters):
                members = flat[assignments == c]
                if len(members) > 0:
                    centroids[c] = np.mean(members, axis=0)

        # Compute representative for each cluster
        new_functions = []
        for c in range(n_clusters):
            members = flat[assignments == c]
            if len(members) > 0:
                if representative == "mean":
                    rep = np.mean(members, axis=0).reshape(shape)
                else:  # "plub"
                    rep = np.max(members, axis=0).reshape(shape)
                new_functions.append(rep)

        if not new_functions:
            new_functions = [self.functions[0]]

        return Potential(self.scope, self.cards, new_functions)


def min_fill_order(scopes: List[List[str]], exclude: set) -> List[str]:
    """
    Compute an elimination ordering using the min-fill heuristic.
    Builds an interaction graph from the potential scopes, then
    greedily eliminates the variable that adds the fewest new edges
    (fill-in edges).

    Args:
        scopes: list of variable-name lists (one per potential).
        exclude: set of variable names NOT to eliminate (query).

    Returns:
        A list of variable names in elimination order.
    """
    # Build interaction graph: undirected edges between variables
    # that appear together in any potential scope
    all_vars = set()
    for s in scopes:
        all_vars.update(s)
    elim_vars = all_vars - exclude
    # adjacency: var -> set of neighbors
    adj = {v: set() for v in all_vars}
    for s in scopes:
        for i, u in enumerate(s):
            for v in s[i + 1:]:
                adj[u].add(v)
                adj[v].add(u)

    order = []
    remaining = set(elim_vars)
    for _ in range(len(elim_vars)):
        # Pick the variable whose elimination adds fewest fill edges
        best_var = None
        best_fill = float('inf')
        for v in remaining:
            # Neighbors of v that are still in remaining or exclude
            nbrs = [u for u in adj[v] if u in remaining or u in exclude]
            # Count fill edges: pairs of neighbors not already connected
            fill = 0
            for i, u in enumerate(nbrs):
                for w in nbrs[i + 1:]:
                    if w not in adj[u]:
                        fill += 1
            if fill < best_fill or (fill == best_fill and
                    (best_var is None or v < best_var)):
                best_fill = fill
                best_var = v

        # Eliminate best_var: add fill edges, remove from graph
        order.append(best_var)
        remaining.remove(best_var)
        nbrs = [u for u in adj[best_var] if u in remaining or u in exclude]
        for i, u in enumerate(nbrs):
            for w in nbrs[i + 1:]:
                adj[u].add(w)
                adj[w].add(u)
        # Remove best_var from all neighbor lists
        for u in adj[best_var]:
            adj[u].discard(best_var)
        del adj[best_var]

    return order
