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

# The min-fill elimination-ordering heuristic shared by the credal-network
# engines (CredalVE, CredalJT, ApproxLP). Free of any pyAgrum / LCN coupling.
# The potentials of credal variable elimination are in pair_potentials.py.

from typing import List


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
