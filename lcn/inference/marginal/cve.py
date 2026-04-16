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

# Credal Variable Elimination for LCNs

import re
import itertools
import time
from typing import Dict, List, Tuple

import numpy as np
import pyagrum as gum  # noqa: N813

# Local
from lcn.core.model import LCN
from lcn.inference.marginal.factorization import Factorization
from lcn.inference.utils.common import check_consistency


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
            extra_count = 0
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
        kept = []
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
        d = flat.shape[1]

        # Initialize centroids: pick k distinct indices at random
        rng = np.random.RandomState(42)
        indices = rng.choice(n, size=n_clusters, replace=False)
        centroids = flat[indices].copy()  # (k, d)

        assignments = np.zeros(n, dtype=int)

        for _ in range(max_iters):
            # Assignment: each function -> nearest centroid (Manhattan)
            new_assignments = np.empty(n, dtype=int)
            for i in range(n):
                dists = np.sum(np.abs(centroids - flat[i]), axis=1)
                new_assignments[i] = np.argmin(dists)

            # Check convergence
            if np.array_equal(assignments, new_assignments):
                break
            assignments = new_assignments

            # Update centroids: mean of assigned functions
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


def _parse_credal_net_vertices(cn: gum.CredalNet) -> Dict:
    """
    Parse the string representation of a CredalNet to extract the extreme
    points (vertices) of each credal set for every node and parent config.

    Returns:
        A dict: {node_name: {parent_config_str: [[v0, v1, ...], ...]}}
    """
    result = {}
    cn_str = str(cn)
    # Split into per-node blocks separated by blank lines
    blocks = cn_str.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue
        # First line: "NodeName:Labelized({0|1|...})"
        header = lines[0]
        node_name = header.split(":")[0].strip()
        result[node_name] = {}
        for line in lines[1:]:
            # Format: "<parent_config> : [[v1, v2], [v3, v4], ...]"
            match = re.match(r"^(<[^>]*>)\s*:\s*\[(.+)\]\s*$", line.strip())
            if not match:
                continue
            parent_config = match.group(1)
            vertices_str = match.group(2)
            # Parse nested lists: [[0.3 , 0.7] , [0.5 , 0.5]]
            vertices = []
            for vm in re.finditer(r"\[([^\[\]]+)\]", vertices_str):
                vals = [float(x.strip()) for x in vm.group(1).split(",")]
                vertices.append(vals)
            result[node_name][parent_config] = vertices
    return result


class CredalVE:
    """
    Credal Variable Elimination for LCNs. Builds a factorization from the
    LCN's chain graph structure, estimates local credal sets via linear
    fractional programs, assembles a pyAgrum CredalNet, and uses LRS to
    enumerate the extreme points of each credal set.
    """

    def __init__(self, lcn: LCN):
        self.lcn = lcn
        self.factorization = None
        self.factors = None
        self.bn_min = None
        self.bn_max = None
        self.credal_net = None
        self.extreme_points = None

    def build(self, verbosity: int = 1,
              factorization_method: str = "linear") -> Dict:
        """
        Build the credal network from the LCN factorization and enumerate
        extreme points via LRS.

        Args:
            verbosity: int
                Verbosity level (0 is silent).
            factorization_method: str
                "linear" for standard LP/fractional LP factorization,
                "nlp" for nonlinear program with pairwise independence.

        Returns:
            A dict of extreme points per node per parent configuration.
        """

        # Step 1: Ensure the LCN structure is built
        if self.lcn.primal_graph is None:
            self.lcn.build_primal_graph()
        if self.lcn.structure_graph is None:
            self.lcn.build_structure_graph()
        if self.lcn.simplified_structure_graph is None:
            self.lcn.simplify_structure_graph()
        assert self.lcn.is_chain_graph(), "The LCN must be a chain graph."
        if self.lcn.families is None:
            self.lcn.process_chain_graph()

        # Step 2: Run the factorization to get local intervals
        self.factorization = Factorization(self.lcn)
        self.factors = self.factorization.build(method=factorization_method)

        if verbosity > 0:
            print(f"[CredalVE] Factorization produced {len(self.factors)} factors.")

        # Step 3: Collect node info from the simplified structure graph
        sg = self.lcn.simplified_structure_graph
        node_names = list(sg.get_nodes())

        # For each node, determine its cardinality (2^k for compound nodes)
        node_card = {}
        node_atoms = {}  # node_name -> list of atoms
        for name in node_names:
            atoms = name.split("-") if "-" in name else [name]
            node_atoms[name] = atoms
            node_card[name] = 2 ** len(atoms)

        # Step 4: Build two BayesNets (bn_min for lower, bn_max for upper)
        self.bn_min = gum.BayesNet("min")
        self.bn_max = gum.BayesNet("max")
        node_ids_min = {}
        node_ids_max = {}

        for name in node_names:
            card = node_card[name]
            nid_min = self.bn_min.add(gum.LabelizedVariable(name, name, card))
            nid_max = self.bn_max.add(gum.LabelizedVariable(name, name, card))
            node_ids_min[name] = nid_min
            node_ids_max[name] = nid_max

        # Add arcs from parents to children (matching the simplified structure)
        for family in self.lcn.families:
            child = family["child"]
            for parent in family["parents"]:
                self.bn_min.addArc(node_ids_min[parent], node_ids_min[child])
                self.bn_max.addArc(node_ids_max[parent], node_ids_max[child])

        # Step 5: Fill CPTs from factorization results
        for factor_idx, factor in enumerate(self.factors):
            sample_entry = factor[0]
            child_name = sample_entry["child"]
            parent_names = sample_entry["parents"]
            scope = sample_entry["scope"]

            child_atoms = node_atoms[child_name]
            n_child_states = node_card[child_name]

            # Determine which scope variables are child vs parent atoms
            # scope = [child_atoms..., parent_atoms...]
            n_child_atoms = len(child_atoms)

            # Build a mapping from (parent_config, child_state) to (lobo, upbo)
            # parent_config: tuple of 0/1 for each parent atom
            # child_state: integer index (binary encoding of child atoms)
            bounds = {}  # (parent_config_tuple, child_state_int) -> (lobo, upbo)

            for i, entry in factor.items():
                interp = entry["interpretation"]
                child_vals = interp[:n_child_atoms]
                parent_vals = interp[n_child_atoms:]
                # Convert child vals to a state index (binary -> int)
                child_state = 0
                for bit in child_vals:
                    child_state = (child_state << 1) | bit
                parent_config = tuple(parent_vals)
                bounds[(parent_config, child_state)] = (entry["lobo"], entry["upbo"])

            # Now fill the CPTs using pyAgrum's Instantiation to get correct ordering
            nid_min = node_ids_min[child_name]
            nid_max = node_ids_max[child_name]

            if len(parent_names) == 0:
                # No parents: just fill the marginal
                lower_vals = []
                upper_vals = []
                for cs in range(n_child_states):
                    lo, up = bounds.get(((), cs), (0.0, 1.0))
                    lo = lo if lo is not None else 0.0
                    up = up if up is not None else 1.0
                    lower_vals.append(abs(lo))
                    upper_vals.append(abs(up))
                self.bn_min.cpt(nid_min).fillWith(lower_vals)
                self.bn_max.cpt(nid_max).fillWith(upper_vals)
            else:
                # With parents: iterate using pyAgrum's Instantiation order
                # to build the flat CPT array
                parent_atoms = []
                for pname in parent_names:
                    parent_atoms.extend(node_atoms[pname])

                inst = gum.Instantiation(self.bn_min.cpt(nid_min))
                lower_flat = [0.0] * inst.domainSize()
                upper_flat = [0.0] * inst.domainSize()

                inst.setFirst()
                flat_idx = 0
                while not inst.end():
                    # Extract child state from the instantiation
                    child_state = inst.val(inst.variable(child_name))

                    # Extract parent values in scope order (matching factorization)
                    parent_config = []
                    for patom in parent_atoms:
                        # Find which node this atom belongs to
                        for pname in parent_names:
                            if patom in node_atoms[pname]:
                                pnode = pname
                                break
                        p_node_val = inst.val(inst.variable(pnode))
                        p_node_atoms = node_atoms[pnode]
                        if len(p_node_atoms) == 1:
                            parent_config.append(p_node_val)
                        else:
                            # Compound parent: decode the state into individual bits
                            n_bits = len(p_node_atoms)
                            atom_idx = p_node_atoms.index(patom)
                            bit = (p_node_val >> (n_bits - 1 - atom_idx)) & 1
                            parent_config.append(bit)

                    parent_config = tuple(parent_config)
                    lo, up = bounds.get((parent_config, child_state), (0.0, 1.0))
                    lo = lo if lo is not None else 0.0
                    up = up if up is not None else 1.0
                    lower_flat[flat_idx] = abs(lo)
                    upper_flat[flat_idx] = abs(up)
                    flat_idx += 1
                    inst.inc()

                self.bn_min.cpt(nid_min).fillWith(lower_flat)
                self.bn_max.cpt(nid_max).fillWith(upper_flat)

        if verbosity > 0:
            print("[CredalVE] Lower BN CPTs:")
            for name in node_names:
                print(f"  {name}: {self.bn_min.cpt(node_ids_min[name])}")
            print("[CredalVE] Upper BN CPTs:")
            for name in node_names:
                print(f"  {name}: {self.bn_max.cpt(node_ids_max[name])}")

        # Step 6: Create the CredalNet and run LRS vertex enumeration
        self.credal_net = gum.CredalNet(self.bn_min, self.bn_max)
        self.credal_net.intervalToCredal()

        if verbosity > 0:
            print("[CredalVE] CredalNet vertices:")
            print(self.credal_net)

        # Step 7: Extract and store extreme points
        self.extreme_points = _parse_credal_net_vertices(self.credal_net)

        if verbosity > 0:
            print("[CredalVE] Extreme points per node:")
            for node, configs in self.extreme_points.items():
                for config, vertices in configs.items():
                    print(f"  {node} {config}: {len(vertices)} vertices")
                    for v in vertices:
                        v_str = ", ".join(f"{x:.4f}" for x in v)
                        print(f"    [{v_str}]")

        return self.extreme_points

    @staticmethod
    def _min_fill_order(scopes: List[List[str]], exclude: set) -> List[str]:
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

    def run(self, query: str, evidence: dict = {},
            elim_heuristic: str = "topological", epsilon: float = None,
            verbosity: int = 1):
        """
        Compute lower and upper bounds on P(query_var | evidence) using
        bucket-based variable elimination over the credal network's
        extreme points.

        Args:
            query: str
                Name of the query variable (a node in the credal network).
            evidence: dict
                {variable_name: value} for observed variables.
            elim_heuristic: str
                Elimination ordering heuristic: "topological" (default)
                or "min-fill".
            epsilon: float or None
                If not None, use epsilon-approximate pruning instead of
                exact pruning. Larger values prune more aggressively,
                producing wider (outer) bounds but faster computation.
            verbosity: int
                Verbosity level (0 is silent).
        """
        if epsilon is not None:
            return self.run_approx(query, evidence, epsilon,
                                   elim_heuristic, verbosity)

        assert elim_heuristic in ("topological", "min-fill"), \
            f"Unknown heuristic '{elim_heuristic}'. Use 'topological' or 'min-fill'."
        assert self.extreme_points is not None, \
            "Must call build() before run()."
        assert self.bn_min is not None

        t_start = time.time()

        bn = self.bn_min  # use for DAG structure
        node_names = [bn.variable(n).name() for n in bn.nodes()]
        assert query in node_names, \
            f"Query variable '{query}' not found in credal network."

        # Variable cardinalities
        cards = {}
        for nid in bn.nodes():
            name = bn.variable(nid).name()
            cards[name] = bn.variable(nid).domainSize()

        # Step 1: Build initial potentials from extreme points
        potentials = []
        for node_name, configs in self.extreme_points.items():
            nid = bn.idFromName(node_name)
            parent_ids = sorted(bn.parents(nid))
            parent_names = [bn.variable(pid).name() for pid in parent_ids]

            # Scope of the CPT potential: [node_name] + parent_names
            # We need to match the pyAgrum parent config strings to array indices
            scope = [node_name] + parent_names
            shape = tuple(cards[v] for v in scope)

            # Parse parent config strings and build vertex arrays
            # Config string format: "<>" (no parents) or "<P1:v1|P2:v2>"
            n_vertices_per_config = {}
            config_map = {}  # parent_config_tuple -> list of vertices

            for config_str, vertices in configs.items():
                if config_str == "<>":
                    parent_config = ()
                else:
                    # Parse "<B:0|E:1>" -> {B: 0, E: 1}
                    inner = config_str[1:-1]  # strip < >
                    parts = inner.split("|")
                    pvals = {}
                    for p in parts:
                        pname, pval = p.split(":")
                        pvals[pname] = int(pval)
                    parent_config = tuple(pvals[pn] for pn in parent_names)
                config_map[parent_config] = vertices

            # Get all parent configs in order
            if len(parent_names) == 0:
                all_parent_configs = [()]
            else:
                all_parent_configs = list(itertools.product(
                    *[range(cards[pn]) for pn in parent_names]
                ))

            # Count total vertices (product over parent configs)
            vertex_counts = [len(config_map.get(pc, [[]])) for pc in all_parent_configs]
            # Each full CPT vertex is a combination: one vertex per parent config
            vertex_combos = list(itertools.product(
                *[range(c) for c in vertex_counts]
            ))

            functions = []
            for combo in vertex_combos:
                arr = np.zeros(shape)
                for pc_idx, pc in enumerate(all_parent_configs):
                    v_idx = combo[pc_idx]
                    verts = config_map.get(pc, [[1.0 / cards[node_name]] * cards[node_name]])
                    vertex = verts[v_idx]
                    # Fill the array slice for this parent config
                    for child_val in range(cards[node_name]):
                        idx = [slice(None)] * len(scope)
                        idx[0] = child_val  # child is first in scope
                        for pi, pn in enumerate(parent_names):
                            idx[1 + pi] = pc[pi]
                        arr[tuple(idx)] = vertex[child_val]
                functions.append(arr)

            potentials.append(Potential(scope, cards, functions))

        # Step 2: Add evidence potentials (indicator functions)
        for ev_var, ev_val in evidence.items():
            assert ev_var in node_names, \
                f"Evidence variable '{ev_var}' not found."
            indicator = np.zeros(cards[ev_var])
            indicator[ev_val] = 1.0
            potentials.append(Potential([ev_var], cards, [indicator]))

        # Step 3: Determine elimination ordering
        if elim_heuristic == "topological":
            topo = list(bn.topologicalOrder())
            topo_names = [bn.variable(nid).name() for nid in topo]
            elim_order = [v for v in topo_names
                          if v != query and v not in evidence]
            # Evidence variables are already fixed by indicator potentials,
            # but we still need to eliminate them
            elim_order += [v for v in topo_names if v in evidence]
        else:  # min-fill
            scopes = [p.scope for p in potentials]
            elim_order = self._min_fill_order(scopes, exclude={query})

        if verbosity > 0:
            print(f"[CredalVE] Query: {query}")
            print(f"[CredalVE] Evidence: {evidence}")
            print(f"[CredalVE] Elimination order ({elim_heuristic}): {elim_order}")
            total_funcs = sum(len(p.functions) for p in potentials)
            print(f"[CredalVE] Initial potentials: {len(potentials)}, "
                  f"total functions: {total_funcs}")

        # Step 4: Bucket elimination
        for var in elim_order:
            # Collect potentials that mention this variable
            bucket = [p for p in potentials if var in p.scope]
            rest = [p for p in potentials if var not in p.scope]

            if not bucket:
                potentials = rest
                continue

            # Combine all potentials in the bucket
            combined = bucket[0]
            for p in bucket[1:]:
                combined = combined.combine(p)

            # Marginalize out the variable
            result = combined.marginalize(var)

            # Prune dominated functions
            result = result.prune()

            if verbosity > 1:
                print(f"  Eliminated {var}: "
                      f"{len(combined.functions)} -> {len(result.functions)} functions, "
                      f"scope {result.scope}")

            potentials = rest + [result]

        # Step 5: Combine remaining potentials
        if len(potentials) == 0:
            self.lower_bound = 0.0
            self.upper_bound = 1.0
            return

        final = potentials[0]
        for p in potentials[1:]:
            final = final.combine(p)

        # Final potential should have scope = [query]
        assert final.scope == [query] or set(final.scope) == {query}, \
            f"Final potential scope {final.scope} should be [{query}]"

        # Step 6: Extract lower and upper bounds for each query value
        lower_bounds = np.ones(cards[query])
        upper_bounds = np.zeros(cards[query])

        for f in final.functions:
            total = np.sum(f)
            if total <= 0:
                continue
            probs = f / total
            for val in range(cards[query]):
                lower_bounds[val] = min(lower_bounds[val], probs[val])
                upper_bounds[val] = max(upper_bounds[val], probs[val])

        t_end = time.time()

        # Store results for the positive (=1) state by default,
        # but also store the full interval arrays
        self.lower_bound = lower_bounds[1] if cards[query] > 1 else lower_bounds[0]
        self.upper_bound = upper_bounds[1] if cards[query] > 1 else upper_bounds[0]
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        if verbosity > 0:
            print(f"[CredalVE] Results for P({query} | {evidence}):")
            for val in range(cards[query]):
                print(f"  P({query}={val}): "
                      f"[{lower_bounds[val]:.6f}, {upper_bounds[val]:.6f}]")
            print(f"[CredalVE] Time elapsed: {t_end - t_start:.4f} sec")

    def run_approx(self, query: str, evidence: dict,
                   epsilon: float, elim_heuristic: str = "topological",
                   verbosity: int = 1):
        """
        Epsilon-approximate credal variable elimination. Same algorithm as
        run() but uses epsilon-approximate pruning at each elimination step,
        which allows slightly dominated functions to be removed. This bounds
        the size of intermediate potentials, yielding an FPTAS (fully
        polynomial-time approximation scheme) with error at most epsilon.

        See: Mauá et al. (2012), "Solving limited memory influence diagrams"
        and Mauá & Cozman (2020), "Thirty years of credal networks", Sec 5.2.

        Args:
            query: str
                Name of the query variable (a node in the credal network).
            evidence: dict
                {variable_name: value} for observed variables.
            epsilon: float
                Approximation tolerance. Larger values prune more
                aggressively (fewer functions kept, faster, wider bounds).
            elim_heuristic: str
                Elimination ordering heuristic: "topological" or "min-fill".
            verbosity: int
                Verbosity level (0 is silent).
        """
        assert self.extreme_points is not None, \
            "Must call build() before run_approx()."
        assert self.bn_min is not None
        assert epsilon >= 0, "Epsilon must be non-negative."
        assert elim_heuristic in ("topological", "min-fill"), \
            f"Unknown heuristic '{elim_heuristic}'. Use 'topological' or 'min-fill'."

        t_start = time.time()

        bn = self.bn_min
        node_names = [bn.variable(n).name() for n in bn.nodes()]
        assert query in node_names, \
            f"Query variable '{query}' not found in credal network."

        # Variable cardinalities
        cards = {}
        for nid in bn.nodes():
            name = bn.variable(nid).name()
            cards[name] = bn.variable(nid).domainSize()

        # Step 1: Build initial potentials from extreme points
        # (identical to run())
        potentials = []
        for node_name, configs in self.extreme_points.items():
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

        # Step 2: Add evidence potentials
        for ev_var, ev_val in evidence.items():
            assert ev_var in node_names, \
                f"Evidence variable '{ev_var}' not found."
            indicator = np.zeros(cards[ev_var])
            indicator[ev_val] = 1.0
            potentials.append(Potential([ev_var], cards, [indicator]))

        # Step 3: Determine elimination ordering
        if elim_heuristic == "topological":
            topo = list(bn.topologicalOrder())
            topo_names = [bn.variable(nid).name() for nid in topo]
            elim_order = [v for v in topo_names
                          if v != query and v not in evidence]
            elim_order += [v for v in topo_names if v in evidence]
        else:
            scopes = [p.scope for p in potentials]
            elim_order = self._min_fill_order(scopes, exclude={query})

        if verbosity > 0:
            print(f"[CredalVE-approx] Query: {query}, epsilon: {epsilon}")
            print(f"[CredalVE-approx] Evidence: {evidence}")
            print(f"[CredalVE-approx] Elimination order ({elim_heuristic}): "
                  f"{elim_order}")
            total_funcs = sum(len(p.functions) for p in potentials)
            print(f"[CredalVE-approx] Initial potentials: {len(potentials)}, "
                  f"total functions: {total_funcs}")

        # Step 4: Bucket elimination with epsilon-pruning
        for var in elim_order:
            bucket = [p for p in potentials if var in p.scope]
            rest = [p for p in potentials if var not in p.scope]

            if not bucket:
                potentials = rest
                continue

            combined = bucket[0]
            for p in bucket[1:]:
                combined = combined.combine(p)

            result = combined.marginalize(var)

            # Epsilon-approximate pruning instead of exact pruning
            result = result.epsilon_prune(epsilon)

            if verbosity > 1:
                print(f"  Eliminated {var}: "
                      f"{len(combined.functions)} -> "
                      f"{len(result.functions)} functions, "
                      f"scope {result.scope}")

            potentials = rest + [result]

        # Step 5: Combine remaining potentials
        if len(potentials) == 0:
            self.lower_bound = 0.0
            self.upper_bound = 1.0
            return

        final = potentials[0]
        for p in potentials[1:]:
            final = final.combine(p)

        assert final.scope == [query] or set(final.scope) == {query}, \
            f"Final potential scope {final.scope} should be [{query}]"

        # Step 6: Extract lower and upper bounds
        lower_bounds = np.ones(cards[query])
        upper_bounds = np.zeros(cards[query])

        for f in final.functions:
            total = np.sum(f)
            if total <= 0:
                continue
            probs = f / total
            for val in range(cards[query]):
                lower_bounds[val] = min(lower_bounds[val], probs[val])
                upper_bounds[val] = max(upper_bounds[val], probs[val])

        t_end = time.time()

        self.lower_bound = lower_bounds[1] if cards[query] > 1 else lower_bounds[0]
        self.upper_bound = upper_bounds[1] if cards[query] > 1 else upper_bounds[0]
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        if verbosity > 0:
            print(f"[CredalVE-approx] Results for P({query} | {evidence}):")
            for val in range(cards[query]):
                print(f"  P({query}={val}): "
                      f"[{lower_bounds[val]:.6f}, {upper_bounds[val]:.6f}]")
            print(f"[CredalVE-approx] Time elapsed: {t_end - t_start:.4f} sec")


if __name__ == "__main__":

    # Load the LCN
    file_name = "examples/alarm.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Check consistency
    ok = check_consistency(l)
    if ok:
        print("CONSISTENT")
    else:
        print("INCONSISTENT")

    # Build the CredalVE
    cve = CredalVE(lcn=l)
    cve.build(verbosity=1)

    # Variable elimination methods
    queries = [
        ("B", {}),
        ("A", {"B": 0, "E": 0}),
    ]

    for q, ev in queries:
        ev_str = str(ev) if ev else "{}"
        print(f"\n=== P({q} | {ev_str}) ===")

        cve.run(query=q, evidence=ev, verbosity=0)
        print(f"  VE (exact):      P({q}=1) in "
              f"[{cve.lower_bound:.6f}, {cve.upper_bound:.6f}]")

        cve.run(query=q, evidence=ev, epsilon=0.01, verbosity=0)
        print(f"  VE (eps=0.01):   P({q}=1) in "
              f"[{cve.lower_bound:.6f}, {cve.upper_bound:.6f}]")
