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

# Credal Variable Elimination for LCNs.
#
# CredalVE is one of the credal-network inference algorithms. It consumes a
# pre-built CredalNetworkVertices (the directed graph + enumerated extreme
# points of the local credal sets) and computes lower/upper bounds on marginal
# queries via bucket-based variable elimination over the extreme points.

import itertools
import time

import numpy as np

# Local
from lcn.core.model import LCN
from lcn.inference.marginal.cn.potentials import Potential, min_fill_order
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.utils.common import check_consistency


class CredalVE:
    """
    Credal Variable Elimination for LCNs. Operates on a CredalNetworkVertices
    instance (the chain-graph credal network together with the enumerated
    extreme points of each local credal set) and computes bounds on marginal
    queries P(query | evidence) via bucket elimination over the extreme points.
    """

    def __init__(self, cnv: CredalNetworkVertices):
        """
        Args:
            cnv: CredalNetworkVertices
                A built CredalNetworkVertices (extreme points enumerated).
        """
        assert cnv.extreme_points is not None, \
            "CredalNetworkVertices must be built before passing to CredalVE."
        assert cnv.bn_min is not None
        self.cnv = cnv

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
        assert self.cnv.extreme_points is not None, \
            "CredalNetworkVertices must be built before run()."
        assert self.cnv.bn_min is not None

        t_start = time.time()

        bn = self.cnv.bn_min  # use for DAG structure
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
        for node_name, configs in self.cnv.extreme_points.items():
            nid = bn.idFromName(node_name)
            parent_ids = sorted(bn.parents(nid))
            parent_names = [bn.variable(pid).name() for pid in parent_ids]

            # Scope of the CPT potential: [node_name] + parent_names
            # We need to match the pyAgrum parent config strings to array indices
            scope = [node_name] + parent_names
            shape = tuple(cards[v] for v in scope)

            # Parse parent config strings and build vertex arrays
            # Config string format: "<>" (no parents) or "<P1:v1|P2:v2>"
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
            elim_order = min_fill_order(scopes, exclude={query})

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
        assert self.cnv.extreme_points is not None, \
            "CredalNetworkVertices must be built before run_approx()."
        assert self.cnv.bn_min is not None
        assert epsilon >= 0, "Epsilon must be non-negative."
        assert elim_heuristic in ("topological", "min-fill"), \
            f"Unknown heuristic '{elim_heuristic}'. Use 'topological' or 'min-fill'."

        t_start = time.time()

        bn = self.cnv.bn_min
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
        for node_name, configs in self.cnv.extreme_points.items():
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
            elim_order = min_fill_order(scopes, exclude={query})

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

    # Build the credal network vertices (chain-graph factorization +
    # interval local credal sets + extreme-point enumeration)
    cnv = CredalNetworkVertices.from_lcn(l, method="linear", verbosity=1)

    # Credal Variable Elimination algorithm
    cve = CredalVE(cnv=cnv)

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
