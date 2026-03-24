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

# ApproxLP: Approximate inference for credal networks via iterative
# linearization (coordinate descent over extreme points).
# Based on: Antonucci, de Campos, Huber, Zaffalon (2015).
# "Approximate credal network updating by linear programming."

import itertools
import time
from typing import Dict, List, Tuple

import numpy as np

# Local
from lcn.model import LCN
from lcn.inference.marginal.cve import CredalVE
from lcn.inference.utils import check_consistency


class ApproxLP:
    """
    Approximate inference for credal networks via iterative linearization.

    The algorithm reformulates credal marginal inference as a multilinear
    program and solves it by coordinate descent: at each step, all local
    distributions except one are fixed, reducing the problem to selecting
    the best extreme point for that variable. This produces inner bounds
    (the returned interval is contained within the true interval).

    Operates on a CredalVE instance that has already been built.
    """

    def __init__(self, cve: CredalVE):
        assert cve.extreme_points is not None, \
            "CredalVE must have build() called before passing to ApproxLP."
        assert cve.bn_min is not None

        self.cve = cve
        self.lower_bound = None
        self.upper_bound = None
        self.lower_bounds = None
        self.upper_bounds = None

    # ------------------------------------------------------------------
    # Factor graph construction (same as IBP)
    # ------------------------------------------------------------------

    def _build_factors(self):
        """
        Build factor structures from the credal network's extreme points.
        Returns (cards, factors) where each factor has keys:
        'node', 'scope', 'parents', 'vertices'.
        """
        bn = self.cve.bn_min
        cards = {}
        for nid in bn.nodes():
            name = bn.variable(nid).name()
            cards[name] = bn.variable(nid).domainSize()

        factors = []
        for node_name, configs in self.cve.extreme_points.items():
            nid = bn.idFromName(node_name)
            parent_ids = sorted(bn.parents(nid))
            parent_names = [bn.variable(pid).name() for pid in parent_ids]
            scope = [node_name] + parent_names

            vertices = {}
            for config_str, verts in configs.items():
                if config_str == "<>":
                    pc = ()
                else:
                    inner = config_str[1:-1]
                    parts = inner.split("|")
                    pvals = {}
                    for p in parts:
                        pname, pval = p.split(":")
                        pvals[pname] = int(pval)
                    pc = tuple(pvals[pn] for pn in parent_names)
                vertices[pc] = [np.array(v) for v in verts]

            factors.append({
                'node': node_name,
                'scope': scope,
                'parents': parent_names,
                'vertices': vertices
            })
        return cards, factors

    # ------------------------------------------------------------------
    # Distribution initialization and evaluation
    # ------------------------------------------------------------------

    def _init_distributions(self, factors, cards):
        """
        Initialize each local distribution to the center (average) of
        its credal set's extreme points.

        Returns:
            dist: dict mapping (node_name, parent_config) -> np.array
                  representing the selected conditional distribution.
        """
        dist = {}
        for fac in factors:
            node = fac['node']
            for pc, verts in fac['vertices'].items():
                center = np.mean(verts, axis=0)
                # Normalize to ensure it sums to 1
                s = np.sum(center)
                if s > 0:
                    center = center / s
                dist[(node, pc)] = center
        return dist

    def _eval_objective(self, dist, factors, cards, query, evidence):
        """
        Evaluate P(query | evidence) using standard variable elimination
        with the fixed distributions in `dist`.

        Returns P(query=x) as a numpy array (unnormalized joint over query).
        """
        bn = self.cve.bn_min
        node_names = [bn.variable(n).name() for n in bn.nodes()]

        # Build a single joint factor for each node using the fixed dist
        # Each factor is a numpy array over (node, parents)
        node_factors = {}
        for fac in factors:
            node = fac['node']
            parents = fac['parents']
            scope = [node] + parents
            shape = tuple(cards[v] for v in scope)
            arr = np.zeros(shape)

            if len(parents) == 0:
                all_pcs = [()]
            else:
                all_pcs = list(itertools.product(
                    *[range(cards[pn]) for pn in parents]
                ))

            for pc in all_pcs:
                p_dist = dist.get((node, pc))
                if p_dist is None:
                    p_dist = np.ones(cards[node]) / cards[node]
                for child_val in range(cards[node]):
                    idx = [slice(None)] * len(scope)
                    idx[0] = child_val
                    for pi, pn in enumerate(parents):
                        idx[1 + pi] = pc[pi]
                    arr[tuple(idx)] = p_dist[child_val]

            node_factors[node] = (scope, arr)

        # Add evidence indicators
        ev_factors = {}
        for ev_var, ev_val in evidence.items():
            indicator = np.zeros(cards[ev_var])
            indicator[ev_val] = 1.0
            ev_factors[ev_var] = ([ev_var], indicator)

        # Variable elimination
        # Elimination order: all vars except query
        topo = list(bn.topologicalOrder())
        topo_names = [bn.variable(nid).name() for nid in topo]
        elim_order = [v for v in topo_names
                      if v != query and v not in evidence]
        elim_order += [v for v in topo_names if v in evidence]

        # Collect all factors
        all_factors = {}
        for node, (scope, arr) in node_factors.items():
            all_factors[f"node_{node}"] = (scope, arr)
        for ev_var, (scope, arr) in ev_factors.items():
            all_factors[f"ev_{ev_var}"] = (scope, arr)

        # Convert to list of (scope, array) pairs
        factor_list = list(all_factors.values())

        for var in elim_order:
            # Collect factors mentioning var
            bucket = [(s, a) for s, a in factor_list if var in s]
            rest = [(s, a) for s, a in factor_list if var not in s]

            if not bucket:
                factor_list = rest
                continue

            # Combine factors in bucket
            combined_scope, combined_arr = bucket[0]
            for s, a in bucket[1:]:
                combined_scope, combined_arr = self._combine_arrays(
                    combined_scope, combined_arr, s, a, cards)

            # Marginalize out var
            axis = combined_scope.index(var)
            new_scope = [v for v in combined_scope if v != var]
            new_arr = np.sum(combined_arr, axis=axis)

            factor_list = rest + [(new_scope, new_arr)]

        # Combine remaining factors (should be over query only)
        if not factor_list:
            return np.ones(cards[query]) / cards[query]

        result_scope, result_arr = factor_list[0]
        for s, a in factor_list[1:]:
            result_scope, result_arr = self._combine_arrays(
                result_scope, result_arr, s, a, cards)

        # Result should be over query
        # Marginalize out any remaining vars except query
        for v in list(result_scope):
            if v != query:
                axis = result_scope.index(v)
                result_scope = [x for x in result_scope if x != v]
                result_arr = np.sum(result_arr, axis=axis)

        return result_arr

    @staticmethod
    def _combine_arrays(scope1, arr1, scope2, arr2, cards):
        """Combine two factor arrays by broadcasting over joint scope."""
        new_scope = list(scope1)
        for v in scope2:
            if v not in new_scope:
                new_scope.append(v)

        def _expand(arr, src_scope, tgt_scope, cards):
            src_idx = {v: i for i, v in enumerate(src_scope)}
            src_order = [v for v in tgt_scope if v in src_idx]
            transpose_perm = [src_idx[v] for v in src_order]
            a = np.transpose(arr, transpose_perm)
            result_shape = []
            for v in tgt_scope:
                if v in src_idx:
                    result_shape.append(cards[v])
                else:
                    result_shape.append(1)
            return a.reshape(result_shape)

        a1 = _expand(arr1, scope1, new_scope, cards)
        a2 = _expand(arr2, scope2, new_scope, cards)
        return new_scope, a1 * a2

    # ------------------------------------------------------------------
    # Coordinate descent
    # ------------------------------------------------------------------

    def _coordinate_descent(self, factors, cards, query, evidence,
                            sense, n_iters, verbosity):
        """
        Run coordinate descent over extreme points to optimize P(query|evidence).

        Args:
            sense: "min" or "max"
        Returns:
            (objective_value, final_distribution_over_query)
        """
        dist = self._init_distributions(factors, cards)

        # Evaluate initial objective
        q_arr = self._eval_objective(dist, factors, cards, query, evidence)
        total = np.sum(q_arr)
        if total <= 0:
            return None, np.ones(cards[query]) / cards[query]

        best_probs = q_arr / total
        if sense == "min":
            best_obj = best_probs[1] if cards[query] > 1 else best_probs[0]
        else:
            best_obj = best_probs[1] if cards[query] > 1 else best_probs[0]

        for iteration in range(n_iters):
            improved = False

            for fac in factors:
                node = fac['node']
                for pc, verts in fac['vertices'].items():
                    if len(verts) <= 1:
                        continue  # only one vertex, nothing to optimize

                    current_v = dist[(node, pc)].copy()
                    best_v = current_v

                    for v in verts:
                        dist[(node, pc)] = v
                        q_arr = self._eval_objective(
                            dist, factors, cards, query, evidence)
                        total = np.sum(q_arr)
                        if total <= 0:
                            continue
                        probs = q_arr / total
                        obj = probs[1] if cards[query] > 1 else probs[0]

                        if sense == "min" and obj < best_obj:
                            best_obj = obj
                            best_probs = probs
                            best_v = v.copy()
                            improved = True
                        elif sense == "max" and obj > best_obj:
                            best_obj = obj
                            best_probs = probs
                            best_v = v.copy()
                            improved = True

                    dist[(node, pc)] = best_v

            if verbosity > 1:
                print(f"  [{sense}] Iteration {iteration}: obj={best_obj:.6f}")

            if not improved:
                if verbosity > 1:
                    print(f"  [{sense}] Converged at iteration {iteration}")
                break

        return best_obj, best_probs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str, evidence: dict = {},
            n_iters: int = 50, verbosity: int = 1):
        """
        Compute lower and upper bounds on P(query | evidence) using
        ApproxLP (coordinate descent over extreme points).

        Produces inner bounds: the returned interval is contained within
        (or equal to) the true interval.

        Args:
            query: Name of the query variable.
            evidence: {variable_name: value} for observed variables.
            n_iters: Maximum coordinate descent iterations.
            verbosity: 0=silent, 1=summary, 2=detailed.
        """
        t_start = time.time()

        cards, factors = self._build_factors()
        bn = self.cve.bn_min
        node_names = [bn.variable(n).name() for n in bn.nodes()]
        assert query in node_names, \
            f"Query variable '{query}' not found."

        if verbosity > 0:
            print(f"[ApproxLP] Query: {query}")
            print(f"[ApproxLP] Evidence: {evidence}")

        # Minimize for lower bound
        lo_obj, lo_probs = self._coordinate_descent(
            factors, cards, query, evidence, "min", n_iters, verbosity)

        # Maximize for upper bound
        hi_obj, hi_probs = self._coordinate_descent(
            factors, cards, query, evidence, "max", n_iters, verbosity)

        t_end = time.time()

        # Assemble full lower/upper arrays across all query states
        k = cards[query]
        lower_bounds = np.ones(k)
        upper_bounds = np.zeros(k)

        # The min/max coordinate descent optimizes state=1 (or state=0).
        # To get proper per-state bounds, we track both min and max probs
        # from the two runs.
        for probs in [lo_probs, hi_probs]:
            if probs is not None:
                for val in range(k):
                    lower_bounds[val] = min(lower_bounds[val], probs[val])
                    upper_bounds[val] = max(upper_bounds[val], probs[val])

        self.lower_bound = lower_bounds[1] if k > 1 else lower_bounds[0]
        self.upper_bound = upper_bounds[1] if k > 1 else upper_bounds[0]
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        if verbosity > 0:
            print(f"[ApproxLP] Results for P({query} | {evidence}):")
            for val in range(k):
                print(f"  P({query}={val}): "
                      f"[{lower_bounds[val]:.6f}, {upper_bounds[val]:.6f}]")
            print(f"[ApproxLP] Time elapsed: {t_end - t_start:.4f} sec")


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

    # Build the CredalVE (needed for extreme points)
    cve = CredalVE(lcn=l)
    cve.build(verbosity=0)

    # Create ApproxLP solver
    alp = ApproxLP(cve=cve)

    # Run queries
    print("\n=== ApproxLP ===")
    alp.run(query="B", evidence={}, verbosity=1)
    alp.run(query="E", evidence={}, verbosity=1)
    alp.run(query="A", evidence={"B": 0, "E": 0}, verbosity=1)

    # Compare with exact VE
    print("\n=== Exact VE (for comparison) ===")
    cve.run(query="B", evidence={}, verbosity=1)
    cve.run(query="A", evidence={"B": 0, "E": 0}, verbosity=1)
