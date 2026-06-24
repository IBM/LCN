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
from lcn.inference.marginal.cn.coupling import CouplingConstraints
from lcn.inference.marginal.cn.potentials import min_fill_order
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.utils.common import check_consistency, make_ipopt


class ApproxLP:
    """
    Approximate inference for credal networks via iterative linearization.

    The algorithm reformulates credal marginal inference as a multilinear
    program and solves it by coordinate descent: at each step, all local
    distributions except one are fixed, reducing the problem to selecting
    the best extreme point for that variable. This produces inner bounds
    (the returned interval is contained within the true interval).

    Operates on a CredalNetworkVertices instance that has already been built.
    """

    def __init__(self, cnv: CredalNetworkVertices):
        assert cnv.extreme_points is not None, \
            "CredalNetworkVertices must be built before passing to ApproxLP."
        assert cnv.bn_min is not None

        self.cnv = cnv
        self.marginals = None
        self.singleton_marginals = None

    # ------------------------------------------------------------------
    # Factor graph construction (same as IBP)
    # ------------------------------------------------------------------

    def _build_factors(self):
        """
        Build factor structures from the credal network's extreme points.
        Returns (cards, factors) where each factor has keys:
        'node', 'scope', 'parents', 'vertices'.
        """
        bn = self.cnv.bn_min
        cards = {}
        for nid in bn.nodes():
            name = bn.variable(nid).name()
            cards[name] = bn.variable(nid).domainSize()

        factors = []
        for node_name, configs in self.cnv.extreme_points.items():
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
        bn = self.cnv.bn_min
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

        # Variable elimination (min-fill ordering, excluding query)
        # Collect all factors
        all_factors = {}
        for node, (scope, arr) in node_factors.items():
            all_factors[f"node_{node}"] = (scope, arr)
        for ev_var, (scope, arr) in ev_factors.items():
            all_factors[f"ev_{ev_var}"] = (scope, arr)

        # Convert to list of (scope, array) pairs
        factor_list = list(all_factors.values())

        # Compute min-fill elimination order excluding the query variable
        scopes = [list(scope) for scope, _ in factor_list]
        elim_order = min_fill_order(scopes, exclude={query})

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

    def _joint_feasible(self, dist, factors, cards, constraints):
        """
        Scheme D4: True if the full joint reconstructed from the current vertex
        choices `dist` satisfies every cross-family coupling constraint. The
        joint is the product of the per-node factors over ALL nodes, so its
        scope covers every constraint (the checkability gate always passes).
        Returns True immediately when coupling is disabled.
        """
        if constraints is None:
            return True
        # Build the product joint over all nodes (one multi-axis array) by
        # combining each node's fixed-vertex factor.
        scope = None
        joint = None
        for fac in factors:
            node = fac['node']
            parents = fac['parents']
            f_scope = [node] + parents
            shape = tuple(cards[v] for v in f_scope)
            arr = np.zeros(shape)
            all_pcs = ([()] if len(parents) == 0
                       else list(itertools.product(
                           *[range(cards[pn]) for pn in parents])))
            for pc in all_pcs:
                p_dist = dist.get((node, pc))
                if p_dist is None:
                    p_dist = np.ones(cards[node]) / cards[node]
                for child_val in range(cards[node]):
                    idx = [slice(None)] * len(f_scope)
                    idx[0] = child_val
                    for pi, pn in enumerate(parents):
                        idx[1 + pi] = pc[pi]
                    arr[tuple(idx)] = p_dist[child_val]
            if scope is None:
                scope, joint = f_scope, arr
            else:
                scope, joint = self._combine_arrays(scope, joint, f_scope, arr,
                                                    cards)
        node_atoms = self.cnv.cn.node_atoms
        return constraints.is_feasible(joint, scope, node_atoms, cards)

    def _coordinate_descent(self, factors, cards, query, evidence,
                            sense, n_iters, verbosity, constraints=None):
        """
        Run coordinate descent over extreme points to optimize P(query|evidence).

        Args:
            sense: "min" or "max"
            constraints: optional D4 CouplingConstraints; when set, vertex picks
                that make the full reconstructed joint infeasible are rejected.
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
                        # Scheme D4: reject a pick that makes the full joint
                        # violate a cross-family constraint.
                        if not self._joint_feasible(dist, factors, cards,
                                                    constraints):
                            continue
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

    def run(self, evidence: dict = {},
            n_iters: int = 50,
            coupling: str = "off",
            verbosity: int = 1) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute lower and upper bounds on the marginal of EVERY
        non-evidence variable using ApproxLP (coordinate descent over
        extreme points).

        Produces inner bounds: the returned interval is contained within
        (or equal to) the true interval.

        Args:
            evidence: {variable_name: value} for observed variables.
            n_iters: Maximum coordinate descent iterations.
            coupling: Scheme D4 (docs/tighter_approximation.tex). "off"
                (default) selects vertices freely over the strong extension
                (byte-identical to today). "cross-family" rejects any vertex
                pick that makes the fully reconstructed joint violate a
                cross-family LCN sentence or LMC assertion.
                EXPERIMENTAL: ApproxLP is an inner, greedy coordinate-descent
                method and is the loosest fit for D4 (see the design note). The
                reject-pick keeps every visited distribution inside the coupled
                feasible region, but when the center initialization itself
                violates a constraint, coordinate descent can find no feasible
                improving move and remain stuck at the init (e.g. reporting a
                degenerate point such as [0.5, 0.5]). A coupled ApproxLP bound is
                therefore NOT a certified inner bound of the coupled set; for a
                trustworthy coupled bracket use coupled CVE (outer) instead.
            verbosity: 0=silent, 1=summary, 2=detailed.

        Returns:
            Dict mapping variable name to (lower_bounds, upper_bounds)
            numpy arrays. Includes both compound and singleton marginals.
        """
        assert coupling in ("off", "cross-family"), \
            f"Unknown coupling '{coupling}'. Use 'off' or 'cross-family'."
        t_start = time.time()

        cards, factors = self._build_factors()
        bn = self.cnv.bn_min
        node_names = [bn.variable(n).name() for n in bn.nodes()]
        evidence_set = set(evidence.keys())

        # Scheme D4: cross-family coupling constraints (None when disabled).
        constraints = None
        if coupling == "cross-family":
            cc = CouplingConstraints.from_lcn(
                self.cnv.lcn, self.cnv.cn.factorization.factors)
            constraints = cc if len(cc) > 0 else None
            if verbosity > 0:
                print(f"[ApproxLP] D4 coupling: {len(cc)} cross-family "
                      f"constraint(s)")

        if verbosity > 0:
            print(f"[ApproxLP] Computing all marginals")
            print(f"[ApproxLP] Evidence: {evidence}")
            print(f"[ApproxLP] Variables: {node_names}")

        # Compute marginals for each non-evidence variable
        self.marginals = {}
        for query in node_names:
            if query in evidence_set:
                # Evidence variable: point distribution
                k = cards[query]
                lo = np.zeros(k)
                hi = np.zeros(k)
                lo[evidence[query]] = 1.0
                hi[evidence[query]] = 1.0
                self.marginals[query] = (lo, hi)
                continue

            if verbosity > 1:
                print(f"[ApproxLP] Optimizing variable: {query}")

            # Minimize for lower bound
            lo_obj, lo_probs = self._coordinate_descent(
                factors, cards, query, evidence, "min", n_iters, verbosity,
                constraints)

            # Maximize for upper bound
            hi_obj, hi_probs = self._coordinate_descent(
                factors, cards, query, evidence, "max", n_iters, verbosity,
                constraints)

            # Assemble per-state bounds from the two runs
            k = cards[query]
            lower_bounds = np.ones(k)
            upper_bounds = np.zeros(k)
            for probs in [lo_probs, hi_probs]:
                if probs is not None:
                    for val in range(k):
                        lower_bounds[val] = min(lower_bounds[val], probs[val])
                        upper_bounds[val] = max(upper_bounds[val], probs[val])

            self.marginals[query] = (lower_bounds, upper_bounds)

        # Extract singleton marginals from compound variables
        self.singleton_marginals = self._extract_singleton_marginals(
            self.marginals, evidence)

        t_end = time.time()

        if verbosity > 0:
            print(f"[ApproxLP] Compound variable marginals:")
            for var in node_names:
                lo, hi = self.marginals[var]
                for val in range(len(lo)):
                    print(f"  P({var}={val}): "
                          f"[{lo[val]:.6f}, {hi[val]:.6f}]")

            if self.singleton_marginals:
                print(f"[ApproxLP] Singleton variable marginals:")
                for atom in sorted(self.singleton_marginals):
                    lo, hi = self.singleton_marginals[atom]
                    print(f"  P({atom}=0): [{1.0 - hi:.6f}, {1.0 - lo:.6f}]")
                    print(f"  P({atom}=1): [{lo:.6f}, {hi:.6f}]")

            print(f"[ApproxLP] Time elapsed: {t_end - t_start:.4f} sec")

        # Return combined dict of all marginals
        all_marginals = dict(self.marginals)
        for atom, (lo, hi) in self.singleton_marginals.items():
            all_marginals[atom] = (
                np.array([1.0 - hi, lo]),
                np.array([1.0 - lo, hi])
            )
        return all_marginals

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
            marginals: compound variable marginals.
            evidence: evidence dict (singleton atoms in evidence are skipped).

        Returns:
            Dict mapping singleton atom name to (lower_bound, upper_bound)
            for P(atom=1).
        """
        singleton_marginals = {}
        solver = make_ipopt()

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
                    print(f"    P({var}={val}): [{abs(lo[val]):.6f}, {abs(hi[val]):.6f}]")

    # Load the LCN
    file_name = "examples/new.lcn"
    # file_name = "benchmarks/chain/chain_n20_1.lcn"
    # file_name = "benchmarks/real/alarm.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Check consistency
    # ok = check_consistency(l)
    # if ok:
    #     print("CONSISTENT")
    # else:
    #     print("INCONSISTENT")

    # Build the credal network vertices (needed for extreme points)
    cnv = CredalNetworkVertices.from_lcn(l, method="linear", verbosity=1)

    # Create ApproxLP solver
    alp = ApproxLP(cnv=cnv)

    # All marginals (no evidence)
    print("\n=== ApproxLP (no evidence) ===")
    results = alp.run(evidence={}, verbosity=2)
    print_singleton_marginals(results)

    # All marginals (with evidence)
    # print("\n=== ApproxLP (B=0, E=0) ===")
    # results = alp.run(evidence={"B": 0, "E": 0}, verbosity=1)
    # print_singleton_marginals(results)
