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

# Interval Belief Propagation and Variational Inference for Credal Networks

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
from lcn.inference.marginal.cve import CredalVE
from lcn.inference.utils.common import check_consistency


class IntervalBP:
    """
    Interval belief propagation and mean-field variational inference for
    credal networks. Supports multi-valued variables natively via
    interval-vector messages (ApproxLP-style).

    Operates on a CredalVE instance that has already been built (i.e.,
    build() has been called to produce extreme points and the underlying
    pyAgrum BayesNet/CredalNet).
    """

    def __init__(self, cve: CredalVE):
        """
        Args:
            cve: CredalVE
                A CredalVE instance with build() already called.
        """
        assert cve.extreme_points is not None, \
            "CredalVE must have build() called before passing to IBP."
        assert cve.bn_min is not None

        self.cve = cve
        self.marginals = None
        self.singleton_marginals = None

    def _build_factors(self):
        """
        Build factor graph structures for BP from the credal network DAG
        and extreme points. Returns (cards, factors) where each factor is
        a dict with keys: 'node', 'scope', 'parents', 'vertices'.

        'vertices' is a dict mapping parent_config_tuple to a list of
        vertex arrays (each vertex is a 1-D array of length card[node]).
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

    def run(self, evidence: dict = {},
            n_iters: int = 100, threshold: float = 1e-6,
            method: str = "interval",
            verbosity: int = 1) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run belief propagation for credal networks. Computes marginal
        bounds for ALL non-evidence variables.

        Args:
            evidence: dict
                {variable_name: value} for observed variables.
            n_iters: int
                Maximum number of BP iterations.
            threshold: float
                Convergence threshold on maximum message change.
            method: str
                "interval" for interval BP (outer approximation), or
                "variational" for mean-field variational (inner approx).
            verbosity: int
                Verbosity level (0 is silent).

        Returns:
            Dict mapping variable name to (lower_bounds, upper_bounds)
            numpy arrays. Includes both compound and singleton marginals.
        """
        assert method in ("interval", "variational"), \
            f"Unknown method '{method}'. Use 'interval' or 'variational'."

        if method == "variational":
            return self._run_variational(evidence, n_iters,
                                         threshold, verbosity)

        t_start = time.time()

        cards, factors = self._build_factors()
        bn = self.cve.bn_min
        node_names = [bn.variable(n).name() for n in bn.nodes()]

        # Build adjacency: var -> list of factor indices
        var_to_factors = {name: [] for name in node_names}
        for fi, fac in enumerate(factors):
            for v in fac['scope']:
                var_to_factors[v].append(fi)

        # Initialize messages
        msg_v2f = {}
        msg_f2v = {}
        for fi, fac in enumerate(factors):
            for v in fac['scope']:
                msg_v2f[(v, fi)] = (np.zeros(cards[v]), np.ones(cards[v]))
                msg_f2v[(fi, v)] = (np.zeros(cards[v]), np.ones(cards[v]))

        # Clamp evidence
        evidence_set = set(evidence.keys())
        for ev_var, ev_val in evidence.items():
            lo = np.zeros(cards[ev_var])
            hi = np.zeros(cards[ev_var])
            lo[ev_val] = 1.0
            hi[ev_val] = 1.0
            for fi in var_to_factors[ev_var]:
                msg_v2f[(ev_var, fi)] = (lo.copy(), hi.copy())

        if verbosity > 0:
            print(f"[IBP] Computing all marginals, method: {method}")
            print(f"[IBP] Evidence: {evidence}")
            print(f"[IBP] Nodes: {len(node_names)}, "
                  f"Factors: {len(factors)}")

        # Iterative message passing
        for iteration in range(n_iters):
            max_delta = 0.0

            # 1) Update variable-to-factor messages (tightening)
            for var in node_names:
                if var in evidence_set:
                    continue
                neighbor_factors = var_to_factors[var]
                for fi in neighbor_factors:
                    old_lo, old_hi = msg_v2f[(var, fi)]
                    new_lo = np.zeros(cards[var])
                    new_hi = np.ones(cards[var])
                    for fj in neighbor_factors:
                        if fj == fi:
                            continue
                        fj_lo, fj_hi = msg_f2v[(fj, var)]
                        new_lo = np.maximum(new_lo, fj_lo)
                        new_hi = np.minimum(new_hi, fj_hi)
                    new_hi = np.maximum(new_lo, new_hi)
                    msg_v2f[(var, fi)] = (new_lo, new_hi)
                    max_delta = max(max_delta,
                                    np.max(np.abs(new_lo - old_lo)),
                                    np.max(np.abs(new_hi - old_hi)))

            # 2) Update factor-to-variable messages
            for fi, fac in enumerate(factors):
                node = fac['node']
                parents = fac['parents']
                scope = fac['scope']
                vertices = fac['vertices']

                for target_var in scope:
                    old_lo, old_hi = msg_f2v[(fi, target_var)]
                    other_vars = [v for v in scope if v != target_var]

                    new_lo = np.ones(cards[target_var])
                    new_hi = np.zeros(cards[target_var])
                    any_valid = False

                    if len(parents) == 0:
                        all_pcs = [()]
                    else:
                        all_pcs = list(itertools.product(
                            *[range(cards[pn]) for pn in parents]
                        ))

                    other_bounds = {}
                    for ov in other_vars:
                        ov_lo, ov_hi = msg_v2f[(ov, fi)]
                        other_bounds[ov] = (ov_lo, ov_hi)

                    vertex_lists = [vertices.get(pc, [np.ones(cards[node]) / cards[node]])
                                    for pc in all_pcs]
                    vertex_counts = [len(vl) for vl in vertex_lists]
                    vertex_combos = list(itertools.product(
                        *[range(c) for c in vertex_counts]
                    ))

                    for vc in vertex_combos:
                        corner_vars = list(other_vars)
                        n_corners = len(corner_vars)
                        for corner_bits in itertools.product([0, 1],
                                                             repeat=n_corners):
                            target_marginal = np.zeros(cards[target_var])

                            other_probs = {}
                            for ci, ov in enumerate(corner_vars):
                                ov_lo, ov_hi = other_bounds[ov]
                                if corner_bits[ci] == 0:
                                    other_probs[ov] = ov_lo
                                else:
                                    other_probs[ov] = ov_hi

                            for pc_idx, pc in enumerate(all_pcs):
                                v_idx = vc[pc_idx]
                                vertex = vertex_lists[pc_idx][v_idx]

                                weight = 1.0
                                for pi, pn in enumerate(parents):
                                    if pn in other_probs:
                                        weight *= other_probs[pn][pc[pi]]

                                if target_var == node:
                                    target_marginal += vertex * weight
                                else:
                                    target_pi = parents.index(target_var)
                                    target_val = pc[target_pi]
                                    other_parent_w = 1.0
                                    for pi, pn in enumerate(parents):
                                        if pn != target_var and pn in other_probs:
                                            other_parent_w *= other_probs[pn][pc[pi]]
                                    if node in other_probs:
                                        child_contrib = np.dot(vertex, other_probs[node])
                                    else:
                                        child_contrib = np.sum(vertex)
                                    target_marginal[target_val] += child_contrib * other_parent_w

                            total = np.sum(target_marginal)
                            if total > 0:
                                prob = target_marginal / total
                                new_lo = np.minimum(new_lo, prob)
                                new_hi = np.maximum(new_hi, prob)
                                any_valid = True

                    if not any_valid:
                        new_lo = np.zeros(cards[target_var])
                        new_hi = np.ones(cards[target_var])
                    new_hi = np.maximum(new_lo, new_hi)
                    msg_f2v[(fi, target_var)] = (new_lo, new_hi)
                    max_delta = max(max_delta,
                                    np.max(np.abs(new_lo - old_lo)),
                                    np.max(np.abs(new_hi - old_hi)))

            if verbosity > 1:
                print(f"  Iteration {iteration}: max_delta = {max_delta:.8f}")

            if max_delta < threshold:
                if verbosity > 0:
                    print(f"[IBP] Converged after {iteration + 1} iterations "
                          f"(delta={max_delta:.2e})")
                break
        else:
            if verbosity > 0:
                print(f"[IBP] Reached max iterations ({n_iters}), "
                      f"delta={max_delta:.2e}")

        # Extract marginal bounds for all variables
        self.marginals = {}
        for var in node_names:
            lo = np.zeros(cards[var])
            hi = np.ones(cards[var])
            for fi in var_to_factors[var]:
                fi_lo, fi_hi = msg_f2v[(fi, var)]
                lo = np.maximum(lo, fi_lo)
                hi = np.minimum(hi, fi_hi)
            hi = np.maximum(lo, hi)
            self.marginals[var] = (lo, hi)

        # Extract singleton marginals from compound variables
        self.singleton_marginals = self._extract_singleton_marginals(
            self.marginals, evidence)

        t_end = time.time()

        if verbosity > 0:
            print(f"[IBP] Compound variable marginals:")
            for var in node_names:
                lo, hi = self.marginals[var]
                for val in range(len(lo)):
                    print(f"  P({var}={val}): "
                          f"[{lo[val]:.6f}, {hi[val]:.6f}]")

            if self.singleton_marginals:
                print(f"[IBP] Singleton variable marginals:")
                for atom in sorted(self.singleton_marginals):
                    lo, hi = self.singleton_marginals[atom]
                    print(f"  P({atom}=0): [{1.0 - hi:.6f}, {1.0 - lo:.6f}]")
                    print(f"  P({atom}=1): [{lo:.6f}, {hi:.6f}]")

            print(f"[IBP] Time elapsed: {t_end - t_start:.4f} sec")

        # Return combined dict of all marginals
        all_marginals = dict(self.marginals)
        for atom, (lo, hi) in self.singleton_marginals.items():
            all_marginals[atom] = (
                np.array([1.0 - hi, lo]),
                np.array([1.0 - lo, hi])
            )
        return all_marginals

    def _run_variational(self, evidence: dict,
                         n_iters: int, threshold: float,
                         verbosity: int):
        """
        Mean-field variational inference for credal networks. For each
        combination of extreme points (one per local credal set), runs
        standard mean-field coordinate ascent. Tracks bounds across all
        explored vertex combinations for ALL variables.
        """
        t_start = time.time()

        cards, factors = self._build_factors()
        bn = self.cve.bn_min
        node_names = [bn.variable(n).name() for n in bn.nodes()]

        evidence_set = set(evidence.keys())

        # Build factor index
        var_to_factors = {name: [] for name in node_names}
        for fi, fac in enumerate(factors):
            for v in fac['scope']:
                var_to_factors[v].append(fi)

        if verbosity > 0:
            print(f"[IBP-VI] Computing all marginals, method: variational")
            print(f"[IBP-VI] Evidence: {evidence}")

        # Track bounds for ALL variables
        all_lower = {var: np.ones(cards[var]) for var in node_names}
        all_upper = {var: np.zeros(cards[var]) for var in node_names}

        # For each factor, enumerate vertex combos per parent config
        factor_combos = []
        for fi, fac in enumerate(factors):
            parents = fac['parents']
            if len(parents) == 0:
                all_pcs = [()]
            else:
                all_pcs = list(itertools.product(
                    *[range(cards[pn]) for pn in parents]
                ))
            vl = [fac['vertices'].get(pc, [np.ones(cards[fac['node']]) / cards[fac['node']]])
                  for pc in all_pcs]
            vc = [len(v) for v in vl]
            combos = list(itertools.product(*[range(c) for c in vc]))
            factor_combos.append((fi, fac, all_pcs, vl, combos))

        # Generate global vertex combos (one per factor)
        all_global_combos = list(itertools.product(
            *[fc[4] for fc in factor_combos]
        ))

        # Cap the number of combinations
        max_combos = min(len(all_global_combos), 500)
        if len(all_global_combos) > max_combos:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(all_global_combos), max_combos, replace=False)
            selected_combos = [all_global_combos[i] for i in sorted(indices)]
        else:
            selected_combos = all_global_combos

        for global_combo in selected_combos:
            # Fix extreme points, run standard mean-field
            q = {}
            for name in node_names:
                if name in evidence:
                    dist = np.zeros(cards[name])
                    dist[evidence[name]] = 1.0
                    q[name] = dist
                else:
                    q[name] = np.ones(cards[name]) / cards[name]

            for iteration in range(n_iters):
                max_delta = 0.0
                for var in node_names:
                    if var in evidence_set:
                        continue

                    log_q = np.zeros(cards[var])

                    for fvi, (fi, fac, all_pcs, vl, _) in enumerate(factor_combos):
                        vc = global_combo[fvi]
                        node_name = fac['node']
                        parents = fac['parents']

                        for pc_idx, pc in enumerate(all_pcs):
                            v_idx = vc[pc_idx]
                            vertex = vl[pc_idx][v_idx]

                            if node_name == var:
                                parent_weight = 1.0
                                for pi, pn in enumerate(parents):
                                    parent_weight *= q[pn][pc[pi]]
                                for x in range(cards[var]):
                                    if vertex[x] > 0:
                                        log_q[x] += parent_weight * np.log(
                                            vertex[x] + 1e-300)
                            elif var in parents:
                                var_pi = parents.index(var)
                                for x in range(cards[var]):
                                    if pc[var_pi] == x:
                                        other_pw = 1.0
                                        for pi2, pn2 in enumerate(parents):
                                            if pn2 != var:
                                                other_pw *= q[pn2][pc[pi2]]
                                        child_contrib = 0.0
                                        for cx in range(cards[node_name]):
                                            child_contrib += q[node_name][cx] * np.log(
                                                vertex[cx] + 1e-300)
                                        log_q[x] += child_contrib * other_pw

                    log_q -= np.max(log_q)
                    q_new = np.exp(log_q)
                    q_sum = np.sum(q_new)
                    if q_sum > 0:
                        q_new /= q_sum
                    else:
                        q_new = np.ones(cards[var]) / cards[var]

                    max_delta = max(max_delta, np.max(np.abs(q[var] - q_new)))
                    q[var] = q_new

                if max_delta < threshold:
                    break

            # Update bounds for ALL variables
            for var in node_names:
                all_lower[var] = np.minimum(all_lower[var], q[var])
                all_upper[var] = np.maximum(all_upper[var], q[var])

        # Store marginals
        self.marginals = {var: (all_lower[var], all_upper[var])
                          for var in node_names}

        # Extract singleton marginals from compound variables
        self.singleton_marginals = self._extract_singleton_marginals(
            self.marginals, evidence)

        t_end = time.time()

        if verbosity > 0:
            print(f"[IBP-VI] Explored {len(selected_combos)} vertex combinations")
            print(f"[IBP-VI] Compound variable marginals:")
            for var in node_names:
                lo, hi = self.marginals[var]
                for val in range(len(lo)):
                    print(f"  P({var}={val}): "
                          f"[{lo[val]:.6f}, {hi[val]:.6f}]")

            if self.singleton_marginals:
                print(f"[IBP-VI] Singleton variable marginals:")
                for atom in sorted(self.singleton_marginals):
                    lo, hi = self.singleton_marginals[atom]
                    print(f"  P({atom}=0): [{1.0 - hi:.6f}, {1.0 - lo:.6f}]")
                    print(f"  P({atom}=1): [{lo:.6f}, {hi:.6f}]")

            print(f"[IBP-VI] Time elapsed: {t_end - t_start:.4f} sec")

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
    file_name = "examples/lcn_chain_1.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Build the CredalVE (needed for extreme points)
    cve = CredalVE(lcn=l)
    cve.build(verbosity=0)

    # Create the IBP solver
    ibp = IntervalBP(cve=cve)

    # Run interval BP (all marginals, no evidence)
    print("\n=== Interval Belief Propagation (no evidence) ===")
    results = ibp.run(evidence={}, method="interval", verbosity=1)
    print_singleton_marginals(results)

    # Run interval BP (all marginals, with evidence)
    print("\n=== Interval Belief Propagation (x0=0) ===")
    results = ibp.run(evidence={"x0": 0}, method="interval", verbosity=1)
    print_singleton_marginals(results)

    # Run variational inference (all marginals, no evidence)
    print("\n=== Variational Inference (no evidence) ===")
    results = ibp.run(evidence={}, method="variational", n_iters=20, verbosity=1)
    print_singleton_marginals(results)

    # Run variational inference (all marginals, with evidence)
    print("\n=== Variational Inference (x0=0) ===")
    results = ibp.run(evidence={"x0": 0}, method="variational",
                      n_iters=20, verbosity=1)
    print_singleton_marginals(results)
