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

import numpy as np
from pyomo.environ import (
    ConcreteModel, Var, ConstraintList, Objective,
    NonNegativeReals, minimize, maximize, value,
)

# Local
from lcn.core.model import LCN
from lcn.inference.marginal.cn.coupling import (
    CouplingConstraints, warn_conditional_coupling)
from lcn.inference.marginal.cn.junction_nlp import build_and_solve_jt_nlp
from lcn.inference.marginal.cn.potentials import Potential, min_fill_order
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.utils.common import check_consistency, make_ipopt


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

    def _build_coupling(self, coupling: str, verbosity: int):
        """
        Build the D4 cross-family coupling constraints from the credal network's
        source LCN and its (possibly merged) symbolic factorization. Returns
        (constraints, node_atoms): constraints is a CouplingConstraints (or None
        when coupling=="off" or there is no cross-family residual), node_atoms is
        the {node -> atoms} map the feasibility checks need.
        """
        node_atoms = self.cnv.cn.node_atoms
        if coupling == "off":
            return None, node_atoms
        cc = CouplingConstraints.from_lcn(
            self.cnv.lcn, self.cnv.cn.factorization.factors)
        if verbosity > 0:
            print(f"[CredalVE] D4 coupling: {len(cc)} cross-family constraint(s)")
        return (cc if len(cc) > 0 else None), node_atoms

    # ------------------------------------------------------------------
    # Public API: all posterior marginals
    # ------------------------------------------------------------------

    def run(self, evidence: dict = {},
            elim_heuristic: str = "topological", epsilon: float = None,
            coupling: str = "off", d5_solver: str = "scip",
            verbosity: int = 1):
        """
        Compute lower/upper bounds on the posterior marginal of EVERY
        (non-evidence) singleton atom, by iterating the single-query bucket
        elimination over the credal-network nodes.

        For each node, the elimination ordering is recomputed so the node is the
        last variable eliminated (its own bucket survives), one bucket-
        elimination pass yields the node's per-state bounds, and singleton-atom
        marginals P(atom=1 | evidence) are read off (directly for a singleton
        node, or by an LP projection of a compound node's per-state polytope).
        Evidence atoms are skipped.

        Args:
            evidence: {variable_name: value} for observed atoms.
            elim_heuristic: "topological" (default) or "min-fill" per-target
                ordering. With coupling="cross-family" the order is forced to
                min-fill augmented with the constrained node sets.
            epsilon: if not None, epsilon-approximate pruning (FPTAS).
            coupling: "off" (strong extension), "cross-family" (scheme D4, drops
                vertex combinations violating a cross-family constraint), or
                "d5" (scheme D5, the junction-tree EXACT NLP; epsilon ignored).
            d5_solver: backend for D5, "scip" (default, certified global) or
                "ipopt". Inert unless coupling=="d5".
            verbosity: 0 silent, 1 summary, 2 per-target detail.

        Returns:
            Dict mapping variable name to (lower_bounds, upper_bounds) numpy
            arrays. Singleton atoms map to the 2-vector [P(=0), P(=1)] bounds;
            compound nodes (e.g. "C-D") map to their per-state bounds. Also sets
            self.marginals (this dict) and self.singleton_marginals
            ({atom -> (lo, hi)} for P(atom=1)).
        """
        assert coupling in ("off", "cross-family", "d5"), \
            f"Unknown coupling '{coupling}'. Use 'off', 'cross-family' or 'd5'."
        assert self.cnv.extreme_points is not None, \
            "CredalNetworkVertices must be built before run()."
        assert self.cnv.bn_min is not None

        node_atoms = self.cnv.cn.node_atoms
        evidence_set = set(evidence.keys())

        if coupling == "d5":
            if epsilon is not None and verbosity > 0:
                print("[CredalVE] D5 is exact; ignoring epsilon.")
            return self._run_all_d5(evidence, evidence_set, node_atoms,
                                    d5_solver, verbosity)

        assert elim_heuristic in ("topological", "min-fill"), \
            f"Unknown heuristic '{elim_heuristic}'. Use 'topological' or 'min-fill'."
        if coupling == "cross-family":
            warn_conditional_coupling(evidence, "CredalVE", verbosity)

        if verbosity > 0:
            print(f"[CredalVE] Computing all marginals "
                  f"(coupling={coupling}, evidence={evidence})")

        # Iterate over the credal-network nodes; for each non-fully-observed
        # node run one elimination pass with that node as the (last) target.
        self.marginals = {}
        for node in self.cnv.cn.nodes:
            if all(a in evidence_set for a in node_atoms[node]):
                continue  # node fully observed -> nothing to compute
            lo, hi = self._run_single_query(
                node, evidence, elim_heuristic, epsilon, coupling, verbosity)
            self.marginals[node] = (lo, hi)

        # Project to singleton-atom marginals and assemble the return dict.
        self.singleton_marginals = self._extract_singleton_atoms(
            self.marginals, evidence_set)
        results = self._assemble_results(evidence_set)

        if verbosity > 0:
            self._print_marginals(results)
        return results

    def _run_all_d5(self, evidence, evidence_set, node_atoms, d5_solver,
                    verbosity):
        """All-marginals via scheme D5: one junction-tree exact NLP per
        non-evidence singleton atom. Sets self.marginals/self.singleton_marginals
        and self.d5_exact/self.induced_width."""
        if verbosity > 0:
            print(f"[CredalVE] Computing all marginals (coupling=d5, "
                  f"solver={d5_solver}, evidence={evidence})")
        atoms = sorted({a for atoms in node_atoms.values() for a in atoms}
                       - evidence_set)
        self.singleton_marginals = {}
        self.d5_exact = True
        self.induced_width = 0
        for atom in atoms:
            lo, hi, exact, width = self._run_d5_atom(
                atom, evidence, d5_solver, verbosity)
            self.singleton_marginals[atom] = (lo, hi)
            self.d5_exact = self.d5_exact and exact
            self.induced_width = max(self.induced_width, width)
        # D5 yields singleton-level marginals only.
        self.marginals = {}
        results = self._assemble_results(evidence_set)
        if verbosity > 0:
            self._print_marginals(results)
            print(f"[CredalVE] D5 exact={self.d5_exact}, "
                  f"max cluster={self.induced_width} atoms")
        return results

    # ------------------------------------------------------------------
    # Singleton-atom projection + result assembly
    # ------------------------------------------------------------------

    def _extract_singleton_atoms(self, marginals, evidence_set):
        """
        Derive P(atom=1 | evidence) bounds for every non-evidence singleton
        atom from the per-node marginals. A singleton node gives its atom's
        bound directly (state 1); a compound node's atoms are obtained by an LP
        over the node's per-state interval polytope (MSB-first state packing,
        matching cve.py / coupling.py). Skips evidence atoms.
        """
        solver = make_ipopt()
        singleton = {}
        for node, (lo, hi) in marginals.items():
            atoms = self.cnv.cn.node_atoms[node]
            if len(atoms) == 1:
                atom = atoms[0]
                if atom in evidence_set:
                    continue
                singleton[atom] = (float(lo[1]), float(hi[1]))
                continue
            # Compound node: project each atom out of the per-state polytope.
            n_atoms = len(atoms)
            k = 2 ** n_atoms
            for atom_idx, atom in enumerate(atoms):
                if atom in evidence_set:
                    continue
                ones = [s for s in range(k)
                        if (s >> (n_atoms - 1 - atom_idx)) & 1 == 1]
                low = self._solve_singleton_lp(lo, hi, k, ones, minimize, solver)
                up = self._solve_singleton_lp(lo, hi, k, ones, maximize, solver)
                singleton[atom] = (low, up)
        return singleton

    @staticmethod
    def _solve_singleton_lp(lo, hi, k, target_states, sense, solver):
        """min/max sum(p[s] for s in target_states) s.t. the per-state interval
        bounds and the simplex. (Same LP as CredalCTE._solve_singleton_lp.)"""
        model = ConcreteModel()
        model.S = range(k)
        model.p = Var(model.S, within=NonNegativeReals)
        model.constr = ConstraintList()
        model.constr.add(sum(model.p[s] for s in model.S) == 1.0)
        for s in model.S:
            model.constr.add(model.p[s] >= float(lo[s]))
            model.constr.add(model.p[s] <= float(hi[s]))
        model.obj = Objective(
            expr=sum(model.p[s] for s in target_states), sense=sense)
        solver.solve(model, tee=False)
        return float(value(model.obj))

    def _assemble_results(self, evidence_set):
        """Combine the per-node marginals (compound nodes kept as-is) with the
        singleton-atom marginals (as 2-vectors) into one {name -> (lo,hi)} dict
        in the contract the experiment runner consumes."""
        results = {}
        # Compound-node per-state bounds (filtered out downstream by name '-').
        for node, (lo, hi) in self.marginals.items():
            if len(self.cnv.cn.node_atoms[node]) > 1:
                results[node] = (np.asarray(lo), np.asarray(hi))
        # Singleton atoms as [P(=0), P(=1)] 2-vectors.
        for atom, (lo, hi) in self.singleton_marginals.items():
            results[atom] = (np.array([1.0 - hi, lo]),
                             np.array([1.0 - lo, hi]))
        return results

    @staticmethod
    def _print_marginals(results):
        print("[CredalVE] Singleton marginals P(atom=1):")
        for name in sorted(results):
            if "-" in name:
                continue
            lo, hi = results[name]
            print(f"  P({name}=1): [{lo[1]:.6f}, {hi[1]:.6f}]")

    def _run_d5_atom(self, atom, evidence, d5_solver, verbosity):
        """
        Scheme D5 for a single (singleton) atom: replace the vertex propagation
        with a junction-tree exact NLP (cluster-marginal variables + separator
        consistency + all LCN sentences and LMC equalities), giving the exact
        chain-graph bound P(atom=1 | evidence) at treewidth cost. Falls back to
        ExactInference.run_query when the JT-NLP cannot be made exact within the
        cluster budget. Returns (lo, hi, exact, width) for P(atom=1 | evidence).
        """
        from lcn.inference.marginal.exact import ExactInference

        lo, hi, info = build_and_solve_jt_nlp(
            self.cnv, atom, evidence=evidence, solver=d5_solver,
            verbosity=verbosity)
        exact = info["exact"]
        width = info["max_cluster_atoms"]

        if info["fallback"]:
            if verbosity > 0:
                print("[CredalVE] D5 falling back to ExactInference "
                      "(cluster budget exceeded).")
            ei = ExactInference(self.cnv.lcn)
            lo, hi = ei.run_query(atom, evidence=evidence,
                                  solver="local", verbosity=0)

        if verbosity > 1:
            print(f"  [D5] P({atom}=1 | {evidence}) = [{lo:.6f}, {hi:.6f}]  "
                  f"(exact={exact}, max cluster={width} atoms)")
        return lo, hi, exact, width

    def _run_single_query(self, query, evidence, elim_heuristic, epsilon,
                          coupling, verbosity):
        """
        One bucket-elimination pass for a single target node ``query``,
        eliminating every other (non-evidence) variable so the target is last,
        and returning the target node's per-state ``(lower_bounds,
        upper_bounds)`` arrays for P(query-state | evidence). This is the engine
        the public all-marginals :meth:`run` loops over; ``coupling`` is "off"
        or "cross-family" here (the "d5" path is handled per-atom by
        :meth:`_run_d5_atom`).
        """
        if epsilon is not None:
            return self._run_single_query_approx(
                query, evidence, epsilon, elim_heuristic, coupling, verbosity)

        bn = self.cnv.bn_min  # use for DAG structure
        node_names = [bn.variable(n).name() for n in bn.nodes()]
        assert query in node_names, \
            f"Query variable '{query}' not found in credal network."

        # Variable cardinalities
        cards = {}
        for nid in bn.nodes():
            name = bn.variable(nid).name()
            cards[name] = bn.variable(nid).domainSize()

        # Scheme D4: build the cross-family coupling constraints (or None when
        # disabled). node_atoms maps each node to its atoms (for the joint
        # marginalization inside the feasibility checks).
        constraints, node_atoms = self._build_coupling(coupling, verbosity)

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

        # Step 3: Determine elimination ordering. With D4 coupling enabled we
        # force the min-fill heuristic and add each constraint's node set as an
        # extra scope, so the constrained nodes co-occur in a common bucket
        # (otherwise the constraint scope is never assembled and D4 stays inert).
        use_heuristic = elim_heuristic
        if constraints is not None and elim_heuristic == "topological":
            use_heuristic = "min-fill"
        if use_heuristic == "topological":
            topo = list(bn.topologicalOrder())
            topo_names = [bn.variable(nid).name() for nid in topo]
            elim_order = [v for v in topo_names
                          if v != query and v not in evidence]
            # Evidence variables are already fixed by indicator potentials,
            # but we still need to eliminate them
            elim_order += [v for v in topo_names if v in evidence]
        else:  # min-fill
            scopes = [p.scope for p in potentials]
            if constraints is not None:
                scopes = scopes + constraints.constraint_node_sets(node_atoms)
            elim_order = min_fill_order(scopes, exclude={query})

        if verbosity > 0:
            print(f"[CredalVE] Query: {query}")
            print(f"[CredalVE] Evidence: {evidence}")
            print(f"[CredalVE] Elimination order ({use_heuristic}): {elim_order}")
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

            # Scheme D4: drop functions whose assembled joint violates a
            # cross-family constraint. This must happen on `combined` (which
            # still carries the full bucket scope) BEFORE `var` is summed out:
            # once the bucket scope covers a constraint's atoms the check fires,
            # and after marginalization those atoms are gone. The check is a
            # no-op for any constraint the bucket scope does not yet cover.
            if constraints is not None:
                combined = combined.filter_infeasible(constraints, node_atoms)

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
            return np.zeros(cards[query]), np.ones(cards[query])

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

        return lower_bounds, upper_bounds

    def _run_single_query_approx(self, query, evidence, epsilon,
                                 elim_heuristic, coupling, verbosity):
        """
        Epsilon-approximate single-query credal variable elimination: same as
        :meth:`_run_single_query` but uses epsilon-approximate pruning at each
        elimination step, which allows slightly dominated functions to be
        removed. This bounds the size of intermediate potentials, yielding an
        FPTAS (error at most epsilon). Returns the target node's per-state
        ``(lower_bounds, upper_bounds)``.

        See: Mauá et al. (2012), "Solving limited memory influence diagrams"
        and Mauá & Cozman (2020), "Thirty years of credal networks", Sec 5.2.
        """
        assert epsilon >= 0, "Epsilon must be non-negative."

        bn = self.cnv.bn_min
        node_names = [bn.variable(n).name() for n in bn.nodes()]
        assert query in node_names, \
            f"Query variable '{query}' not found in credal network."

        # Variable cardinalities
        cards = {}
        for nid in bn.nodes():
            name = bn.variable(nid).name()
            cards[name] = bn.variable(nid).domainSize()

        # Scheme D4: build the cross-family coupling constraints (or None when
        # disabled). node_atoms maps each node to its atoms (for the joint
        # marginalization inside the feasibility checks).
        constraints, node_atoms = self._build_coupling(coupling, verbosity)

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

        # Step 3: Determine elimination ordering (D4: force augmented min-fill).
        use_heuristic = elim_heuristic
        if constraints is not None and elim_heuristic == "topological":
            use_heuristic = "min-fill"
        if use_heuristic == "topological":
            topo = list(bn.topologicalOrder())
            topo_names = [bn.variable(nid).name() for nid in topo]
            elim_order = [v for v in topo_names
                          if v != query and v not in evidence]
            elim_order += [v for v in topo_names if v in evidence]
        else:
            scopes = [p.scope for p in potentials]
            if constraints is not None:
                scopes = scopes + constraints.constraint_node_sets(node_atoms)
            elim_order = min_fill_order(scopes, exclude={query})

        if verbosity > 0:
            print(f"[CredalVE-approx] Query: {query}, epsilon: {epsilon}")
            print(f"[CredalVE-approx] Evidence: {evidence}")
            print(f"[CredalVE-approx] Elimination order ({use_heuristic}): "
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

            # Scheme D4: filter on the full bucket scope before marginalizing.
            if constraints is not None:
                combined = combined.filter_infeasible(constraints, node_atoms)

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
            return np.zeros(cards[query]), np.ones(cards[query])

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

        return lower_bounds, upper_bounds


if __name__ == "__main__":

    # Load the LCN
    file_name = "examples/alarm.lcn"
    lcn_model = LCN()
    lcn_model.from_lcn(file_name=file_name)
    lcn_model.summary()
    print(lcn_model)

    # Check consistency
    print(f"\n=== Consistency check for {file_name} ===")
    ok = check_consistency(lcn_model)

    # Build the credal network vertices (chain-graph factorization +
    # interval local credal sets + extreme-point enumeration)
    cnv = CredalNetworkVertices.from_lcn(lcn_model, method="linear", verbosity=1)

    # Credal Variable Elimination algorithm. A single run() now computes every
    # singleton atom's posterior marginal by looping the per-target bucket
    # elimination over the credal-network nodes.
    cve = CredalVE(cnv=cnv)

    print("\n=== All marginals (exact, coupling=off) ===")
    results = cve.run(evidence={}, verbosity=0)
    for atom in sorted(cve.singleton_marginals):
        lo, hi = cve.singleton_marginals[atom]
        print(f"  P({atom}=1) in [{lo:.6f}, {hi:.6f}]")

    # With evidence (skips the observed atoms):
    print("\n=== All marginals given B=0, E=0 ===")
    cve.run(evidence={"B": 0, "E": 0}, verbosity=0)
    for atom in sorted(cve.singleton_marginals):
        lo, hi = cve.singleton_marginals[atom]
        print(f"  P({atom}=1 | B=0,E=0) in [{lo:.6f}, {hi:.6f}]")
