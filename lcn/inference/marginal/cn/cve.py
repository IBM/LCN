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
# CredalVE consumes a pre-built CredalNetworkVertices (the directed graph +
# enumerated extreme points of the local credal sets) and computes exact
# lower/upper bounds over the strong extension on P(atom = 1 | evidence) for
# every atom, by variable elimination over potentials of PAIRS of functions
# (Maua & Cozman, IJAR 2020, Algorithm 2); see pair_potentials.py.

import time

import numpy as np

# Local
from lcn.core.model import LCN
from lcn.inference.marginal.cn.pair_potentials import build_cpt_pairs
from lcn.inference.marginal.cn.potentials import min_fill_order
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.exact import _is_vacuous
from lcn.inference.utils.common import check_consistency


class CredalVE:
    """
    Credal Variable Elimination (Maua & Cozman, IJAR 2020, Algorithm 2) over the
    strong extension of a credal network, with POTENTIALS OF PAIRS (p, q).

    For every credal-network node ``Z`` the engine eliminates every OTHER node
    first (min-fill or reverse topological order). Until the query enters, all
    pairs have p == q; after each sum-out such a potential is rescaled and
    reduced to the vertices of the convex hull of its rays
    (``prune_query_free``, exact). The node elimination is shared by Z's atoms:
    for each non-evidence atom ``a`` of Z the query potential {(1 - t, t)}
    (``t`` the indicator of ``a = 1`` over Z's states) is multiplied in, Z is
    summed out, the scalar pairs are Pareto-pruned in BOTH orders of eqs.
    (7)-(8) (a pair is dropped only when it can be optimal for neither bound),
    and P(a = 1 | e) = q / (p + q) is read off. The minimum and maximum over the
    pairs are the exact lower and upper probabilities over the strong extension
    -- with or without evidence.

    Evidence on an atom is applied as a 0/1 mask over its node's states, which
    also handles atoms of compound (multi-atom) nodes. A compound node's atom is
    queried directly through its indicator, so the bound is exact for the atom
    (no projection of per-state intervals).

    With ``epsilon > 0`` the engine is the FPTAS of Maua, de Campos & Zaffalon
    (IJAR 2012, Sec. 5): every pairwise product and every sum-out is followed
    by the alpha-quotient (one pair per alpha-equivalence class, see
    ``PairPotential.quotient``), and so is the final query step. With N such
    operations in a node's elimination, alpha = (1 + epsilon)^(1 / (2 N)), and
    the returned bounds are INNER bounds within a factor 1 + epsilon:
        P_up / (1 + eps) <= P_up_eps <= P_up,
        P_lo <= P_lo_eps <= (1 + eps) P_lo.
    (Each quotient loses at most alpha^2 on any downstream odds ratio; the exact
    Pareto and convex-hull prunes lose nothing; factors multiply through
    products and are preserved by sums.)
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

    # ------------------------------------------------------------------
    # Public API: all posterior marginals
    # ------------------------------------------------------------------

    def run(self, evidence: dict = None, elim_heuristic: str = "min-fill",
            epsilon: float = None, verbosity: int = 1):
        """
        Lower/upper bounds on P(atom = 1 | evidence) for EVERY non-evidence atom.

        Args:
            evidence: {atom_name: value} for observed atoms.
            elim_heuristic: "min-fill" (default) or "topological" ordering of
                the node eliminations (computed once per node).
            epsilon: None or 0 -> exact. > 0 -> the FPTAS: inner bounds within
                a factor (1 + epsilon) of the exact ones (see the class doc).
            verbosity: 0 silent, 1 summary, 2 per-atom detail.

        Returns:
            {atom -> (lower, upper)} with each a 2-vector [P(=0), P(=1)].
            Also sets self.marginals (the same dict), self.singleton_marginals
            ({atom -> (lo, hi)} for P(atom=1)), self.max_potential_size,
            self.epsilon, self.alpha (the smallest per-node alpha used, 1.0 when
            exact), self.n_quotient_ops (the largest per-node N),
            self.guarantee, the running times build_time / elimination_time /
            total_time, and self.degenerate.
        """
        evidence = {} if evidence is None else evidence
        assert elim_heuristic in ("topological", "min-fill"), \
            f"Unknown heuristic '{elim_heuristic}'. Use 'topological' or 'min-fill'."
        eps = 0.0 if epsilon is None else float(epsilon)
        assert eps >= 0, "Epsilon must be non-negative."
        self.epsilon = eps
        self.guarantee = "exact" if eps == 0 else \
            f"inner, within a factor 1+{eps:g}"

        node_atoms = self.cnv.cn.node_atoms
        node_of_atom = {a: v for v, ats in node_atoms.items() for a in ats}
        for a in evidence:
            assert a in node_of_atom, f"Evidence variable '{a}' not found."

        if verbosity > 0:
            print(f"[CredalVE] Computing all marginals (evidence={evidence}, "
                  f"epsilon={eps:g})")

        t_elim_start = time.perf_counter()
        cards, base, topo = self._base_potentials(evidence)
        self.max_potential_size = max((p.m for p in base), default=0)
        self.alpha = 1.0
        self.n_quotient_ops = 0

        self.singleton_marginals = {}
        for node in self.cnv.cn.nodes:
            todo = [(i, a) for i, a in enumerate(node_atoms[node])
                    if a not in evidence]
            if not todo:
                continue
            order = self._order(base, topo, node, elim_heuristic)
            alpha = 1.0
            if eps > 0:
                n_ops = self._count_quotients(base, order) + 1  # + query
                alpha = (1.0 + eps) ** (1.0 / (2 * n_ops))
                self.n_quotient_ops = max(self.n_quotient_ops, n_ops)
                self.alpha = alpha if self.alpha == 1.0 else \
                    min(self.alpha, alpha)
            # One elimination per NODE; every atom of the node (several for a
            # compound node) is then read off it with its own query indicator.
            node_pot = self._node_potential(base, order, alpha)
            for i, atom in todo:
                target = self._atom_indicator(node, i)
                lo, hi = self._query(node_pot, node, target, alpha)
                self.singleton_marginals[atom] = (lo, hi)
                if verbosity > 1:
                    print(f"  P({atom}=1 | e) in [{lo:.6f}, {hi:.6f}]")

        results = {a: (np.array([1.0 - hi, lo]), np.array([1.0 - lo, hi]))
                   for a, (lo, hi) in self.singleton_marginals.items()}
        self.marginals = results
        self.elimination_time = time.perf_counter() - t_elim_start
        self._record_times(verbosity)
        self._flag_degenerate(evidence, verbosity)

        if verbosity > 0:
            self._print_marginals(results)
            self._print_times()
        return results

    # ------------------------------------------------------------------
    # Potentials
    # ------------------------------------------------------------------

    def _base_potentials(self, evidence):
        """CPT pair potentials of every node, with the evidence masks already
        multiplied in. Returns (cards, potentials, topological node order)."""
        bn = self.cnv.bn_min
        cards = {bn.variable(n).name(): bn.variable(n).domainSize()
                 for n in bn.nodes()}
        topo = [bn.variable(n).name() for n in bn.topologicalOrder()]
        node_atoms = self.cnv.cn.node_atoms

        pots = []
        for node, configs in self.cnv.extreme_points.items():
            nid = bn.idFromName(node)
            parents = [bn.variable(p).name() for p in sorted(bn.parents(nid))]
            by_cfg = {}
            for cfg, verts in configs.items():
                if cfg == "<>":
                    by_cfg[()] = verts
                    continue
                vals = dict(part.split(":") for part in cfg[1:-1].split("|"))
                by_cfg[tuple(int(vals[p]) for p in parents)] = verts
            pot = build_cpt_pairs(node, parents, cards, by_cfg)
            # Evidence on atoms of this node: mask its child axis.
            for i, a in enumerate(node_atoms[node]):
                if a in evidence:
                    pot = pot.mask(node, self._atom_indicator(
                        node, i, value=int(evidence[a])))
            pots.append(pot.drop_zero_and_duplicates())
        return cards, pots, topo

    def _atom_indicator(self, node, atom_index, value=1):
        """0/1 indicator over ``node``'s states of ``atom = value``, using the
        MSB-first state packing of compound nodes (the cjt convention)."""
        k = len(self.cnv.cn.node_atoms[node])
        n_states = 2 ** k
        bit = k - 1 - atom_index
        return np.array([((s >> bit) & 1) == value for s in range(n_states)],
                        dtype=float)

    # ------------------------------------------------------------------
    # One query: eliminate everything, read off min/max q/(p+q)
    # ------------------------------------------------------------------

    @staticmethod
    def _order(base, topo, node, elim_heuristic):
        """Elimination order of every node except ``node`` (eliminated last)."""
        if elim_heuristic == "topological":
            return [v for v in reversed(topo) if v != node]
        return min_fill_order([p.scope for p in base], exclude={node})

    @staticmethod
    def _count_quotients(base, order):
        """Number of alpha-quotients the elimination along ``order`` performs:
        one per pairwise product (bucket products + the final products of what
        is left) and one per sum-out. Depends only on the scopes, so the FPTAS
        budget is fixed before any numeric work."""
        scopes = [set(p.scope) for p in base]
        n = 0
        for var in order:
            bucket = [sc for sc in scopes if var in sc]
            if not bucket:
                continue
            scopes = [sc for sc in scopes if var not in sc]
            n += len(bucket)          # bucket products + its sum-out
            scopes.append(set().union(*bucket) - {var})
        return n + max(0, len(scopes) - 1)

    def _combine(self, a, b, alpha):
        """Pairwise product, followed by the alpha-quotient when alpha > 1."""
        c = a.product(b)
        self.max_potential_size = max(self.max_potential_size, c.m)
        if alpha > 1.0:
            c = c.quotient(alpha)
        return c.drop_zero_and_duplicates()

    def _node_potential(self, base, order, alpha):
        """Eliminate every node along ``order`` and return the product of what
        is left: a query-free potential (p == q) over the remaining node.

        Because the query potential has not been multiplied in, every pair has
        p == q throughout; after each sum-out the potential is reduced to the
        vertices of its convex hull (``prune_query_free``, exact). The p/q
        split happens only in the final one-variable step of :meth:`_query`, as
        in Maua & Cozman's algorithm with the query eliminated last."""
        pots = list(base)
        for var in order:
            bucket = [p for p in pots if var in p.scope]
            if not bucket:
                continue
            pots = [p for p in pots if var not in p.scope]
            combined = bucket[0]
            for p in bucket[1:]:
                combined = self._combine(combined, p, alpha)
            res = combined.sum_out(var)
            if alpha > 1.0:
                res = res.quotient(alpha)
            pots.append(res.prune_query_free())

        final = pots[0]
        for p in pots[1:]:
            final = self._combine(final, p, alpha).prune_query_free()
        self.max_potential_size = max(self.max_potential_size, final.m)
        return final

    def _query(self, node_pot, node, target, alpha):
        """Bounds on P(target | e) from the query-free potential over ``node``:
        multiply in {(1 - t, t)}, sum ``node`` out, (alpha-quotient), exact
        Pareto prune in both orders, and read min/max q / (p + q)."""
        pot = node_pot.query(node, target).sum_out(node)
        if alpha > 1.0:
            pot = pot.quotient(alpha)
        return pot.prune("both").bounds()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _print_marginals(results):
        print("[CredalVE] Singleton marginals P(atom=1):")
        for name in sorted(results):
            lo, hi = results[name]
            print(f"  P({name}=1): [{lo[1]:.6f}, {hi[1]:.6f}]")

    def _record_times(self, verbosity):
        """build_time (credal network + extreme points, from the
        CredalNetworkVertices), elimination_time, total_time (seconds)."""
        self.build_time = float(getattr(self.cnv, "build_time", None) or 0.0)
        self.total_time = self.build_time + self.elimination_time

    def _print_times(self):
        print("[CredalVE] Running times (seconds):")
        print(f"  build time:       {self.build_time:.4f}")
        print(f"  running time:     {self.elimination_time:.4f}")
        print(f"  total time:       {self.total_time:.4f}")

    def _flag_degenerate(self, evidence, verbosity):
        """``self.degenerate`` is True when EVERY atom bound is the vacuous
        [0, 1] (usually an inconsistent LCN or evidence), None if no atom."""
        atoms = self.singleton_marginals
        if not atoms:
            self.degenerate = None
            return
        n_vacuous = sum(1 for (lo, hi) in atoms.values() if _is_vacuous(lo, hi))
        self.degenerate = (n_vacuous == len(atoms))
        if self.degenerate and verbosity > 0:
            print("[CredalVE] WARNING: the solution is DEGENERATE -- every "
                  "singleton marginal is the vacuous [0, 1].")
            if evidence:
                print(f"[CredalVE] The evidence {evidence} may be inconsistent "
                      f"with the LCN, or have probability zero.")
            else:
                print("[CredalVE] The LCN is most likely INCONSISTENT; run "
                      "check_consistency on the model.")


if __name__ == "__main__":

    # Load the LCN
    # file_name = "examples/chain.lcn"
    file_name = "benchmarks/tree_large_fr/tree_fr_n20_1.lcn"
    lcn_model = LCN()
    lcn_model.from_lcn(file_name=file_name)
    lcn_model.summary()
    print(lcn_model)

    verbosity = 2

    # Check consistency
    print(f"\n=== Consistency check for {file_name} ===")
    check_consistency(lcn_model)

    # Build the credal network vertices (chain-graph factorization +
    # interval local credal sets + extreme-point enumeration). CredalVE
    # consumes the enumerated extreme points, so keep enumerate_vertices=True.
    cnv = CredalNetworkVertices.from_lcn(
        lcn_model,
        method="linear",
        merge_budget=1,
        solver="ipopt",
        verbosity=verbosity,
    )

    # CredalVE computes all singleton marginals over the strong extension by
    # looping the per-target bucket elimination over the credal-network nodes.
    cve = CredalVE(cnv=cnv)
    print("\n=== All marginals (CredalVE) ===")
    cve.run(evidence={}, verbosity=verbosity)
    for atom in sorted(cve.singleton_marginals):
        lo, hi = cve.singleton_marginals[atom]
        print(f"  P({atom}=1) in [{lo:.6f}, {hi:.6f}]")
