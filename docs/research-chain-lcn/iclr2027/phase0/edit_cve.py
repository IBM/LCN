"""Rewrite cve.py around Maua-Cozman pair potentials (one-off)."""
import sys

P = sys.argv[1]
s = open(P).read()

a = s.index("import itertools\nimport time")
b = s.index('if __name__ == "__main__":')

NEW = r'''import time

import numpy as np

# Local
from lcn.core.model import LCN
from lcn.inference.marginal.cn.pair_potentials import (
    PairPotential, build_cpt_pairs)
from lcn.inference.marginal.cn.potentials import min_fill_order
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.exact import _is_vacuous
from lcn.inference.utils.common import check_consistency


class CredalVE:
    """
    Credal Variable Elimination (Maua & Cozman, IJAR 2020, Algorithm 2) over the
    strong extension of a credal network, with POTENTIALS OF PAIRS (p, q).

    For every non-evidence atom ``a`` the engine multiplies the query potential
    {(1 - t, t)} (``t`` the indicator of ``a = 1`` over the states of a's node)
    into the network, eliminates every node, and reads
    P(a = 1 | e) = q / (p + q) off the resulting scalar pairs; the minimum and
    maximum over the pairs are the exact lower and upper probabilities over the
    strong extension -- with or without evidence. After each summation the
    potential is Pareto-pruned in BOTH orders of eqs. (7)-(8) (a pair is dropped
    only when it can be optimal for neither bound), so one elimination per atom
    yields both bounds.

    Evidence on an atom is applied as a 0/1 mask over its node's states, which
    also handles atoms of compound (multi-atom) nodes. A compound node's atom is
    queried directly through its indicator, so the bound is exact for the atom
    (no projection of per-state intervals).
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

    def run(self, evidence: dict = {},
            elim_heuristic: str = "min-fill", epsilon: float = None,
            coupling: str = "off", n_clusters: int = 0,
            cluster_representative: str = "mean", verbosity: int = 1):
        """
        Lower/upper bounds on P(atom = 1 | evidence) for EVERY non-evidence atom.

        Args:
            evidence: {atom_name: value} for observed atoms.
            elim_heuristic: "min-fill" (default) or "topological" ordering of
                the node eliminations (computed once per query atom).
            epsilon: if not None, epsilon-approximate Pareto pruning (FPTAS).
            coupling: only "off" is supported (kept for call compatibility).
            n_clusters: if > 0, cluster every intermediate potential into at
                most ``n_clusters`` pairs before pruning (an approximation).
            cluster_representative: "mean" (centroid; heuristic, no bound
                guarantee) or "plub" (componentwise min p / max q).
            verbosity: 0 silent, 1 summary, 2 per-atom detail.

        Returns:
            {atom -> (lower, upper)} with each a 2-vector [P(=0), P(=1)].
            Also sets self.marginals (the same dict), self.singleton_marginals
            ({atom -> (lo, hi)} for P(atom=1)), self.max_potential_size, the
            running times build_time / elimination_time / total_time, and
            self.degenerate.
        """
        assert coupling == "off", \
            f"coupling='{coupling}' is no longer supported by CredalVE " \
            f"(use the CredalJT engine for exact LCN bounds)."
        assert cluster_representative in ("plub", "mean"), \
            f"Unknown cluster_representative '{cluster_representative}'. " \
            f"Use 'plub' or 'mean'."
        assert elim_heuristic in ("topological", "min-fill"), \
            f"Unknown heuristic '{elim_heuristic}'. Use 'topological' or 'min-fill'."
        if epsilon is not None:
            assert epsilon >= 0, "Epsilon must be non-negative."

        node_atoms = self.cnv.cn.node_atoms
        node_of_atom = {a: v for v, ats in node_atoms.items() for a in ats}
        for a in evidence:
            assert a in node_of_atom, f"Evidence variable '{a}' not found."

        if verbosity > 0:
            print(f"[CredalVE] Computing all marginals (evidence={evidence})")

        t_elim_start = time.perf_counter()
        cards, base, topo = self._base_potentials(evidence)
        self.max_potential_size = max((p.m for p in base), default=0)
        eps = 0.0 if epsilon is None else float(epsilon)

        self.singleton_marginals = {}
        for node in self.cnv.cn.nodes:
            for i, atom in enumerate(node_atoms[node]):
                if atom in evidence:
                    continue
                target = self._atom_indicator(node, i)
                lo, hi = self._query(base, cards, topo, node, target,
                                     elim_heuristic, eps, n_clusters,
                                     cluster_representative, verbosity)
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
        MSB-first state packing of compound nodes (coupling / cjt convention)."""
        k = len(self.cnv.cn.node_atoms[node])
        n_states = 2 ** k
        bit = k - 1 - atom_index
        return np.array([((s >> bit) & 1) == value for s in range(n_states)],
                        dtype=float)

    # ------------------------------------------------------------------
    # One query: eliminate everything, read off min/max q/(p+q)
    # ------------------------------------------------------------------

    def _query(self, base, cards, topo, node, target, elim_heuristic, eps,
               n_clusters, rep, verbosity):
        pots = list(base)
        # Multiply the query potential into the node's own CPT (it is the one
        # potential guaranteed to contain the node).
        for j, p in enumerate(pots):
            if p.scope[0] == node:
                pots[j] = p.query(node, target)
                break

        if elim_heuristic == "topological":
            order = list(reversed(topo))
        else:
            order = min_fill_order([p.scope for p in pots], exclude=set())

        for var in order:
            bucket = [p for p in pots if var in p.scope]
            if not bucket:
                continue
            pots = [p for p in pots if var not in p.scope]
            combined = bucket[0]
            for p in bucket[1:]:
                combined = combined.product(p)
                if n_clusters > 0 and combined.m > n_clusters:
                    combined = combined.cluster(n_clusters, rep)
            res = combined.sum_out(var)
            if n_clusters > 0:
                res = res.cluster(n_clusters, rep)
            res = res.prune("both", eps)
            self.max_potential_size = max(self.max_potential_size,
                                          combined.m)
            pots.append(res)

        # All remaining potentials are scalars (empty scope): multiply them.
        final = pots[0]
        for p in pots[1:]:
            final = final.product(p).prune("both", eps)
        return final.bounds()

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


'''
s = s[:a] + NEW + s[b:]
s = s.replace('''    cve.run(evidence={}, elim_heuristic="min-fill", verbosity=verbosity)''',
              '''    cve.run(evidence={}, verbosity=verbosity)''')
s = s.replace('''    print("\\n=== All marginals (CredalVE, coupling=off) ===")''',
              '''    print("\\n=== All marginals (CredalVE) ===")''')
s = s.replace('''# CredalVE is one of the credal-network inference algorithms. It consumes a
# pre-built CredalNetworkVertices (the directed graph + enumerated extreme
# points of the local credal sets) and computes lower/upper bounds on marginal
# queries via bucket-based variable elimination over the extreme points.''',
              '''# CredalVE consumes a pre-built CredalNetworkVertices (the directed graph +
# enumerated extreme points of the local credal sets) and computes exact
# lower/upper bounds over the strong extension on P(atom = 1 | evidence) for
# every atom, by variable elimination over potentials of PAIRS of functions
# (Maua & Cozman, IJAR 2020, Algorithm 2); see pair_potentials.py.''')
open(P, 'w').write(s)
print("ok")
