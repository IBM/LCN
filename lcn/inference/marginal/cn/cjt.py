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

# Credal Junction Tree (CJT; internally "scheme D5"): exact marginal inference
# for chain-graph LCNs by one nonlinear program over junction-tree cluster
# marginals, of size exponential in the tree width rather than in n.
#
#   * Tree. Bucket elimination over the family scopes of the chain graph,
#     augmented by the scopes of cross-family sentences, along either a min-fill
#     order or a reverse topological order of the component DAG (the "ordered"
#     tree). The tree has the running-intersection property by construction
#     (H1). The narrowest tree on which H3 holds is used.
#   * Variables. One distribution q_C per cluster; adjacent clusters agree on
#     their separator (linear equalities). No joint, no separator variables, no
#     credal-network local sets or extreme points.
#   * Sentences. Each is imposed on a cluster containing its atoms (H2); cleared
#     Type-2 rows are linear.
#   * Independence assertions (H3). On a min-fill tree each assertion
#     X _||_ Y | Z is reduced exactly to X _||_ Y_in | Z, with Y_in the part of Y
#     connected to X in G* after deleting Z (empty => entailed by the tree); the
#     reduced assertion is hosted, after widening the tree if needed and within
#     budget, or else relaxed to hostable consequences (sound, but H3 then
#     fails). On the ordered tree one row per node implies every assertion.
#   * Queries. P(a) is read off a cluster containing a; P(a | e) is solved in
#     Charnes-Cooper form (variables scaled by 1 / P(e)), which is exact without
#     a denominator floor; when a and e share no cluster (H4) a tree augmented by
#     their node set is built for that atom.
#   * Solving. SCIP; an endpoint not certified within the time limit is reported
#     by its dual bound, so every interval is an outer bound, exact when H1-H4
#     hold and both endpoints are certified. ipopt is available as a local
#     (uncertified, inner-bound) backend.
#
# See the CJT appendix of docs/research-chain-lcn/iclr2027/iclr_2027.tex for the
# formulation and the proofs.

import logging
import math
import time
from typing import Dict, List

import numpy as np
from pyomo.environ import (
    ConcreteModel,
    Set, NonNegativeReals,
    Var, ConstraintList,
    Objective, minimize, maximize,
    SolverStatus, TerminationCondition,
    value,
)

# Local
from lcn.inference.marginal.cn.coupling import CouplingConstraints
from lcn.inference.marginal.cn.potentials import min_fill_order
from lcn.inference.marginal.exact import make_scip, _read_gap, _is_vacuous
from lcn.core.model import LCN, SentenceType
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.utils.common import (
    make_ipopt, build_truth_table,
    lmc_constraint_groups_vec, eval_indicator, make_conjunction,
)

_N_RESTARTS = 4
_VACUOUS_TOL = 1e-6


# ----------------------------------------------------------------------
# Junction-tree construction
# ----------------------------------------------------------------------

def _node_scopes(cnv):
    """The per-node CPT scopes [node] + parent_names, in cn.factors order
    (identical to what CVE builds and to _compute_induced_width).

    Derived from the CredalNetwork factors (cnv.cn), NOT the pyAgrum bn_min /
    enumerated extreme_points, so it works for the vertex-free build the
    CredalJT (D5) engine uses (extreme_points may be None there). The
    factor/child order matches the old extreme_points iteration order, and each
    factor[0] carries the same parents as bn_min's arcs, so the scopes are
    identical to the pyAgrum-derived ones."""
    scopes = []
    for factor in cnv.cn.factors:
        entry = factor[0]
        scopes.append([entry["child"]] + list(entry["parents"]))
    return scopes


def _jt_from_scopes(scopes, elim_order):
    """
    Bucket-tree construction restricted to the structural outputs CJT
    needs. Returns (parent, children, effective_scope) where effective_scope[v]
    is the node-name set of bucket v's cluster (the clique v lives in). Tree
    edges are (v, parent[v]); the construction gives a running-intersection
    clique tree.
    """
    buckets = {var: [] for var in elim_order}
    remaining = list(scopes)
    for var in elim_order:
        keep, rest = [], []
        for s in remaining:
            (keep if var in s else rest).append(s)
        buckets[var] = keep
        remaining = rest
    if remaining:
        buckets[elim_order[-1]].extend(remaining)

    parent = {var: None for var in elim_order}
    children = {var: [] for var in elim_order}
    effective_scope = {var: set() for var in elim_order}
    for var in elim_order:
        for s in buckets[var]:
            effective_scope[var].update(s)

    for i, var in enumerate(elim_order):
        msg_scope = effective_scope[var] - {var}
        for j in range(i + 1, len(elim_order)):
            candidate = elim_order[j]
            if candidate in msg_scope:
                parent[var] = candidate
                children[candidate].append(var)
                effective_scope[candidate].update(msg_scope)
                break

    return parent, children, effective_scope


def _topological_nodes(cnv):
    """Chain-graph node DAG (factor child <- parents) in a deterministic
    topological order (lexicographic tie-break)."""
    import networkx as nx
    G = nx.DiGraph()
    for factor in cnv.cn.factors:
        entry = factor[0]
        G.add_node(entry["child"])
        for p in entry["parents"]:
            G.add_edge(p, entry["child"])
    return list(nx.lexicographical_topological_sort(G))


def _node_parents(cnv):
    """node -> set of parent nodes in the chain-graph node DAG."""
    return {f[0]["child"]: set(f[0]["parents"]) for f in cnv.cn.factors}


class _JTClusters:
    """
    Atom-level junction tree: clusters (atom sets), tree edges, separators.

    The clique tree is built by bucket elimination over the family scopes,
    augmented by extra "clique scopes": the node set of every cross-family
    sentence, any widening scopes added by the LMC triage and, for a
    single-query build, the query+evidence node set. Forcing those node sets to
    co-occur guarantees a cluster covering each of them, so every such
    constraint has a host. ``order`` is "min-fill" or "topological" (reverse
    topological order of the component DAG; query-independent builds only).
    """

    def __init__(self, cnv, query, evidence, extra_node_sets, order="min-fill"):
        """
        Build the augmented junction tree.

        ``query`` may be ``None`` (the query-INDEPENDENT tree used by the
        all-marginals CredalJT engine): then only the cross-family constraint
        node sets augment the scopes. Every singleton atom is still hosted in some cluster
        (each node's own scope is a clique), so any atom's marginal can be read
        off a hosting cluster -- one tree serves all marginals. When ``query`` is
        a node name (single-query build_and_solve_jt_nlp), the query+evidence
        node set is added as an extra clique, so some cluster hosts the query
        and the evidence together (H4). Every node, the query's included, is
        eliminated: excluding one would leave the buckets whose message is just
        that node without a parent, giving a forest in which several clusters
        share the node with no separator tying their copies (an H1 violation).
        """
        self.cnv = cnv
        node_atoms = cnv.cn.node_atoms
        # One global atom order so every cluster/separator packs MSB-first
        # consistently with coupling._node_state_to_atom_bits.
        self.atom_order = list(cnv.lcn.atoms.keys())
        self._rank = {a: i for i, a in enumerate(self.atom_order)}

        node_scopes = _node_scopes(cnv)
        aug = [list(s) for s in extra_node_sets]
        exclude = set()
        if query is not None:
            # Augmenting clique: the query+evidence node set. No node is
            # excluded from elimination (see the docstring).
            node_of_atom = {}
            for node, atoms in node_atoms.items():
                for a in atoms:
                    node_of_atom[a] = node
            q_nodes = sorted({node_of_atom[a]
                              for a in ([query] + list(evidence.keys()))
                              if a in node_of_atom})
            if q_nodes:
                aug = [q_nodes] + aug
        aug = [s for s in aug if s]

        scopes = node_scopes + aug
        self.order = order
        if order == "topological" and query is None:
            # Ordered-Markov tree: eliminate the chain-graph node DAG in REVERSE
            # topological order. Then node x's cluster is {x} u N_x with N_x a set
            # of topological PREDECESSORS of x containing pa(x); imposing
            # (x _||_ N_x \ pa(x) | pa(x)) on that cluster makes every
            # reconstructed joint factorize as prod_x p(x | pa(x)) -- the DAG
            # factorization -- so every LMC assertion holds (see _triage_lmc).
            elim_order = list(reversed(_topological_nodes(cnv)))
        else:
            self.order = "min-fill"
            elim_order = min_fill_order(scopes, exclude=exclude)
        self.elim_order = list(elim_order)
        parent, children, eff = _jt_from_scopes(scopes, elim_order)
        # Node-level elimination clique of every node (before leaf pruning):
        # eff[v] is the node set of v's bucket cluster.
        self.node_clique = {v: set(eff[v]) for v in elim_order}

        self.cluster_ids = list(elim_order)
        self.atoms = {}          # cid -> sorted atom list
        for cid in self.cluster_ids:
            atoms = set()
            for node in eff[cid]:
                atoms.update(node_atoms[node])
            self.atoms[cid] = self._sort(atoms)

        # Prune subsumed leaf clusters: a leaf whose atoms are a subset of its
        # parent's atoms is redundant (the parent already represents that joint),
        # so drop it to avoid duplicate variables + separator constraints. Safe
        # because a leaf has no children to reattach. Iterate until stable.
        changed = True
        while changed and len(self.cluster_ids) > 1:
            changed = False
            for cid in list(self.cluster_ids):
                par = parent.get(cid)
                if par is None:
                    continue
                is_leaf = all(parent.get(o) != cid for o in self.cluster_ids)
                if is_leaf and set(self.atoms[cid]).issubset(self.atoms[par]):
                    self.cluster_ids.remove(cid)
                    self.atoms.pop(cid, None)
                    parent.pop(cid, None)
                    changed = True

        # Tree edges (child, parent) and their separators (atom intersection).
        self.edges = []
        self.sep = {}
        self.parent = {}      # cid -> parent cid (root has no entry)
        self.children = {cid: [] for cid in self.cluster_ids}
        for cid in self.cluster_ids:
            par = parent.get(cid)
            if par is None or par not in self.atoms:
                continue
            shared = set(self.atoms[cid]) & set(self.atoms[par])
            self.edges.append((cid, par))
            self.sep[(cid, par)] = self._sort(shared)
            self.parent[cid] = par
            self.children[par].append(cid)

    def rip_holds(self):
        """H1, checked explicitly: for every atom, the clusters containing it
        induce a connected subtree of the cluster tree (forest). Then the
        separator equalities tie every copy of the atom together, which the
        reconstruction (lem:rip) needs."""
        nbrs = {c: set(self.children.get(c, [])) for c in self.cluster_ids}
        for c, p in self.parent.items():
            nbrs[c].add(p)
        for a in self.atom_order:
            holders = {c for c in self.cluster_ids if a in self.atoms[c]}
            if not holders:
                continue
            start = next(iter(holders))
            seen, stack = {start}, [start]
            while stack:
                c = stack.pop()
                for nb in nbrs[c]:
                    if nb in holders and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            if seen != holders:
                return False
        return True

    def roots(self):
        """Cluster ids with no parent (one per connected component)."""
        return [cid for cid in self.cluster_ids if cid not in self.parent]

    def _sort(self, atoms):
        return sorted(atoms, key=lambda a: self._rank[a])

    def host(self, needed_atoms):
        """First cluster whose atoms contain `needed_atoms`, or None."""
        need = set(needed_atoms)
        for cid in self.cluster_ids:
            if need.issubset(self.atoms[cid]):
                return cid
        return None

    def max_cluster_size(self):
        return max(len(self.atoms[c]) for c in self.cluster_ids)

    def _subtree(self, child, parent):
        """Cluster ids in the component containing `child` after removing the
        tree edge (child, parent) -- i.e. `child` and everything reachable from
        it without crossing back through `parent`."""
        seen = set()
        stack = [child]
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            for nb in self.children.get(c, []) + ([self.parent[c]]
                                                  if c in self.parent else []):
                if nb != parent and nb not in seen:
                    stack.append(nb)
        return seen

    def _atoms_of(self, cluster_ids):
        atoms = set()
        for c in cluster_ids:
            atoms.update(self.atoms[c])
        return atoms

    # ------------------------------------------------------------------
    # Separation in the chordal graph G* and the LMC reduction
    # ------------------------------------------------------------------

    def _adjacency(self):
        """Adjacency of the chordal graph G*: two atoms are adjacent iff they
        co-occur in some cluster (the clusters are G*'s maximal cliques)."""
        if getattr(self, "_adj", None) is None:
            adj = {a: set() for a in self.atom_order}
            for cid in self.cluster_ids:
                ats = self.atoms[cid]
                for a in ats:
                    adj[a].update(b for b in ats if b != a)
            self._adj = adj
        return self._adj

    def reachable(self, X, Z):
        """Atoms reachable from the set ``X`` in G* after deleting ``Z``
        (``X`` itself included, ``Z`` excluded)."""
        adj = self._adjacency()
        Zs = set(Z)
        seen = set(a for a in X if a not in Zs)
        stack = list(seen)
        while stack:
            a = stack.pop()
            for b in adj.get(a, ()):
                if b not in Zs and b not in seen:
                    seen.add(b)
                    stack.append(b)
        return seen

    def separates(self, X, Y, Z):
        """True iff ``Z`` separates ``X`` from ``Y`` in G* (every path from X to
        Y meets Z). This is the separator-entailment test of the paper's
        Definition (cluster-feasible set): every joint reconstructed from a
        separator-consistent cluster family factorizes over G*, hence satisfies
        G*'s global Markov property (no positivity needed), hence X perp Y | Z."""
        Ys = set(Y) - set(Z)
        return not (self.reachable(X, Z) & Ys)

    def reduce_assertion(self, X, Y, Z):
        """
        Exact reduction of ``X perp Y | Z`` against G*.

        Let K = reachable(X, Z). Split Y into Y_in = Y & K and Y_out = Y - K.
        Then, for every joint p that factorizes over G* (in particular every
        joint the cluster NLP can reconstruct):

            X perp Y | Z   <=>   X perp Y_in | Z.

        (=>) is decomposition. (<=) Z separates K (which contains X and Y_in)
        from Y_out in G*, so global Markov gives (X u Y_in) perp Y_out | Z, and
        weak union gives X perp Y_out | Z, Y_in; together with X perp Y_in | Z,
        contraction yields X perp Y_in u Y_out | Z. Neither step needs
        positivity.

        Returns Y_in (sorted). An empty result means the assertion is fully
        entailed by G* and need not be imposed.
        """
        K = self.reachable(X, Z)
        return sorted(set(Y) & K - set(Z))

    def _rip_implied(self, X, Y, Z):
        """Alias of :meth:`separates`: True iff ``X perp Y | Z`` is entailed by
        G*. Note that Z itself must separate X from Y; a tree separator S that
        merely contains Z only enforces X perp Y | S."""
        return self.separates(X, Y, Z)

    def describe(self, sentences_by_host=None, lmc_by_host=None,
                 query_cluster=None) -> str:
        """
        Render the junction tree and the separator messages as a string.

        The D5 NLP does not pass numeric messages; it ties adjacent clusters with
        Lauritzen separator-consistency equalities and solves them jointly. The
        "message" on a tree edge (c -> parent p) with separator S is therefore
        the marginal of the cluster onto S, which both endpoints must agree on:
        ``M_{c->S} q_c == M_{p->S} q_p``. This method lists the clusters (with
        their atom scopes and what each hosts), the tree as an indented forest,
        and the per-edge separator messages.

        Args:
            sentences_by_host / lmc_by_host: optional {cluster -> [sentence ids /
                assertions]} maps (as built in build_and_solve_jt_nlp) so the
                description shows which constraints each cluster carries.
            query_cluster: optional cluster id holding the query (+ evidence).
        """
        sentences_by_host = sentences_by_host or {}
        lmc_by_host = lmc_by_host or {}
        lines = []
        n_atoms = len(self.atom_order)
        lines.append(
            f"Junction tree: {len(self.cluster_ids)} cluster(s), "
            f"max cluster {self.max_cluster_size()} atoms "
            f"(of {n_atoms} total); {len(self.edges)} edge(s).")

        # Clusters and what they host.
        lines.append("Clusters (cluster: atoms [size]  -> hosted):")
        for cid in self.cluster_ids:
            atoms = self.atoms[cid]
            tags = []
            if cid == query_cluster:
                tags.append("QUERY")
            sids = sentences_by_host.get(cid, [])
            if sids:
                tags.append("sentences=" + ",".join(str(s) for s in sids))
            indeps = lmc_by_host.get(cid, [])
            if indeps:
                tags.append("LMC=" + "; ".join(str(a) for a in indeps))
            host = ("  -> " + " | ".join(tags)) if tags else ""
            lines.append(f"  {cid}: {{{', '.join(atoms)}}} "
                         f"[2^{len(atoms)}={2 ** len(atoms)} states]{host}")

        # Tree as an indented forest (root has no parent).
        lines.append("Tree (root at top, children indented):")

        def _walk(cid, depth):
            lines.append("    " * depth + f"- {cid} {{{', '.join(self.atoms[cid])}}}")
            for ch in sorted(self.children.get(cid, [])):
                _walk(ch, depth + 1)
        for r in sorted(self.roots()):
            _walk(r, 0)

        # Separator messages, one per edge (child -> parent).
        lines.append("Messages (separator-consistency equalities, child -> parent):")
        if not self.edges:
            lines.append("  (no edges -- single cluster or disconnected)")
        for (c, p) in self.edges:
            S = self.sep[(c, p)]
            sep_str = "{" + ", ".join(S) + "}" if S else "{} (empty separator)"
            lines.append(
                f"  {c} -> {p}  over separator {sep_str}:  "
                f"q[{c}] marginalized to {sep_str} == "
                f"q[{p}] marginalized to {sep_str}")
        return "\n".join(lines)

    def describe_detailed(self, sentences_by_host=None, lmc_by_host=None,
                          query_cluster=None) -> str:
        """
        Like :meth:`describe`, but expands every separator-consistency message
        into its explicit per-state equalities -- the exact linear rows the D5
        NLP enforces. For a tree edge (c -> p) with separator S, each state s of
        S yields one equality

            sum_{j: c-state j projects to s} q[c][j]
              == sum_{k: p-state k projects to s} q[p][k]

        i.e. the marginal of cluster c onto S equals the marginal of cluster p
        onto S. The cluster states are rendered as readable atom assignments
        (MSB-first). This is the message-level detail of the junction-tree
        calibration.
        """
        lines = [self.describe(sentences_by_host, lmc_by_host, query_cluster)]
        lines.append("")
        lines.append("Detailed separator-consistency equalities "
                     "(one per separator state, per edge):")
        if not self.edges:
            lines.append("  (no edges -- single cluster or disconnected)")
        for (c, p) in self.edges:
            S = self.sep[(c, p)]
            atoms_c, atoms_p = self.atoms[c], self.atoms[p]
            sep_str = "{" + ", ".join(S) + "}"
            lines.append(f"  edge {c} -> {p}, separator {sep_str}"
                         f"{' (empty)' if not S else ''}:")
            if not S:
                # Empty separator: the only shared marginal is the total mass,
                # which both simplices already pin to 1.
                lines.append("    sum(q[%s]) == sum(q[%s]) == 1 "
                             "(no shared atoms)" % (c, p))
                continue
            Mc = _marginal_matrix(atoms_c, S)
            Mp = _marginal_matrix(atoms_p, S)
            for r in range(Mc.shape[0]):
                s_label = _state_label(S, r)
                lhs = " + ".join(
                    f"q[{c}|{_state_label(atoms_c, j)}]"
                    for j in np.nonzero(Mc[r])[0])
                rhs = " + ".join(
                    f"q[{p}|{_state_label(atoms_p, k)}]"
                    for k in np.nonzero(Mp[r])[0])
                lines.append(f"    P({s_label}):  {lhs}")
                lines.append(f"      == {rhs}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Marginalization matrix (cluster joint -> separator joint)
# ----------------------------------------------------------------------

def _state_label(atoms: List[str], idx: int) -> str:
    """Human-readable label of state `idx` (MSB-first) over `atoms`,
    e.g. atoms=[A,C], idx=2 -> "A=1,C=0"."""
    n = len(atoms)
    bits = [(idx >> (n - 1 - i)) & 1 for i in range(n)]
    return ",".join(f"{a}={b}" for a, b in zip(atoms, bits))


def _marginal_matrix(atoms_c: List[str], atoms_s: List[str]) -> np.ndarray:
    """
    0/1 matrix M of shape (2^|S|, 2^|C|): M[r, j] = 1 iff cluster interpretation
    j (MSB-first over atoms_c) projects to separator row r (MSB-first over
    atoms_s). So (M @ q_c)[r] = P(separator = r). atoms_s must be a subset of
    atoms_c (verified by the caller via the host/RIP structure).
    """
    nc = len(atoms_c)
    ns = len(atoms_s)
    tbl_c = build_truth_table(nc)
    col_c = {a: i for i, a in enumerate(atoms_c)}
    s_cols = [col_c[a] for a in atoms_s]
    M = np.zeros((2 ** ns, 2 ** nc), dtype=float)
    for j in range(2 ** nc):
        row_bits = tbl_c[j]
        r = 0
        for c in s_cols:
            r = (r << 1) | int(row_bits[c])
        M[r, j] = 1.0
    return M


# ----------------------------------------------------------------------
# NLP build + solve
# ----------------------------------------------------------------------

def _dotq(vec, q, cid, n):
    """Pyomo linear expression vec @ q[cid, .] over the 2^n cluster states."""
    return sum(float(vec[i]) * q[(cid, i)] for i in range(n))


def _build_constraint_model(jt, cnv, sentences_by_host, lmc_by_host,
                            evidence=None):
    """
    Build the constraint-only Pyomo JT model: cluster variables + per-cluster
    simplex + separator-consistency equalities + sentence rows + LMC bilinear
    rows. NO objective. This part is QUERY-INDEPENDENT, so the all-marginals
    engine builds it once and reuses it for every atom (only the objective,
    added by ``_atom_objective``, changes per atom). Returns (model, csize).

    With ``evidence`` (a non-empty {atom: value} dict) the model is built in
    CHARNES-COOPER form for the conditional queries P(a | e): the variables are
    y_C = r * q_C with a scale r = 1 / P(e) >= 0, the simplex rows become
    sum_i y_{C,i} = r, and one normalization row <A_e, y_{C_e}> = 1 is added on
    a cluster C_e containing all evidence atoms. Each row is made homogeneous
    by multiplying its lower-degree terms by powers of r: the Type-1 sentence
    rows lo <= <A, q> <= hi become lo * r <= <A, y> <= hi * r, and the marginal
    LMC rows <A_a, q> = <A_b, q><A_c, q> become r <A_a, y> = <A_b, y><A_c, y>.
    The separator rows, the cleared Type-2 rows and the conditional LMC rows
    (product = product) are already homogeneous. Each scaled row holds for y
    iff the original holds for q = y / r. P(a | e) = P(a & e) / P(e) then equals
    the LINEAR objective <A_{a&e}, y>. This is exact for every q with P(e) > 0,
    needs no denominator floor, and is infeasible iff P(e) = 0 for every q.
    ``model.ev_cluster`` records C_e (None if no cluster hosts the evidence).
    """
    lcn = cnv.lcn
    model = ConcreteModel()
    evidence = evidence or {}

    # Cluster variables: one flat indexed Var over (cid, state).
    keys = []
    csize = {}
    for cid in jt.cluster_ids:
        n = 2 ** len(jt.atoms[cid])
        csize[cid] = n
        keys.extend((cid, i) for i in range(n))
    model.KEYS = Set(initialize=keys, dimen=2)
    model.constr = ConstraintList()
    model.ev_cluster = None
    if evidence:
        # Scaled variables y = r q (unbounded above); r = 1 / P(e).
        model.q = Var(model.KEYS, within=NonNegativeReals)
        model.r = Var(within=NonNegativeReals)
        mass = model.r
    else:
        model.q = Var(model.KEYS, within=NonNegativeReals, bounds=(0.0, 1.0))
        mass = 1.0

    # Per-cluster simplex (scaled: total mass r).
    for cid in jt.cluster_ids:
        model.constr.add(
            sum(model.q[(cid, i)] for i in range(csize[cid])) == mass)

    if evidence:
        ce = jt.host(list(evidence.keys()))
        model.ev_cluster = ce
        if ce is not None:
            atoms_c = jt.atoms[ce]
            interps = [dict(zip(atoms_c, row))
                       for row in build_truth_table(len(atoms_c))]
            Fe = make_conjunction(variables=list(evidence.keys()),
                                  literals=evidence)
            E = eval_indicator(Fe, interps)
            model.constr.add(_dotq(E, model.q, ce, csize[ce]) == 1.0)
            # Valid bounds of the scaled program (thm:d5): r = sum_i y_C,i >=
            # <A_e, y_Ce> = 1, and y = r q <= r since q <= 1. Without them the
            # program is unbounded and spatial branch-and-bound stalls.
            # Kept in their own list: they are valid inequalities (implied by
            # the program), not rows of the formulation counted by
            # _model_stats / thm:cjt-size.
            model.r.setlb(1.0)
            model.cc_bounds = ConstraintList()
            for key in keys:
                model.cc_bounds.add(model.q[key] <= model.r)

    # Separator consistency: M_{c->S} q_c == M_{d->S} q_d for each edge.
    for (c, d) in jt.edges:
        S = jt.sep[(c, d)]
        if not S:
            continue  # disconnected component share nothing -> no tie needed
        Mc = _marginal_matrix(jt.atoms[c], S)
        Md = _marginal_matrix(jt.atoms[d], S)
        for r in range(Mc.shape[0]):
            model.constr.add(
                _dotq(Mc[r], model.q, c, csize[c])
                == _dotq(Md[r], model.q, d, csize[d]))

    # Sentence rows on their host cluster.
    for cid, sids in sentences_by_host.items():
        atoms_c = jt.atoms[cid]
        interps = [dict(zip(atoms_c, row))
                   for row in build_truth_table(len(atoms_c))]
        n = csize[cid]
        for sid in sids:
            s = lcn.sentences.get(sid)
            lo = s.get_lower_bound()
            hi = s.get_upper_bound()
            if s.type == SentenceType.Type1:
                # lo <= P(phi) <= hi; in the scaled model P(phi) = <A, y> / r,
                # so the constant bounds scale with the mass r.
                A = eval_indicator(s.phi_formula, interps)
                expr = _dotq(A, model.q, cid, n)
                model.constr.add(expr >= lo * mass)
                model.constr.add(expr <= hi * mass)
            else:
                Aqr = eval_indicator(s.phi_and_psi_formula, interps)
                Ar = eval_indicator(s.psi_formula, interps)
                eqr = _dotq(Aqr, model.q, cid, n)
                er = _dotq(Ar, model.q, cid, n)
                model.constr.add(eqr >= lo * er)
                model.constr.add(eqr <= hi * er)

    # LMC bilinear rows on their host cluster.
    for cid, indeps in lmc_by_host.items():
        atoms_c = jt.atoms[cid]
        table = build_truth_table(len(atoms_c))
        col_of = {v: i for i, v in enumerate(atoms_c)}
        n = csize[cid]
        for indep in indeps:
            for group in lmc_constraint_groups_vec(indep, table, col_of):
                if group[0] == 'conditional':
                    _, Aa, Ab, Ac, Ad = group
                    v1 = _dotq(Aa, model.q, cid, n) * _dotq(Ab, model.q, cid, n)
                    v2 = _dotq(Ac, model.q, cid, n) * _dotq(Ad, model.q, cid, n)
                    model.constr.add(v1 - v2 == 0.0)
                else:
                    # Marginal form P(x, y) = P(x) P(y): degree 1 vs degree 2,
                    # so in the scaled model (q = y / r) the left side carries
                    # one factor of the mass r.
                    _, Aa, Ab, Ac = group
                    v1 = mass * _dotq(Aa, model.q, cid, n)
                    v2 = _dotq(Ab, model.q, cid, n) * _dotq(Ac, model.q, cid, n)
                    model.constr.add(v1 - v2 == 0.0)

    return model, csize


def _model_stats(jt, csize, sentences_by_host, lmc_by_host, cnv):
    """
    Size of the (query-independent) constraint NLP that _build_constraint_model
    builds: variable and constraint counts, broken down by category, plus the
    number of nonlinear (bilinear) constraint rows.

    Counts are derived structurally from the junction tree + host assignments
    (they match the rows _build_constraint_model actually adds), so this is cheap
    and does not need the built Pyomo model. The objective adds at most one more
    variable (obj_var, only under evidence) and one ratio constraint per atom;
    those are per-atom and excluded here since the constraint model is shared.

    Returns a dict:
        n_clusters, max_cluster_atoms, n_atoms_total, n_edges,
        n_variables (= sum_c 2^|atoms(c)|),
        n_constraints (total), and the per-category counts
        n_simplex, n_separator, n_sentence, n_lmc,
        n_nonlinear (rows that are nonlinear in q: the LMC rows only).
        Sentence rows are always linear: a Type-2 sentence is imposed in its
        cleared form lo*P(psi) <= P(phi & psi) <= hi*P(psi), and lo, hi are
        constants.
    """
    # Variables: one per (cluster, state).
    n_variables = sum(csize[cid] for cid in jt.cluster_ids)

    # Simplex: one equality per cluster.
    n_simplex = len(jt.cluster_ids)

    # Separator consistency: 2^|S| equalities per (non-empty-separator) edge.
    n_separator = 0
    for (c, d) in jt.edges:
        S = jt.sep[(c, d)]
        if S:
            n_separator += 2 ** len(S)

    # Sentence rows: two per sentence (>= lo, <= hi), all LINEAR in q. A Type-1
    # row is lo <= <A_phi, q> <= hi; a Type-2 row is the cleared
    # lo*<A_psi, q> <= <A_{phi&psi}, q> <= hi*<A_psi, q> with constant lo, hi.
    n_sentence = 2 * sum(len(sids) for sids in sentences_by_host.values())

    # LMC rows: one equality per constraint group. 'conditional' groups are
    # <A_a,q><A_b,q> == <A_c,q><A_d,q> and 'marginal' groups are
    # <A_a,q> == <A_b,q><A_c,q>; both contain a product of two linear forms, so
    # every LMC row is bilinear (nonlinear).
    n_lmc = 0
    n_lmc_nonlinear = 0
    for cid, indeps in lmc_by_host.items():
        atoms_c = jt.atoms[cid]
        table = build_truth_table(len(atoms_c))
        col_of = {v: i for i, v in enumerate(atoms_c)}
        for indep in indeps:
            for _group in lmc_constraint_groups_vec(indep, table, col_of):
                n_lmc += 1
                n_lmc_nonlinear += 1

    n_constraints = n_simplex + n_separator + n_sentence + n_lmc
    n_nonlinear = n_lmc_nonlinear

    return {
        "n_clusters": len(jt.cluster_ids),
        "max_cluster_atoms": jt.max_cluster_size(),
        "n_atoms_total": len(jt.atom_order),
        "n_edges": len(jt.edges),
        "n_variables": n_variables,
        "n_constraints": n_constraints,
        "n_simplex": n_simplex,
        "n_separator": n_separator,
        "n_sentence": n_sentence,
        "n_lmc": n_lmc,
        "n_nonlinear": n_nonlinear,
    }


def _format_model_stats(stats) -> str:
    """One-block human-readable rendering of the _model_stats dict."""
    return (
        f"  clusters:        {stats['n_clusters']} "
        f"(max {stats['max_cluster_atoms']} atoms; "
        f"{stats['n_edges']} edges; {stats['n_atoms_total']} atoms total)\n"
        f"  variables:       {stats['n_variables']} "
        f"(cluster-state simplex vars)\n"
        f"  constraints:     {stats['n_constraints']} total "
        f"({stats['n_nonlinear']} nonlinear/bilinear)\n"
        f"    simplex:       {stats['n_simplex']}\n"
        f"    separator:     {stats['n_separator']}\n"
        f"    sentence:      {stats['n_sentence']}\n"
        f"    LMC:           {stats['n_lmc']}")


def _atom_objective(model, jt, csize, query, query_cluster, evidence, solver):
    """
    Objective expression for P(query=1 | evidence), read off ``query_cluster``
    (a cluster whose atoms contain the query and the evidence atoms).

    Without evidence this is the linear <A_query, q>. With evidence the model
    must have been built in Charnes-Cooper form (``_build_constraint_model`` with
    the same ``evidence``), and the objective is the linear <A_{query & e}, y>,
    which equals P(query | e) for the unscaled q = y / r. ``solver`` is unused
    (kept for call compatibility). Returns the objective expression.
    """
    cid = query_cluster
    atoms_c = jt.atoms[cid]
    interps = [dict(zip(atoms_c, row))
               for row in build_truth_table(len(atoms_c))]
    n = csize[cid]
    Fq = make_conjunction(variables=[query], literals={query: 1})
    A_q = eval_indicator(Fq, interps)
    if evidence:
        assert hasattr(model, "r"), \
            "conditional query needs the Charnes-Cooper model (pass evidence)"
        Fe = make_conjunction(variables=list(evidence.keys()), literals=evidence)
        E = eval_indicator(Fe, interps)
        return _dotq(A_q * E, model.q, cid, n)
    return _dotq(A_q, model.q, cid, n)


def _clear_atom_objective(model):
    """Remove the per-atom objective components added by _atom_objective so the
    shared constraint model can be reused for the next atom."""
    for name in ("objective",):
        if hasattr(model, name):
            model.del_component(name)


def _build_model(jt, cnv, query, evidence, sentences_by_host,
                 lmc_by_host, query_cluster, solver):
    """
    Single-query convenience: build the constraint model and attach the query
    objective. Used by build_and_solve_jt_nlp. Returns (model, obj_expr, csize).
    """
    model, csize = _build_constraint_model(jt, cnv, sentences_by_host,
                                           lmc_by_host, evidence=evidence)
    obj_expr = _atom_objective(model, jt, csize, query, query_cluster,
                               evidence, solver)
    return model, obj_expr, csize


def _is_suspicious(val, ok, sense):
    if not ok or val is None:
        return True
    if sense == 'max' and val >= 1.0 - _VACUOUS_TOL:
        return True
    if sense == 'min' and val <= _VACUOUS_TOL:
        return True
    return False


def _init_q(model, jt, csize, rng=None):
    """Starting point for the local solver: uniform (or random) cluster
    distributions; in the Charnes-Cooper model scaled by r = 1."""
    scale = 1.0
    if hasattr(model, "r"):
        model.r.value = 1.0
    for cid in jt.cluster_ids:
        n = csize[cid]
        if rng is None:
            start = np.full(n, 1.0 / n)
        else:
            start = rng.random(n)
            start = start / start.sum()
        for i in range(n):
            model.q[(cid, i)].value = scale * float(start[i])


def _print_realized_messages(model, jt, csize, prefix=""):
    """
    Print the separator messages REALIZED by the current solution loaded in
    ``model.q`` -- i.e. the actual numeric marginal each cluster sends over each
    separator at the optimum (both endpoints agree on it by the separator-
    consistency equalities). Read-only; call after a solve loaded a solution.
    """
    for (c, p) in jt.edges:
        S = jt.sep[(c, p)]
        if not S:
            continue
        Mc = _marginal_matrix(jt.atoms[c], S)
        try:
            qc = np.array([float(value(model.q[(c, j)]))
                           for j in range(csize[c])])
        except Exception:
            return  # no solution loaded
        msg = Mc @ qc
        cells = ", ".join(f"P({_state_label(S, r)})={msg[r]:.4f}"
                          for r in range(len(msg)))
        print(f"{prefix}  message {c} -> {p} over {{{', '.join(S)}}}: {cells}")


def _scip_bounds(results):
    """(primal, dual, gap) as reported by SCIP through the AMPL/NL interface
    (any may be None). For a minimization dual <= optimum <= primal; for a
    maximization primal <= optimum <= dual."""
    def _f(key):
        try:
            v = results.solver[0][key].value
        except Exception:
            try:
                v = results.solver[0][key]
            except Exception:
                return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(v) or math.isinf(v) else v
    return _f('Primal bound'), _f('Dual bound'), _read_gap(results)


# Termination conditions that certify a GLOBAL optimum for SCIP. Note that
# ``feasible`` (an incumbent without a proof) is deliberately NOT included.
_SCIP_CERTIFIED = {TerminationCondition.optimal,
                   TerminationCondition.globallyOptimal}
# Default absolute primal-dual tolerance accepted as a certificate, in
# probability units: an endpoint is "certified" when SCIP proves optimality or
# |primal - dual| <= cert_tol. The reported endpoint is always the DUAL bound
# when not proven optimal, so it is sound regardless of this tolerance; the
# tolerance only decides whether the endpoint is called tight.
_CERT_ABS_TOL = 1e-4
# Terminations with no meaningful objective value.
_SCIP_NO_SOLUTION = {TerminationCondition.infeasible,
                     TerminationCondition.unbounded,
                     TerminationCondition.infeasibleOrUnbounded}


def _is_prob(v, tol=1e-6):
    """True iff ``v`` is a number in [0, 1] up to ``tol``: every CJT objective is
    a probability, so anything else (e.g. SCIP's +-1e20 sentinels) is invalid."""
    return v is not None and -tol <= v <= 1.0 + tol


def _solve_sense(model, jt, obj_expr, csize, sense, solver, time_limit,
                 gap_tol, verbosity, status_out=None, cert_tol=_CERT_ABS_TOL):
    """
    Solve one sense (min/max) with the chosen backend; return the endpoint value
    or None (=> the caller reports the vacuous endpoint).

    SCIP: the returned endpoint is always SOUND. If SCIP certifies global
    optimality (termination optimal, or gap <= gap_tol) the optimum is returned;
    otherwise (time limit, node limit, ...) its DUAL bound is returned instead of
    the incumbent -- the incumbent of an unfinished min lies ABOVE the true
    minimum, so reporting it would make the interval an unsound inner bound.
    The dual bound is <= the true min (resp. >= the true max), so the interval
    stays an outer bound, just possibly loose.

    ``status_out`` (optional list) receives one dict per call:
    {sense, value, primal, dual, gap, termination, certified}. ipopt solves are
    always recorded with certified=False (local solver).
    """
    if hasattr(model, 'objective'):
        model.del_component('objective')
    model.objective = Objective(
        expr=obj_expr, sense=(minimize if sense == 'min' else maximize))

    def _record(**kw):
        if status_out is not None:
            status_out.append(dict(sense=sense, **kw))

    if solver == "scip":
        opt = make_scip(time_limit=(time_limit or 3600.0), gap_tol=gap_tol)
        try:
            results = opt.solve(model, load_solutions=False,
                                tee=(verbosity > 2))
        except Exception as ex:
            if verbosity > 1:
                print(f"[CJT] scip exception: {ex}")
            _record(value=None, primal=None, dual=None, gap=None,
                    termination="exception", certified=False)
            return None
        tc = results.solver.termination_condition
        if tc in _SCIP_NO_SOLUTION:
            # Infeasible / unbounded: SCIP reports +-1e20 sentinels as its
            # bounds. Never treat them as an optimum; the caller reports the
            # vacuous endpoint. (For a conditional query, an infeasible scaled
            # program means P(e) = 0 for every feasible point.)
            _record(value=None, primal=None, dual=None, gap=None,
                    cert_tol=cert_tol, termination=str(tc), certified=False)
            return None
        primal, dual, gap = _scip_bounds(results)
        primal = primal if _is_prob(primal) else None
        dual = dual if _is_prob(dual) else None
        incumbent = None
        try:
            model.solutions.load_from(results)
            v = value(model.objective, exception=False)
            incumbent = float(v) if v is not None else None
        except Exception:
            incumbent = None
        if not _is_prob(incumbent):
            incumbent = primal
        # Certificate: SCIP proved optimality, OR the primal and dual bounds
        # have met (|primal - dual| <= abs tol). SCIP's NL "Gap" is a relative
        # PERCENTAGE and is reported as >0 when the objective is ~0, so it is
        # not used directly; the primal/dual comparison is the sound test.
        bounds_met = (primal is not None and dual is not None
                      and abs(primal - dual) <= cert_tol)
        # With gap_tol > 0 SCIP reports "optimal" once the RELATIVE gap is
        # below gap_tol, so "optimal" proves optimality only when gap_tol == 0.
        proven = tc in _SCIP_CERTIFIED and not (gap_tol and gap_tol > 0)
        certified = (incumbent is not None) and (proven or bounds_met)
        if proven:
            val = incumbent
        else:
            # Not proven optimal (certified within cert_tol, or not at all):
            # report the sound dual bound (None if absent, which the caller
            # turns into the vacuous endpoint), never the incumbent.
            val = dual
        _record(value=val, primal=primal, dual=dual, gap=gap,
                cert_tol=cert_tol,
                termination=str(tc), certified=bool(certified))
        return val

    # ipopt multi-restart
    opt = make_ipopt(debug=(verbosity > 2), mode="exact")
    if time_limit is not None:
        opt.options['max_cpu_time'] = float(time_limit)
        opt.options['max_wall_time'] = float(time_limit)

    def _once():
        try:
            res = opt.solve(model, tee=(verbosity > 2))
            tc = res.solver.termination_condition
            st = res.solver.status
            v = value(model.objective, exception=False)
            v = float(v) if v is not None else None
            ok = (st == SolverStatus.ok
                  and tc == TerminationCondition.optimal) \
                or tc in (TerminationCondition.locallyOptimal,
                          TerminationCondition.feasible) \
                or str(tc).lower() == 'acceptable'
            return v, ok
        except Exception as ex:
            if verbosity > 1:
                print(f"[CJT] ipopt exception: {ex}")
            return None, False

    _init_q(model, jt, csize)
    best, ok = _once()
    if _is_suspicious(best, ok, sense):
        for k in range(_N_RESTARTS):
            rng = np.random.default_rng(abs(hash((sense, k))) % (2 ** 32))
            _init_q(model, jt, csize, rng=rng)
            v, okk = _once()
            if okk and v is not None:
                if best is None or not ok:
                    best, ok = v, True
                elif sense == 'max':
                    best = max(best, v)
                else:
                    best = min(best, v)
            if not _is_suspicious(best, ok, sense):
                break
    _record(value=(best if ok else None), primal=best, dual=None, gap=None,
            termination=("local" if ok else "failed"), certified=False)
    return best if ok else None


def _h3_report(**kw):
    """Per-instance record of how the LMC assertions were discharged. Keys:
    n (assertions), entailed (separated in G*, nothing imposed), hosted
    (imposed exactly, possibly after reduction), relaxed (unhostable even
    after reduction: only hostable sub-assertions imposed -- sound, outer),
    dropped (relaxed with nothing hostable at all), widened (extra cliques
    added to make reduced assertions hostable), holds (H3 holds: relaxed == 0,
    so the cluster program is EXACT, given H1/H2/H4 and a certified solve),
    mode ("ordered" or "min-fill"); in ordered mode ``implied`` counts the
    assertions implied by the ordered rows and ``ordered_rows`` the rows
    imposed."""
    return dict(kw)


def _relaxed_subassertions(jt, X, Y_in, Z):
    """Sound relaxation for a reduced assertion X perp Y_in | Z with no host:
    impose every hostable X perp {y} | Z, y in Y_in (decomposition). Each is
    implied by the original, so imposing it never cuts a joint of D_LCN; the
    cluster program stays an OUTER bound, merely not guaranteed tight."""
    from lcn.core.independencies import IndependenceAssertion
    out = []
    for y in Y_in:
        sc = sorted(set(X) | {y} | set(Z))
        h = jt.host(sc)
        if h is not None:
            out.append((h, IndependenceAssertion(list(X), [y], list(Z))))
    return out


def _lmc_entails(lcn, x_atoms, rest_atoms, cond_atoms):
    """True iff (x_atoms _||_ rest_atoms | cond_atoms) follows from a SINGLE LMC
    assertion of the LCN by symmetry + decomposition (same conditioning set).
    Deliberately conservative: pairwise covers are not combined (pairwise
    independence does not imply joint independence)."""
    if not rest_atoms:
        return True
    for a in lcn.independencies.get_assertions():
        E1, E2, E3 = set(a.event1), set(a.event2), set(a.event3)
        if E3 != set(cond_atoms):
            continue
        if set(x_atoms) <= E1 and set(rest_atoms) <= E2:
            return True
        if set(x_atoms) <= E2 and set(rest_atoms) <= E1:
            return True
    return False


def _triage_ordered(cnv, jt):
    """
    LMC discharge for the ORDERED-Markov tree (``jt.order == 'topological'``).

    For every node x with elimination clique {x} u N_x (N_x = earlier nodes,
    pa(x) subset N_x), impose the single assertion
        (atoms(x) _||_ atoms(N_x - pa(x)) | atoms(pa(x)))
    on x's cluster -- after checking it is ENTAILED by the LCN's own LMC
    (soundness: imposing it never removes a joint of D_LCN).

    Exactness: a separator-consistent cluster family reconstructs (Lemma RIP,
    leaf by leaf along the elimination) to p = prod_x p(x | N_x); the imposed
    equalities turn this into prod_x p(x | pa(x)), the node-DAG factorization,
    which implies every d-separation of the DAG -- in particular every LMC
    assertion of the LCN (verified per assertion below). No positivity is used.

    Returns (lmc_by_host, ok). ok=False (=> caller uses the min-fill triage)
    if some ordered assertion is not entailed or some LCN assertion is not
    d-separated in the node DAG.
    """
    import networkx as nx
    from lcn.core.independencies import IndependenceAssertion
    lcn = cnv.lcn
    na = cnv.cn.node_atoms
    parents = _node_parents(cnv)
    lmc_by_host: Dict[str, List] = {}
    for x in jt.elim_order:
        clique = jt.node_clique.get(x, {x}) - {x}
        pa = parents.get(x, set())
        if not pa <= clique:
            return {}, False
        rest = clique - pa
        if not rest:
            continue
        xa = sorted(na[x])
        ra = sorted({a for v in rest for a in na[v]})
        za = sorted({a for v in pa for a in na[v]})
        if not _lmc_entails(lcn, xa, ra, za):
            return {}, False
        host = jt.host(sorted(set(xa) | set(ra) | set(za)))
        if host is None:
            return {}, False
        # Multi-atom compound child: impose the joint statement atom-block-wise
        # via the singleton-X encoder by chaining x_i _||_ rest | pa, x_<i
        # (chain rule for CI: equivalent to the block statement).
        for i, xi in enumerate(xa):
            lmc_by_host.setdefault(host, []).append(
                IndependenceAssertion([xi], ra, za + xa[:i]))

    # Safety net: every LCN LMC assertion must be d-separated in the node DAG.
    G = nx.DiGraph()
    for v, ps in parents.items():
        G.add_node(v)
        for p in ps:
            G.add_edge(p, v)
    node_of = {a: v for v, ats in na.items() for a in ats}
    for a in lcn.independencies.get_assertions():
        X = {node_of[t] for t in a.event1}
        Y = {node_of[t] for t in a.event2}
        Z = {node_of[t] for t in a.event3}
        if (X & Y) or (X & Z) or (Y & Z):
            return {}, False   # assertion splits a compound node: not covered
        # Each of X, Y, Z must be a union of WHOLE nodes: conditioning on one
        # atom of a compound node is not conditioning on the node, so the
        # node-level d-separation below would not certify it.
        for atoms_, nodes_ in ((a.event1, X), (a.event2, Y), (a.event3, Z)):
            if set(atoms_) != {t for v in nodes_ for t in na[v]}:
                return {}, False
        if not nx.is_d_separator(G, X, Y, Z):
            return {}, False
    return lmc_by_host, True


def _triage_lmc(cnv, query, evidence, extra_node_sets, jt, max_cluster_atoms):
    """
    Discharge every LMC assertion X perp Y | Z of the LCN against the junction
    tree, returning (jt, lmc_by_host); the (possibly rebuilt) tree carries an
    ``h3`` report (see _h3_report).

    1. Reduce: Y_in = jt.reduce_assertion(X, Y, Z). This is an EQUIVALENCE for
       every joint the cluster program can reconstruct (they all factorize over
       G*), so nothing is lost. Y_in empty => entailed, impose nothing.
    2. Host: if X u Y_in u Z fits one cluster, impose the reduced assertion
       there exactly.
    3. Widen: otherwise add X u Y_in u Z (as a node set) as an extra clique and
       rebuild the tree; repeat to a fixed point (widening changes G*, which can
       change the reductions) while the largest cluster stays within
       ``max_cluster_atoms``.
    4. Relax: anything still unhostable is replaced by its hostable pairwise
       sub-assertions (sound; H3 fails, bound is outer). Assertions are never
       dropped: CJT optimizes over D_LCN, not over a strong extension, so an
       unimposed assertion would widen the bound.
    """
    lcn = cnv.lcn
    node_of_atom = {a: nd for nd, ats in cnv.cn.node_atoms.items() for a in ats}
    assertions = []
    for indep in lcn.independencies.get_assertions():
        X, Y, Z = (sorted(indep.event1), sorted(indep.event2),
                   sorted(indep.event3))
        if X and (Y or Z):
            assertions.append((X, Y, Z))

    # Drop symmetric duplicates (x perp {y} | Z and y perp {x} | Z are the same
    # constraint once reduced to singletons) -- each would add identical
    # bilinear rows. Exact duplicates are dropped too.
    seen, uniq = set(), []
    for X, Y, Z in assertions:
        key = (frozenset(X), frozenset(Y), frozenset(Z))
        sym = (frozenset(Y), frozenset(X), frozenset(Z))
        if key in seen or (len(X) == 1 and sym in seen):
            continue
        seen.add(key)
        uniq.append((X, Y, Z))
    n_input = len(assertions)
    assertions = uniq

    extra = [list(s) for s in extra_node_sets]
    widened = 0
    for _round in range(len(assertions) + 1):
        need = []
        for X, Y, Z in assertions:
            Y_in = jt.reduce_assertion(X, Y, Z)
            if Y_in and jt.host(sorted(set(X) | set(Y_in) | set(Z))) is None:
                need.append(sorted({node_of_atom[a]
                                    for a in set(X) | set(Y_in) | set(Z)}))
        if not need:
            break
        trial = _JTClusters(cnv, query, evidence, extra + need)
        if trial.max_cluster_size() > max_cluster_atoms:
            break  # widening would exceed the budget -> relax the remainder
        extra = extra + need
        widened += len(need)
        jt = trial

    from lcn.core.independencies import IndependenceAssertion
    lmc_by_host: Dict[str, List] = {}
    n_ent = n_host = n_relax = n_drop = 0
    imposed = set()   # reduced (x, Y_in, Z) already imposed (dedupe after reduction)

    def _impose(h, X, Yr, Z):
        key = (frozenset(X), frozenset(Yr), frozenset(Z))
        sym = (frozenset(Yr), frozenset(X), frozenset(Z))
        if key in imposed or (len(Yr) == 1 and sym in imposed):
            return
        imposed.add(key)
        lmc_by_host.setdefault(h, []).append(
            IndependenceAssertion(list(X), list(Yr), list(Z)))

    for X, Y, Z in assertions:
        Y_in = jt.reduce_assertion(X, Y, Z)
        if not Y_in:
            n_ent += 1
            continue
        host = jt.host(sorted(set(X) | set(Y_in) | set(Z)))
        if host is not None:
            _impose(host, X, Y_in, Z)
            n_host += 1
            continue
        subs = _relaxed_subassertions(jt, X, Y_in, Z)
        n_relax += 1
        n_drop += not subs
        for h, ia in subs:
            _impose(h, sorted(ia.event1), sorted(ia.event2), sorted(ia.event3))

    jt.h3 = _h3_report(n=n_input, entailed=n_ent, hosted=n_host,
                       relaxed=n_relax, dropped=n_drop, widened=widened,
                       holds=(n_relax == 0))
    return jt, lmc_by_host


def _build_jt_and_hosts(cnv, query, evidence, max_cluster_atoms):
    """
    Build the augmented junction tree and assign every LCN sentence / LMC
    assertion (and the query+evidence) to a host cluster. Shared by
    build_and_solve_jt_nlp and print_junction_tree. Returns
    (jt, sentences_by_host, lmc_by_host, query_cluster, over_budget, fallback):
      - over_budget: True if a cluster exceeds max_cluster_atoms (cannot be made
        exact within the treewidth budget);
      - fallback: True if some constraint/query has no host (degenerate).
    """
    lcn = cnv.lcn
    if lcn.primal_graph is None:
        lcn.build_primal_graph()
    if lcn.independencies is None:
        lcn.local_markov_condition()

    # Cross-family constraints. Cross-family SENTENCES always augment the
    # elimination order (forcing their atoms into one cluster). LMC assertions
    # are NOT augmented wholesale -- most are entailed by G*-separation, and
    # augmenting with their full scopes would needlessly inflate the treewidth
    # (a Markov chain would collapse to a single 2^n cluster). Instead
    # _triage_lmc reduces each assertion exactly against G* and widens only by
    # the (much smaller) reduced scopes that are still unhostable.
    cc = CouplingConstraints.from_lcn(lcn, cnv.cn.factorization.factors)
    extra_node_sets = cc.constraint_node_sets(cnv.cn.node_atoms,
                                              kinds=("type1", "type2"))

    # Candidate 1 (query-independent build only): the ordered-Markov tree,
    # exact by construction (H3 holds) with no widening -- see _triage_ordered.
    chosen = None
    if query is None:
        jt_o = _JTClusters(cnv, None, evidence, extra_node_sets,
                           order="topological")
        if jt_o.max_cluster_size() <= max_cluster_atoms:
            lmc_o, ok = _triage_ordered(cnv, jt_o)
            if ok:
                n_lmc = len(list(lcn.independencies.get_assertions()))
                # In ordered mode no LCN assertion is imposed individually:
                # all are IMPLIED by the ordered rows via the DAG factorization
                # (counted as "implied", not "entailed" by G*-separation).
                jt_o.h3 = _h3_report(
                    n=n_lmc, entailed=0, implied=n_lmc, hosted=0, relaxed=0,
                    dropped=0, widened=0, holds=True, mode="ordered",
                    ordered_rows=sum(len(v) for v in lmc_o.values()))
                chosen = (jt_o, lmc_o)

    # Candidate 2: min-fill tree + exact G*-reduction + budget-limited widening.
    jt = _JTClusters(cnv, query, evidence, extra_node_sets)
    if jt.max_cluster_size() > max_cluster_atoms and chosen is None:
        jt.h3 = _h3_report(n=0, entailed=0, hosted=0, relaxed=0, dropped=0,
                           widened=0, holds=False, mode="min-fill")
        return jt, {}, {}, None, True, True
    if jt.max_cluster_size() <= max_cluster_atoms:
        jt_m, lmc_m = _triage_lmc(cnv, query, evidence, extra_node_sets, jt,
                                  max_cluster_atoms)
        jt_m.h3["mode"] = "min-fill"
        # Prefer an exact tree; among exact trees, the narrower one.
        if chosen is None or (
                jt_m.h3["holds"]
                and jt_m.max_cluster_size() < chosen[0].max_cluster_size()):
            chosen = (jt_m, lmc_m)
    jt, lmc_by_host = chosen

    sentences_by_host: Dict[str, List] = {}
    fallback = False

    # Sentences must be hosted (or the tree cannot represent them exactly).
    for sid, s in lcn.sentences.items():
        atoms = list(s.get_atoms().keys())
        if not atoms:
            # A constant sentence (no atoms) holds for every joint or for
            # none; it used to be skipped silently, which hid an
            # inconsistent LCN. Such sentences are not supported.
            raise ValueError(f"Sentence {sid} has no atoms; constant "
                             f"sentences are not supported by CJT.")
        host = jt.host(atoms)
        if host is None:
            fallback = True
        else:
            sentences_by_host.setdefault(host, []).append(sid)

    # Query-cluster assignment only for the single-query build. For the
    # query-independent build (query is None), each atom's host is resolved
    # later, per atom, by the all-marginals engine.
    query_cluster = None
    if query is not None:
        q_atoms = list(evidence.keys()) + [query]
        query_cluster = jt.host(q_atoms)
        if query_cluster is None:
            fallback = True

    # H1 is checked explicitly rather than assumed (the augmented build once
    # produced a forest that broke it).
    jt.rip = jt.rip_holds()

    return jt, sentences_by_host, lmc_by_host, query_cluster, False, fallback


def print_junction_tree(cnv, query, evidence=None, max_cluster_atoms=16,
                        file=None):
    """
    Print the D5 junction tree and the separator messages it propagates, without
    solving the NLP. ``query`` may be a node name (the single-query tree, where
    the query+evidence are forced into one cluster) or ``None`` (the
    query-independent tree the all-marginals CredalJT engine uses).

    Shows each cluster (atom scope, number of states, and the sentences / LMC
    assertions it hosts), the tree as an indented forest, and the per-edge
    separator-consistency messages (q[child] |S == q[parent] |S, where S is the
    separator). These equalities are the D5 analogue of junction-tree messages:
    the message a cluster passes to a neighbour is its marginal over the shared
    separator.

    Returns the same `jt` object so callers can inspect it programmatically.
    """
    import sys
    out = file if file is not None else sys.stdout
    evidence = evidence or {}
    jt, sentences_by_host, lmc_by_host, query_cluster, over_budget, fallback = \
        _build_jt_and_hosts(cnv, query, evidence, max_cluster_atoms)
    ev_str = (' | ' + str(evidence)) if evidence else ''
    if query is None:
        header = f"=== Junction tree (all marginals{ev_str}) ==="
    else:
        header = f"=== Junction tree for P({query}{ev_str}) ==="
    print(header, file=out)
    if over_budget:
        print(f"max cluster {jt.max_cluster_size()} atoms exceeds budget "
              f"{max_cluster_atoms}; D5 would fall back to ExactInference.",
              file=out)
    elif fallback:
        print("a constraint/query has no host cluster (degenerate structure); "
              "D5 would fall back to ExactInference.", file=out)
    print(jt.describe(sentences_by_host, lmc_by_host, query_cluster), file=out)
    return jt


def build_and_solve_jt_nlp(cnv, query, evidence=None, solver="ipopt",
                           max_cluster_atoms=16, time_limit=None,
                           gap_tol=0.0, verbosity=1):
    """
    Build and solve the D5 junction-tree exact NLP for P(query=1 | evidence).

    Returns (lo, hi, info) where info = {"exact": bool, "max_cluster_atoms": int,
    "fallback": bool} and, when the cluster NLP is actually built (not the budget/
    unhostable fallback), "nlp_stats": the size of that NLP (variable/constraint
    counts by category; see _model_stats). exact/fallback flag whether every
    constraint found a host cluster within the budget; the caller may fall back to
    ExactInference when fallback is True.
    """
    evidence = evidence or {}
    lcn = cnv.lcn

    jt, sentences_by_host, lmc_by_host, query_cluster, over_budget, fallback = \
        _build_jt_and_hosts(cnv, query, evidence, max_cluster_atoms)

    # Bail out (caller falls back to ExactInference) if any cluster blew past
    # the treewidth budget after augmentation.
    if over_budget:
        info = {"exact": False, "fallback": True,
                "max_cluster_atoms": jt.max_cluster_size()}
        if verbosity > 0:
            print(f"[CJT] max cluster {jt.max_cluster_size()} atoms exceeds "
                  f"budget {max_cluster_atoms}; falling back to ExactInference.")
        return None, None, info

    if verbosity > 1:
        print(jt.describe(sentences_by_host, lmc_by_host, query_cluster))

    h3 = dict(getattr(jt, "h3", {}) or {})
    info = {"exact": ((not fallback) and bool(h3.get("holds", False))
                      and bool(getattr(jt, "rip", False))),
            "fallback": fallback, "h3": h3,
            "max_cluster_atoms": jt.max_cluster_size()}
    if fallback:
        if verbosity > 0:
            print("[CJT] a constraint/query has no host cluster; "
                  "caller should fall back to ExactInference.")
        return None, None, info

    if verbosity > 0:
        print(f"[CJT] junction tree: {len(jt.cluster_ids)} clusters, "
              f"max cluster {jt.max_cluster_size()} atoms "
              f"(n={len(lcn.atoms)} atoms total)")

    logging.getLogger('pyomo.core').setLevel(logging.ERROR)
    model, obj_expr, csize = _build_model(
        jt, cnv, query, evidence, sentences_by_host, lmc_by_host,
        query_cluster, solver)
    info["nlp_stats"] = _model_stats(
        jt, csize, sentences_by_host, lmc_by_host, cnv)
    if verbosity > 0:
        print("[CJT] constraint NLP size:")
        print(_format_model_stats(info["nlp_stats"]))
    st = []
    lo = _solve_sense(model, jt, obj_expr, csize, 'min', solver, time_limit,
                      gap_tol, verbosity, status_out=st)
    hi = _solve_sense(model, jt, obj_expr, csize, 'max', solver, time_limit,
                      gap_tol, verbosity, status_out=st)
    info["solve_status"] = st
    info["certified"] = len(st) == 2 and all(x["certified"] for x in st)
    info["exact"] = info["exact"] and info["certified"]
    lo = 0.0 if lo is None else max(0.0, lo)
    hi = 1.0 if hi is None else min(1.0, hi)
    return lo, hi, info


class CredalJT:
    """
    Junction-tree exact marginal inference for chain-graph LCNs (scheme D5).

    Builds ONE junction tree and ONE constraint NLP (cluster-marginal variables
    + per-cluster simplex + separator-consistency equalities + the hosted
    sentence rows + the independence rows chosen by the LMC triage), then
    computes the posterior bounds of EVERY non-evidence singleton atom by
    swapping only the objective P(atom=1 | evidence) per atom and solving
    min/max. The constraint structure is query-independent, so the tree and
    model are built once; only the objective and the two solves are per atom.

    Exactness: under H1-H4 (running intersection, every sentence hosted, the
    LMC discharged with nothing relaxed, query and evidence co-hosted) the global
    optimum of the cluster program is the exact LCN bound; ``d5_exact`` records
    whether the structure satisfies H1-H3, ``certified`` whether every endpoint
    was certified by SCIP, and ``exact`` both. When the query and the evidence
    share no cluster, a tree augmented by their node set is built for that atom.
    Only when a tree exceeds the cluster budget (or a sentence cannot be hosted)
    does the engine fall back to the direct 2^n formulation, solved locally and
    therefore NOT certified (``atom_status[...]["route"] == "direct-local"``).
    """

    def __init__(self, cnv):
        # CredalJT's NLP is formulated from the LCN sentences and LMC equalities,
        # so it needs only the chain-graph STRUCTURE carried by the CredalNetwork
        # factors (child/parents/scope) -- NOT the per-family interval bounds and
        # NOT the enumerated extreme points. cnv may therefore be a structure-only
        # build (solve_families=False, which also implies enumerate_vertices=
        # False): the factor lobo/upbo are placeholders and extreme_points/
        # credal_net are None. The assert below only checks the factors exist.
        assert cnv.cn is not None and cnv.cn.factors, \
            "CredalNetwork must be built before passing to CredalJT."
        self.cnv = cnv

    def run(self, evidence: dict = {}, solver: str = "scip",
            max_cluster_atoms: int = 16, time_limit: float = None,
            gap_tol: float = 0.0, verbosity: int = 1,
            cert_tol: float = _CERT_ABS_TOL):
        """
        Compute exact lower/upper bounds on the posterior marginal of every
        non-evidence singleton atom.

        Returns {name -> (lower_bounds, upper_bounds)} (singleton atoms as the
        2-vector [P(=0), P(=1)]). Also sets self.singleton_marginals
        ({atom -> (lo, hi)} for P(atom=1)), self.d5_exact, self.induced_width,
        the running-time stats (build_time / elimination_time / total_time),
        self.degenerate (all marginals vacuous [0,1]), and self.nlp_stats -- the
        size of the shared constraint NLP (variable/constraint counts by
        category; see _model_stats). self.nlp_stats is None on the exact-fallback
        path (over budget / unhostable constraint), where no cluster NLP is built.

        Args:
            evidence: {atom -> value} for observed atoms (skipped as queries).
            solver: "scip" (default, certified global -- needed for the
                nonconvex cluster NLP to be exact) or "ipopt" (local).
            max_cluster_atoms / time_limit / gap_tol: JT-NLP budget and solver
                limits (see build_and_solve_jt_nlp).
            cert_tol: absolute primal-dual tolerance (probability units) under
                which a SCIP endpoint counts as certified. Uncertified endpoints
                are reported as SCIP's dual bound, so they are always sound.
            verbosity: 0 silent, 1 summary, 2 also prints the junction tree
                with the detailed separator-consistency messages, and the
                solver's own progress.
        """
        evidence = evidence or {}
        evidence_set = set(evidence.keys())
        node_atoms = self.cnv.cn.node_atoms
        atoms = sorted({a for ats in node_atoms.values() for a in ats}
                       - evidence_set)

        t0 = time.perf_counter()
        # Build the query-independent JT and assign constraints to host clusters
        # ONCE (query=None => no per-query augmentation).
        jt, sentences_by_host, lmc_by_host, _, over_budget, fallback = \
            _build_jt_and_hosts(self.cnv, None, evidence, max_cluster_atoms)
        # Structural exactness (Theorem: H1 = running intersection, checked by
        # jt.rip_holds(); H2 = every sentence
        # hosted, H3 = every LMC entailed or hosted after reduction/widening).
        # H4 is per atom and checked below. d5_exact is True only if the
        # STRUCTURE guarantees exactness; ``certified`` additionally requires
        # every solve to be a certified global optimum.
        self.h3 = dict(getattr(jt, "h3", {}) or {})
        self.rip = bool(getattr(jt, "rip", False))
        self.d5_exact = (not (over_budget or fallback)
                         and bool(self.h3.get("holds", False)) and self.rip)
        self.induced_width = jt.max_cluster_size()
        self.solve_status = []      # one dict per min/max solve
        self.atom_status = {}       # atom -> {"exact","certified","route"}

        if verbosity > 0:
            print(f"[CredalJT] Computing all marginals (D5, solver={solver}, "
                  f"evidence={evidence})")
            print(f"[CredalJT] junction tree: {len(jt.cluster_ids)} clusters, "
                  f"max cluster {jt.max_cluster_size()} atoms "
                  f"(n={len(self.cnv.lcn.atoms)} atoms total)")
        if verbosity > 1:
            print(jt.describe_detailed(sentences_by_host, lmc_by_host))

        self.singleton_marginals = {}
        self.nlp_stats = None
        if over_budget or fallback:
            # Cannot host every constraint within budget -> per-atom exact fallback.
            self._all_exact_fallback(atoms, evidence, verbosity)
        else:
            logging.getLogger('pyomo.core').setLevel(logging.ERROR)
            model, csize = _build_constraint_model(
                jt, self.cnv, sentences_by_host, lmc_by_host,
                evidence=evidence)
            # Record the size of the shared (query-independent) constraint NLP.
            self.nlp_stats = _model_stats(
                jt, csize, sentences_by_host, lmc_by_host, self.cnv)
            if verbosity > 0:
                print("[CredalJT] Constraint NLP size (shared across all "
                      "marginals; per-atom objective adds 1 ratio var/constraint "
                      "under evidence):")
                print(_format_model_stats(self.nlp_stats))
            for atom in atoms:
                host = jt.host([atom] + list(evidence.keys()))
                if host is None:
                    # Atom + evidence not co-hosted on the shared tree (H4
                    # fails there): build a tree augmented by the atom+evidence
                    # node set for this atom alone (H4 then holds by
                    # construction); fall back to the direct 2^n formulation
                    # only if that tree is over budget or a sentence is unhosted.
                    lo, hi, st, ex = self._augmented_atom(
                        atom, evidence, solver, max_cluster_atoms, time_limit,
                        gap_tol, cert_tol, verbosity)
                    if st is None:
                        lo, hi = self._exact_atom(atom, evidence)
                        self.atom_status[atom] = {"exact": False,
                                                  "certified": False,
                                                  "structural": False,
                                                  "route": "direct-local"}
                    else:
                        for s_ in st:
                            s_["atom"] = atom
                        self.solve_status.extend(st)
                        cert = all(s_["certified"] for s_ in st) and len(st) == 2
                        self.atom_status[atom] = {
                            "exact": bool(ex and cert), "certified": bool(cert),
                            "structural": bool(ex), "route": "augmented"}
                else:
                    _clear_atom_objective(model)
                    obj = _atom_objective(model, jt, csize, atom, host,
                                          evidence, solver)
                    if verbosity > 1:
                        print(f"  [CJT] atom {atom} (host cluster {host}):")
                    st = []
                    lo = _solve_sense(model, jt, obj, csize, 'min', solver,
                                      time_limit, gap_tol, verbosity,
                                      status_out=st, cert_tol=cert_tol)
                    if verbosity > 1:
                        print(f"    min P({atom}=1) = "
                              f"{0.0 if lo is None else lo:.6f}; realized "
                              f"separator messages at the minimizer:")
                        _print_realized_messages(model, jt, csize, prefix="  ")
                    hi = _solve_sense(model, jt, obj, csize, 'max', solver,
                                      time_limit, gap_tol, verbosity,
                                      status_out=st, cert_tol=cert_tol)
                    if verbosity > 1:
                        print(f"    max P({atom}=1) = "
                              f"{1.0 if hi is None else hi:.6f}; realized "
                              f"separator messages at the maximizer:")
                        _print_realized_messages(model, jt, csize, prefix="  ")
                    lo = 0.0 if lo is None else max(0.0, lo)
                    hi = 1.0 if hi is None else min(1.0, hi)
                    for s in st:
                        s["atom"] = atom
                    self.solve_status.extend(st)
                    cert = all(s["certified"] for s in st) and len(st) == 2
                    self.atom_status[atom] = {
                        "exact": bool(self.d5_exact and cert),
                        "certified": bool(cert),
                        "structural": bool(self.d5_exact), "route": "cluster"}
                self.singleton_marginals[atom] = (lo, hi)
                if verbosity > 1:
                    print(f"  [CJT] P({atom}=1 | {evidence}) = "
                          f"[{lo:.6f}, {hi:.6f}]")

        # Instance-level guarantees.
        self.certified = bool(self.atom_status) and all(
            s["certified"] for s in self.atom_status.values())
        # Exact iff every atom is exact on its own route: an augmented atom's
        # tree has its own H3, which the shared tree's d5_exact does not cover.
        self.exact = bool(self.d5_exact and self.certified) and all(
            s["exact"] for s in self.atom_status.values())

        self.elimination_time = time.perf_counter() - t0
        self.build_time = float(getattr(self.cnv, "build_time", None) or 0.0)
        self.total_time = self.build_time + self.elimination_time

        results = {atom: (np.array([1.0 - hi, lo]), np.array([1.0 - lo, hi]))
                   for atom, (lo, hi) in self.singleton_marginals.items()}
        self.marginals = dict(results)

        # Flag a degenerate (all-vacuous) result.
        if not self.singleton_marginals:
            self.degenerate = None
        else:
            n_vac = sum(1 for (lo, hi) in self.singleton_marginals.values()
                        if _is_vacuous(lo, hi))
            self.degenerate = (n_vac == len(self.singleton_marginals))

        if verbosity > 0:
            print("[CredalJT] Singleton marginals P(atom=1):")
            for atom in sorted(self.singleton_marginals):
                lo, hi = self.singleton_marginals[atom]
                print(f"  P({atom}=1): [{lo:.6f}, {hi:.6f}]")
            print(f"[CredalJT] structurally exact (H1-H3)={self.d5_exact}, "
                  f"all solves certified={self.certified}, "
                  f"EXACT={self.exact}, max cluster={self.induced_width} atoms")
            print(f"[CredalJT] LMC triage: {self.h3}")
            print("[CredalJT] Running times (seconds):")
            print(f"  build time:    {self.build_time:.4f}")
            print(f"  inference time:{self.elimination_time:.4f}")
            print(f"  total time:    {self.total_time:.4f}")
            if self.degenerate:
                print("[CredalJT] WARNING: the solution is DEGENERATE -- every "
                      "singleton marginal is the vacuous [0, 1] (uninformative; "
                      "the LCN/evidence is likely inconsistent).")
        return results

    def _augmented_atom(self, atom, evidence, solver, max_cluster_atoms,
                        time_limit, gap_tol, cert_tol, verbosity):
        """Bounds on P(atom | evidence) from a junction tree augmented by the
        atom+evidence node set (so H4 holds). Returns (lo, hi, solve_status,
        structurally_exact); solve_status is None if the augmented tree is over
        budget or leaves a sentence unhosted (caller falls back)."""
        jt, sbh, lbh, qc, over, fb = _build_jt_and_hosts(
            self.cnv, atom, evidence, max_cluster_atoms)
        if over or fb or qc is None:
            return None, None, None, False
        model, csize = _build_constraint_model(jt, self.cnv, sbh, lbh,
                                               evidence=evidence)
        obj = _atom_objective(model, jt, csize, atom, qc, evidence, solver)
        st = []
        lo = _solve_sense(model, jt, obj, csize, 'min', solver, time_limit,
                          gap_tol, verbosity, status_out=st, cert_tol=cert_tol)
        hi = _solve_sense(model, jt, obj, csize, 'max', solver, time_limit,
                          gap_tol, verbosity, status_out=st, cert_tol=cert_tol)
        lo = 0.0 if lo is None else max(0.0, lo)
        hi = 1.0 if hi is None else min(1.0, hi)
        return lo, hi, st, (bool(getattr(jt, "h3", {}).get("holds", False))
                            and bool(getattr(jt, "rip", False)))

    def _exact_atom(self, atom, evidence):
        """Exact P(atom=1 | evidence) bound via the full-joint ExactInference."""
        from lcn.inference.marginal.exact import ExactInference
        ei = ExactInference(self.cnv.lcn)
        return ei.run_query(atom, evidence=evidence, solver="ipopt",
                            verbosity=0)

    def _all_exact_fallback(self, atoms, evidence, verbosity):
        if verbosity > 0:
            print("[CredalJT] cluster budget exceeded / unhosted constraint; "
                  "falling back to ExactInference per atom.")
        for atom in atoms:
            self.singleton_marginals[atom] = self._exact_atom(atom, evidence)
            self.atom_status[atom] = {"exact": False, "certified": False,
                                      "structural": False,
                                      "route": "direct-local"}


if __name__ == "__main__":

    # Load the LCN
    # file_name = "examples/chain.lcn"
    # file_name = "benchmarks/tree_large_fr/tree_fr_n10_1.lcn"
    file_name = "benchmarks/real/alarm_consistent.lcn"
    lcn_model = LCN()
    lcn_model.from_lcn(file_name=file_name)
    lcn_model.summary()
    print(lcn_model)

    verbosity = 1

    # Build the credal network for scheme D5. CredalJT formulates its NLP from
    # the LCN sentences and LMC equalities, so it needs only the chain-graph
    # STRUCTURE -- not the per-family interval bounds and not the enumerated
    # extreme points. Build with solve_families=False (skip the expensive
    # per-family min/max solves; also forces enumerate_vertices=False) so the
    # build time is just the symbolic factorization.
    cnv = CredalNetworkVertices.from_lcn(
        lcn_model,
        method="linear",
        merge_budget=1,
        solver="ipopt",
        solve_families=False,
        verbosity=verbosity,
    )

    # At verbosity 2, show the D5 junction tree and the separator messages it
    # propagates (one tree serves all marginals -- query-independent).
    if verbosity >= 2:
        print()
        print_junction_tree(cnv, query=None, evidence={})

    # CredalJT computes the EXACT marginals (scheme D5) with one junction tree
    # for all atoms.
    jt = CredalJT(cnv=cnv)
    print("\n=== All marginals (CredalJT, exact D5) ===")
    jt.run(evidence={}, solver="scip", verbosity=verbosity)
    for atom in sorted(jt.singleton_marginals):
        lo, hi = jt.singleton_marginals[atom]
        print(f"  P({atom}=1) in [{lo:.6f}, {hi:.6f}]")
