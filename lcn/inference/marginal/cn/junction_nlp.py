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

# Junction-tree exact NLP for chain-graph LCNs -- scheme D5 of
# docs/tighter_approximation.tex.
#
# Credal VE / CTE / ApproxLP bound the STRONG EXTENSION of the credal network
# (a free product of local extreme points), which is a superset of the LCN's
# distribution set; schemes D1-D4 tighten that approximation but do not in
# general reach the exact LCN bounds. D5 computes the EXACT chain-graph marginal
# by a single optimization that never materializes the strong extension:
#
#   * Build a junction (clique) tree from the chain-graph elimination order
#     (reusing the bucket-tree construction CCTE already uses).
#   * Introduce one variable per CLUSTER, ranging over the joint distribution of
#     that cluster's atoms (2^|atoms(cluster)| simplex variables).
#   * Tie adjacent clusters with LAURITZEN separator-consistency equalities: the
#     marginal of a cluster onto a separator equals the neighbour's marginal onto
#     the same separator.
#   * Impose every LCN sentence and LMC equality on a cluster that contains its
#     atom scope (a "host" cluster).
#   * Read the query marginal P(query | evidence) off the cluster containing the
#     query and the evidence atoms.
#
# For a chain graph the clique tree has the running-intersection property, so a
# separator-consistent, constraint-satisfying family of cluster marginals
# projects exactly to the marginals of some global joint in the LCN's set, and
# conversely. Hence min/max of the query marginal over this NLP equals the exact
# ExactInference bound -- at cost Sum_c 2^|atoms(c)| (treewidth-bounded), not
# 2^n. When a constraint's atoms fit no single cluster (a cross-family constraint
# spanning cliques, or an evidence atom outside the query cluster) the offending
# clusters are merged ("widened") so the constraint has a host; if widening would
# exceed a budget, the caller falls back to ExactInference.

import logging
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
from lcn.core.model import SentenceType
from lcn.inference.utils.common import (
    make_ipopt, build_truth_table,
    lmc_constraint_groups_vec, eval_indicator, make_conjunction,
)

_N_RESTARTS = 4
_VACUOUS_TOL = 1e-6
_DEN_FLOOR = 1e-6


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
    Port of CCTE._build_bucket_tree restricted to the structural outputs D5
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


class _JTClusters:
    """
    Atom-level junction tree: clusters (atom sets), tree edges, separators.

    The clique tree is built ONCE with the min-fill elimination order augmented
    by extra "clique scopes": the query+evidence node set and the node set of
    every cross-family constraint. Forcing those node sets to co-occur (each
    becomes a clique in the interaction graph) guarantees the resulting tree has
    a single cluster covering each constraint's atoms -- so every constraint has
    a host with no fragile post-hoc tree surgery. This is the JT analogue of
    D4's elimination-order augmentation (cve.py / coupling.constraint_node_sets).
    """

    def __init__(self, cnv, query, evidence, extra_node_sets):
        """
        Build the augmented junction tree.

        ``query`` may be ``None`` (the query-INDEPENDENT tree used by the
        all-marginals CredalJT engine): then the tree is built over a plain
        min-fill order with only the cross-family constraint node sets as
        augmenting cliques. Every singleton atom is still hosted in some cluster
        (each node's own scope is a clique), so any atom's marginal can be read
        off a hosting cluster -- one tree serves all marginals. When ``query`` is
        a node name (single-query build_and_solve_jt_nlp), the query+evidence
        node set is added as an extra clique and the query node is kept
        un-eliminated so its bucket holds the query.
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
            # Augmenting clique: the query+evidence node set; keep the query node
            # un-eliminated so its bucket holds the query.
            node_of_atom = {}
            for node, atoms in node_atoms.items():
                for a in atoms:
                    node_of_atom[a] = node
            q_nodes = sorted({node_of_atom[a]
                              for a in ([query] + list(evidence.keys()))
                              if a in node_of_atom})
            if q_nodes:
                aug = [q_nodes] + aug
            exclude = {query}
        aug = [s for s in aug if s]

        scopes = node_scopes + aug
        elim_order = min_fill_order(scopes, exclude=exclude)
        parent, children, eff = _jt_from_scopes(scopes, elim_order)

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

    def _rip_implied(self, X, Y, Z):
        """
        True if the independence (X perp Y | Z) is already enforced by the
        junction tree's separators (running-intersection): there is a tree edge
        whose separator S contains Z and whose two sides carry, beyond S, all of
        X on one side and all of Y on the other. Such an assertion need not be
        imposed as an explicit equality -- any separator-consistent family of
        cluster marginals satisfies it. For a chain graph every LMC assertion is
        RIP-implied by the family-scope tree, which is what keeps the treewidth
        small.
        """
        Xs, Ys, Zs = set(X), set(Y), set(Z)
        for (c, p) in self.edges:
            S = set(self.sep[(c, p)])
            if not Zs.issubset(S):
                continue
            sideA = self._atoms_of(self._subtree(c, p)) - S
            sideB = self._atoms_of(self._subtree(p, c)) - S
            if (Xs.issubset(sideA) and Ys.issubset(sideB)) or \
               (Xs.issubset(sideB) and Ys.issubset(sideA)):
                return True
        return False

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


def _build_constraint_model(jt, cnv, sentences_by_host, lmc_by_host):
    """
    Build the constraint-only Pyomo JT model: cluster variables + per-cluster
    simplex + separator-consistency equalities + sentence rows + LMC bilinear
    rows. NO objective. This part is QUERY-INDEPENDENT, so the all-marginals
    engine builds it once and reuses it for every atom (only the objective,
    added by ``_atom_objective``, changes per atom). Returns (model, csize).
    """
    lcn = cnv.lcn
    model = ConcreteModel()

    # Cluster variables: one flat indexed Var over (cid, state).
    keys = []
    csize = {}
    for cid in jt.cluster_ids:
        n = 2 ** len(jt.atoms[cid])
        csize[cid] = n
        keys.extend((cid, i) for i in range(n))
    model.KEYS = Set(initialize=keys, dimen=2)
    model.q = Var(model.KEYS, within=NonNegativeReals, bounds=(0.0, 1.0))
    model.constr = ConstraintList()

    # Per-cluster simplex.
    for cid in jt.cluster_ids:
        model.constr.add(
            sum(model.q[(cid, i)] for i in range(csize[cid])) == 1.0)

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
                A = eval_indicator(s.phi_formula, interps)
                expr = _dotq(A, model.q, cid, n)
                model.constr.add(expr >= lo)
                model.constr.add(expr <= hi)
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
                    _, Aa, Ab, Ac = group
                    v1 = _dotq(Aa, model.q, cid, n)
                    v2 = _dotq(Ab, model.q, cid, n) * _dotq(Ac, model.q, cid, n)
                    model.constr.add(v1 - v2 == 0.0)

    return model, csize


def _atom_objective(model, jt, csize, query, query_cluster, evidence, solver):
    """
    Build (and attach to ``model``) the objective expression for
    P(query=1 | evidence), reading it off ``query_cluster`` (a cluster whose
    atoms contain the query and the evidence atoms). No evidence -> linear
    ``A_query . q``; with evidence -> the fractional ``obj_var`` ratio aux. The
    evidence-ratio aux constraint is query-specific; it is added under the fixed
    component name ``obj_var`` (the caller clears it between atoms via
    ``_clear_atom_objective`` so the constraint model can be reused).
    Returns the objective expression.
    """
    cid = query_cluster
    atoms_c = jt.atoms[cid]
    interps = [dict(zip(atoms_c, row))
               for row in build_truth_table(len(atoms_c))]
    n = csize[cid]
    Fq = make_conjunction(variables=[query], literals={query: 1})
    A_q = eval_indicator(Fq, interps)
    if evidence:
        ev_vars = list(evidence.keys())
        Fe = make_conjunction(variables=ev_vars, literals=evidence)
        E = eval_indicator(Fe, interps)
        AE = A_q * E
        AE_expr = _dotq(AE, model.q, cid, n)
        E_expr = _dotq(E, model.q, cid, n)
        model.obj_den_floor = ConstraintList()
        if solver == "scip":
            model.obj_den_floor.add(E_expr >= _DEN_FLOOR)
        model.obj_var = Var(within=NonNegativeReals, bounds=(0.0, 1.0))
        model.obj_ratio = ConstraintList()
        model.obj_ratio.add(model.obj_var * E_expr == AE_expr)
        return model.obj_var
    return _dotq(A_q, model.q, cid, n)


def _clear_atom_objective(model):
    """Remove the per-atom objective components added by _atom_objective so the
    shared constraint model can be reused for the next atom."""
    for name in ("objective", "obj_var", "obj_ratio", "obj_den_floor"):
        if hasattr(model, name):
            model.del_component(name)


def _build_model(jt, cnv, query, evidence, sentences_by_host,
                 lmc_by_host, query_cluster, solver):
    """
    Single-query convenience: build the constraint model and attach the query
    objective. Used by build_and_solve_jt_nlp. Returns (model, obj_expr, csize).
    """
    model, csize = _build_constraint_model(jt, cnv, sentences_by_host,
                                           lmc_by_host)
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
    for cid in jt.cluster_ids:
        n = csize[cid]
        if rng is None:
            start = np.full(n, 1.0 / n)
        else:
            start = rng.random(n)
            start = start / start.sum()
        for i in range(n):
            model.q[(cid, i)].value = float(start[i])


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


def _solve_sense(model, jt, obj_expr, csize, sense, solver, time_limit,
                 gap_tol, verbosity):
    """Solve one sense (min/max) with the chosen backend; return value or None."""
    if hasattr(model, 'objective'):
        model.del_component('objective')
    model.objective = Objective(
        expr=obj_expr, sense=(minimize if sense == 'min' else maximize))

    if solver == "scip":
        opt = make_scip(time_limit=(time_limit or 3600.0), gap_tol=gap_tol)
        try:
            results = opt.solve(model, load_solutions=False,
                                tee=(verbosity > 2))
            _read_gap(results)
            model.solutions.load_from(results)
            v = value(model.objective, exception=False)
            return float(v) if v is not None else None
        except Exception as ex:
            if verbosity > 1:
                print(f"[D5] scip exception: {ex}")
            return None

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
                print(f"[D5] ipopt exception: {ex}")
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
    return best if ok else None


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

    # Cross-family constraints. Only cross-family SENTENCES augment the
    # elimination order (forcing their atoms into one cluster); LMC assertions
    # are NOT used to augment -- they are structural and the junction tree's
    # running-intersection separators already enforce them, so augmenting with
    # them would needlessly inflate the treewidth (e.g. a clean Markov chain
    # would collapse to a single 2^n cluster instead of width-2).
    cc = CouplingConstraints.from_lcn(lcn, cnv.cn.factorization.factors)
    extra_node_sets = cc.constraint_node_sets(cnv.cn.node_atoms,
                                              kinds=("type1", "type2"))

    jt = _JTClusters(cnv, query, evidence, extra_node_sets)

    if jt.max_cluster_size() > max_cluster_atoms:
        return jt, {}, {}, None, True, True

    sentences_by_host: Dict[str, List] = {}
    lmc_by_host: Dict[str, List] = {}
    fallback = False

    # Sentences must be hosted (or the tree cannot represent them exactly).
    for sid, s in lcn.sentences.items():
        atoms = list(s.get_atoms().keys())
        if not atoms:
            continue
        host = jt.host(atoms)
        if host is None:
            fallback = True
        else:
            sentences_by_host.setdefault(host, []).append(sid)

    # LMC assertions: impose one only if it has a host cluster AND is not already
    # enforced by the tree's separators (running-intersection). An LMC that is
    # RIP-implied is redundant; one with no host is, for a chain-graph product,
    # inert (the strong extension satisfies every LMC by construction), so it is
    # safe to drop. This is what lets a chain stay at treewidth.
    for indep in lcn.independencies.get_assertions():
        X, Y, Z = (sorted(indep.event1), sorted(indep.event2),
                   sorted(indep.event3))
        atoms = sorted(set(X) | set(Y) | set(Z))
        if not atoms:
            continue
        if jt._rip_implied(X, Y, Z):
            continue  # enforced structurally by the separators
        host = jt.host(atoms)
        if host is not None:
            lmc_by_host.setdefault(host, []).append(indep)
        # else: non-implied, unhostable cross-family LMC -> drop (inert).

    # Query-cluster assignment only for the single-query build. For the
    # query-independent build (query is None), each atom's host is resolved
    # later, per atom, by the all-marginals engine.
    query_cluster = None
    if query is not None:
        q_atoms = list(evidence.keys()) + [query]
        query_cluster = jt.host(q_atoms)
        if query_cluster is None:
            fallback = True

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
    "fallback": bool}. exact/fallback flag whether every constraint found a host
    cluster within the budget; the caller may fall back to ExactInference when
    fallback is True.
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
            print(f"[D5] max cluster {jt.max_cluster_size()} atoms exceeds "
                  f"budget {max_cluster_atoms}; falling back to ExactInference.")
        return None, None, info

    if verbosity > 1:
        print(jt.describe(sentences_by_host, lmc_by_host, query_cluster))

    info = {"exact": not fallback, "fallback": fallback,
            "max_cluster_atoms": jt.max_cluster_size()}
    if fallback:
        if verbosity > 0:
            print("[D5] a constraint/query has no host cluster; "
                  "caller should fall back to ExactInference.")
        return None, None, info

    if verbosity > 0:
        print(f"[D5] junction tree: {len(jt.cluster_ids)} clusters, "
              f"max cluster {jt.max_cluster_size()} atoms "
              f"(n={len(lcn.atoms)} atoms total)")

    logging.getLogger('pyomo.core').setLevel(logging.ERROR)
    model, obj_expr, csize = _build_model(
        jt, cnv, query, evidence, sentences_by_host, lmc_by_host,
        query_cluster, solver)
    lo = _solve_sense(model, jt, obj_expr, csize, 'min', solver, time_limit,
                      gap_tol, verbosity)
    hi = _solve_sense(model, jt, obj_expr, csize, 'max', solver, time_limit,
                      gap_tol, verbosity)
    lo = 0.0 if lo is None else max(0.0, lo)
    hi = 1.0 if hi is None else min(1.0, hi)
    return lo, hi, info


class CredalJT:
    """
    Junction-tree exact marginal inference for chain-graph LCNs (scheme D5).

    Builds ONE junction tree and ONE constraint NLP (cluster-marginal variables
    + per-cluster simplex + Lauritzen separator-consistency equalities + all LCN
    sentence and LMC rows), then computes the exact posterior bounds of EVERY
    non-evidence singleton atom by swapping only the objective P(atom=1 |
    evidence) per atom and solving min/max. The constraint structure is
    query-independent, so unlike the old CredalVE(coupling="d5") path it does not
    rebuild the tree/model per atom -- only the (cheap) objective and the two
    NLP solves are per-atom.

    Exactness: the clique tree has the running-intersection property and every
    sentence/LMC assertion is hosted in a containing cluster, so the
    separator-consistent, constraint-satisfying cluster marginals project exactly
    to the true joint's marginals; cost is bounded by treewidth, not 2^n. When a
    cluster exceeds the treewidth budget, the engine falls back to
    ExactInference.run_query per atom (still exact, at 2^n).
    """

    def __init__(self, cnv):
        # CredalJT's NLP is formulated from the interval local credal sets, so
        # it needs the built CredalNetwork (factors) but NOT the enumerated
        # extreme points -- cnv may be a vertex-free build
        # (enumerate_vertices=False), in which case extreme_points/credal_net
        # are None.
        assert cnv.cn is not None and cnv.cn.factors, \
            "CredalNetwork must be built before passing to CredalJT."
        self.cnv = cnv

    def run(self, evidence: dict = {}, solver: str = "scip",
            max_cluster_atoms: int = 16, time_limit: float = None,
            gap_tol: float = 0.0, verbosity: int = 1):
        """
        Compute exact lower/upper bounds on the posterior marginal of every
        non-evidence singleton atom.

        Returns {name -> (lower_bounds, upper_bounds)} (singleton atoms as the
        2-vector [P(=0), P(=1)]). Also sets self.singleton_marginals
        ({atom -> (lo, hi)} for P(atom=1)), self.d5_exact, self.induced_width,
        the running-time stats (build_time / elimination_time / total_time), and
        self.degenerate (all marginals vacuous [0,1]).

        Args:
            evidence: {atom -> value} for observed atoms (skipped as queries).
            solver: "scip" (default, certified global -- needed for the
                nonconvex cluster NLP to be exact) or "ipopt" (local).
            max_cluster_atoms / time_limit / gap_tol: JT-NLP budget and solver
                limits (see build_and_solve_jt_nlp).
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
        self.d5_exact = not (over_budget or fallback)
        self.induced_width = jt.max_cluster_size()

        if verbosity > 0:
            print(f"[CredalJT] Computing all marginals (D5, solver={solver}, "
                  f"evidence={evidence})")
            print(f"[CredalJT] junction tree: {len(jt.cluster_ids)} clusters, "
                  f"max cluster {jt.max_cluster_size()} atoms "
                  f"(n={len(self.cnv.lcn.atoms)} atoms total)")
        if verbosity > 1:
            print(jt.describe_detailed(sentences_by_host, lmc_by_host))

        self.singleton_marginals = {}
        if over_budget or fallback:
            # Cannot host every constraint within budget -> per-atom exact fallback.
            self._all_exact_fallback(atoms, evidence, verbosity)
        else:
            logging.getLogger('pyomo.core').setLevel(logging.ERROR)
            model, csize = _build_constraint_model(
                jt, self.cnv, sentences_by_host, lmc_by_host)
            for atom in atoms:
                host = jt.host([atom] + list(evidence.keys()))
                if host is None:
                    # Atom + evidence not co-hosted -> exact fallback for it.
                    lo, hi = self._exact_atom(atom, evidence)
                else:
                    _clear_atom_objective(model)
                    obj = _atom_objective(model, jt, csize, atom, host,
                                          evidence, solver)
                    if verbosity > 1:
                        print(f"  [D5] atom {atom} (host cluster {host}):")
                    lo = _solve_sense(model, jt, obj, csize, 'min', solver,
                                      time_limit, gap_tol, verbosity)
                    if verbosity > 1:
                        print(f"    min P({atom}=1) = "
                              f"{0.0 if lo is None else lo:.6f}; realized "
                              f"separator messages at the minimizer:")
                        _print_realized_messages(model, jt, csize, prefix="  ")
                    hi = _solve_sense(model, jt, obj, csize, 'max', solver,
                                      time_limit, gap_tol, verbosity)
                    if verbosity > 1:
                        print(f"    max P({atom}=1) = "
                              f"{1.0 if hi is None else hi:.6f}; realized "
                              f"separator messages at the maximizer:")
                        _print_realized_messages(model, jt, csize, prefix="  ")
                    lo = 0.0 if lo is None else max(0.0, lo)
                    hi = 1.0 if hi is None else min(1.0, hi)
                self.singleton_marginals[atom] = (lo, hi)
                if verbosity > 1:
                    print(f"  [D5] P({atom}=1 | {evidence}) = "
                          f"[{lo:.6f}, {hi:.6f}]")

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
            print(f"[CredalJT] exact={self.d5_exact}, "
                  f"max cluster={self.induced_width} atoms")
            print("[CredalJT] Running times (seconds):")
            print(f"  build time:    {self.build_time:.4f}")
            print(f"  inference time:{self.elimination_time:.4f}")
            print(f"  total time:    {self.total_time:.4f}")
            if self.degenerate:
                print("[CredalJT] WARNING: the solution is DEGENERATE -- every "
                      "singleton marginal is the vacuous [0, 1] (uninformative; "
                      "the LCN/evidence is likely inconsistent).")
        return results

    def _exact_atom(self, atom, evidence):
        """Exact P(atom=1 | evidence) bound via the full-joint ExactInference."""
        from lcn.inference.marginal.exact import ExactInference
        ei = ExactInference(self.cnv.lcn)
        return ei.run_query(atom, evidence=evidence, solver="local",
                            verbosity=0)

    def _all_exact_fallback(self, atoms, evidence, verbosity):
        if verbosity > 0:
            print("[CredalJT] cluster budget exceeded / unhosted constraint; "
                  "falling back to ExactInference per atom.")
        for atom in atoms:
            self.singleton_marginals[atom] = self._exact_atom(atom, evidence)
