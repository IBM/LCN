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

# Family-realizability of LCN sentences (paper: Definition def:realizable and
# the characterization lemma lem:invariance).
#
# A sentence s is family-realizable in family v = (ch_v, pa_v) if its atoms lie
# in ch_v u pa_v and its value is determined by the conditional table
# P(ch_v | pa_v) alone, whatever the parents' marginal. For a parent
# configuration pi write phi_pi, psi_pi for the formulas restricted to pa_v =
# pi (formulas over ch_v). Then (characterization):
#
#   Type-1 P(phi):  realizable in v  iff  pa_v is empty, or phi is constant
#                   (a tautology or a contradiction) on ch_v u pa_v;
#   Type-2 P(phi | psi):  realizable in v  iff  at most one pi has psi_pi
#                   satisfiable, or the conditional value is constant: either
#                   psi_pi |= phi_pi for every such pi (value 1) or
#                   psi_pi |= !phi_pi for every such pi (value 0).
#
# The forms R1 (Type-1 on a root block) and R2 (atoms(phi) in ch_v,
# atoms(psi) in pa_v, psi pinning one full parent configuration) used by the
# problem generator are special cases. ``realizability_report`` gives the
# hypotheses of the sufficiency proposition (prop:suff) besides consistency:
# a locally feasible, factorizing, product-Markov (every product of family
# tables satisfies the LMC) LCN whose every sentence is realizable in some
# family has P_SE = D_LCN, with the exact local sets (row polytopes for
# compound children). DAG-structured LCNs are factorizing and product-Markov;
# factorization is not tested here (it can fail for compound components
# without positivity).

import itertools
from typing import Dict, List, Sequence

from lcn.core.model import SentenceType


def _models(formula, scope: Sequence[str]):
    """Truth table of ``formula`` over ``scope``: {assignment tuple: bool}."""
    atoms = set(formula.atoms.values())
    out = {}
    for vals in itertools.product((0, 1), repeat=len(scope)):
        table = dict(zip(scope, vals))
        sub = {a: table[a] for a in atoms}
        out[vals] = bool(formula.evaluate(table=sub)) if sub else None
    return out


def _restrict(models, scope, child, pi_assign):
    """{child assignment: truth} of a formula with the parents fixed."""
    idx_c = [scope.index(c) for c in child]
    res = {}
    for vals, t in models.items():
        if all(vals[scope.index(p)] == v for p, v in pi_assign.items()):
            res[tuple(vals[i] for i in idx_c)] = t
    return res


def realizable_in_family(sentence, child: Sequence[str],
                         parents: Sequence[str]) -> bool:
    """Whether ``sentence`` is family-realizable in the family
    (``child`` atoms, ``parents`` atoms) -- the exact characterization."""
    child, parents = list(child), list(parents)
    scope = child + parents
    atoms = set(sentence.get_atoms().keys())
    if not atoms <= set(scope):
        return False
    configs = [dict(zip(parents, v))
               for v in itertools.product((0, 1), repeat=len(parents))]
    phi = _models(sentence.phi_formula, scope)
    if sentence.type == SentenceType.Type1:
        if not parents:
            return True
        vals = set(phi.values())
        return len(vals) == 1                     # phi constant on the scope
    psi = _models(sentence.psi_formula, scope)
    live = []                                     # pi with psi_pi satisfiable
    for pi in configs:
        ps = _restrict(psi, scope, child, pi)
        if any(ps.values()):
            live.append(pi)
    if len(live) <= 1:
        return True
    forced = set()                                # constant value per pi
    for pi in live:
        ps = _restrict(psi, scope, child, pi)
        ph = _restrict(phi, scope, child, pi)
        sat = [c for c, t in ps.items() if t]
        if all(ph[c] for c in sat):
            forced.add(1)
        elif not any(ph[c] for c in sat):
            forced.add(0)
        else:
            return False                          # value depends on the row
    return len(forced) == 1


def is_r1_or_r2(sentence, child: Sequence[str], parents: Sequence[str]) -> bool:
    """The syntactic forms emitted by the generator: R1 (Type-1, atoms in a
    root block) or R2 (Type-2, atoms(phi) in ch_v, atoms(psi) in pa_v, psi
    satisfied by exactly one parent configuration)."""
    child, parents = list(child), list(parents)
    phi_atoms = set(sentence.phi_formula.atoms.values())
    if sentence.type == SentenceType.Type1:
        return not parents and phi_atoms <= set(child)
    psi_atoms = set(sentence.psi_formula.atoms.values())
    if not (phi_atoms <= set(child) and psi_atoms <= set(parents)):
        return False
    n = 0
    for v in itertools.product((0, 1), repeat=len(parents)):
        table = dict(zip(parents, v))
        if sentence.psi_formula.evaluate(
                table={a: table[a] for a in psi_atoms}):
            n += 1
    return n == 1


def families_of(cnv) -> List[Dict]:
    """[{node, child atoms, parent atoms}] of a compiled credal network."""
    bn = cnv.bn_min
    na = cnv.cn.node_atoms
    out = []
    for nid in bn.nodes():
        node = bn.variable(nid).name()
        par = [bn.variable(p).name() for p in sorted(bn.parents(nid))]
        out.append({"node": node, "child": list(na[node]),
                    "parents": [a for p in par for a in na[p]]})
    return out


def _family_tables_random(cnv, rng):
    """One random table per family (Dirichlet rows), as arrays over
    [node] + parents in node-state space."""
    import numpy as np
    bn = cnv.bn_min
    tabs = {}
    for nid in bn.nodes():
        node = bn.variable(nid).name()
        par = [bn.variable(p).name() for p in sorted(bn.parents(nid))]
        k = bn.variable(nid).domainSize()
        shape = (k,) + tuple(bn.variable(bn.idFromName(p)).domainSize()
                             for p in par)
        T = rng.dirichlet(np.ones(k) * 0.7, size=int(np.prod(shape[1:])))
        tabs[node] = (par, T.T.reshape(shape))
    return tabs


def lmc_factorized(lcn, cnv, trials: int = 3, max_atoms: int = 14,
                   tol: float = 1e-9):
    """Whether every LMC assertion of the LCN holds for EVERY product of
    family tables of the compiled network -- i.e. whether the free product
    of the strong extension satisfies the LMC (paper: prop:suff). True for
    every DAG-structured LCN; for compound nodes it fails exactly when the
    LMC constrains a family's table (e.g. C _|_ E | D inside a root path
    C - D - E). Decided by evaluating the assertions on random products
    (generic tables: a polynomial identity that holds on random points holds
    everywhere with probability one) when the LCN has at most ``max_atoms``
    atoms; None (unknown) above that."""
    import numpy as np
    na = cnv.cn.node_atoms
    if all(len(a) == 1 for a in na.values()):
        return True
    atoms = sorted({a for ats in na.values() for a in ats})
    if len(atoms) > max_atoms:
        return None
    if lcn.independencies is None:
        lcn.local_markov_condition()
    assertions = lcn.independencies.get_assertions()
    nodes = list(na)
    rng = np.random.default_rng(0)
    for _ in range(trials):
        tabs = _family_tables_random(cnv, rng)
        # joint over node states, then over atoms
        joint = {}
        for states in itertools.product(*[range(2 ** len(na[v]))
                                          for v in nodes]):
            st = dict(zip(nodes, states))
            pr = 1.0
            for v in nodes:
                par, T = tabs[v]
                pr *= T[(st[v],) + tuple(st[p] for p in par)]
            asg = {}
            for v in nodes:
                for i, a in enumerate(na[v]):
                    asg[a] = (st[v] >> (len(na[v]) - 1 - i)) & 1
            joint[tuple(asg[a] for a in atoms)] = pr
        P = np.zeros((2,) * len(atoms))
        for key, pr in joint.items():
            P[key] = pr
        ax = {a: i for i, a in enumerate(atoms)}

        def marg(keep):
            drop = tuple(i for i in range(len(atoms)) if atoms[i] not in keep)
            m = P.sum(axis=drop) if drop else P
            order = sorted(keep, key=lambda a: ax[a])
            return m, order
        for ia in assertions:
            X, Y, S = set(ia.event1), set(ia.event2), set(ia.event3)
            mxys, oxys = marg(X | Y | S)
            for vals in itertools.product((0, 1), repeat=len(oxys)):
                d = dict(zip(oxys, vals))
                pick = lambda vs: tuple(d[a] for a in sorted(vs, key=lambda a: ax[a]))  # noqa: E731
                ms, _ = marg(S)
                mxs, _ = marg(X | S)
                mys, _ = marg(Y | S)
                lhs = mxys[tuple(vals)] * (ms[pick(S)] if S else 1.0)
                rhs = mxs[pick(X | S)] * mys[pick(Y | S)]
                if abs(lhs - rhs) > tol:
                    return False
    return True


def realizability_report(lcn, cnv) -> Dict:
    """Per sentence the families it is realizable in; plus the other
    hypotheses of prop:suff besides consistency (DAG-structured, locally
    feasible -- the latter read off the compiled network's infeasible rows)."""
    fams = families_of(cnv)
    per = {}
    for label, s in lcn.sentences.items():
        per[label] = [f["node"] for f in fams
                      if realizable_in_family(s, f["child"], f["parents"])]
    infeasible = cnv.cn.infeasible_rows
    factorized = lmc_factorized(lcn, cnv)
    return {"realizable_in": per,
            "fully_realizable": all(per.values()),
            "dag_structured": all(len(f["child"]) == 1 for f in fams),
            # product-Markov: every LMC assertion holds for every product of
            # family tables (True if DAG-structured)
            "product_markov": factorized,
            "locally_feasible": not infeasible,
            "infeasible_rows": infeasible}
