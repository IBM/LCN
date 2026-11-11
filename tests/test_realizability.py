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

# Family-realizability (paper Sec. 4.2 and its appendix):
#   * the exact characterization (lem:invariance) against brute-force
#     P(pa)-invariance on random families and sentences;
#   * R1/R2 are sufficient forms, not necessary ones;
#   * the counterexamples that make the DAG-structured hypothesis of
#     prop:suff necessary for the compiled (interval-box) credal network;
#   * prop:suff on the shipped DAG-structured family-realizable benchmarks:
#     CVE (exact for the strong extension) equals certified exact inference.

import contextlib
import io
import itertools
import os
import random

import numpy as np
import pytest

from lcn.core.model import Sentence
from lcn.inference.marginal.cn.cjt import CredalJT
from lcn.inference.marginal.cn.cve import CredalVE
from lcn.inference.marginal.cn.realizability import (
    is_r1_or_r2, realizability_report, realizable_in_family)
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.exact import ExactInference
from test_cve import _cnv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


# ----------------------------------------------------------------------
# The characterization vs brute-force invariance
# ----------------------------------------------------------------------

def _value(sent, child, parents, table, w):
    """g_s at the joint w(pi) * table(c | pi), or None if P(psi) = 0."""
    scope = child + parents
    num = den = 0.0
    for pv in itertools.product((0, 1), repeat=len(parents)):
        for cv in itertools.product((0, 1), repeat=len(child)):
            asg = dict(zip(scope, cv + pv))
            pr = w[pv] * table[pv][cv]
            phi = sent.phi_formula.evaluate(
                table={a: asg[a] for a in sent.phi_formula.atoms.values()})
            if sent.psi_formula is None:
                num += pr * phi
                den += pr
            else:
                psi = sent.psi_formula.evaluate(
                    table={a: asg[a] for a in sent.psi_formula.atoms.values()})
                num += pr * (phi and psi)
                den += pr * psi
    return None if den <= 1e-12 else num / den


def _invariant(sent, child, parents, rng, trials=40):
    """Numerically: for random tables, does g_s depend on the parents'
    marginal? (True = invariant on every trial.)"""
    pcs = list(itertools.product((0, 1), repeat=len(parents)))
    ccs = list(itertools.product((0, 1), repeat=len(child)))
    for _ in range(trials):
        # include deterministic rows, which expose the forced cases
        table = {}
        for pv in pcs:
            r = rng.dirichlet(np.ones(len(ccs)) * 0.5)
            if rng.random() < 0.3:
                r = np.eye(len(ccs))[rng.integers(len(ccs))]
            table[pv] = dict(zip(ccs, r))
        vals = []
        for _ in range(6):
            w = dict(zip(pcs, rng.dirichlet(np.ones(len(pcs)) * 0.5)))
            v = _value(sent, child, parents, table, w)
            if v is not None:
                vals.append(v)
        if vals and max(vals) - min(vals) > 1e-9:
            return False
    return True


_ATOMS = {"child": [["C"], ["C", "D"]],
          "parents": [[], ["U"], ["U", "W"]]}
_LITS = ["C", "!C", "D", "!D", "U", "!U", "W", "!W"]


def _random_formula(rng, atoms, depth=2):
    lits = [x for x in _LITS if x.strip("!") in atoms]
    if depth == 0 or rng.random() < 0.4:
        return rng.choice(lits)
    op = rng.choice(["and", "or", "xor"])
    return (f"({_random_formula(rng, atoms, depth - 1)} {op} "
            f"{_random_formula(rng, atoms, depth - 1)})")


def test_characterization_matches_brute_force_invariance():
    rng = np.random.default_rng(0)
    prng = random.Random(0)
    n = 0
    for child in _ATOMS["child"]:
        for parents in _ATOMS["parents"]:
            atoms = child + parents
            for k in range(25):
                phi = _random_formula(prng, atoms)
                psi = None if k % 3 == 0 else _random_formula(prng, atoms)
                s = Sentence(label=f"s{k}", phi=phi, psi=psi)
                exact = realizable_in_family(s, child, parents)
                assert exact == _invariant(s, child, parents, rng), \
                    (child, parents, phi, psi, exact)
                n += 1
    assert n == 150


def test_r1_r2_are_sufficient_not_necessary():
    # R1 and R2 are realizable ...
    assert realizable_in_family(Sentence("a", "C"), ["C", "D"], [])
    assert is_r1_or_r2(Sentence("a", "C"), ["C", "D"], [])
    r2 = Sentence("b", "C", "U and !W")
    assert is_r1_or_r2(r2, ["C"], ["U", "W"])
    assert realizable_in_family(r2, ["C"], ["U", "W"])
    # ... but these are realizable and neither R1 nor R2:
    for s, ch, pa in [(Sentence("c", "C", "D"), ["C", "D"], []),       # psi on child
                      (Sentence("d", "C", "D and U"), ["C", "D"], ["U"]),
                      (Sentence("e", "C and U", "U"), ["C"], ["U"])]:  # parent in phi
        assert realizable_in_family(s, ch, pa) and not is_r1_or_r2(s, ch, pa)
    # mixture of two parent configurations: not realizable
    assert not realizable_in_family(Sentence("f", "C", "U"), ["C"], ["U", "W"])
    # Type-1 on a non-root family: not realizable unless constant
    assert not realizable_in_family(Sentence("g", "C"), ["C"], ["U"])
    assert realizable_in_family(Sentence("h", "C or !C"), ["C"], ["U"])


# ----------------------------------------------------------------------
# Counterexamples: the DAG-structured hypothesis is necessary
# ----------------------------------------------------------------------

def test_box_counterexample_compound_family():
    # Fully realizable (all R1/R2), compound child, complete family graph.
    # With the interval box of each row the strong-extension bounds are
    # strictly wider than the LCN bounds; with the exact row polytopes the
    # compiler now stores, they are equal.
    path = os.path.join(_ROOT, "examples", "fr_counter_box.lcn")
    cnv = _cnv(path)
    rep = realizability_report(cnv.lcn, cnv)
    assert rep["fully_realizable"] and not rep["dag_structured"]
    assert rep["product_markov"]
    ex = _quiet(ExactInference(lcn=cnv.lcn).run, evidence={}, solver="scip",
                verbosity=0, time_limit=60, progress_bar=False)
    exact_rows = _quiet(CredalVE(cnv).run, evidence={}, verbosity=0)
    assert exact_rows["C"][0][1] == pytest.approx(ex["C"][0][1], abs=1e-4)
    assert exact_rows["C"][1][1] == pytest.approx(ex["C"][1][1], abs=1e-4)
    # the box: drop the exact rows and re-enumerate the interval boxes
    cnv.cn.row_vertices = {}
    box = CredalNetworkVertices(cnv.cn)
    _quiet(box.build, verbosity=0)
    boxed = _quiet(CredalVE(box).run, evidence={}, verbosity=0)
    assert boxed["C"][0][1] < ex["C"][0][1] - 0.2      # 0.03 vs 0.32
    assert boxed["C"][1][1] > ex["C"][1][1] + 0.2      # 0.82 vs 0.58


def test_lmc_counterexample_root_path_component():
    # Fully realizable (R1 only) root component C - D - E: the LMC requires
    # C _|_ E | D inside the single row, which the compiled vertices violate.
    path = os.path.join(_ROOT, "examples", "fr_counter_lmc_path.lcn")
    cnv = _cnv(path)
    rep = realizability_report(cnv.lcn, cnv)
    assert rep["fully_realizable"] and not rep["dag_structured"]
    assert not rep["product_markov"]
    assert [str(a) for a in cnv.lcn.independencies.get_assertions()] == \
        ["(C ⟂ E | D)"]
    node = next(k for k in cnv.cn.node_atoms if "-" in k)
    V = np.array(next(iter(cnv.extreme_points[node].values())))

    def gap(p):
        p = p.reshape(2, 2, 2)
        g = 0.0
        for d in (0, 1):
            m = p[:, d, :]
            if m.sum() > 1e-12:
                j = m / m.sum()
                g = max(g, np.abs(j - np.outer(j.sum(1), j.sum(0))).max())
        return g
    assert max(gap(v) for v in V) > 0.1


# ----------------------------------------------------------------------
# prop:suff on the shipped DAG-structured family-realizable sets
# ----------------------------------------------------------------------

_SHIPPED = [os.path.join(_ROOT, "benchmarks", d, f) for d, f in [
    ("tree_small_fr", "tree_fr_n5_1.lcn"),
    ("polytree_small_fr", "polytree_fr_n5_3.lcn"),
    ("dag_small_fr", "dag_fr_tw3_n5_1.lcn"),
    ("ktree_small_fr", "ktree_fr_k3_n5_1.lcn")]]


# (path, evidence): evidence only on the singly connected instances, where
# the certified CJT solves stay fast.
_SHIPPED_CASES = [(p, {}) for p in _SHIPPED] + \
    [(p, {"x0": 1}) for p in _SHIPPED[:2]]


@pytest.mark.parametrize("path,ev", _SHIPPED_CASES)
def test_shipped_sets_satisfy_the_hypotheses_and_bounds_agree(path, ev):
    # Reference: CredalJT with SCIP, certified endpoints only.
    cnv = _cnv(path)
    rep = realizability_report(cnv.lcn, cnv)
    assert rep["dag_structured"] and rep["fully_realizable"], rep
    assert rep["locally_feasible"], rep["infeasible_rows"]
    cve = _quiet(CredalVE(cnv).run, evidence=ev, verbosity=0)
    cjt = CredalJT(cnv=cnv)
    _quiet(cjt.run, evidence=ev, solver="scip", verbosity=0, time_limit=60)
    checked = 0
    for st in cjt.solve_status:
        if not st.get("certified"):
            continue
        a, j = st["atom"], (0 if st["sense"] == "min" else 1)
        assert cve[a][j][1] == pytest.approx(st["value"], abs=1e-4), (a, st)
        checked += 1
    assert checked > 0


def test_local_feasibility_counterexample():
    # DAG-structured and fully realizable, but the row U=1 carries two
    # contradictory sentences: D_LCN forces P(U) = 0, the strong extension
    # does not.
    path = os.path.join(_ROOT, "examples", "fr_counter_lf.lcn")
    cnv = _cnv(path)
    rep = realizability_report(cnv.lcn, cnv)
    assert rep["fully_realizable"] and rep["dag_structured"]
    assert not rep["locally_feasible"]
    assert rep["infeasible_rows"] == [("X", {"U": 1})]
    cve = _quiet(CredalVE(cnv).run, evidence={}, verbosity=0)
    ei = ExactInference(lcn=cnv.lcn)
    ex = _quiet(ei.run, evidence={}, solver="scip", verbosity=0,
                time_limit=60, progress_bar=False)
    assert ei.status["U"]["max"][2] == "confirmed"
    assert ex["U"][1][1] == pytest.approx(0.0, abs=1e-6)
    assert cve["U"][1][1] == pytest.approx(1.0, abs=1e-6)


def test_non_factorizing_compound_lcn_breaks_inclusion():
    # Z -> Y -> {C,D}. The member Z ~ Bern(1/2), Y _|_ Z, C = D = not Z lies
    # in D_LCN (every sentence and both LMC assertions hold: C and D determine
    # each other) but does not factorize along the component DAG, so it is not
    # in the strong extension. It gives P(C | Z=1) = 0, below CVE's lower
    # bound: CVE (exact for D_SE) is not outer for D_LCN on this LCN.
    path = os.path.join(_ROOT, "examples", "fr_counter_factorizing.lcn")
    cnv = _cnv(path)
    lcn = cnv.lcn
    atoms = ["Z", "Y", "C", "D"]
    joint = {}
    for z, y in itertools.product((0, 1), repeat=2):
        joint[(z, y, 1 - z, 1 - z)] = 0.25

    def prob(formula, extra=None):
        tot = 0.0
        for vals, pr in joint.items():
            t = dict(zip(atoms, vals))
            if extra is not None and not extra(t):
                continue
            if formula is None or formula.evaluate(
                    table={a: t[a] for a in formula.atoms.values()}):
                tot += pr
        return tot

    for s in lcn.sentences.values():
        if s.psi_formula is None:
            v = prob(s.phi_formula)
        else:
            v = prob(s.phi_and_psi_formula) / prob(s.psi_formula)
        assert s.lower_prob - 1e-12 <= v <= s.upper_prob + 1e-12, s.label

    def marg(keep):
        out = {}
        for vals, pr in joint.items():
            k = tuple(v for a, v in zip(atoms, vals) if a in keep)
            out[k] = out.get(k, 0.0) + pr
        return out, [a for a in atoms if a in keep]

    assertions = lcn.independencies.get_assertions()
    assert len(assertions) == 2
    for ia in assertions:
        X, Y, S = set(ia.event1), set(ia.event2), set(ia.event3)
        mxys, o = marg(X | Y | S)
        ms, os_ = marg(S)
        mxs, ox = marg(X | S)
        mys, oy = marg(Y | S)
        for vals in itertools.product((0, 1), repeat=len(o)):
            d = dict(zip(o, vals))
            lhs = mxys.get(vals, 0.0) * ms.get(tuple(d[a] for a in os_), 0.0)
            rhs = mxs.get(tuple(d[a] for a in ox), 0.0) * \
                mys.get(tuple(d[a] for a in oy), 0.0)
            assert lhs == pytest.approx(rhs, abs=1e-12), str(ia)

    p_c_given_z = prob(None, lambda t: t["C"] == 1 and t["Z"] == 1) / \
        prob(None, lambda t: t["Z"] == 1)
    assert p_c_given_z == 0.0
    cve = _quiet(CredalVE(cnv).run, evidence={"Z": 1}, verbosity=0)
    assert cve["C"][0][1] == pytest.approx(0.12, abs=1e-6)   # > 0: not outer
    assert not realizability_report(lcn, cnv)["dag_structured"]
