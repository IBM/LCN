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

# Unit tests for the CredalJT junction-tree exact engine (scheme D5).
#
# CredalJT builds ONE junction tree + ONE constraint NLP and computes every
# non-evidence singleton atom's exact posterior bound by swapping only the
# objective per atom. The tests run on examples/d4_biting.lcn -- small (3 atoms)
# and crafted with a cross-family sentence P(B and C) <= 0.05 so the tree and
# the constraint hosting are non-trivial yet fast to solve with SCIP.

import contextlib
import io
import os

import numpy as np
import pytest
from pyomo.environ import SolverFactory

from lcn.core.model import LCN
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.cn.junction_nlp import CredalJT
from lcn.inference.marginal.exact import ExactInference


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_BITING = os.path.join(_ROOT, "examples", "d4_biting.lcn")
_CHAIN = os.path.join(_ROOT, "examples", "chain.lcn")

# SCIP is required for D5 to be exact (the cluster NLP is nonconvex); skip the
# SCIP-backed assertions when the solver is not on PATH.
_SCIP_OK = SolverFactory("scip").available(exception_flag=False)
_needs_scip = pytest.mark.skipif(not _SCIP_OK, reason="SCIP solver not available")

# True exact P(atom=1) bounds for d4_biting (from ExactInference solver=global).
_EXACT = {"A": (0.4, 0.6), "B": (0.04, 0.72), "C": (0.08, 0.80)}
_TOL = 3e-3


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


@pytest.fixture(scope="module")
def cnv():
    """A structure-only credal network for d4_biting (shared across the module).

    CredalJT reads only the chain-graph structure, so this uses the realistic D5
    build path solve_families=False -- no per-family interval solves. (Parity
    with the interval-solving build is checked in TestStructureOnlyBuild.)"""
    lcn = LCN()
    _quiet(lcn.from_lcn, file_name=_BITING)
    return _quiet(CredalNetworkVertices.from_lcn, lcn,
                  method="linear", solve_families=False, verbosity=0)


@pytest.fixture(scope="module")
def chain_cnv():
    """A structure-only credal network for the 8-atom Markov chain
    examples/chain.lcn (solve_families=False -- the D5 build path)."""
    lcn = LCN()
    _quiet(lcn.from_lcn, file_name=_CHAIN)
    return _quiet(CredalNetworkVertices.from_lcn, lcn,
                  method="linear", solve_families=False, verbosity=0)


# ======================================================================
# 1. Result shape and instance attributes
# ======================================================================

class TestResultContract:

    @_needs_scip
    def test_returns_all_singleton_atoms(self, cnv):
        jt = CredalJT(cnv=cnv)
        results = _quiet(jt.run, evidence={}, solver="scip", verbosity=0)
        # Every non-evidence singleton atom is present.
        assert set(results.keys()) == {"A", "B", "C"}
        assert set(jt.singleton_marginals.keys()) == {"A", "B", "C"}

    @_needs_scip
    def test_result_arrays_are_two_state_and_valid(self, cnv):
        jt = CredalJT(cnv=cnv)
        results = _quiet(jt.run, evidence={}, solver="scip", verbosity=0)
        for atom, (lo, hi) in results.items():
            assert isinstance(lo, np.ndarray) and isinstance(hi, np.ndarray)
            assert lo.shape == (2,) and hi.shape == (2,)
            # [P(=0), P(=1)] bounds; lower <= upper, both in [0, 1].
            assert np.all(lo <= hi + 1e-9)
            assert np.all(lo >= -1e-9) and np.all(hi <= 1 + 1e-9)
            # singleton_marginals holds the P(atom=1) interval = (lo[1], hi[1]).
            slo, shi = jt.singleton_marginals[atom]
            assert slo == pytest.approx(float(lo[1]), abs=1e-9)
            assert shi == pytest.approx(float(hi[1]), abs=1e-9)

    @_needs_scip
    def test_sets_status_attributes(self, cnv):
        jt = CredalJT(cnv=cnv)
        _quiet(jt.run, evidence={}, solver="scip", verbosity=0)
        # d4_biting fits one cluster of 3 atoms; exact, not degenerate.
        assert jt.d5_exact is True
        assert jt.induced_width == 3
        assert jt.degenerate is False
        # Timing stats are recorded and consistent.
        assert jt.build_time >= 0.0
        assert jt.elimination_time >= 0.0
        assert jt.total_time == pytest.approx(
            jt.build_time + jt.elimination_time, abs=1e-9)

    @_needs_scip
    def test_records_nlp_stats(self, cnv):
        jt = CredalJT(cnv=cnv)
        _quiet(jt.run, evidence={}, solver="scip", verbosity=0)
        st = jt.nlp_stats
        assert st is not None
        assert st["n_atoms_total"] == 3  # d4_biting has 3 atoms A, B, C

        # Structural invariants (topology-independent):
        # one simplex row per cluster; total variables = sum of cluster sizes;
        # the category counts partition the total; nonlinear rows are a subset.
        assert st["n_simplex"] == st["n_clusters"] >= 1
        assert st["n_edges"] == st["n_clusters"] - 1  # a (connected) tree
        assert st["n_variables"] >= 2 * st["n_clusters"]  # each cluster >= 1 atom
        assert st["max_cluster_atoms"] >= 1
        assert st["n_constraints"] == (
            st["n_simplex"] + st["n_separator"]
            + st["n_sentence"] + st["n_lmc"])
        assert 0 <= st["n_nonlinear"] <= st["n_constraints"]
        # Sentence rows come in pairs (>= lo, <= hi).
        assert st["n_sentence"] % 2 == 0
        # d4_biting: the cross-family sentence P(B and C) <= 0.05 forces a
        # 3-atom host cluster, so at least one cluster carries all 3 atoms.
        assert st["max_cluster_atoms"] == 3


# ======================================================================
# 2. Exactness vs the certified-global oracle
# ======================================================================

class TestExactness:

    @_needs_scip
    def test_unconditional_matches_exact_inference(self, cnv):
        jt = CredalJT(cnv=cnv)
        _quiet(jt.run, evidence={}, solver="scip", verbosity=0)
        for atom, (elo, ehi) in _EXACT.items():
            lo, hi = jt.singleton_marginals[atom]
            assert lo == pytest.approx(elo, abs=_TOL), f"P({atom}=1) lower"
            assert hi == pytest.approx(ehi, abs=_TOL), f"P({atom}=1) upper"

    @_needs_scip
    def test_matches_exact_inference_global_recomputed(self, cnv):
        # Cross-check against a freshly run ExactInference (global) so the test
        # does not rely only on the hard-coded _EXACT constants.
        jt = CredalJT(cnv=cnv)
        _quiet(jt.run, evidence={}, solver="scip", verbosity=0)
        lcn = LCN()
        _quiet(lcn.from_lcn, file_name=_BITING)
        ei = ExactInference(lcn)
        for atom in ("A", "B", "C"):
            elo, ehi = _quiet(ei.run_query, atom, evidence={},
                              solver="global", verbosity=0, time_limit=60)
            lo, hi = jt.singleton_marginals[atom]
            assert lo == pytest.approx(elo, abs=_TOL)
            assert hi == pytest.approx(ehi, abs=_TOL)

    @_needs_scip
    def test_conditional_query_skips_evidence_atom(self, cnv):
        jt = CredalJT(cnv=cnv)
        _quiet(jt.run, evidence={"C": 1}, solver="scip", verbosity=0)
        # The observed atom C is not among the computed marginals.
        assert "C" not in jt.singleton_marginals
        assert set(jt.singleton_marginals.keys()) == {"A", "B"}
        # Conditional bounds are valid probabilities, and contained in the
        # exact conditional bound (D5 is exact, so equal within tol). Compare to
        # ExactInference for P(B | C=1).
        lcn = LCN()
        _quiet(lcn.from_lcn, file_name=_BITING)
        ei = ExactInference(lcn)
        elo, ehi = _quiet(ei.run_query, "B", evidence={"C": 1},
                          solver="global", verbosity=0, time_limit=60)
        lo, hi = jt.singleton_marginals["B"]
        assert lo == pytest.approx(elo, abs=_TOL)
        assert hi == pytest.approx(ehi, abs=_TOL)


# ======================================================================
# 3. Solver backends and degenerate detection
# ======================================================================

class TestBackendsAndDegenerate:

    def test_ipopt_backend_runs(self, cnv):
        # ipopt is a local solver: it must run and return valid intervals
        # (not necessarily the certified-global bound), with no exceptions.
        jt = CredalJT(cnv=cnv)
        results = _quiet(jt.run, evidence={}, solver="ipopt", verbosity=0)
        assert set(results.keys()) == {"A", "B", "C"}
        for lo, hi in jt.singleton_marginals.values():
            assert 0.0 - 1e-9 <= lo <= hi + 1e-9 <= 1.0 + 1e-9

    @_needs_scip
    def test_degenerate_flag_on_all_vacuous(self, tmp_path):
        # An LCN whose atoms are entirely unconstrained -> every marginal [0,1]
        # -> degenerate.
        path = tmp_path / "vacuous.lcn"
        path.write_text(
            "A1: 0.0 <= P(A) <= 1.0\n"
            "B1: 0.0 <= P(B) <= 1.0\n"
            "C1: 0.0 <= P(C) <= 1.0\n")
        lcn = LCN()
        _quiet(lcn.from_lcn, file_name=str(path))
        vac_cnv = _quiet(CredalNetworkVertices.from_lcn, lcn,
                         method="linear", verbosity=0)
        jt = CredalJT(cnv=vac_cnv)
        _quiet(jt.run, evidence={}, solver="scip", verbosity=0)
        assert jt.degenerate is True
        for lo, hi in jt.singleton_marginals.values():
            assert lo == pytest.approx(0.0, abs=1e-6)
            assert hi == pytest.approx(1.0, abs=1e-6)


# ======================================================================
# 4. Build-once invariant (one junction tree for all marginals)
# ======================================================================

class TestBuildOnce:

    @_needs_scip
    def test_builds_constraint_model_once(self, cnv, monkeypatch):
        # The whole point of CredalJT: the constraint NLP is built ONCE, not
        # once per atom. Count calls to _build_constraint_model during a run.
        import lcn.inference.marginal.cn.junction_nlp as J
        calls = {"n": 0}
        orig = J._build_constraint_model

        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        monkeypatch.setattr(J, "_build_constraint_model", counting)
        jt = CredalJT(cnv=cnv)
        _quiet(jt.run, evidence={}, solver="scip", verbosity=0)
        assert calls["n"] == 1, \
            f"constraint model built {calls['n']} times, expected 1"


# ======================================================================
# 4b. Structure-only build (solve_families=False): parity + no solves
# ======================================================================

class TestStructureOnlyBuild:

    def test_no_family_solves_when_structure_only(self):
        # solve_families=False must skip the per-family interval solves entirely:
        # _solve_family is never called during the credal-network build.
        import lcn.inference.marginal.cn.credal_network as CN
        calls = {"n": 0}
        orig = CN._solve_family

        lcn = LCN()
        _quiet(lcn.from_lcn, file_name=_BITING)

        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        import unittest.mock as mock
        with mock.patch.object(CN, "_solve_family", counting):
            _quiet(CredalNetworkVertices.from_lcn, lcn,
                   method="linear", solve_families=False, verbosity=0)
        assert calls["n"] == 0, \
            f"_solve_family called {calls['n']} times with solve_families=False"

    def test_structure_only_forces_no_vertices(self):
        # With placeholder bounds there is nothing to enumerate, so
        # solve_families=False must coerce enumerate_vertices off.
        lcn = LCN()
        _quiet(lcn.from_lcn, file_name=_BITING)
        cnv = _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                     solve_families=False, enumerate_vertices=True, verbosity=0)
        assert cnv.extreme_points is None
        assert cnv.credal_net is None

    @_needs_scip
    def test_marginals_match_interval_solving_build(self):
        # The whole point: CredalJT marginals are identical whether or not the
        # per-family bounds were solved -- the skipped intervals feed no D5 row.
        lcn1 = LCN()
        _quiet(lcn1.from_lcn, file_name=_BITING)
        cnv_solved = _quiet(CredalNetworkVertices.from_lcn, lcn1,
                            method="linear", solve_families=True, verbosity=0)
        r_solved = _quiet(CredalJT(cnv=cnv_solved).run,
                          evidence={}, solver="scip", verbosity=0)

        lcn2 = LCN()
        _quiet(lcn2.from_lcn, file_name=_BITING)
        cnv_struct = _quiet(CredalNetworkVertices.from_lcn, lcn2,
                            method="linear", solve_families=False, verbosity=0)
        r_struct = _quiet(CredalJT(cnv=cnv_struct).run,
                          evidence={}, solver="scip", verbosity=0)

        assert set(r_solved) == set(r_struct)
        for atom in r_solved:
            for k in (0, 1):  # (lo, hi) arrays
                assert np.allclose(r_solved[atom][k], r_struct[atom][k],
                                   atol=1e-6), f"mismatch on {atom}"


# ======================================================================
# 5. Treewidth advantage on a Markov chain
# ======================================================================

class TestTreewidth:

    def test_chain_collapses_to_treewidth(self, chain_cnv):
        # On the 8-atom Markov chain, CredalJT must build n-1=7 clusters of two
        # atoms each (treewidth 1) -- NOT a single 2^8 joint. The LMC assertions
        # are RIP-implied by the separators, so none are imposed.
        from lcn.inference.marginal.cn.junction_nlp import _build_jt_and_hosts
        jt, sbh, lbh, _, over, fb = _build_jt_and_hosts(
            chain_cnv, None, {}, 16)
        assert not over and not fb
        assert jt.max_cluster_size() == 2, \
            f"max cluster {jt.max_cluster_size()} atoms; chain treewidth is 1"
        assert len(jt.cluster_ids) == 8  # 7 two-atom clusters + the {x7} leaf
        # every cluster has at most two atoms
        assert all(len(jt.atoms[c]) <= 2 for c in jt.cluster_ids)
        # all LMC assertions RIP-implied -> none imposed
        assert sum(len(v) for v in lbh.values()) == 0

    def test_rip_implied_predicate(self, chain_cnv):
        # The Markov-chain LMC (x_{i+1} perp earlier | x_i) is RIP-implied;
        # a fabricated non-separator conditioning is not.
        from lcn.inference.marginal.cn.junction_nlp import _build_jt_and_hosts
        jt, *_ = _build_jt_and_hosts(chain_cnv, None, {}, 16)
        assert jt._rip_implied(["x2"], ["x0"], ["x1"]) is True
        # x0 and x2 are NOT independent given x4 (x4 is not on the x0-x2 path)
        assert jt._rip_implied(["x0"], ["x2"], ["x4"]) is False

    @_needs_scip
    def test_chain_matches_exact(self, chain_cnv):
        jt = CredalJT(chain_cnv)
        _quiet(jt.run, evidence={}, solver="scip", verbosity=0)
        assert jt.induced_width == 2
        lcn = LCN()
        _quiet(lcn.from_lcn, file_name=_CHAIN)
        ei = ExactInference(lcn)
        for atom in ("x0", "x2", "x4", "x7"):
            elo, ehi = _quiet(ei.run_query, atom, evidence={},
                              solver="local", verbosity=0)
            lo, hi = jt.singleton_marginals[atom]
            assert lo == pytest.approx(elo, abs=_TOL), f"{atom} lower"
            assert hi == pytest.approx(ehi, abs=_TOL), f"{atom} upper"
