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

# Tests of the experiment harness: the run registry and its naming scheme,
# the CLI, the record schema, the per-instance evidence, and the analysis
# (certified-only reference, scored runs, inner / outer rates, guarantee
# checks, solved counts). The smoke test runs part of the grid end to end on
# a tiny instance.

import argparse
import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

import analyze_results as ar  # noqa: E402
import run_algorithm as ra  # noqa: E402
import run_experiment as rx  # noqa: E402
import runs as R  # noqa: E402

_ALARM = os.path.join(_ROOT, "examples", "alarm.lcn")


def _parse(argv):
    p = argparse.ArgumentParser()
    ra.add_algorithm_arguments(p)
    return p.parse_args(argv)


# ----------------------------------------------------------------------
# Registry and names
# ----------------------------------------------------------------------

def test_the_nine_runs():
    assert R.RUNS == ["exact_scip", "exact_ipopt", "ariel", "cjt_scip",
                      "cjt_ipopt", "cve", "cve_eps_0.1", "ivbp", "cdve"]
    assert R.ALGORITHMS == ["exact", "ariel", "cjt", "cve", "ivbp", "cdve"]
    assert ra.ALGORITHMS == R.ALGORITHMS


@pytest.mark.parametrize("name", R.RUNS + ["cve_eps_0.01", "cve_eps_0.5",
                                           "cdve_r0", "cdve_it10_r2",
                                           "ariel_it20", "ivbp_it50"])
def test_names_round_trip(name):
    algo, params = R.parse_run(name)
    assert R.run_name(algo, params) == name


@pytest.mark.parametrize("bad", [
    "exact", "cjt", "cjt_l", "cjt_g", "exact_g", "exact_l",   # solver implicit
    "approxlp", "ibp", "ccte", "ijgp", "cve_cm", "cve_cp", "cve_d4",
    "cve_e0.1", "cve_eps", "cve_eps_0", "cve_eps0.1",          # eps naming
    "cdve_r8", "ariel_it10", "ivbp_r2"])                       # non-canonical
def test_removed_and_non_canonical_names_are_rejected(bad):
    with pytest.raises(ValueError):
        R.parse_run(bad)


def test_cve_eps_naming():
    assert R.run_name("cve", {"epsilon": 0.1}) == "cve_eps_0.1"
    assert R.run_name("cve", {"epsilon": 0.001}) == "cve_eps_0.001"
    assert R.run_name("cve", {"epsilon": 0.0}) == "cve"      # eps 0 = exact
    assert R.parse_run("cve_eps_0.1") == ("cve", {"epsilon": 0.1})


def test_solver_is_required_for_exact_and_cjt():
    for algo in ("exact", "cjt"):
        with pytest.raises(ValueError):
            R.run_name(algo, {})
        assert R.run_name(algo, {"solver": "ipopt"}) == f"{algo}_ipopt"
        with pytest.raises(ValueError):
            ra.run_single(_ALARM, algo, verbosity=0)


def test_guarantees_and_labels():
    g = {n: R.guarantee(n) for n in R.RUNS}
    assert g == {"exact_scip": "certified", "exact_ipopt": "none",
                 "ariel": "none", "cjt_scip": "certified", "cjt_ipopt": "none",
                 "cve": "exact_se", "cve_eps_0.1": "inner_eps",
                 "ivbp": "exact_poly", "cdve": "inner"}
    assert R.label("cjt_scip") == "CJT (SCIP)"
    assert R.label("ivbp") == "IVBP" and R.label("cdve") == "CDVE"
    assert "0.1" in R.label("cve_eps_0.1")
    assert sorted(R.RUNS, key=R.order_key) == R.RUNS


# ----------------------------------------------------------------------
# CLI and records
# ----------------------------------------------------------------------

def test_cli_flags_map_to_kwargs():
    kw = ra.kwargs_from_args(_parse([
        "--solver", "ipopt", "--epsilon", "0.1", "--cn-solver", "scip",
        "--n-iters", "7", "--n-restarts", "2", "--no-cache"]))
    assert kw["solver"] == "ipopt" and kw["epsilon"] == 0.1
    assert kw["cn_solver"] == "scip"
    assert kw["n_iters"] == 7 and kw["n_restarts"] == 2
    assert kw["cache"] is False
    assert ra.kwargs_from_args(_parse([])) == {}      # no hidden solver


def test_record_identity_fields():
    r = ra.run_single(_ALARM, "cdve", verbosity=0, n_restarts=0)
    assert r["status"] == "ok"
    assert r["run"] == "cdve_r0" and r["algorithm"] == "cdve"
    assert r["params"] == {"n_iters": 50, "n_restarts": 0}
    assert r["guarantee"] == "inner"
    assert "cdve_sweeps" in r and "variant" not in r


def test_timeout_record_carries_identity():
    r = ra.run_single(_ALARM, "cve", verbosity=0, time_limit=0.01,
                      epsilon=0.1)
    assert r["status"] in ("timeout", "ok", "error")
    assert r["run"] == "cve_eps_0.1" and r["params"] == {"epsilon": 0.1}


def test_exact_engines_split_the_wall_budget_per_solve():
    per = ra.per_solve_limit(_ALARM, 1200)
    assert per == pytest.approx(0.9 * 1200 / 10)
    assert 2 * 5 * per < 1200


def test_per_instance_evidence_is_deterministic():
    e1 = rx.sample_evidence(_ALARM, 2, seed=0)
    assert e1 == rx.sample_evidence(_ALARM, 2, seed=0)
    assert len(e1) == 2 and set(e1.values()) <= {0, 1}
    assert rx.sample_evidence(_ALARM, 0) == {}
    assert any(rx.sample_evidence(_ALARM, 2, seed=s) != e1 for s in range(1, 6))


# ----------------------------------------------------------------------
# Analysis on synthetic records
# ----------------------------------------------------------------------

def _rec(run, marg, **extra):
    r = {"run": run, "algorithm": R.parse_run(run)[0], "status": "ok",
         "instance": "i.lcn", "benchmark": "b", "evidence_atoms": 0,
         "evidence": {}, "graph_type": "tree_fr", "num_vars": 2,
         "build_time": 0.0, "run_time": 0.0, "total_time": 0.0,
         "marginals": {a: {"lower": [1 - hi, lo], "upper": [1 - lo, hi]}
                       for a, (lo, hi) in marg.items()}}
    r.update(extra)
    return r


def _row(rows, run):
    return [r for r in rows if r["run"] == run][0]


def test_reference_uses_certified_endpoints_only():
    cjt = _rec("cjt_scip", {"x": (0.2, 0.9), "y": (0.1, 0.5)}, cjt_solves=[
        {"atom": "x", "sense": "min", "certified": True},
        {"atom": "x", "sense": "max", "certified": False},
        {"atom": "y", "sense": "min", "certified": True},
        {"atom": "y", "sense": "max", "certified": True}],
        cjt_atom_structural={"x": True, "y": True})
    ivbp = _rec("ivbp", {"x": (0.1, 0.3), "y": (0.1, 0.6)})
    rows, _ = ar.analyze([cjt, ivbp], verbose=False)
    r = _row(rows, "ivbp")
    assert r["mae_lb"] == pytest.approx(0.05)    # x 0.1, y 0
    assert r["mae_ub"] == pytest.approx(0.1)     # y only (x upper uncertified)


def test_reference_requires_structural_exactness():
    """A certified CJT endpoint whose route is not structurally exact (H1-H4,
    e.g. a relaxed H3) is only an outer bound and is not a reference."""
    solves = [{"atom": a, "sense": s, "certified": True}
              for a in ("x", "y") for s in ("min", "max")]
    cjt = _rec("cjt_scip", {"x": (0.2, 0.9), "y": (0.1, 0.5)},
               cjt_solves=solves, cjt_atom_structural={"x": True, "y": False})
    assert ar.certified_endpoints(cjt) == {"x": (True, True),
                                           "y": (False, False)}
    legacy = _rec("cjt_scip", {"x": (0.2, 0.9)}, cjt_solves=solves[:2],
                  cjt_structurally_exact=True)
    assert ar.certified_endpoints(legacy) == {"x": (True, True)}


def test_exact_scip_certification_is_read():
    ex = _rec("exact_scip", {"x": (0.2, 0.9)},
              exact_status={"x": {"min": "confirmed", "max": "unsolved"}})
    assert ar.certified_endpoints(ex) == {"x": (True, False)}


def test_ipopt_runs_and_the_other_certified_engine_are_scored():
    ref = _rec("cjt_scip", {"x": (0.2, 0.6)})
    recs = [ref, _rec("cjt_ipopt", {"x": (0.25, 0.6)}),
            _rec("exact_scip", {"x": (0.2, 0.6)}),
            _rec("exact_ipopt", {"x": (0.2, 0.5)})]
    rows, _ = ar.analyze(recs, verbose=False)
    assert {r["run"] for r in rows} == {"cjt_ipopt", "exact_scip",
                                        "exact_ipopt"}
    assert _row(rows, "exact_scip")["mae_lb"] == 0.0


def test_inner_outer_rates_and_guarantee_violations():
    ref = _rec("cjt_scip", {"x": (0.2, 0.6), "y": (0.1, 0.4)})
    cdve = _rec("cdve", {"x": (0.25, 0.5), "y": (0.05, 0.4)})   # y not inner
    ivbp_loopy = _rec("ivbp", {"x": (0.1, 0.7), "y": (0.1, 0.4)},
                      ivbp_exact=False)
    rows, viol = ar.analyze([ref, cdve, ivbp_loopy], verbose=False)
    c = _row(rows, "cdve")
    assert c["inner"] == pytest.approx(0.5) and c["viol"] == 1
    assert _row(rows, "ivbp")["outer"] == 1.0
    assert _row(rows, "ivbp")["viol"] == 0       # loopy: no guarantee
    assert [v[3:6] for v in viol] == [("cdve", "y", "lower")]


def test_exact_poly_and_cve_eps_checks():
    ref = _rec("cjt_scip", {"x": (0.2, 0.6)})
    ivbp = _rec("ivbp", {"x": (0.2, 0.65)}, ivbp_exact=True)
    cve = _rec("cve", {"x": (0.2, 0.6)})
    ok = _rec("cve_eps_0.1", {"x": (0.21, 0.56)})
    rows, _ = ar.analyze([ref, ivbp, cve, ok], verbose=False)
    assert _row(rows, "ivbp")["viol"] == 1
    assert _row(rows, "cve_eps_0.1")["viol"] == 0
    bad = _rec("cve_eps_0.1", {"x": (0.3, 0.6)})       # 0.3 > 1.1 * 0.2
    rows, _ = ar.analyze([ref, cve, bad], verbose=False)
    assert _row(rows, "cve_eps_0.1")["viol"] == 1


def test_evidence_atoms_are_not_scored_and_solved_counts():
    ref = _rec("cjt_scip", {"x": (0.2, 0.6), "e": (1.0, 1.0)},
               evidence={"e": 1}, evidence_atoms=1)
    cve = _rec("cve", {"x": (0.2, 0.6), "e": (0.0, 1.0)},
               evidence={"e": 1}, evidence_atoms=1)
    tout = dict(_rec("cdve", {}, evidence_atoms=1), status="timeout")
    rows, _ = ar.analyze([ref, cve, tout], verbose=False)
    r = _row(rows, "cve")
    assert r["n_atoms"] == 1 and r["evidence"] == 1 and r["viol"] == 0
    rows_all = ar.analyze([ref, cve, tout, dict(
        _rec("cdve", {"x": (0.2, 0.6)}, evidence_atoms=1,
             evidence={"e": 1}))], verbose=False)[0]
    # the timeout record counts in 'solved', the ok one is scored
    c = _row(rows_all, "cdve")
    assert c["timeout"] == 1 and c["ok"] == 1


def test_group_by_fields():
    ref = _rec("cjt_scip", {"x": (0.2, 0.6)})
    cve = _rec("cve", {"x": (0.2, 0.6)})
    rows, _ = ar.analyze([ref, cve], group_by=["evidence", "type"],
                         verbose=False)
    assert set(rows[0]) >= {"evidence", "type", "run"}
    with pytest.raises(ValueError):
        ar.analyze([ref, cve], group_by=["nonsense"], verbose=False)


def test_records_without_run_field_are_skipped(tmp_path):
    (tmp_path / "old.jsonl").write_text(json.dumps(
        {"algorithm": "approxlp", "status": "ok"}) + "\n")
    recs, skipped = ar.load_records(str(tmp_path))
    assert recs == [] and skipped == 1


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------

def test_smoke_grid_end_to_end(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    src = os.path.join(_ROOT, "benchmarks", "tree_small_fr", "tree_fr_n5_1.lcn")
    (bench / "tree_fr_n5_1.lcn").write_text(open(src).read())
    out = tmp_path / "results"
    exp = os.path.join(_ROOT, "experiments")
    runs = ["ariel", "cjt_scip", "cve", "cve_eps_0.1", "ivbp", "cdve"]
    for k in (0, 1):
        subprocess.run(
            [sys.executable, os.path.join(exp, "run_experiment.py"),
             "--input-dir", str(bench), "--output-dir", str(out),
             "--runs", *runs, "--evidence-atoms", str(k),
             "--time-limit", "120", "--verbosity", "0"],
            # pytest leaves fd 0 unusable; the time-limit worker spawned by
            # multiprocessing would inherit it and fail to start.
            check=True, capture_output=True, cwd=_ROOT,
            stdin=subprocess.DEVNULL)
    files = sorted(p.relative_to(out).as_posix()
                   for p in out.rglob("*.jsonl"))
    assert files == sorted(f"bench/ev{k}/{r}.jsonl"
                           for k in (0, 1) for r in runs)
    rec = json.loads((out / "bench/ev1/cve_eps_0.1.jsonl").read_text())
    assert rec["run"] == "cve_eps_0.1" and rec["evidence_atoms"] == 1
    assert len(rec["evidence"]) == 1 and rec["status"] == "ok", rec["error"]
    records, _ = ar.load_records(str(out))
    rows, viol = ar.analyze(records, verbose=False)
    assert {(r["evidence"], r["run"]) for r in rows} == {
        (k, r) for k in (0, 1) for r in runs if r != "cjt_scip"}
    # a family-realizable tree: every declared guarantee holds
    assert not [v for v in viol if R.guarantee(v[3]) != "none"]
    csv_path = tmp_path / "a.csv"
    ar.write_csv(rows, str(csv_path))
    subprocess.run(
        [sys.executable, os.path.join(exp, "plot_results.py"), "--csv",
         str(csv_path), "--output", str(tmp_path / "p"), "--by-evidence"],
        check=True, capture_output=True, cwd=_ROOT, stdin=subprocess.DEVNULL)
    assert (tmp_path / "p_mae.pdf").exists()
