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

# Fast tests of the experiment harness (no solver-heavy runs): the algorithm
# list, the variant flags and their file suffixes, the CLI flag mapping, and
# the certified-only reference of the analysis.

import argparse
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "experiments"))

import analyze_results as ar  # noqa: E402
import run_algorithm as ra  # noqa: E402


def _parse(argv):
    p = argparse.ArgumentParser()
    ra.add_algorithm_arguments(p)
    return p.parse_args(argv)


def test_algorithm_list_is_exactly_the_six():
    assert ra.ALGORITHMS == ["exact", "ariel", "cjt", "ivbp", "cve", "approxlp"]
    assert ra.STAGES == ["compile", "enumerate"]


@pytest.mark.parametrize("old", ["ibp", "ccte", "ccte_e", "ijgp", "cjt_l",
                                 "cjt_g", "exact_g", "exact_l", "cve_cm",
                                 "cve_cp", "cve_d4"])
def test_removed_names_are_rejected(old):
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", choices=ra.ALGORITHMS)
    with pytest.raises(SystemExit):
        p.parse_args(["--algorithm", old])


@pytest.mark.parametrize("algo,kw,suffix", [
    ("cve", {}, ""),
    ("cve", {"epsilon": 0.01}, "_e0.01"),
    ("cve", {"epsilon": 0.1}, "_e0.1"),
    ("cve", {"epsilon": 0.0}, ""),            # epsilon 0 is exact CVE
    ("cjt", {"solver": "scip"}, ""),
    ("cjt", {"solver": "ipopt"}, "_ipopt"),
    ("exact", {"solver": "ipopt"}, "_ipopt"),
    ("ivbp", {"n_iters": 100}, ""),
    ("ivbp", {"n_iters": 20}, "_it20"),
    ("approxlp", {}, ""),
    ("approxlp", {"n_restarts": 0}, "_r0"),
    ("approxlp", {"n_iters": 10, "n_restarts": 2}, "_it10_r2"),
    ("ariel", {"solver": "ipopt"}, ""),   # solver is not an ariel variant
])
def test_variant_suffix(algo, kw, suffix):
    assert ra.variant_suffix(algo, kw) == suffix


def test_variants_get_distinct_files():
    names = {"cve" + ra.variant_suffix("cve", kw) for kw in
             ({}, {"epsilon": 0.1}, {"epsilon": 0.01})}
    assert len(names) == 3


def test_cli_flags_map_to_kwargs():
    kw = ra.kwargs_from_args(_parse([
        "--solver", "ipopt", "--epsilon", "0.1",
        "--cn-solver", "scip", "--n-iters", "7", "--no-cache"]))
    assert kw["solver"] == "ipopt"
    assert kw["epsilon"] == 0.1
    assert kw["cn_solver"] == "scip"        # credal-network build solver
    assert kw["n_iters"] == 7 and kw["cache"] is False
    # defaults: the engine solver is SCIP, the build solver key is absent
    kw0 = ra.kwargs_from_args(_parse([]))
    assert kw0 == {"solver": "scip"}


def test_run_name_matches_output_file_name():
    rec = {"algorithm": "cve", "variant": {"epsilon": 0.1}}
    assert ar.run_name(rec) == "cve_e0.1"
    assert ar.run_name({"algorithm": "cjt", "variant": {"solver": "scip"}}) == "cjt"


def _rec(algo, marg, **extra):
    r = {"algorithm": algo, "variant": ra.variant_of(algo, {}), "status": "ok",
         "instance": "i.lcn", "graph_type": "tree_fr", "num_vars": 2,
         "build_time": 0.0, "run_time": 0.0, "total_time": 0.0,
         "marginals": {a: {"lower": [1 - hi, lo], "upper": [1 - lo, hi]}
                       for a, (lo, hi) in marg.items()}}
    r.update(extra)
    return r


def test_reference_uses_certified_endpoints_only(capsys):
    # CJT certifies x's lower endpoint only; the uncertified upper endpoint
    # must not enter the upper-bound error, and y (fully certified) must.
    cjt = _rec("cjt", {"x": (0.2, 0.9), "y": (0.1, 0.5)}, cjt_solves=[
        {"atom": "x", "sense": "min", "certified": True},
        {"atom": "x", "sense": "max", "certified": False},
        {"atom": "y", "sense": "min", "certified": True},
        {"atom": "y", "sense": "max", "certified": True}])
    ivbp = _rec("ivbp", {"x": (0.1, 0.3), "y": (0.1, 0.6)})
    ar.analyze([cjt, ivbp], reference="cjt")
    out = capsys.readouterr().out
    assert "ivbp" in out
    # lower errors: |0.2-0.1| (x) and 0 (y) -> mean 0.05;
    # upper errors: y only, |0.6-0.5| = 0.1 (x's uncertified upper excluded)
    line = [l for l in out.splitlines() if " ivbp " in f" {l} "][0]
    nums = [float(t) for t in line.split()[3:5]]   # mae_lb, std_lb
    assert nums[0] == pytest.approx(0.05)
    assert "0.100000" in line                       # mae_ub


def test_exact_engines_are_not_scored():
    assert ar._is_excluded("cjt", "ariel")
    assert ar._is_excluded("cjt_ipopt", "cjt")
    assert ar._is_excluded("exact", "cjt")
    assert not ar._is_excluded("cve_e0.1", "cjt")


def test_timeout_record_carries_variant():
    # A run killed by the watchdog must still be nameable by the analysis.
    lcn = os.path.join(os.path.dirname(_HERE), "examples", "alarm.lcn")
    r = ra.run_single(lcn, "cve", verbosity=0, time_limit=0.01,
                      epsilon=0.1)
    assert r["status"] in ("timeout", "ok", "error")
    assert r["algorithm"] == "cve"
    assert r["variant"] == {"epsilon": 0.1}
    assert ar.run_name(r) == "cve_e0.1"


def test_exact_engines_split_the_wall_budget_per_solve():
    # 5 atoms -> 10 endpoint solves; all must fit within 90% of the budget so
    # the engine stops cleanly before the watchdog fires.
    lcn = os.path.join(os.path.dirname(_HERE), "examples", "alarm.lcn")
    per = ra.per_solve_limit(lcn, 1200)
    assert per == pytest.approx(0.9 * 1200 / 10)
    assert 2 * 5 * per < 1200
