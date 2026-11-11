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

# An infeasible local program (no joint on the family scope satisfies the
# in-scope sentences with the parent configuration at positive probability)
# must give a flagged, vacuous row -- not the meaningless value a local
# fallback solver returns -- on every compile path, and the flag must survive
# the .cn round trip.

import contextlib
import io
import json
import os
import warnings

import pytest

from lcn.core.model import LCN
from lcn.inference.marginal.cn.credal_network import CredalNetwork

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LF = os.path.join(_ROOT, "examples", "fr_counter_lf.lcn")


def _lcn(path=_LF):
    lcn = LCN()
    with contextlib.redirect_stdout(io.StringIO()):
        lcn.from_lcn(file_name=path)
    return lcn


def _row_bounds(cn, child, u):
    """(lobo, upbo) of P(child = 1 | U = u)."""
    for factor in cn.factors:
        for e in factor.values():
            if e["child"] == child and list(e["interpretation"]) == [1, u]:
                return e["lobo"], e["upbo"], e.get("infeasible", False)
    raise KeyError(child)


@pytest.mark.parametrize("method,solver,n_jobs", [
    ("linear", "ipopt", 1), ("linear", "ipopt", 2), ("linear", "scip", 1),
    ("linear-tight", "ipopt", 1)])
def test_infeasible_row_is_flagged_and_vacuous(method, solver, n_jobs):
    with pytest.warns(UserWarning, match="infeasible"):
        with contextlib.redirect_stdout(io.StringIO()):
            cn = CredalNetwork.from_lcn(_lcn(), method=method, solver=solver,
                                        n_jobs=n_jobs, verbosity=0)
    assert cn.infeasible_rows == [("X", {"U": 1})]
    assert _row_bounds(cn, "X", 1) == (0.0, 1.0, True)
    lo, hi, flag = _row_bounds(cn, "X", 0)                # feasible row
    assert (lo, hi, flag) == (pytest.approx(0.3), pytest.approx(0.6), False)


def test_feasible_networks_have_no_flags_and_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with contextlib.redirect_stdout(io.StringIO()):
            cn = CredalNetwork.from_lcn(
                _lcn(os.path.join(_ROOT, "examples", "alarm.lcn")),
                verbosity=0)
    assert cn.infeasible_rows == []
    assert not any("infeasible" in e for f in cn.factors for e in f.values())


def test_flag_survives_the_cn_round_trip(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            cn = CredalNetwork.from_lcn(_lcn(), verbosity=0)
    path = str(tmp_path / "lf.cn")
    cn.save_cn(path, method="linear", merge_budget=1, solver="ipopt")
    back = CredalNetwork.load_cn(path, _lcn())
    assert back.infeasible_rows == [("X", {"U": 1})]


def test_old_caches_are_stale(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            cn = CredalNetwork.from_lcn(_lcn(), verbosity=0)
    path = str(tmp_path / "lf.cn")
    cn.save_cn(path, method="linear", merge_budget=1, solver="ipopt")
    CN_FORMAT_VERSION = CredalNetwork.CN_FORMAT_VERSION
    assert CredalNetwork.cn_metadata(path)["version"] == CN_FORMAT_VERSION
    doc = json.load(open(path))
    for old in range(1, CN_FORMAT_VERSION):
        doc["version"] = old
        json.dump(doc, open(path, "w"))
        assert CredalNetwork.cn_metadata(path) is None     # cache miss
