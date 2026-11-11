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

# Tests for the generator's CredalJT-exactness gate (``cjt_exact``).
#
# The gate accepts an instance only if CredalJT is provably exact on it:
# "structural" checks H1-H4 of the CJT exactness theorem on the junction tree
# CredalJT itself builds (every sentence hosted, every LMC assertion discharged
# with nothing relaxed, width within budget); "certified" also solves every
# marginal with SCIP and requires every endpoint to be certified.

import contextlib
import io
import os

import pytest
from pyomo.environ import SolverFactory

from lcn.benchmarks.generator import Generator, cjt_exactness_report
from lcn.core.model import LCN

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SCIP_OK = SolverFactory("scip").available(exception_flag=False)
_needs_scip = pytest.mark.skipif(not _SCIP_OK, reason="SCIP not available")


def _load(path):
    lcn = LCN()
    with contextlib.redirect_stdout(io.StringIO()):
        lcn.from_lcn(file_name=path)
    return lcn


@pytest.mark.parametrize("graph_type,kw", [
    ("tree-fr", {}),
    ("polytree-fr", {}),
    ("ktree-fr", {"k": 3}),
    ("dag-fr", {"max_treewidth": 3}),
    ("dag", {"max_treewidth": 3}),
])
def test_structural_gate_accepts_and_reports(graph_type, kw):
    gen = Generator(seed=3)
    out = gen.generate(num_vars=12, graph_type=graph_type, num_instances=3,
                       num_extras=2, cjt_exact="structural", verbosity=0, **kw)
    assert len(out) == 3
    for lcn in out:
        rep = lcn.cjt_report
        assert rep["structural"] and rep["exact"]
        assert rep["h3"]["holds"] and rep["h3"]["relaxed"] == 0
        assert rep["width"] <= 12
        # Re-checking a shipped instance gives the same verdict.
        again = cjt_exactness_report(lcn, max_width=12)
        assert again["exact"]


def test_gate_rejects_over_budget_width():
    # A k-tree with k=4 has clusters of 5 atoms; a width budget of 4 must
    # reject every candidate.
    gen = Generator(seed=5)
    out = gen.generate(num_vars=8, graph_type="ktree-fr", k=4,
                       num_instances=1, num_extras=2, cjt_exact="structural",
                       cjt_max_width=4, max_retries=3, verbosity=0)
    assert out == []


def test_report_flags_unhostable_compound_assertions():
    # smokers.lcn has an undirected component whose LMC assertions split a
    # compound node: the ordered tree does not apply and the min-fill tree
    # needs the whole joint, so a tight width budget must reject it.
    rep = cjt_exactness_report(_load(os.path.join(_ROOT, "examples",
                                                  "smokers.lcn")), max_width=6)
    assert not rep["exact"]
    assert "budget" in rep["reason"] or "H3" in rep["reason"]


def test_default_gate_does_not_change_the_stream():
    a = Generator(seed=42).generate(num_vars=8, graph_type="polytree-fr",
                                    num_instances=2, num_extras=2, verbosity=0)
    b = Generator(seed=42).generate(num_vars=8, graph_type="polytree-fr",
                                    num_instances=2, num_extras=2,
                                    cjt_exact=None, verbosity=0)
    assert [str(s) for x in a for s in x.sentences.values()] == \
           [str(s) for x in b for s in x.sentences.values()]
    assert not any(hasattr(x, "cjt_report") for x in a)


def test_saved_header_records_report_and_reparses(tmp_path):
    gen = Generator(seed=9)
    [lcn] = gen.generate(num_vars=8, graph_type="dag-fr", max_treewidth=3,
                         num_instances=1, num_extras=2,
                         cjt_exact="structural", verbosity=0)
    path = tmp_path / "x.lcn"
    gen.save(lcn, str(path))
    text = path.read_text()
    assert text.startswith("# CredalJT exactness")
    assert len(_load(str(path)).sentences) == len(lcn.sentences)


@_needs_scip
def test_certified_gate_certifies_every_endpoint():
    gen = Generator(seed=11)
    out = gen.generate(num_vars=7, graph_type="polytree-fr", num_instances=1,
                       num_extras=2, cjt_exact="certified", cjt_time_limit=30,
                       max_retries=10, verbosity=0)
    assert len(out) == 1
    rep = out[0].cjt_report
    assert rep["certified"] and rep["exact"]
    for lo, hi in rep["marginals"].values():
        assert 0.0 <= lo <= hi <= 1.0
