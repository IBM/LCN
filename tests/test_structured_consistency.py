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

# Tests for the structure-exploiting junction-tree consistency checker
# (lcn.inference.utils.structured_consistency) and its generator wiring.
#
# Properties:
#   (1) A genuinely inconsistent bounded-treewidth instance is proven
#       INCONSISTENT (the benchmarks/junkyu n24 file: independent reviewers make
#       P(all four) a product below the component lower bound). Its dropped-fix
#       counterpart is CONSISTENT.
#   (2) A tiny contradictory chain is INCONSISTENT.
#   (3) Completeness: it certifies CONSISTENT an instance the conservative
#       product witness false-rejects (benchmarks/real/cancer.lcn).
#   (4) The generator's consistency_mode="structured" gate accepts genuinely
#       consistent instances.
#
# INCONSISTENT verdicts need the global solver (SCIP); those assertions skip if
# SCIP is unavailable. CONSISTENT verdicts are sound under ipopt too.

import contextlib
import io
import os

import pytest

from lcn.core.model import LCN
from lcn.benchmarks.generator import Generator
from lcn.inference.utils.common import check_consistency_product_witness
from lcn.inference.utils.structured_consistency import (
    check_consistency_structured, is_consistent_structured,
    CONSISTENT, INCONSISTENT,
)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_JUNKYU = os.path.join(_REPO, "benchmarks", "junkyu")
_REAL = os.path.join(_REPO, "benchmarks", "real")


def _scip_available():
    try:
        from lcn.inference.marginal.exact import make_scip
        make_scip(time_limit=5.0)
        return True
    except Exception:
        return False


def _load(path):
    lcn = LCN()
    with contextlib.redirect_stdout(io.StringIO()):
        lcn.from_lcn(path)
    return lcn


requires_scip = pytest.mark.skipif(
    not _scip_available(), reason="SCIP solver not available")


@requires_scip
def test_known_inconsistent_junkyu_n24():
    """The original n24 clinic-referral instance is provably inconsistent."""
    path = os.path.join(_JUNKYU, "benchmark_scaling_directions_n24_2.lcn")
    r = check_consistency_structured(_load(path), solver="scip", time_limit=300)
    assert r.status == INCONSISTENT
    assert r.treewidth is not None and r.treewidth <= 6


def test_fixed_junkyu_n24_consistent():
    """The dropped-component fix is consistent (ipopt incumbent is a sound
    CONSISTENT certificate, so no SCIP needed)."""
    path = os.path.join(_JUNKYU, "benchmark_scaling_directions_n24_2_consistent.lcn")
    r = check_consistency_structured(_load(path), solver="ipopt", time_limit=120)
    assert r.status == CONSISTENT


@requires_scip
def test_contradictory_chain_inconsistent(tmp_path):
    """A tiny chain whose bounds cannot be jointly satisfied is INCONSISTENT."""
    p = tmp_path / "inc.lcn"
    p.write_text(
        "s1: 0.9 <= P(a) <= 1.0 ; True\n"
        "s2: 0.9 <= P(b | a) <= 1.0 ; True\n"
        "s3: 0.0 <= P(a and b) <= 0.05 ; True\n"
    )
    r = check_consistency_structured(_load(str(p)), solver="scip", time_limit=60)
    assert r.status == INCONSISTENT


def test_completeness_beats_product_witness():
    """Structured accepts a consistent instance the product witness rejects."""
    path = os.path.join(_REAL, "cancer.lcn")
    lcn = _load(path)
    # Product witness is conservative: it false-rejects this non-product model.
    assert check_consistency_product_witness(lcn, restarts=40) is False
    # Structured is complete: it certifies consistency (ipopt incumbent is sound).
    assert is_consistent_structured(lcn, solver="ipopt", time_limit=120) is True


def test_generator_structured_mode_accepts_consistent():
    """generate(consistency_mode='structured') yields genuinely consistent
    instances."""
    g = Generator(seed=7)
    with contextlib.redirect_stdout(io.StringIO()):
        inst = g.generate(num_vars=7, graph_type="polytree", num_instances=1,
                          consistency_mode="structured", verbosity=0)
    assert len(inst) == 1
    assert is_consistent_structured(inst[0], solver="ipopt", time_limit=120) is True
