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

# The tau flag after ';' in an .lcn sentence must be parsed literally:
# "; False" is tau = False (it used to be bool("False") = True).

import contextlib
import io

import pytest

from lcn.core.model import LCN


def _tau(tmp_path, flag):
    f = tmp_path / "t.lcn"
    f.write_text(f"s1: 0.5 <= P(A or B) <= 0.7{flag}\n")
    lcn = LCN()
    with contextlib.redirect_stdout(io.StringIO()):
        lcn.from_lcn(file_name=str(f))
    return lcn.sentences["s1"].tau


@pytest.mark.parametrize("flag,expected", [
    ("", False), (" ; True", True), (" ; False", False), ("; true", True),
    (" ; false", False), (" ;", False)])
def test_tau_flag(tmp_path, flag, expected):
    assert _tau(tmp_path, flag) is expected


def test_bad_tau_flag_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _tau(tmp_path, " ; yes")
