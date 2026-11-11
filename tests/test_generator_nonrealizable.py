# Tests of Generator.perturb / gen_nonrealizable: controlled perturbation of a
# family-realizable instance by non-realizable sentences.

import contextlib
import io
import os
import sys

import pytest

from lcn.benchmarks.gen_nonrealizable import generate, nr_file_name
from lcn.benchmarks.generator import Generator
from lcn.core.model import LCN, SentenceType
from lcn.inference.marginal.cn.realizability import realizability_report
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.utils.common import check_consistency_product_witness_scoped

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DAG = os.path.join(_ROOT, "benchmarks", "dag_small_fr", "dag_fr_tw3_n10_1.lcn")
_TREE = os.path.join(_ROOT, "benchmarks", "tree_small_fr", "tree_fr_n10_1.lcn")


def _load(path):
    with contextlib.redirect_stdout(io.StringIO()):
        lcn = LCN()
        lcn.from_lcn(file_name=path)
    return lcn


def _roundtrip(lcn, tmp_path, name="p.lcn"):
    path = str(tmp_path / name)
    Generator.save(lcn, path)
    return _load(path), path


def _structure(lcn):
    sg = lcn.structure_graph
    return (sorted(tuple(e[:2]) for e in sg.directed_edges()),
            sorted(tuple(sorted(e[:2])) for e in sg.undirected_edges()),
            sorted((tuple(sorted(a.event1)), tuple(sorted(a.event2)),
                    tuple(sorted(a.event3)))
                   for a in lcn.independencies.get_assertions()))


@pytest.fixture(scope="module")
def dag_out():
    with contextlib.redirect_stdout(io.StringIO()):
        return Generator(seed=3).perturb(_load(_DAG), ms=(0, 1, 2, 4, 8),
                                         delta=0.1, base_name="dag",
                                         cjt_exact="structural")


def test_nested_and_counts(dag_out):
    base = _load(_DAG)
    assert set(dag_out) == {0, 1, 2, 4, 8}
    labels = {m: [s for s in lc.sentences if s.startswith("nr")]
              for m, lc in dag_out.items()}
    assert labels[0] == []
    for m, lc in dag_out.items():
        assert len(lc.sentences) == len(base.sentences) + m
        assert labels[m] == [f"nr{j}" for j in range(m)]
    big = dag_out[8]
    for m, lc in dag_out.items():        # prefixes of one sequence
        for lab in labels[m]:
            assert str(lc.sentences[lab]) == str(big.sentences[lab])
    assert set(dag_out[8].nr_report["kinds"]) == {"nr1", "nr2", "nr3"}


def test_structure_lmc_unchanged_and_nonrealizable(dag_out, tmp_path):
    base = _load(_DAG)
    ref = _structure(base)
    assert ref[0] and ref[2]                    # not vacuous
    for m, lc in dag_out.items():
        lc2, _ = _roundtrip(lc, tmp_path, f"m{m}.lcn")
        assert _structure(lc2) == ref, m
        with contextlib.redirect_stdout(io.StringIO()):
            cnv = CredalNetworkVertices.from_lcn(lc2, method="linear",
                                                 verbosity=0, cache=False)
        rep = realizability_report(lc2, cnv)
        assert rep["dag_structured"]
        bad = sorted(k for k, v in rep["realizable_in"].items() if not v)
        assert bad == [f"nr{j}" for j in range(m)], (m, bad)
        assert check_consistency_product_witness_scoped(lc2)


def test_kinds_have_the_promised_form(dag_out):
    fam = Generator._families_of(_load(_DAG))
    big = dag_out[8]
    kinds = dict(lab.split(":") for lab in big.nr_report["labels"])
    for lab, kind in kinds.items():
        s = big.sentences[lab]
        phi = set(s.phi_formula.atoms.values())
        if kind == "nr1":
            assert s.type == SentenceType.Type1 and len(phi) == 1
            assert fam[next(iter(phi))]                 # has parents
        elif kind == "nr2":
            assert s.type == SentenceType.Type2 and len(phi) == 1
            (x,) = phi
            assert set(s.psi_formula.atoms.values()) <= set(fam[x])
            assert len(fam[x]) >= 2
        else:
            assert s.type == SentenceType.Type1 and len(phi) == 2
            assert not s.tau
            u, v = sorted(phi)
            assert not any({u, v} <= ({a} | set(p)) for a, p in fam.items())
        assert 0.0 <= s.lower_prob <= s.upper_prob <= 1.0
        assert s.upper_prob - s.lower_prob <= 0.2 + 2e-6


def test_tree_has_no_mixture_kind(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        out = Generator(seed=5).perturb(_load(_TREE), ms=(4,), delta=0.2)
    assert set(out[4].nr_report["kinds"]) == {"nr1", "nr3"}  # one parent each


def test_cjt_report_and_header(dag_out, tmp_path):
    lc = dag_out[4]
    assert lc.cjt_report["structural"]
    _, path = _roundtrip(lc, tmp_path)
    head = open(path).read()
    assert "nr_m=4 nr_delta=0.1" in head and "CredalJT exactness" in head


def test_requires_dag_structured_base():
    path = os.path.join(_ROOT, "examples", "fr_counter_box.lcn")
    with pytest.raises(ValueError):
        Generator(seed=0).perturb(_load(path), ms=(1,))


def test_cli_file_names_parse(tmp_path):
    out = generate([os.path.dirname(_DAG)], "dag", sizes=[10], ms=[0, 2],
                   kinds=list(Generator.NR_KINDS), deltas=[0.05],
                   output_dir=str(tmp_path), num_bases=1, cjt_exact=None,
                   verbosity=0)
    names = sorted(os.path.basename(p) for p in out)
    assert names == ["dag_nrmix_m0_d5_n10_1.lcn", "dag_nrmix_m2_d5_n10_1.lcn"]
    assert nr_file_name("tree", ["nr1"], 4, 0.3, 20, 7) == \
        "tree_nr1_m4_d30_n20_7.lcn"
    sys.path.insert(0, os.path.join(_ROOT, "experiments"))
    from analyze_results import _group_key
    from run_experiment import _parse_instance_info
    assert _parse_instance_info(out[1]) == ("dag_nrmix_m2_d5", 10)
    rec = {"instance": out[1], "num_vars": 10}
    assert _group_key(rec, ["type", "size", "nr_kind", "nr_m", "nr_delta"]) \
        == ("dag", 10, "mix", 2, 0.05)
