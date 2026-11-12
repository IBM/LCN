# Regression: a structure with a directed edge inside an undirected component
# (A -> B together with A - C - B) is not a chain graph. It used to be accepted
# because the contraction silently dropped the intra-component edge.

import contextlib
import io

from lcn.core.model import LCN


def _load(text, tmp_path):
    f = tmp_path / "x.lcn"
    f.write_text(text)
    lcn = LCN()
    with contextlib.redirect_stdout(io.StringIO()):
        lcn.from_lcn(file_name=str(f))
        lcn.build_primal_graph()
        lcn.build_structure_graph()
    return lcn


def test_intra_component_directed_edge_is_not_a_chain_graph(tmp_path):
    lcn = _load("s1: 0.2 <= P(A) <= 0.6\n"
                "s2: 0.3 <= P(B | A) <= 0.7\n"
                "s3: 0.1 <= P(A and C) <= 0.5 ; True\n"
                "s4: 0.1 <= P(C and B) <= 0.5 ; True\n", tmp_path)
    assert lcn.intra_component_directed_edges() == [("A", "B")]
    with contextlib.redirect_stdout(io.StringIO()):
        assert lcn.is_chain_graph() is False


def test_dag_is_a_chain_graph(tmp_path):
    lcn = _load("s1: 0.2 <= P(A) <= 0.6\n"
                "s2: 0.3 <= P(B | A) <= 0.7\n", tmp_path)
    assert lcn.intra_component_directed_edges() == []
    with contextlib.redirect_stdout(io.StringIO()):
        assert lcn.is_chain_graph() is True


def test_directed_edge_parallel_to_undirected_is_absorbed(tmp_path):
    """P(b | a) plus a tau = True sentence on a and b: a -> b is parallel to
    a - b, which already expresses the dependence; still a chain graph."""
    lcn = _load("s1: 0.9 <= P(a) <= 1.0 ; True\n"
                "s2: 0.9 <= P(b | a) <= 1.0 ; True\n"
                "s3: 0.0 <= P(a and b) <= 0.05 ; True\n", tmp_path)
    assert lcn.intra_component_directed_edges() == []
    with contextlib.redirect_stdout(io.StringIO()):
        assert lcn.is_chain_graph() is True
