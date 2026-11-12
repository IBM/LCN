# Regression: the polytree generator must produce a DAG whose undirected
# skeleton is a forest. The old invariant (no two DIRECTED paths between a
# pair) admitted undirected cycles such as a -> c <- b, a -> d <- b.

import random

import networkx as nx
import pytest

from lcn.benchmarks.generator import Generator


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("n", [6, 12, 30])
def test_polytree_skeleton_is_a_forest(seed, n):
    g = Generator(seed=seed)
    scopes = g._graph_polytree(n, max_parents=2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for sc in scopes:
        child, parents = sc[-1], sc[:-1]
        G.add_edges_from((p, child) for p in parents)
    assert nx.is_forest(G)
    assert max(len(sc) - 1 for sc in scopes) <= 2
