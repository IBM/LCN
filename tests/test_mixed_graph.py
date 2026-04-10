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

import os
import pytest
import networkx as nx

from lcn.core.mixed_graph import MixedGraph


# ======================================================================
# 1. Node operations
# ======================================================================

class TestNodeOperations:

    def test_add_node(self):
        g = MixedGraph()
        g.add_node("A")
        assert g.has_node("A")
        assert g.number_of_nodes() == 1

    def test_add_node_with_attributes(self):
        g = MixedGraph()
        g.add_node("A", color="red")
        assert g.nodes["A"]["color"] == "red"

    def test_add_node_idempotent_updates_attrs(self):
        g = MixedGraph()
        g.add_node("A", color="red")
        g.add_node("A", color="blue", shape="circle")
        assert g.number_of_nodes() == 1
        assert g.nodes["A"]["color"] == "blue"
        assert g.nodes["A"]["shape"] == "circle"

    def test_add_nodes_from(self):
        g = MixedGraph()
        g.add_nodes_from(["A", "B", ("C", {"weight": 1})])
        assert g.number_of_nodes() == 3
        assert g.nodes["C"]["weight"] == 1

    def test_add_nodes_from_with_common_attrs(self):
        g = MixedGraph()
        g.add_nodes_from(["A", "B"], color="red")
        assert g.nodes["A"]["color"] == "red"
        assert g.nodes["B"]["color"] == "red"

    def test_remove_node(self):
        g = MixedGraph()
        g.add_node("A")
        g.add_node("B")
        g.remove_node("A")
        assert not g.has_node("A")
        assert g.has_node("B")
        assert g.number_of_nodes() == 1

    def test_remove_node_removes_incident_edges(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("A", "C")
        g.remove_node("A")
        assert not g.has_edge("A", "B")
        assert not g.has_edge("A", "C")

    def test_remove_nonexistent_node_raises(self):
        g = MixedGraph()
        with pytest.raises(KeyError):
            g.remove_node("Z")

    def test_remove_nodes_from(self):
        g = MixedGraph()
        g.add_nodes_from(["A", "B", "C"])
        g.remove_nodes_from(["A", "C"])
        assert g.number_of_nodes() == 1
        assert g.has_node("B")

    def test_order(self):
        g = MixedGraph()
        g.add_nodes_from(["A", "B"])
        assert g.order() == 2


# ======================================================================
# 2. Edge operations
# ======================================================================

class TestEdgeOperations:

    def test_add_directed_edge(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        assert g.has_directed_edge("A", "B")
        assert not g.has_directed_edge("B", "A")

    def test_add_undirected_edge(self):
        g = MixedGraph()
        g.add_undirected_edge("A", "B")
        assert g.has_undirected_edge("A", "B")
        assert g.has_undirected_edge("B", "A")

    def test_edge_auto_creates_nodes(self):
        g = MixedGraph()
        g.add_directed_edge("X", "Y")
        assert g.has_node("X")
        assert g.has_node("Y")

    def test_coexistence_directed_and_undirected(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B", weight=1)
        g.add_undirected_edge("A", "B", weight=2)
        assert g.has_directed_edge("A", "B")
        assert g.has_undirected_edge("A", "B")
        assert g.number_of_edges() == 2

    def test_edge_attributes(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B", color="red")
        data = g.get_edge_data("A", "B", edge_type="directed")
        assert data["color"] == "red"

    def test_get_edge_data_both_types(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B", w=1)
        g.add_undirected_edge("A", "B", w=2)
        data = g.get_edge_data("A", "B")
        assert "directed" in data
        assert "undirected" in data
        assert data["directed"]["w"] == 1
        assert data["undirected"]["w"] == 2

    def test_get_edge_data_nonexistent(self):
        g = MixedGraph()
        g.add_node("A")
        g.add_node("B")
        assert g.get_edge_data("A", "B") is None

    def test_remove_directed_edge(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.remove_directed_edge("A", "B")
        assert not g.has_directed_edge("A", "B")
        # Nodes remain
        assert g.has_node("A")
        assert g.has_node("B")

    def test_remove_undirected_edge(self):
        g = MixedGraph()
        g.add_undirected_edge("A", "B")
        g.remove_undirected_edge("A", "B")
        assert not g.has_undirected_edge("A", "B")

    def test_remove_edge_generic(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("A", "B")
        g.remove_edge("A", "B")
        assert not g.has_edge("A", "B")

    def test_remove_edge_nonexistent_raises(self):
        g = MixedGraph()
        g.add_node("A")
        g.add_node("B")
        with pytest.raises(nx.NetworkXError):
            g.remove_edge("A", "B")

    def test_has_edge(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        assert g.has_edge("A", "B")
        assert not g.has_edge("B", "A")  # directed only A->B

    def test_add_directed_edges_from(self):
        g = MixedGraph()
        g.add_directed_edges_from([("A", "B"), ("B", "C")])
        assert g.has_directed_edge("A", "B")
        assert g.has_directed_edge("B", "C")

    def test_add_undirected_edges_from(self):
        g = MixedGraph()
        g.add_undirected_edges_from([("A", "B"), ("B", "C")])
        assert g.has_undirected_edge("A", "B")
        assert g.has_undirected_edge("B", "C")

    def test_add_edges_from_with_data(self):
        g = MixedGraph()
        g.add_directed_edges_from([("A", "B", {"w": 1})])
        assert g.get_edge_data("A", "B", edge_type="directed")["w"] == 1

    def test_bidirectional_directed_not_collapsed(self):
        """Bidirectional directed edges do NOT auto-collapse into undirected."""
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_directed_edge("B", "A")
        assert g.has_directed_edge("A", "B")
        assert g.has_directed_edge("B", "A")
        assert not g.has_undirected_edge("A", "B")
        assert g.number_of_directed_edges() == 2


# ======================================================================
# 3. Edge iteration and counts
# ======================================================================

class TestEdgeViews:

    def test_directed_edges(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_directed_edge("B", "C")
        assert set(g.directed_edges()) == {("A", "B"), ("B", "C")}

    def test_undirected_edges(self):
        g = MixedGraph()
        g.add_undirected_edge("A", "B")
        edges = list(g.undirected_edges())
        assert len(edges) == 1
        assert edges[0] == ("A", "B") or edges[0] == ("B", "A")

    def test_edges_mixed(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("C", "D")
        all_edges = list(g.edges())
        assert len(all_edges) == 2
        types = {e[2] for e in all_edges}
        assert types == {"directed", "undirected"}

    def test_edges_with_data(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B", w=1)
        edges = list(g.edges(data=True))
        assert len(edges) == 1
        assert edges[0][3]["w"] == 1  # (u, v, type, data)

    def test_number_of_edges(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("C", "D")
        assert g.number_of_directed_edges() == 1
        assert g.number_of_undirected_edges() == 1
        assert g.number_of_edges() == 2
        assert g.size() == 2


# ======================================================================
# 4. Neighbor / degree queries
# ======================================================================

class TestNeighborDegree:

    def _build_sample(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_directed_edge("C", "A")
        g.add_undirected_edge("A", "D")
        return g

    def test_successors(self):
        g = self._build_sample()
        assert set(g.successors("A")) == {"B"}

    def test_predecessors(self):
        g = self._build_sample()
        assert set(g.predecessors("A")) == {"C"}

    def test_undirected_neighbors(self):
        g = self._build_sample()
        assert set(g.undirected_neighbors("A")) == {"D"}

    def test_neighbors_union(self):
        g = self._build_sample()
        assert set(g.neighbors("A")) == {"B", "C", "D"}

    def test_neighbor_query_nonexistent_raises(self):
        g = MixedGraph()
        with pytest.raises(KeyError):
            list(g.successors("Z"))
        with pytest.raises(KeyError):
            list(g.predecessors("Z"))
        with pytest.raises(KeyError):
            list(g.undirected_neighbors("Z"))
        with pytest.raises(KeyError):
            list(g.neighbors("Z"))

    def test_in_degree(self):
        g = self._build_sample()
        assert g.in_degree("A") == 1   # C->A
        assert g.in_degree("B") == 1   # A->B

    def test_out_degree(self):
        g = self._build_sample()
        assert g.out_degree("A") == 1  # A->B
        assert g.out_degree("D") == 0

    def test_undirected_degree(self):
        g = self._build_sample()
        assert g.undirected_degree("A") == 1  # A--D
        assert g.undirected_degree("D") == 1

    def test_total_degree(self):
        g = self._build_sample()
        # A: in=1(C->A) + out=1(A->B) + undirected=1(A--D) = 3
        assert g.degree("A") == 3

    def test_degree_all_nodes(self):
        g = self._build_sample()
        deg = g.degree()
        assert deg["A"] == 3
        assert deg["B"] == 1
        assert deg["C"] == 1
        assert deg["D"] == 1

    def test_degree_nonexistent_raises(self):
        g = MixedGraph()
        with pytest.raises(KeyError):
            g.degree("Z")


# ======================================================================
# 5. Algorithms
# ======================================================================

class TestAlgorithms:

    def test_has_path_undirected(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("B", "C")
        assert g.has_path("A", "C")

    def test_has_path_respect_direction(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        assert g.has_path("A", "B", respect_direction=True)
        assert not g.has_path("B", "A", respect_direction=True)

    def test_has_path_undirected_edges_bidirectional(self):
        """Undirected edges should be traversable in both directions
        even when respect_direction=True."""
        g = MixedGraph()
        g.add_undirected_edge("A", "B")
        assert g.has_path("A", "B", respect_direction=True)
        assert g.has_path("B", "A", respect_direction=True)

    def test_shortest_path(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("B", "C")
        path = g.shortest_path("A", "C")
        assert path == ["A", "B", "C"]

    def test_all_simple_paths(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("A", "C")
        g.add_undirected_edge("B", "D")
        g.add_undirected_edge("C", "D")
        paths = list(g.all_simple_paths("A", "D"))
        assert len(paths) == 2

    def test_connected_components(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("C", "D")
        g.add_node("E")
        comps = list(g.connected_components())
        assert len(comps) == 3
        comp_sets = [frozenset(c) for c in comps]
        assert frozenset({"A", "B"}) in comp_sets
        assert frozenset({"C", "D"}) in comp_sets
        assert frozenset({"E"}) in comp_sets

    def test_get_undirected_cliques_basic(self):
        """Directed edges are ignored; only undirected edges form cliques."""
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("C", "D")
        g.add_node("E")
        cliques = g.get_undirected_cliques()
        clique_sets = [frozenset(c) for c in cliques]
        # C-D form a clique of size 2; singletons are excluded
        assert frozenset({"C", "D"}) in clique_sets
        assert len(clique_sets) == 1

    def test_get_undirected_cliques_triangle(self):
        """Three mutually undirected-connected nodes form a single clique."""
        g = MixedGraph()
        g.add_undirected_edge("A", "B")
        g.add_undirected_edge("B", "C")
        g.add_undirected_edge("A", "C")
        cliques = g.get_undirected_cliques()
        clique_sets = [frozenset(c) for c in cliques]
        assert frozenset({"A", "B", "C"}) in clique_sets
        assert len(clique_sets) == 1

    def test_get_undirected_cliques_directed_does_not_complete(self):
        """A directed edge cannot complete a clique."""
        g = MixedGraph()
        g.add_undirected_edge("A", "B")
        g.add_undirected_edge("B", "C")
        g.add_directed_edge("A", "C")  # directed, not undirected
        cliques = g.get_undirected_cliques()
        clique_sets = [frozenset(c) for c in cliques]
        # Missing undirected A-C means no triangle clique
        assert frozenset({"A", "B"}) in clique_sets
        assert frozenset({"B", "C"}) in clique_sets
        assert len(clique_sets) == 2

    def test_get_undirected_cliques_no_undirected_edges(self):
        """Only directed edges: no cliques returned."""
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_directed_edge("B", "C")
        cliques = g.get_undirected_cliques()
        assert len(cliques) == 0

    def test_is_connected(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("B", "C")
        assert g.is_connected()

    def test_is_not_connected(self):
        g = MixedGraph()
        g.add_node("A")
        g.add_node("B")
        assert not g.is_connected()

    def test_subgraph(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B", w=1)
        g.add_undirected_edge("B", "C", w=2)
        g.add_directed_edge("C", "D")

        sg = g.subgraph(["A", "B", "C"])
        assert sg.number_of_nodes() == 3
        assert sg.has_directed_edge("A", "B")
        assert sg.has_undirected_edge("B", "C")
        assert not sg.has_node("D")
        # Edge attributes preserved
        assert sg.get_edge_data("A", "B", edge_type="directed")["w"] == 1

    def test_subgraph_is_independent_copy(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        sg = g.subgraph(["A", "B"])
        sg.add_node("Z")
        assert not g.has_node("Z")


# ======================================================================
# 6. Conversion / NetworkX interop
# ======================================================================

class TestConversion:

    def test_to_networkx_digraph(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("C", "D")

        dg = g.to_networkx_digraph()
        assert isinstance(dg, nx.DiGraph)
        assert dg.has_edge("A", "B")
        assert dg["A"]["B"]["edge_type"] == "directed"
        # Undirected becomes symmetric
        assert dg.has_edge("C", "D")
        assert dg.has_edge("D", "C")
        assert dg["C"]["D"]["edge_type"] == "undirected"

    def test_to_networkx_graph(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("C", "D")

        ug = g.to_networkx_graph()
        assert isinstance(ug, nx.Graph)
        assert ug.has_edge("A", "B")
        assert ug.has_edge("C", "D")

    def test_from_networkx_digraph(self):
        dg = nx.DiGraph()
        dg.add_edge("A", "B", type="directed", color="red")
        dg.add_edge("C", "D", type="undirected")
        dg.add_edge("D", "C", type="undirected")

        mg = MixedGraph.from_networkx(dg)
        assert mg.has_directed_edge("A", "B")
        assert mg.has_undirected_edge("C", "D")
        # Symmetric undirected pair collapsed into one
        assert mg.number_of_undirected_edges() == 1
        assert not mg.has_directed_edge("C", "D")

    def test_from_networkx_graph(self):
        ug = nx.Graph()
        ug.add_edge("A", "B")
        ug.add_edge("C", "D")

        mg = MixedGraph.from_networkx(ug)
        assert mg.has_undirected_edge("A", "B")
        assert mg.has_undirected_edge("C", "D")
        assert mg.number_of_directed_edges() == 0

    def test_from_networkx_preserves_node_attrs(self):
        dg = nx.DiGraph()
        dg.add_node("A", color="blue")
        dg.add_edge("A", "B", type="directed")
        mg = MixedGraph.from_networkx(dg)
        assert mg.nodes["A"]["color"] == "blue"

    def test_from_networkx_strips_type_attr_from_edges(self):
        """The edge_type_attr ('type') should be consumed, not stored."""
        dg = nx.DiGraph()
        dg.add_edge("A", "B", type="directed", w=1)
        mg = MixedGraph.from_networkx(dg)
        data = mg.get_edge_data("A", "B", edge_type="directed")
        assert "type" not in data
        assert data["w"] == 1

    def test_copy(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B", w=1)
        g.add_undirected_edge("C", "D")
        g.add_node("E", label="e")

        c = g.copy()
        assert c == g

        # Independence
        c.add_node("Z")
        assert not g.has_node("Z")

    def test_is_directed(self):
        g = MixedGraph()
        assert not g.is_directed()

    def test_is_multigraph(self):
        g = MixedGraph()
        assert not g.is_multigraph()

    def test_roundtrip_digraph(self):
        """MixedGraph -> DiGraph -> MixedGraph preserves structure."""
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("C", "D")
        g.add_node("E")

        dg = g.to_networkx_digraph()
        mg2 = MixedGraph.from_networkx(dg, edge_type_attr="edge_type")
        assert mg2.has_directed_edge("A", "B")
        assert mg2.has_undirected_edge("C", "D")
        assert mg2.has_node("E")
        assert mg2.number_of_directed_edges() == g.number_of_directed_edges()
        assert mg2.number_of_undirected_edges() == g.number_of_undirected_edges()


# ======================================================================
# 7. Integration — load asia.lcn, build structure graph, convert
# ======================================================================

class TestIntegration:

    @pytest.fixture
    def asia_structure_graph(self):
        from lcn.core.model import LCN
        lcn_model = LCN()
        examples_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "examples"
        )
        lcn_model.from_lcn(os.path.join(examples_dir, "asia.lcn"))
        return lcn_model.build_structure_graph()

    def test_convert_structure_graph(self, asia_structure_graph):
        """from_networkx handles the structure graph produced by
        build_structure_graph()."""
        G = asia_structure_graph
        assert isinstance(G, nx.DiGraph)

        mg = MixedGraph.from_networkx(G)
        assert mg.number_of_nodes() == G.number_of_nodes()

        # Every node in the original should be in the mixed graph
        for n in G.nodes():
            assert mg.has_node(n)

    def test_structure_graph_edge_types(self, asia_structure_graph):
        """Directed and undirected edges are correctly categorized."""
        G = asia_structure_graph
        mg = MixedGraph.from_networkx(G)

        # The structure graph should have both directed and undirected edges
        assert mg.number_of_directed_edges() > 0
        assert mg.number_of_undirected_edges() > 0

    def test_structure_graph_connectivity(self, asia_structure_graph):
        G = asia_structure_graph
        mg = MixedGraph.from_networkx(G)
        # The asia model should be connected
        assert mg.is_connected()


# ======================================================================
# 8. Protocol / dunder methods
# ======================================================================

class TestProtocol:

    def test_contains(self):
        g = MixedGraph()
        g.add_node("A")
        assert "A" in g
        assert "Z" not in g

    def test_len(self):
        g = MixedGraph()
        g.add_nodes_from(["A", "B", "C"])
        assert len(g) == 3

    def test_iter(self):
        g = MixedGraph()
        g.add_nodes_from(["A", "B", "C"])
        assert set(g) == {"A", "B", "C"}

    def test_getitem(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B", w=1)
        g.add_undirected_edge("A", "C")
        nbrs = g["A"]
        assert "B" in nbrs
        assert "C" in nbrs

    def test_getitem_nonexistent_raises(self):
        g = MixedGraph()
        with pytest.raises(KeyError):
            g["Z"]

    def test_str(self):
        g = MixedGraph()
        g.add_directed_edge("A", "B")
        g.add_undirected_edge("C", "D")
        s = str(g)
        assert "4 nodes" in s
        assert "1 directed" in s
        assert "1 undirected" in s

    def test_repr(self):
        g = MixedGraph()
        r = repr(g)
        assert r.startswith("MixedGraph(")

    def test_eq(self):
        g1 = MixedGraph()
        g1.add_directed_edge("A", "B")
        g1.add_undirected_edge("C", "D")

        g2 = MixedGraph()
        g2.add_directed_edge("A", "B")
        g2.add_undirected_edge("C", "D")

        assert g1 == g2

    def test_eq_not_equal(self):
        g1 = MixedGraph()
        g1.add_directed_edge("A", "B")
        g2 = MixedGraph()
        g2.add_undirected_edge("A", "B")
        assert g1 != g2

    def test_eq_not_mixed_graph(self):
        g = MixedGraph()
        assert g != "not a graph"


# ======================================================================
# 9. Old MixedGraph in utils.py is still importable
# ======================================================================

class TestOldMixedGraph:

    def test_old_mixed_graph_importable(self):
        from lcn.utils import MixedGraph as OldMixedGraph
        g = OldMixedGraph()
        g.add_node("A")
        assert "A" in g.node
