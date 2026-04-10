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

# A MixedGraph class supporting both directed and undirected edges
# using composition over two internal NetworkX graphs.

import copy
import networkx as nx


class MixedGraph:
    """
    A graph that supports both directed and undirected edges.

    Uses composition: internally wraps an ``nx.DiGraph`` (for directed edges)
    and an ``nx.Graph`` (for undirected edges) with a shared node set.
    A node pair can have both a directed and an undirected edge simultaneously.
    """

    def __init__(self, **graph_attr):
        self._nodes = {}           # node_id -> {attr_dict}
        self._directed = nx.DiGraph()
        self._undirected = nx.Graph()
        self.graph = graph_attr

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, n, **attr):
        """Add a single node, optionally with attributes."""
        if n in self._nodes:
            self._nodes[n].update(attr)
        else:
            self._nodes[n] = attr
        self._directed.add_node(n)
        self._undirected.add_node(n)

    def add_nodes_from(self, nodes, **attr):
        """Add nodes from an iterable. Each element may be a node id or
        a ``(node, attr_dict)`` tuple."""
        for item in nodes:
            if isinstance(item, tuple) and len(item) == 2:
                n, node_attr = item
                merged = {**attr, **node_attr}
                self.add_node(n, **merged)
            else:
                self.add_node(item, **attr)

    def remove_node(self, n):
        """Remove node *n* and all its incident edges."""
        if n not in self._nodes:
            raise KeyError(n)
        del self._nodes[n]
        self._directed.remove_node(n)
        self._undirected.remove_node(n)

    def remove_nodes_from(self, nodes):
        """Remove multiple nodes."""
        for n in list(nodes):
            self.remove_node(n)

    def has_node(self, n):
        return n in self._nodes

    def number_of_nodes(self):
        return len(self._nodes)

    def order(self):
        return self.number_of_nodes()

    @property
    def nodes(self):
        """Dict-like view of nodes and their attributes."""
        return self._nodes

    def get_nodes(self) -> list:
        """Return a list of nodes."""
        return list(self._nodes.keys())
    # ------------------------------------------------------------------
    # Edge management — directed
    # ------------------------------------------------------------------

    def add_directed_edge(self, u, v, **attr):
        """Add a directed edge u -> v (auto-creates missing nodes)."""
        if u not in self._nodes:
            self.add_node(u)
        if v not in self._nodes:
            self.add_node(v)
        self._directed.add_edge(u, v, **attr)

    def add_directed_edges_from(self, edges, **attr):
        """Add directed edges from an iterable of ``(u, v)`` or
        ``(u, v, data)`` tuples."""
        for e in edges:
            if len(e) == 3:
                u, v, d = e
                merged = {**attr, **d}
                self.add_directed_edge(u, v, **merged)
            else:
                u, v = e
                self.add_directed_edge(u, v, **attr)

    def remove_directed_edge(self, u, v):
        """Remove the directed edge u -> v."""
        self._directed.remove_edge(u, v)

    # ------------------------------------------------------------------
    # Edge management — undirected
    # ------------------------------------------------------------------

    def add_undirected_edge(self, u, v, **attr):
        """Add an undirected edge between u and v (auto-creates nodes)."""
        if u not in self._nodes:
            self.add_node(u)
        if v not in self._nodes:
            self.add_node(v)
        self._undirected.add_edge(u, v, **attr)

    def add_undirected_edges_from(self, edges, **attr):
        """Add undirected edges from an iterable of ``(u, v)`` or
        ``(u, v, data)`` tuples."""
        for e in edges:
            if len(e) == 3:
                u, v, d = e
                merged = {**attr, **d}
                self.add_undirected_edge(u, v, **merged)
            else:
                u, v = e
                self.add_undirected_edge(u, v, **attr)

    def remove_undirected_edge(self, u, v):
        """Remove the undirected edge between u and v."""
        self._undirected.remove_edge(u, v)

    # ------------------------------------------------------------------
    # Edge management — generic removal and queries
    # ------------------------------------------------------------------

    def remove_edge(self, u, v):
        """Remove any edge (directed or undirected) between u and v."""
        removed = False
        if self._directed.has_edge(u, v):
            self._directed.remove_edge(u, v)
            removed = True
        if self._undirected.has_edge(u, v):
            self._undirected.remove_edge(u, v)
            removed = True
        if not removed:
            raise nx.NetworkXError(
                f"The edge {u}-{v} is not in the graph."
            )

    def has_directed_edge(self, u, v):
        return self._directed.has_edge(u, v)

    def has_undirected_edge(self, u, v):
        return self._undirected.has_edge(u, v)

    def has_edge(self, u, v):
        return self.has_directed_edge(u, v) or self.has_undirected_edge(u, v)

    def get_edge_data(self, u, v, edge_type=None):
        """Return edge attribute dict(s).

        Parameters
        ----------
        edge_type : str or None
            ``'directed'``, ``'undirected'``, or ``None`` (returns a dict
            with both types if they exist).
        """
        if edge_type == "directed":
            return self._directed.get_edge_data(u, v)
        if edge_type == "undirected":
            return self._undirected.get_edge_data(u, v)
        result = {}
        d = self._directed.get_edge_data(u, v)
        if d is not None:
            result["directed"] = d
        ud = self._undirected.get_edge_data(u, v)
        if ud is not None:
            result["undirected"] = ud
        return result if result else None

    # ------------------------------------------------------------------
    # Edge iteration / counts
    # ------------------------------------------------------------------

    def directed_edges(self, data=False):
        """Iterate over directed edges."""
        return self._directed.edges(data=data)

    def undirected_edges(self, data=False):
        """Iterate over undirected edges."""
        return self._undirected.edges(data=data)

    def edges(self, data=False):
        """Iterate over all edges, yielding ``(u, v, 'directed'|'undirected'[, data])``."""
        for e in self._directed.edges(data=data):
            if data:
                yield (e[0], e[1], "directed", e[2])
            else:
                yield (e[0], e[1], "directed")
        for e in self._undirected.edges(data=data):
            if data:
                yield (e[0], e[1], "undirected", e[2])
            else:
                yield (e[0], e[1], "undirected")

    def number_of_directed_edges(self):
        return self._directed.number_of_edges()

    def number_of_undirected_edges(self):
        return self._undirected.number_of_edges()

    def number_of_edges(self):
        return self.number_of_directed_edges() + self.number_of_undirected_edges()

    def size(self):
        return self.number_of_edges()

    # ------------------------------------------------------------------
    # Neighbor queries
    # ------------------------------------------------------------------

    def successors(self, n):
        """Nodes reachable via a directed edge from *n*."""
        if n not in self._nodes:
            raise KeyError(n)
        return iter(self._directed.successors(n))

    def predecessors(self, n):
        """Nodes with a directed edge into *n*."""
        if n not in self._nodes:
            raise KeyError(n)
        return iter(self._directed.predecessors(n))

    def undirected_neighbors(self, n):
        """Nodes connected to *n* via an undirected edge."""
        if n not in self._nodes:
            raise KeyError(n)
        return iter(self._undirected.neighbors(n))

    def neighbors(self, n):
        """All neighbors (union of successors, predecessors, undirected)."""
        if n not in self._nodes:
            raise KeyError(n)
        nbrs = set(self._directed.successors(n))
        nbrs.update(self._directed.predecessors(n))
        nbrs.update(self._undirected.neighbors(n))
        return iter(nbrs)

    # ------------------------------------------------------------------
    # Degree
    # ------------------------------------------------------------------

    def in_degree(self, n=None):
        """Directed in-degree. If *n* is None, return dict for all nodes."""
        if n is not None:
            if n not in self._nodes:
                raise KeyError(n)
            return self._directed.in_degree(n)
        return {v: self._directed.in_degree(v) for v in self._nodes}

    def out_degree(self, n=None):
        """Directed out-degree. If *n* is None, return dict for all nodes."""
        if n is not None:
            if n not in self._nodes:
                raise KeyError(n)
            return self._directed.out_degree(n)
        return {v: self._directed.out_degree(v) for v in self._nodes}

    def undirected_degree(self, n=None):
        """Undirected degree. If *n* is None, return dict for all nodes."""
        if n is not None:
            if n not in self._nodes:
                raise KeyError(n)
            return self._undirected.degree(n)
        return {v: self._undirected.degree(v) for v in self._nodes}

    def degree(self, n=None):
        """Total degree (directed in + out + undirected).
        If *n* is None, return dict for all nodes."""
        if n is not None:
            if n not in self._nodes:
                raise KeyError(n)
            return (self._directed.in_degree(n)
                    + self._directed.out_degree(n)
                    + self._undirected.degree(n))
        return {v: self.degree(v) for v in self._nodes}

    # ------------------------------------------------------------------
    # Algorithms (delegate to NetworkX)
    # ------------------------------------------------------------------

    def _as_undirected(self):
        """Return a plain ``nx.Graph`` treating every edge as undirected."""
        G = nx.Graph()
        G.add_nodes_from(self._nodes)
        G.add_edges_from(self._directed.edges())
        G.add_edges_from(self._undirected.edges())
        return G

    def _as_directed(self):
        """Return a plain ``nx.DiGraph`` respecting direction for directed
        edges and adding both orientations for undirected edges."""
        G = nx.DiGraph()
        G.add_nodes_from(self._nodes)
        G.add_edges_from(self._directed.edges())
        for u, v in self._undirected.edges():
            G.add_edge(u, v)
            G.add_edge(v, u)
        return G

    def has_path(self, source, target, respect_direction=False):
        G = self._as_directed() if respect_direction else self._as_undirected()
        return nx.has_path(G, source, target)

    def shortest_path(self, source, target, respect_direction=False):
        G = self._as_directed() if respect_direction else self._as_undirected()
        return nx.shortest_path(G, source, target)

    def all_simple_paths(self, source, target, respect_direction=False):
        G = self._as_directed() if respect_direction else self._as_undirected()
        return nx.all_simple_paths(G, source, target)

    def connected_components(self):
        """Return connected components treating all edges as undirected."""
        return nx.connected_components(self._as_undirected())

    def get_undirected_cliques(self):
        """Return the maximal cliques in the undirected subgraph.

        An undirected clique is a subgraph in which every pair of nodes
        is connected by an undirected edge.  Only undirected edges are
        considered; directed edges are ignored.

        Returns
        -------
        list[list]
            A list of maximal cliques, where each clique is a list of
            node ids.
        """
        return [list(c) for c in nx.find_cliques(self._undirected) if len(c) > 1]

    def is_connected(self):
        G = self._as_undirected()
        return nx.is_connected(G)

    def subgraph(self, nodes):
        """Return a new MixedGraph induced on the given node subset."""
        node_set = set(nodes)
        sg = MixedGraph(**copy.deepcopy(self.graph))
        for n in node_set:
            if n in self._nodes:
                sg.add_node(n, **copy.deepcopy(self._nodes[n]))
        for u, v, d in self._directed.edges(data=True):
            if u in node_set and v in node_set:
                sg.add_directed_edge(u, v, **copy.deepcopy(d))
        for u, v, d in self._undirected.edges(data=True):
            if u in node_set and v in node_set:
                sg.add_undirected_edge(u, v, **copy.deepcopy(d))
        return sg

    # ------------------------------------------------------------------
    # Conversion / NetworkX interop
    # ------------------------------------------------------------------

    def to_networkx_digraph(self):
        """Convert to ``nx.DiGraph``. Undirected edges become symmetric
        directed pairs with attribute ``edge_type='undirected'``."""
        G = nx.DiGraph()
        for n, attr in self._nodes.items():
            G.add_node(n, **attr)
        for u, v, d in self._directed.edges(data=True):
            G.add_edge(u, v, edge_type="directed", **d)
        for u, v, d in self._undirected.edges(data=True):
            G.add_edge(u, v, edge_type="undirected", **d)
            G.add_edge(v, u, edge_type="undirected", **d)
        return G

    def to_networkx_graph(self):
        """Convert to ``nx.Graph``. All edges become undirected;
        original direction info stored in ``edge_type`` attribute."""
        G = nx.Graph()
        for n, attr in self._nodes.items():
            G.add_node(n, **attr)
        for u, v, d in self._directed.edges(data=True):
            G.add_edge(u, v, edge_type="directed", **d)
        for u, v, d in self._undirected.edges(data=True):
            G.add_edge(u, v, edge_type="undirected", **d)
        return G

    @classmethod
    def from_networkx(cls, G, edge_type_attr="type"):
        """Construct a MixedGraph from a NetworkX graph.

        Edges whose ``edge_type_attr`` attribute equals ``'undirected'`` are
        added as undirected edges.  For ``nx.DiGraph`` inputs, symmetric
        directed pairs (u->v *and* v->u both with ``'undirected'``) are
        collapsed into a single undirected edge.  All other edges are added
        as directed.

        Parameters
        ----------
        G : nx.Graph or nx.DiGraph
        edge_type_attr : str
            The edge attribute key that distinguishes directed from undirected.
        """
        mg = cls()

        # Add nodes with attributes
        for n, attr in G.nodes(data=True):
            mg.add_node(n, **attr)

        if isinstance(G, nx.DiGraph):
            # Track which pairs have been added as undirected
            undirected_added = set()
            for u, v, d in G.edges(data=True):
                d = dict(d)  # copy so we don't mutate original
                etype = d.pop(edge_type_attr, None)
                if etype == "undirected":
                    pair = (min(u, v), max(u, v)) if u != v else (u, v)
                    if pair not in undirected_added:
                        mg.add_undirected_edge(u, v, **d)
                        undirected_added.add(pair)
                else:
                    mg.add_directed_edge(u, v, **d)
        else:
            # Plain Graph — all edges undirected
            for u, v, d in G.edges(data=True):
                d = dict(d)
                d.pop(edge_type_attr, None)
                mg.add_undirected_edge(u, v, **d)

        return mg

    def copy(self):
        """Return a deep copy of this MixedGraph."""
        mg = MixedGraph(**copy.deepcopy(self.graph))
        for n, attr in self._nodes.items():
            mg.add_node(n, **copy.deepcopy(attr))
        for u, v, d in self._directed.edges(data=True):
            mg.add_directed_edge(u, v, **copy.deepcopy(d))
        for u, v, d in self._undirected.edges(data=True):
            mg.add_undirected_edge(u, v, **copy.deepcopy(d))
        return mg

    def is_directed(self):
        """A mixed graph is not purely directed."""
        return False

    def is_multigraph(self):
        return False

    # ------------------------------------------------------------------
    # Dunder / protocol methods
    # ------------------------------------------------------------------

    def __contains__(self, n):
        return n in self._nodes

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __getitem__(self, n):
        """Return a dict of all neighbors of *n* with edge attribute data."""
        if n not in self._nodes:
            raise KeyError(n)
        result = {}
        for v in self._directed.successors(n):
            result[v] = {"directed": dict(self._directed[n][v])}
        for v in self._directed.predecessors(n):
            entry = result.setdefault(v, {})
            entry["directed_in"] = dict(self._directed[v][n])
        for v in self._undirected.neighbors(n):
            entry = result.setdefault(v, {})
            entry["undirected"] = dict(self._undirected[n][v])
        return result

    def __str__(self):
        return (
            f"MixedGraph with {self.number_of_nodes()} nodes, "
            f"{self.number_of_directed_edges()} directed edges, "
            f"{self.number_of_undirected_edges()} undirected edges"
        )

    def __repr__(self):
        return (
            f"MixedGraph(nodes={self.number_of_nodes()}, "
            f"directed_edges={self.number_of_directed_edges()}, "
            f"undirected_edges={self.number_of_undirected_edges()})"
        )

    def __eq__(self, other):
        if not isinstance(other, MixedGraph):
            return NotImplemented
        return (
            self._nodes == other._nodes
            and set(self._directed.edges()) == set(other._directed.edges())
            and set(self._undirected.edges()) == set(other._undirected.edges())
        )
