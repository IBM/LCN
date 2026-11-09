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

# Static graph drawing (matplotlib + networkx) for LCN graphs: the primal
# graph (an ``nx.DiGraph``) and the structure graph (a ``MixedGraph``).


import networkx as nx

from lcn.core.mixed_graph import MixedGraph

# Node ``shape`` attribute (as set by LCN.build_primal_graph) -> matplotlib marker
_SHAPE_TO_MARKER = {"ellipse": "o", "circle": "o", "rectangle": "s", "box": "s"}


def _tint(color, amount: float = 0.75):
    """Opaque light version of ``color`` (blend towards white)."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _layout(G: nx.Graph, layout: str, seed: int) -> dict:
    if layout == "spring":
        return nx.spring_layout(G, seed=seed)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G)
    if layout == "circular":
        return nx.circular_layout(G)
    if layout == "shell":
        return nx.shell_layout(G)
    if layout == "multipartite":
        # Layered top-down drawing; only meaningful for DAGs.
        H = G.copy()
        for i, layer in enumerate(nx.topological_generations(H)):
            for n in layer:
                H.nodes[n]["_layer"] = i
        pos = nx.multipartite_layout(H, subset_key="_layer", align="horizontal")
        return {n: (x, -y) for n, (x, y) in pos.items()}
    raise ValueError(f"Unknown layout: {layout!r}")


def plot_graph(
    G,
    ax=None,
    layout: str = "kamada_kawai",
    pos: dict | None = None,
    seed: int = 42,
    label_attr: str = "label",
    node_size: int = 900,
    font_size: int = 9,
    figsize: tuple[float, float] = (8, 6),
    title: str | None = None,
):
    """Draw an LCN graph with matplotlib.

    Accepts either an ``nx.DiGraph`` (e.g. ``LCN.build_primal_graph()``) or a
    ``MixedGraph`` (e.g. ``LCN.build_structure_graph()``). Node/edge ``color``
    and node ``shape`` attributes are honoured when present.

    - ``nx.DiGraph``: every edge is drawn with an arrow; reciprocal pairs
      (u->v and v->u) are drawn as two slightly curved arcs so both are visible.
    - ``MixedGraph``: directed edges are drawn with arrows, undirected edges as
      plain lines without arrowheads.

    Args:
        G: an ``nx.DiGraph``, ``nx.Graph`` or ``MixedGraph``.
        ax: matplotlib Axes to draw on (a new figure is created if None).
        layout: 'spring', 'kamada_kawai', 'circular', 'shell' or
            'multipartite' (layered, DAGs only). Ignored if ``pos`` is given.
        pos: optional precomputed ``{node: (x, y)}`` positions.
        seed: random seed for the spring layout.
        label_attr: node attribute used as label (falls back to the node id).

    Returns:
        The matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if isinstance(G, MixedGraph):
        directed = [(u, v, d) for u, v, d in G.directed_edges(data=True)]
        undirected = [(u, v, d) for u, v, d in G.undirected_edges(data=True)]
        nodes = dict(G.nodes)
        base = G.to_networkx_graph()
    else:
        if G.is_directed():
            directed = list(G.edges(data=True))
            undirected = []
        else:
            directed = []
            undirected = list(G.edges(data=True))
        nodes = dict(G.nodes(data=True))
        base = G

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if pos is None:
        pos = _layout(base, layout, seed)

    # Node styling: atoms (and anything not a rectangle) are opaque markers;
    # formula nodes (shape='rectangle') are text boxes sized to their label.
    labels = {}
    for n, attr in nodes.items():
        lab = attr.get(label_attr, n)
        labels[n] = str(n) if lab in (None, "") else str(lab)

    def _is_box(n):
        return _SHAPE_TO_MARKER.get(nodes[n].get("shape", "ellipse"), "o") == "s"

    # Per-node "size" (marker area, points^2) used to clip edges at node borders.
    # networkx shortens each edge end by sqrt(size)/2 points. Boxes are wide
    # and short, so use a compromise between their half-height and half-width.
    def _text_width(n):
        return 0.62 * font_size * len(labels[n])

    def _circle_size(n):
        return max(node_size, (_text_width(n) + 12) ** 2)

    def _clip_size(n):
        if _is_box(n):
            return (2.2 * font_size + 0.3 * _text_width(n)) ** 2
        return _circle_size(n)

    # Edges first so nodes are drawn on top of them.
    edge_set = {(u, v) for u, v, _ in directed}
    for u, v, d in directed:
        # Curve reciprocal pairs (u->v and v->u) so both directions are visible.
        rad = 0.15 if (v, u) in edge_set else 0.0
        nx.draw_networkx_edges(
            base, pos, ax=ax, edgelist=[(u, v)], nodelist=[u, v],
            node_size=[_clip_size(u), _clip_size(v)],
            edge_color=d.get("color", "black"), width=1.3,
            arrows=True, arrowstyle="-|>", arrowsize=14,
            connectionstyle=f"arc3,rad={rad}",
        )
    for u, v, d in undirected:
        # arrows=True with a plain '-' style gives a patch clipped at the nodes.
        nx.draw_networkx_edges(
            base, pos, ax=ax, edgelist=[(u, v)], nodelist=[u, v],
            node_size=[_clip_size(u), _clip_size(v)],
            edge_color=d.get("color", "black"), width=1.8,
            arrows=True, arrowstyle="-",
        )

    for n, attr in nodes.items():
        color = attr.get("color", "tab:blue")
        face = _tint(color)
        x, y = pos[n]
        if _is_box(n):
            ax.text(x, y, labels[n], ha="center", va="center", fontsize=font_size,
                    zorder=4, bbox={"boxstyle": "round,pad=0.3", "fc": face,
                                        "ec": color, "lw": 1.3})
        else:
            ax.scatter([x], [y], s=_circle_size(n), c=[face], edgecolors=[color],
                       linewidths=1.5, zorder=3)
            ax.text(x, y, labels[n], ha="center", va="center",
                    fontsize=font_size, zorder=4)

    if title:
        ax.set_title(title)
    ax.set_axis_off()
    ax.margins(0.1)
    return ax
