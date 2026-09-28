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

# Symbolic chain-graph factorization for LCNs.
#
# An LCN whose structure is a chain graph factorizes as a product of local
# conditional factors P(child | parents), one per family of the chain graph.
# This module builds that *symbolic* factorization: for each family it records
# the child, its parents and the LCN sentences that constrain the family, plus
# the flattened atom scope. It does NOT enumerate interpretations and does NOT
# compute any probability intervals — those local credal sets are computed by
# `CredalNetwork` (via `local_credal_sets.LocalCredalSetSolver`).

# Local
import networkx as nx

from lcn.core.model import LCN


def _flatten(node: str) -> list:
    """Split a (possibly compound "A-B") node name into its atoms."""
    return node.split("-") if "-" in node else [node]


class ChainGraphFactorization:
    """
    The symbolic chain-graph factorization of an LCN.

    For a chain graph, the joint factorizes as a product of local conditional
    factors P(child | parents) over the families of the simplified structure
    graph. This class enumerates those families and exposes one symbolic factor
    descriptor per family. Intervals (local credal sets) are NOT computed here.
    """

    def __init__(self, lcn: LCN):
        self.lcn = lcn
        self.factors = []

    def build(self, verbosity: int = 0, merge_budget: int = 1):
        """
        Build the symbolic factors P(child | parents), one per family.

        The LCN must already be postprocessed (structure graph and families
        computed). Each returned factor is a dict with:
            - "child":       child node name (possibly compound "A-B")
            - "parents":     list of parent node names (as in the chain graph)
            - "parents_lst": flattened parent atoms (compound nodes split on "-")
            - "child_lst":   flattened child atoms
            - "scope":       full flattened atom scope (child atoms then parents)
            - "sentences":   sentence ids attached to the family

        Args:
            verbosity: int
                Verbosity level (0 is silent).
            merge_budget: int
                Scheme D2 (docs/tighter_approximation.tex): the maximum flattened
                scope size of a merged super-family. ``1`` (the default) performs
                no merging -- the factors are exactly the LCN families. With a
                larger budget, adjacent families whose combined flattened scope is
                at most ``merge_budget`` are merged into a single joint factor, so
                cross-family LCN constraints act jointly (tighter local credal
                sets). A budget at least the total atom count collapses everything
                into one family, i.e. exact inference.

        Returns:
            A list of symbolic factor descriptors (one per family).
        """
        # Ensure that the LCN has been postprocessed (structure, families)
        assert self.lcn.structure_graph is not None
        assert self.lcn.simplified_structure_graph is not None
        assert self.lcn.families is not None

        if merge_budget is not None and merge_budget > 1:
            families = self._merge_families(
                self.lcn.families, merge_budget, verbosity)
        else:
            families = self.lcn.families

        self.factors = []
        for family in families:
            child = family["child"]
            parents = family["parents"]
            sentences = family["sentences"]

            parents_lst = []
            scope = [child] if "-" not in child else child.split("-")
            for par in parents:
                if "-" in par:
                    scope += par.split("-")
                    parents_lst += par.split("-")
                else:
                    scope += [par]
                    parents_lst += [par]

            child_lst = child.split("-") if "-" in child else [child]

            if verbosity > 0:
                print(f"Symbolic factor: {child} <-- {parents}")
                print(f"  parents_lst: {parents_lst}")
                print(f"  scope: {scope}")

            self.factors.append({
                "child": child,
                "parents": parents,
                "parents_lst": parents_lst,
                "child_lst": child_lst,
                "scope": scope,
                "sentences": sentences,
            })

        return self.factors

    def _merge_families(self, families: list, merge_budget: int,
                        verbosity: int = 0) -> list:
        """
        Scheme D2: greedily merge adjacent chain-graph families into joint
        super-families whose flattened scope is at most ``merge_budget``.

        The merge is a sequence of arc contractions on the family DAG (one node
        per family, an arc ``u -> v`` when ``child(u)`` is a parent of ``v``).
        Each contraction folds two families into one whose child node is the
        "-"-joined union of their child atoms, whose parents are the external
        parents only (parents not produced inside the group), and whose attached
        sentences are recomputed as every LCN sentence with scope inside the
        merged scope -- exactly the rule ``process_chain_graph`` uses, so a
        merged scope (a superset of each member scope) can only *gain* sentences.

        Contractions that would create a cycle are skipped, so the result stays
        a DAG. ``self.lcn.families`` is never mutated. Returns a list of family
        dicts of the same shape ``{child, parents, sentences}`` that ``build``
        consumes; with ``merge_budget <= 1`` this method is not called and the
        original families are used verbatim.
        """
        # Family DAG: node per family child-name; arc child(u) -> v when child(u)
        # is a parent of v. Each node carries its current child-atom set, its
        # external parent node-names, and the flattened scope.
        groups = {}
        for fam in families:
            child = fam["child"]
            groups[child] = {
                "child_atoms": set(_flatten(child)),
                "parents": set(fam["parents"]),
            }

        def scope_atoms(node_key):
            g = groups[node_key]
            atoms = set(g["child_atoms"])
            for p in g["parents"]:
                atoms.update(_flatten(p))
            return atoms

        # Build the arc set (parent-node -> child-node) over the *current* keys.
        def build_arcs():
            arcs = []
            for v, g in groups.items():
                for p in g["parents"]:
                    if p in groups:  # p names another family's child node
                        arcs.append((p, v))
            return arcs

        # Greedy contraction. Process candidate arcs in a deterministic order
        # (sorted by the pair of node names) so the result is reproducible across
        # serial and parallel builds.
        changed = True
        while changed:
            changed = False
            for u, v in sorted(build_arcs()):
                if u not in groups or v not in groups or u == v:
                    continue
                merged_atoms = scope_atoms(u) | scope_atoms(v)
                if len(merged_atoms) > merge_budget:
                    continue
                # Tentatively contract u into v and check the result stays a DAG.
                new_child_atoms = groups[u]["child_atoms"] | groups[v]["child_atoms"]
                new_key = "-".join(sorted(new_child_atoms))
                # External parents: parents of u or v that are not now-internal
                # child nodes of the merged group.
                internal = {u, v, new_key}
                new_parents = set()
                for src in (u, v):
                    for p in groups[src]["parents"]:
                        if p not in internal and p not in new_child_atoms:
                            new_parents.add(p)

                # Rewire: any other family that had u or v as a parent now points
                # at new_key instead.
                trial = {k: {"child_atoms": set(val["child_atoms"]),
                             "parents": set(val["parents"])}
                         for k, val in groups.items() if k not in (u, v)}
                trial[new_key] = {"child_atoms": new_child_atoms,
                                  "parents": new_parents}
                for k, val in trial.items():
                    if k == new_key:
                        continue
                    if u in val["parents"] or v in val["parents"]:
                        val["parents"].discard(u)
                        val["parents"].discard(v)
                        val["parents"].add(new_key)

                # Acyclicity check on the trial family DAG.
                dg = nx.DiGraph()
                dg.add_nodes_from(trial.keys())
                for k, val in trial.items():
                    for p in val["parents"]:
                        if p in trial:
                            dg.add_edge(p, k)
                if not nx.is_directed_acyclic_graph(dg):
                    continue

                groups = trial
                changed = True
                if verbosity > 0:
                    print(f"[D2] merged {u} + {v} -> {new_key} "
                          f"(scope size {len(merged_atoms)})")
                break  # restart the scan after a structural change

        # Emit family dicts; recompute attached sentences over the merged scope.
        merged_families = []
        for key in sorted(groups.keys()):
            g = groups[key]
            scope = scope_atoms(key)
            sentences = []
            for sid, s in self.lcn.sentences.items():
                if set(s.get_atoms().keys()).issubset(scope):
                    sentences.append(sid)
            merged_families.append({
                "child": key,
                "parents": sorted(g["parents"]),
                "sentences": sentences,
            })
        return merged_families


if __name__ == "__main__":

    # Load the LCN
    # file_name = "examples/alarm.lcn"
    # file_name = "benchmarks/chain/chain_n20_1.lcn"
    file_name = "examples/smokers.lcn"
    lcn_model = LCN()
    lcn_model.from_lcn(file_name=file_name)
    print(lcn_model)

    # Factorize (symbolic)
    lcn_model.build_primal_graph(formula_labels=True)
    lcn_model.build_structure_graph()

    # check if the LCN is a chain graph
    ok = lcn_model.is_chain_graph()
    print(f"Is the LCN a chain graph? {ok}")

    # get the families of each node in the chain graph
    families = lcn_model.process_chain_graph()
    print("Families of each node:")
    for family in families:
        node = family["child"]
        print(f"{node}: {family}")

    cgf = ChainGraphFactorization(lcn_model)
    factors = cgf.build(verbosity=1)

    print("\nSymbolic chain-graph factors P(child | parents):")
    for factor in factors:
        child = factor["child"]
        parents = factor["parents"]
        scope = factor["scope"]
        print(f"  P({child} | {parents})  scope={scope}")

    # To obtain the local credal sets (intervals) for these symbolic factors,
    # build a CredalNetwork:
    from lcn.inference.marginal.cn.credal_network import CredalNetwork
    cn = CredalNetwork.from_lcn(lcn_model, method="linear", verbosity=0)
    print("\nLocal credal sets (intervals) per factor:")
    for factor in cn.factors:
        child = factor[0]["child"]
        parents = factor[0]["parents"]
        scope = factor[0]["scope"]
        print(f"\nFactor: {child} | parents={parents}, scope={scope}")
        for i, entry in factor.items():
            interp = entry["interpretation"]
            lobo = entry["lobo"]
            upbo = entry["upbo"]
            literals = dict(zip(scope, interp))
            lit_str = ", ".join(f"{k}={v}" for k, v in literals.items())
            lo_str = f"{abs(lobo):.4f}" if lobo is not None else "None"
            up_str = f"{abs(upbo):.4f}" if upbo is not None else "None"
            print(f"  {lit_str}  =>  [{lo_str}, {up_str}]")
