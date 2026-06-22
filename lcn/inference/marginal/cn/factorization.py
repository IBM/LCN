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
from lcn.core.model import LCN


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

    def build(self, verbosity: int = 0):
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

        Returns:
            A list of symbolic factor descriptors (one per family).
        """
        # Ensure that the LCN has been postprocessed (structure, families)
        assert self.lcn.structure_graph is not None
        assert self.lcn.simplified_structure_graph is not None
        assert self.lcn.families is not None

        self.factors = []
        for family in self.lcn.families:
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
