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

# CredalNetwork: the imprecise Bayesian network induced by an LCN chain graph.
#
# A CredalNetwork pairs the directed graph of the chain-graph factorization
# (the symbolic factors P(child | parents)) with the *interval* local credal
# sets computed from the LCN constraints. For each symbolic factor it
# enumerates the 2^|scope| interpretations and solves the corresponding
# (fractional) optimization problems to obtain the lower/upper bound of each
# conditional probability.
#
# This class is pure data: it holds the graph and the interval factors and has
# NO pyAgrum coupling. Enumerating the extreme points of the local credal sets
# is the job of `CredalNetworkVertices` (see vertices.py).

import itertools
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

from lcn.core.model import LCN
from lcn.inference.marginal.cn.factorization import ChainGraphFactorization
from lcn.inference.marginal.cn.local_credal_sets import LocalCredalSetSolver


def _solve_family(symbolic_factor: Dict, lcn: LCN, method: str,
                  solver_backend: str, time_limit, gap_tol: float,
                  verbosity: int) -> Dict:
    """
    Compute the interval local credal set for one symbolic factor.

    Enumerates all 2^|scope| interpretations of the family scope and solves the
    min/max problems for each, returning the interval factor dict keyed by
    interpretation index. This is a module-level function so it can be shipped
    to worker processes for parallel per-family solving.
    """
    solver = LocalCredalSetSolver(
        lcn, method=method, solver=solver_backend,
        time_limit=time_limit, gap_tol=gap_tol, verbosity=verbosity)

    child = symbolic_factor["child"]
    parents = symbolic_factor["parents"]
    parents_lst = symbolic_factor["parents_lst"]
    child_lst = symbolic_factor["child_lst"]
    scope = symbolic_factor["scope"]
    sentences = symbolic_factor["sentences"]

    factor = {}
    interpretations = list(itertools.product([0, 1], repeat=len(scope)))
    for i, interpretation in enumerate(interpretations):
        literals = dict(zip(scope, interpretation))
        lobo = solver.solve(scope, literals, child, parents_lst, sentences, sense="min")
        upbo = solver.solve(scope, literals, child, parents_lst, sentences, sense="max")
        factor[i] = {
            "interpretation": interpretation,
            "scope": scope,
            "child": child,
            "parents": parents,
            "parents_lst": parents_lst,
            "child_lst": child_lst,
            "lobo": lobo if lobo is not None else 0.0,
            "upbo": upbo if upbo is not None else 1.0,
        }
    return factor


class CredalNetwork:
    """
    An imprecise Bayesian network derived from an LCN chain graph: the directed
    graph of the chain-graph factorization together with the interval local
    credal sets P(child | parents) computed from the LCN constraints.

    Build with :meth:`from_lcn`. The resulting object exposes:
        - factors:    list of interval factors (one per family); each is a dict
                      keyed by interpretation index, with per-entry "lobo"/"upbo"
        - nodes:      list of node names in the simplified structure graph
        - node_atoms: dict node_name -> list of atoms (compound nodes split "-")
        - node_card:  dict node_name -> cardinality (2^len(atoms))
        - lcn:        the source LCN
    """

    def __init__(self, lcn: LCN, factorization: ChainGraphFactorization,
                 factors: List[Dict], nodes: List[str],
                 node_atoms: Dict[str, List[str]], node_card: Dict[str, int]):
        self.lcn = lcn
        self.factorization = factorization
        self.factors = factors
        self.nodes = nodes
        self.node_atoms = node_atoms
        self.node_card = node_card

    @classmethod
    def from_lcn(cls, lcn: LCN, method: str = "linear",
                 solver: str = "ipopt", time_limit: float = None,
                 gap_tol: float = 0.0, n_jobs: int = 1,
                 merge_budget: int = 1,
                 verbosity: int = 1) -> "CredalNetwork":
        """
        Build a CredalNetwork from an LCN: derive the chain-graph structure,
        build the symbolic factorization and compute the interval local credal
        sets for every family.

        Args:
            lcn: LCN
                The source model.
            method: str
                Factorization method: "linear" (in-scope-sentence LP /
                fractional LP) or "linear-tight" (scheme D1: additionally
                imposes the scope-restricted Local Markov Condition equalities,
                a bilinear program).
            solver: str
                Solver backend for the per-family solves: "ipopt" (default,
                hardened local solver) or "scip" (global solver; requires the
                SCIP CLI on PATH).
            time_limit: float or None
                Per-solve wall-clock limit in seconds (passed to the solver).
            gap_tol: float
                SCIP relative optimality gap to stop at (ignored by ipopt).
            n_jobs: int
                Number of worker processes for the per-family solves. 1 (the
                default) solves serially. Each family is one task; the result
                is order-preserving and identical to the serial computation.
            merge_budget: int
                Scheme D2: maximum flattened scope of a merged super-family. 1
                (the default) performs no merging (the factors are the LCN
                families); a larger budget merges adjacent families whose
                combined scope is at most this size, so cross-family constraints
                tighten the local credal sets. See
                ``ChainGraphFactorization.build``.
            verbosity: int
                Verbosity level (0 is silent).

        Returns:
            A CredalNetwork holding the interval factors.
        """
        assert method in ("linear", "linear-tight"), \
            f"Unknown method '{method}'. Use 'linear' or 'linear-tight'."
        assert solver in ("ipopt", "scip"), \
            f"Unknown solver '{solver}'. Use 'ipopt' or 'scip'."

        # Step 1: Ensure the LCN chain-graph structure is built
        if lcn.primal_graph is None:
            lcn.build_primal_graph()
        if lcn.structure_graph is None:
            lcn.build_structure_graph()
        if lcn.simplified_structure_graph is None:
            lcn.simplify_structure_graph()
        assert lcn.is_chain_graph(), "The LCN must be a chain graph."
        if lcn.families is None:
            lcn.process_chain_graph()

        # Step 2: Build the symbolic chain-graph factorization (optionally
        # merging adjacent families per the D2 merge_budget).
        factorization = ChainGraphFactorization(lcn)
        symbolic_factors = factorization.build(
            verbosity=verbosity, merge_budget=merge_budget)

        if verbosity > 0:
            print(f"[CredalNetwork] Chain-graph factorization produced "
                  f"{len(symbolic_factors)} symbolic factors.")

        # Step 3: Compute the interval local credal set for each family
        if n_jobs and n_jobs > 1 and len(symbolic_factors) > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                factors = list(executor.map(
                    _solve_family,
                    symbolic_factors,
                    itertools.repeat(lcn),
                    itertools.repeat(method),
                    itertools.repeat(solver),
                    itertools.repeat(time_limit),
                    itertools.repeat(gap_tol),
                    itertools.repeat(verbosity),
                ))
        else:
            factors = [
                _solve_family(sf, lcn, method, solver, time_limit,
                              gap_tol, verbosity)
                for sf in symbolic_factors
            ]

        # Step 4: Collect node info from the (possibly merged) symbolic factors,
        # NOT from the simplified structure graph -- under a merge_budget > 1 the
        # factor children are merged super-nodes that the structure graph does
        # not contain. With merge_budget == 1 the factor children are exactly the
        # structure-graph nodes, so this is identical to the unmerged behavior.
        node_names = [sf["child"] for sf in symbolic_factors]
        node_atoms = {}
        node_card = {}
        for name in node_names:
            atoms = name.split("-") if "-" in name else [name]
            node_atoms[name] = atoms
            node_card[name] = 2 ** len(atoms)

        return cls(lcn, factorization, factors, node_names,
                   node_atoms, node_card)


if __name__ == "__main__":

    file_name = "examples/alarm.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    cn = CredalNetwork.from_lcn(l, method="linear", verbosity=1)

    print(f"\n[CredalNetwork] nodes: {cn.nodes}")
    print("[CredalNetwork] interval factors:")
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
            print(f"  {lit_str}  =>  [{abs(lobo):.4f}, {abs(upbo):.4f}]")
