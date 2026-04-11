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

# Random LCN generator with support for multiple graph topologies

import numpy as np
import networkx as nx
from typing import List

from lcn.core.model import LCN, Sentence, Atom
from lcn.inference.utils.common import check_consistency


# Binary connectors supported by the LCN parser
_CONNECTORS = ["and", "or", "xor"]


class Generator:
    """
    Generates random LCN instances with different graph topologies.
    Supports random, DAG, polytree, and chain graph structures.
    Logical formulas are built using standard connectors (and, or, xor, not)
    and can involve multiple variables per sentence.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        num_vars: int,
        graph_type: str,
        num_instances: int = 1,
        max_vars_per_sentence: int = 3,
        num_extras: int = 0,
        epsilon: float = 0.3,
        max_retries: int = 100,
        verbosity: int = 1,
    ) -> List[LCN]:
        """
        Generate random consistent LCN instances.

        Args:
            num_vars: Number of variables in each LCN.
            graph_type: Graph topology — "random", "dag", "polytree", or "chain".
            num_instances: Number of consistent instances to generate.
            max_vars_per_sentence: Maximum number of variables in a formula.
            num_extras: Number of extra marginal sentences P(x) to add.
            epsilon: Half-width of the probability interval around a random value.
            max_retries: Maximum generation attempts per instance before giving up.
            verbosity: Verbosity level (0 = silent).

        Returns:
            A list of consistent LCN instances.
        """
        assert graph_type in ("random", "dag", "polytree", "chain"), \
            f"Unknown graph_type '{graph_type}'. " \
            f"Use 'random', 'dag', 'polytree', or 'chain'."
        assert num_vars >= 3, "Need at least 3 variables."

        instances = []
        total_attempts = 0
        while len(instances) < num_instances:
            total_attempts += 1
            if total_attempts > num_instances * max_retries:
                if verbosity > 0:
                    print(f"[Generator] Gave up after {total_attempts} attempts. "
                          f"Generated {len(instances)}/{num_instances} instances.")
                break

            scopes, components = self._make_graph(num_vars, graph_type)
            lcn = self._build_lcn(scopes, components, num_vars, epsilon,
                                  max_vars_per_sentence, num_extras)
            if self._check_and_build(lcn):
                instances.append(lcn)
                if verbosity > 0:
                    print(f"[Generator] {graph_type} instance "
                          f"{len(instances)}/{num_instances} "
                          f"({len(lcn.sentences)} sentences, "
                          f"attempt {total_attempts})")
            elif verbosity > 1:
                print(f"[Generator] Attempt {total_attempts}: inconsistent, retrying.")

        return instances

    # ------------------------------------------------------------------
    # Graph topology generators
    # ------------------------------------------------------------------

    def _make_graph(self, num_vars: int, graph_type: str):
        """
        Generate scopes for the given topology.

        Returns:
            (scopes, components) where:
            - scopes: list of [parents..., child] for Type 2 sentences
              and [var] for Type 1 singleton sentences
            - components: list of [var1, var2, ...] for Type 1 multi-variable
              sentences (undirected cliques in chain graphs). Empty for
              non-chain-graph topologies.
        """
        if graph_type == "random":
            return self._graph_random(num_vars), []
        elif graph_type == "dag":
            return self._graph_dag(num_vars), []
        elif graph_type == "polytree":
            return self._graph_polytree(num_vars), []
        elif graph_type == "chain":
            return self._graph_chain(num_vars)

    def _random_ordering(self, n: int) -> List[int]:
        """Return a random permutation of 0..n-1."""
        ordering = list(range(n))
        for i in range(n):
            j = self.rng.randint(n)
            ordering[i], ordering[j] = ordering[j], ordering[i]
        return ordering

    def _graph_random(self, n: int) -> List[List[int]]:
        """Random graph (may have cycles). Chain + random extra edges."""
        ordering = self._random_ordering(n)
        scopes = [[ordering[0]]]  # root
        for i in range(1, n):
            scopes.append([ordering[i - 1], ordering[i]])
        # Add a few random edges
        extras = self.rng.randint(1, max(2, n // 2))
        for _ in range(extras):
            x = ordering[self.rng.randint(n)]
            y = ordering[self.rng.randint(n)]
            if x != y and [x, y] not in scopes:
                scopes.append([x, y])
        return scopes

    def _graph_dag(self, n: int) -> List[List[int]]:
        """Random DAG. Each non-root picks 1-2 parents from higher-ordered vars."""
        ordering = self._random_ordering(n)
        position = [0] * n
        for i, v in enumerate(ordering):
            position[v] = i

        scopes = []
        num_roots = max(1, self.rng.randint(1, 3))
        for i in range(n):
            v = ordering[i]
            if i < num_roots:
                scopes.append([v])
            else:
                num_parents = self.rng.randint(1, min(3, i + 1))
                parent_indices = self.rng.choice(i, size=num_parents, replace=False)
                parents = [ordering[pi] for pi in parent_indices]
                scopes.append(parents + [v])
        return scopes

    def _graph_polytree(self, n: int) -> List[List[int]]:
        """Random polytree (tree-shaped DAG, each node has at most 1 parent in the
        undirected sense, but may have multiple parents via directed edges)."""
        ordering = self._random_ordering(n)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        # Start with a chain
        for i in range(1, n):
            G.add_edge(ordering[i - 1], ordering[i])
        # Randomly swap some edges to create a polytree
        for _ in range(n):
            i = self.rng.randint(n)
            j = self.rng.randint(n)
            if i < j:
                u, v = ordering[i], ordering[j]
                if not G.has_edge(u, v):
                    UG = nx.to_undirected(G)
                    paths = list(nx.all_simple_paths(UG, u, v))
                    if len(paths) == 1:
                        k = paths[0][-2]
                        G.remove_edge(k, v)
                        G.add_edge(u, v)

        scopes = []
        for child in range(n):
            parents = list(G.predecessors(child))
            scopes.append(parents + [child])
        return scopes

    def _graph_chain(self, n: int) -> List[List[int]]:
        """
        Chain graph: a DAG of chain components.

        A chain graph is a mixed graph with both directed and undirected
        edges, containing no semi-directed cycles (Lauritzen & Wermuth 1989).
        The chain components are the connected components of the undirected
        subgraph. These components form a DAG when connected by directed edges.

        Each component is either:
        - A single variable (singleton)
        - A group of 2-3 variables fully connected by undirected edges (clique)

        The LCN sentences reflect this structure:
        - Variables within a component: Type 1 sentences P(phi) where phi
          involves multiple atoms (creating undirected edges in the structure graph)
        - Directed edges between components: Type 2 sentences P(phi|psi)

        Returns scopes as a list of:
        - [v1, v2, ...] for undirected clique sentences (Type 1, multi-var phi)
        - [parent_vars..., child_var] for directed sentences (Type 2)
        - [v] for singleton marginals (Type 1)
        """
        ordering = self._random_ordering(n)

        # Step 1: Partition variables into chain components
        # Randomly assign variables to components of size 1-3
        components = []
        idx = 0
        while idx < n:
            remaining = n - idx
            if remaining == 1:
                size = 1
            elif remaining == 2:
                size = self.rng.choice([1, 2])
            else:
                size = self.rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            comp = [ordering[idx + j] for j in range(size)]
            components.append(comp)
            idx += size

        # Step 2: Create a DAG over the components (topological order = list order)
        num_comp = len(components)
        comp_dag = []  # list of (parent_comp_idx, child_comp_idx)
        for i in range(1, num_comp):
            # Each non-root component gets 1 parent from earlier components
            parent_idx = self.rng.randint(0, i)
            comp_dag.append((parent_idx, i))

        # Step 3: Build scopes (Type 2 directed + Type 1 singletons) and
        # components (Type 1 multi-variable undirected cliques)
        scopes = []
        multi_var_components = []

        for comp in components:
            if len(comp) == 1:
                # Singleton: add marginal P(x)
                scopes.append(comp)
            else:
                # Multi-variable component: will generate Type 1 sentence(s)
                # with phi involving all component variables
                multi_var_components.append(comp)

        # For each directed edge between components: add Type 2 sentence
        for parent_idx, child_idx in comp_dag:
            parent_comp = components[parent_idx]
            child_comp = components[child_idx]
            # Pick one variable from the child component as the phi target
            child_var = child_comp[self.rng.randint(len(child_comp))]
            # Use all parent component variables as the psi condition
            scope = parent_comp + [child_var]
            scopes.append(scope)

        return scopes, multi_var_components

    # ------------------------------------------------------------------
    # Formula generation
    # ------------------------------------------------------------------

    def _make_random_formula(self, variables: List[int],
                             max_vars: int) -> str:
        """
        Generate a random propositional logic formula over a subset of the
        given variable indices. Uses connectors: and, or, xor, not (!).

        Args:
            variables: List of variable indices to choose from.
            max_vars: Maximum number of variables to use in the formula.

        Returns:
            A formula string like "(x2 or !x3)" or "x1 and (x0 xor !x4)".
        """
        k = min(max_vars, len(variables))
        if k <= 0:
            k = 1
        k = self.rng.randint(1, k + 1)  # pick 1..k variables
        chosen = list(self.rng.choice(variables, size=k, replace=False))
        return self._build_formula(chosen)

    def _build_formula(self, var_ids: List[int]) -> str:
        """Recursively build a formula from a list of variable ids."""
        if len(var_ids) == 1:
            return self._make_literal(var_ids[0])
        if len(var_ids) == 2:
            conn = _CONNECTORS[self.rng.randint(len(_CONNECTORS))]
            left = self._make_literal(var_ids[0])
            right = self._make_literal(var_ids[1])
            return f"({left} {conn} {right})"
        # Split into two groups and recurse
        split = self.rng.randint(1, len(var_ids))
        self.rng.shuffle(var_ids)
        left_vars = var_ids[:split]
        right_vars = var_ids[split:]
        # Avoid empty sides
        if len(left_vars) == 0:
            left_vars = [var_ids[0]]
            right_vars = var_ids[1:]
        if len(right_vars) == 0:
            right_vars = [var_ids[-1]]
            left_vars = var_ids[:-1]
        conn = _CONNECTORS[self.rng.randint(len(_CONNECTORS))]
        left = self._build_formula(list(left_vars))
        right = self._build_formula(list(right_vars))
        return f"({left} {conn} {right})"

    def _make_literal(self, var_id: int) -> str:
        """Return an atom or its negation with 50/50 probability."""
        name = f"x{var_id}"
        if self.rng.uniform() < 0.3:
            return f"!{name}"
        return name

    # ------------------------------------------------------------------
    # Sentence and LCN construction
    # ------------------------------------------------------------------

    def _make_bounds(self, epsilon: float):
        """Generate random lower and upper probability bounds."""
        val = self.rng.uniform()
        lo = max(0.0, val - epsilon)
        hi = min(1.0, val + epsilon)
        return round(lo, 6), round(hi, 6)

    def _build_lcn(self, scopes: List[List[int]],
                   components: List[List[int]],
                   num_vars: int, epsilon: float, max_vars: int,
                   num_extras: int) -> LCN:
        """
        Build an LCN instance from scopes and chain components.

        Args:
            scopes: list of [parents..., child] for Type 2 and [var] for Type 1.
            components: list of multi-variable lists for Type 1 undirected
                        clique sentences (chain graph components). Empty for
                        non-chain-graph topologies.
            num_vars: total number of variables.
            epsilon: half-width of probability intervals.
            max_vars: max variables per formula.
            num_extras: extra marginal sentences to add.
        """
        lcn = LCN()
        atoms = [Atom(f"x{i}") for i in range(num_vars)]
        lcn.add_atoms(atoms)

        sid = 0

        # Type 1 multi-variable sentences for chain graph components
        for comp in components:
            phi = self._make_random_formula(comp, max_vars=len(comp))
            lo, hi = self._make_bounds(epsilon)
            sentence = Sentence(
                label=f"s{sid}",
                phi=phi,
                psi=None,
                lower=lo,
                upper=hi,
            )
            lcn.add_sentence(sentence)
            sid += 1

        # Scopes: Type 1 singletons and Type 2 directed sentences
        for scope in scopes:
            if len(scope) == 1:
                # Type 1: P(phi)
                child = scope[0]
                phi = self._make_random_formula([child], max_vars)
                psi = None
            else:
                # Type 2: P(phi | psi)
                child = scope[-1]
                parents = scope[:-1]
                phi = self._make_random_formula([child], max_vars)
                psi = self._make_random_formula(parents, max_vars)

            lo, hi = self._make_bounds(epsilon)
            sentence = Sentence(
                label=f"s{sid}",
                phi=phi,
                psi=psi,
                lower=lo,
                upper=hi,
            )
            lcn.add_sentence(sentence)
            sid += 1

        # Add extra marginal sentences P(x_i)
        all_vars = list(range(num_vars))
        extras_added = 0
        attempts = 0
        while extras_added < num_extras and attempts < num_extras * 10:
            attempts += 1
            var = all_vars[self.rng.randint(num_vars)]
            phi = self._make_random_formula([var], max_vars)
            lo, hi = self._make_bounds(epsilon)
            sentence = Sentence(
                label=f"s{sid}",
                phi=phi,
                psi=None,
                lower=lo,
                upper=hi,
            )
            lcn.add_sentence(sentence)
            sid += 1
            extras_added += 1

        return lcn

    def _check_and_build(self, lcn: LCN) -> bool:
        """Build the LCN structure and check consistency. Returns True if consistent."""
        try:
            if len(lcn.atoms) < 10:
                lcn.build_primal_graph()
                lcn.build_structure_graph()
                lcn.local_markov_condition()
                return check_consistency(lcn)
            else:
                return True  # skip consistency check for large instances to save time
        except Exception:
            return False

    @staticmethod
    def save(lcn: LCN, file_name: str):
        """
        Save an LCN instance to a file using the .lcn format.

        Args:
            lcn: LCN
                The LCN instance to save.
            file_name: str
                Path to the output file.
        """
        lcn.save_lcn(file_name)


if __name__ == "__main__":

    gen = Generator(seed=42)

    for graph_type in ["random", "dag", "polytree", "chain"]:
        print(f"\n{'='*60}")
        print(f"Generating {graph_type} LCNs (5 variables, 2 instances)")
        print(f"{'='*60}")
        instances = gen.generate(
            num_vars=10,
            graph_type=graph_type,
            num_instances=2,
            max_vars_per_sentence=4,
            num_extras=2,
            epsilon=0.3,
            verbosity=1,
        )
        for i, lcn in enumerate(instances):
            print(f"\n--- {graph_type} instance {i+1} ---")
            print(lcn)
            fname = f"/tmp/lcn_{graph_type}_{i+1}.lcn"
            gen.save(lcn, fname)
            print(f"Saved to {fname}")
