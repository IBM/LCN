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

import contextlib
import io
import numpy as np
from typing import Dict, List, Optional

from lcn.core.model import LCN, Sentence, Atom, SentenceType
from lcn.inference.utils.common import (
    check_consistency, check_consistency_product_witness,
    check_consistency_product_witness_scoped,
)


# Binary connectors supported by the LCN parser
_CONNECTORS = ["and", "or", "xor"]

# An LCN instance is predicted "easy" for the global (SCIP) solver when the
# total number of bilinear LMC equality constraints it induces stays small.
# Each Local Markov assertion (X _||_ Y | S) contributes 2^(|Y|+|S|) bilinear
# equalities to the global model (see estimate_difficulty); below this many the
# spatial branch-and-bound certifies all marginals quickly.
EASY_TOTAL_CAP = 256


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
        num_sentences: int = None,
        max_vars_per_sentence: int = 3,
        num_extras: int = 0,
        epsilon: float = 0.3,
        max_retries: int = 100,
        max_component_size: int = 3,
        max_parents: int = 2,
        k: int = 2,
        consistency_restarts: int = 40,
        consistency_mode: str = "product",
        strategy: str = "linear",
        coverage: float = 1.0,
        core: int = 3,
        difficulty_cap: int = 64,
        base_topology: str = "polytree",
        verify_time_limit: float = 5.0,
        verify_gap_tol: float = 0.0,
        verbosity: int = 1,
    ) -> List[LCN]:
        """
        Generate random consistent LCN instances.

        Args:
            num_vars: Number of variables in each LCN.
            graph_type: Graph topology — "random", "dag", "polytree", "tree",
                "tree-fr", "polytree-fr", "chain", "ktree", "ktree-fr", or
                "easy". A "ktree" is the maximal graph of treewidth exactly
                ``k`` (see ``_graph_ktree``); it is oriented as a DAG so every
                atom conditions on the full conjunction of its ``k``
                clique-parents, and its chain-graph junction tree stays at
                treewidth ``k``. "ktree-fr" has the SAME topology but restricts
                the extra marginals to root atoms, so it contains no
                non-family-realizable *sentence* (every Type-1 is on a root, and
                the full-conjunction conditionals each pin a single
                parent-config). CAVEAT: unlike "tree-fr"/"polytree-fr", this does
                NOT make the strong-extension engines (Credal VE, Interval BP)
                exact for ``k >= 2`` — a k-tree with ``k >= 2`` is not
                singly-connected (the ``k`` clique-parents are moralized into a
                loop), a *structural* source of non-realizability the "-fr"
                extras placement cannot remove. So "ktree-fr" means "no
                non-realizable sentences", NOT "exact for CVE/IBP"; use CredalJT
                / ExactInference(solver="global") for exact bounds. (Only for
                ``k == 1``, where the k-tree is a plain tree, is "ktree-fr" fully
                family-realizable.) The "-fr"
                (family-realizable) variants have the same topology as
                "tree"/"polytree"/"ktree" but place the extra marginal sentences
                on root atoms only, so every *sentence* is family-realizable; for
                "tree-fr"/"polytree-fr" (singly-connected) the strong-extension
                engines (Credal VE, Interval BP) are then exact
                (see docs/strong_extension_exactness.tex). "easy" produces
                instances designed to be quick for the SCIP global solver to
                certify; see the strategy/coverage args below.
            num_instances: Number of consistent instances to generate.
            num_sentences: Number of sentences per instance (only used when
                graph_type="random"). Defaults to num_vars if not specified.
            max_vars_per_sentence: Maximum number of variables in a formula.
            num_extras: Number of extra marginal sentences P(x) to add.
            epsilon: Half-width of the probability interval around a random value.
            max_retries: Maximum generation attempts per instance before giving up.
            max_component_size: Maximum number of variables in a chain component
                (only used when graph_type="chain").
            max_parents: Maximum number of parents per child node
                (only used when graph_type="dag" or "polytree").
            k: Treewidth parameter (only used when graph_type="ktree"). Each
                non-seed atom conditions on exactly ``k`` clique-parents, so the
                induced junction-tree treewidth is exactly ``k``. Requires
                num_vars >= k + 1; k=1 degenerates to a random rooted tree.
            strategy: For graph_type="easy", how to make instances easy:
                "linear" (default) conditions each atom on its predecessors so
                most/all Local Markov assertions vanish (coverage controls how
                many); "sparse" produces a small, bounded number of small
                assertions independent of n (the regime of
                examples/linear5a.lcn; controlled by core); "bounded" generates
                a base_topology candidate and rejects it unless its induced
                bilinear load stays under difficulty_cap; "verified" additionally
                runs SCIP and keeps only instances whose marginals all certify
                within verify_time_limit.
            coverage: For strategy="linear", fraction of predecessors each atom
                conditions on. 1.0 (default) = full coverage => zero independence
                assertions => a pure-LP global model (the reliably-easy regime).
                NOTE: difficulty is essentially a cliff, not a gradient -- any
                coverage < 1.0 leaves at least one atom with a large
                non-parent-non-descendant set Y (or a large conditioning set S),
                which re-introduces a ~2^9-term bilinear assertion at n=10 and is
                about as hard as a raw chain. Lower coverage is exposed for
                experimentation, but keep it at 1.0 for guaranteed-easy
                instances; use strategy="sparse" for a few small assertions or
                strategy="verified" to SCIP-filter anything else.
            core: For strategy="sparse", the size of the leading sparse-spine
                region. The first ``core`` atoms condition only on their
                immediate predecessor while later atoms use full coverage,
                yielding ~core-2 assertions each of size <= 2^(core-1),
                independent of n. Keep it small (3-5); core <= 2 degenerates to
                full coverage (zero assertions).
            difficulty_cap: For strategy="bounded", the maximum allowed single
                assertion size 2^(|Y|+|S|) (a parallel total cap of
                EASY_TOTAL_CAP applies to the sum).
            base_topology: For strategy="bounded", the topology of the candidate
                to filter ("dag", "polytree", or "chain").
            verify_time_limit: For strategy="verified", the per-solve SCIP wall
                limit (seconds) under which every marginal must certify.
            verify_gap_tol: For strategy="verified", the SCIP gap tolerance.
            verbosity: Verbosity level (0 = silent).

        Returns:
            A list of consistent LCN instances.
        """
        assert graph_type in ("random", "dag", "polytree", "polytree-fr",
                              "tree", "tree-fr", "chain", "ktree", "ktree-fr",
                              "easy"), \
            f"Unknown graph_type '{graph_type}'. " \
            f"Use 'random', 'dag', 'polytree', 'polytree-fr', 'tree', " \
            f"'tree-fr', 'chain', 'ktree', 'ktree-fr', or 'easy'."
        assert num_vars >= 3, "Need at least 3 variables."
        assert max_component_size >= 1, "max_component_size must be >= 1."
        assert max_parents >= 1, "max_parents must be >= 1."
        assert k >= 1, "k must be >= 1."
        if graph_type in ("ktree", "ktree-fr"):
            assert num_vars >= k + 1, "ktree needs num_vars >= k + 1."
        assert consistency_mode in ("product", "full"), \
            f"Unknown consistency_mode '{consistency_mode}'. Use 'product' or 'full'."
        assert strategy in ("linear", "sparse", "bounded", "verified"), \
            f"Unknown strategy '{strategy}'. " \
            f"Use 'linear', 'sparse', 'bounded', or 'verified'."
        assert 0.0 <= coverage <= 1.0, "coverage must be in [0, 1]."
        assert core >= 2, "core must be >= 2."

        if num_sentences is None:
            num_sentences = num_vars

        instances = []
        total_attempts = 0
        while len(instances) < num_instances:
            total_attempts += 1
            if total_attempts > num_instances * max_retries:
                if verbosity > 0:
                    print(f"[Generator] Gave up after {total_attempts} attempts. "
                          f"Generated {len(instances)}/{num_instances} "
                          f"consistent instances.")
                break

            # Heartbeat: consistency checking for n=10 can be slow, so show
            # progress periodically rather than appearing to hang.
            if verbosity > 0 and total_attempts % 25 == 0:
                print(f"[Generator] {graph_type} n={num_vars}: "
                      f"{len(instances)}/{num_instances} consistent so far "
                      f"after {total_attempts} attempts...")

            if graph_type == "random":
                lcn = self._build_random_lcn(num_vars, num_sentences,
                                             max_vars_per_sentence, epsilon)
            elif graph_type == "easy":
                lcn = self._build_easy_lcn(
                    num_vars, strategy, coverage, difficulty_cap, base_topology,
                    epsilon, max_vars_per_sentence, num_extras,
                    max_component_size, max_parents, core,
                    verify_time_limit, verify_gap_tol)
                if lcn is None:
                    # "bounded"/"verified" rejected this candidate; fall through
                    # to the retry loop (counts against max_retries).
                    continue
            else:
                scopes, components = self._make_graph(num_vars, graph_type,
                                                      max_component_size,
                                                      max_parents, k)
                # The "-fr" classes keep the same topology but place extra
                # marginals on root atoms only.
                extras_on_roots_only = graph_type in (
                    "tree-fr", "polytree-fr", "ktree-fr")
                # k-tree scopes must condition on ALL k clique-parents (not a
                # max_vars subsample) or the treewidth-k guarantee breaks.
                lcn = self._build_lcn(scopes, components, num_vars, epsilon,
                                      max_vars_per_sentence, num_extras,
                                      extras_on_roots_only=extras_on_roots_only,
                                      full_parents=graph_type in (
                                          "ktree", "ktree-fr"))
            if self._check_and_build(lcn, consistency_restarts, verbosity,
                                     consistency_mode):
                instances.append(lcn)
                if verbosity > 0:
                    print(f"[Generator] {graph_type} instance "
                          f"{len(instances)}/{num_instances} "
                          f"({len(lcn.sentences)} sentences, "
                          f"attempt {total_attempts})")
            elif verbosity > 1:
                print(f"[Generator] Attempt {total_attempts}: inconsistent, retrying.")

        return instances

    def generate_easy(
        self,
        num_vars: int,
        num_instances: int = 1,
        strategy: str = "linear",
        coverage: float = 1.0,
        **kwargs,
    ) -> List[LCN]:
        """
        Convenience wrapper for ``generate(graph_type="easy", ...)``.

        Produces consistent LCN instances designed to be quick for the SCIP
        global solver to certify. See ``generate`` for the strategy/coverage
        semantics and the remaining keyword arguments.
        """
        return self.generate(num_vars=num_vars, graph_type="easy",
                             num_instances=num_instances, strategy=strategy,
                             coverage=coverage, **kwargs)

    # ------------------------------------------------------------------
    # Easy-instance generation (low SCIP global-solve difficulty)
    # ------------------------------------------------------------------

    def _graph_full_cover(self, n: int, coverage: float = 1.0):
        """
        Scopes where each atom conditions on its nearest predecessors.

        With ``coverage == 1.0`` every atom x_i conditions on ALL earlier atoms
        x_0..x_{i-1}; then for every atom its parents (all earlier) and
        descendants (all later) cover the rest of the graph, so the Local Markov
        non-parent-non-descendant set Y is empty and NO independence assertions
        are emitted -- the global model becomes a pure LP that SCIP certifies
        immediately. Lower coverage conditions on only the k nearest
        predecessors; this is NOT reliably easier (the un-conditioned earlier
        atoms fall into Y, re-creating large bilinear assertions -- see the
        coverage note in ``generate``), and is exposed only for experimentation.

        Returns ``(scopes, [])`` matching the ``_make_graph`` contract: each
        scope is ``[parents..., child]`` (or ``[var]`` for the root).
        """
        scopes = [[0]]  # root: Type 1 marginal P(x0)
        for i in range(1, n):
            k = max(1, round(coverage * i))
            k = min(k, i)
            parents = list(range(i - k, i))  # the k nearest predecessors
            scopes.append(parents + [i])
        return scopes, []

    def _graph_sparse_linear(self, n: int, core: int = 3):
        """
        Scopes that induce a SMALL, BOUNDED number of small LMC assertions,
        independent of n (the regime of ``examples/linear5a.lcn``).

        Construction: the first ``core`` atoms form a sparse spine -- each x_i
        (1 <= i < core) conditions only on its immediate predecessor x_{i-1} --
        while every atom from index ``core`` on conditions on ALL earlier atoms
        (full coverage). Intuition:

          * The full-coverage tail has parents = all-earlier and descendants =
            all-later, so those atoms emit no assertion (Y = empty).
          * Each sparse early atom x_i skips predecessors x_0..x_{i-2}; one of
            them becomes its single non-parent-non-descendant Y, producing an
            assertion (x_i _||_ Y | x_{i-1}) whose conditioning set lives in the
            low-index region, so |Y|+|S| <= core. The number of assertions is
            ~core-2 and the largest is ~2^(core-1) -- both bounded by ``core``,
            NOT by n.

        With ``core <= 2`` this degenerates to full coverage (zero assertions);
        ``core = 3`` gives one tiny assertion, ``core = 4`` gives two, etc. Keep
        ``core`` small (3-5) for genuinely easy instances. Returns
        ``(scopes, [])`` matching the ``_make_graph`` contract.
        """
        core = max(2, min(core, n))
        scopes = [[0]]  # root: Type 1 marginal P(x0)
        for i in range(1, n):
            if i < core:
                parents = [i - 1]              # sparse spine in the early region
            else:
                parents = list(range(0, i))    # full coverage afterwards
            scopes.append(parents + [i])
        return scopes, []

    def _make_conjunction_formula(self, var_ids: List[int]) -> str:
        """
        Build a conjunction ``x_a and x_b and ...`` over the given variables
        (each possibly negated via ``_make_literal``). Unlike
        ``_make_random_formula`` this uses ALL given variables and only the
        ``and`` connector, so the conditioning set is exactly ``var_ids`` -- the
        property the full-coverage Y=empty argument relies on.
        """
        lits = [self._make_literal(v) for v in var_ids]
        if len(lits) == 1:
            return lits[0]
        return "(" + " and ".join(lits) + ")"

    def _build_easy_from_scopes(self, scopes, num_vars: int, epsilon: float,
                                max_vars: int, num_extras: int) -> LCN:
        """
        Build an LCN from ``[parents..., child]`` scopes, mirroring ``_build_lcn``
        but using a full ``and``-conjunction of the parents as the conditioning
        formula (so the intended predecessors are exactly the structural
        parents). Conditional sentences keep the ``Sentence`` default
        ``tau=True`` -- required for the predecessor->parent derivation, and thus
        for the Y=empty property -- so ``tau`` is never overridden here.
        """
        lcn = LCN()
        lcn.add_atoms([Atom(f"x{i}") for i in range(num_vars)])

        sid = 0
        for scope in scopes:
            if len(scope) == 1:
                phi = self._make_random_formula([scope[0]], max_vars)
                psi = None
            else:
                child = scope[-1]
                parents = scope[:-1]
                phi = self._make_random_formula([child], max_vars)
                psi = self._make_conjunction_formula(parents)
            lo, hi = self._make_bounds(epsilon)
            lcn.add_sentence(Sentence(label=f"s{sid}", phi=phi, psi=psi,
                                      lower=lo, upper=hi))
            sid += 1

        # Extra marginal sentences P(x_i) (kept identical to _build_lcn).
        all_vars = list(range(num_vars))
        extras_added = 0
        attempts = 0
        while extras_added < num_extras and attempts < num_extras * 10:
            attempts += 1
            var = all_vars[self.rng.randint(num_vars)]
            phi = self._make_random_formula([var], max_vars)
            lo, hi = self._make_bounds(epsilon)
            lcn.add_sentence(Sentence(label=f"s{sid}", phi=phi, psi=None,
                                      lower=lo, upper=hi))
            sid += 1
            extras_added += 1

        return lcn

    def _build_easy_lcn(self, num_vars, strategy, coverage, difficulty_cap,
                        base_topology, epsilon, max_vars, num_extras,
                        max_component_size, max_parents, core,
                        verify_time_limit, verify_gap_tol) -> Optional[LCN]:
        """
        Build a single easy-instance candidate. Returns None when a "bounded" or
        "verified" candidate fails its filter (the caller's retry loop handles
        it); "linear"/"sparse" candidates are constructively easy and always
        returned.
        """
        if strategy == "linear":
            scopes, _ = self._graph_full_cover(num_vars, coverage)
            return self._build_easy_from_scopes(scopes, num_vars, epsilon,
                                                max_vars, num_extras)

        if strategy == "sparse":
            scopes, _ = self._graph_sparse_linear(num_vars, core)
            return self._build_easy_from_scopes(scopes, num_vars, epsilon,
                                                max_vars, num_extras)

        if strategy == "bounded":
            scopes, components = self._make_graph(num_vars, base_topology,
                                                  max_component_size, max_parents)
            lcn = self._build_lcn(scopes, components, num_vars, epsilon,
                                  max_vars, num_extras)
            d = estimate_difficulty(lcn)
            if d["max_single"] > difficulty_cap or d["total_bilinear"] > EASY_TOTAL_CAP:
                return None
            return lcn

        # strategy == "verified": build a linear candidate, keep only if SCIP
        # certifies every marginal within the budget.
        scopes, _ = self._graph_full_cover(num_vars, coverage)
        lcn = self._build_easy_from_scopes(scopes, num_vars, epsilon,
                                           max_vars, num_extras)
        if is_globally_easy(lcn, time_limit=verify_time_limit,
                            gap_tol=verify_gap_tol):
            return lcn
        return None

    # ------------------------------------------------------------------
    # Graph topology generators
    # ------------------------------------------------------------------

    def _make_graph(self, num_vars: int, graph_type: str,
                    max_component_size: int = 3,
                    max_parents: int = 2, k: int = 2):
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
        if graph_type == "dag":
            return self._graph_dag(num_vars, max_parents), []
        elif graph_type in ("polytree", "polytree-fr"):
            # "polytree-fr" is the same topology as "polytree"; it differs only
            # in that extra marginals are restricted to root atoms (family-
            # realizable), handled in generate() / _build_lcn.
            return self._graph_polytree(num_vars, max_parents), []
        elif graph_type in ("tree", "tree-fr"):
            # "tree-fr": same topology as "tree", family-realizable extras.
            return self._graph_tree(num_vars), []
        elif graph_type in ("ktree", "ktree-fr"):
            # "ktree-fr": same topology as "ktree"; differs only in restricting
            # extra marginals to root atoms (handled in generate() / _build_lcn).
            return self._graph_ktree(num_vars, k), []
        elif graph_type == "chain":
            return self._graph_chain(num_vars, max_component_size)

    def _random_ordering(self, n: int) -> List[int]:
        """Return a random permutation of 0..n-1."""
        ordering = list(range(n))
        for i in range(n):
            j = self.rng.randint(n)
            ordering[i], ordering[j] = ordering[j], ordering[i]
        return ordering

    def _graph_dag(self, n: int, max_parents: int = 2) -> List[List[int]]:
        """Random DAG. Each non-root picks 1..max_parents parents from
        higher-ordered vars."""
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
                num_parents = self.rng.randint(1, min(max_parents + 1, i + 1))
                parent_indices = self.rng.choice(i, size=num_parents, replace=False)
                parents = [ordering[pi] for pi in parent_indices]
                scopes.append(parents + [v])
        return scopes

    def _graph_polytree(self, n: int,
                        max_parents: int = 1) -> List[List[int]]:
        """Random singly connected DAG: for every pair of nodes (a, b) there
        is at most one directed path from a to b.

        Built in topological order.  Each non-root node picks 1..max_parents
        parents from earlier nodes, accepting a candidate only if the
        singly-connected invariant is preserved.

        Args:
            n: Number of variables.
            max_parents: Maximum number of parents per child node.
        """
        ordering = self._random_ordering(n)

        # ancestors[v] = set of all nodes that can reach v (including v)
        ancestors = {ordering[i]: {ordering[i]} for i in range(n)}

        parents_of = {ordering[i]: [] for i in range(n)}  # parent lists
        num_roots = max(1, self.rng.randint(1, 3))

        for i in range(num_roots, n):
            v = ordering[i]
            # candidates: all earlier nodes in topological order
            candidates = list(range(i))
            self.rng.shuffle(candidates)
            k = self.rng.randint(1, min(max_parents, i) + 1)

            added = 0
            for ci in candidates:
                p = ordering[ci]
                # Adding edge p -> v is safe iff ancestors of p and
                # current ancestors of v are disjoint (no node already
                # reaches v through another path).
                if ancestors[p].isdisjoint(ancestors[v]):
                    parents_of[v].append(p)
                    ancestors[v] |= ancestors[p]
                    added += 1
                    if added >= k:
                        break

        scopes = []
        for i in range(n):
            v = ordering[i]
            scopes.append(parents_of[v] + [v])
        return scopes

    def _graph_tree(self, n: int) -> List[List[int]]:
        """Random rooted directed tree: a single root, and every non-root node
        has EXACTLY ONE parent chosen from the earlier nodes.

        This is the branching analogue of a chain and the single-parent special
        case of a polytree (matches examples/tree.lcn). Because no node has two
        parents, there are no colliders: the moral graph equals the skeleton, so
        the chain-graph junction tree stays at treewidth 1 and inference engines
        such as ARIEL / CredalJT are exact on it (see docs/ariel_exactness.tex).

        Args:
            n: Number of variables.
        """
        ordering = self._random_ordering(n)
        scopes = [[ordering[0]]]  # the single root (Type-1 prior)
        for i in range(1, n):
            v = ordering[i]
            # exactly one parent, drawn uniformly from the earlier nodes
            parent = ordering[self.rng.randint(i)]
            scopes.append([parent, v])
        return scopes

    def _graph_ktree(self, n: int, k: int = 2) -> List[List[int]]:
        """Random k-tree, oriented as a DAG in construction order.

        A k-tree is the maximal graph of treewidth exactly ``k`` (Arnborg &
        Proskurowski): start with a (k+1)-clique, then repeatedly add a new
        vertex adjacent to exactly ``k`` vertices that already form a clique,
        creating a fresh (k+1)-clique. Its treewidth is exactly ``k``.

        We orient it using the construction order so it fits the LCN
        ``[parents..., child]`` scope contract:

          * The seed (k+1)-clique {v_0..v_k} is a directed cascade -- v_0 is a
            Type-1 root prior and each v_i (1<=i<=k) conditions on ALL earlier
            v_0..v_{i-1}. Every earlier vertex is a parent, so all pairs in the
            seed are moralized -> the seed alone has treewidth k.
          * Each later vertex v attaches to a random existing (k+1)-clique with
            one member dropped, giving exactly ``k`` clique-parents.

        Because every parent set is itself a clique, the moralized graph equals
        the (undirected) k-tree, so the chain-graph junction tree stays at
        treewidth ``k``. The child must condition on the FULL conjunction of its
        k parents (handled via ``_build_lcn(full_parents=True)``) for this to
        hold. ``k`` is clamped to ``[1, n-1]``; ``k=1`` reduces to a random
        rooted tree (treewidth 1).

        Returns ``scopes`` (list of ``[parents..., child]``), matching the
        ``_make_graph`` contract; there are no undirected chain components.
        """
        k = max(1, min(k, n - 1))
        ordering = self._random_ordering(n)

        # Seed (k+1)-clique rendered as a directed cascade.
        scopes = [[ordering[0]]]  # root prior P(x_{v0})
        for i in range(1, k + 1):
            parents = [ordering[j] for j in range(i)]
            scopes.append(parents + [ordering[i]])

        # Existing (k+1)-cliques we can attach new vertices to (start: the seed).
        cliques = [[ordering[j] for j in range(k + 1)]]
        for i in range(k + 1, n):
            v = ordering[i]
            base = cliques[self.rng.randint(len(cliques))]  # host clique
            # Drop one of the k+1 members to get a k-clique parent set.
            drop = self.rng.randint(len(base))
            parents = [base[j] for j in range(len(base)) if j != drop]
            scopes.append(parents + [v])
            cliques.append(parents + [v])  # the new (k+1)-clique
        return scopes

    def _graph_chain(self, n: int,
                     max_component_size: int = 3) -> List[List[int]]:
        """
        Chain graph: a DAG of chain components.

        A chain graph is a mixed graph with both directed and undirected
        edges, containing no semi-directed cycles (Lauritzen & Wermuth 1989).
        The chain components are the connected components of the undirected
        subgraph. These components form a DAG when connected by directed edges.

        Each component is either:
        - A single variable (singleton)
        - A group of variables fully connected by undirected edges (clique),
          with size bounded by max_component_size

        The LCN sentences reflect this structure:
        - Variables within a component: Type 1 sentences P(phi) where phi
          involves multiple atoms (creating undirected edges in the structure graph)
        - Directed edges between components: Type 2 sentences P(phi|psi)

        Args:
            n: Number of variables.
            max_component_size: Maximum number of variables per chain component.

        Returns scopes as a list of:
        - [v1, v2, ...] for undirected clique sentences (Type 1, multi-var phi)
        - [parent_vars..., child_var] for directed sentences (Type 2)
        - [v] for singleton marginals (Type 1)
        """
        ordering = self._random_ordering(n)

        # Step 1: Partition variables into chain components
        # Randomly assign variables to components of size 1..max_component_size
        components = []
        idx = 0
        while idx < n:
            remaining = n - idx
            max_size = min(max_component_size, remaining)
            size = self.rng.randint(1, max_size + 1)
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
        lo = max(0.01, val - epsilon)
        hi = min(1.0, val + epsilon)
        return round(lo, 6), round(hi, 6)

    @staticmethod
    def _phi_is_negated(phi: str, var: int) -> bool:
        """True if the single-atom formula phi is the negation !xVAR.

        Single-atom marginal formulas are exactly ``xN`` or ``!xN`` (see
        _make_literal), so a substring test on the stripped string suffices.
        """
        return phi.replace(" ", "") == f"!x{var}"

    def _interval_on_positive(self, phi: str, var: int, lo: float, hi: float):
        """Map a bound [lo, hi] on P(phi) to the implied interval on P(xVAR=1).

        For phi = xVAR this is [lo, hi]; for phi = !xVAR, P(x=1)=1-P(!x), so it
        is [1-hi, 1-lo]. Used to accumulate a consistent per-atom marginal."""
        if self._phi_is_negated(phi, var):
            return (1.0 - hi, 1.0 - lo)
        return (lo, hi)

    def _nested_bounds_for_phi(self, phi: str, var: int, pos_interval):
        """Draw a random sub-interval of ``pos_interval`` (an interval on
        P(xVAR=1)) and return it as a bound on P(phi), respecting phi's polarity.

        Guarantees the resulting sentence is jointly satisfiable with the
        existing marginal on ``var``: the drawn interval is contained in the
        current feasible interval for P(xVAR=1), so the intersection is nonempty.
        """
        plo, phi_hi = pos_interval
        # Degenerate/near-empty feasible interval: just reuse it verbatim.
        if phi_hi - plo <= 1e-6:
            a, b = plo, phi_hi
        else:
            u1 = self.rng.uniform(plo, phi_hi)
            u2 = self.rng.uniform(plo, phi_hi)
            a, b = (u1, u2) if u1 <= u2 else (u2, u1)
        # Map the sub-interval [a, b] on P(x=1) back to a bound on P(phi).
        if self._phi_is_negated(phi, var):
            lo, hi = 1.0 - b, 1.0 - a
        else:
            lo, hi = a, b
        return round(lo, 6), round(hi, 6)

    def _build_random_lcn(self, num_vars: int, num_sentences: int,
                          max_vars: int, epsilon: float) -> LCN:
        """
        Build a random LCN by generating m sentences of the form P(q) or
        P(q|r) over n variables.

        Each sentence is randomly (50/50) either:
        - Type 1 P(q) with tau=False
        - Type 2 P(q|r) with tau=True, where q and r use disjoint variables

        Args:
            num_vars: Number of variables (n).
            num_sentences: Number of sentences (m).
            max_vars: Maximum number of variables per formula.
            epsilon: Half-width of probability intervals.
        """
        lcn = LCN()
        all_vars = list(range(num_vars))
        atoms = [Atom(f"x{i}") for i in range(num_vars)]
        lcn.add_atoms(atoms)

        for sid in range(num_sentences):
            lo, hi = self._make_bounds(epsilon)
            is_conditional = self.rng.uniform() < 0.5

            if is_conditional and num_vars >= 2:
                # Type 2: P(q | r) with disjoint variable sets, tau=True
                total = min(2 * max_vars, num_vars)
                chosen = list(self.rng.choice(all_vars, size=total, replace=False))
                split = self.rng.randint(1, total)
                q_vars = chosen[:split]
                r_vars = chosen[split:]
                phi = self._make_random_formula(list(q_vars), max_vars)
                psi = self._make_random_formula(list(r_vars), max_vars)
                sentence = Sentence(
                    label=f"s{sid}",
                    phi=phi,
                    psi=psi,
                    lower=lo,
                    upper=hi,
                    tau=True,
                )
            else:
                # Type 1: P(q), tau=False
                phi = self._make_random_formula(all_vars, max_vars)
                sentence = Sentence(
                    label=f"s{sid}",
                    phi=phi,
                    psi=None,
                    lower=lo,
                    upper=hi,
                    tau=False,
                )

            lcn.add_sentence(sentence)

        return lcn

    def _build_lcn(self, scopes: List[List[int]],
                   components: List[List[int]],
                   num_vars: int, epsilon: float, max_vars: int,
                   num_extras: int,
                   extras_on_roots_only: bool = False,
                   full_parents: bool = False) -> LCN:
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
            extras_on_roots_only: when False (default) the extra marginals are
                drawn from all atoms (original behavior); when True they are
                restricted to root atoms so every sentence is family-realizable
                (the "-fr" tree/polytree classes). See the loop below.
            full_parents: when False (default) each Type-2 conditioning formula
                psi is a random subformula over a max_vars subsample of the
                parents (original behavior); when True psi is the full
                ``and``-conjunction of ALL parents (via
                ``_make_conjunction_formula``). Used by the "ktree" topology,
                where conditioning on every clique-parent is what keeps the
                induced junction-tree treewidth equal to k.
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

        # Track, per atom, the interval on P(atom=1) implied by an existing
        # single-atom Type-1 marginal (root prior). Used below so a family-
        # realizable extra on that same atom draws a CONSISTENT sub-interval
        # instead of an independent (and often contradictory) one.
        marginal_on_atom: Dict[int, tuple] = {}

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
                if full_parents:
                    # Condition on ALL parents (k-tree: preserves treewidth k).
                    psi = self._make_conjunction_formula(parents)
                else:
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
            if psi is None:
                marginal_on_atom[child] = self._interval_on_positive(
                    phi, child, lo, hi)
            sid += 1

        # Add extra marginal sentences P(x_i).
        #
        # By default (``extras_on_roots_only=False``) the extra marginals are
        # drawn from ALL atoms -- the original generator behavior. Note that a
        # Type-1 marginal P(x) on a NON-root child x is an *effective*
        # cross-family constraint: the child's marginal is
        # sum_{pa} P(x | pa) P(pa), which couples the family P(x | parents) with
        # its parents' marginals across families. No single per-family local
        # credal set can enforce it, so the strong-extension engines (Credal VE,
        # Interval BP) can inflate that marginal beyond the LCN bound -- a Gap B
        # blowup -- even though the atom-scope cross-family test (which sees only
        # the single atom {x}) reports the sentence as family-local. Such
        # instances are legitimate (and exactly solved by CredalJT / ARIEL);
        # they are simply not "family-realizable".
        #
        # When ``extras_on_roots_only=True`` the extras are restricted to ROOT
        # atoms (children of a length-1 scope), whose family P(x) *is* the
        # marginal, so the bound is enforced in-family and no Gap B blowup
        # occurs -- yielding a fully family-realizable instance on which the
        # strong-extension engines are exact. This is the "-fr" tree/polytree
        # class. See docs/strong_extension_exactness.tex, docs/cve_exactness.tex,
        # docs/ibp_exactness.tex.
        if extras_on_roots_only:
            extra_vars = sorted({scope[0] for scope in scopes
                                 if len(scope) == 1})
            if not extra_vars:  # defensive: fall back if no plain root
                extra_vars = list(range(num_vars))
        else:
            extra_vars = list(range(num_vars))
        extras_added = 0
        attempts = 0
        while extras_added < num_extras and attempts < num_extras * 10:
            attempts += 1
            var = extra_vars[self.rng.randint(len(extra_vars))]
            phi = self._make_random_formula([var], max_vars)
            # If this atom already carries a Type-1 marginal (e.g. a root prior,
            # which is always the case for the family-realizable classes where
            # extra_vars are exactly the roots), draw a CONSISTENT sub-interval
            # of that existing marginal instead of an independent one. Otherwise
            # two random P(x) intervals on the same atom routinely fail to
            # overlap -> an inconsistent instance (see the tree_fr_n20_1 bug).
            # The sub-interval is nested in the existing bound (on P(x=1)) and
            # then mapped back to phi's polarity, so both sentences are jointly
            # satisfiable by construction.
            if var in marginal_on_atom:
                lo, hi = self._nested_bounds_for_phi(
                    phi, var, marginal_on_atom[var])
            else:
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
            # Fold the new (consistent) bound into the running interval so a
            # further extra on the same atom stays consistent with both.
            new_pos = self._interval_on_positive(phi, var, lo, hi)
            prev = marginal_on_atom.get(var)
            if prev is None:
                marginal_on_atom[var] = new_pos
            else:
                marginal_on_atom[var] = (max(prev[0], new_pos[0]),
                                         min(prev[1], new_pos[1]))

        return lcn

    def _check_and_build(self, lcn: LCN, consistency_restarts: int = 40,
                         verbosity: int = 1,
                         consistency_mode: str = "product") -> bool:
        """Check consistency of an instance. Returns True if consistent.

        Every instance is checked -- the fast product-distribution witness is
        O(n) and sound, so there is no size at which we blindly accept (an
        earlier version skipped the check for n > 10, which let inconsistent
        large instances -- e.g. contradictory duplicate root marginals -- ship
        silently). Any failure during the check is treated as inconsistent
        (reject), so an inconsistent instance can never be silently accepted.

        Two modes (``consistency_mode``):
        - ``"product"`` (default): the fast SOUND product-distribution witness
          (``check_consistency_product_witness``). It searches per-atom marginals
          only (n vars) and relies on the fact that any product distribution
          satisfies every Local Markov Condition independence automatically, so
          it does NOT build the primal/structure graph or the LMC here. ~0.2s at
          n=10 (and still cheap at larger n) vs ~minutes for the full check.
          Conservative (rejects consistent-but-non-product instances), which
          rejection sampling absorbs.
        - ``"full"``: the exact joint-LMC oracle (``check_consistency``), which
          builds the graphs + LMC. Slower (~minutes at n=10) but accepts any
          consistent instance. Only feasible for small n.

        For n > 10 both the unscoped product witness and the full oracle build a
        2^n table and are intractable, so regardless of ``consistency_mode`` we
        use the SCOPE-LOCAL product witness
        (``check_consistency_product_witness_scoped``): the same sound,
        complete-for-product-consistency search, but evaluating each sentence over
        its own bounded atom scope so there is no 2^n term. This covers every
        topology (tree/polytree/dag/chain/random), catching multi-atom clique and
        cross-scope contradictions -- not just single-literal marginal collisions.

        ``consistency_restarts`` caps the restart budget of the chosen search.
        """
        try:
            n = len(lcn.atoms)
            if n <= 10:
                if consistency_mode == "product":
                    # No graph/LMC build needed: the product witness satisfies
                    # the LMC by construction.
                    return check_consistency_product_witness(
                        lcn, restarts=consistency_restarts)
                # Full joint-LMC oracle (small instances only).
                lcn.build_primal_graph()
                lcn.build_structure_graph()
                lcn.local_markov_condition()
                # check_consistency emits diagnostics; silence unless verbose.
                if verbosity > 1:
                    return check_consistency(
                        lcn, max_slsqp_restarts=consistency_restarts)
                with contextlib.redirect_stdout(io.StringIO()):
                    return check_consistency(
                        lcn, max_slsqp_restarts=consistency_restarts)
            # n > 10: the unscoped product witness and the full oracle both build
            # a 2^n table, so neither scales here. Use the SCOPE-LOCAL product
            # witness, which evaluates every sentence over its own (bounded) atom
            # scope -- no 2^n term -- so it scales to any n while keeping the same
            # sound, complete-for-product-consistency guarantee. This catches not
            # just contradictory single-literal marginals (the tree_fr_n20_1 bug)
            # but also multi-atom clique / cross-scope contradictions that a
            # single-literal structural screen would miss (chain / random). Never
            # blindly accepts.
            return check_consistency_product_witness_scoped(
                lcn, restarts=consistency_restarts)
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


# ----------------------------------------------------------------------
# Difficulty analysis (global-solver hardness of an LCN instance)
# ----------------------------------------------------------------------

def estimate_difficulty(lcn: LCN) -> dict:
    """
    Estimate how hard an LCN is for the global (SCIP) marginal solver.

    The global model adds, for each Local Markov assertion (X _||_ Y | S),
    exactly ``2 ** (|Y| + |S|)`` bilinear equality constraints -- the only
    source of nonconvexity, and the dominant driver of SCIP's branch-and-bound
    cost. This sums those counts over all assertions.

    Builds the primal/structure graphs and the Local Markov Condition as a side
    effect (quietly; those builders print progress). Safe to call on a freshly
    parsed LCN.

    Returns a dict with:
        num_assertions: number of Local Markov independence assertions.
        total_bilinear: sum over assertions of 2^(|Y|+|S|) (the model load).
        max_single:     the largest single 2^(|Y|+|S|) (0 if no assertions).
        predicted:      "easy" if total_bilinear <= EASY_TOTAL_CAP else "hard".
    """
    with contextlib.redirect_stdout(io.StringIO()):
        lcn.build_primal_graph()
        lcn.build_structure_graph()
        lcn.local_markov_condition()

    per = [2 ** (len(a.event2) + len(a.event3))
           for a in lcn.independencies.get_assertions()]
    total = sum(per)
    return {
        "num_assertions": len(per),
        "total_bilinear": total,
        "max_single": max(per, default=0),
        "predicted": "easy" if total <= EASY_TOTAL_CAP else "hard",
    }


def is_globally_easy(lcn: LCN, time_limit: float = 5.0, gap_tol: float = 0.0,
                     den_floor: float = 1e-6) -> bool:
    """
    Ground-truth easy check: run the SCIP global solver and report whether every
    singleton marginal certifies (proven optimal) within ``time_limit`` seconds
    per solve.

    Requires the optional SCIP CLI on PATH; if SCIP is unavailable this returns
    False (the instance cannot be *certified* easy without the global solver)
    rather than raising, so callers degrade gracefully.
    """
    # Imported lazily so the generator does not pull in the inference stack (or
    # require SCIP) unless this check is actually used.
    from lcn.inference.marginal.exact import ExactInference

    with contextlib.redirect_stdout(io.StringIO()):
        lcn.build_primal_graph()
        lcn.build_structure_graph()
        lcn.local_markov_condition()
        algo = ExactInference(lcn=lcn)
        try:
            algo.run(evidence={}, solver="global", verbosity=0,
                     progress_bar=False, time_limit=time_limit,
                     gap_tol=gap_tol, den_floor=den_floor)
        except RuntimeError:
            return False  # SCIP binary not found

    if not algo.feasible or not algo.status:
        return False
    return all(rec['min'][2] == "confirmed" and rec['max'][2] == "confirmed"
               for rec in algo.status.values())


if __name__ == "__main__":

    gen = Generator(seed=42)

    for graph_type in ["random", "dag", "polytree", "tree", "chain"]:
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

    # Easy instances (designed for fast SCIP global certification).
    print(f"\n{'='*60}")
    print("Generating EASY LCNs")
    print(f"{'='*60}")
    for strategy, kw in [("linear", {"coverage": 1.0}),
                         ("sparse", {"core": 3})]:
        print(f"--- strategy={strategy} {kw} ---")
        for n in [5, 8, 10]:
            easy = gen.generate_easy(num_vars=n, num_instances=1,
                                     strategy=strategy, epsilon=0.4,
                                     verbosity=0, **kw)
            for lcn in easy:
                d = estimate_difficulty(lcn)
                print(f"  {strategy} n={n}: assertions={d['num_assertions']} "
                      f"total_bilinear={d['total_bilinear']} "
                      f"max_single={d['max_single']} predicted={d['predicted']}")
