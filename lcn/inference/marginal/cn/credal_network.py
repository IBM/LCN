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
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from lcn.core.model import LCN
from lcn.inference.marginal.cn.factorization import ChainGraphFactorization
from lcn.inference.marginal.cn.local_credal_sets import LocalCredalSetSolver


def _entry_shell(symbolic_factor: Dict, interpretation) -> Dict:
    """Build the structural (bound-less) skeleton of one interval-factor entry.

    Shared by the serial and threaded solve paths so both produce byte-identical
    factor dicts; callers fill in ``lobo``/``upbo`` afterwards.
    """
    return {
        "interpretation": interpretation,
        "scope": symbolic_factor["scope"],
        "child": symbolic_factor["child"],
        "parents": symbolic_factor["parents"],
        "parents_lst": symbolic_factor["parents_lst"],
        "child_lst": symbolic_factor["child_lst"],
        "lobo": 0.0,
        "upbo": 1.0,
    }


def _solve_family(symbolic_factor: Dict, solver: "LocalCredalSetSolver") -> Dict:
    """
    Compute the interval local credal set for one symbolic factor, serially.

    Enumerates all 2^|scope| interpretations of the family scope and solves the
    min/max problems for each with the supplied (stateless, shareable)
    :class:`LocalCredalSetSolver`, returning the interval factor dict keyed by
    interpretation index. Used by the ``n_jobs <= 1`` path; the threaded path
    dispatches the same per-interpretation solves individually across a
    :class:`~concurrent.futures.ThreadPoolExecutor` (see
    :meth:`CredalNetwork.from_lcn`).
    """
    child = symbolic_factor["child"]
    parents_lst = symbolic_factor["parents_lst"]
    scope = symbolic_factor["scope"]
    sentences = symbolic_factor["sentences"]

    factor = {}
    interpretations = list(itertools.product([0, 1], repeat=len(scope)))
    for i, interpretation in enumerate(interpretations):
        literals = dict(zip(scope, interpretation))
        lobo = solver.solve(scope, literals, child, parents_lst, sentences, sense="min")
        upbo = solver.solve(scope, literals, child, parents_lst, sentences, sense="max")
        entry = _entry_shell(symbolic_factor, interpretation)
        entry["lobo"] = lobo if lobo is not None else 0.0
        entry["upbo"] = upbo if upbo is not None else 1.0
        factor[i] = entry
    return factor


def _solve_families_threaded(symbolic_factors: List[Dict],
                             solver: "LocalCredalSetSolver",
                             n_jobs: int) -> List[Dict]:
    """
    Compute every family's interval local credal set concurrently on a thread
    pool, at *per-solve* granularity.

    Each family contributes ``2 * 2^|scope|`` independent min/max solves; all of
    them are flattened into one task list and dispatched to a single
    ``ThreadPoolExecutor(max_workers=n_jobs)``. This load-balances across families
    -- a large family no longer serializes one worker while the others idle -- and
    keeps a bounded ``n_jobs`` concurrent ipopt subprocesses in flight. Results are
    written back by (family, interpretation) index, so the returned list is in the
    original family order and numerically identical to the serial path.

    The dominant cost of each task is the ipopt subprocess (Pyomo NL/AMPL
    interface), which blocks on subprocess I/O and thus releases the GIL, so the
    threads make real progress in parallel. The scipy SLSQP fallback runs
    in-process (it is only reached on suspicious/failed ipopt results and is always
    re-verified against the real constraints); see the LCN_ISOLATE_SLSQP note
    below.
    """
    # Pre-shape the output: one dict per family, each pre-populated with the
    # structural entry skeletons so worker threads only write scalar bounds.
    factors: List[Dict] = []
    interps_per_family: List[list] = []
    for sf in symbolic_factors:
        interpretations = list(itertools.product([0, 1], repeat=len(sf["scope"])))
        interps_per_family.append(interpretations)
        factors.append({i: _entry_shell(sf, interp)
                        for i, interp in enumerate(interpretations)})

    # Flatten to individual (family_idx, interp_idx, sense) solve tasks.
    tasks = []
    for fi, sf in enumerate(symbolic_factors):
        for ii, interpretation in enumerate(interps_per_family[fi]):
            literals = dict(zip(sf["scope"], interpretation))
            for sense in ("min", "max"):
                tasks.append((fi, ii, sense, sf, literals))

    def _one(task):
        fi, ii, sense, sf, literals = task
        val = solver.solve(sf["scope"], literals, sf["child"],
                           sf["parents_lst"], sf["sentences"], sense=sense)
        return fi, ii, sense, val

    # Threads must not fork: the SLSQP crash-isolation (run_isolated) forks via
    # multiprocessing on Linux, and forking from a worker thread of a live pool is
    # a deadlock hazard. Force it in-process for the duration of the pool. This is
    # sound: the SLSQP fallback re-verifies every optimum against the real
    # constraints and degrades to a "no result" sentinel on failure, so we lose
    # only a native-SIGABRT guard that no longer applies without a child process.
    prev_isolate = os.environ.get("LCN_ISOLATE_SLSQP")
    os.environ["LCN_ISOLATE_SLSQP"] = "0"
    try:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            for fut in as_completed(executor.submit(_one, t) for t in tasks):
                fi, ii, sense, val = fut.result()
                key = "lobo" if sense == "min" else "upbo"
                default = 0.0 if sense == "min" else 1.0
                factors[fi][ii][key] = val if val is not None else default
    finally:
        if prev_isolate is None:
            os.environ.pop("LCN_ISOLATE_SLSQP", None)
        else:
            os.environ["LCN_ISOLATE_SLSQP"] = prev_isolate

    return factors


def _build_structural_factor(symbolic_factor: Dict) -> Dict:
    """
    Build a STRUCTURE-ONLY interval factor for one symbolic factor: the same
    per-interpretation dict shape as :func:`_solve_family`, but with placeholder
    ``lobo=0.0``/``upbo=1.0`` bounds and NO solver calls.

    Used by the ``solve_families=False`` build path (scheme D5 / CredalJT), whose
    NLP is formulated from the LCN sentences and LMC equalities and never reads
    the local-credal-set intervals -- only the structural keys (child, parents,
    parents_lst, child_lst, scope, interpretation). The bounds here are junk and
    MUST NOT be consumed by any engine that reasons over the intervals.
    """
    child = symbolic_factor["child"]
    parents = symbolic_factor["parents"]
    parents_lst = symbolic_factor["parents_lst"]
    child_lst = symbolic_factor["child_lst"]
    scope = symbolic_factor["scope"]

    factor = {}
    interpretations = list(itertools.product([0, 1], repeat=len(scope)))
    for i, interpretation in enumerate(interpretations):
        factor[i] = {
            "interpretation": interpretation,
            "scope": scope,
            "child": child,
            "parents": parents,
            "parents_lst": parents_lst,
            "child_lst": child_lst,
            "lobo": 0.0,
            "upbo": 1.0,
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
        # Wall-clock cost of the per-family interval solves. Populated by
        # from_lcn (freshly compiled) or load_cn (read from the .cn header);
        # None when unknown.
        self.compile_time = None

    @classmethod
    def from_lcn(cls, lcn: LCN, method: str = "linear",
                 solver: str = "ipopt", time_limit: float = None,
                 gap_tol: float = 0.0, n_jobs: int = 1,
                 merge_budget: int = 1,
                 solve_families: bool = True,
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
                Number of worker threads for the per-family solves. 1 (the
                default) solves serially. When > 1, the individual per-family
                min/max solves are dispatched across a thread pool: each solve
                spends most of its time in the ipopt subprocess (Pyomo's NL/AMPL
                interface), which releases the GIL, so threads give real
                parallelism without the K x interpreter memory, pickling cost, or
                fork-inside-worker hang of a process pool. The result is
                order-preserving and numerically identical to the serial path.
            merge_budget: int
                Scheme D2: maximum flattened scope of a merged super-family. 1
                (the default) performs no merging (the factors are the LCN
                families); a larger budget merges adjacent families whose
                combined scope is at most this size, so cross-family constraints
                tighten the local credal sets. See
                ``ChainGraphFactorization.build``.
            solve_families: bool
                When True (default), compute the interval local credal set of
                every family by solving the min/max problems (the expensive
                Step 3). When False, SKIP those solves: the factors carry only
                the structural keys (child, parents, scope, interpretations)
                with placeholder ``lobo=0.0``/``upbo=1.0`` bounds. Intended for
                the CredalJT (scheme D5) engine, whose junction-tree NLP is
                formulated from the LCN sentences and LMC equalities and never
                reads the intervals -- so the per-family solves are pure
                overhead there. Any engine that reasons over the intervals
                (CredalVE, IntervalBP, CredalCTE, ApproxLP, CredalIJGP) MUST
                keep this True.
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

        # Step 3: Compute the interval local credal set for each family.
        # When solve_families is False (e.g. CredalJT / scheme D5), skip the
        # min/max solves entirely and build structure-only factors -- the
        # consumer formulates its NLP from the LCN sentences/LMC, not the
        # intervals, so the per-family solves would be pure overhead.
        if not solve_families:
            if verbosity > 0:
                print("[CredalNetwork] solve_families=False: skipping per-family "
                      "interval solves (structure-only factors).")
            factors = [_build_structural_factor(sf) for sf in symbolic_factors]
        else:
            # Suppress the benign Pyomo solver warnings unless full solver
            # progress was requested (verbosity >= 2). All solves now run in this
            # interpreter (serial or threaded), so a single set here suffices.
            if verbosity < 2:
                logging.getLogger('pyomo').setLevel(logging.ERROR)

            # The "linear-tight" path reads the Local Markov Condition, which the
            # LCN computes lazily and memoizes. Force that compute now (before any
            # threading) so the workers never race on the lazy initialization.
            if method == "linear-tight":
                if lcn.primal_graph is None:
                    lcn.build_primal_graph()
                if lcn.independencies is None:
                    lcn.local_markov_condition()

            # A single stateless LocalCredalSetSolver is shared across all solves
            # (serial and threaded): it holds only the LCN + config, and each
            # .solve() builds its own Pyomo model, so there is no shared mutable
            # solver state to race on.
            lcs = LocalCredalSetSolver(
                lcn, method=method, solver=solver, time_limit=time_limit,
                gap_tol=gap_tol, verbosity=verbosity)

            if n_jobs and n_jobs > 1 and len(symbolic_factors) > 1:
                factors = _solve_families_threaded(
                    symbolic_factors, lcs, n_jobs)
            else:
                factors = [_solve_family(sf, lcs) for sf in symbolic_factors]

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

    # ------------------------------------------------------------------
    # Serialization: the compiled credal network as a portable .cn file
    # ------------------------------------------------------------------

    # Bump when the on-disk schema changes in a backward-incompatible way.
    CN_FORMAT_VERSION = 1

    def save_cn(self, file_name: str, method: str = None,
                merge_budget: int = None, solver: str = None,
                compile_time: float = None, n_jobs: int = None) -> None:
        """
        Serialize this compiled credal network to a JSON ``.cn`` file.

        The document stores the directed structure (nodes/atoms/cardinalities)
        and the *interval* local credal sets (each factor's per-interpretation
        ``lobo``/``upbo`` bounds). It does NOT store extreme points -- those are
        derived cheaply on demand by :class:`CredalNetworkVertices`, so the
        ``.cn`` stays pyAgrum-free.

        The ``method``/``merge_budget``/``solver`` header is provenance AND the
        cache-match key: an engine reuses a ``.cn`` only when these match its
        requested build (see :meth:`cn_metadata`). ``compile_time`` records the
        wall-clock cost of the per-family interval solves (the expensive step),
        so a cache hit can report it as build time. ``n_jobs`` is informational
        only -- results are worker-count invariant, so it is NOT part of the
        cache-match key.

        Args:
            file_name: str
                Full path to the output ``.cn`` file.
            method, merge_budget, solver:
                Compilation parameters recorded in the header; the cache-match
                key.
            compile_time: float or None
                Wall-clock seconds spent on the per-family interval solves.
            n_jobs: int or None
                Worker-process count used for the solves (informational).
        """
        # itertools.product yields tuples; JSON round-trips them to lists.
        # We keep lists on disk and normalize back to tuples on load so the
        # in-memory shape stays bit-identical to a freshly built CredalNetwork.
        doc = {
            "format": "lcn-credal-network",
            "version": self.CN_FORMAT_VERSION,
            "source_lcn": getattr(self.lcn, "file_name", None),
            "method": method,
            "merge_budget": merge_budget,
            "solver": solver,
            "compile_time": compile_time,
            "n_jobs": n_jobs,
            "nodes": self.nodes,
            "node_atoms": self.node_atoms,
            "node_card": self.node_card,
            "factors": self.factors,
        }
        with open(file_name, "w") as f:
            json.dump(doc, f, indent=2)

    @classmethod
    def cn_metadata(cls, file_name: str) -> Dict:
        """
        Read only the header of a ``.cn`` file -- format/version and the
        compilation provenance (method/merge_budget/solver/compile_time/n_jobs)
        -- WITHOUT reconstructing the network. Cheap enough to call before every
        cached build to decide whether the file matches the requested settings.

        Returns:
            The header dict on success, or ``None`` if the file is absent,
            unreadable, not valid JSON, or not a recognized ``.cn`` of the
            current format version.
        """
        if not file_name or not os.path.exists(file_name):
            return None
        try:
            with open(file_name) as f:
                doc = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(doc, dict):
            return None
        if doc.get("format") != "lcn-credal-network":
            return None
        if doc.get("version") != cls.CN_FORMAT_VERSION:
            return None
        return {
            "format": doc.get("format"),
            "version": doc.get("version"),
            "method": doc.get("method"),
            "merge_budget": doc.get("merge_budget"),
            "solver": doc.get("solver"),
            "compile_time": doc.get("compile_time"),
            "n_jobs": doc.get("n_jobs"),
        }

    @classmethod
    def load_cn(cls, file_name: str, lcn: LCN) -> "CredalNetwork":
        """
        Reconstruct a CredalNetwork from a ``.cn`` file (produced by
        :meth:`save_cn`) plus its source LCN.

        The symbolic :class:`ChainGraphFactorization` is rebuilt from the LCN
        (it is cheap and purely structural); the interval factors are read from
        the file, so the expensive per-family solves are NOT repeated. The
        resulting object is bit-identical to the one that produced the file.

        Args:
            file_name: str
                Full path to the input ``.cn`` file.
            lcn: LCN
                The source LCN the file was compiled from.

        Returns:
            A CredalNetwork holding the deserialized interval factors.
        """
        with open(file_name) as f:
            doc = json.load(f)

        if doc.get("format") != "lcn-credal-network":
            raise ValueError(f"{file_name} is not an LCN credal-network file.")
        if doc.get("version") != cls.CN_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported .cn format version {doc.get('version')} "
                f"(expected {cls.CN_FORMAT_VERSION}).")

        # Rebuild the structural symbolic factorization from the LCN so the
        # engines that read cn.factorization (e.g. D4 coupling) keep working.
        if lcn.primal_graph is None:
            lcn.build_primal_graph()
        if lcn.structure_graph is None:
            lcn.build_structure_graph()
        if lcn.simplified_structure_graph is None:
            lcn.simplify_structure_graph()
        if lcn.families is None:
            lcn.process_chain_graph()
        factorization = ChainGraphFactorization(lcn)
        factorization.build(verbosity=0, merge_budget=doc.get("merge_budget") or 1)

        # JSON keys are strings and product-tuples became lists; normalize.
        factors = []
        for factor in doc["factors"]:
            entries = {}
            for k, entry in factor.items():
                entry = dict(entry)
                entry["interpretation"] = tuple(entry["interpretation"])
                entries[int(k)] = entry
            factors.append(entries)

        cn = cls(lcn, factorization, factors, doc["nodes"],
                 doc["node_atoms"],
                 {k: int(v) for k, v in doc["node_card"].items()})
        cn.compile_time = doc.get("compile_time")
        return cn


if __name__ == "__main__":

    file_name = "examples/alarm.lcn"
    lcn_model = LCN()
    lcn_model.from_lcn(file_name=file_name)
    print(lcn_model)

    cn = CredalNetwork.from_lcn(lcn_model, method="linear", verbosity=1, n_jobs=5)

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
