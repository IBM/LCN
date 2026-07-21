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

# CredalNetworkVertices: the extreme-point (vertex) representation of a
# CredalNetwork. Given the interval local credal sets of a CredalNetwork, it
# assembles a pair of pyAgrum BayesNets (lower/upper bounds), builds a
# gum.CredalNet, and uses LRS (via intervalToCredal) to enumerate the extreme
# points of every local credal set. The resulting `extreme_points` dict is the
# common input consumed by all credal-network inference algorithms (CredalVE,
# IntervalBP, CredalCTE, ApproxLP, CredalIJGP).
#
# All pyAgrum coupling lives here; CredalNetwork itself is pyAgrum-free.

import json
import logging
import os
import re
import time
from typing import Dict

import pyagrum as gum  # noqa: N813

from lcn.core.model import LCN
from lcn.inference.marginal.cn.credal_network import CredalNetwork


def _cache_matches(meta: Dict, method: str, merge_budget: int,
                   solver: str) -> bool:
    """
    Decide whether a compiled .cn (via its cn_metadata header) was produced
    with the same settings as the requested build. The cache-match key is
    (method, merge_budget, solver); compile_time / n_jobs are not part of it
    (n_jobs does not affect the result, and compile_time is an output).
    """
    return (meta.get("method") == method
            and (meta.get("merge_budget") or 1) == (merge_budget or 1)
            and meta.get("solver") == solver)


def _parse_credal_net_vertices(cn: gum.CredalNet) -> Dict:
    """
    Parse the string representation of a CredalNet to extract the extreme
    points (vertices) of each credal set for every node and parent config.

    Returns:
        A dict: {node_name: {parent_config_str: [[v0, v1, ...], ...]}}
    """
    result = {}
    cn_str = str(cn)
    # Split into per-node blocks separated by blank lines
    blocks = cn_str.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue
        # First line: "NodeName:Labelized({0|1|...})"
        header = lines[0]
        node_name = header.split(":")[0].strip()
        result[node_name] = {}
        for line in lines[1:]:
            # Format: "<parent_config> : [[v1, v2], [v3, v4], ...]"
            match = re.match(r"^(<[^>]*>)\s*:\s*\[(.+)\]\s*$", line.strip())
            if not match:
                continue
            parent_config = match.group(1)
            vertices_str = match.group(2)
            # Parse nested lists: [[0.3 , 0.7] , [0.5 , 0.5]]
            vertices = []
            for vm in re.finditer(r"\[([^\[\]]+)\]", vertices_str):
                vals = [float(x.strip()) for x in vm.group(1).split(",")]
                vertices.append(vals)
            result[node_name][parent_config] = vertices
    return result


class CredalNetworkVertices:
    """
    The extreme-point representation of a CredalNetwork.

    Wraps a :class:`CredalNetwork` (the directed graph + interval local credal
    sets) and, on :meth:`build`, enumerates the extreme points of every local
    credal set via pyAgrum's LRS. Exposes:
        - extreme_points: {node: {parent_config_str: [[vertex], ...]}}
        - bn_min, bn_max: the lower/upper pyAgrum BayesNets
        - credal_net:     the gum.CredalNet
        - build_time:     wall-clock seconds for the build (see from_lcn for the
                          whole-pipeline timing)
        - lcn:            the source LCN (via the wrapped CredalNetwork)
    """

    def __init__(self, cn: CredalNetwork):
        self.cn = cn
        self.bn_min = None
        self.bn_max = None
        self.credal_net = None
        self.extreme_points = None
        self.build_time = None
        # Cache bookkeeping (set by from_lcn): whether the interval factors came
        # from a compiled .cn on disk, and the compile_time recorded there.
        self.loaded_from_cache = False
        self.compile_time = None
        # Vertex-cache bookkeeping: whether the extreme points were loaded from a
        # .vtx on disk, and the enumeration_time recorded there. When freshly
        # enumerated, enumeration_time is the wall-clock of the LRS section.
        self.loaded_vertices_from_cache = False
        self.enumeration_time = None

    @property
    def lcn(self) -> LCN:
        return self.cn.lcn

    def build(self, verbosity: int = 1,
              enumerate_vertices: bool = True) -> Dict:
        """
        Assemble the lower/upper BayesNets from the CredalNetwork's interval
        factors and enumerate the extreme points via LRS.

        Args:
            verbosity: int
                Verbosity level (0 is silent).
            enumerate_vertices: bool
                When True (default) run LRS extreme-point enumeration and store
                the result in ``extreme_points``. When False, build the
                lower/upper BayesNets but SKIP the (potentially expensive) LRS
                enumeration -- ``credal_net``/``extreme_points`` stay None. Used
                by the CredalJT (scheme D5) engine, whose constraint NLP is
                formulated from the interval local credal sets and never
                consumes the enumerated vertices.

        Returns:
            A dict with the build summary: {"build_time", "n_nodes",
            "n_vertices"} (build_time covers only this method; for the
            whole-pipeline timing use from_lcn).
        """
        t0 = time.perf_counter()
        self._build(verbosity=verbosity, enumerate_vertices=enumerate_vertices)
        self.build_time = time.perf_counter() - t0
        return self._summary()

    @classmethod
    def from_lcn(cls, lcn: LCN, method: str = "linear",
                 solver: str = "ipopt", time_limit: float = None,
                 gap_tol: float = 0.0, n_jobs: int = 1,
                 merge_budget: int = 1,
                 enumerate_vertices: bool = True,
                 solve_families: bool = True,
                 lcn_file: str = None, cache: bool = True,
                 verbosity: int = 1,
                 use_highs: bool = True) -> "CredalNetworkVertices":
        """
        Build the full pipeline from an LCN: CredalNetwork (chain-graph
        factorization + interval local credal sets) followed by extreme-point
        enumeration. `build_time` measures the WHOLE pipeline (interval solves
        plus vertex enumeration) with a single perf_counter bracket, which is
        what experiments report.

        Args:
            lcn: LCN
                The source model.
            method: str
                Factorization method ("linear" or "linear-tight").
            solver: str
                Solver backend for the per-family solves: "ipopt" (default) or
                "scip" (global; requires the SCIP CLI on PATH).
            time_limit: float or None
                Per-solve wall-clock limit in seconds.
            gap_tol: float
                SCIP relative optimality gap (ignored by ipopt).
            n_jobs: int
                Worker processes for the per-family interval solves.
            merge_budget: int
                Scheme D2: maximum flattened scope of a merged super-family
                (1 = no merging; see CredalNetwork.from_lcn).
            enumerate_vertices: bool
                When True (default) run LRS extreme-point enumeration. When
                False, skip it (see :meth:`build`); ``extreme_points`` stays
                None. The CredalJT (D5) engine passes False -- its NLP uses the
                interval local credal sets, not the enumerated vertices.
            solve_families: bool
                Forwarded to :meth:`CredalNetwork.from_lcn`. When True (default)
                compute the interval local credal set of every family. When
                False, SKIP the per-family min/max solves entirely (structure-
                only factors with placeholder bounds) -- the CredalJT (scheme D5)
                engine passes False because its junction-tree NLP is built from
                the LCN sentences/LMC and never reads the intervals. Because the
                placeholder bounds carry no usable credal set, ``solve_families=
                False`` forces ``enumerate_vertices=False`` (there is nothing to
                enumerate); any engine that consumes the intervals or vertices
                MUST keep this True.
            lcn_file: str or None
                Path to the source ``.lcn`` file. When given (and ``cache`` is
                True), two on-disk caches next to it (same basename) are reused
                if their recorded ``method``/``merge_budget``/``solver`` match
                this build:
                  - ``.vtx`` (enumerated extreme points, checked FIRST): a hit
                    skips BOTH the per-family interval solves and the LRS
                    enumeration -- only the cheap chain-graph structure is
                    rebuilt (from a matching ``.cn`` if present, else structure-
                    only from the ``.lcn``). Reported ``build_time`` reuses the
                    stored ``compile_time + enumeration_time``.
                  - ``.cn`` (interval local credal sets): a hit skips the
                    per-family solves; the LRS enumeration still runs.
                When None the caches are inactive (the LCN object does not store
                its own source path), so direct library callers keep the
                uncached behavior.
            cache: bool
                Enable the ``.vtx``/``.cn`` caches described under ``lcn_file``
                (default True). Set False to always recompute in memory even
                when matching cache files exist.
            verbosity: int
                Verbosity level. 0 is silent. At verbosity < 2 the ipopt/scip
                solver warnings/errors emitted during the per-family solves
                (e.g. Pyomo's "Loading a SolverResults object with a warning
                status" / "termination condition: other" notices, which are
                benign here -- the hardened multi-restart handles them) are
                SUPPRESSED. At verbosity >= 2 the solver's own progress log is
                streamed (ipopt/scip ``tee``) and those messages are shown.
            use_highs: bool
                Forwarded to :meth:`CredalNetwork.from_lcn`. When True (default)
                and method="linear"/solver="ipopt", each per-family LP is solved
                in-process with HiGHS (numerically identical, far faster than
                launching ipopt per solve); set False to force the legacy ipopt
                path (A/B testing). No effect for method "linear-tight",
                solver "scip", or a cache hit (which skips solving). Not part of
                the ``.cn``/``.vtx`` provenance key.
        """
        # Placeholder bounds from a structure-only build carry no usable credal
        # set, so there is nothing for LRS to enumerate: force enumerate_vertices
        # off when the families are not solved.
        if not solve_families and enumerate_vertices:
            if verbosity > 0:
                print("[CredalNetworkVertices] solve_families=False forces "
                      "enumerate_vertices=False (no intervals to enumerate).")
            enumerate_vertices = False

        # Transparent VERTEX cache: reuse enumerated extreme points from a .vtx
        # next to the source .lcn when its recorded settings match. This is the
        # cheapest path -- it skips BOTH the per-family interval solves and the
        # LRS enumeration. Only meaningful when vertices are actually wanted.
        if (cache and solve_families and enumerate_vertices and lcn_file):
            vtx_path = os.path.splitext(lcn_file)[0] + ".vtx"
            vmeta = cls.vtx_metadata(vtx_path)
            if vmeta is not None and _cache_matches(
                    vmeta, method, merge_budget, solver):
                t0 = time.perf_counter()
                loaded = cls.load_extreme_points(vtx_path)
                extreme_points, _ = loaded
                # We need cn/bn_min for STRUCTURE only (cardinalities, arcs). Use
                # a matching compiled .cn if present; otherwise rebuild just the
                # chain-graph structure (no per-family interval solves, no LRS).
                cn_path = os.path.splitext(lcn_file)[0] + ".cn"
                cn_meta = CredalNetwork.cn_metadata(cn_path)
                if cn_meta is not None and _cache_matches(
                        cn_meta, method, merge_budget, solver):
                    cn = CredalNetwork.load_cn(cn_path, lcn)
                else:
                    cn = CredalNetwork.from_lcn(
                        lcn, method=method, solver=solver,
                        time_limit=time_limit, gap_tol=gap_tol, n_jobs=n_jobs,
                        merge_budget=merge_budget, solve_families=False,
                        verbosity=verbosity, use_highs=use_highs)
                cnv = cls(cn)
                # Build bn_min/bn_max (cheap structure) but SKIP LRS, then inject
                # the loaded vertices.
                cnv._build(verbosity=verbosity, enumerate_vertices=False)
                cnv.extreme_points = extreme_points
                cnv.credal_net = None
                struct_time = time.perf_counter() - t0
                cnv.loaded_from_cache = True
                cnv.loaded_vertices_from_cache = True
                cnv.compile_time = vmeta.get("compile_time")
                cnv.enumeration_time = vmeta.get("enumeration_time")
                base = (vmeta.get("compile_time") or 0.0) + \
                       (vmeta.get("enumeration_time") or 0.0)
                cnv.build_time = base + struct_time
                if verbosity > 0:
                    print(f"[CredalNetworkVertices] Loaded vertices from "
                          f"{vtx_path} (compile_time="
                          f"{vmeta.get('compile_time') or 0.0:.4f}s, "
                          f"enumeration_time="
                          f"{vmeta.get('enumeration_time') or 0.0:.4f}s).")
                return cnv
            elif vmeta is not None and verbosity > 0:
                print(f"[CredalNetworkVertices] Ignoring {vtx_path}: recorded "
                      f"settings do not match the requested build "
                      f"(method={method}, merge_budget={merge_budget}, "
                      f"solver={solver}); re-enumerating.")

        # Transparent cache: reuse a compiled .cn next to the source .lcn when
        # its recorded settings match this build. Only when families are solved
        # (a structure-only build carries no intervals to cache) and a source
        # path was provided.
        if cache and solve_families and lcn_file:
            cn_path = os.path.splitext(lcn_file)[0] + ".cn"
            meta = CredalNetwork.cn_metadata(cn_path)
            if meta is not None and _cache_matches(
                    meta, method, merge_budget, solver):
                t0 = time.perf_counter()
                cn = CredalNetwork.load_cn(cn_path, lcn)
                cnv = cls(cn)
                # The per-family interval solves are what the compile_time
                # measured; the vertex enumeration still runs here and is added
                # so build_time stays an honest whole-pipeline wall-clock.
                pyomo_logger = logging.getLogger('pyomo')
                prev_level = pyomo_logger.level
                if verbosity < 2:
                    pyomo_logger.setLevel(logging.ERROR)
                try:
                    cnv._build(verbosity=verbosity,
                               enumerate_vertices=enumerate_vertices,
                               n_jobs=n_jobs)
                finally:
                    pyomo_logger.setLevel(prev_level)
                enum_time = time.perf_counter() - t0
                cnv.loaded_from_cache = True
                cnv.compile_time = meta.get("compile_time")
                base = meta.get("compile_time") or 0.0
                cnv.build_time = base + enum_time
                if verbosity > 0:
                    print(f"[CredalNetworkVertices] Loaded compiled network from "
                          f"{cn_path} (compile_time={base:.4f}s, "
                          f"enumeration={enum_time:.4f}s).")
                return cnv
            elif meta is not None and verbosity > 0:
                print(f"[CredalNetworkVertices] Ignoring {cn_path}: recorded "
                      f"settings (method={meta.get('method')}, "
                      f"merge_budget={meta.get('merge_budget')}, "
                      f"solver={meta.get('solver')}) do not match the requested "
                      f"build (method={method}, merge_budget={merge_budget}, "
                      f"solver={solver}); recompiling.")

        t0 = time.perf_counter()
        # Suppress the (benign) Pyomo solver warnings during the build, unless
        # verbosity >= 2 where the user asked to see full solver progress. This
        # covers the serial per-family solves and the vertex enumeration in this
        # process; under n_jobs > 1 the per-family solves run in worker
        # processes and emit on their own streams.
        pyomo_logger = logging.getLogger('pyomo')
        prev_level = pyomo_logger.level
        if verbosity < 2:
            pyomo_logger.setLevel(logging.ERROR)
        try:
            cn = CredalNetwork.from_lcn(
                lcn, method=method, solver=solver, time_limit=time_limit,
                gap_tol=gap_tol, n_jobs=n_jobs, merge_budget=merge_budget,
                solve_families=solve_families, verbosity=verbosity,
                use_highs=use_highs)
            cnv = cls(cn)
            cnv._build(verbosity=verbosity,
                       enumerate_vertices=enumerate_vertices,
                       n_jobs=n_jobs)
        finally:
            pyomo_logger.setLevel(prev_level)
        cnv.build_time = time.perf_counter() - t0
        if verbosity > 0:
            print(f"[CredalNetworkVertices] Build time (whole pipeline): "
                  f"{cnv.build_time:.4f} sec")
        return cnv

    def _summary(self) -> Dict:
        n_vertices = 0
        if self.extreme_points is not None:
            for configs in self.extreme_points.values():
                for vertices in configs.values():
                    n_vertices += len(vertices)
        return {
            "build_time": self.build_time,
            "n_nodes": len(self.cn.nodes),
            "n_vertices": n_vertices,
        }

    def _build(self, verbosity: int = 1, enumerate_vertices: bool = True,
               n_jobs: int = 1):
        """Build bn_min/bn_max, and (unless enumerate_vertices is False) the
        parsed extreme points.

        The LRS enumeration is timed and stored in ``self.enumeration_time``
        (0.0 when enumeration is skipped). It runs serially on the monolithic
        ``gum.CredalNet`` -- it is not parallelized (LRS is cheap and
        treewidth-bounded, and pyAgrum's LRS holds the GIL and is not
        thread-safe). ``n_jobs`` is accepted only to drive the compile-time
        per-family solves in :meth:`CredalNetwork.from_lcn`; it has no effect on
        enumeration.
        """
        cn = self.cn
        node_names = cn.nodes
        node_card = cn.node_card
        node_atoms = cn.node_atoms

        # Build two BayesNets (bn_min for lower, bn_max for upper)
        self.bn_min = gum.BayesNet("min")
        self.bn_max = gum.BayesNet("max")
        node_ids_min = {}
        node_ids_max = {}

        for name in node_names:
            card = node_card[name]
            nid_min = self.bn_min.add(gum.LabelizedVariable(name, name, card))
            nid_max = self.bn_max.add(gum.LabelizedVariable(name, name, card))
            node_ids_min[name] = nid_min
            node_ids_max[name] = nid_max

        # Add arcs from parents to children. Use the built factors (not
        # cn.lcn.families) so the arcs match the possibly-merged factor
        # structure; at merge_budget == 1 the factor children/parents are
        # exactly the LCN families, so this is identical to the unmerged arcs.
        for factor in cn.factors:
            sample_entry = factor[0]
            child = sample_entry["child"]
            for parent in sample_entry["parents"]:
                self.bn_min.addArc(node_ids_min[parent], node_ids_min[child])
                self.bn_max.addArc(node_ids_max[parent], node_ids_max[child])

        # Fill CPTs from the interval factors
        for factor in cn.factors:
            sample_entry = factor[0]
            child_name = sample_entry["child"]
            parent_names = sample_entry["parents"]

            child_atoms = node_atoms[child_name]
            n_child_states = node_card[child_name]
            n_child_atoms = len(child_atoms)

            # Map (parent_config, child_state) -> (lobo, upbo)
            bounds = {}
            for i, entry in factor.items():
                interp = entry["interpretation"]
                child_vals = interp[:n_child_atoms]
                parent_vals = interp[n_child_atoms:]
                # Convert child vals to a state index (binary -> int)
                child_state = 0
                for bit in child_vals:
                    child_state = (child_state << 1) | bit
                parent_config = tuple(parent_vals)
                bounds[(parent_config, child_state)] = (entry["lobo"], entry["upbo"])

            nid_min = node_ids_min[child_name]
            nid_max = node_ids_max[child_name]

            if len(parent_names) == 0:
                # No parents: just fill the marginal
                lower_vals = []
                upper_vals = []
                for cs in range(n_child_states):
                    lo, up = bounds.get(((), cs), (0.0, 1.0))
                    lo = lo if lo is not None else 0.0
                    up = up if up is not None else 1.0
                    lower_vals.append(abs(lo))
                    upper_vals.append(abs(up))
                self.bn_min.cpt(nid_min).fillWith(lower_vals)
                self.bn_max.cpt(nid_max).fillWith(upper_vals)
            else:
                # With parents: iterate using pyAgrum's Instantiation order
                # to build the flat CPT array
                parent_atoms = []
                for pname in parent_names:
                    parent_atoms.extend(node_atoms[pname])

                inst = gum.Instantiation(self.bn_min.cpt(nid_min))
                lower_flat = [0.0] * inst.domainSize()
                upper_flat = [0.0] * inst.domainSize()

                inst.setFirst()
                flat_idx = 0
                while not inst.end():
                    # Extract child state from the instantiation
                    child_state = inst.val(inst.variable(child_name))

                    # Extract parent values in scope order (matching factorization)
                    parent_config = []
                    for patom in parent_atoms:
                        # Find which node this atom belongs to
                        for pname in parent_names:
                            if patom in node_atoms[pname]:
                                pnode = pname
                                break
                        p_node_val = inst.val(inst.variable(pnode))
                        p_node_atoms = node_atoms[pnode]
                        if len(p_node_atoms) == 1:
                            parent_config.append(p_node_val)
                        else:
                            # Compound parent: decode the state into individual bits
                            n_bits = len(p_node_atoms)
                            atom_idx = p_node_atoms.index(patom)
                            bit = (p_node_val >> (n_bits - 1 - atom_idx)) & 1
                            parent_config.append(bit)

                    parent_config = tuple(parent_config)
                    lo, up = bounds.get((parent_config, child_state), (0.0, 1.0))
                    lo = lo if lo is not None else 0.0
                    up = up if up is not None else 1.0
                    lower_flat[flat_idx] = abs(lo)
                    upper_flat[flat_idx] = abs(up)
                    flat_idx += 1
                    inst.inc()

                self.bn_min.cpt(nid_min).fillWith(lower_flat)
                self.bn_max.cpt(nid_max).fillWith(upper_flat)

        if verbosity > 0:
            print("[CredalNetworkVertices] Lower BN CPTs:")
            for name in node_names:
                print(f"  {name}: {self.bn_min.cpt(node_ids_min[name])}")
            print("[CredalNetworkVertices] Upper BN CPTs:")
            for name in node_names:
                print(f"  {name}: {self.bn_max.cpt(node_ids_max[name])}")

        if not enumerate_vertices:
            # Vertex-free build (e.g. CredalJT / scheme D5): the interval
            # local credal sets carried by bn_min/bn_max are all the consumer
            # needs; skip the (potentially expensive) LRS enumeration.
            self.credal_net = None
            self.extreme_points = None
            self.enumeration_time = 0.0
            if verbosity > 0:
                print("[CredalNetworkVertices] Skipping LRS extreme-point "
                      "enumeration (enumerate_vertices=False).")
            return

        # Run LRS vertex enumeration (timed), serially, on the monolithic
        # gum.CredalNet. Enumeration is NOT parallelized: it is cheap
        # (treewidth-bounded per docs/lrs.tex -- typically milliseconds), and
        # pyAgrum's LRS (intervalToCredal) both holds the GIL and is not
        # thread-safe (aGrUM vendors classic lrslib.c, whose file-scope globals
        # lrs_global_list/lrs_global_count are shared across calls), so threads
        # cannot parallelize it and a process pool was pure overhead (spawn +
        # per-worker `import pyagrum` + pickling dwarfed the LRS work, and it was
        # a hang source). n_jobs is retained on the signature only because it
        # drives the compile-time per-family solves in CredalNetwork.from_lcn.
        t_enum = time.perf_counter()
        self.credal_net = gum.CredalNet(self.bn_min, self.bn_max)
        self.credal_net.intervalToCredal()
        self.extreme_points = _parse_credal_net_vertices(self.credal_net)
        self.enumeration_time = time.perf_counter() - t_enum

        if verbosity > 0 and self.credal_net is not None:
            print("[CredalNetworkVertices] CredalNet vertices:")
            print(self.credal_net)

        if verbosity > 0:
            print("[CredalNetworkVertices] Extreme points per node:")
            for node, configs in self.extreme_points.items():
                for config, vertices in configs.items():
                    print(f"  {node} {config}: {len(vertices)} vertices")
                    for v in vertices:
                        v_str = ", ".join(f"{x:.4f}" for x in v)
                        print(f"    [{v_str}]")

    # ------------------------------------------------------------------
    # Serialization: the enumerated extreme points as a portable .vtx file
    # ------------------------------------------------------------------

    # Bump when the on-disk .vtx schema changes incompatibly.
    VTX_FORMAT_VERSION = 1

    def save_vtx(self, file_name: str, method: str = None,
                 merge_budget: int = None, solver: str = None,
                 enumeration_time: float = None, compile_time: float = None,
                 n_jobs: int = None) -> None:
        """
        Serialize the enumerated extreme points to a JSON ``.vtx`` file.

        Stores the LRS output (``extreme_points``: {node: {config: [[...]]}}) with
        a provenance header. The ``(method, merge_budget, solver)`` triple is the
        cache-match key (an engine reuses a ``.vtx`` only when these match its
        requested build). ``enumeration_time`` is the LRS wall-clock;
        ``compile_time`` is carried through from the ``.cn`` so a cache hit can
        report ``build_time = compile_time + enumeration_time``. ``n_jobs`` is
        informational (results are worker-count invariant).
        """
        assert self.extreme_points is not None, \
            "No extreme points to save (enumerate_vertices was False)."
        n_vertices = sum(len(vs) for cfgs in self.extreme_points.values()
                         for vs in cfgs.values())
        doc = {
            "format": "lcn-credal-vertices",
            "version": self.VTX_FORMAT_VERSION,
            "method": method,
            "merge_budget": merge_budget,
            "solver": solver,
            "enumeration_time": enumeration_time,
            "compile_time": compile_time,
            "n_jobs": n_jobs,
            "n_vertices": n_vertices,
            "extreme_points": self.extreme_points,
        }
        with open(file_name, "w") as f:
            json.dump(doc, f, indent=2)

    @classmethod
    def vtx_metadata(cls, file_name: str) -> Dict:
        """
        Read only the header of a ``.vtx`` file (format/version and provenance:
        method/merge_budget/solver/enumeration_time/compile_time/n_jobs) WITHOUT
        loading the extreme points. Returns the header dict, or ``None`` if the
        file is absent, unreadable, not valid JSON, or not a recognized ``.vtx``
        of the current format version.
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
        if doc.get("format") != "lcn-credal-vertices":
            return None
        if doc.get("version") != cls.VTX_FORMAT_VERSION:
            return None
        return {
            "format": doc.get("format"),
            "version": doc.get("version"),
            "method": doc.get("method"),
            "merge_budget": doc.get("merge_budget"),
            "solver": doc.get("solver"),
            "enumeration_time": doc.get("enumeration_time"),
            "compile_time": doc.get("compile_time"),
            "n_jobs": doc.get("n_jobs"),
        }

    @classmethod
    def load_extreme_points(cls, file_name: str):
        """
        Load the extreme points and header from a ``.vtx`` file.

        Returns ``(extreme_points, meta)`` on success, or ``None`` if the file is
        absent/invalid. ``extreme_points`` is the {node: {config: [[...]]}} dict
        the engines consume; ``meta`` is the header from :meth:`vtx_metadata`.
        """
        meta = cls.vtx_metadata(file_name)
        if meta is None:
            return None
        with open(file_name) as f:
            doc = json.load(f)
        return doc.get("extreme_points"), meta


if __name__ == "__main__":

    file_name = "examples/alarm.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    cnv = CredalNetworkVertices.from_lcn(l, method="linear", verbosity=1)
    print(f"\n[CredalNetworkVertices] build_time = {cnv.build_time:.4f} sec")
