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

import re
import time
from typing import Dict

import pyagrum as gum  # noqa: N813

from lcn.core.model import LCN
from lcn.inference.marginal.cn.credal_network import CredalNetwork


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

    @property
    def lcn(self) -> LCN:
        return self.cn.lcn

    def build(self, verbosity: int = 1) -> Dict:
        """
        Assemble the lower/upper BayesNets from the CredalNetwork's interval
        factors and enumerate the extreme points via LRS.

        Args:
            verbosity: int
                Verbosity level (0 is silent).

        Returns:
            A dict with the build summary: {"build_time", "n_nodes",
            "n_vertices"} (build_time covers only this method; for the
            whole-pipeline timing use from_lcn).
        """
        t0 = time.perf_counter()
        self._build(verbosity=verbosity)
        self.build_time = time.perf_counter() - t0
        return self._summary()

    @classmethod
    def from_lcn(cls, lcn: LCN, method: str = "linear",
                 solver: str = "ipopt", time_limit: float = None,
                 gap_tol: float = 0.0, n_jobs: int = 1,
                 verbosity: int = 1) -> "CredalNetworkVertices":
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
                Factorization method ("linear" or "nlp").
            solver: str
                Solver backend for the per-family solves: "ipopt" (default) or
                "scip" (global; requires the SCIP CLI on PATH).
            time_limit: float or None
                Per-solve wall-clock limit in seconds.
            gap_tol: float
                SCIP relative optimality gap (ignored by ipopt).
            n_jobs: int
                Worker processes for the per-family interval solves.
            verbosity: int
                Verbosity level (0 is silent).
        """
        t0 = time.perf_counter()
        cn = CredalNetwork.from_lcn(
            lcn, method=method, solver=solver, time_limit=time_limit,
            gap_tol=gap_tol, n_jobs=n_jobs, verbosity=verbosity)
        cnv = cls(cn)
        cnv._build(verbosity=verbosity)
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

    def _build(self, verbosity: int = 1):
        """Build bn_min/bn_max, the CredalNet, and parse the extreme points."""
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

        # Add arcs from parents to children (matching the simplified structure)
        for family in cn.lcn.families:
            child = family["child"]
            for parent in family["parents"]:
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

        # Create the CredalNet and run LRS vertex enumeration
        self.credal_net = gum.CredalNet(self.bn_min, self.bn_max)
        self.credal_net.intervalToCredal()

        if verbosity > 0:
            print("[CredalNetworkVertices] CredalNet vertices:")
            print(self.credal_net)

        # Extract and store extreme points
        self.extreme_points = _parse_credal_net_vertices(self.credal_net)

        if verbosity > 0:
            print("[CredalNetworkVertices] Extreme points per node:")
            for node, configs in self.extreme_points.items():
                for config, vertices in configs.items():
                    print(f"  {node} {config}: {len(vertices)} vertices")
                    for v in vertices:
                        v_str = ", ".join(f"{x:.4f}" for x in v)
                        print(f"    [{v_str}]")


if __name__ == "__main__":

    file_name = "examples/alarm.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    cnv = CredalNetworkVertices.from_lcn(l, method="linear", verbosity=1)
    print(f"\n[CredalNetworkVertices] build_time = {cnv.build_time:.4f} sec")
