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

# Verifier: does the chain-graph credal network support the LCN's LMC?
#
# An LCN induces a set of conditional-independence assertions via the Local
# Markov Condition (LCN.local_markov_condition(), stored on lcn.independencies).
# Separately, the cn/ pipeline compiles the LCN's chain-graph factorization into
# a credal network: a directed structure
#
#     P(V) = prod_node  P(node | parents)
#
# whose local factors are interval credal sets, with the extreme points of every
# local set enumerated by CredalNetworkVertices.
#
# Theory (docs/ISIPTA2025_LCNs.pdf Sec. 3, docs/LCN_IJAR_Revised.pdf): for a CHAIN
# LCN the chain-graph Gibbs factorization reproduces EXACTLY the LMC
# independencies (the LMC coincides with the chain-graph global Markov condition,
# under positivity). This script is an executable check of that claim on a given
# instance.
#
# WHAT IT CHECKS
# Choosing one extreme point per local credal set (per node, per parent
# configuration) yields a fully specified product Bayesian network -- i.e. ONE
# joint distribution P over all atoms:
#
#     P(atom_config) = prod_node  vertex_node[ parent_config ][ child_state ]
#
# The extreme points of the JOINT credal set are exactly these products of local
# extreme points. So the credal network "supports" an assertion (X _||_ Y | S)
# iff EVERY such product joint satisfies the bilinear LMC equality
#
#     P(x,y,s) * P(s) = P(x,s) * P(y,s)     (for all configurations)
#
# which is precisely the residual that lmc_constraint_groups_vec() encodes (and
# that verify_bruteforce.py already trusts). The test is therefore exact
# (closed-form joints + algebraic residuals) and uses no solver -- results are
# deterministic and independent of ipopt/SCIP. The --solver flag only affects how
# the upstream interval bounds (hence the vertices) were computed; tiny non-zero
# residuals (~1e-7) just reflect that solver precision, which is what --tol allows.
#
# SCOPE
# Targets chain LCNs (structure is a chain graph) -- exactly the class for which
# the factorization is claimed to reproduce the LMC. The numerical test runs on
# any instance the cn/ pipeline can build; a FAIL is the informative outcome.
#
# COST. The verifier's own work (joint reconstruction + residual evaluation) is
# vectorized and cheap. Wall-clock is dominated by the UPSTREAM build
# (CredalNetworkVertices.from_lcn: per-family interval solves + LRS vertex
# enumeration), which is expensive for instances with large compound nodes
# (e.g. smokers.lcn has card-8 nodes and takes minutes just to build). When the
# number of joint extreme-point combinations exceeds --max-joints the verifier
# samples instead of enumerating (and says so) -- a SUPPORTED verdict under
# sampling is evidence, not a proof, over the whole joint credal set.
#
# PROGRESS. The two phases are reported separately so a long run is observable:
# a "[1/2] Building credal network..." heads-up before the slow build and its
# elapsed time + node/vertex counts after, then "[2/2] Verifying..." with a
# tqdm progress bar over the tested joints (showing the running worst residual).
# A final "Time: build=..s, verify=..s" line makes the build/verify split
# explicit. Use --verbosity 2 to stream the credal-network build log (the bar is
# then suppressed) and --no-progress-bar / --verbosity 0 to silence the bar.

import argparse
import itertools
import time

import numpy as np
from tqdm import tqdm

# Local
from lcn.core.model import LCN
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.utils.common import (
    build_truth_table,
    lmc_constraint_groups_vec,
)


# ----------------------------------------------------------------------
# Joint reconstruction from a choice of local extreme points
# ----------------------------------------------------------------------
def _node_state(node_atoms_of_node, row, col_of):
    """
    Encode the state index of a (possibly compound) node from a truth-table row.

    Mirrors the child-state encoding in cn/vertices.py: the atoms of the node are
    packed MSB-first, so for a compound node "C-D" with C,D the state is
    (C << 1) | D. For a singleton node it is just the atom's bit.
    """
    st = 0
    for a in node_atoms_of_node:
        st = (st << 1) | int(row[col_of[a]])
    return st


def _parent_config_str(parents, node_atoms, row, col_of):
    """
    Build the parent-configuration key string in the SAME format used as the
    keys of CredalNetworkVertices.extreme_points (the pyAgrum CredalNet print
    format), e.g. "<>", "<A:0>", "<B:0|E:1>".

    Parent order follows the family's `parents` list; a compound parent
    contributes its packed integer state (matching cn/vertices.py decoding).
    """
    if not parents:
        return "<>"
    parts = []
    for p in parents:
        st = _node_state(node_atoms[p], row, col_of)
        parts.append(f"{p}:{st}")
    return "<" + "|".join(parts) + ">"


def _build_residuals(lcn, table, col_of):
    """
    Build the list of LMC residual callables p -> float (one per constraint
    group, per assertion). Each must be ~0 for a distribution that satisfies the
    independence. Reuses lmc_constraint_groups_vec (the same construction as
    verify_bruteforce.py). Returns a list of (assertion_str, residual_fn).
    """
    residuals = []
    for indep in lcn.independencies.get_assertions():
        astr = str(indep)
        for group in lmc_constraint_groups_vec(indep, table, col_of):
            if group[0] == 'conditional':
                _, Aa, Ab, Ac, Ad = group
                residuals.append((astr, lambda p, Aa=Aa, Ab=Ab, Ac=Ac, Ad=Ad:
                                  float(Aa @ p) * float(Ab @ p)
                                  - float(Ac @ p) * float(Ad @ p)))
            else:  # 'marginal'
                _, Aa, Ab, Ac = group
                residuals.append((astr, lambda p, Aa=Aa, Ab=Ab, Ac=Ac:
                                  float(Aa @ p) - float(Ab @ p) * float(Ac @ p)))
    return residuals


class _JointReconstructor:
    """
    Reconstructs the joint P over all atoms from a choice of one local extreme
    point per (node, parent_config).

    For every node it precomputes, for each of the 2^n truth-table rows, the
    parent-config key and the child state. These are baked into numpy index
    arrays so that building one joint is a handful of vectorized fancy-index
    multiplies (no per-row Python loop) -- the per-joint cost stays negligible
    even at n ~ 9 (512 rows) where many joints are sampled.
    """

    def __init__(self, cnv, table, col_of):
        self.table = table
        self.nodes = cnv.cn.nodes
        node_atoms = cnv.cn.node_atoms
        fam_by_child = {f["child"]: f for f in cnv.lcn.families}
        n_rows = len(table)

        # Per node, give each distinct parent-config a contiguous integer id, and
        # store row->config_id and row->child_state as numpy index arrays. A
        # joint then reads, per node, a (n_configs, card) matrix of the chosen
        # vertices and gathers it with p *= mat[config_of_row, state_of_row] --
        # fully vectorized, no per-row Python.
        self.config_ids = {}       # node -> {pc_str: int}
        self.config_of_row = {}    # node -> np.ndarray[int] (len n_rows)
        self.state_of_row = {}     # node -> np.ndarray[int] (len n_rows)
        for node in self.nodes:
            parents = fam_by_child[node]["parents"]
            ids = {}
            cfg_row = np.empty(n_rows, dtype=np.intp)
            state_row = np.empty(n_rows, dtype=np.intp)
            for r, row in enumerate(table):
                pc = _parent_config_str(parents, node_atoms, row, col_of)
                if pc not in ids:
                    ids[pc] = len(ids)
                cfg_row[r] = ids[pc]
                state_row[r] = _node_state(node_atoms[node], row, col_of)
            self.config_ids[node] = ids
            self.config_of_row[node] = cfg_row
            self.state_of_row[node] = state_row

    def joint(self, choice):
        """
        choice: dict (node, parent_config_str) -> vertex (list[float]).
        Returns the length-2^n joint as a numpy array.
        """
        p = np.ones(len(self.table), dtype=float)
        for node in self.nodes:
            ids = self.config_ids[node]
            # Stack the chosen vertices into a (n_configs, card) matrix in id order.
            mat = np.array([choice[(node, pc)] for pc, _ in
                            sorted(ids.items(), key=lambda kv: kv[1])],
                           dtype=float)
            p *= mat[self.config_of_row[node], self.state_of_row[node]]
        return p


# ----------------------------------------------------------------------
# Enumeration / sampling of vertex choices
# ----------------------------------------------------------------------
def _choice_keys(extreme_points):
    """
    Flatten the extreme_points dict into a list of (node, parent_config_str,
    vertices) triples -- one independent multiple-choice slot per local credal
    set / parent configuration.
    """
    keys = []
    for node, configs in extreme_points.items():
        for pc, vertices in configs.items():
            keys.append((node, pc, vertices))
    return keys


def _total_combinations(keys):
    total = 1
    for _, _, vertices in keys:
        total *= max(1, len(vertices))
    return total


def _iter_choices(keys, max_joints, seed):
    """
    Yield (choice_dict, exhaustive_flag) over the vertex-choice space.

    If the number of combinations is <= max_joints, enumerate ALL of them (an
    exhaustive check over the joint extreme points). Otherwise yield a
    deterministic seeded sample of max_joints choices, always including the
    all-first-vertex and all-last-vertex corners so the common cases are never
    skipped. The second return value reports whether coverage was exhaustive.
    """
    total = _total_combinations(keys)
    if total <= max_joints:
        option_lists = [range(len(v)) for _, _, v in keys]
        for combo in itertools.product(*option_lists):
            choice = {(node, pc): vertices[idx]
                      for (node, pc, vertices), idx in zip(keys, combo)}
            yield choice, True
        return

    # Sampled coverage. Emit the two canonical corners first, then random draws.
    def corner(pick):
        return {(node, pc): vertices[pick(vertices)]
                for (node, pc, vertices) in keys}

    yield corner(lambda v: 0), False
    yield corner(lambda v: len(v) - 1), False

    rng = np.random.default_rng(seed)
    for _ in range(max(0, max_joints - 2)):
        choice = {}
        for node, pc, vertices in keys:
            idx = int(rng.integers(0, len(vertices)))
            choice[(node, pc)] = vertices[idx]
        yield choice, False


# ----------------------------------------------------------------------
# Main verification routine
# ----------------------------------------------------------------------
def verify(lcn, method="linear", solver="ipopt", tol=1e-7,
           max_joints=20000, seed=12345, time_limit=None, gap_tol=0.0,
           verbosity=1, progress_bar=True):
    """
    Verify that the chain-graph credal network built from `lcn` supports every
    LMC independence of `lcn`.

    Args (progress-related):
        verbosity: int
            0 silent; 1 phase headers + a per-joint progress bar + report;
            2 additionally streams the credal-network build log and suppresses
            the bar (the build log would interleave with it).
        progress_bar: bool
            Show the tqdm progress bar over the verified joints (default True).
            Forced off at verbosity 0 and verbosity 2.

    Returns a dict summary: {"supported", "max_residual", "n_joints",
    "exhaustive", "per_assertion" (assertion_str -> max|residual|), "worst"
    ((assertion_str, residual) or None), "applicable" (False when the LCN is not
    a chain graph -- the factorization, and hence this check, does not apply),
    "build_time", "verify_time"}.
    """
    t_start = time.time()

    # The chain-graph factorization only exists for chain LCNs. Bail out cleanly
    # (rather than letting CredalNetwork.from_lcn's assertion raise) for cyclic
    # structures -- there the property is simply not applicable.
    if not lcn.is_chain_graph():
        if verbosity > 0:
            print("[verify_lmc] The LCN is NOT a chain graph -- its structure "
                  "admits no chain-graph factorization, so this check does not "
                  "apply. VERDICT: N/A")
        return {"applicable": False, "supported": None, "max_residual": None,
                "n_joints": 0, "exhaustive": None, "per_assertion": {},
                "worst": None, "build_time": 0.0, "verify_time": 0.0}

    # Build the credal network + enumerate extreme points. This dominates the
    # wall-clock for large compound-node instances, so announce it up front and
    # report how long it took (the loop below is comparatively cheap).
    if verbosity > 0:
        print(f"[verify_lmc] [1/2] Building credal network "
              f"(method={method}, solver={solver}) -- this can take a while "
              f"for large compound nodes...", flush=True)
    t_build = time.time()
    cnv = CredalNetworkVertices.from_lcn(
        lcn, method=method, solver=solver, time_limit=time_limit,
        gap_tol=gap_tol, verbosity=(verbosity - 1 if verbosity > 1 else 0))
    build_time = time.time() - t_build
    n_vertices = sum(len(v) for cfgs in cnv.extreme_points.values()
                     for v in cfgs.values())
    if verbosity > 0:
        print(f"[verify_lmc] [1/2] Credal network built in {build_time:.2f}s: "
              f"{len(cnv.cn.nodes)} nodes, {n_vertices} local extreme points.",
              flush=True)

    vars_list = list(lcn.atoms.keys())
    n = len(vars_list)
    table = build_truth_table(n)
    col_of = {v: i for i, v in enumerate(vars_list)}

    assertions = lcn.independencies.get_assertions()

    if verbosity > 0:
        factor_str = " ".join(
            f"P({f['child']}|{','.join(f['parents'])})" if f['parents']
            else f"P({f['child']})"
            for f in lcn.families)
        print(f"[verify_lmc] LCN atoms ({n}): {vars_list}")
        print(f"[verify_lmc] LMC independencies: {len(assertions)}")
        for a in assertions:
            print(f"[verify_lmc]   {a}")
        print(f"[verify_lmc] Credal net nodes: {cnv.cn.nodes}")
        print(f"[verify_lmc] Factorization: {factor_str}")

    # No independencies -> nothing to violate; trivially supported.
    if len(assertions) == 0:
        if verbosity > 0:
            print("[verify_lmc] No LMC independencies; trivially SUPPORTED.")
        return {"applicable": True, "supported": True, "max_residual": 0.0,
                "n_joints": 0, "exhaustive": True, "per_assertion": {},
                "worst": None, "build_time": build_time, "verify_time": 0.0}

    residuals = _build_residuals(lcn, table, col_of)
    recon = _JointReconstructor(cnv, table, col_of)
    keys = _choice_keys(cnv.extreme_points)
    total = _total_combinations(keys)
    exhaustive_possible = total <= max_joints

    # Number of joints we will actually test (used for the progress bar total).
    n_to_test = total if exhaustive_possible else max_joints

    if verbosity > 0:
        coverage = (f"exhaustive ({total} combinations)" if exhaustive_possible
                    else f"SAMPLED {max_joints} of {total} combinations "
                         f"(not exhaustive)")
        print(f"[verify_lmc] [2/2] Verifying LMC over joint extreme points: "
              f"{coverage}", flush=True)
        if not exhaustive_possible:
            print("[verify_lmc] WARNING: coverage is sampled, not exhaustive -- "
                  "a SUPPORTED verdict does not prove the property over the whole "
                  "joint credal set. Raise --max-joints to enumerate all.")

    per_assertion = {}
    max_residual = 0.0
    worst = None
    n_joints = 0
    exhaustive = True
    t_verify = time.time()

    # Per-joint progress bar; shows the running worst residual so a violation is
    # visible the moment it appears. Off at verbosity 0 and (to avoid interleaving
    # with the streamed build log) at verbosity 2.
    effective_pbar = progress_bar and verbosity == 1
    pbar = tqdm(total=n_to_test, desc="[verify_lmc] joints",
                disable=(not effective_pbar))
    try:
        for choice, is_exhaustive in _iter_choices(keys, max_joints, seed):
            exhaustive = exhaustive and is_exhaustive
            p = recon.joint(choice)

            # Self-check: a valid product factorization must sum to 1. A gross
            # deviation signals a reconstruction (indexing/parent-config) bug
            # rather than an LMC violation, so surface it loudly.
            s = float(p.sum())
            if abs(s - 1.0) > 1e-3:
                raise RuntimeError(
                    f"reconstructed joint does not sum to 1 (sum={s:.6f}); "
                    f"the joint reconstruction is inconsistent with the credal "
                    f"net layout -- this is a bug in the verifier, not an LMC "
                    f"violation.")

            for astr, fn in residuals:
                r = abs(fn(p))
                if r > per_assertion.get(astr, 0.0):
                    per_assertion[astr] = r
                if r > max_residual:
                    max_residual = r
                    worst = (astr, r)
            n_joints += 1
            pbar.set_postfix_str(f"max|res|={max_residual:.2e}")
            pbar.update(1)
    finally:
        pbar.close()

    verify_time = time.time() - t_verify
    supported = max_residual <= tol
    elapsed = time.time() - t_start

    if verbosity > 0:
        print("[verify_lmc] Per-assertion max |residual|:")
        for astr in sorted(per_assertion):
            ok = "OK" if per_assertion[astr] <= tol else "VIOLATED"
            print(f"[verify_lmc]   {astr}:  {per_assertion[astr]:.3e}  {ok}")
        verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
        cov = "exhaustive" if exhaustive else "sampled"
        print(f"[verify_lmc] VERDICT: {verdict}  "
              f"(max|residual|={max_residual:.3e} {'<=' if supported else '>'} "
              f"{tol:.1e}, {n_joints} joints, {cov})")
        if not supported and worst is not None:
            print(f"[verify_lmc] Worst offender: {worst[0]}  "
                  f"residual={worst[1]:.3e}")
        print(f"[verify_lmc] Time: build={build_time:.2f}s, "
              f"verify={verify_time:.2f}s, total={elapsed:.2f}s")

    return {"applicable": True, "supported": supported,
            "max_residual": max_residual,
            "n_joints": n_joints, "exhaustive": exhaustive,
            "per_assertion": per_assertion, "worst": worst,
            "build_time": build_time, "verify_time": verify_time}


def main():
    parser = argparse.ArgumentParser(
        description="Verify that the chain-graph credal network supports the "
                    "LCN's LMC independencies.")
    parser.add_argument("--file", default="examples/alarm.lcn",
                        help="Path to the .lcn file (default examples/alarm.lcn).")
    parser.add_argument("--method", choices=["linear", "nlp"], default="linear",
                        help="Factorization method for the local credal sets.")
    parser.add_argument("--solver", choices=["ipopt", "scip"], default="ipopt",
                        help="Backend for the per-family interval solves.")
    parser.add_argument("--tol", type=float, default=1e-7,
                        help="Max |LMC residual| tolerated (default 1e-7).")
    parser.add_argument("--max-joints", type=int, default=20000,
                        help="Enumerate all vertex-choice combinations when their "
                             "count is <= this, else sample this many (default "
                             "20000).")
    parser.add_argument("--seed", type=int, default=12345,
                        help="RNG seed for the sampled-coverage branch.")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Per-family solve time limit (seconds) for the "
                             "upstream credal-set build.")
    parser.add_argument("--gap", type=float, default=0.0,
                        help="SCIP relative gap for the upstream build "
                             "(ignored by ipopt).")
    parser.add_argument("--verbosity", type=int, default=1,
                        help="0 silent, 1 report + progress bar, 2 also stream "
                             "the credal-network build (suppresses the bar).")
    parser.add_argument("--no-progress-bar", action="store_true",
                        help="Disable the per-joint tqdm progress bar.")
    args = parser.parse_args()

    lcn = LCN()
    lcn.from_lcn(file_name=args.file)

    if args.verbosity > 0:
        print(f"=== LMC factorization verifier for {args.file} ===")

    result = verify(lcn, method=args.method, solver=args.solver, tol=args.tol,
                    max_joints=args.max_joints, seed=args.seed,
                    time_limit=args.time_limit, gap_tol=args.gap,
                    verbosity=args.verbosity,
                    progress_bar=not args.no_progress_bar)

    # CI-friendly exit codes: 0 = supported or not-applicable (not a chain
    # graph), 1 = a real LMC violation in an applicable instance.
    if not result["applicable"]:
        raise SystemExit(0)
    raise SystemExit(0 if result["supported"] else 1)


if __name__ == "__main__":
    main()
