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

# Credal-network compiler CLI.
#
# Compiles an input LCN ``.lcn`` file into its corresponding credal network
# (the chain-graph factorization together with the interval local credal sets
# P(child | parents)) and writes it, as JSON, to a ``.cn`` file saved alongside
# the input (same basename, ``.cn`` extension).
#
# The per-family interval solves are the expensive step; they parallelize
# across worker threads via ``--n-jobs`` (each individual min/max solve is one
# task on a shared ThreadPoolExecutor, handled by CredalNetwork.from_lcn -- the
# solves spend their time in the ipopt subprocess, which releases the GIL).
# Extreme points are NOT enumerated here -- the stored ``.cn`` carries the
# interval local credal sets only and stays pyAgrum-free; engines derive
# vertices on demand.
#
# Usage:
#   python -m lcn.inference.marginal.cn.compile_cn examples/alarm.lcn --n-jobs 5
#   python -m lcn.inference.marginal.cn.compile_cn foo.lcn -o bar.cn --method linear-tight

import argparse
import os
import time

from lcn.core.model import LCN
from lcn.inference.marginal.cn.credal_network import CredalNetwork


def compile_lcn_to_cn(
        input_file: str,
        output_file: str = None,
        method: str = "linear",
        solver: str = "ipopt",
        merge_budget: int = 1,
        n_jobs: int = 1,
        time_limit: float = None,
        gap_tol: float = 0.0,
        verbosity: int = 1,
        use_highs: bool = True) -> str:
    """
    Compile an LCN file into a credal-network ``.cn`` file.

    Args:
        input_file: str
            Path to the input ``.lcn`` file.
        output_file: str or None
            Path to the output ``.cn`` file. When None (the default), the
            output is the input path with its extension replaced by ``.cn``
            (saved alongside the input).
        method, solver, merge_budget, time_limit, gap_tol:
            Passed through to :meth:`CredalNetwork.from_lcn`.
        n_jobs: int
            Number of worker threads for the per-family interval solves.
        verbosity: int
            Verbosity level (0 is silent).
        use_highs: bool
            When True (default), solve each "linear"/"ipopt" per-family LP
            in-process with HiGHS (numerically identical, far faster); set
            False to force the legacy ipopt subprocess path. No effect for
            method "linear-tight" or solver "scip".

    Returns:
        The path to the written ``.cn`` file.
    """
    if output_file is None:
        base, _ = os.path.splitext(input_file)
        output_file = base + ".cn"

    lcn = LCN()
    lcn.from_lcn(file_name=input_file)

    if verbosity > 0:
        print(f"[compile_cn] Compiling {input_file} "
              f"(method={method}, solver={solver}, merge_budget={merge_budget}, "
              f"n_jobs={n_jobs}) ...")

    t0 = time.time()
    cn = CredalNetwork.from_lcn(
        lcn, method=method, solver=solver, time_limit=time_limit,
        gap_tol=gap_tol, n_jobs=n_jobs, merge_budget=merge_budget,
        solve_families=True, verbosity=verbosity, use_highs=use_highs)
    elapsed = time.time() - t0

    cn.save_cn(output_file, method=method, merge_budget=merge_budget,
               solver=solver, compile_time=round(elapsed, 4), n_jobs=n_jobs)

    if verbosity > 0:
        print(f"[compile_cn] Wrote {output_file} "
              f"({len(cn.factors)} factors, {elapsed:.2f}s).")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Compile an LCN .lcn file into its credal network (.cn).")
    parser.add_argument("input", help="Path to the input .lcn file.")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output .cn path (default: input path with a .cn extension).")
    parser.add_argument(
        "--method", choices=("linear", "linear-tight"), default="linear",
        help="Factorization method (default: linear).")
    parser.add_argument(
        "--solver", choices=("ipopt", "scip"), default="ipopt",
        help="Per-family solver backend (default: ipopt).")
    parser.add_argument(
        "--merge-budget", type=int, default=1,
        help="Scheme D2 max merged super-family scope (default: 1, no merge).")
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Worker threads for the per-family solves (default: 1).")
    parser.add_argument(
        "--time-limit", type=float, default=None,
        help="Per-solve wall-clock limit in seconds (default: none).")
    parser.add_argument(
        "--gap-tol", type=float, default=0.0,
        help="SCIP relative optimality gap to stop at (ignored by ipopt).")
    parser.add_argument(
        "-v", "--verbosity", type=int, default=1,
        help="Verbosity level (0 is silent; default: 1).")
    parser.add_argument(
        "--no-highs", dest="use_highs", action="store_false",
        help="Force the legacy ipopt subprocess path for method=linear/"
             "solver=ipopt (default: use the in-process HiGHS LP).")
    args = parser.parse_args()

    compile_lcn_to_cn(
        input_file=args.input,
        output_file=args.output,
        method=args.method,
        solver=args.solver,
        merge_budget=args.merge_budget,
        n_jobs=args.n_jobs,
        time_limit=args.time_limit,
        gap_tol=args.gap_tol,
        verbosity=args.verbosity,
        use_highs=args.use_highs)


if __name__ == "__main__":
    main()
