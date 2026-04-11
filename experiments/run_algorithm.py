"""Run a single inference algorithm on a single LCN instance.

Core building block for the experimental pipeline. Can be used standalone
or called from run_experiment.py.

Usage:
    python experiments/run_algorithm.py \
        --instance benchmarks/chain/chain_n10_1.lcn \
        --algorithm ccte
"""

import argparse
import json
import time
import numpy as np

from lcn.core.model import LCN
from lcn.inference.marginal.exact import ExactInference
from lcn.inference.marginal.ariel import ArielInference
from lcn.inference.marginal.cve import CredalVE
from lcn.inference.marginal.ibp import IntervalBP
from lcn.inference.marginal.ccte import CredalCTE
from lcn.inference.marginal.approxlp import ApproxLP

ALGORITHMS = ["exact", "ariel", "ibp", "ccte", "approxlp"]


def _filter_singletons(results):
    """Keep only singleton variables (no '-' in name) from a results dict."""
    return {k: v for k, v in results.items() if "-" not in k}


def _marginals_to_dict(results):
    """Convert numpy marginals to JSON-serializable dict."""
    out = {}
    for var, (lo, hi) in results.items():
        out[var] = {
            "lower": [round(float(x), 8) for x in lo],
            "upper": [round(float(x), 8) for x in hi],
        }
    return out


def run_single(lcn_file, algorithm, evidence=None, verbosity=0, **kwargs):
    """
    Run one algorithm on one LCN instance.

    Args:
        lcn_file: path to .lcn file
        algorithm: one of "exact", "ariel", "ibp", "ccte", "approxlp"
        evidence: dict of evidence (default: {})
        verbosity: 0=silent
        **kwargs: algorithm-specific params

    Returns:
        dict with keys: algorithm, time_seconds, status, marginals, error
    """
    if evidence is None:
        evidence = {}

    # Load LCN
    l = LCN()
    l.from_lcn(file_name=lcn_file)

    result = {
        "algorithm": algorithm,
        "build_time": 0.0,
        "run_time": 0.0,
        "total_time": 0.0,
        "status": "ok",
        "marginals": {},
        "error": None,
    }

    try:
        t_start = time.time()

        if algorithm == "exact":
            algo = ExactInference(lcn=l)
            raw = algo.run(evidence=evidence, verbosity=verbosity)
            marginals = _filter_singletons(raw)
            t_end = time.time()
            result["run_time"] = round(t_end - t_start, 4)

        elif algorithm == "ariel":
            n_iters = kwargs.get("n_iters", 10)
            threshold = kwargs.get("threshold", 1e-6)
            algo = ArielInference(lcn=l)
            raw = algo.run(
                n_iters=n_iters, threshold=threshold,
                evidence=evidence, verbosity=verbosity)
            marginals = _filter_singletons(raw)
            t_end = time.time()
            result["run_time"] = round(t_end - t_start, 4)

        elif algorithm in ("ibp", "ccte", "approxlp"):
            # Build factorization (CredalVE) — timed separately
            t_build_start = time.time()
            cve = CredalVE(lcn=l)
            cve.build(verbosity=max(0, verbosity - 1))
            t_build_end = time.time()
            result["build_time"] = round(t_build_end - t_build_start, 4)

            # Run the algorithm — timed separately
            t_run_start = time.time()
            if algorithm == "ibp":
                n_iters = kwargs.get("n_iters", 100)
                threshold = kwargs.get("threshold", 1e-6)
                algo = IntervalBP(cve=cve)
                raw = algo.run(
                    evidence=evidence, n_iters=n_iters,
                    threshold=threshold, method="interval",
                    verbosity=verbosity)
            elif algorithm == "ccte":
                epsilon = kwargs.get("epsilon", None)
                algo = CredalCTE(cve=cve)
                raw = algo.run(
                    evidence=evidence, epsilon=epsilon,
                    verbosity=verbosity)
            elif algorithm == "approxlp":
                n_iters = kwargs.get("n_iters", 50)
                algo = ApproxLP(cve=cve)
                raw = algo.run(
                    evidence=evidence, n_iters=n_iters,
                    verbosity=verbosity)
            t_run_end = time.time()
            result["run_time"] = round(t_run_end - t_run_start, 4)

            marginals = _filter_singletons(raw)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        result["total_time"] = round(
            result["build_time"] + result["run_time"], 4)
        result["marginals"] = _marginals_to_dict(marginals)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["total_time"] = round(time.time() - t_start, 4)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run a single inference algorithm on an LCN instance.")
    parser.add_argument(
        "--instance", type=str, required=True,
        help="Path to the .lcn file")
    parser.add_argument(
        "--algorithm", type=str, required=True, choices=ALGORITHMS,
        help="Algorithm to run")
    parser.add_argument(
        "--evidence", type=str, default="{}",
        help="Evidence as JSON string (default: {})")
    parser.add_argument(
        "--verbosity", type=int, default=1,
        help="Verbosity level (default: 1)")
    args = parser.parse_args()

    evidence = json.loads(args.evidence)
    result = run_single(
        args.instance, args.algorithm,
        evidence=evidence, verbosity=args.verbosity)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
