"""Generate easy-to-solve-globally LCN benchmark instances.

These instances are designed so the SCIP global marginal solver certifies every
singleton bound quickly (gap=0, no timeouts). The difficulty comes from the
bilinear Local Markov constraints in the global model, whose count is
sum over assertions of 2^(|Y|+|S|); the "linear" strategy conditions each atom
on its predecessors so those assertions vanish (full coverage => pure LP). See
``Generator.generate`` / ``estimate_difficulty`` for details.
"""

import os
import csv
import argparse

from lcn.benchmarks.generator import Generator, estimate_difficulty


def generate(output_dir, sizes, num_instances, seed, epsilon, strategy,
             coverage, core, difficulty_cap, base_topology, verify_time_limit,
             max_vars, num_extras, report, verbosity):
    os.makedirs(output_dir, exist_ok=True)
    gen = Generator(seed=seed)

    rows = []
    for n in sizes:
        instances = gen.generate(
            num_vars=n,
            graph_type="easy",
            num_instances=num_instances,
            strategy=strategy,
            coverage=coverage,
            core=core,
            difficulty_cap=difficulty_cap,
            base_topology=base_topology,
            verify_time_limit=verify_time_limit,
            max_vars_per_sentence=max_vars,
            num_extras=num_extras,
            epsilon=epsilon,
            verbosity=max(0, verbosity - 1),
        )
        for i, lcn in enumerate(instances):
            fname = os.path.join(output_dir,
                                 f"easy_{strategy}_n{n}_{i + 1}.lcn")
            gen.save(lcn, fname)
            if verbosity > 0:
                print(f"Saved {fname} ({len(lcn.sentences)} sentences)")
            if report:
                d = estimate_difficulty(lcn)
                rows.append({"file": os.path.basename(fname), "n": n,
                             "num_assertions": d["num_assertions"],
                             "total_bilinear": d["total_bilinear"],
                             "max_single": d["max_single"],
                             "predicted": d["predicted"]})
                if verbosity > 0:
                    print(f"  difficulty: assertions={d['num_assertions']} "
                          f"total_bilinear={d['total_bilinear']} "
                          f"max_single={d['max_single']} "
                          f"predicted={d['predicted']}")

    if report and rows:
        csv_path = os.path.join(output_dir, "difficulty_stats.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        if verbosity > 0:
            print(f"Wrote difficulty report to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate easy-to-solve-globally LCN benchmark instances.")
    parser.add_argument("--output-dir", type=str, default="benchmarks/easy",
                        help="Output directory (default: benchmarks/easy)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 8, 10],
                        help="Number of variables (default: 5 8 10)")
    parser.add_argument("--num-instances", type=int, default=3,
                        help="Instances per size (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--epsilon", type=float, default=0.4,
                        help="Half-width of probability interval (default: 0.4)")
    parser.add_argument("--strategy", type=str, default="linear",
                        choices=["linear", "sparse", "bounded", "verified"],
                        help="Easy-generation strategy (default: linear)")
    parser.add_argument("--coverage", type=float, default=1.0,
                        help="Predecessor coverage for linear/verified "
                             "(1.0 => zero independence assertions; default: 1.0)")
    parser.add_argument("--core", type=int, default=3,
                        help="Leading sparse-spine size for 'sparse' "
                             "(~core-2 small assertions; default: 3)")
    parser.add_argument("--difficulty-cap", type=int, default=64,
                        help="Max single 2^(|Y|+|S|) for 'bounded' (default: 64)")
    parser.add_argument("--base-topology", type=str, default="polytree",
                        choices=["dag", "polytree", "chain"],
                        help="Candidate topology for 'bounded' (default: polytree)")
    parser.add_argument("--verify-time-limit", type=float, default=5.0,
                        help="Per-solve SCIP limit for 'verified' (default: 5.0s)")
    parser.add_argument("--max-vars", type=int, default=3,
                        help="Max variables per child formula (default: 3)")
    parser.add_argument("--num-extras", type=int, default=0,
                        help="Extra marginal sentences per instance (default: 0)")
    parser.add_argument("--report", action="store_true",
                        help="Compute and save a difficulty report (CSV).")
    parser.add_argument("--verbosity", type=int, default=1,
                        help="Verbosity level (default: 1)")
    args = parser.parse_args()
    generate(args.output_dir, args.sizes, args.num_instances, args.seed,
             args.epsilon, args.strategy, args.coverage, args.core,
             args.difficulty_cap, args.base_topology, args.verify_time_limit,
             args.max_vars, args.num_extras, args.report, args.verbosity)
