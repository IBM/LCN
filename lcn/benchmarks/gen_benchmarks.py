"""Master benchmark generator for LCN instances.

Generates LCN problem instances of different graph topologies and sizes.
Each instance is saved as a .lcn file in a per-topology subdirectory.

Usage examples:
    # Generate all default benchmarks (chain, polytree, random)
    python -m lcn.benchmarks.gen_benchmarks

    # Only chains and polytrees, sizes 5 and 10
    python -m lcn.benchmarks.gen_benchmarks --types chain polytree --sizes 5 10

    # All types including DAG, more instances
    python -m lcn.benchmarks.gen_benchmarks --types chain polytree dag random \
        --sizes 5 10 20 50 --num-instances 5 --output-dir my_benchmarks

    # Easy-to-solve-globally instances (quick for the SCIP global solver)
    python -m lcn.benchmarks.gen_benchmarks --types easy --sizes 5 8 10
"""

import os
import argparse

from lcn.benchmarks.generator import Generator

TOPOLOGIES = ["chain", "polytree", "random", "dag"]
# "easy" is available via --types but excluded from the default set (it targets
# SCIP-easy instances rather than a graph topology; see Generator.generate).
ALL_TYPES = TOPOLOGIES + ["easy"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate LCN benchmark instances of different types and sizes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s                                    # all defaults
  %(prog)s --types chain --sizes 5 10         # chains only, small
  %(prog)s --types chain polytree dag random  # all topologies
  %(prog)s --num-instances 5 --seed 123       # more instances, custom seed
""")
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks",
        help="root output directory (default: benchmarks)")
    parser.add_argument(
        "--types", type=str, nargs="+", default=TOPOLOGIES,
        choices=ALL_TYPES,
        help="instance types to generate (default: chain polytree random dag; "
             "'easy' produces SCIP-easy instances)")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[5, 10, 15, 20, 30],
        help="number of variables per instance (default: 5 10 15 20 30)")
    parser.add_argument(
        "--num-instances", type=int, default=3,
        help="instances per (type, size) pair (default: 3)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed (default: 42)")
    parser.add_argument(
        "--epsilon", type=float, default=0.3,
        help="half-width of probability interval (default: 0.3)")
    parser.add_argument(
        "--max-vars", type=int, default=3,
        help="max variables per sentence formula (default: 3)")
    parser.add_argument(
        "--num-extras", type=int, default=2,
        help="extra marginal sentences per instance (default: 2)")
    parser.add_argument(
        "--max-component-size", type=int, default=3,
        help="max variables per chain component (default: 3, chain only)")
    parser.add_argument(
        "--max-parents", type=int, default=2,
        help="max parents per child node (default: 2, polytree/dag only)")
    parser.add_argument(
        "--strategy", type=str, default="linear",
        choices=["linear", "sparse", "bounded", "verified"],
        help="easy-instance strategy (default: linear, 'easy' type only)")
    parser.add_argument(
        "--coverage", type=float, default=1.0,
        help="predecessor coverage for easy/linear (1.0 => zero independence "
             "assertions; default: 1.0, 'easy' type only)")
    parser.add_argument(
        "--core", type=int, default=3,
        help="leading sparse-spine size for easy/sparse "
             "(~core-2 small assertions; default: 3)")
    parser.add_argument(
        "--difficulty-cap", type=int, default=64,
        help="max single 2^(|Y|+|S|) for easy/bounded (default: 64)")
    parser.add_argument(
        "--base-topology", type=str, default="polytree",
        choices=["dag", "polytree", "chain"],
        help="candidate topology for easy/bounded (default: polytree)")
    parser.add_argument(
        "--verify-time-limit", type=float, default=5.0,
        help="per-solve SCIP limit for easy/verified (default: 5.0s)")
    parser.add_argument(
        "--verbosity", type=int, default=1,
        help="verbosity level: 0=silent, 1=summary (default: 1)")
    args = parser.parse_args()

    gen = Generator(seed=args.seed)
    total_saved = 0

    for graph_type in args.types:
        type_dir = os.path.join(args.output_dir, graph_type)
        os.makedirs(type_dir, exist_ok=True)

        if args.verbosity > 0:
            print(f"\n--- Generating {graph_type} instances ---")

        for n in args.sizes:
            instances = gen.generate(
                num_vars=n,
                graph_type=graph_type,
                num_instances=args.num_instances,
                max_vars_per_sentence=args.max_vars,
                num_extras=args.num_extras,
                epsilon=args.epsilon,
                max_component_size=args.max_component_size,
                max_parents=args.max_parents,
                strategy=args.strategy,
                coverage=args.coverage,
                core=args.core,
                difficulty_cap=args.difficulty_cap,
                base_topology=args.base_topology,
                verify_time_limit=args.verify_time_limit,
                verbosity=max(0, args.verbosity - 1),
            )
            # Encode the strategy in easy filenames (mirrors gen_easy.py) so
            # linear/bounded/verified sets don't overwrite one another.
            prefix = (f"easy_{args.strategy}" if graph_type == "easy"
                      else graph_type)
            for i, lcn in enumerate(instances):
                fname = os.path.join(
                    type_dir, f"{prefix}_n{n}_{i + 1}.lcn")
                gen.save(lcn, fname)
                total_saved += 1
                if args.verbosity > 0:
                    print(f"  Saved {fname} ({len(lcn.sentences)} sentences)")

    if args.verbosity > 0:
        expected = len(args.types) * len(args.sizes) * args.num_instances
        print(f"\nDone. Saved {total_saved}/{expected} instances "
              f"in {args.output_dir}/")


if __name__ == "__main__":
    main()
