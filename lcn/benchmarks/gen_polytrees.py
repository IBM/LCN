"""Generate polytree LCN benchmark instances of increasing size."""

import os
import argparse

from lcn.benchmarks.generator import Generator


def generate(output_dir, sizes, num_instances, seed, epsilon,
             max_vars, num_extras, verbosity):
    os.makedirs(output_dir, exist_ok=True)
    gen = Generator(seed=seed)

    for n in sizes:
        instances = gen.generate(
            num_vars=n,
            graph_type="polytree",
            num_instances=num_instances,
            max_vars_per_sentence=max_vars,
            num_extras=num_extras,
            epsilon=epsilon,
            verbosity=max(0, verbosity - 1),
        )
        for i, lcn in enumerate(instances):
            fname = os.path.join(output_dir, f"polytree_n{n}_{i + 1}.lcn")
            gen.save(lcn, fname)
            if verbosity > 0:
                print(f"Saved {fname} ({len(lcn.sentences)} sentences)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate polytree LCN benchmark instances.")
    parser.add_argument("--output-dir", type=str, default="benchmarks/polytree",
                        help="Output directory (default: benchmarks/polytree)")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[5, 10, 15, 20, 30],
                        help="Number of variables (default: 5 10 15 20 30)")
    parser.add_argument("--num-instances", type=int, default=3,
                        help="Instances per size (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="Half-width of probability interval (default: 0.3)")
    parser.add_argument("--max-vars", type=int, default=3,
                        help="Max variables per sentence formula (default: 3)")
    parser.add_argument("--num-extras", type=int, default=2,
                        help="Extra marginal sentences per instance (default: 2)")
    parser.add_argument("--verbosity", type=int, default=1,
                        help="Verbosity level (default: 1)")
    args = parser.parse_args()
    generate(args.output_dir, args.sizes, args.num_instances, args.seed,
             args.epsilon, args.max_vars, args.num_extras, args.verbosity)
