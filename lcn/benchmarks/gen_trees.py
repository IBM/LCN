"""Generate rooted directed-tree LCN benchmark instances of increasing size.

A tree is the branching analogue of a chain: a single root and every non-root
node has exactly one parent, so there are no colliders. Inference engines such
as ARIEL and CredalJT are exact on trees (treewidth 1); see
docs/ariel_exactness.tex and examples/tree.lcn.
"""

import os
import argparse

from lcn.benchmarks.generator import Generator


def generate(output_dir, sizes, num_instances, seed, epsilon,
             max_vars, num_extras, verbosity, family_realizable=False):
    os.makedirs(output_dir, exist_ok=True)
    gen = Generator(seed=seed)

    # The family-realizable class ("tree-fr") has the same topology but places
    # the extra marginals on root atoms only, so every sentence is
    # family-realizable and the strong-extension engines (Credal VE, Interval
    # BP) are exact on it (see docs/strong_extension_exactness.tex).
    graph_type = "tree-fr" if family_realizable else "tree"
    prefix = "tree_fr" if family_realizable else "tree"

    for n in sizes:
        instances = gen.generate(
            num_vars=n,
            graph_type=graph_type,
            num_instances=num_instances,
            max_vars_per_sentence=max_vars,
            num_extras=num_extras,
            epsilon=epsilon,
            verbosity=max(0, verbosity - 1),
        )
        for i, lcn in enumerate(instances):
            fname = os.path.join(output_dir, f"{prefix}_n{n}_{i + 1}.lcn")
            gen.save(lcn, fname)
            if verbosity > 0:
                print(f"Saved {fname} ({len(lcn.sentences)} sentences)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate rooted directed-tree LCN benchmark instances.")
    parser.add_argument("--output-dir", type=str, default="benchmarks/tree",
                        help="Output directory (default: benchmarks/tree)")
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
    parser.add_argument("--family-realizable", action="store_true",
                        help="Generate the family-realizable class 'tree-fr' "
                             "(extra marginals on root atoms only, so Credal VE "
                             "/ Interval BP are exact); files prefixed tree_fr_.")
    parser.add_argument("--verbosity", type=int, default=1,
                        help="Verbosity level (default: 1)")
    args = parser.parse_args()
    generate(args.output_dir, args.sizes, args.num_instances, args.seed,
             args.epsilon, args.max_vars, args.num_extras, args.verbosity,
             family_realizable=args.family_realizable)
