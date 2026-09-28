"""Generate DAG LCN benchmark instances of increasing size.

A "dag" instance is a random directed acyclic graph in which every non-root node
has up to ``--max-parents`` parents (in-degree tunable, default 2; DAG-ness
guaranteed by drawing parents only from earlier vertices in a random ordering).
Unlike ``ktree`` -- whose treewidth is fixed exactly at ``k`` by construction --
the DAG topology is UNSTRUCTURED; its moralized chain-graph treewidth is bounded
ABOVE by ``--max-treewidth`` (default 4) via rejection sampling: candidates whose
moralized min-fill induced width exceeds the cap are resampled. This bound holds
regardless of ``--max-parents`` (higher fan-in just means more candidates are
rejected before one meets the cap). Every accepted instance also passes the
standard product-witness consistency check (the same one used for
trees/polytrees/k-trees), so it is both treewidth-bounded and consistent. See
Generator._graph_dag / Generator.generate.

With ``--family-realizable`` this generates the "dag-fr" class instead, which
differs in two ways. (1) Family-realizability: extra marginals go on root atoms
only and each conditional is on the FULL parent conjunction, so every sentence
bounds a single row of one family's conditional table. (2) The treewidth cap holds
BY CONSTRUCTION -- parents are drawn from a k-clique of an underlying partial
k-tree (Generator._graph_dag_bounded) rather than by rejection sampling, whose
acceptance rate collapses as n grows (<1% at n=50, 0% at n>=30 for 3 parents),
making large bounded-width DAGs effectively ungeneratable the other way.
NOTE that "dag-fr" is still NOT singly connected, so Credal VE / Interval BP are
not exact on it (same caveat as "ktree-fr" for k >= 2); it is a bounded-treewidth,
fully realizable benchmark class for CredalJT / global exact inference.
"""

import os
import argparse

from lcn.benchmarks.generator import Generator


def generate(output_dir, sizes, num_instances, seed, epsilon,
             max_vars, num_extras, max_parents, max_treewidth, verbosity,
             family_realizable=False):
    os.makedirs(output_dir, exist_ok=True)
    gen = Generator(seed=seed)

    graph_type = "dag-fr" if family_realizable else "dag"

    # The family-realizable class encodes the treewidth cap in the filename so
    # runs with different --max-treewidth coexist in one directory (mirrors the
    # _k{k} convention in gen_ktrees.py). Plain "dag" keeps its flat name.
    prefix = f"dag_fr_tw{max_treewidth}" if family_realizable else "dag"

    for n in sizes:
        instances = gen.generate(
            num_vars=n,
            graph_type=graph_type,
            num_instances=num_instances,
            max_vars_per_sentence=max_vars,
            num_extras=num_extras,
            epsilon=epsilon,
            max_parents=max_parents,
            max_treewidth=max_treewidth,
            verbosity=max(0, verbosity - 1),
        )
        for i, lcn in enumerate(instances):
            fname = os.path.join(output_dir, f"{prefix}_n{n}_{i + 1}.lcn")
            gen.save(lcn, fname)
            if verbosity > 0:
                print(f"Saved {fname} ({len(lcn.sentences)} sentences)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate DAG LCN benchmark instances (<=2 parents, "
                    "treewidth bounded above by --max-treewidth).")
    parser.add_argument("--output-dir", type=str, default="benchmarks/dag",
                        help="Output directory (default: benchmarks/dag)")
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
    parser.add_argument("--max-parents", type=int, default=2,
                        help="Max parents per child node (default: 2; fully "
                             "tunable -- treewidth stays bounded by "
                             "--max-treewidth regardless of fan-in).")
    parser.add_argument("--max-treewidth", type=int, default=4,
                        help="Cap on the moralized chain-graph treewidth "
                             "(default: 4). DAGs are rejection-sampled to stay "
                             "at or below this width; use 3 for a tighter cap.")
    parser.add_argument("--family-realizable", action="store_true",
                        help="Generate the family-realizable class 'dag-fr': "
                             "extra marginals on root atoms only AND conditionals "
                             "on the full parent conjunction, so every sentence "
                             "is family-realizable. Its treewidth cap also holds "
                             "BY CONSTRUCTION rather than by rejection, which is "
                             "what makes large (n >= 30) bounded-width DAGs "
                             "generatable at all. Files prefixed "
                             "dag_fr_tw{max_treewidth}_. CAVEAT: like 'ktree-fr' "
                             "with k >= 2 this does NOT make Credal VE / Interval "
                             "BP exact -- a collider with >= 2 parents moralizes "
                             "into a loop, so use CredalJT / "
                             "ExactInference(solver='global') for exact bounds.")
    parser.add_argument("--verbosity", type=int, default=1,
                        help="Verbosity level (default: 1)")
    args = parser.parse_args()
    generate(args.output_dir, args.sizes, args.num_instances, args.seed,
             args.epsilon, args.max_vars, args.num_extras, args.max_parents,
             args.max_treewidth, args.verbosity,
             family_realizable=args.family_realizable)
