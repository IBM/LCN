"""Generate k-tree LCN benchmark instances of increasing size.

A k-tree is the maximal graph of treewidth exactly ``k`` (Arnborg &
Proskurowski): start with a (k+1)-clique, then repeatedly add a vertex adjacent
to exactly ``k`` vertices that already form a clique. The generator orients it
as a DAG so every atom conditions on the full conjunction of its ``k``
clique-parents; because each parent set is a clique, the moralized graph equals
the k-tree and the chain-graph junction tree stays at treewidth ``k`` (see
Generator._graph_ktree). This lets inference scaling be studied as a function of
treewidth rather than raw variable count. ``k=1`` degenerates to a random rooted
tree (treewidth 1); ``k`` requires n >= k + 1.
"""

import os
import argparse

from lcn.benchmarks.generator import Generator


def generate(output_dir, sizes, num_instances, seed, epsilon,
             max_vars, num_extras, k, verbosity, family_realizable=False):
    os.makedirs(output_dir, exist_ok=True)
    gen = Generator(seed=seed)

    # The family-realizable class "ktree-fr" has the same topology but places
    # the extra marginals on root atoms only, so the instance has no
    # non-family-realizable SENTENCE. CAVEAT: for k >= 2 the k-tree is still not
    # singly-connected, so unlike tree-fr/polytree-fr this does NOT make Credal
    # VE / Interval BP exact -- use CredalJT / ExactInference(solver="global").
    # See Generator.generate and docs/strong_extension_exactness.tex.
    graph_type = "ktree-fr" if family_realizable else "ktree"
    slug = "ktree_fr" if family_realizable else "ktree"

    # Encode k in the filename prefix so different k values coexist in the same
    # directory (mirrors gen_benchmarks.py).
    prefix = f"{slug}_k{k}"

    for n in sizes:
        if n < k + 1:
            if verbosity > 0:
                print(f"Skipping n={n}: k-tree needs n >= k + 1 (k={k}).")
            continue
        instances = gen.generate(
            num_vars=n,
            graph_type=graph_type,
            num_instances=num_instances,
            max_vars_per_sentence=max_vars,
            num_extras=num_extras,
            epsilon=epsilon,
            k=k,
            verbosity=max(0, verbosity - 1),
        )
        for i, lcn in enumerate(instances):
            fname = os.path.join(output_dir, f"{prefix}_n{n}_{i + 1}.lcn")
            gen.save(lcn, fname)
            if verbosity > 0:
                print(f"Saved {fname} ({len(lcn.sentences)} sentences)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate k-tree LCN benchmark instances (bounded treewidth k).")
    parser.add_argument("--output-dir", type=str, default="benchmarks/ktree",
                        help="Output directory (default: benchmarks/ktree)")
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
    parser.add_argument("--k", type=int, default=2,
                        help="Treewidth parameter: each atom conditions on "
                             "exactly k clique-parents, so the junction-tree "
                             "treewidth is exactly k (default: 2; needs "
                             "n >= k + 1). k=1 is a random rooted tree.")
    parser.add_argument("--family-realizable", action="store_true",
                        help="Generate the family-realizable class 'ktree-fr' "
                             "(extra marginals on root atoms only, so the "
                             "instance has no non-realizable SENTENCE); files "
                             "prefixed ktree_fr_. NOTE: for k>=2 this is still "
                             "NOT Credal VE / Interval BP exact (a k-tree is "
                             "structurally loopy) -- use CredalJT / "
                             "ExactInference(solver='global').")
    parser.add_argument("--verbosity", type=int, default=1,
                        help="Verbosity level (default: 1)")
    args = parser.parse_args()
    generate(args.output_dir, args.sizes, args.num_instances, args.seed,
             args.epsilon, args.max_vars, args.num_extras, args.k,
             args.verbosity, family_realizable=args.family_realizable)
