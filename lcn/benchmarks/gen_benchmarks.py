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

TOPOLOGIES = ["chain", "tree", "polytree", "random", "dag", "ktree"]
# "easy" targets SCIP-easy instances rather than a graph topology; the "-fr"
# entries are the family-realizable classes (extra marginals on root atoms only,
# and conditionals on the full parent conjunction, so every sentence bounds a
# single row of one family's conditional table -- see
# docs/strong_extension_exactness.tex). For the singly-connected "tree-fr" /
# "polytree-fr" that also makes Credal VE / Interval BP exact; "ktree-fr" (k >= 2)
# and "dag-fr" are realizable but loopy, so they need CredalJT / global inference.
# "dag-fr" additionally bounds its treewidth by construction rather than by
# rejection, which is what makes large bounded-width DAGs generatable.
# All are available via --types but excluded from the default set.
# See Generator.generate.
ALL_TYPES = TOPOLOGIES + ["tree-fr", "polytree-fr", "ktree-fr", "dag-fr", "easy"]


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
        help="instance types to generate (default: chain tree polytree random "
             "dag ktree; 'tree-fr'/'polytree-fr' are the family-realizable "
             "classes on which Credal VE / Interval BP are exact; 'ktree-fr' has "
             "root-only extras so it has no non-realizable SENTENCES, but for "
             "k>=2 it is still not CVE/IBP-exact (structurally loopy); 'easy' "
             "produces SCIP-easy instances)")
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
        help="max parents per child node (default: 2, polytree/dag only; "
             "fully tunable for dag -- treewidth stays bounded by "
             "--max-treewidth regardless)")
    parser.add_argument(
        "--max-treewidth", type=int, default=4,
        help="cap on the moralized chain-graph treewidth for 'dag' (default: 4; "
             "DAGs are rejection-sampled to stay at or below this width)")
    parser.add_argument(
        "--k", type=int, default=2,
        help="k-tree treewidth parameter: each atom conditions on exactly k "
             "clique-parents so the junction-tree treewidth is exactly k "
             "(default: 2, 'ktree' type only; needs n >= k + 1)")
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
        "--consistency-mode", type=str, default="product",
        choices=("product", "full", "structured"),
        help="consistency gate: 'product' (default, fast sound product "
             "witness), 'full' (exact 2^n joint-LMC oracle, small n only), or "
             "'structured' (junction-tree feasibility -- sound AND complete at "
             "bounded treewidth, scales to any n)")
    parser.add_argument(
        "--verbosity", type=int, default=1,
        help="verbosity level: 0=silent, 1=summary (default: 1)")
    args = parser.parse_args()

    gen = Generator(seed=args.seed)
    total_saved = 0

    for graph_type in args.types:
        # Use a filesystem-friendly slug ("tree-fr" -> "tree_fr") for the
        # subdirectory and filename prefix.
        slug = graph_type.replace("-", "_")
        type_dir = os.path.join(args.output_dir, slug)
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
                max_treewidth=args.max_treewidth,
                k=args.k,
                strategy=args.strategy,
                coverage=args.coverage,
                core=args.core,
                difficulty_cap=args.difficulty_cap,
                base_topology=args.base_topology,
                verify_time_limit=args.verify_time_limit,
                consistency_mode=args.consistency_mode,
                verbosity=max(0, args.verbosity - 1),
            )
            # Encode the strategy in easy filenames (mirrors gen_easy.py) so
            # linear/bounded/verified sets don't overwrite one another; likewise
            # encode k in ktree filenames so different k values coexist.
            if graph_type == "easy":
                prefix = f"easy_{args.strategy}"
            elif graph_type in ("ktree", "ktree-fr"):
                # slug is "ktree" or "ktree_fr"; append the k value.
                prefix = f"{slug}_k{args.k}"
            elif graph_type == "dag-fr":
                # Encode the treewidth cap (the discriminating topology param
                # here, as k is for ktree) so different caps coexist.
                prefix = f"{slug}_tw{args.max_treewidth}"
            else:
                prefix = slug
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
