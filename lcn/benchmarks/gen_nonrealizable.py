"""Generate almost-family-realizable LCN benchmarks by controlled perturbation.

Each base instance is a shipped family-realizable, DAG-structured instance
(``benchmarks/{tree,polytree,ktree,dag}_*_fr``). ``Generator.perturb`` adds ``m``
non-realizable sentences of the requested kinds:

  nr1  non-root marginal       P(l), l a literal of an atom with parents
  nr2  mixture conditional     P(l | psi), psi over >= 1 parents, satisfied by
                               >= 2 but not all parent configurations
  nr3  cross-family sentence   P(phi) over two atoms sharing no family, tau=False

The new bounds are [v - delta, v + delta] around the value v under the base
instance's product witness, so every instance is consistent. The structure and
the LMC are unchanged. The instances for the different ``m`` are nested (prefixes
of one drawn sequence). With ``--cjt-exact structural`` (default) each instance
must keep the CJT reference structurally exact.

Output: one directory per run, with files

    {topology}_nr{kind}_m{m}_d{100*delta}_n{n}_{i}.lcn

where ``kind`` is ``mix`` when all three kinds are used, and ``i`` is the base
instance index. ``m = 0`` is a copy of the base, so a directory is
self-contained. The runner's ``_parse_instance_info`` reads the trailing
``_n{n}_{i}``; ``analyze_results`` reads ``nr_kind`` / ``nr_m`` / ``nr_delta``
from the prefix.

Example (the main sweep of the rerun plan):

    python -m lcn.benchmarks.gen_nonrealizable \\
        --bases benchmarks/tree_small_fr benchmarks/tree_large_fr \\
        --topology tree --sizes 10 20 --ms 0 1 2 4 8 --deltas 0.1 \\
        --output-dir benchmarks/nr_tree
"""

import argparse
import contextlib
import glob
import io
import os
import re

from lcn.core.model import LCN
from lcn.benchmarks.generator import Generator

_BASE_RE = re.compile(r".*_n(\d+)_(\d+)\.lcn$")


def nr_file_name(topology, kinds, m, delta, n, i):
    kind = "mix" if set(kinds) == set(Generator.NR_KINDS) else "".join(
        k[2:] for k in kinds)         # e.g. "1", "13"
    return f"{topology}_nr{kind}_m{m}_d{round(100 * delta)}_n{n}_{i}.lcn"


def base_instances(dirs, sizes, num_bases):
    """[(path, n, i)] of the base files with n in ``sizes``, i <= num_bases."""
    out = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.lcn"))):
            mt = _BASE_RE.match(os.path.basename(path))
            if not mt:
                continue
            n, i = int(mt.group(1)), int(mt.group(2))
            if n in sizes and i <= num_bases:
                out.append((path, n, i))
    return sorted(out, key=lambda t: (t[1], t[2]))


def generate(bases, topology, sizes, ms, kinds, deltas, output_dir,
             num_bases=10, seed=42, cjt_exact="structural", cjt_max_width=12,
             verbosity=1):
    os.makedirs(output_dir, exist_ok=True)
    gen = Generator(seed=seed)
    written = []
    for path, n, i in base_instances(bases, sizes, num_bases):
        with contextlib.redirect_stdout(io.StringIO()):
            base = LCN()
            base.from_lcn(file_name=path)
        for delta in deltas:
            with contextlib.redirect_stdout(io.StringIO()):
                out = gen.perturb(base, ms=ms, kinds=kinds, delta=delta,
                                  base_name=os.path.basename(path),
                                  cjt_exact=cjt_exact,
                                  cjt_max_width=cjt_max_width)
            for m, lcn in out.items():
                fname = os.path.join(output_dir, nr_file_name(
                    topology, kinds, m, delta, n, i))
                gen.save(lcn, fname)
                written.append(fname)
                if verbosity > 0:
                    print(f"Saved {fname} ({len(lcn.sentences)} sentences)")
    return written


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Perturb family-realizable LCNs with non-realizable "
                    "sentences (controlled degradation benchmarks).")
    p.add_argument("--bases", nargs="+", required=True,
                   help="directories of family-realizable base instances")
    p.add_argument("--topology", required=True,
                   help="name prefix of the output files (tree, dag, ...)")
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 20])
    p.add_argument("--num-bases", type=int, default=10,
                   help="base instances per size (default: 10)")
    p.add_argument("--ms", type=int, nargs="+", default=[0, 1, 2, 4, 8],
                   help="numbers of added non-realizable sentences")
    p.add_argument("--kinds", nargs="+", default=list(Generator.NR_KINDS),
                   choices=list(Generator.NR_KINDS))
    p.add_argument("--deltas", type=float, nargs="+", default=[0.1],
                   help="half-widths of the added intervals")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cjt-exact", default="structural",
                   choices=["none", "structural", "certified"])
    p.add_argument("--cjt-max-width", type=int, default=12)
    p.add_argument("--verbosity", type=int, default=1)
    a = p.parse_args()
    generate(a.bases, a.topology, a.sizes, a.ms, a.kinds, a.deltas,
             a.output_dir, num_bases=a.num_bases, seed=a.seed,
             cjt_exact=None if a.cjt_exact == "none" else a.cjt_exact,
             cjt_max_width=a.cjt_max_width, verbosity=a.verbosity)
