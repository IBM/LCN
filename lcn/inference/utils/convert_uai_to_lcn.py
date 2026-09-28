"""Convert an LCN problem instance from UAI format to LCN format.

Usage:
    python -m lcn.inference.utils.convert_uai_to_lcn --input file.uai --output file.lcn
    python -m lcn.inference.utils.convert_uai_to_lcn --input benchmarks/uai/asia.uai --output /tmp/asia.lcn
"""

import argparse

from lcn.core.model import LCN


def convert(input_file, output_file, lmc=True, verbosity=1):
    """Convert a UAI file to LCN format.

    Args:
        input_file: path to the input .uai file.
        output_file: path to the output .lcn file.
        lmc: if True, apply the Local Markov Condition.
        verbosity: 0=silent, 1=summary.
    """
    l = LCN()
    ok = l.from_uai(file_name=input_file, lmc=lmc)
    if not ok:
        raise RuntimeError(f"Failed to parse {input_file}")
    l.save_lcn(output_file)
    if verbosity > 0:
        print(f"Converted {input_file} -> {output_file} "
              f"({len(l.sentences)} sentences)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert UAI format to LCN format.")
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input .uai file")
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output .lcn file")
    parser.add_argument(
        "--no-lmc", action="store_true",
        help="Skip Local Markov Condition")
    parser.add_argument(
        "--verbosity", type=int, default=1,
        help="Verbosity level (default: 1)")
    args = parser.parse_args()
    convert(args.input, args.output, lmc=not args.no_lmc,
            verbosity=args.verbosity)
