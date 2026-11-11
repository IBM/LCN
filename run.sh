#!/bin/bash
#
# Usage:
#   ./run.sh <benchmark_dir> <name> <factorization> [time_limit] [flags...]
#
# Arguments:
#   benchmark_dir  - directory containing .lcn benchmark files
#   name           - an algorithm: exact, ariel, cjt, ivbp, cve, approxlp
#                    or a build stage: compile, enumerate
#   factorization  - the factorization method (linear or linear-tight)
#   time_limit     - optional wall-clock limit in seconds per instance
#   flags...       - any further run_experiment.py flags, passed through, e.g.
#                      --solver ipopt                  (exact, cjt; default scip)
#                      --epsilon 0.1                   (cve -> FPTAS, cve_e0.1)
#                      --n-iters 50                    (ivbp, approxlp, ariel)
#                    Results go to results/<benchmark>/<name><variant>.jsonl.
#
# Examples:
#   ./run.sh benchmarks/tree_small_fr compile linear
#   ./run.sh benchmarks/tree_small_fr enumerate linear
#   ./run.sh benchmarks/tree_small_fr cjt linear 3600
#   ./run.sh benchmarks/tree_small_fr cjt linear 3600 --solver ipopt
#   ./run.sh benchmarks/tree_small_fr exact linear 3600
#   ./run.sh benchmarks/tree_small_fr cve linear 3600
#   ./run.sh benchmarks/tree_small_fr cve linear 3600 --epsilon 0.1
#   ./run.sh benchmarks/tree_small_fr ivbp linear 3600

b=$1
a=$2
f=$3
t=$4
shift 4 2>/dev/null || shift $#
extra="$*"

ALGORITHMS="exact ariel cjt ivbp cve approxlp"
STAGES="compile enumerate"

if [ -z "$b" ] || [ -z "$a" ] || [ -z "$f" ]; then
    echo "Usage: $0 <benchmark_dir> <name> <factorization> [time_limit] [flags...]"
    echo ""
    echo "Algorithms: $ALGORITHMS"
    echo "Stages:     $STAGES"
    exit 1
fi

mkdir -p logs

if [[ " $STAGES " == *" $a "* ]]; then
    # Build stages: per-family interval solves in 16 parallel workers.
    cmd="python experiments/run_experiment.py --input-dir $b --stages $a --factorization-method $f --memory-limit 30 --n-jobs 16"
elif [[ " $ALGORITHMS " == *" $a "* ]]; then
    cmd="python experiments/run_experiment.py --input-dir $b --algorithms $a --factorization-method $f --memory-limit 30"
else
    echo "Unknown algorithm or stage: $a"
    echo "Algorithms: $ALGORITHMS"
    echo "Stages:     $STAGES"
    exit 1
fi

if [ -n "$t" ]; then
    cmd="$cmd --time-limit $t"
fi
if [ -n "$extra" ]; then
    cmd="$cmd $extra"
fi

# Log name carries the variant flags so variants do not share a log.
tag=$(echo "$extra" | tr -s ' -' '_' | sed 's/^_//; s/_$//')
l="logs/$(basename $b)_$a${tag:+_$tag}.log"

echo "Running: $cmd"
echo "Log: $l"

# Run the command and redirect output to log file
$cmd >& $l

# Running with timeout perl script to avoid memory leaks in Python (e.g., with SCIP)
# ./timeout -m 30000000 $cmd >& $l
