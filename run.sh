#!/bin/bash
#
# Usage:
#   ./run.sh <benchmark_dir> <run> <factorization> [time_limit] [flags...]
#
# Arguments:
#   benchmark_dir  - directory containing .lcn benchmark files
#   run            - a run of the evaluation grid (experiments/runs.py):
#                      exact_scip exact_ipopt ariel cjt_scip cjt_ipopt
#                      cve cve_eps_<E> ivbp cdve        (or 'all' for the grid)
#                    or a build stage: compile, enumerate
#   factorization  - the factorization method (linear or linear-tight)
#   time_limit     - optional wall-clock limit in seconds per instance
#   flags...       - any further run_experiment.py flags, passed through, e.g.
#                      --evidence-atoms 3              (observed atoms per instance)
#                      --evidence-seed 0
#                      --n-iters 50                    (ariel, ivbp, cdve)
#                      --n-restarts 8                  (cdve)
#                    Results go to results/<benchmark>/ev<K>/<run>.jsonl.
#
# Examples:
#   ./run.sh benchmarks/tree_small_fr compile linear
#   ./run.sh benchmarks/tree_small_fr enumerate linear
#   ./run.sh benchmarks/tree_small_fr cjt_scip linear 3600
#   ./run.sh benchmarks/tree_small_fr cjt_ipopt linear 3600
#   ./run.sh benchmarks/tree_small_fr exact_scip linear 3600
#   ./run.sh benchmarks/tree_small_fr cve linear 3600 --evidence-atoms 1
#   ./run.sh benchmarks/tree_small_fr cve_eps_0.1 linear 3600
#   ./run.sh benchmarks/tree_small_fr all linear 3600 --evidence-atoms 3

b=$1
a=$2
f=$3
t=$4
shift 4 2>/dev/null || shift $#
extra="$*"

RUNS="exact_scip exact_ipopt ariel cjt_scip cjt_ipopt cve cve_eps_<E> ivbp cdve all"
STAGES="compile enumerate"

if [ -z "$b" ] || [ -z "$a" ] || [ -z "$f" ]; then
    echo "Usage: $0 <benchmark_dir> <run> <factorization> [time_limit] [flags...]"
    echo ""
    echo "Runs:   $RUNS"
    echo "Stages: $STAGES"
    exit 1
fi

mkdir -p logs

if [[ " $STAGES " == *" $a "* ]]; then
    # Build stages: per-family interval solves in 16 parallel workers.
    cmd="python experiments/run_experiment.py --input-dir $b --stages $a --factorization-method $f --memory-limit 30 --n-jobs 16"
else
    # run_experiment.py validates the run name (experiments/runs.py).
    cmd="python experiments/run_experiment.py --input-dir $b --runs $a --factorization-method $f --memory-limit 30"
fi

if [ -n "$t" ]; then
    cmd="$cmd --time-limit $t"
fi
if [ -n "$extra" ]; then
    cmd="$cmd $extra"
fi

# Log name carries the run and the extra flags so settings do not share a log.
tag=$(echo "$extra" | tr -s ' -' '_' | sed 's/^_//; s/_$//')
l="logs/$(basename $b)_$a${tag:+_$tag}.log"

echo "Running: $cmd"
echo "Log: $l"

# Run the command and redirect output to log file
$cmd >& $l

# Running with timeout perl script to avoid memory leaks in Python (e.g., with SCIP)
# ./timeout -m 30000000 $cmd >& $l
