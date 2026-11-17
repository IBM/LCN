#!/bin/bash
# Local counterpart of experiments/slurm/run_job.sh: runs the evaluation grid
# on this machine, one task per (benchmark, run, instance), at most JOBS at once.
#
#   [RUNS="r1 r2"] [SHARDS="0 1"] experiments/local/run_grid.sh [JOBS] [DIR...]
#   (default: all nine runs, every instance, 7 jobs, real junkyu)
#
# Same settings as the SLURM grid: 3600 s, 8 GB, 1 thread, --no-cache,
# --exact-threshold 15, ev0. RLIMIT_AS is not enforced on macOS, so run
# experiments/local/mem_watchdog.py alongside (it SIGKILLs a worker above 8 GB,
# which run_algorithm.py records as status "memory").
set -u
cd "$(dirname "$0")/../.." || exit 1
JOBS=${1:-7}; shift || true
DIRS=("${@:-real}")
[ $# -eq 0 ] && DIRS=(real junkyu)
RUNS=(${RUNS:-exact_scip cjt_scip exact_ipopt cjt_ipopt cve cve_eps_0.01 cdve ivbp ariel})
SHARDS=${SHARDS:-}   # optional subset of shard indices, e.g. "0 1 2"
OUT_DIR=${OUT_DIR:-results}
TIME_LIMIT=${TIME_LIMIT:-3600}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=$OUT_DIR/local_logs; mkdir -p "$LOG"
for run in "${RUNS[@]}"; do for d in "${DIRS[@]}"; do
    n=$(ls benchmarks/$d/*.lcn | wc -l | tr -d ' ')
    if [ -n "$SHARDS" ]; then for i in $SHARDS; do echo "$d $run $i $n"; done
    else for ((i = 0; i < n; i++)); do echo "$d $run $i $n"; done; fi
done; done | xargs -P "$JOBS" -L 1 sh -c '
    uv run --no-sync python experiments/run_experiment.py \
        --input-dir "benchmarks/$0" --benchmark "$0" --runs "$1" \
        --shard "$2/$3" --output-dir "'"$OUT_DIR"'" --evidence-atoms 0 \
        --time-limit "'"$TIME_LIMIT"'" --memory-limit 8 --num-threads 1 \
        --no-cache --exact-threshold 15 --verbosity 1 \
        < /dev/null > "'"$LOG"'/$0_$1_$2.log" 2>&1
    echo "$(date +%H:%M:%S) done $0 $1 $2/$3 (exit $?)"'
