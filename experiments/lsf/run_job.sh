#!/bin/bash
# One LSF job = one (benchmark dir, run) pair, on 16 cores.
# The 16 shards of the pair run concurrently, one core each, each capped at
# 32 GB (RLIMIT_AS); the instance time limit is 1 hour.
#
# Under a job array the pair is line $LSB_JOBINDEX of manifest.txt. For a
# one-off test set JOB_SPEC="<dir> <run>" instead.
#
#   bsub -q normal ... ; LSB_JOBINDEX is set by LSF for array elements.

set -u

REPO=/dccstor/radu-data/git/IBM/LCN
ENV=$HOME/storage/envs/lcn
PY=$ENV/bin/python
N_SHARDS=${N_SHARDS:-16}
TIME_LIMIT=3600      # seconds per instance per algorithm
MEM_GB=32            # per instance/algorithm combination

# ipopt and scip are found through PATH by Pyomo.
export PATH=$ENV/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO" || exit 1

if [ -n "${JOB_SPEC:-}" ]; then
    read -r DIR RUN <<< "$JOB_SPEC"
else
    read -r DIR RUN < <(sed -n "${LSB_JOBINDEX}p" experiments/lsf/manifest.txt)
fi
if [ -z "${DIR:-}" ] || [ -z "${RUN:-}" ]; then
    echo "no (dir, run) for job index ${LSB_JOBINDEX:-none}" >&2
    exit 1
fi

echo "=== $(date) job ${LSB_JOBID:-local} index ${LSB_JOBINDEX:-0}: $DIR $RUN"
echo "=== host $(hostname), python $("$PY" -V 2>&1)"

pids=()
for ((i = 0; i < N_SHARDS; i++)); do
    "$PY" experiments/run_experiment.py \
        --input-dir "benchmarks/$DIR" --benchmark "$DIR" \
        --runs "$RUN" --shard "$i/$N_SHARDS" \
        --output-dir results --evidence-atoms 0 \
        --time-limit "$TIME_LIMIT" --memory-limit "$MEM_GB" \
        --num-threads 1 --no-cache --exact-threshold 15 \
        --verbosity 1 < /dev/null &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
echo "=== $(date) finished $DIR $RUN (worst shard exit status $status)"
exit $status
