#!/bin/bash
# One SLURM job = one (benchmark dir, run) pair, on 16 CPUs.
# The 16 shards of the pair run concurrently, one CPU each; each
# instance/algorithm combination is capped at 8 GB (RLIMIT_AS) and 1 hour.
#
# Under a job array the pair is line $SLURM_ARRAY_TASK_ID of manifest.txt.
# For a one-off test set JOB_SPEC="<dir> <run>" (and N_SHARDS) instead.

set -u

REPO=/u/radu/git/LCN
ENV=/proj/guardian/radu/envs/lcn
PY=$ENV/bin/python
N_SHARDS=${N_SHARDS:-16}
TIME_LIMIT=3600      # seconds per instance per algorithm
MEM_GB=8             # per instance/algorithm combination
EVIDENCE=${EVIDENCE:-0}
OUT_DIR=${OUT_DIR:-results}

# ipopt and scip are found through PATH by Pyomo.
export PATH=$ENV/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Solver scratch files (Pyomo's .nl/.sol) on shared storage, one dir per job:
# an eviction clears the node-local /tmp mid-solve, which used to turn the
# interrupted instance into a spurious FileNotFoundError record.
TMP_ROOT=${TMP_ROOT:-/proj/guardian/radu/data/lcn_tmp}
export TMPDIR=$TMP_ROOT/${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}_$$
mkdir -p "$TMPDIR" || exit 1

cd "$REPO" || exit 1

if [ -n "${JOB_SPEC:-}" ]; then
    read -r DIR RUN <<< "$JOB_SPEC"
else
    read -r DIR RUN < <(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/slurm/manifest.txt)
fi
if [ -z "${DIR:-}" ] || [ -z "${RUN:-}" ]; then
    echo "no (dir, run) for array index ${SLURM_ARRAY_TASK_ID:-none}" >&2
    exit 1
fi

echo "=== $(date) job ${SLURM_JOB_ID:-local} index ${SLURM_ARRAY_TASK_ID:-0}: $DIR $RUN"
echo "=== host $(hostname), python $("$PY" -V 2>&1), $(ipopt --version | head -1), $(scip --version | head -1 | cut -c1-20)"

pids=()
# On eviction/cancel SLURM sends SIGTERM; pass it on to every shard, which
# exits without writing a record (run_experiment.py), so the requeued job
# reruns the interrupted instances.
trap 'echo "=== $(date) SIGTERM: stopping shards"; kill -TERM "${pids[@]}" 2>/dev/null; wait; rm -rf "$TMPDIR"; exit 143' TERM
for ((i = 0; i < N_SHARDS; i++)); do
    "$PY" experiments/run_experiment.py \
        --input-dir "benchmarks/$DIR" --benchmark "$DIR" \
        --runs "$RUN" --shard "$i/$N_SHARDS" \
        --output-dir "$OUT_DIR" --evidence-atoms "$EVIDENCE" \
        --time-limit "$TIME_LIMIT" --memory-limit "$MEM_GB" \
        --num-threads 1 --no-cache --exact-threshold 15 \
        --verbosity 1 < /dev/null &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
rm -rf "$TMPDIR"
echo "=== $(date) finished $DIR $RUN (worst shard exit status $status)"
exit $status
