#!/bin/bash
# Build the manifest of (benchmark dir, run) pairs and submit the grid as one
# SLURM job array: 16 CPUs / 128 GB per element (16 shards x 8 GB), 1 h per
# instance/algorithm, at most 20 elements running at once.
#
#   experiments/slurm/submit_grid.sh            # submit the grid
#   experiments/slurm/submit_grid.sh --smoke    # one 4-CPU test job (tree_small_fr, ariel)

set -eu

REPO=/u/radu/git/LCN
cd "$REPO"
SL_DIR=experiments/slurm
LOG_DIR=results/slurm_logs
mkdir -p "$LOG_DIR"

DIRS=(tree_small_fr tree_large_fr polytree_small_fr polytree_large_fr
      ktree_small_fr ktree_large_fr dag_small_fr dag_large_fr
      nr_tree nr_dag nr_dag_ablation)
RUNS=(exact_scip exact_ipopt ariel cjt_scip cjt_ipopt cve cve_eps_0.01 ivbp cdve)

COMMON=(--account=slurm_grp_big_job_lease --partition=cpu --qos=cpu
        --nodes=1 --ntasks=1 --mem-per-cpu=8G)
WALLTIME=8:00:00     # 100-instance sets: 7 instances per shard x 1 h, plus margin
CONCURRENCY=20       # array elements running at once

if [ "${1:-}" = "--smoke" ]; then
    sbatch "${COMMON[@]}" --cpus-per-task=4 --time=1:00:00 -J lcn_smoke \
        -o "$LOG_DIR/smoke_%j.out" -e "$LOG_DIR/smoke_%j.err" \
        --export=ALL,JOB_SPEC="tree_small_fr ariel",N_SHARDS=4 \
        "$SL_DIR/run_job.sh"
    exit 0
fi

MANIFEST="$SL_DIR/manifest.txt"
: > "$MANIFEST"
for d in "${DIRS[@]}"; do
    for r in "${RUNS[@]}"; do
        echo "$d $r" >> "$MANIFEST"
    done
done
N=$(wc -l < "$MANIFEST")
echo "manifest: $N pairs in $MANIFEST"

sbatch "${COMMON[@]}" --cpus-per-task=16 --time="$WALLTIME" \
    -J lcn_grid --array="1-$N%$CONCURRENCY" \
    -o "$LOG_DIR/%A_%a.out" -e "$LOG_DIR/%A_%a.err" \
    "$SL_DIR/run_job.sh"
