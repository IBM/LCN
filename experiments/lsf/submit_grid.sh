#!/bin/bash
# Build the manifest of (benchmark dir, run) pairs and submit the grid as one
# LSF job array: 16 cores per element, 32 GB per instance/algorithm
# combination (16 per job), 1 h per instance/algorithm, at most 20 elements
# running at once.
#
#   experiments/lsf/submit_grid.sh            # submit the grid
#   experiments/lsf/submit_grid.sh --smoke    # one small test job (tree_small_fr, ariel)

set -eu

REPO=/dccstor/radu-data/git/IBM/LCN
cd "$REPO"
LSF_DIR=experiments/lsf
LOG_DIR=results/lsf_logs
mkdir -p "$LOG_DIR"

DIRS=(tree_small_fr tree_large_fr polytree_small_fr polytree_large_fr
      ktree_small_fr ktree_large_fr dag_small_fr dag_large_fr
      nr_tree nr_dag nr_dag_ablation real junkyu)
RUNS=(exact_scip exact_ipopt ariel cjt_scip cjt_ipopt cve cve_eps_0.1 ivbp cdve)

WALLTIME=9:00        # 16 shards x up to 7 instances x 1 h, plus margin
CONCURRENCY=20       # array elements running at once

if [ "${1:-}" = "--smoke" ]; then
    # 4 slots instead of 16: same per-shard logic, but it fits the queue now.
    bsub -q normal -n 4 -R "span[hosts=1] rusage[mem=32768]" -M 512 \
        -W 1:00 -J "lcn_smoke" \
        -o "$LOG_DIR/smoke_%J.out" -e "$LOG_DIR/smoke_%J.err" \
        env JOB_SPEC="tree_small_fr ariel" N_SHARDS=4 bash "$LSF_DIR/run_job.sh" \
        < /dev/null
    exit 0
fi

MANIFEST="$LSF_DIR/manifest.txt"
: > "$MANIFEST"
for d in "${DIRS[@]}"; do
    for r in "${RUNS[@]}"; do
        echo "$d $r" >> "$MANIFEST"
    done
done
N=$(wc -l < "$MANIFEST")
echo "manifest: $N pairs in $MANIFEST"

bsub -q normal -n 16 -R "span[hosts=1] rusage[mem=32768]" -M 512 \
    -W "$WALLTIME" -J "lcn_grid[1-$N]%$CONCURRENCY" \
    -o "$LOG_DIR/%J_%I.out" -e "$LOG_DIR/%J_%I.err" \
    bash "$LSF_DIR/run_job.sh" < /dev/null
