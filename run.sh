#!/bin/bash

# Usage:
#   ./run.sh <benchmark_dir> <algorithm> <time_limit> [epsilon]
#
# Arguments:
#   benchmark_dir  - path to benchmark instances (e.g., benchmarks/polytree)
#   algorithm      - one of: exact, ariel, ibp, ccte, ccte_e, ijgp, ijgp_e approxlp
#   time_limit     - (optional) time limit in seconds per instance (e.g., 300)
#   epsilon        - (optional) epsilon value for ccte_e, ijgp_e algorithm
#   ibound         - (optional) ibound value for ijgp, ijgp_e algorithms
#
# Examples:
#   ./run.sh benchmarks/chain ccte 300
#   ./run.sh benchmarks/polytree ccte_e 600 0.01
#   ./run.sh benchmarks/random ariel 120

b=$1
a=$2
t=$3
e=$4
i=$5

if [ -z "$b" ] || [ -z "$a" ]; then
    echo "Usage: $0 <benchmark_dir> <algorithm> [time_limit] [epsilon] [ibound]"
    exit 1
fi

mkdir -p logs

l="logs/$(basename $b)_$a.log"

# Build the command
cmd="python experiments/run_experiment.py --input-dir $b --algorithms $a"

if [ -n "$t" ]; then
    cmd="$cmd --time-limit $t"
fi

    if [ -n "$e" ]; then
    cmd="$cmd --epsilon $e"
fi

if [ -n "$i" ]; then
    cmd="$cmd --ibound $i"
fi

./timeout -m 30000000 $cmd >& $l
