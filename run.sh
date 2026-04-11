#!/bin/bash

# 1st benchmark dir (e.g., benchmarks/polytree)
# 2nd argument is algorithm (e.g., exact, ccte, ibp, ariel, approxlp)

b=$1
a=$2

l="logs/$(basename $b)_$a.log"
./timeout -m 30000000 python experiments/run_experiment.py --input-dir $b \
    --algorithms $a >& $l




