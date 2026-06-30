#!/bin/bash

# Usage:
#   ./run.sh <benchmark_dir> <algorithm> [options...]
#
# Arguments:
#   benchmark_dir  - path to benchmark instances (e.g., benchmarks/polytree)
#   algorithm      - one of: exact_l, exact_g, ariel, ibp, ijgp, ijgp_e,
#                    ijgp_cp, ijgp_cm, ccte, ccte_e, ccte_cp, ccte_cm, approxlp,
#                    cve, cve_e, cve_d4, cjt
#                    (exact_l = local/ipopt backend; exact_g = global/SCIP backend)
#   factorization  - the factorization to use (e.g., linear, nlp, exact)
#
# Options (positional, after algorithm):
#   time_limit     - time limit in seconds per instance (e.g., 300)
#   epsilon        - epsilon value for *_e, *_cp, *_cm algorithms
#   ibound         - i-bound for ijgp* algorithms
#   n_clusters     - number of clusters for *_cp, *_cm algorithms
#
# Examples:
#   ./run.sh benchmarks/chain exact_l linear 300
#   ./run.sh benchmarks/chain exact_g linear 300
#   ./run.sh benchmarks/chain ccte linear 300
#   ./run.sh benchmarks/polytree ccte_e 600 0.01
#   ./run.sh benchmarks/polytree ijgp 300 "" 4
#   ./run.sh benchmarks/polytree ijgp_e 600 0.01 4
#   ./run.sh benchmarks/polytree ijgp_cp 600 0.01 4 10
#   ./run.sh benchmarks/polytree ccte_cp 600 0.01 "" 10
#   ./run.sh benchmarks/chain cve linear 300
#   ./run.sh benchmarks/chain cjt linear 300

b=$1
a=$2
f=$3
t=$4
e=$5
i=$6
k=$7

if [ -z "$b" ] || [ -z "$a" ] || [ -z "$f" ]; then
    echo "Usage: $0 <benchmark_dir> <algorithm> <factorization> [time_limit] [epsilon] [ibound] [n_clusters]"
    echo ""
    echo "Algorithms: exact_l, exact_g, ariel, ibp, ijgp, ijgp_e, ijgp_cp,"
    echo "            ijgp_cm, ccte, ccte_e, ccte_cp, ccte_cm, approxlp,"
    echo "            cve, cve_e, cve_d4, cjt"
    exit 1
fi

mkdir -p logs

l="logs/$(basename $b)_$a.log"

# Build the command based on the algorithm
cmd="python experiments/run_experiment.py --input-dir $b --algorithms $a --factorization-method $f"

# Add time limit if provided
if [ -n "$t" ]; then
    cmd="$cmd --time-limit $t"
fi

if [ "$a" = "exact_l" ]; then
    # exact_l: exact inference with the local backend (ipopt + SLSQP fallback).
    # The backend is selected by the algorithm name; no extra arguments.
    :

elif [ "$a" = "exact_g" ]; then
    # exact_g: exact inference with the global backend (SCIP, certified).
    # The backend is selected by the algorithm name; no extra arguments.
    :

elif [ "$a" = "ariel" ]; then
    # ariel: no extra arguments
    :

elif [ "$a" = "ibp" ]; then
    # ibp: no extra arguments
    :

elif [ "$a" = "ijgp" ]; then
    # ijgp: optional ibound
    if [ -n "$i" ]; then
        cmd="$cmd --ibound $i"
    fi

elif [ "$a" = "ijgp_e" ]; then
    # ijgp_e: epsilon + optional ibound
    if [ -n "$e" ]; then
        cmd="$cmd --epsilon $e"
    fi
    if [ -n "$i" ]; then
        cmd="$cmd --ibound $i"
    fi

elif [ "$a" = "ijgp_cp" ]; then
    # ijgp_cp: ibound + n_clusters, cluster_representative=plub
    if [ -n "$i" ]; then
        cmd="$cmd --ibound $i"
    fi
    if [ -n "$k" ]; then
        cmd="$cmd --n-clusters $k"
    fi

elif [ "$a" = "ijgp_cm" ]; then
    # ijgp_cm: ibound + n_clusters, cluster_representative=mean
    if [ -n "$i" ]; then
        cmd="$cmd --ibound $i"
    fi
    if [ -n "$k" ]; then
        cmd="$cmd --n-clusters $k --cluster-representative mean"
    fi

elif [ "$a" = "ccte" ]; then
    # ccte: no extra arguments
    :

elif [ "$a" = "ccte_e" ]; then
    # ccte_e: epsilon required
    if [ -n "$e" ]; then
        cmd="$cmd --epsilon $e"
    fi

elif [ "$a" = "ccte_cp" ]; then
    # ccte_cp: n_clusters, cluster_representative=plub
    if [ -n "$k" ]; then
        cmd="$cmd --n-clusters $k --cluster-representative plub"
    fi

elif [ "$a" = "ccte_cm" ]; then
    # ccte_cm: n_clusters, cluster_representative=mean
    if [ -n "$k" ]; then
        cmd="$cmd --n-clusters $k --cluster-representative mean"
    fi

elif [ "$a" = "approxlp" ]; then
    # approxlp: no extra arguments
    :

elif [ "$a" = "cve" ]; then
    # cve: Credal Variable Elimination (coupling off); no extra arguments
    :

elif [ "$a" = "cve_e" ]; then
    # cve_e: epsilon required
    if [ -n "$e" ]; then
        cmd="$cmd --epsilon $e"
    fi

elif [ "$a" = "cve_d4" ]; then
    # cve_d4: CVE with cross-family coupling (D4); no extra arguments
    :

elif [ "$a" = "cjt" ]; then
    # cjt: CredalJT junction-tree exact NLP (scheme D5). Use the certified
    # global backend (scip) so the cluster NLP is solved exactly.
    cmd="$cmd --solver scip"

else
    echo "Unknown algorithm: $a"
    echo "Allowed: exact_l, exact_g, ariel, ibp, ijgp, ijgp_e, ijgp_cp,"
    echo "         ijgp_cm, ccte, ccte_e, ccte_cp, ccte_cm, approxlp,"
    echo "         cve, cve_e, cve_d4, cjt"
    exit 1
fi

echo "Running: $cmd"
echo "Log: $l"

./timeout -m 30000000 $cmd >& $l
