#!/usr/bin/env bash
# Sweep server_setup across the full config matrix.
#
# Usage:
#   ./scripts/run_server_setup.sh                    # FrodoPIR only (default)
#   IKPIR_BENCH_BACKENDS=simple       ./scripts/run_server_setup.sh
#   IKPIR_BENCH_BACKENDS=frodo,simple ./scripts/run_server_setup.sh  # both backends
#   MAX_MEM_GB=20.0 ./scripts/run_server_setup.sh   # raise OOM guard (default 12 GB)
#
# Memory: dominant cost is the per-segment LWE public matrix A, summed across
# segments to ≈ num_buckets × lwe_dim × 4 B.  After the HintMaterial refactor
# `server.setup()` no longer copies A into the bundle (A is re-expanded
# client-side from the seed), so the server-side bench holds A exactly once.
# This shell guard intentionally multiplies by 2 as a conservative safety
# margin (slack for the table cells, transient bundle clones during the wire-
# size readout, and OS overhead).  This guard is also FrodoPIR-shaped — it
# does not divide by the SimplePIR √N reshape factor and will therefore
# over-skip large SimplePIR configs that would actually fit; if SimplePIR runs
# need to cover the largest configs, run via `scripts/run_classical.sh`
# instead (its in-bench guard is backend-aware).  The bench skips any config
# whose estimated peak exceeds MAX_MEM_GB (default 12.0).  Raise it if your
# machine has 32 GB+ RAM.
#
# Config matrix (scripts/configs.sh):
#   12 (arity, bucket_size, num_buckets) tuples × 3 value_bits = 36 runs/backend
#   (some may be skipped by the OOM guard).
#
# Output:
#   ikpir-server/results/ikpir_server_setup.csv
#
# Columns: backend, arity, num_buckets, bucket_size, value_bits, lwe_dim,
#   mean_setup_ms, min_setup_ms, max_setup_ms, stddev_setup_ms,
#   setup_bundle_bytes, hint_bytes_per_segment, server_params_bytes_per_segment

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# shellcheck source=./configs.sh
source "$WORKSPACE_DIR/scripts/configs.sh"

cd "$WORKSPACE_DIR"

RESULTS_DIR="$WORKSPACE_DIR/ikpir-server/results"
mkdir -p "$RESULTS_DIR"

if [[ -t 1 ]]; then
    BLUE='\033[34m'; GREEN='\033[32m'; YELLOW='\033[33m'; MAGENTA='\033[35m'; RESET='\033[0m'
else
    BLUE=''; GREEN=''; YELLOW=''; MAGENTA=''; RESET=''
fi

step()         { echo -e "${BLUE}── $* ──${RESET}"; }
note()         { echo -e "${YELLOW}  $*${RESET}"; }
ok()           { echo -e "${GREEN}  $*${RESET}"; }
backend_note() { echo -e "${MAGENTA}▶ backend=$1${RESET}"; }

MAX_MEM_GB=${MAX_MEM_GB:-12.0}
MAX_MEM_BYTES=$(awk "BEGIN { printf \"%d\", $MAX_MEM_GB * 1e9 }")

CARGO=(cargo bench -p ikpir-server --bench server_setup)

step "server_setup  (trials=5, warmup=2, max_mem=${MAX_MEM_GB} GB)"
rm -f "$RESULTS_DIR/ikpir_server_setup.csv"

for backend in "${BACKENDS_ARR[@]}"; do
    backend_note "$backend"
    lwe=$(backend_lwe_dim "$backend")
    for cfg in "${BENCH_CONFIGS[@]}"; do
        IFS=':' read -r arity bs nb n_label m_label <<< "$cfg"
        pb=$(backend_plaintext_bits "$backend" "$m_label")
        for i in "${!VALUE_BITS[@]}"; do
            vb=${VALUE_BITS[$i]}
            w=${W_LABELS[$i]}

            # Memory guard: peak = server (A) + bundle (A) = 2 × num_buckets × lwe × 4 B.
            # server.setup() clones A into the bundle; both are live simultaneously
            # during the one warmup trial that reads wire sizes.
            peak_bytes=$(( nb * lwe * 4 * 2 ))
            if (( peak_bytes > MAX_MEM_BYTES )); then
                peak_gb=$(awk "BEGIN { printf \"%.1f\", $peak_bytes / 1e9 }")
                a_gb=$(awk "BEGIN { printf \"%.1f\", $nb * $lwe * 4 / 1e9 }")
                note "Skip (OOM guard): estimated peak ${peak_gb} GB > MAX_MEM_GB=${MAX_MEM_GB} (nb=$nb lwe=$lwe, A=${a_gb} GB/copy × 2). Raise MAX_MEM_GB on machines with more RAM."
                continue
            fi

            note "arity=$arity bs=$bs nb=$nb (n=$n_label, m=$m_label) w=$w lwe=$lwe pb=$pb"
            "${CARGO[@]}" -- \
                --backend         "$backend" \
                --arity           "$arity"   \
                --num-buckets     "$nb"      \
                --bucket-size     "$bs"      \
                --value-bits      "$vb"      \
                --plaintext-bits  "$pb"      \
                --lwe-dim         "$lwe"     \
                --trials 5 --warmup 2
        done
    done
done

echo
ok "Done.  Results:"
ok "  $RESULTS_DIR/ikpir_server_setup.csv"
