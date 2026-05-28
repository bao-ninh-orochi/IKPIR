#!/usr/bin/env bash
# Sweep server_setup across the SAME config matrix as the mutation_throughput
# sweep (BENCH_CONFIGS × VALUE_BITS), for BOTH backends by default.
#
# Measures the honest single-threaded, non-SIMD server setup cost
# (IkpirServer::new: derive the LWE matrix A from the seed, then compute the
# hint H = Aᵀ·D). The preprocessing acceleration (rayon + wide SIMD) has been
# reverted, so setup is inherently single-threaded — no RAYON_NUM_THREADS to
# set. Expect tens of seconds to minutes per build at paper scale.
#
# Usage:
#   ./scripts/run_server_setup.sh                                    # both backends (default)
#   IKPIR_BENCH_BACKENDS=frodo        ./scripts/run_server_setup.sh  # one backend only
#   IKPIR_BENCH_BACKENDS=simple       ./scripts/run_server_setup.sh
#   TRIALS=5 WARMUP=2 ./scripts/run_server_setup.sh   # more samples (slower)
#   MAX_MEM_GB=20.0   ./scripts/run_server_setup.sh   # override OOM guard (default 85 GB)
#
# Trials/warmup: each (config, value_bits, backend) times WARMUP+TRIALS
# IkpirServer::new builds; only the TRIALS post-warmup samples feed the
# reported mean/stddev. Defaults TRIALS=3, WARMUP=1 (override via env). Single-
# threaded setup is slow, so each extra trial costs a full rebuild.
#
# Memory: dominant cost is the per-segment LWE public matrix A, summed across
# segments to ≈ num_buckets × lwe_dim × 4 B. After the HintMaterial refactor
# `server.setup()` no longer copies A into the bundle (A is re-expanded
# client-side from the seed), so the server-side bench holds A exactly once.
# The shell guard below multiplies by 2 as a conservative safety margin (table
# cells, transient bundle clones during the wire-size readout, OS overhead).
# It is FrodoPIR-shaped — it does not divide by the SimplePIR √N reshape
# factor and so over-estimates SimplePIR — but even the over-estimate clears
# 85 GB for every BENCH_CONFIGS entry (largest peak ≈ 4194304 × 1566 × 4 × 2
# ≈ 52.5 GB), so nothing is skipped at the default MAX_MEM_GB. The guard still
# protects smaller machines.
#
# Config matrix (scripts/configs.sh) — identical to run_mutation.sh:
#   12 BENCH_CONFIGS × 3 VALUE_BITS = 36 runs/backend (× 2 backends = 72).
#   Note: server_setup populates to TableFull (not --load-factor 0.90 like the
#   mutation sweep); the (arity, bucket_size, num_buckets, value_bits,
#   plaintext_bits, lwe_dim) tuples are identical and the fill level does not
#   change compute_hint's row count, so setup time is comparable.
#
# Output:
#   ikpir-server/results/ikpir_server_setup.csv
#
# Columns: backend, arity, num_buckets, bucket_size, value_bits, plaintext_bits,
#   lwe_dim, mean_setup_ms, min_setup_ms, max_setup_ms, stddev_setup_ms,
#   setup_bundle_bytes, hint_bytes_per_segment, server_params_bytes_per_segment,
#   cells_per_slot, row_width, segment_rows, db_rows, db_cols, load_factor

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# Default to BOTH backends (matches "for both frodo and simple"); env overrides.
# Must be set before sourcing configs.sh, whose own `:-frodo` default then
# becomes a no-op because the variable is already non-empty.
IKPIR_BENCH_BACKENDS="${IKPIR_BENCH_BACKENDS:-frodo,simple}"

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

MAX_MEM_GB=${MAX_MEM_GB:-85.0}
MAX_MEM_BYTES=$(awk "BEGIN { printf \"%d\", $MAX_MEM_GB * 1e9 }")

# Warmup/trials per (config, value_bits, backend). Single-threaded setup is
# expensive, so these default low; raise for a tighter mean/stddev.
TRIALS=${TRIALS:-3}
WARMUP=${WARMUP:-1}

CARGO=(cargo bench -p ikpir-server --bench server_setup)

step "server_setup  (single-threaded; trials=${TRIALS}, warmup=${WARMUP}, max_mem=${MAX_MEM_GB} GB)"
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
            # FrodoPIR-shaped (no SimplePIR √N reshape divisor); see header.
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
                --trials "$TRIALS" --warmup "$WARMUP"
        done
    done
done

echo
ok "Done.  Results:"
ok "  $RESULTS_DIR/ikpir_server_setup.csv"
