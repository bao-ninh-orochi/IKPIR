#!/usr/bin/env bash
# Sweep server_answer + client_query + client_decode across the full config
# matrix using the `classical_throughput` merged bench.  Fill and hint-matrix
# computation are paid once per config invocation instead of three times,
# cutting sweep time roughly 3× compared to running the individual bench
# scripts for those three operations.
#
# Usage:
#   ./scripts/run_classical.sh                    # FrodoPIR only (default)
#   IKPIR_BENCH_BACKENDS=simple       ./scripts/run_classical.sh
#   IKPIR_BENCH_BACKENDS=frodo,simple ./scripts/run_classical.sh  # both backends
#   MAX_MEM_GB=20.0 ./scripts/run_classical.sh   # raise OOM guard (default 12 GB)
#
# Memory: dominant cost is the per-segment LWE public matrix A, held in
# B::HintMaterial. After the HintMaterial refactor setup() no longer ships A
# over the wire and the server drops its copy immediately after setup; peak
# coexisting copies = 1 (only the active client). The pre-built `queries` Vec
# held across all three measurements (batch × arity × b_len × 4 B) adds ~1 GB
# at paper-scale Frodo + batch=64 and is negligible for SimplePIR.
#
# Per-segment A shape is backend-aware:
#   FrodoPIR:  (n_rows, lwe_dim)
#   SimplePIR: (reshape_rows, lwe_dim) — ~100× smaller via the √N reshape.
#
# The in-bench estimator (classical_throughput.rs) accounts for table + A +
# queries Vec and skips any config whose estimated peak exceeds MAX_MEM_GB
# (default 12.0). Raise it on 32 GB+ machines.
#
# Config matrix (scripts/configs.sh):
#   12 (arity, bucket_size, num_buckets) tuples × 3 value_bits = 36 runs/backend
#   (some may be skipped by the OOM guard).
#
# Output (three CSV files in results/):
#   results/ikpir_server_answer.csv
#   results/ikpir_client_query.csv
#   results/ikpir_client_decode.csv
#
# WARNING: These CSV files share their schema with the output of the individual
# bench scripts (ikpir-server/scripts/run_benches.sh server_answer and
# ikpir-client/scripts/run_benches.sh client_query / client_decode).
# Run this script OR those scripts for a given config set — not both —
# to avoid duplicate rows in the CSV files.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# shellcheck source=./configs.sh
source "$WORKSPACE_DIR/scripts/configs.sh"

cd "$WORKSPACE_DIR"

# The bench binary (cargo bench -p ikpir-client) runs with CWD = ikpir-client/,
# so helpers::csv_writer("results/...") writes to ikpir-client/results/.
RESULTS_DIR="$WORKSPACE_DIR/ikpir-client/results"
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

CARGO=(cargo bench -p ikpir-client --bench classical_throughput)

step "classical_throughput  (server_answer + client_query + client_decode, shared fill+setup, max_table=${MAX_MEM_GB}GB)"
rm -f "$RESULTS_DIR/ikpir_server_answer.csv"
rm -f "$RESULTS_DIR/ikpir_client_query.csv"
rm -f "$RESULTS_DIR/ikpir_client_decode.csv"

for backend in "${BACKENDS_ARR[@]}"; do
    backend_note "$backend"
    lwe=$(backend_lwe_dim "$backend")
    for cfg in "${BENCH_CONFIGS[@]}"; do
        IFS=':' read -r arity bs nb n_label m_label <<< "$cfg"
        pb=$(backend_plaintext_bits "$backend" "$m_label")
        for i in "${!VALUE_BITS[@]}"; do
            vb=${VALUE_BITS[$i]}
            w=${W_LABELS[$i]}
            note "arity=$arity bs=$bs nb=$nb (n=$n_label, m=$m_label) w=$w lwe=$lwe pb=$pb"
            "${CARGO[@]}" -- \
                --backend         "$backend"     \
                --arity           "$arity"       \
                --num-buckets     "$nb"          \
                --bucket-size     "$bs"          \
                --value-bits      "$vb"          \
                --plaintext-bits  "$pb"          \
                --lwe-dim         "$lwe"         \
                --batch           64             \
                --max-mem-gb      "$MAX_MEM_GB"
        done
    done
done

echo
ok "Done.  Results:"
ok "  $RESULTS_DIR/ikpir_server_answer.csv"
ok "  $RESULTS_DIR/ikpir_client_query.csv"
ok "  $RESULTS_DIR/ikpir_client_decode.csv"
