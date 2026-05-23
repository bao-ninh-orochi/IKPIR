#!/usr/bin/env bash
# Sweep server-side and client-side mutation throughput across the full
# mutation config matrix using the `mutation_throughput` merged bench.
# Fill + IkpirServer::new are shared between the server measurement and
# the client measurement for each kind, cutting per-config setup cost
# (7 expensive IkpirServer::new calls → 3) compared to running the
# individual server_mutation and client_mutation scripts.
#
# Usage:
#   ./scripts/run_mutation.sh                                    # FrodoPIR only (default)
#   IKPIR_BENCH_BACKENDS=simple       ./scripts/run_mutation.sh
#   IKPIR_BENCH_BACKENDS=frodo,simple ./scripts/run_mutation.sh  # both backends
#   MAX_MEM_GB=20.0 ./scripts/run_mutation.sh   # raise OOM guard (default 12 GB)
#
# Memory: dominant allocation is the per-segment LWE public matrix A (in
# B::HintMaterial). After the HintMaterial refactor setup() no longer ships
# A over the wire. run_server_kind keeps A resident throughout its mutation
# loop (no drop_hint_material — that would re-expand A on the first
# mutation and bias the timing) and the server drops at function return,
# before the client is built. The client runs in empty-queue mode (no
# precompute_queries / precompute_decodes), so no prepared-query queue
# coexists with A. Peak coexisting A copies = 1 (server during mutations,
# then the client during apply_delta).
#
# Per-segment A shape is backend-aware:
#   FrodoPIR:  (n_rows, lwe_dim)
#   SimplePIR: (reshape_rows, lwe_dim) — ~100× smaller via the √N reshape.
#
# The in-bench estimator (mutation_throughput.rs) accounts for table + A
# and skips any config whose estimated peak exceeds MAX_MEM_GB
# (default 12.0). Raise it on 32 GB+ machines.
#
# Config matrix (scripts/configs.sh):
#   6 (arity, bucket_size, num_buckets) MUTATION_CONFIGS × 3 value_bits
#   = 18 runs/backend (some may be skipped by the OOM guard).
#   n_mutations = capacity / 100  (1 % of capacity per config).
#   load_factor = MUTATION_LOAD_FACTOR (0.90).
#
# Output (two CSV files in results/):
#   ikpir-client/results/ikpir_server_mutation.csv
#   ikpir-client/results/ikpir_client_mutation.csv
#
# Note: ikpir_server_mutation.csv lands under ikpir-client/results/
# because the combined bench runs from the client crate (mirrors how
# classical_throughput places ikpir_server_answer.csv there).
#
# WARNING: These CSV files share their schema with the output of the
# individual bench scripts (ikpir-server/scripts/run_benches.sh
# server_mutation and ikpir-client/scripts/run_benches.sh
# client_mutation, which write to ikpir-server/results/ and
# ikpir-client/results/ respectively).  Run this script OR those
# scripts for a given config set — not both — to avoid duplicate rows.

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

CARGO=(cargo bench -p ikpir-client --bench mutation_throughput)

step "mutation_throughput  (server insert/update/delete + client apply_delta, shared populate+setup, max_mem=${MAX_MEM_GB}GB)"
rm -f "$RESULTS_DIR/ikpir_server_mutation.csv"
rm -f "$RESULTS_DIR/ikpir_client_mutation.csv"

for backend in "${BACKENDS_ARR[@]}"; do
    backend_note "$backend"
    lwe=$(backend_lwe_dim "$backend")
    for cfg in "${MUTATION_CONFIGS[@]}"; do
        IFS=':' read -r arity bs nb n_label m_label <<< "$cfg"
        capacity=$(( nb * bs ))
        n_mut=$(( capacity / 100 ))   # 1 % of capacity, same rule as today
        for i in "${!VALUE_BITS[@]}"; do
            vb=${VALUE_BITS[$i]}
            w=${W_LABELS[$i]}
            note "arity=$arity bs=$bs nb=$nb (n=$n_label, m=$m_label) w=$w lwe=$lwe N=$n_mut lf=$MUTATION_LOAD_FACTOR"
            "${CARGO[@]}" -- \
                --backend      "$backend"              \
                --arity        "$arity"                \
                --num-buckets  "$nb"                   \
                --bucket-size  "$bs"                   \
                --value-bits   "$vb"                   \
                --lwe-dim      "$lwe"                  \
                --n-mutations  "$n_mut"                \
                --load-factor  "$MUTATION_LOAD_FACTOR" \
                --max-mem-gb   "$MAX_MEM_GB"
        done
    done
done

echo
ok "Done.  Results:"
ok "  $RESULTS_DIR/ikpir_server_mutation.csv"
ok "  $RESULTS_DIR/ikpir_client_mutation.csv"
