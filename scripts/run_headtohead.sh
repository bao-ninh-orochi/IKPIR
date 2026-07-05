#!/usr/bin/env bash
# Sweep server_answer + client_query + client_decode across the head-to-head
# config matrix using the `headtohead_throughput` bench. Unlike `run_classical.sh`
# (which fixes DB size and populates `until_full`), this fixes the keyword
# count `num_keys` and lets the CSV report the resulting DB size so the
# expansion rate vs ChalametPIR / Hao et al. 2025 can be computed at analysis
# time. Fill + hint-matrix computation are paid once per config.
#
# Usage:
#   ./scripts/run_headtohead.sh                    # FrodoPIR only (default)
#   IKPIR_BENCH_BACKENDS=simple       ./scripts/run_headtohead.sh
#   IKPIR_BENCH_BACKENDS=frodo,simple ./scripts/run_headtohead.sh  # both backends
#   MAX_MEM_GB=20.0 ./scripts/run_headtohead.sh    # override OOM guard (default 85 GB)
#
# Memory: same dominant terms as run_classical.sh (per-segment LWE matrix A,
# pre-built queries Vec). The in-bench estimator
# (headtohead_throughput.rs) accounts for table + A + queries Vec and skips
# any config whose estimated peak exceeds MAX_MEM_GB (default 85.0; tuned
# for ~96 GB servers — lower on smaller machines).
#
# Config matrix (scripts/configs.sh::HEADTOHEAD_CONFIGS):
#   10 (num_keys, arity, bucket_size, num_buckets) tuples × 3 value_bits
#   = 30 runs/backend (some may be skipped by the OOM guard).
#
# Output (three CSV files in ikpir-client/results/):
#   ikpir_headtohead_server_answer.csv
#   ikpir_headtohead_client_query.csv
#   ikpir_headtohead_client_decode.csv
#
# Each CSV row carries `num_keys` and `db_size_slots` so the per-row
# expansion rate is just `db_size_slots / num_keys` (or `load_factor` for
# the reciprocal). All other columns mirror run_classical.sh's CSVs.

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

MAX_MEM_GB=${MAX_MEM_GB:-85.0}

CARGO=(cargo bench -p ikpir-client --bench headtohead_throughput)

step "headtohead_throughput  (server_answer + client_query + client_decode, fixed N, shared fill+setup, max_mem=${MAX_MEM_GB}GB)"
rm -f "$RESULTS_DIR/ikpir_headtohead_server_answer.csv"
rm -f "$RESULTS_DIR/ikpir_headtohead_client_query.csv"
rm -f "$RESULTS_DIR/ikpir_headtohead_client_decode.csv"

for backend in "${BACKENDS_ARR[@]}"; do
    backend_note "$backend"
    lwe=$(backend_lwe_dim "$backend")
    for cfg in "${HEADTOHEAD_CONFIGS[@]}"; do
        IFS=':' read -r num_keys arity bs nb m_label <<< "$cfg"
        for i in "${!VALUE_BITS[@]}"; do
            vb=${VALUE_BITS[$i]}
            w=${W_LABELS[$i]}
            # pb depends on value_bits for the SimplePIR backend, so the
            # lookup lives inside the value-bits loop.
            pb=$(backend_plaintext_bits "$backend" "$arity" "$bs" "$nb" "$vb")
            note "N=$num_keys arity=$arity bs=$bs nb=$nb (m=$m_label) w=$w lwe=$lwe pb=$pb"
            "${CARGO[@]}" -- \
                --backend         "$backend"     \
                --num-keys        "$num_keys"    \
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
ok "  $RESULTS_DIR/ikpir_headtohead_server_answer.csv"
ok "  $RESULTS_DIR/ikpir_headtohead_client_query.csv"
ok "  $RESULTS_DIR/ikpir_headtohead_client_decode.csv"
