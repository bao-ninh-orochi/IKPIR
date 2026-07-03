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
#   IKPIR_BENCH_PATCH_MODES=entry ./scripts/run_mutation.sh      # one realization (default entry,row)
#   MAX_MEM_GB=20.0 ./scripts/run_mutation.sh   # override OOM guard (default 85 GB)
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
# (default 85.0; tuned for ~96 GB servers — lower on smaller machines).
#
# Config matrix (scripts/configs.sh):
#   12 (arity, bucket_size, num_buckets) BENCH_CONFIGS × 3 value_bits
#   = 36 runs/backend (some may be skipped by the OOM guard).
#   (BENCH_CONFIGS is shared with the classical sweep — the historical
#   separate MUTATION_CONFIGS array was unified into BENCH_CONFIGS.)
#   n_mutations = min(capacity / 100, N_MUTATIONS_CAP)  (see below).
#   load_factor = MUTATION_LOAD_FACTOR (0.90).
#   patch modes = IKPIR_BENCH_PATCH_MODES (default entry,row): each run
#   emits one server + one client CSV row per (patch mode, kind) pair,
#   sharing one populate + IkpirServer::new across all pairs — the extra
#   mode adds only its timed loops, not another expensive setup.
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

MAX_MEM_GB=${MAX_MEM_GB:-85.0}

# Cap on n_mutations per (config, kind). Mutation throughput is a *rate*:
# the reported ops/s is stable regardless of how many mutations are timed
# (empirically within ~1 %: insert ops/s = 1450 at N=10485 vs 1440 at
# N=41943 on the same operating point). capacity/100 grows to ~42k at
# paper scale, which makes the timed loop the dominant cost for large
# value_bits without improving the rate estimate. Capping it keeps the
# loop short. Raise N_MUTATIONS_CAP for a tighter estimate, or set it
# huge (e.g. 100000000) to restore the pure capacity/100 rule.
N_MUTATIONS_CAP=${N_MUTATIONS_CAP:-2000}

CARGO=(cargo bench -p ikpir-client --bench mutation_throughput)

step "mutation_throughput  (server insert/update/delete + client apply_delta, shared populate+setup, max_mem=${MAX_MEM_GB}GB)"
rm -f "$RESULTS_DIR/ikpir_server_mutation.csv"
rm -f "$RESULTS_DIR/ikpir_client_mutation.csv"

for backend in "${BACKENDS_ARR[@]}"; do
    backend_note "$backend"
    lwe=$(backend_lwe_dim "$backend")
    for cfg in "${BENCH_CONFIGS[@]}"; do
        IFS=':' read -r arity bs nb n_label m_label <<< "$cfg"
        pb=$(backend_plaintext_bits "$backend" "$m_label")
        capacity=$(( nb * bs ))
        n_mut=$(( capacity / 100 ))   # 1 % of capacity
        if (( n_mut > N_MUTATIONS_CAP )); then n_mut=$N_MUTATIONS_CAP; fi
        for i in "${!VALUE_BITS[@]}"; do
            vb=${VALUE_BITS[$i]}
            w=${W_LABELS[$i]}
            note "arity=$arity bs=$bs nb=$nb (n=$n_label, m=$m_label) w=$w lwe=$lwe pb=$pb N=$n_mut lf=$MUTATION_LOAD_FACTOR modes=$IKPIR_BENCH_PATCH_MODES"
            "${CARGO[@]}" -- \
                --backend         "$backend"                 \
                --patch-mode      "$IKPIR_BENCH_PATCH_MODES" \
                --arity           "$arity"                   \
                --num-buckets     "$nb"                      \
                --bucket-size     "$bs"                      \
                --value-bits      "$vb"                      \
                --plaintext-bits  "$pb"                      \
                --lwe-dim         "$lwe"                     \
                --n-mutations     "$n_mut"                   \
                --load-factor     "$MUTATION_LOAD_FACTOR"    \
                --max-mem-gb      "$MAX_MEM_GB"
        done
    done
done

echo
ok "Done.  Results:"
ok "  $RESULTS_DIR/ikpir_server_mutation.csv"
ok "  $RESULTS_DIR/ikpir_client_mutation.csv"
