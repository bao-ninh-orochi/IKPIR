#!/usr/bin/env bash
# Run every ikpir-client bench across the full config matrix once per backend.
#
# Usage:
#   ./scripts/run_benches.sh                    # all 3 benches
#   ./scripts/run_benches.sh client_query       # one bench
#   IKPIR_BENCH_BACKENDS=frodo                  # default: FrodoPIR only
#   IKPIR_BENCH_BACKENDS=simple                 # SimplePIR only
#   IKPIR_BENCH_BACKENDS=frodo,simple           # both backends (~2× runtime)
#   IKPIR_BENCH_PATCH_MODES=entry               # client_mutation: one hint-patch realization (default entry,row)
#
# Config matrix (scripts/configs.sh):
#   client_query    12 cfgs × 3 value_bits = 36 runs/backend  (warm-bc)
#   client_decode   12 cfgs × 3 value_bits = 36 runs/backend  (warm-bc)
#   client_mutation 12 cfgs × 3 value_bits × N=1% capacity = 36 runs/backend (empty-queue)
#                   (each run emits one CSV row per (patch mode, kind) pair)
# (All three sweeps now share BENCH_CONFIGS — the historical separate
# MUTATION_CONFIGS array was unified into BENCH_CONFIGS.)
#
# One invocation = one CSV row (append-mode). The CSV is rm'd before each
# sweep so prior rows don't accumulate. Criterion's target/criterion/ is
# left untouched.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CRATE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_DIR=$(cd "$CRATE_DIR/.." && pwd)

# shellcheck source=../../scripts/configs.sh
source "$WORKSPACE_DIR/scripts/configs.sh"

cd "$WORKSPACE_DIR"

RESULTS_DIR="$CRATE_DIR/results"
mkdir -p "$RESULTS_DIR"

CARGO=(cargo bench -p ikpir-client --bench)

if [[ -t 1 ]]; then
    BLUE='\033[34m'; GREEN='\033[32m'; YELLOW='\033[33m'; MAGENTA='\033[35m'; RESET='\033[0m'
else
    BLUE=''; GREEN=''; YELLOW=''; MAGENTA=''; RESET=''
fi

step()         { echo -e "${BLUE}── $* ──${RESET}"; }
note()         { echo -e "${YELLOW}  $*${RESET}"; }
ok()           { echo -e "${GREEN}  $*${RESET}"; }
backend_note() { echo -e "${MAGENTA}▶ backend=$1${RESET}"; }

# ─────────────────────────────────────────────────────────────────────────────
# Helper: iterate BENCH_CONFIGS × VALUE_BITS (36 pairs).
# Calls `$1 arity bucket_size num_buckets n_label m_label value_bits w_label`
# ─────────────────────────────────────────────────────────────────────────────

for_each_bench_config() {
    local cb=$1
    local cfg arity bs nb n_label m_label
    local i vb w
    for cfg in "${BENCH_CONFIGS[@]}"; do
        IFS=':' read -r arity bs nb n_label m_label <<< "$cfg"
        pb=$(backend_plaintext_bits "$backend" "$m_label")
        for i in "${!VALUE_BITS[@]}"; do
            vb=${VALUE_BITS[$i]}
            w=${W_LABELS[$i]}
            "$cb" "$arity" "$bs" "$nb" "$n_label" "$m_label" "$vb" "$w"
        done
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# client_query — 36 runs/backend  (warm-bc)
# ─────────────────────────────────────────────────────────────────────────────
run_client_query() {
    step "client_query"
    rm -f "$RESULTS_DIR/ikpir_client_query.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        local lwe pb; lwe=$(backend_lwe_dim "$backend")
        _query_one() {
            local arity=$1 bs=$2 nb=$3 n_label=$4 m_label=$5 vb=$6 w=$7
            note "arity=$arity bs=$bs nb=$nb (n=$n_label, m=$m_label) w=$w lwe=$lwe pb=$pb"
            "${CARGO[@]}" client_query -- \
                --backend "$backend" \
                --arity "$arity" --num-buckets "$nb" \
                --bucket-size "$bs" --value-bits "$vb" \
                --plaintext-bits "$pb" \
                --lwe-dim "$lwe" --batch 64
        }
        for_each_bench_config _query_one
    done
    ok "→ $RESULTS_DIR/ikpir_client_query.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# client_decode — 36 runs/backend  (warm-bc)
# ─────────────────────────────────────────────────────────────────────────────
run_client_decode() {
    step "client_decode"
    rm -f "$RESULTS_DIR/ikpir_client_decode.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        local lwe pb; lwe=$(backend_lwe_dim "$backend")
        _decode_one() {
            local arity=$1 bs=$2 nb=$3 n_label=$4 m_label=$5 vb=$6 w=$7
            note "arity=$arity bs=$bs nb=$nb (n=$n_label, m=$m_label) w=$w lwe=$lwe pb=$pb"
            "${CARGO[@]}" client_decode -- \
                --backend "$backend" \
                --arity "$arity" --num-buckets "$nb" \
                --bucket-size "$bs" --value-bits "$vb" \
                --plaintext-bits "$pb" \
                --lwe-dim "$lwe" --batch 64
        }
        for_each_bench_config _decode_one
    done
    ok "→ $RESULTS_DIR/ikpir_client_decode.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# client_mutation — 36 runs/backend  (BENCH_CONFIGS × VALUE_BITS, N=1% capacity)
# ─────────────────────────────────────────────────────────────────────────────
run_client_mutation() {
    step "client_mutation"
    rm -f "$RESULTS_DIR/ikpir_client_mutation.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        local lwe; lwe=$(backend_lwe_dim "$backend")
        local cfg arity bs nb n_label m_label n_mut pb i vb w
        for cfg in "${BENCH_CONFIGS[@]}"; do
            IFS=':' read -r arity bs nb n_label m_label <<< "$cfg"
            pb=$(backend_plaintext_bits "$backend" "$m_label")
            n_mut=$(( nb * bs / 100 ))
            for i in "${!VALUE_BITS[@]}"; do
                vb=${VALUE_BITS[$i]}
                w=${W_LABELS[$i]}
                note "arity=$arity bs=$bs nb=$nb (m=$m_label) w=$w N=$n_mut lwe=$lwe pb=$pb modes=$IKPIR_BENCH_PATCH_MODES"
                "${CARGO[@]}" client_mutation -- \
                    --backend "$backend" \
                    --patch-mode "$IKPIR_BENCH_PATCH_MODES" \
                    --arity "$arity" --num-buckets "$nb" \
                    --bucket-size "$bs" --value-bits "$vb" \
                    --plaintext-bits "$pb" \
                    --lwe-dim "$lwe" \
                    --n-mutations "$n_mut" --load-factor "$MUTATION_LOAD_FACTOR"
            done
        done
    done
    ok "→ $RESULTS_DIR/ikpir_client_mutation.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

ALL_BENCHES=(client_query client_decode client_mutation)

run_one_bench() {
    case "$1" in
        client_query)    run_client_query ;;
        client_decode)   run_client_decode ;;
        client_mutation) run_client_mutation ;;
        *) echo "unknown bench: $1"; echo "valid: ${ALL_BENCHES[*]}"; exit 1 ;;
    esac
}

if (( $# == 0 )); then
    for b in "${ALL_BENCHES[@]}"; do run_one_bench "$b"; done
else
    for b in "$@"; do run_one_bench "$b"; done
fi

echo
ok "Done. CSVs under $RESULTS_DIR/"
