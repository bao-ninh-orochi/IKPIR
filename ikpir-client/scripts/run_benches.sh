#!/usr/bin/env bash
# Run every ikpir-client bench across the paper-derived config matrices.
#
# Usage:
#   ./scripts/run_benches.sh                      # all benches, default profile
#   ./scripts/run_benches.sh query_throughput     # one bench
#   IKPIR_BENCH_PROFILE=quick ./scripts/run_benches.sh   # smaller matrix
#
# Per-bench config matrix:
#   query_throughput          A (m sweep) × G (mode) + B (w sweep, cold mode)
#   decode_throughput         A (m sweep) × G (mode) + B (w sweep, cold mode)
#   apply_delta_throughput    F-like (num_buckets sweep)
#   preprocess_throughput     A (m sweep, batch=64)
#   client_setup_latency      A (m sweep, with --with-precompute)
#   client_memory_footprint   A (m sweep) × G (mode); closed-form, fast

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
    BLUE='\033[34m'; GREEN='\033[32m'; YELLOW='\033[33m'; RESET='\033[0m'
else
    BLUE=''; GREEN=''; YELLOW=''; RESET=''
fi

step() { echo -e "${BLUE}── $* ──${RESET}"; }
note() { echo -e "${YELLOW}  $*${RESET}"; }
ok()   { echo -e "${GREEN}  $*${RESET}"; }

# ─────────────────────────────────────────────────────────────────────────────
# query_throughput — A × G + B (cold mode)
# ─────────────────────────────────────────────────────────────────────────────
run_query_throughput() {
    step "query_throughput"
    rm -f "$RESULTS_DIR/ikpir_client_query_throughput.csv"

    # A × G: m sweep across all preprocessing modes.
    for nb in "${MATRIX_A_NB_ARITY2[@]}"; do
        for mode in "${MATRIX_G_MODES[@]}"; do
            note "A×G: num_buckets=$nb mode=$mode"
            "${CARGO[@]}" query_throughput -- \
                --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                --lwe-dim "$MATRIX_A_LWE_DIM" --mode "$mode" --batch 64
        done
    done

    # B (value-size sweep, cold mode only — keep CSV size in check).
    for i in "${!MATRIX_B_VALUE_BITS[@]}"; do
        vb=${MATRIX_B_VALUE_BITS[$i]}; w=${MATRIX_B_W_LABELS[$i]}
        note "B: num_buckets=$MATRIX_B_NB_ARITY2 value_bits=$vb (w=$w) mode=cold"
        "${CARGO[@]}" query_throughput -- \
            --arity "$MATRIX_B_ARITY" --num-buckets "$MATRIX_B_NB_ARITY2" \
            --bucket-size "$MATRIX_B_BUCKET_SIZE" --value-bits "$vb" \
            --lwe-dim "$MATRIX_B_LWE_DIM" --mode cold --batch 64
    done
    ok "→ $RESULTS_DIR/ikpir_client_query_throughput.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# decode_throughput — A × G + B (cold mode)
# ─────────────────────────────────────────────────────────────────────────────
run_decode_throughput() {
    step "decode_throughput"
    rm -f "$RESULTS_DIR/ikpir_client_decode_throughput.csv"

    for nb in "${MATRIX_A_NB_ARITY2[@]}"; do
        for mode in "${MATRIX_G_MODES[@]}"; do
            note "A×G: num_buckets=$nb mode=$mode"
            "${CARGO[@]}" decode_throughput -- \
                --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                --lwe-dim "$MATRIX_A_LWE_DIM" --mode "$mode" --batch 64 --trials 5 --warmup 2
        done
    done

    for i in "${!MATRIX_B_VALUE_BITS[@]}"; do
        vb=${MATRIX_B_VALUE_BITS[$i]}; w=${MATRIX_B_W_LABELS[$i]}
        note "B: num_buckets=$MATRIX_B_NB_ARITY2 value_bits=$vb (w=$w) mode=cold"
        "${CARGO[@]}" decode_throughput -- \
            --arity "$MATRIX_B_ARITY" --num-buckets "$MATRIX_B_NB_ARITY2" \
            --bucket-size "$MATRIX_B_BUCKET_SIZE" --value-bits "$vb" \
            --lwe-dim "$MATRIX_B_LWE_DIM" --mode cold --batch 64 --trials 5 --warmup 2
    done
    ok "→ $RESULTS_DIR/ikpir_client_decode_throughput.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# apply_delta_throughput — m sweep × {precomputed_slots=0, batch}
# ─────────────────────────────────────────────────────────────────────────────
run_apply_delta_throughput() {
    step "apply_delta_throughput"
    rm -f "$RESULTS_DIR/ikpir_client_apply_delta_throughput.csv"

    for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
        nb=${MATRIX_A_NB_ARITY2[$i]}; m=${MATRIX_A_M_LABELS[$i]}
        # Apply-delta batch size: scale with capacity but cap at 4096.
        capacity=$(( nb * 4 ))
        batch=$(( capacity / 8 ))   # 12.5% of capacity worth of inserts
        if (( batch > 4096 )); then batch=4096; fi
        if (( batch < 64 )); then batch=64; fi
        note "A: num_buckets=$nb (m≈$m) batch=$batch precomputed_slots=0"
        "${CARGO[@]}" apply_delta_throughput -- \
            --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
            --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
            --lwe-dim "$MATRIX_A_LWE_DIM" --batch "$batch" \
            --precomputed-slots 0 --trials 5 --warmup 1 --load-factor 0.50
        note "A: num_buckets=$nb (m≈$m) batch=$batch precomputed_slots=$batch (warm path)"
        "${CARGO[@]}" apply_delta_throughput -- \
            --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
            --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
            --lwe-dim "$MATRIX_A_LWE_DIM" --batch "$batch" \
            --precomputed-slots "$batch" --trials 5 --warmup 1 --load-factor 0.50
    done
    ok "→ $RESULTS_DIR/ikpir_client_apply_delta_throughput.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# preprocess_throughput — A (m sweep)
# ─────────────────────────────────────────────────────────────────────────────
run_preprocess_throughput() {
    step "preprocess_throughput"
    rm -f "$RESULTS_DIR/ikpir_client_preprocess_throughput.csv"

    for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
        nb=${MATRIX_A_NB_ARITY2[$i]}; m=${MATRIX_A_M_LABELS[$i]}
        note "A: num_buckets=$nb (m≈$m) batch=64"
        "${CARGO[@]}" preprocess_throughput -- \
            --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
            --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
            --lwe-dim "$MATRIX_A_LWE_DIM" --batch 64 --trials 5 --warmup 2
    done
    ok "→ $RESULTS_DIR/ikpir_client_preprocess_throughput.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# client_setup_latency — A (m sweep) + --with-precompute variant
# ─────────────────────────────────────────────────────────────────────────────
run_client_setup_latency() {
    step "client_setup_latency"
    rm -f "$RESULTS_DIR/ikpir_client_setup_latency.csv"

    for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
        nb=${MATRIX_A_NB_ARITY2[$i]}; m=${MATRIX_A_M_LABELS[$i]}
        note "A: num_buckets=$nb (m≈$m) (cold)"
        "${CARGO[@]}" client_setup_latency -- \
            --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
            --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
            --lwe-dim "$MATRIX_A_LWE_DIM" --trials 5 --warmup 2 --batch 64
        note "A: num_buckets=$nb (m≈$m) (with precompute)"
        "${CARGO[@]}" client_setup_latency -- \
            --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
            --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
            --lwe-dim "$MATRIX_A_LWE_DIM" --trials 5 --warmup 2 --batch 64 \
            --with-precompute
    done
    ok "→ $RESULTS_DIR/ikpir_client_setup_latency.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# client_memory_footprint — A × G (closed-form, fast)
# ─────────────────────────────────────────────────────────────────────────────
run_client_memory_footprint() {
    step "client_memory_footprint"
    rm -f "$RESULTS_DIR/ikpir_client_memory_footprint.csv"

    for nb in "${MATRIX_A_NB_ARITY2[@]}"; do
        for mode in "${MATRIX_G_MODES[@]}"; do
            note "A×G: num_buckets=$nb mode=$mode"
            "${CARGO[@]}" client_memory_footprint -- \
                --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                --lwe-dim "$MATRIX_A_LWE_DIM" --mode "$mode" --batch 64
        done
    done
    ok "→ $RESULTS_DIR/ikpir_client_memory_footprint.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

ALL_BENCHES=(
    query_throughput
    decode_throughput
    apply_delta_throughput
    preprocess_throughput
    client_setup_latency
    client_memory_footprint
)

run_one() {
    case "$1" in
        query_throughput)         run_query_throughput ;;
        decode_throughput)        run_decode_throughput ;;
        apply_delta_throughput)   run_apply_delta_throughput ;;
        preprocess_throughput)    run_preprocess_throughput ;;
        client_setup_latency)     run_client_setup_latency ;;
        client_memory_footprint)  run_client_memory_footprint ;;
        *) echo "unknown bench: $1"; echo "valid: ${ALL_BENCHES[*]}"; exit 1 ;;
    esac
}

if (( $# == 0 )); then
    for b in "${ALL_BENCHES[@]}"; do run_one "$b"; done
else
    for b in "$@"; do run_one "$b"; done
fi

echo
ok "Done. CSVs under $RESULTS_DIR/"
ok "Next: cd $CRATE_DIR && python scripts/plot.py"
