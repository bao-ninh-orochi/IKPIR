#!/usr/bin/env bash
# Run every ikpir-client bench across the paper-derived config matrices,
# once per backend selected by IKPIR_BENCH_BACKENDS.
#
# Usage:
#   ./scripts/run_benches.sh                      # all benches, default profile
#   ./scripts/run_benches.sh query_throughput     # one bench
#   IKPIR_BENCH_PROFILE=quick ./scripts/run_benches.sh   # smaller matrix
#   IKPIR_BENCH_BACKENDS=frodo                            # default: FrodoPIR only
#   IKPIR_BENCH_BACKENDS=simple                           # SimplePIR only
#   IKPIR_BENCH_BACKENDS=frodo,simple                     # both (~2× runtime)
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
    BLUE='\033[34m'; GREEN='\033[32m'; YELLOW='\033[33m'; MAGENTA='\033[35m'; RESET='\033[0m'
else
    BLUE=''; GREEN=''; YELLOW=''; MAGENTA=''; RESET=''
fi

step() { echo -e "${BLUE}── $* ──${RESET}"; }
note() { echo -e "${YELLOW}  $*${RESET}"; }
ok()   { echo -e "${GREEN}  $*${RESET}"; }
backend_note() { echo -e "${MAGENTA}▶ backend=$1${RESET}"; }

# ─────────────────────────────────────────────────────────────────────────────
# query_throughput — A × G + B (cold mode) × {backend}
# ─────────────────────────────────────────────────────────────────────────────
run_query_throughput() {
    step "query_throughput"
    rm -f "$RESULTS_DIR/ikpir_client_query_throughput.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        lwe=$(backend_lwe_dim "$backend")
        for nb in "${MATRIX_A_NB_ARITY2[@]}"; do
            for mode in "${MATRIX_G_MODES[@]}"; do
                note "A×G: num_buckets=$nb mode=$mode lwe_dim=$lwe"
                "${CARGO[@]}" query_throughput -- \
                    --backend "$backend" \
                    --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                    --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                    --lwe-dim "$lwe" --mode "$mode" --batch 64
            done
        done

        for i in "${!MATRIX_B_VALUE_BITS[@]}"; do
            vb=${MATRIX_B_VALUE_BITS[$i]}; w=${MATRIX_B_W_LABELS[$i]}
            note "B: num_buckets=$MATRIX_B_NB_ARITY2 value_bits=$vb (w=$w) mode=cold lwe_dim=$lwe"
            "${CARGO[@]}" query_throughput -- \
                --backend "$backend" \
                --arity "$MATRIX_B_ARITY" --num-buckets "$MATRIX_B_NB_ARITY2" \
                --bucket-size "$MATRIX_B_BUCKET_SIZE" --value-bits "$vb" \
                --lwe-dim "$lwe" --mode cold --batch 64
        done
    done
    ok "→ $RESULTS_DIR/ikpir_client_query_throughput.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# decode_throughput — A × G + B (cold mode) × {backend}
# ─────────────────────────────────────────────────────────────────────────────
run_decode_throughput() {
    step "decode_throughput"
    rm -f "$RESULTS_DIR/ikpir_client_decode_throughput.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        lwe=$(backend_lwe_dim "$backend")
        for nb in "${MATRIX_A_NB_ARITY2[@]}"; do
            for mode in "${MATRIX_G_MODES[@]}"; do
                note "A×G: num_buckets=$nb mode=$mode lwe_dim=$lwe"
                "${CARGO[@]}" decode_throughput -- \
                    --backend "$backend" \
                    --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                    --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                    --lwe-dim "$lwe" --mode "$mode" --batch 64
            done
        done

        for i in "${!MATRIX_B_VALUE_BITS[@]}"; do
            vb=${MATRIX_B_VALUE_BITS[$i]}; w=${MATRIX_B_W_LABELS[$i]}
            note "B: num_buckets=$MATRIX_B_NB_ARITY2 value_bits=$vb (w=$w) mode=cold lwe_dim=$lwe"
            "${CARGO[@]}" decode_throughput -- \
                --backend "$backend" \
                --arity "$MATRIX_B_ARITY" --num-buckets "$MATRIX_B_NB_ARITY2" \
                --bucket-size "$MATRIX_B_BUCKET_SIZE" --value-bits "$vb" \
                --lwe-dim "$lwe" --mode cold --batch 64
        done
    done
    ok "→ $RESULTS_DIR/ikpir_client_decode_throughput.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# apply_delta_throughput — m sweep × {precomputed_slots=0, batch} × {backend}
# ─────────────────────────────────────────────────────────────────────────────
run_apply_delta_throughput() {
    step "apply_delta_throughput"
    rm -f "$RESULTS_DIR/ikpir_client_apply_delta_throughput.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        lwe=$(backend_lwe_dim "$backend")
        for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
            nb=${MATRIX_A_NB_ARITY2[$i]}; m=${MATRIX_A_M_LABELS[$i]}
            capacity=$(( nb * 4 ))
            batch=$(( capacity / 8 ))
            if (( batch > 4096 )); then batch=4096; fi
            if (( batch < 64 )); then batch=64; fi
            note "A: num_buckets=$nb (m≈$m) batch=$batch precomputed_slots=0 lwe_dim=$lwe"
            "${CARGO[@]}" apply_delta_throughput -- \
                --backend "$backend" \
                --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                --lwe-dim "$lwe" --batch "$batch" \
                --precomputed-slots 0 --load-factor 0.50
            note "A: num_buckets=$nb (m≈$m) batch=$batch precomputed_slots=$batch (warm path) lwe_dim=$lwe"
            "${CARGO[@]}" apply_delta_throughput -- \
                --backend "$backend" \
                --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                --lwe-dim "$lwe" --batch "$batch" \
                --precomputed-slots "$batch" --load-factor 0.50
        done
    done
    ok "→ $RESULTS_DIR/ikpir_client_apply_delta_throughput.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# preprocess_throughput — A (m sweep) × {backend}
# ─────────────────────────────────────────────────────────────────────────────
run_preprocess_throughput() {
    step "preprocess_throughput"
    rm -f "$RESULTS_DIR/ikpir_client_preprocess_throughput.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        lwe=$(backend_lwe_dim "$backend")
        for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
            nb=${MATRIX_A_NB_ARITY2[$i]}; m=${MATRIX_A_M_LABELS[$i]}
            note "A: num_buckets=$nb (m≈$m) batch=64 lwe_dim=$lwe"
            "${CARGO[@]}" preprocess_throughput -- \
                --backend "$backend" \
                --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                --lwe-dim "$lwe" --batch 64
        done
    done
    ok "→ $RESULTS_DIR/ikpir_client_preprocess_throughput.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# client_setup_latency — A (m sweep) + --with-precompute variant × {backend}
# ─────────────────────────────────────────────────────────────────────────────
run_client_setup_latency() {
    step "client_setup_latency"
    rm -f "$RESULTS_DIR/ikpir_client_setup_latency.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        lwe=$(backend_lwe_dim "$backend")
        for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
            nb=${MATRIX_A_NB_ARITY2[$i]}; m=${MATRIX_A_M_LABELS[$i]}
            note "A: num_buckets=$nb (m≈$m) (cold) lwe_dim=$lwe"
            "${CARGO[@]}" client_setup_latency -- \
                --backend "$backend" \
                --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                --lwe-dim "$lwe" --trials 5 --warmup 2 --batch 64
            note "A: num_buckets=$nb (m≈$m) (with precompute) lwe_dim=$lwe"
            "${CARGO[@]}" client_setup_latency -- \
                --backend "$backend" \
                --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                --lwe-dim "$lwe" --trials 5 --warmup 2 --batch 64 \
                --with-precompute
        done
    done
    ok "→ $RESULTS_DIR/ikpir_client_setup_latency.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# client_memory_footprint — A × G (closed-form, fast) × {backend}
# ─────────────────────────────────────────────────────────────────────────────
run_client_memory_footprint() {
    step "client_memory_footprint"
    rm -f "$RESULTS_DIR/ikpir_client_memory_footprint.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        lwe=$(backend_lwe_dim "$backend")
        for nb in "${MATRIX_A_NB_ARITY2[@]}"; do
            for mode in "${MATRIX_G_MODES[@]}"; do
                note "A×G: num_buckets=$nb mode=$mode lwe_dim=$lwe"
                "${CARGO[@]}" client_memory_footprint -- \
                    --backend "$backend" \
                    --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                    --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                    --lwe-dim "$lwe" --mode "$mode" --batch 64
            done
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
