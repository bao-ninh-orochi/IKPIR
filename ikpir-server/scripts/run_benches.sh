#!/usr/bin/env bash
# Run every ikpir-server bench across the paper-derived config matrices.
#
# Usage:
#   ./scripts/run_benches.sh                    # all benches, default profile
#   ./scripts/run_benches.sh setup_latency      # one bench
#   IKPIR_BENCH_PROFILE=quick ./scripts/run_benches.sh   # smaller matrix
#   IKPIR_BENCH_PROFILE=full  ./scripts/run_benches.sh   # add 2^22 keys row
#
# Per-bench config matrix:
#   setup_latency           A (m sweep) + C (arity sweep)
#   answer_throughput       A (m sweep) + B (w sweep) + C (arity sweep)
#   incremental_vs_rebuild  F (n_mutations × num_buckets)
#   end_to_end_fpr          D (fingerprint_bits sweep)
#   failure_modes           single config (rejection-path microbench)
#   wire_sizes              A (m sweep) + B (w sweep) — wire-size catalogue
#   setup_to_first_query    G (mode × num_buckets)
#   steady_state_workload   A (m sweep, subset)
#
# Each bench CSV is `rm`'d before its sweep starts so rows from prior runs
# don't accumulate. Criterion's own report under target/criterion/ is left
# untouched.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CRATE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
WORKSPACE_DIR=$(cd "$CRATE_DIR/.." && pwd)

# shellcheck source=../../scripts/configs.sh
source "$WORKSPACE_DIR/scripts/configs.sh"

cd "$WORKSPACE_DIR"

RESULTS_DIR="$CRATE_DIR/results"
mkdir -p "$RESULTS_DIR"

CARGO=(cargo bench -p ikpir-server --bench)

# Color helpers (no-op if not a TTY).
if [[ -t 1 ]]; then
    BLUE='\033[34m'; GREEN='\033[32m'; YELLOW='\033[33m'; RESET='\033[0m'
else
    BLUE=''; GREEN=''; YELLOW=''; RESET=''
fi

step() { echo -e "${BLUE}── $* ──${RESET}"; }
note() { echo -e "${YELLOW}  $*${RESET}"; }
ok()   { echo -e "${GREEN}  $*${RESET}"; }

# ─────────────────────────────────────────────────────────────────────────────
# setup_latency — A (database-size sweep) + C (arity sweep)
# ─────────────────────────────────────────────────────────────────────────────
run_setup_latency() {
    step "setup_latency"
    rm -f "$RESULTS_DIR/ikpir_server_setup_latency.csv"

    # Matrix A: arity=2, vary num_buckets (m sweep)
    for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
        nb=${MATRIX_A_NB_ARITY2[$i]}
        m=${MATRIX_A_M_LABELS[$i]}
        note "A: arity=2 num_buckets=$nb (m≈$m) value_bits=$MATRIX_A_VALUE_BITS"
        "${CARGO[@]}" setup_latency -- \
            --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
            --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
            --lwe-dim "$MATRIX_A_LWE_DIM" --trials 5 --warmup 2
    done

    # Matrix C: arity sweep at fixed m≈2^16
    for i in "${!MATRIX_C_ARITY[@]}"; do
        ar=${MATRIX_C_ARITY[$i]}
        nb=${MATRIX_C_NUM_BUCKETS[$i]}
        note "C: arity=$ar num_buckets=$nb value_bits=$MATRIX_C_VALUE_BITS"
        "${CARGO[@]}" setup_latency -- \
            --arity "$ar" --num-buckets "$nb" \
            --bucket-size "$MATRIX_C_BUCKET_SIZE" --value-bits "$MATRIX_C_VALUE_BITS" \
            --lwe-dim "$MATRIX_C_LWE_DIM" --trials 5 --warmup 2
    done
    ok "→ $RESULTS_DIR/ikpir_server_setup_latency.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# answer_throughput — A (m sweep) + B (w sweep) + C (arity sweep)
# ─────────────────────────────────────────────────────────────────────────────
run_answer_throughput() {
    step "answer_throughput"
    rm -f "$RESULTS_DIR/ikpir_server_answer_throughput.csv"

    # Matrix A
    for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
        nb=${MATRIX_A_NB_ARITY2[$i]}; m=${MATRIX_A_M_LABELS[$i]}
        note "A: arity=2 num_buckets=$nb (m≈$m)"
        "${CARGO[@]}" answer_throughput -- \
            --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
            --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
            --lwe-dim "$MATRIX_A_LWE_DIM" --batch 64
    done

    # Matrix B (value-size sweep)
    for i in "${!MATRIX_B_VALUE_BITS[@]}"; do
        vb=${MATRIX_B_VALUE_BITS[$i]}; w=${MATRIX_B_W_LABELS[$i]}
        note "B: arity=2 num_buckets=$MATRIX_B_NB_ARITY2 value_bits=$vb (w=$w)"
        "${CARGO[@]}" answer_throughput -- \
            --arity "$MATRIX_B_ARITY" --num-buckets "$MATRIX_B_NB_ARITY2" \
            --bucket-size "$MATRIX_B_BUCKET_SIZE" --value-bits "$vb" \
            --lwe-dim "$MATRIX_B_LWE_DIM" --batch 64
    done

    # Matrix C (arity sweep)
    for i in "${!MATRIX_C_ARITY[@]}"; do
        ar=${MATRIX_C_ARITY[$i]}; nb=${MATRIX_C_NUM_BUCKETS[$i]}
        note "C: arity=$ar num_buckets=$nb"
        "${CARGO[@]}" answer_throughput -- \
            --arity "$ar" --num-buckets "$nb" \
            --bucket-size "$MATRIX_C_BUCKET_SIZE" --value-bits "$MATRIX_C_VALUE_BITS" \
            --lwe-dim "$MATRIX_C_LWE_DIM" --batch 64
    done
    ok "→ $RESULTS_DIR/ikpir_server_answer_throughput.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# incremental_vs_rebuild — F (n_mutations × num_buckets, all 3 mutation kinds)
# ─────────────────────────────────────────────────────────────────────────────
run_incremental_vs_rebuild() {
    step "incremental_vs_rebuild"
    rm -f "$RESULTS_DIR/ikpir_server_incremental_vs_rebuild.csv"

    for nb in "${MATRIX_F_NUM_BUCKETS[@]}"; do
        for n in "${MATRIX_F_N_MUTATIONS[@]}"; do
            note "F: num_buckets=$nb n_mutations=$n"
            "${CARGO[@]}" incremental_vs_rebuild -- \
                --arity "$MATRIX_F_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_F_BUCKET_SIZE" --value-bits "$MATRIX_F_VALUE_BITS" \
                --lwe-dim "$MATRIX_F_LWE_DIM" --n-mutations "$n" \
                --load-factor 0.50
        done
    done
    ok "→ $RESULTS_DIR/ikpir_server_incremental_vs_rebuild.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# end_to_end_fpr — D (fingerprint_bits sweep)
# ─────────────────────────────────────────────────────────────────────────────
run_end_to_end_fpr() {
    step "end_to_end_fpr"
    rm -f "$RESULTS_DIR/ikpir_end_to_end_fpr.csv"

    for fb in "${MATRIX_D_FINGERPRINT_BITS[@]}"; do
        note "D: fingerprint_bits=$fb (n_queried=$MATRIX_D_N_QUERIED)"
        "${CARGO[@]}" end_to_end_fpr -- \
            --arity "$MATRIX_D_ARITY" --num-buckets "$MATRIX_D_NUM_BUCKETS" \
            --bucket-size "$MATRIX_D_BUCKET_SIZE" --value-bits "$MATRIX_D_VALUE_BITS" \
            --fingerprint-bits "$fb" --lwe-dim "$MATRIX_D_LWE_DIM" \
            --n-queried "$MATRIX_D_N_QUERIED"
    done
    ok "→ $RESULTS_DIR/ikpir_end_to_end_fpr.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# failure_modes — single config (rejection-path microbench)
# ─────────────────────────────────────────────────────────────────────────────
run_failure_modes() {
    step "failure_modes"
    rm -f "$RESULTS_DIR/ikpir_failure_modes.csv"

    # full_num_buckets must satisfy each arity's constraint (power of 2 for
    # arity 2/4; 3·2^t for arity 3).
    for ar in 2 3 4; do
        case $ar in
            3) nb=24576; full_nb=12 ;;
            *) nb=16384; full_nb=16 ;;
        esac
        note "arity=$ar num_buckets=$nb full_num_buckets=$full_nb"
        "${CARGO[@]}" failure_modes -- \
            --arity "$ar" --num-buckets "$nb" \
            --bucket-size 4 --value-bits 256 \
            --num-trials 2000 --full-num-buckets "$full_nb"
    done
    ok "→ $RESULTS_DIR/ikpir_failure_modes.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# wire_sizes — A (m sweep) + B (w sweep)
# ─────────────────────────────────────────────────────────────────────────────
run_wire_sizes() {
    step "wire_sizes"
    rm -f "$RESULTS_DIR/ikpir_wire_sizes.csv"

    # Matrix A
    for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
        nb=${MATRIX_A_NB_ARITY2[$i]}; m=${MATRIX_A_M_LABELS[$i]}
        note "A: arity=2 num_buckets=$nb (m≈$m)"
        "${CARGO[@]}" wire_sizes -- \
            --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
            --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
            --lwe-dim "$MATRIX_A_LWE_DIM" --load-factor 0.50
    done

    # Matrix B (value-size sweep)
    for i in "${!MATRIX_B_VALUE_BITS[@]}"; do
        vb=${MATRIX_B_VALUE_BITS[$i]}; w=${MATRIX_B_W_LABELS[$i]}
        note "B: arity=2 num_buckets=$MATRIX_B_NB_ARITY2 value_bits=$vb (w=$w)"
        "${CARGO[@]}" wire_sizes -- \
            --arity "$MATRIX_B_ARITY" --num-buckets "$MATRIX_B_NB_ARITY2" \
            --bucket-size "$MATRIX_B_BUCKET_SIZE" --value-bits "$vb" \
            --lwe-dim "$MATRIX_B_LWE_DIM" --load-factor 0.50
    done
    ok "→ $RESULTS_DIR/ikpir_wire_sizes.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# setup_to_first_query — G (mode × num_buckets)
# ─────────────────────────────────────────────────────────────────────────────
run_setup_to_first_query() {
    step "setup_to_first_query"
    rm -f "$RESULTS_DIR/ikpir_setup_to_first_query.csv"

    for nb in "${MATRIX_G_NUM_BUCKETS[@]}"; do
        for mode in "${MATRIX_G_MODES[@]}"; do
            note "G: mode=$mode num_buckets=$nb"
            "${CARGO[@]}" setup_to_first_query -- \
                --arity "$MATRIX_G_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_G_BUCKET_SIZE" --value-bits "$MATRIX_G_VALUE_BITS" \
                --lwe-dim "$MATRIX_G_LWE_DIM" --mode "$mode" --trials 5 --warmup 2
        done
    done
    ok "→ $RESULTS_DIR/ikpir_setup_to_first_query.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# steady_state_workload — A subset (m sweep, smaller workload to keep runtime sane)
# ─────────────────────────────────────────────────────────────────────────────
run_steady_state_workload() {
    step "steady_state_workload"
    rm -f "$RESULTS_DIR/ikpir_steady_state_workload.csv"

    for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
        nb=${MATRIX_A_NB_ARITY2[$i]}; m=${MATRIX_A_M_LABELS[$i]}
        # Cap workload size at nb/2 inserts to stay below TableFull from a 50% load start.
        n_inserts=$(( nb / 2 ))
        if (( n_inserts > 4096 )); then n_inserts=4096; fi
        n_queries=$(( n_inserts / 10 ))
        if (( n_queries < 20 )); then n_queries=20; fi
        note "A: num_buckets=$nb (m≈$m) n_inserts=$n_inserts n_queries=$n_queries"
        "${CARGO[@]}" steady_state_workload -- \
            --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
            --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
            --lwe-dim "$MATRIX_A_LWE_DIM" \
            --n-inserts "$n_inserts" --n-queries "$n_queries" --query-ratio 10 \
            --trials 3 --warmup 1 --load-factor 0.50
    done
    ok "→ $RESULTS_DIR/ikpir_steady_state_workload.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

ALL_BENCHES=(
    setup_latency
    answer_throughput
    incremental_vs_rebuild
    end_to_end_fpr
    failure_modes
    wire_sizes
    setup_to_first_query
    steady_state_workload
)

run_one() {
    case "$1" in
        setup_latency)           run_setup_latency ;;
        answer_throughput)       run_answer_throughput ;;
        incremental_vs_rebuild)  run_incremental_vs_rebuild ;;
        end_to_end_fpr)          run_end_to_end_fpr ;;
        failure_modes)           run_failure_modes ;;
        wire_sizes)              run_wire_sizes ;;
        setup_to_first_query)    run_setup_to_first_query ;;
        steady_state_workload)   run_steady_state_workload ;;
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
