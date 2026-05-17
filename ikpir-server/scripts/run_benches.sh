#!/usr/bin/env bash
# Run every ikpir-server bench across the paper-derived config matrices,
# once per backend selected by IKPIR_BENCH_BACKENDS.
#
# Usage:
#   ./scripts/run_benches.sh                    # all benches, default profile
#   ./scripts/run_benches.sh setup_latency      # one bench
#   IKPIR_BENCH_PROFILE=quick ./scripts/run_benches.sh   # smaller matrix
#   IKPIR_BENCH_PROFILE=full  ./scripts/run_benches.sh   # add 2^22 keys row
#   IKPIR_BENCH_BACKENDS=frodo                          # default: FrodoPIR only
#   IKPIR_BENCH_BACKENDS=simple                         # SimplePIR only
#   IKPIR_BENCH_BACKENDS=frodo,simple                   # both backends (~2× runtime)
#
# Per-bench config matrix:
#   setup_latency           A (m sweep) + C (arity sweep)
#   answer_throughput       A (m sweep) + B (w sweep) + C (arity sweep)
#   incremental_vs_rebuild  F (n_mutations × num_buckets)
#   failure_modes           single config (rejection-path microbench)
#   wire_sizes              A (m sweep) + B (w sweep) — wire-size catalogue
#   setup_to_first_query    G (mode × num_buckets)
#   steady_state_workload   A (m sweep, subset)
#
# Backend dimension:
#   For each backend in BACKENDS_ARR, every bench is re-invoked with
#   --backend $backend --lwe-dim $(backend_lwe_dim $backend). FrodoPIR
#   defaults to 1774, SimplePIR to 1024 — both paper-recommended for
#   128-bit security at q=2^32.
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
# setup_latency — A (database-size sweep) + C (arity sweep) × {backend}
# ─────────────────────────────────────────────────────────────────────────────
run_setup_latency() {
    step "setup_latency"
    rm -f "$RESULTS_DIR/ikpir_server_setup_latency.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        lwe=$(backend_lwe_dim "$backend")
        for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
            nb=${MATRIX_A_NB_ARITY2[$i]}
            m=${MATRIX_A_M_LABELS[$i]}
            note "A: arity=2 num_buckets=$nb (m≈$m) value_bits=$MATRIX_A_VALUE_BITS lwe_dim=$lwe"
            "${CARGO[@]}" setup_latency -- \
                --backend "$backend" \
                --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                --lwe-dim "$lwe" --trials 5 --warmup 2
        done

        for i in "${!MATRIX_C_ARITY[@]}"; do
            ar=${MATRIX_C_ARITY[$i]}
            nb=${MATRIX_C_NUM_BUCKETS[$i]}
            note "C: arity=$ar num_buckets=$nb value_bits=$MATRIX_C_VALUE_BITS lwe_dim=$lwe"
            "${CARGO[@]}" setup_latency -- \
                --backend "$backend" \
                --arity "$ar" --num-buckets "$nb" \
                --bucket-size "$MATRIX_C_BUCKET_SIZE" --value-bits "$MATRIX_C_VALUE_BITS" \
                --lwe-dim "$lwe" --trials 5 --warmup 2
        done
    done
    ok "→ $RESULTS_DIR/ikpir_server_setup_latency.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# answer_throughput — A (m sweep) + B (w sweep) + C (arity sweep) × {backend}
# ─────────────────────────────────────────────────────────────────────────────
run_answer_throughput() {
    step "answer_throughput"
    rm -f "$RESULTS_DIR/ikpir_server_answer_throughput.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        lwe=$(backend_lwe_dim "$backend")

        for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
            nb=${MATRIX_A_NB_ARITY2[$i]}; m=${MATRIX_A_M_LABELS[$i]}
            note "A: arity=2 num_buckets=$nb (m≈$m) lwe_dim=$lwe"
            "${CARGO[@]}" answer_throughput -- \
                --backend "$backend" \
                --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                --lwe-dim "$lwe" --batch 64
        done

        for i in "${!MATRIX_B_VALUE_BITS[@]}"; do
            vb=${MATRIX_B_VALUE_BITS[$i]}; w=${MATRIX_B_W_LABELS[$i]}
            note "B: arity=2 num_buckets=$MATRIX_B_NB_ARITY2 value_bits=$vb (w=$w) lwe_dim=$lwe"
            "${CARGO[@]}" answer_throughput -- \
                --backend "$backend" \
                --arity "$MATRIX_B_ARITY" --num-buckets "$MATRIX_B_NB_ARITY2" \
                --bucket-size "$MATRIX_B_BUCKET_SIZE" --value-bits "$vb" \
                --lwe-dim "$lwe" --batch 64
        done

        for i in "${!MATRIX_C_ARITY[@]}"; do
            ar=${MATRIX_C_ARITY[$i]}; nb=${MATRIX_C_NUM_BUCKETS[$i]}
            note "C: arity=$ar num_buckets=$nb lwe_dim=$lwe"
            "${CARGO[@]}" answer_throughput -- \
                --backend "$backend" \
                --arity "$ar" --num-buckets "$nb" \
                --bucket-size "$MATRIX_C_BUCKET_SIZE" --value-bits "$MATRIX_C_VALUE_BITS" \
                --lwe-dim "$lwe" --batch 64
        done
    done
    ok "→ $RESULTS_DIR/ikpir_server_answer_throughput.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# incremental_vs_rebuild — F (n_mutations × num_buckets, all 3 mutation kinds)
# ─────────────────────────────────────────────────────────────────────────────
run_incremental_vs_rebuild() {
    step "incremental_vs_rebuild"
    rm -f "$RESULTS_DIR/ikpir_server_incremental_vs_rebuild.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        lwe=$(backend_lwe_dim "$backend")

        for nb in "${MATRIX_F_NUM_BUCKETS[@]}"; do
            for n in "${MATRIX_F_N_MUTATIONS[@]}"; do
                note "F: num_buckets=$nb n_mutations=$n lwe_dim=$lwe"
                "${CARGO[@]}" incremental_vs_rebuild -- \
                    --backend "$backend" \
                    --arity "$MATRIX_F_ARITY" --num-buckets "$nb" \
                    --bucket-size "$MATRIX_F_BUCKET_SIZE" --value-bits "$MATRIX_F_VALUE_BITS" \
                    --lwe-dim "$lwe" --n-mutations "$n" \
                    --load-factor 0.50
            done
        done
    done
    ok "→ $RESULTS_DIR/ikpir_server_incremental_vs_rebuild.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# failure_modes — single config (rejection-path microbench) × {backend}
# ─────────────────────────────────────────────────────────────────────────────
run_failure_modes() {
    step "failure_modes"
    rm -f "$RESULTS_DIR/ikpir_failure_modes.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        lwe=$(backend_lwe_dim "$backend")
        for ar in 2 3 4; do
            case $ar in
                3) nb=24576; full_nb=12 ;;
                *) nb=16384; full_nb=16 ;;
            esac
            note "arity=$ar num_buckets=$nb full_num_buckets=$full_nb lwe_dim=$lwe"
            "${CARGO[@]}" failure_modes -- \
                --backend "$backend" \
                --arity "$ar" --num-buckets "$nb" \
                --bucket-size 4 --value-bits 256 \
                --lwe-dim "$lwe" \
                --num-trials 2000 --full-num-buckets "$full_nb"
        done
    done
    ok "→ $RESULTS_DIR/ikpir_failure_modes.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# wire_sizes — A (m sweep) + B (w sweep) × {backend}
# ─────────────────────────────────────────────────────────────────────────────
run_wire_sizes() {
    step "wire_sizes"
    rm -f "$RESULTS_DIR/ikpir_wire_sizes.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        lwe=$(backend_lwe_dim "$backend")

        for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
            nb=${MATRIX_A_NB_ARITY2[$i]}; m=${MATRIX_A_M_LABELS[$i]}
            note "A: arity=2 num_buckets=$nb (m≈$m) lwe_dim=$lwe"
            "${CARGO[@]}" wire_sizes -- \
                --backend "$backend" \
                --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                --lwe-dim "$lwe" --load-factor 0.50
        done

        for i in "${!MATRIX_B_VALUE_BITS[@]}"; do
            vb=${MATRIX_B_VALUE_BITS[$i]}; w=${MATRIX_B_W_LABELS[$i]}
            note "B: arity=2 num_buckets=$MATRIX_B_NB_ARITY2 value_bits=$vb (w=$w) lwe_dim=$lwe"
            "${CARGO[@]}" wire_sizes -- \
                --backend "$backend" \
                --arity "$MATRIX_B_ARITY" --num-buckets "$MATRIX_B_NB_ARITY2" \
                --bucket-size "$MATRIX_B_BUCKET_SIZE" --value-bits "$vb" \
                --lwe-dim "$lwe" --load-factor 0.50
        done
    done
    ok "→ $RESULTS_DIR/ikpir_wire_sizes.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# setup_to_first_query — G (mode × num_buckets) × {backend}
# ─────────────────────────────────────────────────────────────────────────────
run_setup_to_first_query() {
    step "setup_to_first_query"
    rm -f "$RESULTS_DIR/ikpir_setup_to_first_query.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        lwe=$(backend_lwe_dim "$backend")
        for nb in "${MATRIX_G_NUM_BUCKETS[@]}"; do
            for mode in "${MATRIX_G_MODES[@]}"; do
                note "G: mode=$mode num_buckets=$nb lwe_dim=$lwe"
                "${CARGO[@]}" setup_to_first_query -- \
                    --backend "$backend" \
                    --arity "$MATRIX_G_ARITY" --num-buckets "$nb" \
                    --bucket-size "$MATRIX_G_BUCKET_SIZE" --value-bits "$MATRIX_G_VALUE_BITS" \
                    --lwe-dim "$lwe" --mode "$mode" --trials 5 --warmup 2
            done
        done
    done
    ok "→ $RESULTS_DIR/ikpir_setup_to_first_query.csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# steady_state_workload — A subset (m sweep) × {backend}
# ─────────────────────────────────────────────────────────────────────────────
run_steady_state_workload() {
    step "steady_state_workload"
    rm -f "$RESULTS_DIR/ikpir_steady_state_workload.csv"

    for backend in "${BACKENDS_ARR[@]}"; do
        backend_note "$backend"
        lwe=$(backend_lwe_dim "$backend")
        for i in "${!MATRIX_A_NB_ARITY2[@]}"; do
            nb=${MATRIX_A_NB_ARITY2[$i]}; m=${MATRIX_A_M_LABELS[$i]}
            n_inserts=$(( nb / 2 ))
            if (( n_inserts > 4096 )); then n_inserts=4096; fi
            n_queries=$(( n_inserts / 10 ))
            if (( n_queries < 20 )); then n_queries=20; fi
            note "A: num_buckets=$nb (m≈$m) n_inserts=$n_inserts n_queries=$n_queries lwe_dim=$lwe"
            "${CARGO[@]}" steady_state_workload -- \
                --backend "$backend" \
                --arity "$MATRIX_A_ARITY" --num-buckets "$nb" \
                --bucket-size "$MATRIX_A_BUCKET_SIZE" --value-bits "$MATRIX_A_VALUE_BITS" \
                --lwe-dim "$lwe" \
                --n-inserts "$n_inserts" --n-queries "$n_queries" --query-ratio 10 \
                --trials 3 --warmup 1 --load-factor 0.50
        done
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
