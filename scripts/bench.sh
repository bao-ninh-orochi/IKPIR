#!/usr/bin/env bash
# Run ONE IKPIR bench at ONE config, then stop. For the PIR benches this
# auto-derives --plaintext-bits (the max the backend's noise budget admits at
# q = 2^32) and --lwe-dim, and routes the CSV to results/<crate>/. This is the
# everyday entry point — no full sweep, no hours-long matrix.
#
# Usage:
#   ./scripts/bench.sh <bench-name> [flags...]
#
# PIR benches (one config per run; all flags optional, defaults shown):
#   server_setup  server_answer  server_mutation  headtohead_answer
#   client_query  client_decode  client_mutation  headtohead_query  headtohead_decode
#
#     --arity N            2 | 3 | 4                         (default 2)
#     --num-buckets N                                        (default: per-arity)
#     --bucket-size N                                        (default 4)
#     --value-bits N       256 | 2048 | 8192 = 32B/256B/1kB  (default 256)
#     --backend B          frodo | simple                    (default frodo)
#     --plaintext-bits N                    (default: max the backend admits)
#     --lwe-dim N                           (default: 1566 frodo / 1275 simple)
#   Mutation benches also take:
#     --n-mutations N      (default: 1% of capacity, capped at 2000)
#     --load-factor F      (default 0.90)
#     --patch-mode M       entry | row | entry,row           (default entry,row)
#   Head-to-head benches also take:
#     --num-keys N         (default: ~90% of capacity)
#   Any other flag (--batch, --fingerprint-bits, --estimate, --max-mem-gb,
#   --trials, --warmup, …) is forwarded to the bench unchanged.
#
# segmented-cuckoo benches (fixed internal config matrix — take no --arity/etc.):
#   load_factor  insert_throughput  lookup_throughput  delete_throughput  fpr
#   degree_distribution  kv_store_insert_throughput  kv_store_lookup_throughput
#   kv_store_delete_throughput
#
# Examples:
#   ./scripts/bench.sh server_answer --arity 4 --num-buckets 65536 --bucket-size 4 --value-bits 256
#   ./scripts/bench.sh client_decode --backend simple
#   ./scripts/bench.sh server_mutation --patch-mode entry
#   ./scripts/bench.sh headtohead_answer --arity 4 --num-buckets 262144 --num-keys 1000000
#   ./scripts/bench.sh insert_throughput

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; }

if [[ $# -eq 0 || "$1" == -h || "$1" == --help ]]; then usage; exit 0; fi

BENCH=$1; shift
if ! CRATE=$(crate_for_bench "$BENCH"); then
    { echo "unknown bench: $BENCH"; echo "valid benches:"; all_benches | sed 's/^/  /'; } >&2
    exit 1
fi

cd "$IKPIR_ROOT"
# Base defaults to <root>/results; smoke.sh points IKPIR_RESULTS_BASE at a
# scratch dir so its tiny rows never touch the real results/<crate>/ CSVs.
RESULTS_BASE="${IKPIR_RESULTS_BASE:-$IKPIR_ROOT/results}"
export IKPIR_RESULTS_DIR="$RESULTS_BASE/$CRATE"
mkdir -p "$IKPIR_RESULTS_DIR"
REL_DIR="${IKPIR_RESULTS_DIR#"$IKPIR_ROOT"/}"

# segmented-cuckoo benches run their own fixed matrix — just dispatch + route.
if [[ "$CRATE" == segmented-cuckoo ]]; then
    log "$BENCH  (segmented-cuckoo — fixed internal config matrix)"
    cargo bench -p segmented-cuckoo --bench "$BENCH" -- "$@"
    ok "CSV(s) under $REL_DIR/"
    exit 0
fi

# ── PIR bench: managed single-config surface ─────────────────────────────────
ARITY=""; NUM_BUCKETS=""; BUCKET_SIZE=""; VALUE_BITS=""; BACKEND=""
PB=""; LWE=""; NUM_KEYS=""; N_MUT=""; LOAD_FACTOR=""; PATCH_MODE=""
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arity)          ARITY=$2;       shift 2 ;;
        --num-buckets)    NUM_BUCKETS=$2; shift 2 ;;
        --bucket-size)    BUCKET_SIZE=$2; shift 2 ;;
        --value-bits)     VALUE_BITS=$2;  shift 2 ;;
        --backend)        BACKEND=$2;     shift 2 ;;
        --plaintext-bits) PB=$2;          shift 2 ;;
        --lwe-dim)        LWE=$2;         shift 2 ;;
        --num-keys)       NUM_KEYS=$2;    shift 2 ;;
        --n-mutations)    N_MUT=$2;       shift 2 ;;
        --load-factor)    LOAD_FACTOR=$2; shift 2 ;;
        --patch-mode)     PATCH_MODE=$2;  shift 2 ;;
        *)                EXTRA+=("$1");  shift ;;
    esac
done

ARITY=${ARITY:-2}
BUCKET_SIZE=${BUCKET_SIZE:-4}
VALUE_BITS=${VALUE_BITS:-256}
BACKEND=${BACKEND:-frodo}
validate_backend "$BACKEND"
NUM_BUCKETS=${NUM_BUCKETS:-$(default_num_buckets "$ARITY")}
[[ -n "$LWE" ]] || LWE=$(backend_lwe_dim "$BACKEND")
if [[ -z "$PB" ]]; then
    PB=$(backend_plaintext_bits "$BACKEND" "$ARITY" "$BUCKET_SIZE" "$NUM_BUCKETS" "$VALUE_BITS") || exit 1
fi

ARGS=(--arity "$ARITY" --num-buckets "$NUM_BUCKETS" --bucket-size "$BUCKET_SIZE"
      --value-bits "$VALUE_BITS" --backend "$BACKEND" --plaintext-bits "$PB" --lwe-dim "$LWE")

if is_mutation_bench "$BENCH"; then
    if [[ -z "$N_MUT" ]]; then
        N_MUT=$(( NUM_BUCKETS * BUCKET_SIZE / 100 ))
        (( N_MUT > 2000 )) && N_MUT=2000
        (( N_MUT < 1 ))    && N_MUT=1
    fi
    ARGS+=(--n-mutations "$N_MUT" --load-factor "${LOAD_FACTOR:-0.90}" --patch-mode "${PATCH_MODE:-entry,row}")
fi

if is_headtohead_bench "$BENCH"; then
    ARGS+=(--num-keys "${NUM_KEYS:-$(( NUM_BUCKETS * BUCKET_SIZE * 90 / 100 ))}")
fi

ARGS+=("${EXTRA[@]+"${EXTRA[@]}"}")

log "$BENCH  (crate=$CRATE backend=$BACKEND arity=$ARITY nb=$NUM_BUCKETS bs=$BUCKET_SIZE vb=$VALUE_BITS pb=$PB lwe=$LWE)"
cargo bench -p "$CRATE" --bench "$BENCH" -- "${ARGS[@]}"
ok "CSV(s) under $REL_DIR/"
