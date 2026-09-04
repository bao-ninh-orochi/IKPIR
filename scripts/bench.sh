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
#   client_query  client_decode  client_mutation  client_rewind_staleness
#   headtohead_query  headtohead_decode
#
#     --arity N            2 | 3 | 4                         (default 2)
#     --num-buckets N                             (default: per-arity, dev scale)
#     --bucket-size N                                        (default 4)
#     --value-bits N       2048 | 8192 = 256B / 1kB          (default 2048)
#                          The paper reports these two widths. 256 (= 32B) still
#                          runs if you pass it; it is just not a paper config.
#     --backend B          frodo | simple                    (default frodo)
#     --fingerprint-bits N (default 64; the paper's width)
#     --plaintext-bits N                    (default: max the backend admits)
#     --lwe-dim N                           (default: 1566 frodo / 1275 simple)
#   server_setup, the mutation benches, and client_rewind_staleness also take:
#     --load-factor F      (default 0.90)
#   client_rewind_staleness also takes:
#     --batch-size N  --staleness-steps N  --queries N   (staleness sweep knobs)
#   server_setup also takes:
#     --setup-impl I       reference | parallel              (default reference)
#                          `reference` is the single-threaded, non-SIMD path the
#                          paper reports, and the ONLY thing server_setup should
#                          normally time. `parallel` times the byte-identical
#                          multi-threaded twin that every OTHER bench already
#                          uses in its untimed preamble — diagnostic, never a
#                          paper number. The CSV's setup_mode column records it
#                          (full vs full_parallel).
#                          IKPIR_SETUP_THREADS sets the worker count everywhere
#                          (default: the machine's available parallelism, and
#                          clamped to MAX_SETUP_THREADS = 1024); set it to 1 to
#                          force the reference schedule.
#   Mutation benches also take:
#     --n-mutations N      (default: 1% of the table's slots — the paper's τ)
#     --patch-mode M       entry | row | entry,row           (default entry,row)
#   Head-to-head benches also take:
#     --num-keys N         (default: ~90% of capacity)
#   Any other flag (--batch, --trials, --warmup, …) is forwarded to the bench
#   unchanged.
#
# Geometry defaults here are DEV SCALE (~2^16 slots), not the paper's. This is
# the everyday one-config runner; the paper's matrix lives in scripts/lib.sh and
# is driven by table{2,3,4,5}.sh.
#
# segmented-cuckoo benches (all flags optional; with none, each runs the paper's
# Table 2 matrix of five (arity, bucket_size) configs — see
# crates/segmented-cuckoo/benches/configs.rs). Flags are forwarded unchanged:
#   cuckoo_filter_load_factor         cuckoo_filter_insert_throughput
#   cuckoo_filter_lookup_throughput   cuckoo_filter_delete_throughput
#   cuckoo_filter_false_positive_rate
#   kv_store_insert_throughput  kv_store_lookup_throughput  kv_store_delete_throughput
#
#     --arity N            2 | 3 | 4          (default: every arity in the matrix)
#     --bucket-size N      1..4               (default: every size in the matrix)
#     --num-buckets N                         (default: per-arity, per Table 2)
#     --fingerprint-bits N                    (default 64)
#     --max-kicks N                           (default 2500)
#     --warmup N / --trials N   (default 3 / 10; load_factor defaults to 20 trials)
#   Filter-bench extras:  --hit-rate (lookup), --num-queries (false_positive_rate)
#   KV-store extras:      --value-bits, --plaintext-bits, --target-items
#
# To reproduce a paper table end to end, use the sweep that owns it:
#   ./scripts/table2.sh   filter: SCF vs standard cuckoo filter
#   ./scripts/table3.sh   online: query / response / answer
#   ./scripts/table4.sh   mutation throughput
#   ./scripts/table5.sh   setup (static rebuild cost)
#
# Examples:
#   ./scripts/bench.sh server_answer --arity 4 --num-buckets 65536 --bucket-size 4 --value-bits 8192
#   ./scripts/bench.sh client_decode --backend simple
#   ./scripts/bench.sh server_mutation --patch-mode entry
#   ./scripts/bench.sh headtohead_answer --arity 4 --num-buckets 262144 --num-keys 1000000
#   ./scripts/bench.sh cuckoo_filter_insert_throughput                    # full Table 2 matrix
#   ./scripts/bench.sh cuckoo_filter_insert_throughput --arity 4 --bucket-size 2   # one cell

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

# segmented-cuckoo benches parse their own flags and default to the paper's
# Table 2 matrix — just dispatch + route the CSV.
if [[ "$CRATE" == segmented-cuckoo ]]; then
    if [[ $# -eq 0 ]]; then
        log "$BENCH  (segmented-cuckoo — paper Table 2 matrix)"
    else
        log "$BENCH  (segmented-cuckoo — $*)"
    fi
    cargo bench -p segmented-cuckoo --bench "$BENCH" -- "$@"
    ok "CSV(s) under $REL_DIR/"
    exit 0
fi

# ── PIR bench: managed single-config surface ─────────────────────────────────
ARITY=""; NUM_BUCKETS=""; BUCKET_SIZE=""; VALUE_BITS=""; BACKEND=""; FINGERPRINT_BITS=""
PB=""; LWE=""; NUM_KEYS=""; N_MUT=""; LOAD_FACTOR=""; PATCH_MODE=""
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arity)             ARITY=$2;            shift 2 ;;
        --num-buckets)       NUM_BUCKETS=$2;       shift 2 ;;
        --bucket-size)       BUCKET_SIZE=$2;       shift 2 ;;
        --value-bits)        VALUE_BITS=$2;        shift 2 ;;
        --backend)           BACKEND=$2;           shift 2 ;;
        --fingerprint-bits)  FINGERPRINT_BITS=$2;  shift 2 ;;
        --plaintext-bits)    PB=$2;                shift 2 ;;
        --lwe-dim)           LWE=$2;               shift 2 ;;
        --num-keys)          NUM_KEYS=$2;          shift 2 ;;
        --n-mutations)       N_MUT=$2;             shift 2 ;;
        --load-factor)       LOAD_FACTOR=$2;       shift 2 ;;
        --patch-mode)        PATCH_MODE=$2;        shift 2 ;;
        *)                   EXTRA+=("$1");        shift ;;
    esac
done

ARITY=${ARITY:-2}
BUCKET_SIZE=${BUCKET_SIZE:-4}
VALUE_BITS=${VALUE_BITS:-2048}
BACKEND=${BACKEND:-frodo}
validate_backend "$BACKEND"
# 64 = the paper's fingerprint width (see crate CLAUDE.md / README notation
# tables). Both backends' plaintext-bits selector needs this explicitly now
# (see backend_plaintext_bits below) — it is no longer frodo-ignorable.
FINGERPRINT_BITS=${FINGERPRINT_BITS:-64}
NUM_BUCKETS=${NUM_BUCKETS:-$(default_num_buckets "$ARITY")}
[[ -n "$LWE" ]] || LWE=$(backend_lwe_dim "$BACKEND")
if [[ -z "$PB" ]]; then
    PB=$(backend_plaintext_bits "$BACKEND" "$ARITY" "$BUCKET_SIZE" "$NUM_BUCKETS" \
             "$VALUE_BITS" "$FINGERPRINT_BITS") || exit 1
fi

ARGS=(--arity "$ARITY" --num-buckets "$NUM_BUCKETS" --bucket-size "$BUCKET_SIZE"
      --value-bits "$VALUE_BITS" --backend "$BACKEND" --fingerprint-bits "$FINGERPRINT_BITS"
      --plaintext-bits "$PB" --lwe-dim "$LWE")

# A flag that does not apply to this bench is parsed above but never forwarded.
# Say so: silently dropping it would run a config the caller did not ask for.
if [[ -n "$LOAD_FACTOR" ]] && ! takes_load_factor "$BENCH"; then
    warn "$BENCH takes no --load-factor (it populates to TableFull, or to --num-keys); ignoring it"
fi
if [[ -n "$N_MUT$PATCH_MODE" ]] && ! is_mutation_bench "$BENCH"; then
    warn "$BENCH is not a mutation bench; ignoring --n-mutations / --patch-mode"
fi
if [[ -n "$NUM_KEYS" ]] && ! is_headtohead_bench "$BENCH"; then
    warn "$BENCH takes no --num-keys (only the headtohead_* benches fix a key count); ignoring it"
fi

if is_mutation_bench "$BENCH"; then
    # τ = PAPER_TAU_PERCENT of the table's slots, the paper's batch rule. There
    # is no upper clamp: the rule is what the paper's method sentence states, and
    # a clamp would only ever bind at paper scale — exactly where it must not.
    ARGS+=(--n-mutations "${N_MUT:-$(tau_for_geometry "$NUM_BUCKETS" "$BUCKET_SIZE")}"
           --patch-mode "${PATCH_MODE:-entry,row}")
fi

if takes_load_factor "$BENCH"; then
    ARGS+=(--load-factor "${LOAD_FACTOR:-$PAPER_LOAD_FACTOR}")
fi

if is_headtohead_bench "$BENCH"; then
    ARGS+=(--num-keys "${NUM_KEYS:-$(keys_at_paper_fill "$NUM_BUCKETS" "$BUCKET_SIZE")}")
fi

ARGS+=("${EXTRA[@]+"${EXTRA[@]}"}")

log "$BENCH  (crate=$CRATE backend=$BACKEND arity=$ARITY nb=$NUM_BUCKETS bs=$BUCKET_SIZE vb=$VALUE_BITS fb=$FINGERPRINT_BITS pb=$PB lwe=$LWE)"
# client_mutation's --update-mode patch sweep needs the bench-only
# HintPatchClient comparator, which is gated behind hint-patch-bench (a
# production build never links it in) — no other bench needs this feature.
if [[ "$BENCH" == client_mutation ]]; then
    cargo bench -p "$CRATE" --bench "$BENCH" --features hint-patch-bench -- "${ARGS[@]}"
else
    cargo bench -p "$CRATE" --bench "$BENCH" -- "${ARGS[@]}"
fi
ok "CSV(s) under $REL_DIR/"
