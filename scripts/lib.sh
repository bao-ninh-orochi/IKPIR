#!/usr/bin/env bash
# Shared helpers for the IKPIR benchmark scripts (bench.sh, smoke.sh).
# Sourced, never executed directly.

# Workspace root = parent of the dir holding this file (scripts/).
IKPIR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Logging ───────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_BLUE=$'\033[34m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RESET=$'\033[0m'
else
    C_BLUE=''; C_GREEN=''; C_YELLOW=''; C_RESET=''
fi
log()  { echo "${C_BLUE}▶ $*${C_RESET}"; }
ok()   { echo "${C_GREEN}✓ $*${C_RESET}"; }
warn() { echo "${C_YELLOW}! $*${C_RESET}" >&2; }
die()  { echo "${C_YELLOW}error:${C_RESET} $*" >&2; exit 1; }

# ── Bench registry ────────────────────────────────────────────────────────────
# PIR benches take the managed single-config surface (arity/num-buckets/…/backend
# with auto-derived plaintext-bits + lwe-dim); the segmented-cuckoo filter benches
# run their own fixed internal config matrix and take no such flags.
PIR_SERVER_BENCHES=(server_setup server_answer server_mutation headtohead_answer)
PIR_CLIENT_BENCHES=(client_query client_decode client_mutation headtohead_query headtohead_decode)
CUCKOO_BENCHES=(load_factor insert_throughput lookup_throughput delete_throughput fpr
                degree_distribution kv_store_insert_throughput kv_store_lookup_throughput
                kv_store_delete_throughput)

_contains() { local x=$1; shift; local e; for e in "$@"; do [[ "$e" == "$x" ]] && return 0; done; return 1; }

# Print the owning crate for a bench name, or fail if unknown.
crate_for_bench() {
    if _contains "$1" "${PIR_SERVER_BENCHES[@]}"; then echo ikpir-server; return 0; fi
    if _contains "$1" "${PIR_CLIENT_BENCHES[@]}"; then echo ikpir-client; return 0; fi
    if _contains "$1" "${CUCKOO_BENCHES[@]}";     then echo segmented-cuckoo; return 0; fi
    return 1
}
is_pir_bench()        { _contains "$1" "${PIR_SERVER_BENCHES[@]}" "${PIR_CLIENT_BENCHES[@]}"; }
is_mutation_bench()   { [[ "$1" == server_mutation || "$1" == client_mutation ]]; }
is_headtohead_bench() { [[ "$1" == headtohead_* ]]; }
all_benches()         { printf '%s\n' "${PIR_SERVER_BENCHES[@]}" "${PIR_CLIENT_BENCHES[@]}" "${CUCKOO_BENCHES[@]}"; }

# ── Backend parameters ────────────────────────────────────────────────────────
validate_backend() {
    case "$1" in frodo|simple) return 0 ;; *) die "unknown backend '$1' (valid: frodo, simple)" ;; esac
}

# LWE dimension for 128-bit security, lattice estimator under the ADPS16 model.
backend_lwe_dim() {
    case "$1" in
        frodo)  echo 1566 ;;
        simple) echo 1275 ;;
        *)      die "backend_lwe_dim: unknown backend '$1'" ;;
    esac
}

# Default num_buckets per arity. Segmented 3-ary needs num_buckets = 3·2^t;
# 2-/4-ary need a power of two. Mirrors helpers::default_num_buckets_for_arity.
default_num_buckets() {
    case "$1" in
        2) echo 16384 ;;
        3) echo 24576 ;;
        4) echo 16384 ;;
        *) die "default_num_buckets: arity must be 2, 3, or 4 (got '$1')" ;;
    esac
}

# Max plaintext_bits admitted by the backend's correctness bound at q = 2^32,
# per (backend, SCF geometry, value_bits). Single source of truth is the
# ikpir-common `max_plaintext_bits` example over ikpir_common::pir_params
# (FrodoPIR paper Eq. 8; SimplePIR paper Theorem C.1). The result depends on
# value_bits for SimplePIR, so callers pass the full geometry.
backend_plaintext_bits() {
    local backend=$1 arity=$2 bucket_size=$3 num_buckets=$4 value_bits=$5 pb
    pb=$(cargo run -q --release -p ikpir-common --example max_plaintext_bits \
             --manifest-path "$IKPIR_ROOT/Cargo.toml" -- \
             "$backend" --arity "$arity" --num-buckets "$num_buckets" \
             --bucket-size "$bucket_size" --value-bits "$value_bits") \
        || die "max_plaintext_bits selector failed (backend=$backend arity=$arity bs=$bucket_size nb=$num_buckets vb=$value_bits)"
    echo "$pb"
}
