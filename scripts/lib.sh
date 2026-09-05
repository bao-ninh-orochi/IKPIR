#!/usr/bin/env bash
# Shared helpers for the IKPIR benchmark scripts (bench.sh, smoke.sh,
# table2.sh, table3.sh, table4.sh, table5.sh). Sourced, never executed directly.
#
# Targets bash 3.2 (the macOS system bash): no associative arrays, no namerefs.
# The paper registry below is therefore a `case` lookup keyed by "arity:bucket_size"
# rather than a map.

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
# with auto-derived plaintext-bits + lwe-dim). The segmented-cuckoo benches parse
# their own flags (see crates/segmented-cuckoo/benches/configs.rs) and default to
# the paper's Table 2 matrix, so bench.sh forwards their flags unchanged.
PIR_SERVER_BENCHES=(server_setup server_answer server_mutation headtohead_answer)
# Nine client benches: three flow-independent (query construction is
# identical on both flows) plus a client-hint-patch / client-rewind pair
# each for decode, mutation, and head-to-head decode.
PIR_CLIENT_BENCHES=(client_query client_hint_patch_decode client_rewind_decode
                     client_hint_patch_mutation client_rewind_mutation
                     client_rewind_staleness headtohead_query
                     headtohead_hint_patch_decode headtohead_rewind_decode)
# The four that populate Table 2, in the order the table's columns read.
CUCKOO_TABLE2_BENCHES=(cuckoo_filter_load_factor cuckoo_filter_insert_throughput
                       cuckoo_filter_lookup_throughput cuckoo_filter_delete_throughput)
# The benches behind each keyword-PIR table, split into the flow-independent
# common leg and each flow's own leg. Read by the sourcing table{3,4}.sh
# (via `table_benches_for_flow`) and table5.sh, never here — hence the
# disables.
# shellcheck disable=SC2034
PIR_TABLE3_COMMON=(headtohead_answer headtohead_query)
# shellcheck disable=SC2034
PIR_TABLE3_HINT_PATCH=(headtohead_hint_patch_decode)
# shellcheck disable=SC2034
PIR_TABLE3_REWIND=(headtohead_rewind_decode)
# shellcheck disable=SC2034
PIR_TABLE4_COMMON=(server_mutation)
# shellcheck disable=SC2034
PIR_TABLE4_HINT_PATCH=(client_hint_patch_mutation)
# shellcheck disable=SC2034
PIR_TABLE4_REWIND=(client_rewind_mutation)
# shellcheck disable=SC2034
PIR_TABLE5_BENCHES=(server_setup)
CUCKOO_BENCHES=("${CUCKOO_TABLE2_BENCHES[@]}" cuckoo_filter_false_positive_rate
                kv_store_insert_throughput kv_store_lookup_throughput
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
is_mutation_bench()   { [[ "$1" == server_mutation || "$1" == client_hint_patch_mutation || "$1" == client_rewind_mutation ]]; }
is_headtohead_bench() { [[ "$1" == headtohead_* ]]; }
# Benches that seed their store to a target fill (--load-factor): the mutation
# benches, server_setup, and client_rewind_staleness. The rest populate to
# TableFull (server_answer, client_query, client_{hint_patch,rewind}_decode)
# or to an exact key count (the headtohead_* quartet) and reject --load-factor.
takes_load_factor()   { is_mutation_bench "$1" || [[ "$1" == server_setup || "$1" == client_rewind_staleness ]]; }
# Benches that accept `--patch-mode`: server_mutation and the client-hint-patch
# mutation bench only. client_rewind_mutation has no --patch-mode flag — the
# client-rewind flow's maintenance is patch-mode-independent.
takes_patch_mode()    { [[ "$1" == server_mutation || "$1" == client_hint_patch_mutation ]]; }
all_benches()         { printf '%s\n' "${PIR_SERVER_BENCHES[@]}" "${PIR_CLIENT_BENCHES[@]}" "${CUCKOO_BENCHES[@]}"; }

# ── Backend parameters ────────────────────────────────────────────────────────
validate_backend() {
    case "$1" in frodo|simple) return 0 ;; *) die "unknown backend '$1' (valid: frodo, simple)" ;; esac
}

# ── Flow selection (client-hint-patch | client-rewind | all) ──────────────────
validate_flow() {
    case "$1" in
        client-hint-patch|client-rewind|all) return 0 ;;
        *) die "unknown flow '$1' (valid: client-hint-patch, client-rewind, all)" ;;
    esac
}

# Bench list for one keyword-PIR table (3 or 4) at the selected flow
# (client-hint-patch | client-rewind | all), one bench name per line so
# bash 3.2 callers can collect it with a `while read` loop. Always includes
# the table's flow-independent common leg (PIR_TABLE{3,4}_COMMON).
table_benches_for_flow() {
    local table=$1 flow=$2
    validate_flow "$flow"
    case "$table" in
        3)
            printf '%s\n' "${PIR_TABLE3_COMMON[@]}"
            case "$flow" in
                client-hint-patch) printf '%s\n' "${PIR_TABLE3_HINT_PATCH[@]}" ;;
                client-rewind)     printf '%s\n' "${PIR_TABLE3_REWIND[@]}" ;;
                all)               printf '%s\n' "${PIR_TABLE3_HINT_PATCH[@]}" "${PIR_TABLE3_REWIND[@]}" ;;
            esac
            ;;
        4)
            printf '%s\n' "${PIR_TABLE4_COMMON[@]}"
            case "$flow" in
                client-hint-patch) printf '%s\n' "${PIR_TABLE4_HINT_PATCH[@]}" ;;
                client-rewind)     printf '%s\n' "${PIR_TABLE4_REWIND[@]}" ;;
                all)               printf '%s\n' "${PIR_TABLE4_HINT_PATCH[@]}" "${PIR_TABLE4_REWIND[@]}" ;;
            esac
            ;;
        *) die "table_benches_for_flow: unknown table '$table' (expected 3 or 4)" ;;
    esac
}

# LWE dimension for 128-bit security, lattice estimator under the ADPS16 model.
backend_lwe_dim() {
    case "$1" in
        frodo)  echo 1566 ;;
        simple) echo 1275 ;;
        *)      die "backend_lwe_dim: unknown backend '$1'" ;;
    esac
}

# Default num_buckets per arity for a *one-off* `bench.sh` run: dev scale
# (~2^16 slots), sub-second per bench. Segmented 3-ary needs num_buckets = 3·2^t;
# 2-/4-ary need a power of two. Mirrors helpers::default_num_buckets_for_arity.
#
# The paper's geometry is much larger and lives in `paper_num_buckets` below —
# the table{3,4,5}.sh sweeps pass it explicitly. A bare `bench.sh` stays small
# on purpose: it is the everyday one-config runner, not a paper reproduction.
default_num_buckets() {
    case "$1" in
        2) echo 16384 ;;
        3) echo 24576 ;;
        4) echo 16384 ;;
        *) die "default_num_buckets: arity must be 2, 3, or 4 (got '$1')" ;;
    esac
}

# Max plaintext_bits admitted by the backend's per-cell correctness budget at
# q = 2^32, per (backend, SCF geometry, value_bits, fingerprint_bits). Single
# source of truth is the ikpir-common `max_plaintext_bits` example over
# ikpir_common::pir_params: both backends target the same delta_cell <=
# 2^-(kappa+1) / (arity * row_width) per-cell budget at kappa = 40 (FrodoPIR's
# explicit Bernstein tail; SimplePIR's Theorem C.1 bound retargeted from its
# old fixed delta = 2^-40). The result depends on the FULL geometry for both
# backends now, including fingerprint_bits, so callers pass it explicitly.
backend_plaintext_bits() {
    local backend=$1 arity=$2 bucket_size=$3 num_buckets=$4 value_bits=$5 fingerprint_bits=$6 pb
    pb=$(cargo run -q --release -p ikpir-common --example max_plaintext_bits \
             --manifest-path "$IKPIR_ROOT/Cargo.toml" -- \
             "$backend" --arity "$arity" --num-buckets "$num_buckets" \
             --bucket-size "$bucket_size" --value-bits "$value_bits" \
             --fingerprint-bits "$fingerprint_bits") \
        || die "max_plaintext_bits selector failed (backend=$backend arity=$arity bs=$bucket_size nb=$num_buckets vb=$value_bits fb=$fingerprint_bits)"
    echo "$pb"
}

# ── The paper's keyword-PIR config matrix ─────────────────────────────────────
#
# Single source of truth for the keyword-PIR tables, the counterpart of
# crates/segmented-cuckoo/benches/configs.rs for Table 2. table{3,4,5}.sh read
# it; bench.sh does not (a one-off run stays at dev scale, see
# `default_num_buckets`).
#
# The (arity, bucket_size) cells mirror `PAPER_CONFIGS` in configs.rs, so the
# filter table and the PIR tables describe the same five KV-SCF shapes.
PAPER_PIR_CONFIGS=(2:4 3:2 3:3 4:1 4:2)

# Table 5 (Setup) covers the arity-2/4 cells only: it is the static-rebuild
# baseline for the headline comparison, and the arity-3 cells are a full-paper
# addition that does not carry one. Read by table5.sh, never here.
# shellcheck disable=SC2034
PAPER_SETUP_CONFIGS=(2:4 4:1 4:2)

# Value widths ℓ the paper reports: 2048 bits = 256 B, 8192 bits = 1 kB.
# 32 B (256 bits) is deliberately absent — pass `--value-bits 256` to run it.
PAPER_VALUE_BITS=(2048 8192)

PAPER_BACKENDS=(frodo simple)

# Keyword count for the head-to-head against ChalametPIR / KPIR^index, which
# both report at this m. Arity-3 cells cannot land near it at a sane fill (see
# `paper_num_keys`), so they use their own fill instead.
PAPER_NUM_KEYS=1000000

# Fill for the mutation and setup tables, and for the arity-3 online cells.
PAPER_LOAD_FACTOR=0.90
# Same fill as an integer percent, for exact slot arithmetic (bash has no floats).
# Derived, so the fill is stated once.
PAPER_LOAD_PERCENT=$(awk -v f="$PAPER_LOAD_FACTOR" 'BEGIN{printf "%d", f*100}')

# Mutation batch size τ, as a percentage of the table's slots N. The paper
# applies one batch of τ = 1% of the slots and divides by the measured time.
PAPER_TAU_PERCENT=1

# Bucket count n_b each (arity, bucket_size) cell runs at.
#
# Every arity-2/4 cell is sized to N = n_b · b = 2^20 slots, so the three share
# one table size and differ only in shape. Arity 3 cannot reach 2^20: a
# segmented 3-ary table must have n_b = 3·2^t, so N = 3·2^t·b lands on a coarser
# ladder. Each arity-3 cell takes the smallest rung of that ladder still holding
# 10^6 keys at fill 0.90 — one rung down gives 707k slots for (3,2) and 531k for
# (3,3), both short of 10^6.
#
#   cell    n_b                    N = n_b · b            fill 0.90 → keys
#   ──────  ─────────────────────  ─────────────────────  ────────────────
#   (2, 4)  2^18   =   262 144     2^20    = 1 048 576       943 718
#   (3, 2)  3·2^18 =   786 432     3·2^19  = 1 572 864     1 415 577
#   (3, 3)  3·2^17 =   393 216     9·2^17  = 1 179 648     1 061 683
#   (4, 1)  2^20   = 1 048 576     2^20    = 1 048 576       943 718
#   (4, 2)  2^19   =   524 288     2^20    = 1 048 576       943 718
paper_num_buckets() {
    case "$1:$2" in
        2:4) echo 262144 ;;
        3:2) echo 786432 ;;
        3:3) echo 393216 ;;
        4:1) echo 1048576 ;;
        4:2) echo 524288 ;;
        *)   die "paper_num_buckets: no paper config (arity=$1, bucket_size=$2); matrix: ${PAPER_PIR_CONFIGS[*]}" ;;
    esac
}

# Keywords to populate for the online (Table 3) benches at a given cell.
#
# Arity 2/4 fix m = 10^6, the count the baselines publish, which fills their
# 2^20-slot tables to 0.954 — under every one of their achieved load factors
# (Table 2), so the comparison is at a real operating point.
#
# Arity 3 has no baseline to match and cannot hold 10^6 keys at a comparable
# fill (10^6 in (3,2)'s 1 572 864 slots is 0.636 — a sparse table that would
# flatter the scheme), so it reports at PAPER_LOAD_FACTOR instead. The CSV's
# num_keys / db_size columns record which regime each row came from.
paper_num_keys() {
    local arity=$1 bucket_size=$2 nb
    nb=$(paper_num_buckets "$arity" "$bucket_size")
    if [[ "$arity" == 3 ]]; then
        keys_at_paper_fill "$nb" "$bucket_size"
    else
        echo "$PAPER_NUM_KEYS"
    fi
}

# floor(num_buckets · bucket_size · PAPER_LOAD_FACTOR) — the key count that
# fills a table to the paper's fill.
keys_at_paper_fill() {
    echo $(( $1 * $2 * PAPER_LOAD_PERCENT / 100 ))
}

# Mutation batch size for a table of `num_buckets · bucket_size` slots.
tau_for_geometry() {
    local n=$(( $1 * $2 * PAPER_TAU_PERCENT / 100 ))
    (( n < 1 )) && n=1
    echo "$n"
}

# ── Paper sweep selectors ─────────────────────────────────────────────────────
#
# Shared flag handling for table{3,4,5}.sh: each narrows the matrix along one
# axis and forwards everything else to the bench. `$1` is the base config list as
# a space-separated string ("2:4 4:1 …") because bash 3.2 has no namerefs.
#
# Sets: PAPER_SEL_CONFIGS, PAPER_SEL_VALUE_BITS, PAPER_SEL_BACKENDS,
#       PAPER_SEL_FLOW, PAPER_EXTRA
paper_select() {
    local base=$1; shift
    local sel_arity="" sel_bs="" c a b be
    local all=()
    PAPER_SEL_CONFIGS=(); PAPER_SEL_VALUE_BITS=(); PAPER_SEL_BACKENDS=(); PAPER_EXTRA=()
    PAPER_SEL_FLOW="all"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --arity)       sel_arity=$2; shift 2 ;;
            --bucket-size) sel_bs=$2;    shift 2 ;;
            --value-bits)  IFS=',' read -ra PAPER_SEL_VALUE_BITS <<< "$2"; shift 2 ;;
            --backend)     IFS=',' read -ra PAPER_SEL_BACKENDS  <<< "$2"; shift 2 ;;
            --flow)        PAPER_SEL_FLOW=$2; shift 2 ;;
            *)             PAPER_EXTRA+=("$1"); shift ;;
        esac
    done

    IFS=' ' read -ra all <<< "$base"
    for c in "${all[@]}"; do
        a=${c%%:*}; b=${c##*:}
        if [[ -n "$sel_arity" && "$a" != "$sel_arity" ]]; then continue; fi
        if [[ -n "$sel_bs"    && "$b" != "$sel_bs"    ]]; then continue; fi
        PAPER_SEL_CONFIGS+=("$c")
    done
    if [[ ${#PAPER_SEL_CONFIGS[@]} -eq 0 ]]; then
        die "no config matches --arity ${sel_arity:-any} --bucket-size ${sel_bs:-any} (this table covers: $base)"
    fi

    [[ ${#PAPER_SEL_VALUE_BITS[@]} -gt 0 ]] || PAPER_SEL_VALUE_BITS=("${PAPER_VALUE_BITS[@]}")
    [[ ${#PAPER_SEL_BACKENDS[@]}   -gt 0 ]] || PAPER_SEL_BACKENDS=("${PAPER_BACKENDS[@]}")
    for be in "${PAPER_SEL_BACKENDS[@]}"; do validate_backend "$be"; done
    validate_flow "$PAPER_SEL_FLOW"
}

# Count of bench invocations a sweep is about to make, for its banner.
paper_run_count() {
    echo $(( ${#PAPER_SEL_CONFIGS[@]} * ${#PAPER_SEL_VALUE_BITS[@]} * ${#PAPER_SEL_BACKENDS[@]} * $1 ))
}
