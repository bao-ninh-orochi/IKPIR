#!/usr/bin/env bash
# Reproduce Table 2 of the CANS 2026 paper: maximum load factor and
# insert/lookup/delete throughput for the d-ary Segmented Cuckoo Filter against a
# standard cuckoo filter, across the six (arity, bucket_size) configurations
# RisePIR uses.
#
# This is the default entry point for the table. It runs the four benches that
# feed its columns, each of which sweeps all six configs on its own (the matrix
# and the tunables live in crates/segmented-cuckoo/benches/configs.rs, the single
# source of truth). With no arguments it reproduces the published table:
#
#   Load factor (%)     ← cuckoo_filter_load_factor
#   Throughput / Insert ← cuckoo_filter_insert_throughput
#   Throughput / Lookup ← cuckoo_filter_lookup_throughput
#   Throughput / Delete ← cuckoo_filter_delete_throughput
#
# Defaults per the paper: fingerprint_bits = 32, max_kicks = 2500, ~10^6 buckets
# (2^20 at arity 2/4; 3·2^19 segmented / 3^13 standard at arity 3).
#
# Any flags given are forwarded to all four benches, which is how you narrow the
# run — e.g. `--arity 4` for the three arity-4 rows only. The false-positive-rate
# and kv_store_* benches are not part of Table 2; run those via
# ./scripts/bench.sh <name>.
#
# Runtime: ~1-2 hours on the paper's machine (an Apple M1 MacBook Pro, 16 GiB).
# Every config fills a ~10^6-bucket table to capacity, repeatedly: the load-factor
# bench alone does 20 fills per scheme per config. Pass `--trials N` to trade
# error bars for wall time, or `--arity N` to run one arity at a time.
#
# Usage:
#   ./scripts/table2.sh                      # the full published table
#   ./scripts/table2.sh --arity 4            # just the arity-4 rows
#   ./scripts/table2.sh --trials 3           # a faster, noisier pass
#
# Output: results/segmented-cuckoo/cuckoo_filter_{load_factor,insert_throughput,
#         lookup_throughput,delete_throughput}.csv — one row per (scheme, config).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; }
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi

log "Table 2 — SCF vs standard cuckoo filter (${#CUCKOO_TABLE2_BENCHES[@]} benches)"
if [[ $# -gt 0 ]]; then
    warn "forwarding flags to every bench: $*"
fi

for b in "${CUCKOO_TABLE2_BENCHES[@]}"; do
    "$SCRIPT_DIR/bench.sh" "$b" "$@"
done

ok "Table 2 complete — CSVs under results/segmented-cuckoo/"
printf '  %s\n' "${CUCKOO_TABLE2_BENCHES[@]/%/.csv}"
