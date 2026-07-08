#!/usr/bin/env bash
# Fast smoke test: run every PIR bench at a tiny config so the whole suite
# finishes in a couple of minutes and every correctness path (setup → answer →
# query → decode → mutation, on both backends) is exercised. `set -e` aborts on
# the first failure — including a decode/correctness failure inside a bench,
# which each decode bench guards with a once-per-config verify_decode.
#
# This is the "test the properties on small configs" entry point.
#
# The segmented-cuckoo filter benches run a fixed large internal matrix (minutes
# each) and are NOT part of this quick smoke — run them one at a time via
# `./scripts/bench.sh <name>`, or exercise the filter / kv-store properties fast
# with `cargo test -p segmented-cuckoo`.
#
# Usage:
#   ./scripts/smoke.sh                              # frodo + simple, tiny config
#   IKPIR_SMOKE_BACKENDS=frodo ./scripts/smoke.sh   # a single backend

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

# Write to a throwaway base so smoke's tiny rows never touch the real
# results/<crate>/ CSVs (bench.sh appends). Cleared on each run.
export IKPIR_RESULTS_BASE="$IKPIR_ROOT/results/.smoke"
rm -rf "$IKPIR_RESULTS_BASE"

IFS=',' read -ra BACKENDS <<< "${IKPIR_SMOKE_BACKENDS:-frodo,simple}"

# Tiny geometry: capacity = 256 × 4 = 1024 slots — sub-second per bench.
TINY=(--arity 2 --num-buckets 256 --bucket-size 4 --value-bits 256)

SMOKE_BENCHES=(server_setup server_answer server_mutation
               client_query client_decode client_mutation
               headtohead_answer headtohead_query headtohead_decode)

for be in "${BACKENDS[@]}"; do
    validate_backend "$be"
    log "smoke: backend=$be (${#SMOKE_BENCHES[@]} benches, tiny config)"
    for b in "${SMOKE_BENCHES[@]}"; do
        "$SCRIPT_DIR/bench.sh" "$b" "${TINY[@]}" --backend "$be"
    done
done
rm -rf "$IKPIR_RESULTS_BASE"
ok "smoke: all ${#SMOKE_BENCHES[@]} benches passed on: ${BACKENDS[*]}"
