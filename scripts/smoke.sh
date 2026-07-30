#!/usr/bin/env bash
# Fast smoke test: run every PIR bench at a tiny config so the whole suite
# finishes in a couple of minutes and every correctness path (setup → answer →
# query → decode → mutation, on both backends) is exercised. `set -e` aborts on
# the first failure — including a decode/correctness failure inside a bench,
# which each decode bench guards with a once-per-config verify_decode.
#
# This is the "test the properties on small configs" entry point.
#
# Every leg runs at bench.sh's default fingerprint_bits = 64 (the paper's
# width), plus one extra tiny leg at --fingerprint-bits 32 (one bench, the
# first requested backend) so the narrow fingerprint path — still legal, just
# not a paper config — keeps smoke coverage too.
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

# Tiny geometry: capacity = 256 × 4 = 1024 slots — sub-second per bench. The
# value width is the paper's narrower one, so the exercised cells_per_slot is
# realistic; at 1024 slots that is still well under a megabyte.
TINY=(--arity 2 --num-buckets 256 --bucket-size 4 --value-bits 2048)

SMOKE_BENCHES=(server_setup server_answer server_mutation
               client_query client_decode client_mutation
               headtohead_answer headtohead_query headtohead_decode)

for be in "${BACKENDS[@]}"; do
    validate_backend "$be"
    log "smoke: backend=$be (${#SMOKE_BENCHES[@]} benches, tiny config, fingerprint_bits=64)"
    for b in "${SMOKE_BENCHES[@]}"; do
        "$SCRIPT_DIR/bench.sh" "$b" "${TINY[@]}" --backend "$be"
    done
done

# One extra leg at the narrow (32-bit) fingerprint width, on the first
# requested backend only — enough to catch a regression in the
# fingerprint-width-dependent path (pb derivation, row_width, cells_per_slot)
# without doubling smoke's runtime.
NARROW_BACKEND="${BACKENDS[0]}"
log "smoke: backend=$NARROW_BACKEND (1 bench, tiny config, fingerprint_bits=32 — narrow-path check)"
"$SCRIPT_DIR/bench.sh" server_answer "${TINY[@]}" --fingerprint-bits 32 --backend "$NARROW_BACKEND"

rm -rf "$IKPIR_RESULTS_BASE"
ok "smoke: all ${#SMOKE_BENCHES[@]} benches passed on: ${BACKENDS[*]} (+ 1 narrow-fingerprint check)"
