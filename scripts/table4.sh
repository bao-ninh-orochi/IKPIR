#!/usr/bin/env bash
# Reproduce the paper's mutation table: insert / update / delete throughput of
# RisePIR-F and RisePIR-S, the incremental half the static baselines cannot do
# at all. table5.sh gives the rebuild cost these throughputs replace.
#
# Runs the flow-independent server leg plus the selected flow's client leg,
# over the paper matrix in scripts/lib.sh, the single source of truth:
#
#   server_mutation                DBMutation throughput + the v1 delta
#                                   transcript (bytes, rows, runs, cells,
#                                   nonzero cells) next to the fresh-hint
#                                   download it competes with
#                                   (setup_bundle_bytes, hint_bytes_total)
#   client_hint_patch_mutation      HintUpdate throughput, client-hint-patch —
#   client_rewind_mutation          the camera-ready's number — or client-rewind
#                                   — the full paper's number; one bench per
#                                   flow, never merged, selected by --flow.
#
# Five (arity, bucket_size) cells × 2 value widths × 2 backends = 20 configs,
# × (1 server bench + 1 or 2 client benches, depending on --flow). Every
# cell's n_b comes from `paper_num_buckets`; the arity-2/4 cells share
# N = 2^20 slots, and arity 3 lands on its own ladder (1 572 864 for (3,2),
# 1 179 648 for (3,3)) — see lib.sh for why it cannot reach 2^20. Every run is
# at fingerprint_bits = 64, the paper's width (bench.sh's default).
#
# Each bench seeds to fill 0.90, applies one batch of τ = 1% of the table's
# slots per (kind[, patch mode]) pair, and divides the batch by the measured
# time. bench.sh derives τ from the geometry (10 485 at N = 2^20) and — for
# server_mutation and client_hint_patch_mutation — sweeps both patch modes, so
# this script passes geometry only — the batch rule lives in one place. Each
# bench builds one server per config and rewinds it between sequences with
# `IkpirServer::reset_for_replay`, so the setup is paid once per config, not
# once per sequence.
#
# Flags narrow the sweep; anything else is forwarded to every bench:
#   --arity N          2 | 3 | 4
#   --bucket-size N
#   --value-bits LIST  comma-separated (default 2048,8192 = 256B,1kB)
#   --backend LIST     comma-separated (default frodo,simple)
#   --flow F           client-hint-patch | client-rewind | all (default all) —
#                      which flow's client mutation leg to run; the common
#                      server_mutation leg always runs once.
#
# Usage:
#   ./scripts/table4.sh                            # the full table, both flows
#   ./scripts/table4.sh --flow client-hint-patch   # camera-ready's HintUpdate column only
#   ./scripts/table4.sh --flow client-rewind       # full paper's HintUpdate column only
#   ./scripts/table4.sh --arity 3                  # just the full-paper arity-3 cells
#   ./scripts/table4.sh --backend simple           # RisePIR-S rows only
#   ./scripts/table4.sh --patch-mode entry         # one patch mode (forwarded;
#                                                     applies to server_mutation
#                                                     and client_hint_patch_mutation)
#
# Output: results/ikpir-server/ikpir_server_mutation.csv
#         results/ikpir-client/ikpir_client_hint_patch_mutation.csv (--flow client-hint-patch|all)
#         results/ikpir-client/ikpir_client_rewind_mutation.csv (--flow client-rewind|all)
#         — one row per (config, value width, backend[, patch mode], kind); the
#         two flows' data lands in separate files and is never merged.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; }
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi

paper_select "${PAPER_PIR_CONFIGS[*]}" "$@"

BENCHES=()
while IFS= read -r b; do BENCHES+=("$b"); done < <(table_benches_for_flow 4 "$PAPER_SEL_FLOW")

log "Table 4 — mutation throughput ($(paper_run_count ${#BENCHES[@]}) runs: \
${#PAPER_SEL_CONFIGS[@]} config(s) × ${#PAPER_SEL_VALUE_BITS[@]} value width(s) × \
${#PAPER_SEL_BACKENDS[@]} backend(s) × ${#BENCHES[@]} benches, flow=$PAPER_SEL_FLOW, \
fill $PAPER_LOAD_FACTOR, τ = $PAPER_TAU_PERCENT% of slots)"
if [[ ${#PAPER_EXTRA[@]} -gt 0 ]]; then
    warn "forwarding to every bench: ${PAPER_EXTRA[*]}"
fi

for cfg in "${PAPER_SEL_CONFIGS[@]}"; do
    arity=${cfg%%:*}; bs=${cfg##*:}
    nb=$(paper_num_buckets "$arity" "$bs")
    for vb in "${PAPER_SEL_VALUE_BITS[@]}"; do
        for be in "${PAPER_SEL_BACKENDS[@]}"; do
            for b in "${BENCHES[@]}"; do
                "$SCRIPT_DIR/bench.sh" "$b" \
                    --arity "$arity" --bucket-size "$bs" --num-buckets "$nb" \
                    --value-bits "$vb" --backend "$be" \
                    ${PAPER_EXTRA[@]+"${PAPER_EXTRA[@]}"}
            done
        done
    done
done

ok "Table 4 complete — CSVs under results/ikpir-{server,client}/"
printf '  %s\n' ikpir_server_mutation.csv
case "$PAPER_SEL_FLOW" in
    client-hint-patch) printf '  %s\n' ikpir_client_hint_patch_mutation.csv ;;
    client-rewind)     printf '  %s\n' ikpir_client_rewind_mutation.csv ;;
    all)               printf '  %s\n' ikpir_client_hint_patch_mutation.csv \
                                        ikpir_client_rewind_mutation.csv ;;
esac
