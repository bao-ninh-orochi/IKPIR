#!/usr/bin/env bash
# Reproduce the paper's mutation table: insert / update / delete throughput of
# RisePIR-F and RisePIR-S, the incremental half the static baselines cannot do
# at all. table5.sh gives the rebuild cost these throughputs replace.
#
# Runs both mutation benches over the paper matrix in scripts/lib.sh, the single
# source of truth:
#
#   client_mutation   HintUpdate throughput — the number the table reports
#   server_mutation   DBMutation throughput + delta wire bytes
#
# Five (arity, bucket_size) cells × 2 value widths × 2 backends = 20 configs, 2
# benches each. Every cell's n_b comes from `paper_num_buckets`; the arity-2/4
# cells share N = 2^20 slots, and arity 3 lands on its own ladder (1 572 864 for
# (3,2), 1 179 648 for (3,3)) — see lib.sh for why it cannot reach 2^20. Every
# run is at fingerprint_bits = 64, the paper's width (bench.sh's default).
#
# Each bench seeds to fill 0.90, applies one batch of τ = 1% of the table's
# slots per (kind, patch mode) pair, and divides the batch by the measured time.
# bench.sh derives τ from the geometry (10 485 at N = 2^20) and sweeps both
# patch modes, so this script passes geometry only — the batch rule lives in one
# place.
#
# Flags narrow the sweep; anything else is forwarded to every bench:
#   --arity N          2 | 3 | 4
#   --bucket-size N
#   --value-bits LIST  comma-separated (default 2048,8192 = 256B,1kB)
#   --backend LIST     comma-separated (default frodo,simple)
#
# Usage:
#   ./scripts/table4.sh                        # the full table
#   ./scripts/table4.sh --arity 3              # just the full-paper arity-3 cells
#   ./scripts/table4.sh --backend simple       # RisePIR-S rows only
#   ./scripts/table4.sh --patch-mode entry     # one patch mode (forwarded)
#
# Output: results/ikpir-server/ikpir_server_mutation.csv
#         results/ikpir-client/ikpir_client_mutation.csv
#         — one row per (config, value width, backend, patch mode, kind).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; }
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi

paper_select "${PAPER_PIR_CONFIGS[*]}" "$@"

log "Table 4 — mutation throughput ($(paper_run_count ${#PIR_TABLE4_BENCHES[@]}) runs: \
${#PAPER_SEL_CONFIGS[@]} config(s) × ${#PAPER_SEL_VALUE_BITS[@]} value width(s) × \
${#PAPER_SEL_BACKENDS[@]} backend(s) × ${#PIR_TABLE4_BENCHES[@]} benches, fill $PAPER_LOAD_FACTOR, \
τ = $PAPER_TAU_PERCENT% of slots)"
if [[ ${#PAPER_EXTRA[@]} -gt 0 ]]; then
    warn "forwarding to every bench: ${PAPER_EXTRA[*]}"
fi

for cfg in "${PAPER_SEL_CONFIGS[@]}"; do
    arity=${cfg%%:*}; bs=${cfg##*:}
    nb=$(paper_num_buckets "$arity" "$bs")
    for vb in "${PAPER_SEL_VALUE_BITS[@]}"; do
        for be in "${PAPER_SEL_BACKENDS[@]}"; do
            for b in "${PIR_TABLE4_BENCHES[@]}"; do
                "$SCRIPT_DIR/bench.sh" "$b" \
                    --arity "$arity" --bucket-size "$bs" --num-buckets "$nb" \
                    --value-bits "$vb" --backend "$be" \
                    ${PAPER_EXTRA[@]+"${PAPER_EXTRA[@]}"}
            done
        done
    done
done

ok "Table 4 complete — CSVs under results/ikpir-{server,client}/"
printf '  %s\n' ikpir_server_mutation.csv ikpir_client_mutation.csv
