#!/usr/bin/env bash
# Reproduce the paper's online table: per-query upload (Qry.), response download
# (Rsp.), and mean server answer latency (Ans.) for RisePIR-F and RisePIR-S,
# beside ChalametPIR and KPIR^index.
#
# Runs the flow-independent common leg plus the selected flow's decode leg —
#
#   Qry.  ← headtohead_query                    (client build_query; query_bytes;
#                                                  flow-independent — build_query
#                                                  is the same code on both flows)
#   Rsp.  ← headtohead_{hint_patch,rewind}_decode (client decode; response_bytes;
#                                                  one bench per flow, never merged)
#   Ans.  ← headtohead_answer                   (server answer latency, mean_qps)
#
# — over the paper matrix in scripts/lib.sh, the single source of truth:
# five (arity, bucket_size) cells × 2 value widths × 2 backends = 20 configs,
# × (2 common benches + 1 or 2 decode benches, depending on --flow). Every
# cell's n_b comes from `paper_num_buckets`. Every run is at
# fingerprint_bits = 64, the paper's width (bench.sh's default).
#
# Two key-count regimes, resolved per cell by `paper_num_keys`:
#
#   arity 2/4  m = 10^6 keys, the count ChalametPIR and KPIR^index publish at.
#              Fills their 2^20-slot tables to 0.954.
#   arity 3    fill 0.90 instead (1 415 577 keys at (3,2), 1 061 683 at (3,3)).
#              A 3-ary table cannot hold 10^6 keys near that fill — its size
#              ladder is 3·2^t·b, and 10^6 in (3,2) would sit at 0.636 — and it
#              has no baseline to line up against anyway. These are the
#              full-paper cells; the CSV's num_keys / db_size columns record
#              which regime each row came from.
#
# Flags narrow the sweep; anything else is forwarded to every bench:
#   --arity N          2 | 3 | 4
#   --bucket-size N
#   --value-bits LIST  comma-separated (default 2048,8192 = 256B,1kB)
#   --backend LIST     comma-separated (default frodo,simple)
#   --flow F           client-hint-patch | client-rewind | all (default all) —
#                      which flow's decode leg to run; the common legs
#                      (headtohead_answer, headtohead_query) always run once.
#
# Usage:
#   ./scripts/table3.sh                            # the full table, both flows
#   ./scripts/table3.sh --flow client-hint-patch   # camera-ready's decode column only
#   ./scripts/table3.sh --flow client-rewind       # full paper's decode column only
#   ./scripts/table3.sh --arity 3                  # just the full-paper arity-3 cells
#   ./scripts/table3.sh --backend frodo            # RisePIR-F rows only
#   ./scripts/table3.sh --value-bits 2048          # the 256 B column only
#
# Output: results/ikpir-server/ikpir_headtohead_server_answer.csv
#         results/ikpir-client/ikpir_headtohead_client_query.csv
#         results/ikpir-client/ikpir_headtohead_client_hint_patch_decode.csv (--flow client-hint-patch|all)
#         results/ikpir-client/ikpir_headtohead_client_rewind_decode.csv (--flow client-rewind|all)
#         — one row per (config, value width, backend); the two flows' decode
#         data lands in separate files and is never merged.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; }
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi

paper_select "${PAPER_PIR_CONFIGS[*]}" "$@"

BENCHES=()
while IFS= read -r b; do BENCHES+=("$b"); done < <(table_benches_for_flow 3 "$PAPER_SEL_FLOW")

log "Table 3 — online cost ($(paper_run_count ${#BENCHES[@]}) runs: \
${#PAPER_SEL_CONFIGS[@]} config(s) × ${#PAPER_SEL_VALUE_BITS[@]} value width(s) × \
${#PAPER_SEL_BACKENDS[@]} backend(s) × ${#BENCHES[@]} benches, flow=$PAPER_SEL_FLOW)"
if [[ ${#PAPER_EXTRA[@]} -gt 0 ]]; then
    warn "forwarding to every bench: ${PAPER_EXTRA[*]}"
fi

for cfg in "${PAPER_SEL_CONFIGS[@]}"; do
    arity=${cfg%%:*}; bs=${cfg##*:}
    nb=$(paper_num_buckets "$arity" "$bs")
    keys=$(paper_num_keys "$arity" "$bs")
    for vb in "${PAPER_SEL_VALUE_BITS[@]}"; do
        for be in "${PAPER_SEL_BACKENDS[@]}"; do
            for b in "${BENCHES[@]}"; do
                "$SCRIPT_DIR/bench.sh" "$b" \
                    --arity "$arity" --bucket-size "$bs" --num-buckets "$nb" \
                    --num-keys "$keys" --value-bits "$vb" --backend "$be" \
                    ${PAPER_EXTRA[@]+"${PAPER_EXTRA[@]}"}
            done
        done
    done
done

ok "Table 3 complete — CSVs under results/ikpir-{server,client}/"
printf '  %s\n' ikpir_headtohead_server_answer.csv \
                ikpir_headtohead_client_query.csv
case "$PAPER_SEL_FLOW" in
    client-hint-patch) printf '  %s\n' ikpir_headtohead_client_hint_patch_decode.csv ;;
    client-rewind)     printf '  %s\n' ikpir_headtohead_client_rewind_decode.csv ;;
    all)               printf '  %s\n' ikpir_headtohead_client_hint_patch_decode.csv \
                                        ikpir_headtohead_client_rewind_decode.csv ;;
esac
