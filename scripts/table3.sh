#!/usr/bin/env bash
# Reproduce the paper's online table: per-query upload (Qry.), response download
# (Rsp.), and mean server answer latency (Ans.) for RisePIR-F and RisePIR-S,
# beside ChalametPIR and KPIR^index.
#
# Runs the three head-to-head benches that feed its columns —
#
#   Qry.  ← headtohead_query    (client build_query; query_bytes)
#   Rsp.  ← headtohead_decode   (client decode; response_bytes)
#   Ans.  ← headtohead_answer   (server answer latency, mean_qps)
#
# — over the paper matrix in scripts/lib.sh, the single source of truth:
# five (arity, bucket_size) cells × 2 value widths × 2 backends = 20 configs,
# 3 benches each. Every cell's n_b comes from `paper_num_buckets`.
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
#
# Usage:
#   ./scripts/table3.sh                        # the full table
#   ./scripts/table3.sh --arity 3              # just the full-paper arity-3 cells
#   ./scripts/table3.sh --backend frodo        # RisePIR-F rows only
#   ./scripts/table3.sh --value-bits 2048      # the 256 B column only
#
# Output: results/ikpir-server/ikpir_headtohead_server_answer.csv
#         results/ikpir-client/ikpir_headtohead_client_{query,decode}.csv
#         — one row per (config, value width, backend).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; }
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi

paper_select "${PAPER_PIR_CONFIGS[*]}" "$@"

log "Table 3 — online cost ($(paper_run_count ${#PIR_TABLE3_BENCHES[@]}) runs: \
${#PAPER_SEL_CONFIGS[@]} config(s) × ${#PAPER_SEL_VALUE_BITS[@]} value width(s) × \
${#PAPER_SEL_BACKENDS[@]} backend(s) × ${#PIR_TABLE3_BENCHES[@]} benches)"
if [[ ${#PAPER_EXTRA[@]} -gt 0 ]]; then
    warn "forwarding to every bench: ${PAPER_EXTRA[*]}"
fi

for cfg in "${PAPER_SEL_CONFIGS[@]}"; do
    arity=${cfg%%:*}; bs=${cfg##*:}
    nb=$(paper_num_buckets "$arity" "$bs")
    keys=$(paper_num_keys "$arity" "$bs")
    for vb in "${PAPER_SEL_VALUE_BITS[@]}"; do
        for be in "${PAPER_SEL_BACKENDS[@]}"; do
            for b in "${PIR_TABLE3_BENCHES[@]}"; do
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
                ikpir_headtohead_client_query.csv \
                ikpir_headtohead_client_decode.csv
