#!/usr/bin/env bash
# Reproduce the paper's preprocessing table: the index-PIR re-encoding time
# (Setup) of RisePIR-F and RisePIR-S — the rebuild a static scheme pays on every
# mutation, and the baseline the table4.sh throughputs are weighed against.
#
# Runs server_setup over PAPER_SETUP_CONFIGS from scripts/lib.sh: the three
# arity-2/4 cells at N = 2^20 slots, × 2 value widths × 2 backends = 12 runs.
# Arity 3 is deliberately out of scope here — it is a full-paper addition to the
# online and mutation tables, and carries no static-rebuild row. To measure it
# anyway: `./scripts/bench.sh server_setup --arity 3 --bucket-size 2 \
# --num-buckets 786432 --value-bits 8192`.
#
# Both benched stores are seeded to fill 0.90, matching table4.sh so the two
# tables describe one store. Setup time does not actually depend on the fill —
# both backends' compute_hint multiply every cell unconditionally, so only the
# matrix shape matters — but the CSV's load_factor column then says what the
# table's caption says.
#
# Flags narrow the sweep; anything else is forwarded to the bench:
#   --arity N          2 | 4
#   --bucket-size N
#   --value-bits LIST  comma-separated (default 2048,8192 = 256B,1kB)
#   --backend LIST     comma-separated (default frodo,simple)
#
# Runtime: this is the slow one — a full run is hours. Setup is the Θ(nNw) hint
# rebuild the whole paper is about avoiding, and the 1 kB × SimplePIR cells run
# to several minutes each on the paper's machine. Pass `--estimate` to time one
# segment and scale by arity (exact, ~arity× faster), or narrow with
# --value-bits / --backend.
#
# Usage:
#   ./scripts/table5.sh                        # the full table (hours)
#   ./scripts/table5.sh --estimate             # same numbers, ~arity× faster
#   ./scripts/table5.sh --backend frodo        # RisePIR-F rows only
#   ./scripts/table5.sh --trials 3             # tighter error bars, slower
#
# Output: results/ikpir-server/ikpir_server_setup.csv
#         — one row per (config, value width, backend).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; }
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi

paper_select "${PAPER_SETUP_CONFIGS[*]}" "$@"

log "Table 5 — setup / static rebuild cost ($(paper_run_count ${#PIR_TABLE5_BENCHES[@]}) runs: \
${#PAPER_SEL_CONFIGS[@]} config(s) × ${#PAPER_SEL_VALUE_BITS[@]} value width(s) × \
${#PAPER_SEL_BACKENDS[@]} backend(s), fill $PAPER_LOAD_FACTOR)"
if [[ ${#PAPER_EXTRA[@]} -gt 0 ]]; then
    warn "forwarding to every bench: ${PAPER_EXTRA[*]}"
fi

for cfg in "${PAPER_SEL_CONFIGS[@]}"; do
    arity=${cfg%%:*}; bs=${cfg##*:}
    nb=$(paper_num_buckets "$arity" "$bs")
    for vb in "${PAPER_SEL_VALUE_BITS[@]}"; do
        for be in "${PAPER_SEL_BACKENDS[@]}"; do
            for b in "${PIR_TABLE5_BENCHES[@]}"; do
                "$SCRIPT_DIR/bench.sh" "$b" \
                    --arity "$arity" --bucket-size "$bs" --num-buckets "$nb" \
                    --value-bits "$vb" --backend "$be" \
                    ${PAPER_EXTRA[@]+"${PAPER_EXTRA[@]}"}
            done
        done
    done
done

ok "Table 5 complete — CSV under results/ikpir-server/"
printf '  %s\n' ikpir_server_setup.csv
