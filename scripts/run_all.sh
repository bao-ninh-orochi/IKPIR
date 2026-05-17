#!/usr/bin/env bash
# Run every IKPIR bench (server + client) across the paper-derived configs,
# then render plots.
#
# Usage:
#   ./scripts/run_all.sh                           # default profile
#   IKPIR_BENCH_PROFILE=quick ./scripts/run_all.sh # smaller config matrix
#   IKPIR_BENCH_PROFILE=full  ./scripts/run_all.sh # include m=2^22 row
#
#   IKPIR_BENCH_BACKENDS=frodo ./scripts/run_all.sh         # default: FrodoPIR only
#   IKPIR_BENCH_BACKENDS=simple ./scripts/run_all.sh        # SimplePIR only
#   IKPIR_BENCH_BACKENDS=frodo,simple ./scripts/run_all.sh  # both (~2× runtime)
#
#   ./scripts/run_all.sh --no-plot                 # skip the plot step
#   ./scripts/run_all.sh --plot-only               # only run plot.py (assume CSVs already populated)
#   ./scripts/run_all.sh --server-only             # ikpir-server benches only
#   ./scripts/run_all.sh --client-only             # ikpir-client benches only
#
# Total runtime (default profile, M1/M2 Mac):
#   ikpir-server: ~20-30 min
#   ikpir-client: ~10-15 min
#   plots:        seconds

set -euo pipefail

WORKSPACE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

DO_SERVER=1
DO_CLIENT=1
DO_PLOT=1

while (( $# > 0 )); do
    case "$1" in
        --no-plot)     DO_PLOT=0 ;;
        --plot-only)   DO_SERVER=0; DO_CLIENT=0 ;;
        --server-only) DO_CLIENT=0 ;;
        --client-only) DO_SERVER=0 ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed -n 's/^# \?//p'
            exit 0
            ;;
        *) echo "unknown flag: $1"; exit 1 ;;
    esac
    shift
done

if [[ -t 1 ]]; then
    BOLD='\033[1m'; RESET='\033[0m'
else
    BOLD=''; RESET=''
fi

echo -e "${BOLD}IKPIR bench orchestrator${RESET} (profile=${IKPIR_BENCH_PROFILE:-default}, backends=${IKPIR_BENCH_BACKENDS:-frodo})"

if (( DO_SERVER )); then
    echo -e "${BOLD}▶ ikpir-server${RESET}"
    bash "$WORKSPACE_DIR/ikpir-server/scripts/run_benches.sh"
fi

if (( DO_CLIENT )); then
    echo -e "${BOLD}▶ ikpir-client${RESET}"
    bash "$WORKSPACE_DIR/ikpir-client/scripts/run_benches.sh"
fi

if (( DO_PLOT )); then
    echo -e "${BOLD}▶ plotting${RESET}"
    if ! python3 -c 'import pandas, matplotlib' 2>/dev/null; then
        echo "  Missing python deps. Installing..."
        pip install -r "$WORKSPACE_DIR/ikpir-server/scripts/requirements.txt"
    fi
    ( cd "$WORKSPACE_DIR/ikpir-server" && python3 scripts/plot.py )
    ( cd "$WORKSPACE_DIR/ikpir-client" && python3 scripts/plot.py )
fi

echo -e "${BOLD}Done.${RESET}"
echo "  Server CSVs: $WORKSPACE_DIR/ikpir-server/results/"
echo "  Client CSVs: $WORKSPACE_DIR/ikpir-client/results/"
echo "  Server PNGs: $WORKSPACE_DIR/ikpir-server/results/plots/"
echo "  Client PNGs: $WORKSPACE_DIR/ikpir-client/results/plots/"
