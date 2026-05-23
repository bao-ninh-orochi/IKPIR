#!/usr/bin/env bash
# Run every IKPIR bench (server + client) across the full config matrix.
#
# Usage:
#   ./scripts/run_all.sh                                    # FrodoPIR only
#   IKPIR_BENCH_BACKENDS=simple       ./scripts/run_all.sh  # SimplePIR only
#   IKPIR_BENCH_BACKENDS=frodo,simple ./scripts/run_all.sh  # both (~2× runtime)
#
#   ./scripts/run_all.sh --server-only   # ikpir-server benches only
#   ./scripts/run_all.sh --client-only   # ikpir-client benches only
#
# Config matrix: 20 (arity, bucket_size, num_buckets) tuples × 3 value_bits
# = 60 classical runs per bench per backend.
# Mutation sweep: × 7 N_MUTATIONS = 420 mutation runs per bench per backend.
# See scripts/configs.sh for the full matrix.
#
# Faster alternatives:
#   * scripts/run_classical.sh — runs server_answer + client_query +
#     client_decode from one shared fill + setup per config (~3× faster
#     than running the individual scripts for those three benches).
#   * scripts/run_mutation.sh — runs server_mutation + client_mutation
#     from one shared populate + setup per kind (cuts 7 expensive
#     IkpirServer::new calls per config to 3).
# server_setup still requires the individual script.

set -euo pipefail

WORKSPACE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

DO_SERVER=1
DO_CLIENT=1

while (( $# > 0 )); do
    case "$1" in
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

echo -e "${BOLD}IKPIR bench orchestrator${RESET} (backends=${IKPIR_BENCH_BACKENDS:-frodo})"

if (( DO_SERVER )); then
    echo -e "${BOLD}▶ ikpir-server${RESET}"
    bash "$WORKSPACE_DIR/ikpir-server/scripts/run_benches.sh"
fi

if (( DO_CLIENT )); then
    echo -e "${BOLD}▶ ikpir-client${RESET}"
    bash "$WORKSPACE_DIR/ikpir-client/scripts/run_benches.sh"
fi

echo -e "${BOLD}Done.${RESET}"
echo "  Server CSVs: $WORKSPACE_DIR/ikpir-server/results/"
echo "  Client CSVs: $WORKSPACE_DIR/ikpir-client/results/"
