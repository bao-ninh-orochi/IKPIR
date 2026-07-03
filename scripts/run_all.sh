#!/usr/bin/env bash
# The canonical full-paper sweep. Chains the three fused sweeps — classical
# throughput, mutation throughput, and server setup — over the shared config
# matrix in scripts/configs.sh. One command → the complete IKPIR dataset.
#
# Delegates to the per-sweep orchestrators (run_classical.sh, run_mutation.sh,
# run_server_setup.sh, and optionally run_headtohead.sh), so their environment
# knobs pass straight through:
#   IKPIR_BENCH_BACKENDS     frodo (default) | simple | frodo,simple
#   IKPIR_BENCH_PATCH_MODES  entry,row (default) | entry | row  (mutation sweep)
#   IKPIR_SETUP_ESTIMATE     1 (default) | 0                    (setup sweep)
#   MAX_MEM_GB               85.0 (default)                     (OOM guards)
#   N_MUTATIONS_CAP          2000 (default)                     (mutation sweep)
#   TRIALS / WARMUP          1 / 0 (default)                    (setup sweep)
#   RESUME_AFTER             ''                                 (setup sweep)
#
# Usage:
#   ./scripts/run_all.sh                     # classical + mutation + setup
#   ./scripts/run_all.sh --skip-setup        # classical + mutation (setup is the long sweep)
#   ./scripts/run_all.sh --classical-only    # one sweep only
#   ./scripts/run_all.sh --mutation-only     #   "
#   ./scripts/run_all.sh --setup-only        #   "
#   ./scripts/run_all.sh --headtohead        # ALSO run the fixed-N head-to-head matrix at the end
#   IKPIR_BENCH_BACKENDS=frodo,simple ./scripts/run_all.sh   # both backends (~2× runtime)
#
# For a single bench at a single config, use the per-crate scripts
# (ikpir-server/scripts/run_benches.sh, ikpir-client/scripts/run_benches.sh)
# or `cargo bench` directly — but do not mix them with this sweep on the same
# CSV files: the fused benches share the individual benches' CSV schemas and
# rows would accumulate as duplicates (see the WARNING in run_mutation.sh /
# run_classical.sh).
#
# Output CSVs:
#   ikpir-client/results/ikpir_server_answer.csv        (classical)
#   ikpir-client/results/ikpir_client_query.csv         (classical)
#   ikpir-client/results/ikpir_client_decode.csv        (classical)
#   ikpir-client/results/ikpir_server_mutation.csv      (mutation)
#   ikpir-client/results/ikpir_client_mutation.csv      (mutation)
#   ikpir-server/results/ikpir_server_setup.csv         (setup; append + RESUME_AFTER)
#   ikpir-client/results/ikpir_headtohead_*.csv         (--headtohead only)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
    sed -n '2,42p' "$0" | sed 's/^# \{0,1\}//'
}

RUN_CLASSICAL=1
RUN_MUTATION=1
RUN_SETUP=1
RUN_HEADTOHEAD=0
for arg in "$@"; do
    case "$arg" in
        --classical-only) RUN_CLASSICAL=1; RUN_MUTATION=0; RUN_SETUP=0 ;;
        --mutation-only)  RUN_CLASSICAL=0; RUN_MUTATION=1; RUN_SETUP=0 ;;
        --setup-only)     RUN_CLASSICAL=0; RUN_MUTATION=0; RUN_SETUP=1 ;;
        --skip-setup)     RUN_SETUP=0 ;;
        --headtohead)     RUN_HEADTOHEAD=1 ;;
        -h|--help)        usage; exit 0 ;;
        *)
            echo "run_all.sh: unknown flag '$arg'" >&2
            echo "valid: --classical-only --mutation-only --setup-only --skip-setup --headtohead" >&2
            exit 1
            ;;
    esac
done

if (( RUN_CLASSICAL )); then
    "$SCRIPT_DIR/run_classical.sh"
fi
if (( RUN_MUTATION )); then
    "$SCRIPT_DIR/run_mutation.sh"
fi
if (( RUN_SETUP )); then
    "$SCRIPT_DIR/run_server_setup.sh"
fi
if (( RUN_HEADTOHEAD )); then
    "$SCRIPT_DIR/run_headtohead.sh"
fi

echo
echo "run_all.sh: done."
