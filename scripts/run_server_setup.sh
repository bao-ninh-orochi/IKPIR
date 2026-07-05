#!/usr/bin/env bash
# Sweep server_setup across the SAME config matrix as the mutation_throughput
# sweep (BENCH_CONFIGS × VALUE_BITS), for BOTH backends by default.
#
# Measures the honest single-threaded, non-SIMD server setup cost
# (IkpirServer::new: derive the LWE matrix A from the seed, then compute the
# per-segment hint H_j = A_jᵀ·D_j). The preprocessing acceleration (rayon +
# wide SIMD) has been reverted, so setup is inherently single-threaded — no
# RAYON_NUM_THREADS to set.
#
# Per-segment estimate (ON by default): IkpirServer::new computes the `arity`
# segment hints in an independent sequential loop, and every segment has the
# same shape, so timing ONE segment's hint and multiplying by `arity` recovers
# the full single-threaded time exactly:
#   full = arity × time(B::server_setup over one segment),
# computing 1/arity of the work (~arity× faster) while staying single-threaded
# / non-SIMD. `mean_setup_ms` always carries the full single-threaded time.
# Toggle via IKPIR_SETUP_ESTIMATE (see scripts/configs.sh):
#   IKPIR_SETUP_ESTIMATE=1  (default) pass --estimate (one segment × arity).
#   IKPIR_SETUP_ESTIMATE=0            time the full IkpirServer::new over all
#                                     segments (slower ground truth; use this
#                                     to validate the estimate).
# The store is built (real per-segment D) in BOTH modes.
#
# Usage:
#   ./scripts/run_server_setup.sh                                    # both backends, per-segment estimate (default)
#   IKPIR_BENCH_BACKENDS=frodo        ./scripts/run_server_setup.sh  # one backend only
#   IKPIR_BENCH_BACKENDS=simple       ./scripts/run_server_setup.sh
#   IKPIR_SETUP_ESTIMATE=0            ./scripts/run_server_setup.sh  # full IkpirServer::new ground truth (slow)
#   TRIALS=3 WARMUP=1 ./scripts/run_server_setup.sh                  # more samples per config (default 1/0)
#   MAX_MEM_GB=20.0   ./scripts/run_server_setup.sh                  # override OOM guard (default 85 GB)
#
# Trials/warmup: each (config, value_bits, backend) times WARMUP+TRIALS
# measurements; only the TRIALS post-warmup samples feed the reported
# mean/stddev. Defaults are TRIALS=1, WARMUP=0 (one measurement per config, no
# warmup) — override via env. NOTE: with WARMUP=0 the single measurement is a
# cold first run; a warmup round trims a small cold-cache bias if you want a
# tighter number (TRIALS=3 WARMUP=1).
#
# RESUME (this sweep is long): the CSV is NEVER cleared here — rows are
# appended, so prior results survive a re-run. To continue after an
# interrupted/partial run, set RESUME_AFTER to the CSV key of the last config
# you already completed:
#       "<backend>,<arity>,<num_buckets>,<bucket_size>,<value_bits>"
# Every (backend, config, value_bits) up to AND INCLUDING that key is skipped;
# the sweep resumes with the next one. Leave RESUME_AFTER='' for a full sweep.
# Iteration order is: backend (frodo, simple) → BENCH_CONFIGS (top→bottom) →
# VALUE_BITS (256, 2048, 8192). Example resume:
#       RESUME_AFTER='frodo,4,4194304,1,2048' \
#           ./scripts/run_server_setup.sh
# To start fresh, delete ikpir-server/results/ikpir_server_setup.csv first.
#
# Memory: dominant cost is the per-segment LWE public matrix A. The full path
# (IkpirServer::new) holds A for all segments at once ≈ num_buckets × lwe_dim
# × 4 B; the estimate path holds only one segment's A ≈ that ÷ arity. Plus the
# store snapshot (built in both modes). After the HintMaterial refactor
# `server.setup()` no longer copies A into the bundle (A is re-expanded
# client-side from the seed), so the server-side bench holds A once. The shell
# guard below uses the full-path bound (2 × num_buckets × lwe_dim × 4, the ×2
# a conservative margin for table cells / transient clones / OS overhead) for
# both modes; it is FrodoPIR-shaped (no SimplePIR √N reshape divisor) and so
# over-estimates SimplePIR and the estimate path — but even the over-estimate
# clears 85 GB for every BENCH_CONFIGS entry (largest peak ≈ 4194304 × 1566 ×
# 4 × 2 ≈ 52.5 GB), so nothing is skipped at the default MAX_MEM_GB. The guard
# still protects smaller machines.
#
# Config matrix (scripts/configs.sh) — identical to run_mutation.sh:
#   12 BENCH_CONFIGS × 3 VALUE_BITS = 36 runs/backend (× 2 backends = 72).
#   Note: server_setup populates to TableFull (not --load-factor 0.90 like the
#   mutation sweep); the (arity, bucket_size, num_buckets, value_bits,
#   plaintext_bits, lwe_dim) tuples are identical and the fill level does not
#   change compute_hint's row count, so setup time is comparable.
#
# Output (appended, never cleared by this script):
#   ikpir-server/results/ikpir_server_setup.csv
#
# Columns: backend, arity, num_buckets, bucket_size, value_bits, plaintext_bits,
#   lwe_dim, mean_setup_ms, min_setup_ms, max_setup_ms, stddev_setup_ms,
#   setup_bundle_bytes, hint_bytes_per_segment, server_params_bytes_per_segment,
#   cells_per_slot, row_width, segment_rows, db_rows, db_cols, load_factor,
#   setup_mode, measured_ms
#
# mean/min/max/stddev_setup_ms always carry the full single-threaded time. The
# trailing columns are traceability: setup_mode = full|per_segment, measured_ms
# = raw per-call time before the arity× scaling (= mean_setup_ms on the full
# path; the one-segment time on the estimate path).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# Default to BOTH backends (matches "for both frodo and simple"); env overrides.
# Must be set before sourcing configs.sh, whose own `:-frodo` default then
# becomes a no-op because the variable is already non-empty.
IKPIR_BENCH_BACKENDS="${IKPIR_BENCH_BACKENDS:-frodo,simple}"

# shellcheck source=./configs.sh
source "$WORKSPACE_DIR/scripts/configs.sh"

cd "$WORKSPACE_DIR"

RESULTS_DIR="$WORKSPACE_DIR/ikpir-server/results"
mkdir -p "$RESULTS_DIR"

if [[ -t 1 ]]; then
    BLUE='\033[34m'; GREEN='\033[32m'; YELLOW='\033[33m'; MAGENTA='\033[35m'; RESET='\033[0m'
else
    BLUE=''; GREEN=''; YELLOW=''; MAGENTA=''; RESET=''
fi

step()         { echo -e "${BLUE}── $* ──${RESET}"; }
note()         { echo -e "${YELLOW}  $*${RESET}"; }
ok()           { echo -e "${GREEN}  $*${RESET}"; }
backend_note() { echo -e "${MAGENTA}▶ backend=$1${RESET}"; }

MAX_MEM_GB=${MAX_MEM_GB:-85.0}
MAX_MEM_BYTES=$(awk "BEGIN { printf \"%d\", $MAX_MEM_GB * 1e9 }")

# One measurement per config, no warmup, by default (override via env).
TRIALS=${TRIALS:-1}; WARMUP=${WARMUP:-0}

# Per-segment estimate vs full ground truth (see scripts/configs.sh).
EST_FLAG=()
if [ "$IKPIR_SETUP_ESTIMATE" = 1 ]; then EST_FLAG=(--estimate); fi

# Resume point (see header). Empty => run the full sweep. Non-empty => skip
# every iteration up to AND INCLUDING this CSV key, then run the rest.
RESUME_AFTER="${RESUME_AFTER:-}"
resume_reached=1
if [ -n "$RESUME_AFTER" ]; then resume_reached=0; fi

CARGO=(cargo bench -p ikpir-server --bench server_setup)

step "server_setup  (single-threaded; trials=${TRIALS}, warmup=${WARMUP}, max_mem=${MAX_MEM_GB} GB)"
if [ -n "$RESUME_AFTER" ]; then
    note "RESUME: skipping through '${RESUME_AFTER}' (inclusive), then continuing. Set RESUME_AFTER='' for a full sweep."
fi
note "Appending to $RESULTS_DIR/ikpir_server_setup.csv (not cleared). Delete it first to start fresh."

for backend in "${BACKENDS_ARR[@]}"; do
    backend_note "$backend"
    lwe=$(backend_lwe_dim "$backend")
    for cfg in "${BENCH_CONFIGS[@]}"; do
        IFS=':' read -r arity bs nb n_label m_label <<< "$cfg"
        for i in "${!VALUE_BITS[@]}"; do
            vb=${VALUE_BITS[$i]}
            w=${W_LABELS[$i]}
            # pb depends on value_bits for the SimplePIR backend, so the
            # lookup lives inside the value-bits loop.
            pb=$(backend_plaintext_bits "$backend" "$arity" "$bs" "$nb" "$vb")
            key="${backend},${arity},${nb},${bs},${vb}"

            # Resume: skip everything up to and including RESUME_AFTER.
            if (( ! resume_reached )); then
                if [[ "$key" == "$RESUME_AFTER" ]]; then
                    resume_reached=1
                    note "↳ resume point '$key' reached (already done) — running everything after it"
                else
                    note "skip (before resume point): $key"
                fi
                continue
            fi

            # Memory guard: full-path peak = server (A, all segments) + clone =
            # 2 × num_buckets × lwe × 4 B. FrodoPIR-shaped; conservative for the
            # estimate path (one segment's A) and SimplePIR. See header.
            peak_bytes=$(( nb * lwe * 4 * 2 ))
            if (( peak_bytes > MAX_MEM_BYTES )); then
                peak_gb=$(awk "BEGIN { printf \"%.1f\", $peak_bytes / 1e9 }")
                a_gb=$(awk "BEGIN { printf \"%.1f\", $nb * $lwe * 4 / 1e9 }")
                note "Skip (OOM guard): estimated peak ${peak_gb} GB > MAX_MEM_GB=${MAX_MEM_GB} (nb=$nb lwe=$lwe, A=${a_gb} GB/copy × 2). Raise MAX_MEM_GB on machines with more RAM."
                continue
            fi

            if [ "$IKPIR_SETUP_ESTIMATE" = 1 ]; then
                note "arity=$arity bs=$bs nb=$nb (n=$n_label, m=$m_label) w=$w lwe=$lwe pb=$pb | per-segment estimate (1 seg × arity)"
            else
                note "arity=$arity bs=$bs nb=$nb (n=$n_label, m=$m_label) w=$w lwe=$lwe pb=$pb | full ground truth"
            fi
            "${CARGO[@]}" -- \
                --backend         "$backend" \
                --arity           "$arity"   \
                --num-buckets     "$nb"      \
                --bucket-size     "$bs"      \
                --value-bits      "$vb"      \
                --plaintext-bits  "$pb"      \
                --lwe-dim         "$lwe"     \
                "${EST_FLAG[@]+"${EST_FLAG[@]}"}" \
                --trials "$TRIALS" --warmup "$WARMUP"
        done
    done
done

if (( ! resume_reached )); then
    echo
    note "WARNING: resume point '$RESUME_AFTER' was never matched — nothing ran. Check the key (backend,arity,num_buckets,bucket_size,value_bits) and iteration order."
fi

echo
ok "Done.  Results:"
ok "  $RESULTS_DIR/ikpir_server_setup.csv"
