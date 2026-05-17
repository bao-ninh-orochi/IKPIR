#!/usr/bin/env bash
# Config matrices derived from ChalametPIR (CCS'24) Table 1/2 and Hao-2025 Table 1/Figure 10.
#
# Sourced by run_*.sh scripts; defines arrays consumed by the orchestrators.
#
# Conventions:
#  * num_buckets is the IKPIR SCF parameter; capacity = num_buckets × bucket_size.
#    With bucket_size=4 and a TableFull populate (~95% load), num_keys ≈ num_buckets × 3.8.
#  * value_bits is the value width in bits per (key, value) entry.
#  * arity ∈ {2, 3, 4} maps to {Segmented2ary, Segmented3ary, Segmented4ary}.
#    Constraints: arity=2 → num_buckets is a power of 2 ≥ 2; arity=3 → num_buckets = 3·2^t;
#    arity=4 → num_buckets is a power of 2 ≥ 4.
#  * lwe_dim=1774 is the FrodoPIR Table-5 default (n=1774, q=2^32, λ=128 bits).
#    SimplePIR's n=1024 (and HintlessPIR's n=1408) are exposed via the LWE-dim sweep.
#
# Paper anchor points (single source of truth for the comparisons):
#   ChalametPIR Table 1: m × w table for FrodoPIR k=3 and k=4
#     m ∈ {2^16, 2^17, 2^18, 2^19, 2^20}, w=1024B  (BFF k=3, BFF k=4)
#     Plus: (m=2^20, w=256B), (m=2^17, w=30 720B), (m=2^14, w=102 400B)
#   ChalametPIR Table 2: same m at w=1024B; reports server/answer/parse timings.
#   Hao-2025 Table 1: m ∈ {2^18, 2^20, 2^22} × w ∈ {32, 64, 128, 256}B
#   Hao-2025 Figure 10: m=2^20 fixed; w ∈ {4, 8, 16, 32, 64, 128, 256, 512, 1024}B
#
# IKPIR mapping:
#   value_bits = w_bytes × 8
#   target_num_keys ≈ m ⇒ num_buckets ≈ m / (bucket_size × load) ≈ m / 3.8

# ─────────────────────────────────────────────────────────────────────────────
# Headline matrix A — database-size sweep (m varies; w fixed at 256 B = 2048 bits)
# Covers ChalametPIR Case I (m=2^20, 256B) and Hao-2025 Table 1 (256B column).
# ─────────────────────────────────────────────────────────────────────────────
#
# arity=2 / arity=4, num_buckets = next power of 2 ≥ ceil(m / 3.8):
#   m=2^14 ≈ 16k        → num_buckets = 2^12 = 4096       (capacity 16384)
#   m=2^16 ≈ 65k        → num_buckets = 2^14 = 16384      (capacity 65536)
#   m=2^17 ≈ 131k       → num_buckets = 2^15 = 32768      (capacity 131072)
#   m=2^18 ≈ 262k       → num_buckets = 2^16 = 65536      (capacity 262144)
#   m=2^19 ≈ 524k       → num_buckets = 2^17 = 131072     (capacity 524288)
#   m=2^20 ≈ 1.05M      → num_buckets = 2^18 = 262144     (capacity 1048576)
#   m=2^22 ≈ 4.19M      → num_buckets = 2^20 = 1048576    (capacity 4194304)
#
# arity=3, num_buckets = 3·2^t closest to m / 3.8:
#   m=2^16 → 3·2^13 = 24576
#   m=2^18 → 3·2^15 = 98304
#   m=2^20 → 3·2^17 = 393216
#
# Default headline matrix: arity=2, bucket_size=4, lwe_dim=1774, value_bits=2048 (256B).
# Coverage: 2^14 .. 2^20 keys (skip 2^22 by default — adds ~1 hour).

# (num_buckets, label) — "label" = approximate m (target key count).
MATRIX_A_NB_ARITY2=(4096 16384 32768 65536 131072 262144)
MATRIX_A_M_LABELS=("2^14" "2^16" "2^17" "2^18" "2^19" "2^20")
MATRIX_A_VALUE_BITS=2048   # 256 B
MATRIX_A_BUCKET_SIZE=4
MATRIX_A_ARITY=2
MATRIX_A_LWE_DIM=1774

# Optional extension: include 2^22 keys (very slow, ~1 hour for setup_latency).
MATRIX_A_FULL_NB_ARITY2=(4096 16384 32768 65536 131072 262144 1048576)
MATRIX_A_FULL_M_LABELS=("2^14" "2^16" "2^17" "2^18" "2^19" "2^20" "2^22")

# arity=3 mirror (Segmented3ary).
MATRIX_A_NB_ARITY3=(6144 12288 24576 49152 98304 196608)
MATRIX_A_M_LABELS_ARITY3=("2^14" "2^15" "2^16" "2^17" "2^18" "2^19")

# arity=4 mirror (Segmented4ary uses power-of-2 ≥ 4).
MATRIX_A_NB_ARITY4=(4096 16384 32768 65536 131072 262144)

# ─────────────────────────────────────────────────────────────────────────────
# Headline matrix B — value-size sweep (w varies; m fixed at ~2^20)
# Covers Hao-2025 Figure 10 directly (m=2^20, value-length curve).
# ─────────────────────────────────────────────────────────────────────────────
#
# Fixed: num_buckets=262144 (arity=2) ⇒ ~1M keys at TableFull
#        bucket_size=4, lwe_dim=1774, arity=2
# Vary:  value_bits ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}
#        (4 B, 8 B, 16 B, 32 B, 64 B, 128 B, 256 B, 512 B, 1 kB)

MATRIX_B_VALUE_BITS=(32 64 128 256 512 1024 2048 4096 8192)
MATRIX_B_W_LABELS=("4B" "8B" "16B" "32B" "64B" "128B" "256B" "512B" "1kB")
MATRIX_B_NB_ARITY2=262144
MATRIX_B_BUCKET_SIZE=4
MATRIX_B_ARITY=2
MATRIX_B_LWE_DIM=1774

# Smaller m for faster runs (m≈2^16 keys).
MATRIX_B_FAST_NB_ARITY2=16384

# ─────────────────────────────────────────────────────────────────────────────
# Matrix C — arity / k comparison (BFF k=3 / k=4 analog)
# Maps to ChalametPIR Table 1 (k=3 vs k=4 columns).
# ─────────────────────────────────────────────────────────────────────────────
#
# Fixed: m≈2^16 keys (small for quick comparison), bucket_size=4, value_bits=2048
# Vary:  arity ∈ {2, 3, 4}
#        For arity=2/4 → num_buckets=16384
#        For arity=3 → num_buckets=24576 (3·2^13)
#
# Three (arity, num_buckets) pairs at the same m anchor.

MATRIX_C_ARITY=(2 3 4)
MATRIX_C_NUM_BUCKETS=(16384 24576 16384)
MATRIX_C_VALUE_BITS=2048
MATRIX_C_BUCKET_SIZE=4
MATRIX_C_LWE_DIM=1774

# ─────────────────────────────────────────────────────────────────────────────
# Matrix E — LWE-dim sweep
# Covers ChalametPIR (n=1774, FrodoPIR), Hao-2025 (n=1024, SimplePIR),
# and HintlessPIR (n=1408).
# ─────────────────────────────────────────────────────────────────────────────

MATRIX_E_LWE_DIM=(1024 1408 1774)
MATRIX_E_LWE_LABELS=("SimplePIR-1024" "Hintless-1408" "Frodo-1774")
MATRIX_E_ARITY=2
MATRIX_E_NUM_BUCKETS=16384
MATRIX_E_BUCKET_SIZE=4
MATRIX_E_VALUE_BITS=2048

# ─────────────────────────────────────────────────────────────────────────────
# Matrix F — incremental crossover sweep (no analog in either paper).
# Justifies the "Incremental" in IKPIR.
# ─────────────────────────────────────────────────────────────────────────────
#
# Fixed: arity=2, bucket_size=4, value_bits=2048, lwe_dim=1774
#        num_buckets sweeps over a small set to show how crossover shifts with DB size
# Vary:  n_mutations ∈ {1, 4, 16, 64, 256, 1024, 4096}

MATRIX_F_N_MUTATIONS=(1 4 16 64 256 1024 4096)
MATRIX_F_NUM_BUCKETS=(4096 16384 65536)
MATRIX_F_NUM_BUCKETS_LABELS=("2^14keys" "2^16keys" "2^18keys")
MATRIX_F_ARITY=2
MATRIX_F_BUCKET_SIZE=4
MATRIX_F_VALUE_BITS=2048
MATRIX_F_LWE_DIM=1774

# ─────────────────────────────────────────────────────────────────────────────
# Matrix G — preprocessing mode sweep
# IKPIR-specific (cold / warm-b / warm-bc) — no paper analog.
# ─────────────────────────────────────────────────────────────────────────────

MATRIX_G_MODES=("cold" "warm-b" "warm-bc")
MATRIX_G_NUM_BUCKETS=(4096 16384 65536)
MATRIX_G_NUM_BUCKETS_LABELS=("2^14keys" "2^16keys" "2^18keys")
MATRIX_G_ARITY=2
MATRIX_G_BUCKET_SIZE=4
MATRIX_G_VALUE_BITS=2048
MATRIX_G_LWE_DIM=1774

# ─────────────────────────────────────────────────────────────────────────────
# Backend selector — per-bench `--backend frodo|simple` dispatch.
# Set IKPIR_BENCH_BACKENDS=frodo (default) or IKPIR_BENCH_BACKENDS=frodo,simple
# (or just simple) to control which backend(s) each sweep covers.
# ─────────────────────────────────────────────────────────────────────────────

IKPIR_BENCH_BACKENDS=${IKPIR_BENCH_BACKENDS:-frodo}
IFS=',' read -ra BACKENDS_ARR <<< "$IKPIR_BENCH_BACKENDS"

# Sanity-check selected backends.
for _b in "${BACKENDS_ARR[@]}"; do
    case "$_b" in
        frodo|simple) ;;
        *) echo "[configs.sh] ERROR: unknown backend '$_b' (valid: frodo, simple)" >&2; exit 1 ;;
    esac
done
unset _b

# Backend-appropriate LWE dimension. FrodoPIR uses 1774 (Table 5; 128-bit
# security), SimplePIR uses 1024 (paper §4.2; 128-bit security at q=2^32,
# σ=6.4). Both are the paper-recommended choices; bench sweeps that want a
# different dim should call the bench with --lwe-dim directly.
backend_lwe_dim() {
    case "$1" in
        frodo)  echo 1774 ;;
        simple) echo 1024 ;;
        *) echo "[configs.sh] backend_lwe_dim: unknown backend '$1'" >&2; return 1 ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
# Profile selector — pick a smaller subset for quick iteration.
# Set IKPIR_BENCH_PROFILE=quick | default | full before sourcing this file
# (or pass IKPIR_BENCH_PROFILE=quick to any run_*.sh script).
# ─────────────────────────────────────────────────────────────────────────────

IKPIR_BENCH_PROFILE=${IKPIR_BENCH_PROFILE:-default}

case "$IKPIR_BENCH_PROFILE" in
    quick)
        # Quick: only smallest 3 m values, smallest 3 w values.
        MATRIX_A_NB_ARITY2=(4096 16384 32768)
        MATRIX_A_M_LABELS=("2^14" "2^16" "2^17")
        MATRIX_A_NB_ARITY3=(6144 12288 24576)
        MATRIX_A_M_LABELS_ARITY3=("2^14" "2^15" "2^16")
        MATRIX_A_NB_ARITY4=(4096 16384 32768)
        MATRIX_B_VALUE_BITS=(64 256 1024 2048)
        MATRIX_B_W_LABELS=("8B" "32B" "128B" "256B")
        MATRIX_B_NB_ARITY2=$MATRIX_B_FAST_NB_ARITY2
        MATRIX_F_NUM_BUCKETS=(4096 16384)
        MATRIX_F_NUM_BUCKETS_LABELS=("2^14keys" "2^16keys")
        MATRIX_F_N_MUTATIONS=(1 16 256 1024)
        MATRIX_G_NUM_BUCKETS=(4096 16384)
        MATRIX_G_NUM_BUCKETS_LABELS=("2^14keys" "2^16keys")
        ;;
    full)
        # Full: include the 2^22 keys row (very slow).
        MATRIX_A_NB_ARITY2=("${MATRIX_A_FULL_NB_ARITY2[@]}")
        MATRIX_A_M_LABELS=("${MATRIX_A_FULL_M_LABELS[@]}")
        ;;
    default|*)
        # Defaults already set above.
        ;;
esac

echo "[configs.sh] profile=$IKPIR_BENCH_PROFILE backends=${IKPIR_BENCH_BACKENDS}"
