#!/usr/bin/env bash
# Bench config matrix for paper evaluation.
#
# Sourced by run_*.sh scripts; defines arrays consumed by the orchestrators.
#
# Conventions:
#  * BENCH_CONFIGS: 20 (arity, bucket_size, num_buckets) tuples covering the
#    full head-to-head matrix: (2,4)×3, (4,1)×3, (4,2)×3, (3,2)×4, (3,3)×3, (4,3)×4.
#  * num_buckets is the SCF parameter; capacity = num_buckets × bucket_size.
#    Constraints: arity=2 → power of 2; arity=3 → 3·2^t; arity=4 → power of 2.
#  * VALUE_BITS sweep: 256/2048/8192 bits (32 B / 256 B / 1 kB).
#  * N_MUTATIONS sweep: 1 4 16 64 256 1024 4096 (server_mutation, client_mutation).
#  * LOAD_FACTOR: default 0.80 for server_mutation and client_mutation.
#  * lwe_dim: 1566 for FrodoPIR, 1275 for SimplePIR (lattice estimator, ADPS16).
#  * plaintext_bits=8 (p=2^8), fingerprint_bits=32, q=2^32 — all fixed.
#
# Total-entries coverage:
#   (2,4) (4,1) (4,2): exact powers of 2 — 2^18, 2^20, 2^22.
#   (3,2) (4,3): 4 entries each ≈ 3·2^{17..20}  (≈ 2^18.6..2^21.6).
#   (3,3):       3 entries     ≈ 9·2^{15,17,19} (≈ 2^18.2, 2^20.2, 2^22.2).

# ─────────────────────────────────────────────────────────────────────────────
# Head-to-head config matrix.
# Each tuple: "arity:bucket_size:num_buckets:n_label:m_label".
#   arity        — SCF arity (2 / 3 / 4)
#   bucket_size  — slots per bucket
#   num_buckets  — total SCF bucket count
#   n_label      — short label for num_buckets
#   m_label      — short label for total entries = num_buckets × bucket_size
# ─────────────────────────────────────────────────────────────────────────────

BENCH_CONFIGS=(
    # arity=2, bucket_size=4 — total_entries ∈ {2^18, 2^20, 2^22}
    "2:4:65536:2^16:2^18"
    "2:4:262144:2^18:2^20"
    "2:4:1048576:2^20:2^22"

    # arity=4, bucket_size=1 — total_entries ∈ {2^18, 2^20, 2^22}
    "4:1:262144:2^18:2^18"
    "4:1:1048576:2^20:2^20"
    "4:1:4194304:2^22:2^22"

    # arity=4, bucket_size=2 — total_entries ∈ {2^18, 2^20, 2^22}
    "4:2:131072:2^17:2^18"
    "4:2:524288:2^19:2^20"
    "4:2:2097152:2^21:2^22"

    # arity=3, bucket_size=2 — total_entries ∈ {3·2^17, 3·2^18, 3·2^19, 3·2^20}
    "3:2:196608:3x2^16:3x2^17"
    "3:2:393216:3x2^17:3x2^18"
    "3:2:786432:3x2^18:3x2^19"
    "3:2:1572864:3x2^19:3x2^20"

    # arity=3, bucket_size=3 — total_entries ∈ {9·2^15, 9·2^17, 9·2^19}
    "3:3:98304:3x2^15:9x2^15"
    "3:3:393216:3x2^17:9x2^17"
    "3:3:1572864:3x2^19:9x2^19"

    # arity=4, bucket_size=3 — total_entries ∈ {3·2^17, 3·2^18, 3·2^19, 3·2^20}
    "4:3:131072:2^17:3x2^17"
    "4:3:262144:2^18:3x2^18"
    "4:3:524288:2^19:3x2^19"
    "4:3:1048576:2^20:3x2^20"
)

# Value-bits sweep applied to every config (32 B / 256 B / 1 kB).
VALUE_BITS=(256 2048 8192)
W_LABELS=("32B" "256B" "1kB")

# ─────────────────────────────────────────────────────────────────────────────
# Mutation-bench config: one entry per (arity, bucket_size), total_entries ≈ 1 M.
# Tuple format: "arity:bucket_size:num_buckets:n_label:m_label"
#   capacity = num_buckets × bucket_size
#   n_mutations = capacity / 100  (1 % of capacity, computed per config in scripts)
# ─────────────────────────────────────────────────────────────────────────────
MUTATION_CONFIGS=(
    "2:4:262144:2^18:2^20"      # capacity 1,048,576
    "4:1:1048576:2^20:2^20"     # capacity 1,048,576
    "4:2:524288:2^19:2^20"      # capacity 1,048,576
    "3:2:786432:3x2^18:3x2^19"  # capacity 1,572,864
    "3:3:393216:3x2^17:9x2^17"  # capacity 1,179,648
    "4:3:524288:2^19:3x2^19"    # capacity 1,572,864
)

MUTATION_LOAD_FACTOR=0.90  # 90 % full

# ─────────────────────────────────────────────────────────────────────────────
# Backend selector — per-bench `--backend frodo|simple` dispatch.
# Set IKPIR_BENCH_BACKENDS=frodo (default), =simple, or =frodo,simple.
# ─────────────────────────────────────────────────────────────────────────────

IKPIR_BENCH_BACKENDS=${IKPIR_BENCH_BACKENDS:-frodo}
IFS=',' read -ra BACKENDS_ARR <<< "$IKPIR_BENCH_BACKENDS"

for _b in "${BACKENDS_ARR[@]}"; do
    case "$_b" in
        frodo|simple) ;;
        *) echo "[configs.sh] ERROR: unknown backend '$_b' (valid: frodo, simple)" >&2; exit 1 ;;
    esac
done
unset _b

# Backend-appropriate LWE dimension for 128-bit security, estimated via
# the lattice estimator under the ADPS16 cost model. FrodoPIR uses 1566.
# SimplePIR uses 1275 (at q=2^32, σ=6.4).
backend_lwe_dim() {
    case "$1" in
        frodo)  echo 1566 ;;
        simple) echo 1275 ;;
        *) echo "[configs.sh] backend_lwe_dim: unknown backend '$1'" >&2; return 1 ;;
    esac
}

echo "[configs.sh] backends=${IKPIR_BENCH_BACKENDS} | ${#BENCH_CONFIGS[@]} configs × ${#VALUE_BITS[@]} value_bits = $(( ${#BENCH_CONFIGS[@]} * ${#VALUE_BITS[@]} )) classical runs/bench/backend | mutation: ${#MUTATION_CONFIGS[@]} configs × ${#VALUE_BITS[@]} value_bits × lf=${MUTATION_LOAD_FACTOR} × N=1% capacity = $(( ${#MUTATION_CONFIGS[@]} * ${#VALUE_BITS[@]} )) runs/bench/backend"
