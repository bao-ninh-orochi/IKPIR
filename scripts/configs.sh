#!/usr/bin/env bash
# Bench config matrix for paper evaluation.
#
# Sourced by run_*.sh scripts; defines arrays consumed by the orchestrators.
#
# Conventions:
#  * BENCH_CONFIGS: 12 (arity, bucket_size, num_buckets) tuples — 6
#    (arity, bucket_size) pairs × 2 DB sizes each, all with total
#    entries > 1 M. Drives BOTH the classical throughput sweep and the
#    mutation throughput sweep (the two matrices were historically
#    separate and are now unified).
#  * num_buckets is the SCF parameter; capacity = num_buckets × bucket_size.
#    Constraints: arity=2 → power of 2; arity=3 → 3·2^t; arity=4 → power of 2.
#  * VALUE_BITS sweep: 256/2048/8192 bits (32 B / 256 B / 1 kB).
#  * N_MUTATIONS: derived per config as capacity/100 (1 % of capacity);
#    no separate sweep — one N value per BENCH_CONFIGS entry.
#  * LOAD_FACTOR: MUTATION_LOAD_FACTOR=0.90 for server_mutation and client_mutation.
#  * lwe_dim: 1566 for FrodoPIR, 1275 for SimplePIR (lattice estimator, ADPS16).
#  * fingerprint_bits=32, q=2^32 — fixed.
#  * plaintext_bits is the per-config maximum supported by the noise budget
#    of each backend at q=2^32. Looked up via `backend_plaintext_bits` (see
#    "Plaintext-bits table" below); maps (backend, m_label) → pb.
#
# Total-entries coverage (all > 1 M; 2 sizes per (arity, bucket_size) pair):
#   (2,4) (4,1) (4,2): 2^20 (1.05 M), 2^22 (4.19 M).
#   (3,2) (4,3):       3·2^19 (1.57 M), 3·2^20 (3.15 M).
#   (3,3):             9·2^17 (1.18 M), 9·2^19 (4.72 M).

# ─────────────────────────────────────────────────────────────────────────────
# Throughput config matrix (classical + mutation sweeps).
# Each tuple: "arity:bucket_size:num_buckets:n_label:m_label".
#   arity        — SCF arity (2 / 3 / 4)
#   bucket_size  — slots per bucket
#   num_buckets  — total SCF bucket count
#   n_label      — short label for num_buckets
#   m_label      — short label for total entries = num_buckets × bucket_size
#                  (also the key into the plaintext-bits table below)
# 12 tuples: 6 (arity, bucket_size) pairs × 2 DB sizes, all > 1 M.
# ─────────────────────────────────────────────────────────────────────────────

BENCH_CONFIGS=(
    # arity=2, bucket_size=4 — total_entries ∈ {2^20, 2^22}
    "2:4:262144:2^18:2^20"
    "2:4:1048576:2^20:2^22"

    # arity=4, bucket_size=1 — total_entries ∈ {2^20, 2^22}
    "4:1:1048576:2^20:2^20"
    "4:1:4194304:2^22:2^22"

    # arity=4, bucket_size=2 — total_entries ∈ {2^20, 2^22}
    "4:2:524288:2^19:2^20"
    "4:2:2097152:2^21:2^22"

    # arity=3, bucket_size=2 — total_entries ∈ {3·2^19, 3·2^20}
    "3:2:786432:3x2^18:3x2^19"
    "3:2:1572864:3x2^19:3x2^20"

    # arity=3, bucket_size=3 — total_entries ∈ {9·2^17, 9·2^19}
    "3:3:393216:3x2^17:9x2^17"
    "3:3:1572864:3x2^19:9x2^19"

    # arity=4, bucket_size=3 — total_entries ∈ {3·2^19, 3·2^20}
    "4:3:524288:2^19:3x2^19"
    "4:3:1048576:2^20:3x2^20"
)

# Value-bits sweep applied to every config (32 B / 256 B / 1 kB).
VALUE_BITS=(256 2048 8192)
W_LABELS=("32B" "256B" "1kB")

# ─────────────────────────────────────────────────────────────────────────────
# Mutation-bench config knobs (the (arity, bucket_size, num_buckets) matrix
# itself is BENCH_CONFIGS above — both sweeps share it).
#   n_mutations = capacity / 100  (1 % of capacity, computed per config in scripts)
# ─────────────────────────────────────────────────────────────────────────────
MUTATION_LOAD_FACTOR=0.90  # 90 % full

# ─────────────────────────────────────────────────────────────────────────────
# Head-to-head config matrix (vs ChalametPIR and Hao et al. 2025).
#
# Unlike BENCH_CONFIGS (which fixes DB size and populates `until_full`), this
# matrix fixes the **keyword count** so every scheme processes the same N
# and reports its own DB size — the fair-comparison setting requested by
# the paper's head-to-head section.
#
# Tuple format: "num_keys:arity:bucket_size:num_buckets:m_label"
#   num_keys     — keyword count to populate (fed to --num-keys)
#   arity        — SCF arity (2 / 3 / 4)
#   bucket_size  — slots per bucket
#   num_buckets  — smallest valid count s.t. capacity ≥ ⌈N / 0.954⌉
#                  (respects arity constraint: power of 2 for 2-/4-ary,
#                  3·2^t for 3-ary). All 10 entries land at ~95.37% load.
#   m_label      — key into the plaintext-bits table (= total entries label)
#
# (arity, bucket_size) ↔ N pairing (per the user's specification — picking
# a (arity, bs) outside the listed N gives a poor load factor):
#   N = 1 M    →  (2,4)  (4,1)  (4,2)
#   N = 1.5 M  →  (3,2)  (4,3)
#   N = 3 M    →  (3,2)  (4,3)
#   N = 4 M    →  (2,4)  (4,1)  (4,2)
# ─────────────────────────────────────────────────────────────────────────────
HEADTOHEAD_CONFIGS=(
    # N = 1 000 000  (capacity = 2^20 ≈ 1.05 M, load 95.37 %)
    "1000000:2:4:262144:2^20"
    "1000000:4:1:1048576:2^20"
    "1000000:4:2:524288:2^20"

    # N = 1 500 000  (capacity = 3·2^19 ≈ 1.57 M, load 95.37 %)
    "1500000:3:2:786432:3x2^19"
    "1500000:4:3:524288:3x2^19"

    # N = 3 000 000  (capacity = 3·2^20 ≈ 3.15 M, load 95.37 %)
    "3000000:3:2:1572864:3x2^20"
    "3000000:4:3:1048576:3x2^20"

    # N = 4 000 000  (capacity = 2^22 ≈ 4.19 M, load 95.37 %)
    "4000000:2:4:1048576:2^22"
    "4000000:4:1:4194304:2^22"
    "4000000:4:2:2097152:2^22"
)

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

# ─────────────────────────────────────────────────────────────────────────────
# Plaintext-bits table — pb_max per (backend, DB size) at q = 2^32.
#
# DB size = num_buckets × bucket_size  (total cuckoo capacity, keyed by
# m_label). For each unique m_label across BENCH_CONFIGS, the table below
# encodes the largest plaintext_bits value satisfying each backend's
# correctness equation; this is the "smallest cells_per_slot" / "fastest
# PIR" operating point each backend supports.
#
# FrodoPIR (paper eq. 8 / §5.1):
#     q ≥ 8 · p^2 · √m          where m = DB size, p = 2^pb
#   ⇒ p_max = √(2^32 / (8 · √m))
#
#   | DB size       | p_max  | pb | slack vs q=2^32 |
#   |---------------|--------|----|-----------------|
#   | 2^18          | 1024   | 10 | 1.00× boundary  |
#   | 2^20          |  724.1 |  9 | 2.00×           |
#   | 2^22          |  512.0 |  9 | 1.00× boundary  |
#   | 3·2^17        |  925.0 |  9 | 3.27×           |
#   | 3·2^18        |  778.1 |  9 | 2.31×           |
#   | 3·2^19        |  654.0 |  9 | 1.63×           |
#   | 3·2^20        |  550.4 |  9 | 1.16×           |
#   | 9·2^15        |  994.3 |  9 | 3.77×           |
#   | 9·2^17        |  703.1 |  9 | 1.89×           |
#   | 9·2^19        |  497.1 |  8 | 3.77×           |
#
# SimplePIR (paper Appendix C.2, eq. 2):
#     ⌊q/p⌋ ≥ √2 · σ · p · N^(1/4) · √(ln(2/δ))
#       where σ = 6.4, δ = 2^-40 ⇒ factor ≈ 48.249
#   ⇒ p_max = √(2^32 / (48.249 · N^(1/4)))
#
#   | DB size       | p_max  | pb | slack vs q=2^32 |
#   |---------------|--------|----|-----------------|
#   | 2^18          | 1983.6 | 10 | 3.76×           |
#   | 2^20          | 1668.3 | 10 | 2.65×           |
#   | 2^22          | 1402.5 | 10 | 1.88×           |
#   | 3·2^17        | 1885.2 | 10 | 3.39×           |
#   | 3·2^18        | 1729.1 | 10 | 2.85×           |
#   | 3·2^19        | 1585.5 | 10 | 2.40×           |
#   | 3·2^20        | 1453.8 | 10 | 2.02×           |
#   | 9·2^15        | 1954.7 | 10 | 3.64×           |
#   | 9·2^17        | 1643.6 | 10 | 2.58×           |
#   | 9·2^19        | 1382.1 | 10 | 1.82×           |
#
# Usage: backend_plaintext_bits frodo  3x2^18   →   9
#        backend_plaintext_bits simple 9x2^19   →  10
# ─────────────────────────────────────────────────────────────────────────────
backend_plaintext_bits() {
    local backend=$1 m_label=$2 pb
    case "$backend" in
        frodo)
            case "$m_label" in
                "2^18")    pb=10 ;;
                "2^20")    pb=9 ;;
                "2^22")    pb=9 ;;
                "3x2^17")  pb=9 ;;
                "3x2^18")  pb=9 ;;
                "3x2^19")  pb=9 ;;
                "3x2^20")  pb=9 ;;
                "9x2^15")  pb=9 ;;
                "9x2^17")  pb=9 ;;
                "9x2^19")  pb=8 ;;
                *) echo "[configs.sh] backend_plaintext_bits: unknown m_label '$m_label' for backend frodo" >&2; return 1 ;;
            esac
            ;;
        simple)
            # All 10 DB sizes covered by the table below comfortably support
            # pb=10 (slack ≥ 1.82×); pb=11 would need p_max ≥ 2048 which fails
            # even at the smallest DB (p_max=1983.6 at 2^18). Entries for
            # m_labels not currently in BENCH_CONFIGS / HEADTOHEAD_CONFIGS
            # are kept for documentation / future extension.
            case "$m_label" in
                "2^18"|"2^20"|"2^22"|\
                "3x2^17"|"3x2^18"|"3x2^19"|"3x2^20"|\
                "9x2^15"|"9x2^17"|"9x2^19") pb=10 ;;
                *) echo "[configs.sh] backend_plaintext_bits: unknown m_label '$m_label' for backend simple" >&2; return 1 ;;
            esac
            ;;
        *) echo "[configs.sh] backend_plaintext_bits: unknown backend '$backend'" >&2; return 1 ;;
    esac
    echo "$pb"
}

# Self-check: every m_label in BENCH_CONFIGS + HEADTOHEAD_CONFIGS must have
# a pb entry for every selected backend. Fail loudly at sourcing time rather
# than mid-sweep on a CSV-killing typo.
#
# Note: BENCH_CONFIGS m_label is field 5; HEADTOHEAD_CONFIGS m_label is
# field 5 too (after num_keys:arity:bucket_size:num_buckets).
_pb_self_check() {
    local cfg m_label backend
    for cfg in "${BENCH_CONFIGS[@]}" "${HEADTOHEAD_CONFIGS[@]}"; do
        m_label=$(echo "$cfg" | cut -d: -f5)
        for backend in "${BACKENDS_ARR[@]}"; do
            if ! backend_plaintext_bits "$backend" "$m_label" >/dev/null 2>&1; then
                echo "[configs.sh] ERROR: no plaintext_bits entry for (backend=$backend, m_label=$m_label)" >&2
                return 1
            fi
        done
    done
}
_pb_self_check || exit 1
unset -f _pb_self_check

echo "[configs.sh] backends=${IKPIR_BENCH_BACKENDS} | ${#BENCH_CONFIGS[@]} configs × ${#VALUE_BITS[@]} value_bits = $(( ${#BENCH_CONFIGS[@]} * ${#VALUE_BITS[@]} )) classical runs/bench/backend | mutation: ${#BENCH_CONFIGS[@]} configs × ${#VALUE_BITS[@]} value_bits × lf=${MUTATION_LOAD_FACTOR} × N=1% capacity = $(( ${#BENCH_CONFIGS[@]} * ${#VALUE_BITS[@]} )) runs/bench/backend | head-to-head: ${#HEADTOHEAD_CONFIGS[@]} configs × ${#VALUE_BITS[@]} value_bits = $(( ${#HEADTOHEAD_CONFIGS[@]} * ${#VALUE_BITS[@]} )) runs/backend | pb=max-per-(backend,m_label)"
