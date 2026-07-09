# OPTIMIZATIONS.md — the `perf/optimized` branch

## What this branch is

`main` is deliberately **single-threaded and free of explicit SIMD /
hardware-specific tuning**: the CANS 2026 paper compares RisePIR against
other baselines under identical, unaccelerated conditions, so the
reference implementation must stay plain.

This branch answers the other question — *how fast does RisePIR go when
you let it use the machine?* It carries every CPU-level optimization
that applies to the implementation, each as its own commit so the
history documents exactly what each technique buys:

| # | Commit | Kind | What it does | Status |
|---|--------|------|--------------|--------|
| 1 | bench harness | infra | kernel-level timing harness (`benches/kernels.rs`) + this file | this commit |
| 2 | `target-cpu=native` | hardware | let the autovectorizer use the full local ISA (NEON / AVX2 / AVX-512) | planned |
| 3 | blocked `Aᵀ·D` | algorithm | cache/register-blocked matrix-matrix kernel for `compute_hint` (setup / rebuild) | planned |
| 4 | blocked `A·s` | algorithm | row-blocked matrix-vector kernel for query sampling | planned |
| 5 | explicit SIMD | hardware | explicit vector kernels where they beat autovectorization (measured) | planned |
| 6 | parallel kernels | parallel | rayon row-partitioned matvec + GEMM (`answer`, `decode`, `compute_c`, `compute_hint`) | planned |
| 7 | parallel segments | parallel | per-segment parallelism in `IkpirServer` / `IkpirClient` (setup, answer, patch) | planned |
| 8 | parallel `sample_a` | parallel | bit-identical chunked ChaCha20 expansion of the LWE matrix `A` | planned |
| 9 | parallel client paths | parallel | precompute queues, Phase-C maintenance, hint patches | planned |

Out of scope for now: GPU acceleration (deliberately deferred) and
async/pipelined I/O (matters for a deployed client↔server system, not
for the compute kernels this branch targets).

## Invariants every optimization must keep

1. **Bit-exact results.** All hot-loop arithmetic is `u32` wrapping
   add/mul mod 2³², which is associative and commutative — any
   regrouping (blocking, SIMD lanes, thread partitions) must leave every
   output word identical to the reference implementation. Unit tests pin
   this against the naive loops.
2. **Deterministic `A` expansion.** Server and client independently
   re-expand the LWE public matrix `A` from a 16-byte seed; a parallel
   expansion must produce a byte-identical stream (chunked
   `ChaCha20Rng::set_word_pos`, tested against the sequential stream).
3. **No secret-dependent schedules.** Query/decode-side kernels touch
   LWE secrets; blocking factors, lane counts, and thread splits may
   depend only on public shapes `(n_rows, row_width, lwe_dim)`, never on
   data values.
4. **Protocol surface unchanged.** Wire bundles, trait contracts, epochs
   and patch semantics are untouched; this branch is drop-in.

## Measuring

```bash
# Kernel-level, one op per line (min over fixed iters is the headline):
cargo bench -p ikpir-common --bench kernels -- --heavy

# End-to-end protocol benches (same harness as main):
./scripts/bench.sh server_answer --arity 4 --num-buckets 65536 --value-bits 256
./scripts/smoke.sh   # correctness smoke across all PIR benches
```

Baseline and per-commit results (Apple M1, 4P+4E, Rust 1.85.0) are
recorded in the table below as commits land.

## Results

*(filled in as optimization commits land; times are `min` from
`benches/kernels.rs` at `n_rows = 16384`, FrodoPIR `lwe_dim = 1566`,
SimplePIR `lwe_dim = 1275`, `plaintext_bits = 10`)*

| op | shape | baseline | after |
|----|-------|----------|-------|
| | | | |
