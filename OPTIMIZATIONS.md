# OPTIMIZATIONS.md — the `perf/optimized` branch

## What this branch is

`main` is deliberately **single-threaded and free of explicit SIMD /
hardware-specific tuning**: the CANS 2026 paper compares RisePIR against
other baselines under identical, unaccelerated conditions, so the
reference implementation must stay plain.

This branch answers the other question — *how fast does RisePIR go when
you let it use the machine?* Every applicable CPU-level optimization,
one commit per technique, so the history documents exactly what each
buys:

| # | Commit | Kind | What it does |
|---|--------|------|--------------|
| 1 | `bench(ikpir-common)` kernel harness | infra | `benches/kernels.rs` timing harness + this file; every later commit cites its before/after numbers |
| 2 | `perf(hw)` target-cpu=native | hardware | full local ISA for the autovectorizer (no-op on M1 whose NEON is the aarch64 default; unlocks AVX2/AVX-512 on x86-64) |
| 3 | `perf(algo)` panel-blocked `Aᵀ·D` | algorithm | GEBP panel blocking for `compute_hint` — H streamed once per panel instead of once per DB row |
| 4 | `perf(algo)` row-blocked `A·s` | algorithm | size-adaptive interleaved dot products for query sampling |
| 5 | `perf(algo)` register-tiled GEMM | algorithm | 4×16 register micro-kernel, contraction innermost, KC-blocked — supersedes #3's inner loop (~29 GMAC/s single-thread on M1) |
| 6 | `perf(parallel)` rayon kernels | parallel | `parallel` cargo feature (default-on): row-partitioned matvec + tile-partitioned GEMM |
| 7 | `perf(parallel)` chunked ChaCha20 | parallel | byte-identical chunk-parallel expansion of the LWE matrix `A` via `set_word_pos` |
| 8 | `perf(parallel)` hint patches | parallel | row-level patches batched into one GEMM; entry-level patches banded across tasks; parallel Phase-C maintenance |
| 9 | `perf(parallel)` client precompute | parallel | Phase-B/Phase-C slot batches fan out across tasks |

Out of scope for now: GPU acceleration (deliberately deferred) and
async/pipelined I/O (matters for a deployed client↔server system, not
for the compute kernels this branch targets).

## Invariants every optimization keeps

1. **Bit-exact results.** All hot-loop arithmetic is `u32` wrapping
   add/mul mod 2³², which is associative and commutative — any
   regrouping (blocking, SIMD lanes, thread partitions) leaves every
   output word identical to the reference implementation. Unit tests pin
   this against the naive loops.
2. **Deterministic `A` expansion.** Server and client independently
   re-expand the LWE public matrix `A` from a 16-byte seed; the parallel
   expansion produces a byte-identical stream (chunked
   `ChaCha20Rng::set_word_pos`, tested against the sequential stream).
3. **No secret-dependent schedules.** Blocking factors, chunk sizes, and
   thread splits depend only on public shapes
   `(n_rows, row_width, lwe_dim)`, never on data values.
4. **Protocol surface unchanged.** Wire bundles, trait contracts, epochs
   and patch semantics are untouched; both `HintPatchMode` realizations
   still produce bit-identical hints. The branch is drop-in.

Disable all threading with `--no-default-features` (the `parallel`
feature is default-on here); the algorithmic and codegen improvements
remain.

## Results

Apple M1 (4P + 4E), Rust 1.85.0. Kernel level: `min` over fixed iters
from `cargo bench -p ikpir-common --bench kernels -- --heavy`, one
segment of `n_rows = 16384`, `plaintext_bits = 10`, FrodoPIR
`lwe_dim = 1566` / SimplePIR `lwe_dim = 1275`.

### Kernel level, `width = 112` (256-bit values)

| op | main | this branch | speedup |
|----|-----:|------------:|--------:|
| frodo setup (per segment) | 509.7 ms | 73.1 ms | 7.0× |
| frodo expand_a | 223.1 ms | 39.2 ms | 5.7× |
| frodo query_cold | 2.12 ms | 2.02 ms | ~1× (DRAM-bound) |
| frodo answer | 128.0 µs | 52.0 µs | 2.5× |
| frodo patch_entry (64×8 burst) | 625.2 µs | 81.3 µs | 7.7× |
| frodo patch_row (64×8 burst) | 1.10 ms | 455.4 µs | 2.4× |
| frodo precompute 16 queries | ~33.9 ms | 22.2 ms | 1.5× (DRAM-bound) |
| simple setup (per segment) | 402.5 ms | 21.4 ms | 18.8× |
| simple query_cold | 167.2 µs | 75.3 µs | 2.2× |
| simple answer | 146.0 µs | 61.0 µs | 2.4× |
| simple decode_cold | 136.8 µs | 59.7 µs | 2.3× |
| simple patch_entry | 763.0 µs | 146.4 µs | 5.2× |
| simple patch_row | 10.90 ms | 847.9 µs | 12.9× |
| simple precompute 16 queries | ~2.7 ms | 552.2 µs | 4.8× |

### Kernel level, `width = 832` (wide values)

| op | main | this branch | speedup |
|----|-----:|------------:|--------:|
| frodo setup | 2.360 s | 203.4 ms | 11.6× |
| frodo answer | 992.8 µs | 922.9 µs | ~1× (DRAM-bound) |
| frodo patch_row | 8.18 ms | 676.3 µs | 12.1× |
| simple setup | 3.527 s | 178.1 ms | 19.8× |
| simple decode_cold | 346.7 µs | 213.0 µs | 1.6× |
| simple patch_row | 27.43 ms | 2.18 ms | 12.6× |

### End-to-end (`scripts/bench.sh`, arity 4, 65536 buckets, 256-bit values)

| bench | main | this branch | speedup |
|-------|-----:|------------:|--------:|
| server_setup (full, estimate) — frodo | 2243 ms | 357 ms | 6.3× |
| server_setup — simple | 1347 ms | 86 ms | 15.7× |
| server_answer — frodo | 675 q/s | 1665 q/s | 2.5× |
| server_answer — simple | 1578 q/s | 1515 q/s | ~1× |
| client_query (warm) — frodo | 95.8 k/s | 106.4 k/s | ~1× |
| client_decode (warm) — simple | 101.7 k/s | 179.0 k/s | 1.8× |

The flat rows are not failures — they are the memory wall: a warm-queue
query/decode was always a queue pop plus a subtract, and SimplePIR's
answer pass on `main` already streamed the 7.3 MB segment at ~46 GB/s,
which is a single M1 core's practical DRAM ceiling. Threads cannot add
bandwidth this machine doesn't have; on a server-class part with more
memory channels (the paper's EPYC), the parallel answer path has real
headroom.

## Design decisions (measured, not assumed)

- **No explicit SIMD.** The register-tiled kernels autovectorize: LLVM
  emits NEON `mla.4s` on Apple Silicon (verified in the disassembly)
  and AVX2 `vpmulld` on x86-64 with `target-cpu=native` (single-uop on
  Zen 2, the paper's bench machine). The fastest published LWE-PIR
  implementations (SimplePIR, YPIR) reach near-DRAM-bandwidth the same
  way — plain code + `-march=native`, no intrinsics. A `wide`/intrinsic
  port would add a dependency and constant-time review surface to
  replicate codegen we already get.
- **No protocol-level (per-segment) parallelism.** With kernels
  parallel, the answer path is DRAM-bound and setup saturates all cores
  inside each segment; parallelizing the `arity ≤ 4` segment loop on
  top measured as noise on M1, and it would force `Send`/`Sync` bounds
  onto the public `IndexPirBackend` associated types. Not worth the API
  churn.
- **Adaptive gates everywhere.** Every parallel path sits behind a
  size gate (`~1M cells` matvec, `2²⁴` MACs GEMM, `2¹⁹` MACs patches)
  so tiny inputs — single-mutation patches, FrodoPIR's narrow decode —
  keep their low-latency sequential path.

## Deferred ideas (next candidates for this branch)

- **Packed DB cells** (SimplePIR packs 3×10-bit cells per `u32`, YPIR
  4×8-bit): ~3× less DRAM traffic on the bandwidth-bound answer pass —
  the single highest-leverage remaining optimization, but it changes
  the cell layout that `segmented-cuckoo` owns and every mutation path
  touches. Needs its own design pass.
- **Multi-query batching** (YPIR-style 8-query register blocks): near-8×
  answer throughput when a server holds several outstanding queries;
  needs an `answer_batch` API.
- **GPU acceleration**: deliberately deferred per project direction.
- **Transparent huge pages** on the Linux bench machine
  (`madvise(MADV_HUGEPAGE)` for the 54–102 MB `A`/DB buffers; Zen 2's
  4K dTLB covers only 12 MB).

## Measuring

```bash
# Kernel-level, one op per line (min over fixed iters is the headline):
cargo bench -p ikpir-common --bench kernels -- --heavy

# End-to-end protocol benches (same harness as main):
./scripts/bench.sh server_answer --arity 4 --num-buckets 65536 --value-bits 256
./scripts/smoke.sh   # correctness smoke across all PIR benches

# Single-threaded (algorithmic + codegen improvements only):
cargo bench -p ikpir-common --no-default-features --bench kernels
```
