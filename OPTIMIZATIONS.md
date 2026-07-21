# OPTIMIZATIONS.md — the `perf/optimized` branch

## What this branch is

`main` is deliberately **single-threaded and free of explicit SIMD /
hardware-specific tuning**: the CANS 2026 paper compares RisePIR against
other baselines under identical, unaccelerated conditions, so the
reference implementation must stay plain.

This branch answers the other question — *how fast does RisePIR go when
you let it use the machine?* Everything upstream ships is here
(`perf/optimized` tracks `orochi-network/IKPIR` `main`, currently through
`8032a2c`); on top of it sit five techniques:

| # | Technique | Where |
|---|---|---|
| 1 | `-C target-cpu=native` — full local ISA for the autovectorizer | `.cargo/config.toml` |
| 2 | Register-tiled `Aᵀ·D` GEMM (`4 × 16` micro-kernel, KC-blocked, contraction innermost) | `backend/gemm.rs` → `compute_hint`, row-level patch |
| 3 | Size-adaptive row-blocked `A·s` for query sampling | `backend/matvec.rs::matvec_rows_accumulate` |
| 4 | Chunk-parallel ChaCha20 expansion of `A` (`set_word_pos`) | `backend/prg.rs` → both `sample_a`s |
| 5 | rayon behind the default-on `parallel` feature: row-partitioned matvec, tile-partitioned GEMM, banded hint patches, slot-parallel client precompute | everywhere above, plus both backends |

`benches/kernels.rs` (in `ikpir-common`) is the timing harness the
numbers below come from.

**The reference implementation is upstream, not a build flag here.**
Two ways to get `main`'s regime without leaving the branch:

- `IKPIR_SETUP_THREADS=1` — the crate's single worker-count knob, which
  `backend::parallel::kernels_parallel()` makes the rayon kernels honour
  too. Every optimized path degrades to its reference schedule; the
  algorithmic and codegen work (1–4) stays.
- `--no-default-features` — drops rayon entirely and restores upstream's
  scoped-thread `ParallelSetupBackend` implementation verbatim.

Out of scope for now: GPU acceleration (deliberately deferred) and
async/pipelined I/O (matters for a deployed client↔server system, not
for the compute kernels this branch targets).

## Invariants every optimization keeps

1. **Bit-exact results.** All hot-loop arithmetic is `u32` wrapping
   add/mul mod 2³², which is associative and commutative — any
   regrouping (blocking, register tiles, thread partitions) leaves every
   output word identical to the reference implementation. Where a
   partition splits the *output* (GEMM tiles, hint bands, keystream
   runs) it is bit-exact by construction, without even needing that
   argument. Unit tests pin every path against the naive loop, on both
   sides of every size gate.
2. **Deterministic `A` expansion.** Server and client independently
   re-expand the LWE public matrix `A` from a 16-byte seed; the parallel
   expansion produces a byte-identical stream (chunked
   `ChaCha20Rng::set_word_pos`, tested against the sequential stream).
3. **No secret-dependent schedules.** Blocking factors, chunk sizes, and
   thread splits depend only on public shapes
   `(n_rows, row_width, lwe_dim)` and the core count, never on data
   values.
4. **Protocol surface unchanged.** Wire bundles, trait contracts, epochs
   and patch semantics are untouched; both `HintPatchMode` realizations
   still produce bit-identical hints; `ParallelSetupBackend` and its
   protocol-level entry points (`IkpirServer::new_parallel`,
   `IkpirClient::from_setup_parallel`, …) still exist and still behave
   as documented. The branch is drop-in against `main`.

## Measuring

```bash
# Kernel-level, one op per line (min over fixed iters is the headline):
cargo bench -p ikpir-common --bench kernels -- --heavy

# End-to-end protocol benches (same harness, same CSV contract as main):
./scripts/bench.sh server_answer --arity 4 --num-buckets 65536 --value-bits 2048
./scripts/smoke.sh   # correctness smoke across all nine PIR benches

# main's regime, without leaving the branch:
IKPIR_SETUP_THREADS=1 cargo bench -p ikpir-common --bench kernels
cargo bench -p ikpir-common --no-default-features --bench kernels
```

## Results

**Machine:** Apple M1 (4 performance + 4 efficiency cores), 16 GB,
macOS 26.5.2, Rust 1.85.0 (the pinned toolchain), `-C target-cpu=native`.

**Baseline:** `orochi-network/IKPIR` `main` at `8032a2c`, in a clean
worktree, with `benches/kernels.rs` copied in unmodified (the harness
lives only on this branch; nothing else about the worktree was changed).
Both sides therefore run the same measurement contract — which matters,
because `6da0b7a` and `44b9eab` changed it, and the pre-merge numbers in
this file's previous revision were taken under the old one. They have
been discarded rather than carried forward.

**Protocol:** one segment of `n_rows = 16384`, `plaintext_bits = 10`,
FrodoPIR `lwe_dim = 1566` / SimplePIR `lwe_dim = 1275`; two runs per
side, per-op minimum across both (the `min` a single run reports is
already the least-noise sample; taking it across runs removes the rest).

The **seq** column is `--no-default-features` on this branch: techniques
1–4 with no threading, i.e. what the algorithmic and codegen work buys
on one core.

### Kernel level, `width = 112` (256 B values)

| op | main | seq | branch | seq × | branch × |
|----|-----:|----:|-------:|------:|---------:|
| frodo setup | 463.7 ms | 294.7 ms | **71.83 ms** | 1.57 | **6.46** |
| frodo expand_a | 180.6 ms | 175.0 ms | 38.35 ms | 1.03 | 4.71 |
| frodo query_cold | 2.169 ms | 2.105 ms | 1.953 ms | 1.03 | 1.11 |
| frodo answer | 125.0 µs | 126.4 µs | 47.67 µs | 0.99 | 2.62 |
| frodo decode_cold | 11.96 µs | 11.96 µs | 11.92 µs | 1.00 | 1.00 |
| frodo precompute B×16 | 36.68 ms | 35.99 ms | 19.89 ms | 1.02 | 1.84 |
| frodo precompute BC×16 | 36.99 ms | 36.21 ms | 19.63 ms | 1.02 | 1.88 |
| frodo patch_entry | 1.077 ms | 1.075 ms | 254.6 µs | 1.00 | 4.23 |
| frodo patch_row | 1.105 ms | 459.0 µs | 457.5 µs | 2.41 | 2.42 |
| simple setup | 411.7 ms | 91.63 ms | **21.34 ms** | 4.49 | **19.3** |
| simple expand_a | 12.18 ms | 11.84 ms | 2.666 ms | 1.03 | 4.57 |
| simple query_cold | 165.0 µs | 133.3 µs | 69.83 µs | 1.24 | 2.36 |
| simple answer | 144.1 µs | 144.7 µs | 61.96 µs | 1.00 | 2.33 |
| simple decode_cold | 135.0 µs | 139.6 µs | 57.33 µs | 0.97 | 2.35 |
| simple precompute B×16 | 2.651 ms | 2.140 ms | 488.2 µs | 1.24 | 5.43 |
| simple precompute BC×16 | 4.870 ms | 4.526 ms | 1.245 ms | 1.08 | 3.91 |
| simple patch_entry | 917.9 µs | 898.7 µs | 288.1 µs | 1.02 | 3.19 |
| simple patch_row | 10.92 ms | 3.564 ms | **877.2 µs** | 3.06 | **12.5** |

### Kernel level, `width = 832` (1 kB values)

| op | main | seq | branch | seq × | branch × |
|----|-----:|----:|-------:|------:|---------:|
| frodo setup | 2.317 s | 903.3 ms | **209.9 ms** | 2.57 | **11.0** |
| frodo expand_a | 181.1 ms | 175.2 ms | 39.26 ms | 1.03 | 4.61 |
| frodo query_cold | 2.186 ms | 2.117 ms | 1.953 ms | 1.03 | 1.12 |
| frodo answer | 1.012 ms | 1.001 ms | 900.4 µs | 1.01 | 1.12 |
| frodo decode_cold | 89.50 µs | 89.33 µs | 43.63 µs | 1.00 | 2.05 |
| frodo precompute B×16 | 38.30 ms | 36.43 ms | 20.15 ms | 1.05 | 1.90 |
| frodo precompute BC×16 | 39.20 ms | 37.76 ms | 19.79 ms | 1.04 | 1.98 |
| frodo patch_entry | 1.082 ms | 1.084 ms | 258.0 µs | 1.00 | 4.19 |
| frodo patch_row | 8.264 ms | 2.706 ms | **690.5 µs** | 3.05 | **12.0** |
| simple setup | 5.291 s | 632.5 ms | **180.8 ms** | 8.37 | **29.3** |
| simple expand_a | 37.09 ms | 35.61 ms | 7.938 ms | 1.04 | 4.67 |
| simple query_cold | 502.4 µs | 454.9 µs | 382.8 µs | 1.10 | 1.31 |
| simple answer | 1.085 ms | 1.078 ms | 927.2 µs | 1.01 | 1.17 |
| simple decode_cold | 346.1 µs | 347.1 µs | 199.0 µs | 1.00 | 1.74 |
| simple precompute B×16 | 8.542 ms | 7.724 ms | 2.726 ms | 1.11 | 3.13 |
| simple precompute BC×16 | 14.09 ms | 13.40 ms | 4.525 ms | 1.05 | 3.11 |
| simple patch_entry | 952.7 µs | 932.5 µs | 290.3 µs | 1.02 | 3.28 |
| simple patch_row | 28.48 ms | 10.20 ms | **2.257 ms** | 2.79 | **12.6** |

### Reading the flat rows

They are not failures, they are the memory wall — and where the wall
sits is legible in the numbers.

- **`frodo answer` is 2.62× at `width = 112` and 1.12× at `width = 832`.**
  The narrow segment is 7.3 MB, which fits M1's 8 MB system-level cache;
  the wide one is 54 MB, which does not. Once the pass is streaming from
  DRAM, threads cannot add bandwidth the machine does not have. On a
  server-class part with more memory channels — the paper's EPYC — the
  parallel answer path has real headroom this laptop cannot show.
- **`frodo query_cold` barely moves** (1.11–1.12×): `A` is
  `16384 × 1566` = 102 MB, streamed once per sampled slot. Same wall.
- **`frodo decode_cold` at `width = 112` is exactly `main`.** The hint is
  `1566 × 112` = 175 k cells, under the 2²⁰-cell fan-out gate, so it runs
  the identical sequential kernel — which is the right call: fork/join
  would cost more than the 12 µs pass.

  There is one honest caveat here. On a heterogeneous machine, a
  sub-gate sequential kernel measured *right after* a parallel one is
  sometimes 25–40% slower (14–17 µs rather than 12 µs), because the
  calling thread can land on an efficiency core once rayon's workers
  have occupied every core. It is a scheduler artifact, not a code-path
  change: `RAYON_NUM_THREADS=1` on the same binary reproduces `main`'s
  12.0 µs exactly. Taking the minimum across runs recovers the
  performance-core number, which is what the table reports. Expect this
  to disappear on a homogeneous server part.

### Where the big wins come from

- **Setup, 6.5–29×.** Two independent factors: the register-tiled GEMM
  (the **seq** column — 1.57–8.37× on its own, biggest for SimplePIR
  where the reshape lets the whole segment go through one dense product)
  and threading on top. SimplePIR's 29.3× at `width = 832` is the
  headline number of the branch.
- **Row-level hint patch, 2.4–12.6×.** Upstream's `5201dbd` made the
  pass stream through one dense buffer instead of materialising the
  whole grouping (~70 MB → ~31 kB); this branch batches the densified
  rows into **chunks** and fires one GEMM per chunk, so the hint is
  streamed once per chunk rather than once per touched row — while a
  cap of 8 MiB on `Δ` plus the gathered `A` rows keeps upstream's memory
  win. Neither alone gets here: the batching is the speed, the cap is
  the memory. A chunk shallower than `2 · MR` rows keeps the reference
  rank-one pass — with one row there is no contraction for the tiles to
  amortise, and routing a single-mutation patch through the GEMM
  measured 0.69–0.80× `main` before that gate went in.
- **Entry-level hint patch, 3.2–4.2×.** Entirely threading: the **seq**
  column is 1.00, because sequential entry-level *is* upstream's
  `TouchedRuns` loop inversion, unchanged. The branch's contribution is
  splitting that single sweep across bands of hint rows, which composes
  with the inversion rather than replacing it.
- **`expand_a`, 4.6–4.7×.** Pure ChaCha20 throughput, and the one
  kernel that is neither memory- nor cache-bound, so it scales close to
  core count until the write bandwidth for `A` caps it.

### End-to-end

Same machine, `./scripts/bench.sh <bench> --arity 4 --num-buckets 65536
--bucket-size 4 --value-bits 2048`, both backends, against the same
`main` worktree. Dev scale (2¹⁸ slots), not the paper's geometry.

| bench (metric) | backend | main | branch | × |
|---|---|---:|---:|---:|
| `server_setup` (ms, lower better) | frodo | 8976 | **850.1** | **10.6** |
| `server_setup` (ms) | simple | 24114 | **856.8** | **28.1** |
| `server_answer` (queries/s) | frodo | 256.0 | 232.6 | 0.91 |
| `server_answer` (queries/s) | simple | 189.5 | 191.2 | 1.01 |
| `client_query` (queries/s) | frodo | 103.6 k | 94.3 k | 0.91 |
| `client_query` (queries/s) | simple | 299.3 k | 259.3 k | 0.87 |
| `client_decode` (decodes/s) | frodo | 82.0 k | 78.4 k | 0.96 |
| `client_decode` (decodes/s) | simple | 57.4 k | 54.8 k | 0.95 |
| `server_mutation` entry (ops/s) | frodo | 7412 / 13075 / 13197 | 7857 / 12622 / 12752 | 1.06 / 0.97 / 0.97 |
| `server_mutation` row (ops/s) | frodo | 4647 / 7287 / 7234 | 4533 / 7270 / 7267 | 0.98 / 1.00 / 1.00 |
| `server_mutation` entry (ops/s) | simple | 3402 / 5843 / 5655 | 3670 / 5690 / 5736 | 1.08 / 0.97 / 1.01 |
| `server_mutation` row (ops/s) | simple | 883 / 1070 / 1583 | 970 / 1594 / 1570 | 1.10 / 1.49 / 0.99 |
| `client_mutation` entry (Δ/s) | frodo | 8112 / 15331 / 15251 | 8113 / 14554 / 13726 | 1.00 / 0.95 / 0.90 |
| `client_mutation` row (Δ/s) | frodo | 4867 / 7758 / 7739 | 4855 / 7724 / 7690 | 1.00 / 1.00 / 0.99 |
| `client_mutation` entry (Δ/s) | simple | 3599 / 6169 / 6131 | 3993 / 6073 / 6141 | 1.11 / 0.98 / 1.00 |
| `client_mutation` row (Δ/s) | simple | 984 / 1623 / 1625 | 1011 / 1637 / 1603 | 1.03 / 1.01 / 0.99 |

Mutation rows are `insert / update / delete`.

Two things this table says, and neither is a disappointment:

- **Setup is where the branch lives.** 10.6× and 28.1× on a whole
  `IkpirServer::new` over four segments — 24 seconds down to 0.86 for
  RisePIR-S. That is the operation whose cost forced the paper's
  geometry down from `N = 2²²` to `2²⁰` in the first place.
- **Everything online is flat, within 0.87–1.01×.** At this geometry the
  per-segment matrix is 12.4 M cells (50 MB), four of them per query;
  the pass is bandwidth-bound on a laptop with one memory controller, so
  threads add coordination and no throughput. The 5–13% shortfall is
  that coordination. The same kernels show 2.3–2.6× at the cache-resident
  `width = 112` shape above, which is the regime a server with more
  memory channels would extend.
- **The mutation benches never reach the branch's mutation work.** They
  apply one mutation per call, so `row_deltas.len() == 1` and every path
  falls back to the reference kernels by design (see
  `patch::ROW_LEVEL_MIN_GEMM_ROWS`). Parity is the correct outcome here;
  the 2.4–12.6× the batched hint patches buy shows up only against a
  burst, which is what `benches/kernels.rs` fires and what a batched
  deployment produces.

## Design decisions (measured, not assumed)

- **No explicit SIMD.** The register-tiled kernels autovectorize: LLVM
  emits NEON `mla.4s` on Apple Silicon and AVX2 `vpmulld` on x86-64 with
  `target-cpu=native` (single-uop on Zen 2, the paper's bench machine).
  The fastest published LWE-PIR implementations (SimplePIR, YPIR) reach
  near-DRAM-bandwidth the same way — plain code plus `-march=native`, no
  intrinsics. A `wide`/intrinsic port would add a dependency and a
  constant-time review surface to replicate codegen we already get.
- **rayon for the online kernels, `std::thread::scope` for setup-only
  fan-out.** Upstream's `backend/parallel.rs` spawns a thread per chunk,
  which is free for a kernel called a handful of times per process and
  ruinous for one called per query: at the 2²⁰-cell gate a matvec is a
  couple of hundred microseconds, and eight `spawn`s on macOS cost about
  as much. rayon's persistent pool is the right tool there. Both
  mechanisms read one worker count and one chunk rule, so
  `IKPIR_SETUP_THREADS` still means what its documentation says.
- **`ParallelSetupBackend`'s twins delegate rather than nest.** With the
  `parallel` feature on, `server_setup` already *is* the optimized path,
  so `server_setup_parallel` calls it. Banding it a second time with
  scoped threads would put a rayon fan-out inside every band and
  oversubscribe the machine. The trait stays because the API is public
  and a downstream consumer pins it; only the implementation collapses.
- **No protocol-level (per-segment) parallelism.** With kernels
  parallel, the answer path is DRAM-bound and setup already saturates
  every core inside each segment; parallelising the `arity ≤ 4` segment
  loop on top measured as noise, and it would force `Send`/`Sync` bounds
  onto the public `IndexPirBackend` associated types. Not worth the API
  churn.
- **Adaptive gates everywhere, and they are load-bearing.** Every
  optimized path sits behind a size gate — ~1 M cells for matvec, 2²⁴
  MACs for the GEMM, 2¹⁹ MACs for hint patches and for Phase-C slot
  maintenance, 2¹⁸ words for the keystream, 8 rows for the row-level
  GEMM — so tiny inputs keep their low-latency reference path. Two of
  those exist because the end-to-end benches caught the branch losing
  without them: the row-level GEMM at one touched row (0.69–0.80×), and
  `patch_slot_c`'s fan-out, which the pre-merge branch gated on *slot
  count* rather than work, so a single-mutation delta against a warm
  16-slot queue was ~25 k multiply-adds spread over eight threads
  (0.69–0.90×). Both now gate on the work, and both are back at parity.
  Each gate has a unit test that crosses it, asserted, so a gate cannot
  silently stop pinning anything.

## What is deliberately unmeasured

Stated plainly, because these numbers end up in a paper.

- **The x86-64 matvec dispatch table is taken on upstream's evidence,
  not re-measured here.** `8b5e0d1` / `7e28c25` replaced the M1
  footprint ladder with an arch-gated table — `width ≤ 128 → R = 16`,
  `129..=4096 → R = 1`, `> 4096 → R = 8` — measured on Zen 4
  (EPYC 9R14, `r7a.xlarge`) and spot-validated on Zen 2 (EPYC 7R32,
  `c5a`), which is the paper's bench machine. This branch adopts it
  verbatim. **It cannot be checked here**: the gate is
  `#[cfg(target_arch = "x86_64")]` and this is an M1, where it is a
  no-op. Every number in this file is from the `aarch64` arm.
- **Nor is that table verified under this branch's threading.**
  Upstream's crossovers were measured single-threaded; here the kernel
  runs once per rayon task, so `N` threads each drive `R` streams. The
  cliffs are properties of the per-core L1 and the per-page prefetchers,
  so the shape should carry over, but the exact boundaries under
  concurrency are an open question. Anyone with a Zen 4 or Zen 2 box
  should re-run the wide-arm sweep in `matvec.rs`'s `# Tuning` section
  with the `parallel` feature on before trusting `R = 8` at scale.
- **No paper-scale run.** Everything here is one segment of 16384 rows,
  or the dev-scale (2¹⁸-slot) end-to-end config. The paper's matrix is
  2²⁰ slots; `scripts/table{3,4,5}.sh` drive it, and nobody has run them
  on this branch.
- **No power or thermal control.** A laptop under a several-minute bench
  throttles. Minimum-over-runs mitigates it; it does not eliminate it.

## Deferred ideas (next candidates for this branch)

None of these landed upstream in the sync, and none is started here.

- **`k`-outermost batch ordering for the entry-level patch.** The
  current sweep is per mutated row, `k` inside — upstream's ordering,
  with bands layered on. Inverting it once more (one hint row at a time,
  every mutated row inside) would keep a single 44 kB hint row hot
  across the whole batch instead of streaming the multi-megabyte hint
  once per row; at the (4, 1) paper config that is ~11 GB of traffic
  against ~112 MB. The catch is `A`: reading `A[row_r, k]` for thousands
  of `r` at fixed `k` is a scattered walk over a gigabyte-scale matrix,
  so it needs a transposed gather of the touched rows (`t × lwe_dim`,
  ~13 MB) first. Worth an experiment, and unmeasurable at dev scale
  where the hint already fits in cache — which is why it is deferred
  rather than done.
- **Packed DB cells** (SimplePIR packs 3×10-bit cells per `u32`, YPIR
  4×8-bit): ~3× less DRAM traffic on the bandwidth-bound answer pass —
  the single highest-leverage remaining optimization, and the one the
  `width = 832` rows above are begging for. It changes the cell layout
  that `segmented-cuckoo` owns and every mutation path touches. Needs
  its own design pass.
- **Multi-query batching** (YPIR-style 8-query register blocks): near-8×
  answer throughput when a server holds several outstanding queries;
  needs an `answer_batch` API.
- **An arch-gated table for `matvec_rows_accumulate`.** Its `R = 8` /
  `R = 2` cutover was measured on M1 only, and it is the same class of
  object as the `matvec_accumulate` table upstream had to split per
  microarchitecture. Nobody has looked at it on x86-64.
- **GPU acceleration**: deliberately deferred per project direction.
- **Transparent huge pages** on the Linux bench machine
  (`madvise(MADV_HUGEPAGE)` for the 54–102 MB `A`/DB buffers; Zen 2's
  4K dTLB covers only 12 MB).
