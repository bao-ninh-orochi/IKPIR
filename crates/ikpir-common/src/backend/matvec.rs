//! Width-adaptive register-blocked accumulate matvec — the shared
//! `acc += qᵀ·D mod 2³²` kernel behind the LWE hot loops.
//!
//! # Purpose
//!
//! Both shipped backends spend essentially all of their online CPU time
//! multiplying a vector against a row-major `u32` matrix and folding the
//! products into a row-wide accumulator: `server_answer` (query × DB),
//! `client_decode`'s cold path and `compute_c` (secret × hint). The
//! naive loop
//!
//! ```text
//! for i in rows { for j in width { acc[j] += q[i] · d[i][j] } }
//! ```
//!
//! touches the accumulator with one load **and one store per matrix
//! cell** (three memory ops per multiply-add) and pays the loop
//! preamble / vector tail once per row. ChalametPIR's reference kernel
//! — a contiguous dot product with the running sum in registers — does
//! two loads per cell and no stores, which is why its measured
//! per-cell throughput exceeded ours by ~25–60% at the paper's shapes.
//! This module closes that gap without explicit SIMD or threads.
//!
//! # Design / architecture
//!
//! - **Register blocking.** [`matvec_accumulate`] processes `R` rows per
//!   pass over the accumulator, so each `acc[j]` load/store and each
//!   loop preamble is amortised across `R` multiply-adds, and the `R`
//!   independent products give the autovectorizer ILP to hide multiply
//!   latency. The tail rows (`n mod R`) run one at a time.
//! - **Width-adaptive `R`.** `R` is the largest power of two `≤ 16`
//!   whose block footprint `R · width` stays within 2048 cells (8 KiB).
//!   Measured on Apple M1 (and consistent with Zen 2's 32 KiB L1 /
//!   4 KiB pages), larger footprints make the `R` parallel row streams
//!   fall off a prefetcher cliff — e.g. `R = 8` at `width = 832` runs
//!   2× *slower* than unblocked — while narrow rows need large `R` to
//!   amortise the per-row overhead (`R = 16` at `width = 29` is 2.7×
//!   faster than unblocked).
//! - **Bit-exact.** `u32` wrapping addition is associative and
//!   commutative, so regrouping the `i`-summation by blocks leaves every
//!   `acc[j]` bit-identical to the naive loop for any `R`. The unit
//!   tests pin this.
//! - **Constant-time.** No data-dependent branches or indices: the
//!   schedule depends only on the public shape `(n, width)`, never on
//!   the values of `q` or `d` — safe for secret-dependent inputs
//!   (`client_decode`, `compute_c`).
//!
//! # Related files
//!
//! - `frodo/backend.rs` / `simple/backend.rs` — the callers
//!   (`server_answer`, `client_decode` cold path, `compute_c`).
//! - Unlike `arith.rs` (tiny, duplicated per backend so each backend
//!   reads as a self-contained reference), this kernel is shared: it is
//!   backend-agnostic linear algebra, and its tuning tables should never
//!   diverge between backends.

/// Fold `qᵀ · D` into `acc` (all arithmetic mod `2³²`).
///
/// `D` is `d`, a row-major `q.len() × acc.len()` matrix: cell `(i, j)`
/// lives at `d[i * acc.len() + j]`.
///
/// # Arguments
///
/// - `acc` — accumulator, one cell per matrix column; updated in place
///   (`acc[j] += Σ_i q[i] · d[i][j]`). Callers wanting a plain product
///   pass a zeroed buffer.
/// - `d`   — row-major matrix; `d.len()` must equal
///   `q.len() * acc.len()`.
/// - `q`   — one coefficient per matrix row.
///
/// # Constraints
///
/// Panics (debug) if `d.len() != q.len() * acc.len()`. An empty `acc`
/// or `q` is a no-op.
///
/// # Complexity
///
/// `Θ(q.len() · acc.len())` wrapping multiply-adds; ~1 memory access
/// per cell once blocked (vs 3 for the naive loop).
pub(crate) fn matvec_accumulate(acc: &mut [u32], d: &[u32], q: &[u32]) {
    debug_assert_eq!(d.len(), q.len() * acc.len(), "matvec shape mismatch");
    // R = largest power of two ≤ 16 with R · width ≤ 2048 cells (8 KiB
    // block footprint) — see the module docs for the measured cliffs
    // that pin both ends of this rule.
    match acc.len() {
        0 => (),
        1..=128 => block_pass::<16>(acc, d, q),
        129..=256 => block_pass::<8>(acc, d, q),
        257..=512 => block_pass::<4>(acc, d, q),
        513..=1024 => block_pass::<2>(acc, d, q),
        _ => block_pass::<1>(acc, d, q),
    }
}

/// One monomorphised blocking level: fold `R` rows per pass over `acc`,
/// then the `q.len() mod R` tail rows one at a time.
///
/// # Rationale
///
/// `chunks_exact` hands the optimiser slices of compile-time-known
/// shape (`R · width` / `R`), so the inner indexing needs no bounds
/// checks and the `r`-loop fully unrolls (`R` is const).
#[inline]
fn block_pass<const R: usize>(acc: &mut [u32], d: &[u32], q: &[u32]) {
    let width = acc.len();
    let n = q.len();
    if width == 0 || n == 0 {
        return;
    }
    let full = n - n % R;

    let blocks = d[..full * width].chunks_exact(R * width);
    for (block, qs) in blocks.zip(q[..full].chunks_exact(R)) {
        for j in 0..width {
            let mut x = acc[j];
            for r in 0..R {
                x = x.wrapping_add(qs[r].wrapping_mul(block[r * width + j]));
            }
            acc[j] = x;
        }
    }

    // Tail rows: identical arithmetic, so the result matches the naive
    // loop bit-for-bit (u32 wrapping add is associative + commutative).
    for (row, &qi) in d[full * width..].chunks_exact(width).zip(&q[full..]) {
        for (x, &cell) in acc.iter_mut().zip(row) {
            *x = x.wrapping_add(qi.wrapping_mul(cell));
        }
    }
}

/// Fold `A · s` into `b` (all arithmetic mod `2³²`): one dot product
/// per matrix row, `b[i] += Σ_k a[i][k] · s[k]`.
///
/// # Purpose
///
/// The query-sampling hot loop (`sample_slot`'s `b = A·s + e`) in both
/// backends. The dual of [`matvec_accumulate`]: output is indexed by
/// *rows*, so the accumulators are per-row scalars rather than a
/// row-wide vector.
///
/// # Arguments
///
/// - `b` — accumulator, one cell per matrix row; updated in place.
/// - `a` — row-major `b.len() × s.len()` matrix.
/// - `s` — one coefficient per matrix column.
///
/// # Rationale
///
/// Size-adaptive row blocking, measured on Apple M1:
///
/// - **`A` cache-resident** (≤ 8 MiB, e.g. SimplePIR's reshaped
///   `~1365 × 1275`): interleaving 8 rows gives 8 independent
///   accumulator chains (ILP) and re-uses the L1-resident `s` stream —
///   ~1.3× over the single-accumulator dot.
/// - **`A` streaming from DRAM** (FrodoPIR's `16384 × 1566` = 102 MiB):
///   the loop is memory-bandwidth-bound (~13 GMAC/s ≈ 52 GB/s reads,
///   the single-core ceiling), and wide interleaving *hurts* (8 row
///   streams defeat the prefetcher, 2× slower); 2 rows matches the
///   plain dot while keeping a second chain for ISAs with spare
///   bandwidth.
///
/// # Constraints
///
/// Panics (debug) if `a.len() != b.len() * s.len()`. Empty `b` or `s`
/// is a no-op. Bit-exact vs the naive per-row dot (wrapping add is
/// associative + commutative); schedule depends only on public shapes.
///
/// # Complexity
///
/// `Θ(b.len() · s.len())` wrapping multiply-adds.
pub(crate) fn matvec_rows_accumulate(b: &mut [u32], a: &[u32], s: &[u32]) {
    debug_assert_eq!(a.len(), b.len() * s.len(), "matvec_rows shape mismatch");
    // 2 MiB of u32 cells = 8 MiB of A — the cache-residency threshold
    // between the ILP-bound and DRAM-bound regimes measured above.
    if a.len() <= 2_097_152 {
        rows_pass::<8>(b, a, s);
    } else {
        rows_pass::<2>(b, a, s);
    }
}

/// One monomorphised row-blocking level for [`matvec_rows_accumulate`]:
/// `R` interleaved dot products per pass, then the `b.len() mod R` tail
/// rows one at a time.
#[inline]
fn rows_pass<const R: usize>(b: &mut [u32], a: &[u32], s: &[u32]) {
    let n = s.len();
    if n == 0 || b.is_empty() {
        return;
    }
    let full = b.len() - b.len() % R;

    for (bc, ab) in b[..full]
        .chunks_exact_mut(R)
        .zip(a[..full * n].chunks_exact(R * n))
    {
        let mut acc = [0u32; R];
        for (k, &sk) in s.iter().enumerate() {
            for r in 0..R {
                acc[r] = acc[r].wrapping_add(ab[r * n + k].wrapping_mul(sk));
            }
        }
        for (bi, &t) in bc.iter_mut().zip(&acc) {
            *bi = bi.wrapping_add(t);
        }
    }

    // Tail rows: plain dot — same arithmetic, bit-identical result.
    for (bi, row) in b[full..].iter_mut().zip(a[full * n..].chunks_exact(n)) {
        let mut acc = *bi;
        for (&x, &y) in row.iter().zip(s) {
            acc = acc.wrapping_add(x.wrapping_mul(y));
        }
        *bi = acc;
    }
}

#[cfg(test)]
mod tests {
    //! Pins the two contracts: bit-exact agreement with the naive loop
    //! (every blocking level, tails included) and in-place accumulation.

    use super::*;

    /// Deterministic xorshift so failures reproduce without a rand dep.
    fn fill_pseudorandom(buf: &mut [u32], mut state: u32) {
        for cell in buf {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *cell = state;
        }
    }

    fn naive(acc: &mut [u32], d: &[u32], q: &[u32]) {
        let width = acc.len();
        for (i, &qi) in q.iter().enumerate() {
            for j in 0..width {
                acc[j] = acc[j].wrapping_add(qi.wrapping_mul(d[i * width + j]));
            }
        }
    }

    /// Every blocking level (widths straddling each `R` threshold) and
    /// every tail length agrees with the naive loop bit-for-bit.
    #[test]
    fn matches_naive_across_shapes() {
        // (n, width): widths pick each R in {16, 8, 4, 2, 1}; n values
        // exercise full blocks, tails, and n < R.
        let shapes = [
            (1usize, 1usize),
            (7, 3),
            (16, 29),
            (33, 128),
            (129, 129),
            (100, 517),
            (65, 1024),
            (5, 2049),
            (3, 208),
        ];
        for (n, width) in shapes {
            let mut d = vec![0u32; n * width];
            let mut q = vec![0u32; n];
            fill_pseudorandom(&mut d, 0xC0FF_EE00 ^ ((n as u32) << 12) ^ width as u32);
            fill_pseudorandom(&mut q, 0xBEEF_0000 | width as u32);

            let mut expected = vec![0u32; width];
            naive(&mut expected, &d, &q);
            let mut got = vec![0u32; width];
            matvec_accumulate(&mut got, &d, &q);
            assert_eq!(got, expected, "mismatch at n={n} width={width}");
        }
    }

    /// `matvec_accumulate` folds into a non-zero accumulator (the decode
    /// path relies on this to compute `residual − sᵀ·H` via negation).
    #[test]
    fn accumulates_in_place() {
        let (n, width) = (10, 40);
        let mut d = vec![0u32; n * width];
        let mut q = vec![0u32; n];
        fill_pseudorandom(&mut d, 0x1234_5678);
        fill_pseudorandom(&mut q, 0x9ABC_DEF0);

        let mut base = vec![0u32; width];
        fill_pseudorandom(&mut base, 0x0F0F_0F0F);

        let mut expected = base.clone();
        naive(&mut expected, &d, &q);
        let mut got = base;
        matvec_accumulate(&mut got, &d, &q);
        assert_eq!(got, expected);
    }

    /// Degenerate shapes are no-ops, not panics.
    #[test]
    fn empty_inputs_are_noops() {
        let mut acc: [u32; 0] = [];
        matvec_accumulate(&mut acc, &[], &[1, 2, 3]);

        let mut acc = [7u32, 8];
        matvec_accumulate(&mut acc, &[], &[]);
        assert_eq!(acc, [7, 8]);
    }

    fn naive_rows(b: &mut [u32], a: &[u32], s: &[u32]) {
        let n = s.len();
        for (bi, row) in b.iter_mut().zip(a.chunks_exact(n)) {
            for (&x, &y) in row.iter().zip(s) {
                *bi = bi.wrapping_add(x.wrapping_mul(y));
            }
        }
    }

    /// Every row-blocking level and tail length agrees with the naive
    /// per-row dot bit-for-bit, and folding starts from the incoming
    /// `b` (the `b = A·s + e` call site seeds `b` with the errors).
    #[test]
    fn rows_kernel_matches_naive_across_shapes() {
        // (rows, n): rows exercise full R=8 blocks, tails, rows < R;
        // n covers odd lengths. All shapes stay under the cache-size
        // threshold, so force the R=2 path explicitly too.
        let shapes = [
            (1usize, 1usize),
            (7, 3),
            (8, 29),
            (17, 128),
            (100, 517),
            (63, 1275),
        ];
        for (rows, n) in shapes {
            let mut a = vec![0u32; rows * n];
            let mut s = vec![0u32; n];
            fill_pseudorandom(&mut a, 0x5EED_0000 ^ ((rows as u32) << 10) ^ n as u32);
            fill_pseudorandom(&mut s, 0xFACE_0000 | n as u32);

            let mut base = vec![0u32; rows];
            fill_pseudorandom(&mut base, 0xE44_0E44);

            let mut expected = base.clone();
            naive_rows(&mut expected, &a, &s);

            let mut got = base.clone();
            matvec_rows_accumulate(&mut got, &a, &s);
            assert_eq!(got, expected, "adaptive mismatch at rows={rows} n={n}");

            // Both monomorphised levels, regardless of the size cutoff.
            let mut got8 = base.clone();
            rows_pass::<8>(&mut got8, &a, &s);
            assert_eq!(got8, expected, "R=8 mismatch at rows={rows} n={n}");
            let mut got2 = base;
            rows_pass::<2>(&mut got2, &a, &s);
            assert_eq!(got2, expected, "R=2 mismatch at rows={rows} n={n}");
        }
    }

    /// Degenerate shapes are no-ops, not panics.
    #[test]
    fn rows_kernel_empty_inputs_are_noops() {
        let mut b: [u32; 0] = [];
        matvec_rows_accumulate(&mut b, &[], &[1, 2]);

        let mut b = [3u32, 4];
        matvec_rows_accumulate(&mut b, &[], &[]);
        assert_eq!(b, [3, 4]);
    }
}
