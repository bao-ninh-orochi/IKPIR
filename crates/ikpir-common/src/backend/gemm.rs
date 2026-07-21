//! Register-tiled `H += Aᵀ·D mod 2³²` — the setup-time matrix-matrix
//! kernel behind `compute_hint` in both backends.
//!
//! # Purpose
//!
//! `server_setup` / `full_rebuild` spend essentially all of their time
//! computing the hint `H[k, j] = Σ_i A[i, k] · D[i, j]`. The reference
//! loop nest (`i, k, j` — one rank-one update per DB row) streams the
//! whole `lwe_dim × width` hint matrix once per DB row: for the paper's
//! shapes that is 16384 passes over an `H` of 0.7–5 MB, so the loop is
//! bound on `H` cache traffic, not multiply throughput.
//!
//! # Design / architecture
//!
//! Goto/BLIS-style GEBP blocking with an `MR × NR = 4 × 16`
//! register-tiled micro-kernel and the contraction (DB-row) loop
//! innermost:
//!
//! 1. **`KC`-blocking.** The row dimension is split into blocks of `KC`
//!    rows (arch-tuned: 512 on aarch64's 128 KiB L1d, 256 on x86-64's
//!    typical 32 KiB) so each `KC × NR` `D`-slab stays L1-resident
//!    while every k-tile consumes it.
//! 2. **Packing.** The `KC × lwe_dim` `A`-block is transpose-packed
//!    once per block into `MR`-interleaved tiles (`Θ(KC · lwe_dim)` —
//!    negligible against the `Θ(KC · lwe_dim · width)` block work), so
//!    the micro-kernel reads `A` coefficients contiguously. The last
//!    tile is zero-padded when `lwe_dim % MR ≠ 0` (`0 · d = 0`, and the
//!    padded rows are simply never written back).
//! 3. **Micro-kernel.** An `MR × NR` grid of `u32` accumulators lives
//!    in registers across the whole `KC` contraction: per DB row, `NR`
//!    contiguous `D` cells and `MR` packed `A` coefficients feed
//!    `MR × NR` independent multiply-adds (NEON `MLA.4S` / AVX2
//!    `vpmulld` after autovectorization). `H` is touched exactly once
//!    per `(KC-block, tile)` — the accumulator traffic that bounded the
//!    reference loop disappears.
//!
//! Measured on Apple M1 against the previous panel-blocked kernel
//! (which itself was ~2× over the reference loop): 18.6 → 26.8 GMAC/s
//! at `width = 112` and 19.4 → 29.1 GMAC/s at `width = 832`
//! single-threaded.
//!
//! # Bit-exactness / constant-time
//!
//! `u32` wrapping addition is associative and commutative, so
//! regrouping the `i`-summation into blocks and register tiles leaves
//! every `H` word bit-identical to the reference loop — pinned by the
//! unit tests. The schedule depends only on the public shape
//! `(n_rows, lwe_dim, width)`, never on cell values.
//!
//! # Related files
//!
//! - `matvec.rs` — the vector-matrix sibling kernels for the online
//!   paths (`server_answer`, decode, query sampling).
//! - `frodo/backend.rs` / `simple/backend.rs` — the `compute_hint`
//!   callers (SimplePIR additionally handles its partial tail reshape
//!   row outside this kernel).

/// Micro-tile rows: `A`-coefficients (= `H` rows) per register tile.
const MR: usize = 4;
/// Micro-tile columns: contiguous `D` (= `H`) cells per register tile.
/// `MR × NR = 64` accumulators = 16 NEON / 8 AVX2 vector registers,
/// leaving room for the two operand loads.
const NR: usize = 16;
/// Contraction block: DB rows per pack-and-consume pass. The `KC × NR`
/// `D`-slab (`KC · 64` bytes) must stay L1-resident: 512 rows = 32 KiB
/// suits aarch64's 128 KiB L1d; x86-64 (Zen 2's 32 KiB L1d) gets 256.
#[cfg(target_arch = "x86_64")]
const KC: usize = 256;
#[cfg(not(target_arch = "x86_64"))]
const KC: usize = 512;

/// Fold `Aᵀ · D` into `h` (all arithmetic mod `2³²`).
///
/// # Arguments
///
/// - `h` — hint accumulator, row-major `lwe_dim × width`; updated in
///   place (`h[k, j] += Σ_i a[i, k] · d[i, j]`). Callers wanting a plain
///   product pass a zeroed buffer.
/// - `a` — row-major `n_rows × lwe_dim` matrix.
/// - `d` — row-major `n_rows × width` matrix (`n_rows` inferred from
///   `a.len() / lwe_dim`).
/// - `lwe_dim`, `width` — public shape; `width == 0` or an empty `a`
///   is a no-op.
///
/// # Constraints
///
/// Panics (debug) on inconsistent slice lengths.
///
/// # Complexity
///
/// `Θ(n_rows · lwe_dim · width)` wrapping multiply-adds — identical to
/// the reference loop; only the memory schedule changes.
pub(crate) fn gemm_at_d_accumulate(
    h: &mut [u32],
    a: &[u32],
    d: &[u32],
    lwe_dim: usize,
    width: usize,
) {
    if lwe_dim == 0 || width == 0 {
        return;
    }
    let n_rows = a.len() / lwe_dim;
    debug_assert_eq!(a.len(), n_rows * lwe_dim, "A shape mismatch");
    debug_assert_eq!(d.len(), n_rows * width, "D shape mismatch");
    debug_assert_eq!(h.len(), lwe_dim * width, "H shape mismatch");

    // Each k-tile owns MR consecutive H rows — one contiguous chunk of
    // `h` — so tiles are independent tasks with disjoint writes: the
    // parallel and sequential paths run the identical per-tile code and
    // produce bit-identical output (the tile schedule carries no
    // arithmetic).
    #[cfg(feature = "parallel")]
    if (n_rows as u64) * (lwe_dim as u64) * (width as u64) >= GEMM_PAR_MIN_MACS
        && crate::backend::parallel::kernels_parallel()
    {
        use rayon::prelude::*;
        let mut i0 = 0;
        while i0 < n_rows {
            let kc = KC.min(n_rows - i0);
            h.par_chunks_mut(MR * width)
                .enumerate()
                .for_each(|(kt, h_tile)| {
                    block_tile(h_tile, a, d, lwe_dim, width, i0..i0 + kc, kt * MR);
                });
            i0 += kc;
        }
        return;
    }

    let mut i0 = 0;
    while i0 < n_rows {
        let kc = KC.min(n_rows - i0);
        for (kt, h_tile) in h.chunks_mut(MR * width).enumerate() {
            block_tile(h_tile, a, d, lwe_dim, width, i0..i0 + kc, kt * MR);
        }
        i0 += kc;
    }
}

/// Minimum multiply-add count before the GEMM fans out to rayon: below
/// this the join overhead outweighs the win. Real segment shapes are
/// three orders of magnitude above it.
#[cfg(feature = "parallel")]
const GEMM_PAR_MIN_MACS: u64 = 1 << 24;

/// One `(KC-block, k-tile)` unit of work: transpose-pack the tile's
/// `A` coefficients for rows `i0 .. i0 + kc`, then fold the block into
/// the tile's `mr` H rows (`h_tile`, row stride `width`).
///
/// # Rationale
///
/// Packing per task (rather than once per KC-block) keeps tasks fully
/// independent for the parallel path; the pack is `Θ(kc · MR)` against
/// `Θ(kc · MR · width)` tile work, and the 8 KiB scratch allocation is
/// noise at that scale.
fn block_tile(
    h_tile: &mut [u32],
    a: &[u32],
    d: &[u32],
    lwe_dim: usize,
    width: usize,
    rows: core::ops::Range<usize>,
    k0: usize,
) {
    let mr = h_tile.len() / width;
    debug_assert!((1..=MR).contains(&mr));
    let (i0, kc) = (rows.start, rows.len());

    // at[i * MR + m] = A[i0 + i, k0 + m], zero-padded when mr < MR.
    let mut at = vec![0u32; kc * MR];
    for i in 0..kc {
        let a_off = (i0 + i) * lwe_dim + k0;
        for m in 0..mr {
            at[i * MR + m] = a[a_off + m];
        }
    }

    let j_full = width - width % NR;
    for j0 in (0..j_full).step_by(NR) {
        micro_full(h_tile, &at, d, i0, kc, j0, width);
    }
    if j_full < width {
        micro_tail(h_tile, &at, d, i0, kc, j_full, width);
    }
}

/// Full-width micro-kernel: one `MR × NR` register tile folded over
/// `kc` contraction rows, then written back to the tile's `mr` live
/// `H` rows (`mr < MR` only on the zero-padded final k-tile).
#[inline]
fn micro_full(
    h_tile: &mut [u32],
    at: &[u32],
    d: &[u32],
    i0: usize,
    kc: usize,
    j0: usize,
    width: usize,
) {
    let mut acc = [[0u32; NR]; MR];
    for i in 0..kc {
        let ar = &at[i * MR..i * MR + MR];
        let dv = &d[(i0 + i) * width + j0..(i0 + i) * width + j0 + NR];
        for m in 0..MR {
            let am = ar[m];
            for (accn, &dn) in acc[m].iter_mut().zip(dv) {
                *accn = accn.wrapping_add(am.wrapping_mul(dn));
            }
        }
    }
    let mr = h_tile.len() / width;
    for m in 0..mr {
        let h_row = &mut h_tile[m * width + j0..m * width + j0 + NR];
        for (hn, &accn) in h_row.iter_mut().zip(&acc[m]) {
            *hn = hn.wrapping_add(accn);
        }
    }
}

/// Width-tail micro-kernel: identical to [`micro_full`] with the column
/// count `width mod NR` decided at run time. Runs once per
/// `(KC-block, k-tile)`, so the variable inner bound costs nothing
/// measurable.
#[inline]
fn micro_tail(
    h_tile: &mut [u32],
    at: &[u32],
    d: &[u32],
    i0: usize,
    kc: usize,
    j0: usize,
    width: usize,
) {
    let jt = width - j0;
    debug_assert!(jt < NR);
    let mut acc = [[0u32; NR]; MR];
    for i in 0..kc {
        let ar = &at[i * MR..i * MR + MR];
        let dv = &d[(i0 + i) * width + j0..(i0 + i) * width + jt + j0];
        for m in 0..MR {
            let am = ar[m];
            for (accn, &dn) in acc[m][..jt].iter_mut().zip(dv) {
                *accn = accn.wrapping_add(am.wrapping_mul(dn));
            }
        }
    }
    let mr = h_tile.len() / width;
    for m in 0..mr {
        let h_row = &mut h_tile[m * width + j0..m * width + j0 + jt];
        for (hn, &accn) in h_row.iter_mut().zip(&acc[m][..jt]) {
            *hn = hn.wrapping_add(accn);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Pins the contract: bit-exact agreement with the reference
    //! `i, k, j` rank-one-update loop across block boundaries and all
    //! three tail dimensions (contraction, k-tile, width).

    use super::*;

    fn fill_pseudorandom(buf: &mut [u32], mut state: u32) {
        for cell in buf {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *cell = state;
        }
    }

    /// The reference loop from the pre-optimization `compute_hint`
    /// (including its `aik == 0` skip, which is a semantic no-op).
    fn naive(h: &mut [u32], a: &[u32], d: &[u32], lwe_dim: usize, width: usize) {
        let n_rows = a.len() / lwe_dim;
        for i in 0..n_rows {
            let a_row = &a[i * lwe_dim..(i + 1) * lwe_dim];
            let d_row = &d[i * width..(i + 1) * width];
            for (k, &aik) in a_row.iter().enumerate() {
                if aik == 0 {
                    continue;
                }
                let h_row = &mut h[k * width..(k + 1) * width];
                for (hj, &dj) in h_row.iter_mut().zip(d_row) {
                    *hj = hj.wrapping_add(aik.wrapping_mul(dj));
                }
            }
        }
    }

    /// Shapes straddle KC blocks (n = KC, n < KC, n mod KC ≠ 0), MR
    /// k-tails (lwe mod 4 ∈ {0,1,2,3}), NR width-tails (w mod 16 ≠ 0),
    /// and degenerate sizes.
    #[test]
    fn matches_naive_across_shapes() {
        let shapes = [
            // (n_rows, lwe_dim, width)
            (1usize, 1usize, 1usize),
            (3, 7, 5),
            (256, 33, 112), // full x86 KC block, k-tail 1, no j-tail
            (513, 34, 112), // KC block + tail rows on both arches
            (300, 64, 29),  // width < NR: pure j-tail tile
            (100, 16, 517), // j-tail 5
            (37, 48, 832),  // wide, no tails except contraction
            (600, 10, 40),  // contraction tail + k-tail 2 + j-tail 8
            (64, 8, 2049),  // j-tail 1
        ];
        for (n, lwe, w) in shapes {
            let mut a = vec![0u32; n * lwe];
            let mut d = vec![0u32; n * w];
            fill_pseudorandom(&mut a, 0xA11C_E500 ^ ((n as u32) << 8) ^ w as u32);
            fill_pseudorandom(&mut d, 0xD00D_0000 | lwe as u32);

            let mut expected = vec![0u32; lwe * w];
            naive(&mut expected, &a, &d, lwe, w);
            let mut got = vec![0u32; lwe * w];
            gemm_at_d_accumulate(&mut got, &a, &d, lwe, w);
            assert_eq!(got, expected, "mismatch at n={n} lwe={lwe} w={w}");
        }
    }

    /// Accumulates into a non-zero `h` (the contract callers rely on).
    #[test]
    fn accumulates_in_place() {
        let (n, lwe, w) = (50, 10, 40);
        let mut a = vec![0u32; n * lwe];
        let mut d = vec![0u32; n * w];
        fill_pseudorandom(&mut a, 0x1357_9BDF);
        fill_pseudorandom(&mut d, 0x2468_ACE0);

        let mut base = vec![0u32; lwe * w];
        fill_pseudorandom(&mut base, 0x0F1E_2D3C);

        let mut expected = base.clone();
        naive(&mut expected, &a, &d, lwe, w);
        let mut got = base;
        gemm_at_d_accumulate(&mut got, &a, &d, lwe, w);
        assert_eq!(got, expected);
    }

    /// Degenerate shapes are no-ops, not panics.
    #[test]
    fn empty_inputs_are_noops() {
        let mut h: [u32; 0] = [];
        gemm_at_d_accumulate(&mut h, &[], &[], 0, 0);
        gemm_at_d_accumulate(&mut h, &[], &[], 4, 0);

        let mut h = [1u32, 2, 3, 4];
        gemm_at_d_accumulate(&mut h, &[], &[], 2, 2);
        assert_eq!(h, [1, 2, 3, 4]);
    }

    /// A shape crossing `GEMM_PAR_MIN_MACS` exercises the rayon path
    /// end-to-end and must still be bit-identical to the reference
    /// loop (disjoint-tile partition carries no arithmetic).
    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_path_matches_naive() {
        let (n, lwe, w) = (2048usize, 66usize, 130usize); // 17.6M MACs ≥ gate
        assert!((n as u64) * (lwe as u64) * (w as u64) >= GEMM_PAR_MIN_MACS);
        let mut a = vec![0u32; n * lwe];
        let mut d = vec![0u32; n * w];
        fill_pseudorandom(&mut a, 0x7A7A_7A7A);
        fill_pseudorandom(&mut d, 0x5B5B_5B5B);

        let mut expected = vec![0u32; lwe * w];
        naive(&mut expected, &a, &d, lwe, w);
        let mut got = vec![0u32; lwe * w];
        gemm_at_d_accumulate(&mut got, &a, &d, lwe, w);
        assert_eq!(got, expected);
    }
}
