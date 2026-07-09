//! Panel-blocked `H += Aᵀ·D mod 2³²` — the setup-time matrix-matrix
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
//! Classic GEBP-style panel blocking, reusing the width-adaptive
//! register-blocked [`matvec_accumulate`] as the micro-kernel:
//!
//! 1. Split the row dimension into panels of `P` rows, `P` chosen so the
//!    `P × width` `D`-panel fits comfortably in L1 (≤ 16384 cells =
//!    64 KiB, capped at 128 rows).
//! 2. Transpose-pack the matching `P × lwe_dim` `A`-panel into a
//!    column-major scratch (`Θ(P · lwe_dim)` — negligible against the
//!    `Θ(P · lwe_dim · width)` panel work) so each hint row's
//!    coefficients are contiguous.
//! 3. For each hint row `k`, fold the panel with
//!    `matvec_accumulate(H[k, ·], D_panel, Aᵀ[k, ·])`.
//!
//! Per panel, `H` is streamed exactly once, so total `H` traffic drops
//! from `n_rows` passes to `n_rows / P` passes (≥ 100× less at the
//! paper's shapes); `D`-panel reads come from L1, and the micro-kernel's
//! register blocking amortises the `H` load/stores further.
//!
//! The reference loop's `aik == 0` skip is dropped: `A` is sampled
//! uniformly over `Z_{2³²}` via ChaCha20, so the branch fires with
//! probability 2⁻³² per cell — dead weight in the hot loop, and adding
//! `0 · d` is arithmetically identical anyway.
//!
//! # Bit-exactness / constant-time
//!
//! `u32` wrapping addition is associative and commutative, so
//! regrouping the `i`-summation into panels (and the micro-kernel's
//! blocks) leaves every `H` word bit-identical to the reference loop —
//! pinned by the unit tests. The schedule depends only on the public
//! shape `(n_rows, lwe_dim, width)`, never on cell values.
//!
//! # Related files
//!
//! - `matvec.rs` — the shared micro-kernel; its tuning tables carry the
//!   per-width register blocking this kernel inherits.
//! - `frodo/backend.rs` / `simple/backend.rs` — the `compute_hint`
//!   callers (SimplePIR additionally handles its partial tail reshape
//!   row outside this kernel).

use crate::backend::matvec::matvec_accumulate;

/// Rows per panel: the largest power of two ≤ 128 whose `D`-panel
/// footprint `P · width` stays within 16384 cells (64 KiB).
///
/// # Rationale
///
/// The micro-kernel re-reads the `D`-panel once per hint row
/// (`lwe_dim` times), so the panel must sit in L1; 64 KiB leaves room
/// for the `H` row and `Aᵀ` traffic beside it. The 128-row cap bounds
/// the transpose scratch and keeps the tail cheap; below 4 rows the
/// panel no longer amortises anything, so wide rows floor there.
fn panel_rows(width: usize) -> usize {
    let mut p = 128;
    while p > 4 && p * width > 16_384 {
        p /= 2;
    }
    p
}

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

    let p = panel_rows(width);
    // Transpose-pack scratch for one A-panel: at[k * rows + r] = A[i0 + r, k].
    let mut at = vec![0u32; p * lwe_dim];

    let mut i0 = 0;
    while i0 < n_rows {
        let rows = p.min(n_rows - i0);
        for r in 0..rows {
            let a_row = &a[(i0 + r) * lwe_dim..(i0 + r + 1) * lwe_dim];
            for (k, &v) in a_row.iter().enumerate() {
                at[k * rows + r] = v;
            }
        }
        let d_panel = &d[i0 * width..(i0 + rows) * width];
        for (k, h_row) in h.chunks_exact_mut(width).enumerate() {
            matvec_accumulate(h_row, d_panel, &at[k * rows..(k + 1) * rows]);
        }
        i0 += rows;
    }
}

#[cfg(test)]
mod tests {
    //! Pins the contract: bit-exact agreement with the reference
    //! `i, k, j` rank-one-update loop across panel boundaries and tails.

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

    /// Shapes straddle panel boundaries (n = P, n < P, n mod P ≠ 0),
    /// micro-kernel blocking levels (widths across the R thresholds),
    /// and degenerate sizes.
    #[test]
    fn matches_naive_across_shapes() {
        let shapes = [
            // (n_rows, lwe_dim, width)
            (1usize, 1usize, 1usize),
            (3, 7, 5),
            (128, 33, 112), // exactly one full panel at width 112
            (129, 33, 112), // one panel + 1-row tail
            (300, 64, 29),  // narrow width → R = 16 micro-kernel
            (100, 16, 517), // width in the R = 2 band, P = 16
            (37, 48, 832),  // wide width → P = 16, partial panel
            (256, 8, 2049), // width past the last blocking threshold
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
}
