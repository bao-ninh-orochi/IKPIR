//! Operating-point selection: the largest plaintext modulus each LWE
//! backend supports at `q = 2³²` without breaking decode correctness.
//!
//! # Purpose
//!
//! Both shipped backends decode by rounding `ans − s·H = e·D + Δ·D[r]`
//! to the nearest multiple of `Δ = q/p`; decoding a cell is correct iff
//! the accumulated noise coordinate stays below `Δ/2` in absolute
//! value. A larger plaintext modulus `p = 2^plaintext_bits` packs more
//! payload bits per `Z_q` cell (fewer cells per slot → faster matvec,
//! smaller responses) but shrinks `Δ`, so for every database geometry
//! there is a maximum `plaintext_bits` that still decodes correctly.
//! This module computes that maximum from each backend's own published
//! correctness bound, applied to the **per-segment** matrix the backend
//! actually multiplies — the IKPIR construction runs one independent
//! index-PIR instance per SCF segment (paper Fig. 5), so the bound must
//! be evaluated at the per-segment shape, not the whole-store size.
//!
//! # The two bounds
//!
//! **FrodoPIR** (Davidson–Pestana–Celi, PoPETs 2023, Theorem 2 /
//! Eq. 8): with a ternary error vector `e` of length `m` and database
//! cells in `[0, p)`, `‖e·D‖_∞ ≤ 4·p·√m` with high probability, so
//! correctness needs
//!
//! ```text
//! q ≥ 8 · p² · √m ,
//! ```
//!
//! where `m` is the number of rows of the matrix the query vector
//! multiplies. The FrodoPIR backend stores each segment as a tall
//! `n_rows × row_width` matrix with one bucket per row, so `m` is the
//! per-segment bucket count `CuckooParams::segment_size() =
//! num_buckets / arity`. The bound is independent of the value width
//! (wider values add columns, not rows).
//!
//! **SimplePIR** (Henzinger et al., USENIX Security 2023,
//! Theorem C.1): with discrete-Gaussian error of width `σ` the noise
//! coordinate is `db[i,:]·e`, a Gaussian of standard deviation
//! `σ·‖db[i,:]‖`, and correctness with per-cell failure probability `δ`
//! needs `Δ/2 ≥ √2·σ·√(ln(2/δ))·‖db[i,:]‖`. The theorem is stated for
//! a square `√N×√N` matrix with **centered** entries of magnitude at
//! most `p/2`, which yields its published form
//! `⌊q/p⌋ ≥ √2·σ·p·N^(1/4)·√(ln(2/δ))`. Two adjustments are required
//! before it applies to this implementation:
//!
//! 1. **True row count.** The SimplePIR backend reshapes each segment
//!    to a near-square `R × C` matrix (`reshape_dims` in the backend) with
//!    `R = ⌈n_rows/k⌉`, `k = max(1, round(√(n_rows/row_width)))`, and
//!    the noise sums over the `R` rows, so `N^(1/4) = √(√N)` must be
//!    replaced by `√R`. Because
//!    `row_width = bucket_size · ⌈(fingerprint_bits + value_bits) /
//!    plaintext_bits⌉`, `R` — and therefore the maximum
//!    `plaintext_bits` — **depends on the value width**, unlike
//!    FrodoPIR.
//! 2. **Uncentered cells.** Cells here live in `[0, p)` (the
//!    high-bits-zero invariant of `FingerprintValueTable`), so the
//!    worst-case row norm is `p·√R`, not `(p/2)·√R`, costing a factor
//!    2 relative to the published constant.
//!
//! The bound actually enforced is therefore
//!
//! ```text
//! Δ = q/p ≥ 2√2 · σ · √(ln(2/δ)) · p · √R ,      δ = 2⁻⁴⁰ ,
//! ```
//!
//! a worst-case-database guarantee matching the spirit of FrodoPIR's
//! Eq. 8. (On uniformly random cells the expected row norm is
//! `p·√(R/3)` ≈ 0.58·p·√R, so real margins are somewhat wider than the
//! bound requires; the `#[ignore]`d noise-margin probes in the two
//! backend test suites measure this empirically.)
//!
//! # History
//!
//! The bench orchestration previously looked plaintext bits up from a
//! table keyed by **total** store capacity `num_buckets × bucket_size`.
//! That mis-sizes both backends: for FrodoPIR it over-counts the rows
//! by `arity × bucket_size` (over-conservative — up to one plaintext
//! bit of throughput left on the table), and for SimplePIR it ignores
//! the value-width dependence entirely (under-conservative — at
//! `value_bits = 8192` and paper-scale segments the old `pb = 10`
//! operating point overflows `Δ/2` on virtually every decode). This
//! module is the single source of truth replacing that table;
//! `scripts/lib.sh::backend_plaintext_bits` shells out to the
//! [`max_plaintext_bits` example](../examples/max_plaintext_bits.rs).
//!
//! # Related files
//!
//! - `backend/frodo/backend.rs` — the tall-matrix layout whose row
//!   count feeds [`frodo_max_plaintext_bits`]; hosts the FrodoPIR
//!   noise-margin probe.
//! - `backend/simple/backend.rs` — `reshape_dims`, re-used here so the
//!   selection can never drift from the layout; hosts the SimplePIR
//!   noise-margin probe.
//! - `scripts/lib.sh` — bench-side consumer via the example CLI.

use crate::backend::simple::reshape_dims;

/// `log₂ q` for both shipped backends (`q = 2³²`, native `u32`
/// wraparound).
const Q_BITS: u32 = 32;

/// Per-cell decode-failure probability budget for the SimplePIR bound:
/// `δ = 2⁻⁴⁰` (the value SimplePIR §4.2 instantiates).
const LOG2_DELTA: f64 = -40.0;

/// Largest `plaintext_bits` FrodoPIR decodes correctly at `q = 2³²`.
///
/// # Arguments
///
/// - `segment_rows` — rows of the per-segment matrix the query
///   multiplies, i.e. buckets per segment
///   (`CuckooParams::segment_size() = num_buckets / arity`). **Not**
///   the total store capacity.
///
/// # Returns
///
/// The largest `pb` such that `q ≥ 8·p²·√m` (FrodoPIR Eq. 8) holds for
/// `p = 2^pb`, `m = segment_rows`. Exact integer arithmetic: the
/// condition is equivalent to `m ≤ 2^(58 − 4·pb)`.
///
/// # Constraints
///
/// Panics if `segment_rows == 0`.
///
/// # Complexity
///
/// `O(1)` — at most 14 shift-and-compare steps.
pub fn frodo_max_plaintext_bits(segment_rows: u32) -> u32 {
    assert!(segment_rows > 0, "segment_rows must be positive");
    // q ≥ 8·p²·√m  ⟺  2^32 ≥ 2^(3 + 2·pb)·√m  ⟺  m ≤ 2^(58 − 4·pb).
    (1..=14u32)
        .rev()
        .find(|pb| u64::from(segment_rows) <= 1u64 << (58 - 4 * pb))
        .expect("pb = 1 admits every u32 segment_rows")
}

/// Largest `plaintext_bits` SimplePIR decodes correctly at `q = 2³²`.
///
/// # Arguments
///
/// - `segment_rows`     — buckets per segment (`num_buckets / arity`),
///   the `n_rows` handed to `SimplePirBackend::server_setup`.
/// - `bucket_size`      — slots per bucket (paper's `b`).
/// - `fingerprint_bits` — fingerprint width in bits (paper's `f`; the
///   benches fix 32).
/// - `value_bits`       — value width in bits (paper's `ℓ`).
/// - `sigma`            — discrete-Gaussian error width `σ`
///   (`SimpleParams::DEFAULT_SIGMA = 6.4`).
///
/// # Returns
///
/// The largest `pb` such that `Δ = 2^(32−pb)` satisfies
/// `Δ ≥ 2√2·σ·√(ln(2/δ))·p·√R` with `δ = 2⁻⁴⁰`, where `R` is the
/// reshape row count of the per-segment matrix **at that same `pb`**
/// (the row width `bucket_size·⌈(fingerprint_bits + value_bits)/pb⌉`
/// changes with `pb`, so the bound is re-evaluated per candidate).
///
/// # Constraints
///
/// Panics if any integer argument is zero, or if `sigma` is not
/// strictly positive and finite.
///
/// # Complexity
///
/// `O(1)` — at most 14 candidate evaluations, each one `reshape_dims`
/// call plus a few flops.
pub fn simple_max_plaintext_bits(
    segment_rows: u32,
    bucket_size: u32,
    fingerprint_bits: u32,
    value_bits: u32,
    sigma: f64,
) -> u32 {
    assert!(segment_rows > 0, "segment_rows must be positive");
    assert!(bucket_size > 0, "bucket_size must be positive");
    assert!(fingerprint_bits > 0, "fingerprint_bits must be positive");
    assert!(value_bits > 0, "value_bits must be positive");
    assert!(
        sigma > 0.0 && sigma.is_finite(),
        "sigma must be positive and finite, got {sigma}"
    );
    // 2√2 · σ · √(ln(2/δ)):  ln(2/δ) = (1 − log₂δ)·ln 2 = 41·ln 2.
    let factor = 2.0 * std::f64::consts::SQRT_2 * sigma * ((1.0 - LOG2_DELTA) * 2f64.ln()).sqrt();
    (1..=14u32)
        .rev()
        .find(|&pb| {
            let cells_per_slot = (fingerprint_bits + value_bits).div_ceil(pb);
            let row_width = bucket_size
                .checked_mul(cells_per_slot)
                .expect("row_width = bucket_size * cells_per_slot overflowed u32");
            let (_, reshape_rows, _) = reshape_dims(segment_rows, row_width);
            let delta = 2f64.powi((Q_BITS - pb) as i32);
            let p = 2f64.powi(pb as i32);
            delta >= factor * p * f64::from(reshape_rows).sqrt()
        })
        .expect("pb = 1 satisfies the bound for every u32-sized geometry")
}

#[cfg(test)]
mod tests {
    //! Pins the selection to hand-checked values at the paper's bench
    //! geometries and verifies the defining property (returned `pb`
    //! satisfies the bound, `pb + 1` violates it).

    use super::*;

    /// FrodoPIR: hand-checked values at the per-segment row counts the
    /// bench matrix produces, including the exact-equality boundary
    /// `s = 2^18` (`8·(2^10)²·√(2^18) = 2^32`).
    #[test]
    fn frodo_pinned_values() {
        assert_eq!(frodo_max_plaintext_bits(1 << 17), 10);
        assert_eq!(frodo_max_plaintext_bits(1 << 18), 10); // equality boundary
        assert_eq!(frodo_max_plaintext_bits((1 << 18) + 1), 9);
        assert_eq!(frodo_max_plaintext_bits(1 << 19), 9);
        assert_eq!(frodo_max_plaintext_bits(1 << 20), 9);
        // Small segments from the unit/integration tests.
        assert_eq!(frodo_max_plaintext_bits(1 << 10), 12);
    }

    /// FrodoPIR: non-increasing in the segment row count.
    #[test]
    fn frodo_monotone_in_rows() {
        let mut last = u32::MAX;
        for log_s in 4..=22 {
            let pb = frodo_max_plaintext_bits(1u32 << log_s);
            assert!(pb <= last, "pb must not grow with segment_rows");
            last = pb;
        }
    }

    /// SimplePIR: pinned values across the 12 bench configs × 3 value
    /// widths (fingerprint 32 bits, σ = 6.4). Derived by evaluating the
    /// documented bound by hand; the value-width dependence (unlike
    /// FrodoPIR) is the point of the test.
    #[test]
    fn simple_pinned_values() {
        // (arity, bucket_size, num_buckets) → per-segment rows.
        let configs: &[(u32, u32, u32)] = &[
            (2, 4, 262_144),
            (2, 4, 1_048_576),
            (4, 1, 1_048_576),
            (4, 1, 4_194_304),
            (4, 2, 524_288),
            (4, 2, 2_097_152),
            (3, 2, 786_432),
            (3, 2, 1_572_864),
            (3, 3, 393_216),
            (3, 3, 1_572_864),
            (4, 3, 524_288),
            (4, 3, 1_048_576),
        ];
        let expected: &[[u32; 3]] = &[
            [9, 9, 9],
            [9, 9, 8],
            [9, 9, 9],
            [9, 9, 8],
            [9, 9, 9],
            [9, 9, 8],
            [9, 9, 9],
            [9, 9, 8],
            [9, 9, 9],
            [9, 9, 8],
            [9, 9, 9],
            [9, 9, 9],
        ];
        for (&(arity, bs, nb), exp) in configs.iter().zip(expected) {
            let s = nb / arity;
            for (i, &vb) in [256u32, 2048, 8192].iter().enumerate() {
                let pb = simple_max_plaintext_bits(s, bs, 32, vb, 6.4);
                assert_eq!(
                    pb, exp[i],
                    "(arity={arity}, bs={bs}, nb={nb}, vb={vb}): got {pb}, want {}",
                    exp[i]
                );
            }
        }
    }

    /// SimplePIR: the returned `pb` satisfies the documented bound and
    /// `pb + 1` violates it (maximality), across a geometry sweep.
    #[test]
    fn simple_returned_pb_is_maximal() {
        let holds = |s: u32, bs: u32, vb: u32, pb: u32| -> bool {
            let rw = bs * (32 + vb).div_ceil(pb);
            let (_, r, _) = reshape_dims(s, rw);
            let factor = 2.0 * std::f64::consts::SQRT_2 * 6.4 * (41.0 * 2f64.ln()).sqrt();
            2f64.powi((32 - pb) as i32) >= factor * 2f64.powi(pb as i32) * f64::from(r).sqrt()
        };
        for log_s in [12u32, 15, 17, 19, 20] {
            for bs in [1u32, 2, 4] {
                for vb in [256u32, 2048, 8192] {
                    let s = 1u32 << log_s;
                    let pb = simple_max_plaintext_bits(s, bs, 32, vb, 6.4);
                    assert!(holds(s, bs, vb, pb), "bound must hold at returned pb");
                    assert!(!holds(s, bs, vb, pb + 1), "pb + 1 must violate the bound");
                }
            }
        }
    }

    /// SimplePIR: wider error (larger σ) can only shrink the operating
    /// point, and the value-width dependence is monotone too.
    #[test]
    fn simple_monotone_in_sigma_and_value_bits() {
        let s = 1u32 << 19;
        assert!(
            simple_max_plaintext_bits(s, 2, 32, 2048, 12.8)
                <= simple_max_plaintext_bits(s, 2, 32, 2048, 6.4)
        );
        assert!(
            simple_max_plaintext_bits(s, 2, 32, 8192, 6.4)
                <= simple_max_plaintext_bits(s, 2, 32, 256, 6.4)
        );
    }
}
