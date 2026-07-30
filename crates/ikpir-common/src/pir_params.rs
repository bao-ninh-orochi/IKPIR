//! Operating-point selection: the largest plaintext modulus each LWE
//! backend supports at `q = 2³²` while staying inside its share of the
//! scheme's per-query correctness budget.
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
//!
//! The paper's corrected correctness lemma (Lemma 2) bounds the
//! per-query failure probabilities `δ, ξ ≤ d·δ_idx + (1 + τ_ins/m)·d·b·2⁻ᶠ`.
//! At the shipped `fingerprint_bits = 64` the second (filter) term sits
//! near `2⁻⁶¹` — negligible — so the whole `κ =` [`KAPPA`] `= 40` target
//! rides on the index term: `d·δ_idx ≤ 2⁻⁽ᵏᵃᵖᵖᵃ⁺¹⁾`, half the budget in
//! log-space. One index-PIR read recovers
//! `row_width = bucket_size·⌈(fingerprint_bits + value_bits) /
//! plaintext_bits⌉` cells in each of the `arity` segments a lookup
//! touches, and fails if any one of them decodes wrong, so a union
//! bound over all `arity · row_width` cells turns that query-level
//! target into a **per-cell** budget
//!
//! ```text
//! δ_cell ≤ 2⁻⁽ᵏᵃᵖᵖᵃ⁺¹⁾ / (arity · row_width) .
//! ```
//!
//! [`frodo_max_plaintext_bits`] and [`simple_max_plaintext_bits`] each
//! compute, for every candidate `pb`, the actual per-cell failure tail
//! their backend's error distribution implies, and return the largest
//! `pb` whose tail stays inside `δ_cell`. Both now depend on the full
//! per-segment geometry — `arity`, `bucket_size`, `fingerprint_bits`,
//! and `value_bits`, not only `segment_rows` (FrodoPIR, before this
//! module targeted `δ_cell` explicitly) or `(segment_rows, value_bits)`
//! (SimplePIR).
//!
//! # The two bounds
//!
//! **FrodoPIR.** The decode noise on one cell is `⟨e, col⟩` for a
//! uniform-ternary error vector `e` of length `m = segment_rows` (iid,
//! per-coordinate variance `2/3`) against one column of database cells
//! of worst-case magnitude `p = 2^pb` — the same uncentered-cells
//! convention as the SimplePIR bound below: cells live in `[0, p)`, not
//! `[−p/2, p/2)`. Decode of a cell is correct iff `|⟨e, col⟩| < Δ/2`,
//! and Bernstein's inequality for a sum of independent, bounded,
//! zero-mean terms gives
//!
//! ```text
//! Pr[|⟨e, col⟩| ≥ t] ≤ 2·exp(−t² / (2V + (2/3)·M·t)) ,
//! V = (2/3)·m·p² ,  M = p ,  t = Δ/2 = 2^(31−pb) .
//! ```
//!
//! [`frodo_max_plaintext_bits`] evaluates this tail directly (in `log₂`,
//! for numerical range) and compares it against `δ_cell` above.
//!
//! **SimplePIR** (Henzinger et al., USENIX Security 2023, Theorem C.1):
//! with discrete-Gaussian error of width `σ` the noise coordinate is
//! `db[i,:]·e`, a Gaussian of standard deviation `σ·‖db[i,:]‖`, and
//! correctness with per-cell failure probability `δ_cell` needs
//! `Δ/2 ≥ √2·σ·√(ln(2/δ_cell))·‖db[i,:]‖`. The theorem is stated for a
//! square `√N×√N` matrix with **centered** entries of magnitude at most
//! `p/2`, which yields its published form
//! `⌊q/p⌋ ≥ √2·σ·p·N^(1/4)·√(ln(2/δ_cell))`. Two adjustments are
//! required before it applies to this implementation:
//!
//! 1. **True row count.** The SimplePIR backend reshapes each segment
//!    to a near-square `R × C` matrix (`reshape_dims` in the backend)
//!    with `R = ⌈n_rows/k⌉`, `k = max(1, round(√(n_rows/row_width)))`,
//!    and the noise sums over the `R` rows, so `N^(1/4) = √(√N)` must
//!    be replaced by `√R`. Because `row_width = bucket_size ·
//!    ⌈(fingerprint_bits + value_bits) / plaintext_bits⌉`, `R` — and
//!    therefore the maximum `plaintext_bits` — depends on the full
//!    slot width.
//! 2. **Uncentered cells.** Cells here live in `[0, p)`, so the
//!    worst-case row norm is `p·√R`, not `(p/2)·√R`, costing a factor 2
//!    relative to the published constant.
//!
//! The bound actually enforced is therefore
//!
//! ```text
//! Δ = q/p ≥ 2√2 · σ · √(ln(2/δ_cell)) · p · √R ,
//! ```
//!
//! with `δ_cell` the **same** per-cell target FrodoPIR uses above —
//! algebraically `ln(2/δ_cell) = ln 2 · (κ + 2 + log₂(arity ·
//! row_width))`, the `+2` folding in both `ln 2`'s own contribution and
//! the `κ+1` from the budget split. [`simple_max_plaintext_bits`]
//! evaluates this directly. (On uniformly random cells the expected row
//! norm is `p·√(R/3)` ≈ 0.58·p·√R, so real margins are somewhat wider
//! than the bound requires; the `#[ignore]`d noise-margin probes in the
//! two backend test suites measure this empirically.)
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
//! module replaced that table with the per-segment derivations above.
//!
//! Until this change, FrodoPIR's own bound was its paper's Eq. 8,
//! `q ≥ 8·p²·√m` (equivalently `m ≤ 2^(58−4·pb)`), derived from
//! `‖e·D‖_∞ ≤ 4·p·√m` "with high probability" — no explicit `δ`. That
//! constant is exactly what the cruder, variance-blind Hoeffding tail
//! `2·exp(−t²/(2·m·p²))` gives at `t = 4p√m`: the exponent reduces to
//! `−8` independent of `m` and `p`, i.e. a fixed per-cell tail near
//! `2·e⁻⁸ ≈ 6.7 × 10⁻⁴` — orders of magnitude short of a `2⁻⁴¹`-scale
//! target, and with no `δ` to dial in. [`frodo_max_plaintext_bits`]
//! replaces it with the explicit, `δ_cell`-targeted Bernstein tail
//! above. Using that same Hoeffding tail in place of Bernstein
//! reproduces every operating point across the paper's five-config
//! matrix at both value widths; the extra sharpness only bites at
//! small `segment_rows` (e.g. the dev-scale `(arity, bucket_size) =
//! (4, 1)` spot check at `segment_rows = 2^12` selects `pb = 11` under
//! Bernstein but only `10` under Hoeffding — see
//! `frodo_dev_scale_spot_values` in this module's tests). Bernstein is
//! enforced throughout because it is the provably tighter bound, not
//! because the two happen to agree at paper scale.
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

/// Overall per-query correctness target: `δ, ξ ≤ 2⁻ᵏᵃᵖᵖᵃ` (paper
/// Lemma 2, corrected). This module owns only the **index** half of
/// that budget in log-space — `d · δ_idx ≤ 2⁻⁽ᵏᵃᵖᵖᵃ⁺¹⁾` — leaving the
/// other half to the **filter** term, which at the shipped
/// `fingerprint_bits = 64` sits near `2⁻⁶¹` and is not this module's
/// concern: it is a property of the Segmented Cuckoo Filter, not of
/// either PIR backend's plaintext modulus.
pub const KAPPA: u32 = 40;

/// Largest `plaintext_bits` FrodoPIR decodes correctly at `q = 2³²`,
/// targeting the per-cell budget `δ_cell` derived in the module docs.
///
/// # Arguments
///
/// - `arity`            — SCF arity `d` (candidate segments per key,
///   `CuckooParams::arity()`). A lookup fails if any cell in any of
///   the `arity` per-segment reads decodes wrong, so the per-cell
///   budget divides by it.
/// - `segment_rows`     — rows of the per-segment matrix the query
///   multiplies, i.e. buckets per segment
///   (`CuckooParams::segment_size() = num_buckets / arity`). **Not**
///   the total store capacity.
/// - `bucket_size`      — slots per bucket (paper's `b`).
/// - `fingerprint_bits` — fingerprint width in bits (paper's `f`).
/// - `value_bits`       — value width in bits (paper's `ℓ`).
///
/// # Returns
///
/// The largest `pb` such that the Bernstein tail bound on
/// `|⟨e, col⟩|` (module docs) stays at or below
/// `δ_cell = 2⁻⁽ᵏᵃᵖᵖᵃ⁺¹⁾ / (arity · row_width)`, with
/// `row_width = bucket_size · ⌈(fingerprint_bits + value_bits) / pb⌉`
/// re-evaluated at each candidate `pb`.
///
/// # Constraints
///
/// Panics if `segment_rows == 0`, or if `arity`, `bucket_size`,
/// `fingerprint_bits`, or `value_bits` is zero.
///
/// # Complexity
///
/// `O(14)` — one Bernstein-tail evaluation per candidate `pb`.
pub fn frodo_max_plaintext_bits(
    arity: u32,
    segment_rows: u32,
    bucket_size: u32,
    fingerprint_bits: u32,
    value_bits: u32,
) -> u32 {
    assert!(arity > 0, "arity must be positive");
    assert!(segment_rows > 0, "segment_rows must be positive");
    assert!(bucket_size > 0, "bucket_size must be positive");
    assert!(fingerprint_bits > 0, "fingerprint_bits must be positive");
    assert!(value_bits > 0, "value_bits must be positive");
    (1..=14u32)
        .rev()
        .find(|&pb| {
            let t = 2f64.powi(31 - pb as i32); // Δ/2
            let p = 2f64.powi(pb as i32);
            let denom = (4.0 / 3.0) * f64::from(segment_rows) * p * p + (2.0 / 3.0) * p * t;
            let log2_tail = 1.0 - (t * t / denom) / std::f64::consts::LN_2;
            let row_width = bucket_size * (fingerprint_bits + value_bits).div_ceil(pb);
            let log2_target = -f64::from(KAPPA + 1) - f64::from(arity * row_width).log2();
            log2_tail <= log2_target
        })
        .expect("pb = 1 admits every u32 segment_rows")
}

/// Largest `plaintext_bits` SimplePIR decodes correctly at `q = 2³²`,
/// targeting the same per-cell budget `δ_cell` as
/// [`frodo_max_plaintext_bits`] (module docs).
///
/// # Arguments
///
/// - `arity`            — SCF arity `d`; see [`frodo_max_plaintext_bits`].
/// - `segment_rows`     — buckets per segment (`num_buckets / arity`),
///   the `n_rows` handed to `SimplePirBackend::server_setup`.
/// - `bucket_size`      — slots per bucket (paper's `b`).
/// - `fingerprint_bits` — fingerprint width in bits (paper's `f`).
/// - `value_bits`       — value width in bits (paper's `ℓ`).
/// - `sigma`            — discrete-Gaussian error width `σ`
///   (`SimpleParams::DEFAULT_SIGMA = 6.4`).
///
/// # Returns
///
/// The largest `pb` such that `Δ = 2^(32−pb)` satisfies
/// `Δ ≥ 2√2·σ·√(ln(2/δ_cell))·p·√R` with `δ_cell` as in the module
/// docs, where `R` is the reshape row count of the per-segment matrix
/// **at that same `pb`** (`row_width` — and so `R` — changes with
/// `pb`, so the bound is re-evaluated per candidate).
///
/// # Constraints
///
/// Panics if `segment_rows`, `arity`, `bucket_size`, `fingerprint_bits`,
/// or `value_bits` is zero, or if `sigma` is not strictly positive and
/// finite.
///
/// # Complexity
///
/// `O(14)` — one `reshape_dims` call plus a few flops per candidate `pb`.
pub fn simple_max_plaintext_bits(
    arity: u32,
    segment_rows: u32,
    bucket_size: u32,
    fingerprint_bits: u32,
    value_bits: u32,
    sigma: f64,
) -> u32 {
    assert!(arity > 0, "arity must be positive");
    assert!(segment_rows > 0, "segment_rows must be positive");
    assert!(bucket_size > 0, "bucket_size must be positive");
    assert!(fingerprint_bits > 0, "fingerprint_bits must be positive");
    assert!(value_bits > 0, "value_bits must be positive");
    assert!(
        sigma > 0.0 && sigma.is_finite(),
        "sigma must be positive and finite, got {sigma}"
    );
    (1..=14u32)
        .rev()
        .find(|&pb| {
            let row_width = bucket_size * (fingerprint_bits + value_bits).div_ceil(pb);
            let (_, reshape_rows, _) = reshape_dims(segment_rows, row_width);
            let ln_2_over_delta = std::f64::consts::LN_2
                * (f64::from(KAPPA + 2) + f64::from(arity * row_width).log2());
            let factor = 2.0 * std::f64::consts::SQRT_2 * sigma * ln_2_over_delta.sqrt();
            2f64.powi((32 - pb) as i32)
                >= factor * 2f64.powi(pb as i32) * f64::from(reshape_rows).sqrt()
        })
        .expect("pb = 1 satisfies the bound for every u32-sized geometry")
}

#[cfg(test)]
mod tests {
    //! Pins the selection to machine-checked values at the paper's
    //! bench geometries and a few dev-scale spot checks, and verifies
    //! the defining properties: maximality (the returned `pb` passes,
    //! `pb + 1` fails), downward-closure (every `pb` below it passes
    //! too, so a caller may safely narrow further), and monotonicity in
    //! the inputs that should only ever shrink the operating point.

    use super::*;

    /// The five paper `(arity, bucket_size, num_buckets)` cells (Table 2
    /// / the keyword-PIR matrix, `PAPER_PIR_CONFIGS` in
    /// `scripts/lib.sh`), independent of value width.
    const PAPER_MATRIX: &[(u32, u32, u32)] = &[
        (2, 4, 262_144),
        (3, 2, 786_432),
        (3, 3, 393_216),
        (4, 1, 1_048_576),
        (4, 2, 524_288),
    ];

    /// Re-derivation of the FrodoPIR pass predicate, independent of
    /// [`frodo_max_plaintext_bits`]'s scan, for the maximality /
    /// downward-closure checks below.
    fn frodo_holds(
        arity: u32,
        segment_rows: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        pb: u32,
    ) -> bool {
        let t = 2f64.powi(31 - pb as i32);
        let p = 2f64.powi(pb as i32);
        let denom = (4.0 / 3.0) * f64::from(segment_rows) * p * p + (2.0 / 3.0) * p * t;
        let log2_tail = 1.0 - (t * t / denom) / std::f64::consts::LN_2;
        let row_width = bucket_size * (fingerprint_bits + value_bits).div_ceil(pb);
        let log2_target = -f64::from(KAPPA + 1) - f64::from(arity * row_width).log2();
        log2_tail <= log2_target
    }

    /// Re-derivation of the SimplePIR pass predicate, independent of
    /// [`simple_max_plaintext_bits`]'s scan.
    fn simple_holds(
        arity: u32,
        segment_rows: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        sigma: f64,
        pb: u32,
    ) -> bool {
        let row_width = bucket_size * (fingerprint_bits + value_bits).div_ceil(pb);
        let (_, reshape_rows, _) = reshape_dims(segment_rows, row_width);
        let ln_2_over_delta =
            std::f64::consts::LN_2 * (f64::from(KAPPA + 2) + f64::from(arity * row_width).log2());
        let factor = 2.0 * std::f64::consts::SQRT_2 * sigma * ln_2_over_delta.sqrt();
        2f64.powi((32 - pb) as i32)
            >= factor * 2f64.powi(pb as i32) * f64::from(reshape_rows).sqrt()
    }

    /// FrodoPIR at `fingerprint_bits = 64` selects `pb = 9` at every
    /// paper cell and both paper value widths — the new δ_cell-targeted
    /// rule costs exactly one bit relative to the old Eq. 8 rule, which
    /// gave `pb = 10` at these same per-segment row counts (see the
    /// History note in the module docs).
    #[test]
    fn frodo_paper_matrix() {
        for &(arity, bs, nb) in PAPER_MATRIX {
            let seg = nb / arity;
            for vb in [2048u32, 8192] {
                let pb = frodo_max_plaintext_bits(arity, seg, bs, 64, vb);
                assert_eq!(
                    pb, 9,
                    "(arity={arity}, bs={bs}, nb={nb}, vb={vb}): got {pb}, want 9"
                );
            }
        }
    }

    /// SimplePIR at `fingerprint_bits = 64`, σ = 6.4: pinned per paper
    /// cell × value width. The two `8`s (arity-2 `(2, 4)` and arity-3
    /// `(3, 2)`, both at `value_bits = 8192`) are genuine boundary
    /// cases — they miss the retargeted bound at `pb = 9` by under 1%,
    /// not by a wide margin.
    #[test]
    fn simple_paper_matrix() {
        let expected: &[[u32; 2]] = &[
            [9, 8], // (2, 4, 262144)
            [9, 8], // (3, 2, 786432)
            [9, 9], // (3, 3, 393216)
            [9, 9], // (4, 1, 1048576)
            [9, 9], // (4, 2, 524288)
        ];
        for (&(arity, bs, nb), exp) in PAPER_MATRIX.iter().zip(expected) {
            let seg = nb / arity;
            for (i, &vb) in [2048u32, 8192].iter().enumerate() {
                let pb = simple_max_plaintext_bits(arity, seg, bs, 64, vb, 6.4);
                assert_eq!(
                    pb, exp[i],
                    "(arity={arity}, bs={bs}, nb={nb}, vb={vb}): got {pb}, want {}",
                    exp[i]
                );
            }
        }
    }

    /// FrodoPIR dev-scale spot checks, at `fingerprint_bits = 64`: the
    /// geometries `scripts/lib.sh::default_num_buckets` picks for a
    /// one-off `bench.sh` run (arity 2/4 at `num_buckets = 16384`,
    /// arity 3 at `24576`).
    #[test]
    fn frodo_dev_scale_spot_values() {
        let cases: &[(u32, u32, u32, u32, u32)] = &[
            // (arity, bucket_size, num_buckets, value_bits, expected pb)
            (4, 1, 16384, 2048, 11),
            (4, 1, 16384, 8192, 11),
            (2, 4, 16384, 2048, 10),
            (2, 4, 16384, 8192, 10),
            (3, 2, 24576, 2048, 10),
            (3, 2, 24576, 8192, 10),
        ];
        for &(arity, bs, nb, vb, exp) in cases {
            let seg = nb / arity;
            let pb = frodo_max_plaintext_bits(arity, seg, bs, 64, vb);
            assert_eq!(pb, exp, "(arity={arity}, bs={bs}, nb={nb}, vb={vb})");
        }
    }

    /// SimplePIR counterpart of [`frodo_dev_scale_spot_values`].
    #[test]
    fn simple_dev_scale_spot_values() {
        let cases: &[(u32, u32, u32, u32, u32)] = &[
            (4, 1, 16384, 2048, 10),
            (4, 1, 16384, 8192, 9),
            (2, 4, 16384, 2048, 9),
            (2, 4, 16384, 8192, 9),
            (3, 2, 24576, 2048, 9),
            (3, 2, 24576, 8192, 9),
        ];
        for &(arity, bs, nb, vb, exp) in cases {
            let seg = nb / arity;
            let pb = simple_max_plaintext_bits(arity, seg, bs, 64, vb, 6.4);
            assert_eq!(pb, exp, "(arity={arity}, bs={bs}, nb={nb}, vb={vb})");
        }
    }

    /// Maximality: the returned `pb` satisfies [`frodo_holds`] and
    /// `pb + 1` violates it, across the paper matrix and both value
    /// widths.
    #[test]
    fn frodo_returned_pb_is_maximal() {
        for &(arity, bs, nb) in PAPER_MATRIX {
            let seg = nb / arity;
            for vb in [2048u32, 8192] {
                let pb = frodo_max_plaintext_bits(arity, seg, bs, 64, vb);
                assert!(
                    frodo_holds(arity, seg, bs, 64, vb, pb),
                    "bound must hold at returned pb={pb} (arity={arity}, bs={bs}, nb={nb}, vb={vb})"
                );
                assert!(
                    !frodo_holds(arity, seg, bs, 64, vb, pb + 1),
                    "pb + 1 must violate the bound (arity={arity}, bs={bs}, nb={nb}, vb={vb})"
                );
            }
        }
    }

    /// SimplePIR counterpart of [`frodo_returned_pb_is_maximal`].
    #[test]
    fn simple_returned_pb_is_maximal() {
        for &(arity, bs, nb) in PAPER_MATRIX {
            let seg = nb / arity;
            for vb in [2048u32, 8192] {
                let pb = simple_max_plaintext_bits(arity, seg, bs, 64, vb, 6.4);
                assert!(
                    simple_holds(arity, seg, bs, 64, vb, 6.4, pb),
                    "bound must hold at returned pb={pb} (arity={arity}, bs={bs}, nb={nb}, vb={vb})"
                );
                assert!(
                    !simple_holds(arity, seg, bs, 64, vb, 6.4, pb + 1),
                    "pb + 1 must violate the bound (arity={arity}, bs={bs}, nb={nb}, vb={vb})"
                );
            }
        }
    }

    /// Downward-closure: every `pb` at or below the chosen one also
    /// passes [`frodo_holds`], across the paper matrix — so a caller
    /// may safely use any narrower `pb`, not just the returned maximum.
    #[test]
    fn frodo_passing_set_is_downward_closed() {
        for &(arity, bs, nb) in PAPER_MATRIX {
            let seg = nb / arity;
            for vb in [2048u32, 8192] {
                let pb = frodo_max_plaintext_bits(arity, seg, bs, 64, vb);
                for p in 1..pb {
                    assert!(
                        frodo_holds(arity, seg, bs, 64, vb, p),
                        "pb={p} must also pass below the chosen {pb} \
                         (arity={arity}, bs={bs}, nb={nb}, vb={vb})"
                    );
                }
            }
        }
    }

    /// SimplePIR counterpart of [`frodo_passing_set_is_downward_closed`].
    #[test]
    fn simple_passing_set_is_downward_closed() {
        for &(arity, bs, nb) in PAPER_MATRIX {
            let seg = nb / arity;
            for vb in [2048u32, 8192] {
                let pb = simple_max_plaintext_bits(arity, seg, bs, 64, vb, 6.4);
                for p in 1..pb {
                    assert!(
                        simple_holds(arity, seg, bs, 64, vb, 6.4, p),
                        "pb={p} must also pass below the chosen {pb} \
                         (arity={arity}, bs={bs}, nb={nb}, vb={vb})"
                    );
                }
            }
        }
    }

    /// FrodoPIR: non-increasing in the segment row count, at a fixed
    /// representative geometry.
    #[test]
    fn frodo_monotone_in_segment_rows() {
        let mut last = u32::MAX;
        for log_s in 4..=22 {
            let pb = frodo_max_plaintext_bits(4, 1u32 << log_s, 1, 64, 8192);
            assert!(
                pb <= last,
                "pb must not grow with segment_rows (log_s={log_s})"
            );
            last = pb;
        }
    }

    /// SimplePIR counterpart of [`frodo_monotone_in_segment_rows`].
    #[test]
    fn simple_monotone_in_segment_rows() {
        let mut last = u32::MAX;
        for log_s in 4..=22 {
            let pb = simple_max_plaintext_bits(4, 1u32 << log_s, 1, 64, 8192, 6.4);
            assert!(
                pb <= last,
                "pb must not grow with segment_rows (log_s={log_s})"
            );
            last = pb;
        }
    }

    /// SimplePIR: wider error (larger σ) and wider values can only
    /// shrink the operating point.
    #[test]
    fn simple_monotone_in_sigma_and_value_bits() {
        let s = 1u32 << 19;
        assert!(
            simple_max_plaintext_bits(4, s, 2, 64, 2048, 12.8)
                <= simple_max_plaintext_bits(4, s, 2, 64, 2048, 6.4)
        );
        assert!(
            simple_max_plaintext_bits(4, s, 2, 64, 8192, 6.4)
                <= simple_max_plaintext_bits(4, s, 2, 64, 256, 6.4)
        );
    }

    /// FrodoPIR is `fingerprint_bits`-dependent only through the
    /// per-cell budget's `row_width` term: a narrower fingerprint can
    /// only widen (never narrow) the selected `pb`. Pinned at the
    /// paper's `(4, 1)` cell, `ℓ = 8192`, where `f = 64` selects
    /// `pb = 9` (the [`frodo_paper_matrix`] value); asserting the
    /// inequality against `f = 1` rather than a second invented pin
    /// keeps the test honest about what is actually being claimed.
    #[test]
    fn frodo_monotone_in_fingerprint_bits() {
        let seg = 1_048_576 / 4;
        let pb_f64 = frodo_max_plaintext_bits(4, seg, 1, 64, 8192);
        assert_eq!(pb_f64, 9);
        let pb_f1 = frodo_max_plaintext_bits(4, seg, 1, 1, 8192);
        assert!(
            pb_f1 >= pb_f64,
            "narrower fingerprint must not shrink the operating point"
        );
    }
}
