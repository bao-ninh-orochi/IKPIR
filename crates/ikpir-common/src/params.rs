//! LWE parameters, filter arity, plaintext encoding, and wire-level constants.
//!
//! The concrete parameter set mirrors [ChalametPIR] at 128-bit classical LWE
//! security. We keep the ciphertext modulus at `q = 2^32` so every ring element
//! is a plain `u32`; this is deliberate — it lets matrix arithmetic reduce to
//! `u32::wrapping_*` with no modular-reduction overhead.
//!
//! [ChalametPIR]: https://eprint.iacr.org/2024/092

use core::fmt;

/// LWE lattice dimension (secret-vector length).
///
/// Matches the reference ChalametPIR value for 128-bit classical LWE security
/// under a uniform ternary secret/error distribution.
pub const LWE_DIMENSION: usize = 1774;

/// Number of bits in the ciphertext modulus `q`.
///
/// Ciphertexts are elements of `Z_q` with `q = 2^32`, so each element fits in a
/// single `u32` and all arithmetic is wrapping.
pub const CIPHERTEXT_MODULUS_BITS: u32 = 32;

/// Length of the server seed that expands the public LWE matrix `A`.
pub const SEED_LEN: usize = 32;

/// Minimum plaintext-element bit width (slot granularity).
///
/// Chosen so that the per-slot LWE error budget is respected even for the
/// largest supported database. See [`mat_elem_bit_len_for`].
pub const MIN_MAT_ELEM_BIT_LEN: u32 = 4;

/// Maximum plaintext-element bit width.
///
/// For tiny databases the noise growth is small enough to support wider slots.
pub const MAX_MAT_ELEM_BIT_LEN: u32 = 14;

/// SCF fingerprint width used by the Phase B encoder.
///
/// 12 bits matches the default exercised by the SCF test suite and keeps the
/// theoretical false-positive rate of the membership filter bounded by
/// `d · b / 2^{fp_bits} ≤ 4 · 4 / 4096 ≈ 0.4 %`.
pub const DEFAULT_FP_BITS: u32 = 12;

/// Slots per bucket in the Segmented Cuckoo Map. Matches the SCF crate's
/// empirical load-factor sweet spot for `b = 4`.
pub const DEFAULT_SLOTS_PER_BUCKET: u32 = 4;

/// Filter arity — how many candidate buckets each keyword maps to.
///
/// Mirrors the six concrete index schemes exposed by
/// `segmented-cuckoo-filter`. Two orthogonal axes: segmentation and
/// candidate-count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arity {
    /// Standard 2-ary (all candidates in `[0, n)`).
    Standard2,
    /// Segmented 2-ary (bipartite: one candidate per half).
    Segmented2,
    /// Standard 3-ary via `xor3` cycling.
    Standard3,
    /// Segmented 3-ary (one candidate per third).
    Segmented3,
    /// Standard 4-ary via `xor4` cycling.
    Standard4,
    /// Segmented 4-ary (one candidate per quarter). **Default.**
    Segmented4,
}

impl Arity {
    /// Number of candidate buckets for this arity (2, 3, or 4).
    pub const fn degree(self) -> u32 {
        match self {
            Self::Standard2 | Self::Segmented2 => 2,
            Self::Standard3 | Self::Segmented3 => 3,
            Self::Standard4 | Self::Segmented4 => 4,
        }
    }

    /// `true` if the arity uses the segmented (bipartite / k-partite) layout.
    pub const fn is_segmented(self) -> bool {
        matches!(self, Self::Segmented2 | Self::Segmented3 | Self::Segmented4)
    }

    /// Target load factor for a Segmented Cuckoo Map at `b = 4`.
    ///
    /// Mirrors the empirical `MAX_LOAD_FACTOR` table in
    /// `segmented-cuckoo-filter/src/filter.rs:83-90` for the `b = 4` column.
    /// Used by [`FilterParams::recommended`] to size `seg` given a target
    /// item count.
    pub fn cuckoo_load_target(self) -> f64 {
        match self.degree() {
            2 => 0.91,
            3 => 0.94,
            _ => 0.95,
        }
    }
}

impl Default for Arity {
    fn default() -> Self {
        Self::Segmented4
    }
}

impl fmt::Display for Arity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Standard2 => "standard-2ary",
            Self::Segmented2 => "segmented-2ary",
            Self::Standard3 => "standard-3ary",
            Self::Segmented3 => "segmented-3ary",
            Self::Standard4 => "standard-4ary",
            Self::Segmented4 => "segmented-4ary",
        };
        f.write_str(s)
    }
}

/// Static parameters of the SCF-backed PIR encoding.
///
/// Captures both the filter shape (arity, bucket count, fingerprint width) and
/// the plaintext layout (elements-per-row, bit width per element). Server and
/// client must agree on the full struct for the protocol to work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterParams {
    /// Index scheme — mirrors one of the six `segmented-cuckoo-filter` variants.
    pub arity: Arity,
    /// Total number of rows in the preprocessing matrix.
    pub num_buckets: u32,
    /// Fingerprint bits per stored item (for the membership filter).
    pub fingerprint_bits: u32,
    /// Number of `u32` elements in a single database row.
    pub row_elems: u32,
    /// Useful bits per plaintext slot (`Δ = 2^{32 − mat_elem_bit_len}`).
    pub mat_elem_bit_len: u32,
    /// Declared maximum value length in bytes (for encoding / decoding).
    pub value_len: u32,
}

impl FilterParams {
    /// Build a `FilterParams` sized for a Segmented-Cuckoo-Map-backed PIR
    /// of `num_items` `(key, value)` pairs with `value_len`-byte values.
    ///
    /// Sizing rules:
    ///   (i)   segment-local load stays below the SCF empirical target
    ///         at `b = 4` — see [`Arity::cuckoo_load_target`];
    ///   (ii)  the per-segment PIR noise budget holds,
    ///         `2 · seg · P² < 2^32` where `P = 2^{mat_elem_bit_len}` and
    ///         `seg = num_buckets / k`;
    ///   (iii) each `value_len`-byte value fits inside `row_elems` plaintext
    ///         slots at `mat_elem_bit_len` bits per slot;
    ///   (iv)  `seg` is a power of 2 (required by every segmented SCF
    ///         variant), so `num_buckets = k · seg`.
    ///
    /// # Errors
    ///
    /// - [`ParamError::ArityNotSupportedForPir`] for any `Standard*` arity —
    ///   the per-segment PIR design requires segmentation.
    /// - [`ParamError::EmptyDatabase`] / [`ParamError::EmptyValue`] for
    ///   zero-sized inputs.
    /// - [`ParamError::TooManyItems`] / [`ParamError::ValueTooLarge`] when
    ///   the resulting dimensions would exceed `u32`.
    pub fn recommended(arity: Arity, num_items: u32, value_len: u32) -> Result<Self, ParamError> {
        if !arity.is_segmented() {
            return Err(ParamError::ArityNotSupportedForPir);
        }
        if num_items == 0 {
            return Err(ParamError::EmptyDatabase);
        }
        if value_len == 0 {
            return Err(ParamError::EmptyValue);
        }

        let b = DEFAULT_SLOTS_PER_BUCKET as u64;
        let load_target = arity.cuckoo_load_target();
        let slots_needed = (num_items as f64 / load_target).ceil() as u64;
        let seg_target = slots_needed.div_ceil(b).max(1);
        let seg_u64 = next_power_of_2_u64(seg_target).max(1);
        let num_buckets_u64 = seg_u64
            .checked_mul(arity.degree() as u64)
            .ok_or(ParamError::TooManyItems)?;
        let num_buckets = u32::try_from(num_buckets_u64).map_err(|_| ParamError::TooManyItems)?;
        let seg = u32::try_from(seg_u64).map_err(|_| ParamError::TooManyItems)?;

        // Noise budget: one PIR query touches only `seg` rows of `D_j`.
        //   2 · seg · P²  <  2^32   ⇒   B < (31 − log2(seg)) / 2
        // For integer B we need B ≤ ⌊(30 − log2(seg)) / 2⌋ (strict < becomes ≤ after
        // subtracting one before the floor), ensuring the product stays strictly below 2^32.
        let log2_seg = 31u32.saturating_sub(seg.leading_zeros());
        let budget = (30u32.saturating_sub(log2_seg)) / 2;
        let mat_elem_bit_len = budget.clamp(MIN_MAT_ELEM_BIT_LEN, MAX_MAT_ELEM_BIT_LEN);

        let value_bits = (value_len as u64) * 8;
        let row_elems = value_bits.div_ceil(mat_elem_bit_len as u64);
        let row_elems = u32::try_from(row_elems).map_err(|_| ParamError::ValueTooLarge)?;

        Ok(Self {
            arity,
            num_buckets,
            fingerprint_bits: DEFAULT_FP_BITS,
            row_elems,
            mat_elem_bit_len,
            value_len,
        })
    }

    /// Number of candidate buckets per keyword (convenience).
    pub fn degree(&self) -> u32 {
        self.arity.degree()
    }

    /// Per-segment row count (`num_buckets / degree`).
    ///
    /// Under the per-segment PIR architecture each of the `degree`
    /// segments owns `seg()` rows; the j-th candidate index for any
    /// keyword lives in `[j · seg(), (j + 1) · seg())`.
    pub fn seg(&self) -> u32 {
        self.num_buckets / self.arity.degree()
    }

    /// Slots per bucket (`b`). Fixed at [`DEFAULT_SLOTS_PER_BUCKET`] for now.
    pub fn slots_per_bucket(&self) -> u32 {
        DEFAULT_SLOTS_PER_BUCKET
    }

    /// `u32` elements per fingerprint slot chunk, LSB-first packing of
    /// `fingerprint_bits` bits into `mat_elem_bit_len`-wide plaintext slots.
    pub fn fp_elems(&self) -> u32 {
        self.fingerprint_bits.div_ceil(self.mat_elem_bit_len)
    }

    /// `u32` elements per value slot chunk (same as the legacy `row_elems`
    /// field: a value is `ceil(value_len·8 / mat_elem_bit_len)` plaintext
    /// slots wide).
    pub fn val_elems(&self) -> u32 {
        self.row_elems
    }

    /// Total `u32` elements per bucket row (`b · (fp_elems + val_elems)`).
    pub fn row_elems_bucket(&self) -> u32 {
        self.slots_per_bucket() * (self.fp_elems() + self.val_elems())
    }

    /// Scale factor `Δ = 2^{32 − mat_elem_bit_len}` used on encoding.
    ///
    /// `Δ` is the distance between adjacent encoded values in `Z_q`. The
    /// decryption tolerance is `Δ / 2`; the LWE error distribution must stay
    /// well below that bound.
    pub fn scale(&self) -> u32 {
        debug_assert!(self.mat_elem_bit_len < CIPHERTEXT_MODULUS_BITS);
        1u32 << (CIPHERTEXT_MODULUS_BITS - self.mat_elem_bit_len)
    }
}

/// Pick a plaintext-element bit width based on database size.
///
/// Smaller databases tolerate wider slots because the accumulated LWE noise
/// during `DB · query` is linear in `num_buckets`. The table is taken from the
/// reference ChalametPIR constants — same for direct comparability.
pub fn mat_elem_bit_len_for(num_items: u32) -> u32 {
    // Log2-bucketed schedule; each step drops one bit of plaintext width.
    // Matches the ChalametPIR "MIN_CIPHER_TEXT_BIT_LEN…MAX_CIPHER_TEXT_BIT_LEN"
    // range for databases up to 2^42 entries.
    match num_items {
        0 => MAX_MAT_ELEM_BIT_LEN,
        1..=2 => MAX_MAT_ELEM_BIT_LEN,
        3..=4 => 13,
        5..=16 => 12,
        17..=64 => 11,
        65..=256 => 10,
        257..=1024 => 9,
        1025..=4096 => 8,
        4097..=16_384 => 7,
        16_385..=65_536 => 6,
        65_537..=262_144 => 5,
        _ => MIN_MAT_ELEM_BIT_LEN,
    }
}

fn next_power_of_2_u64(x: u64) -> u64 {
    if x <= 1 {
        return 1;
    }
    let shift = 64 - (x - 1).leading_zeros();
    1u64 << shift
}

/// Error returned when parameter selection fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamError {
    /// Caller passed `num_items = 0`.
    EmptyDatabase,
    /// Caller passed `value_len = 0`.
    EmptyValue,
    /// `value_len` would require more than `u32::MAX` slots.
    ValueTooLarge,
    /// `num_items` (after bucket overhead) exceeds what the arity can size.
    TooManyItems,
    /// The supplied arity is not one of the segmented variants that the
    /// per-segment PIR architecture supports.
    ArityNotSupportedForPir,
}

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDatabase => f.write_str("database is empty"),
            Self::EmptyValue => f.write_str("value_len must be non-zero"),
            Self::ValueTooLarge => f.write_str("value_len too large for any supported row width"),
            Self::TooManyItems => f.write_str("num_items exceeds the supported maximum"),
            Self::ArityNotSupportedForPir => {
                f.write_str("arity not supported by the per-segment PIR (use Segmented2/3/4)")
            }
        }
    }
}

impl std::error::Error for ParamError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arity_degree_matches_variant() {
        assert_eq!(Arity::Standard2.degree(), 2);
        assert_eq!(Arity::Segmented2.degree(), 2);
        assert_eq!(Arity::Standard3.degree(), 3);
        assert_eq!(Arity::Segmented3.degree(), 3);
        assert_eq!(Arity::Standard4.degree(), 4);
        assert_eq!(Arity::Segmented4.degree(), 4);
    }

    #[test]
    fn default_arity_is_segmented4() {
        assert_eq!(Arity::default(), Arity::Segmented4);
    }

    #[test]
    fn recommended_seg4_is_power_of_2_and_k_times_seg() {
        let p = FilterParams::recommended(Arity::Segmented4, 1000, 32).unwrap();
        assert_eq!(p.num_buckets & (p.num_buckets - 1), 0);
        assert_eq!(p.num_buckets, 4 * p.seg());
        // seg is a power of 2, at least as large as ceil(num_items / (load * b)).
        let seg = p.seg();
        assert_eq!(seg & (seg - 1), 0);
    }

    #[test]
    fn recommended_seg3_is_3_times_pow2() {
        let p = FilterParams::recommended(Arity::Segmented3, 500, 32).unwrap();
        assert_eq!(p.num_buckets % 3, 0);
        let seg = p.num_buckets / 3;
        assert_eq!(seg & (seg - 1), 0); // seg is power of 2
    }

    #[test]
    fn recommended_rejects_standard_arities() {
        for arity in [Arity::Standard2, Arity::Standard3, Arity::Standard4] {
            assert_eq!(
                FilterParams::recommended(arity, 100, 32),
                Err(ParamError::ArityNotSupportedForPir)
            );
        }
    }

    #[test]
    fn empty_inputs_rejected() {
        assert!(FilterParams::recommended(Arity::Segmented4, 0, 32).is_err());
        assert!(FilterParams::recommended(Arity::Segmented4, 100, 0).is_err());
    }

    #[test]
    fn scale_is_consistent_with_mat_elem_bit_len() {
        let p = FilterParams::recommended(Arity::Segmented4, 1000, 32).unwrap();
        assert_eq!(p.scale(), 1u32 << (32 - p.mat_elem_bit_len));
    }

    #[test]
    fn row_elems_is_large_enough() {
        let p = FilterParams::recommended(Arity::Segmented4, 1000, 32).unwrap();
        let total_bits = p.row_elems as u64 * p.mat_elem_bit_len as u64;
        assert!(total_bits >= p.value_len as u64 * 8);
    }

    #[test]
    fn seg_is_num_buckets_over_degree() {
        for &arity in &[Arity::Segmented2, Arity::Segmented3, Arity::Segmented4] {
            let p = FilterParams::recommended(arity, 500, 32).unwrap();
            assert_eq!(p.seg(), p.num_buckets / arity.degree());
            assert!(p.seg() >= 1);
        }
    }

    #[test]
    fn next_power_of_2_cases() {
        assert_eq!(next_power_of_2_u64(0), 1);
        assert_eq!(next_power_of_2_u64(1), 1);
        assert_eq!(next_power_of_2_u64(2), 2);
        assert_eq!(next_power_of_2_u64(3), 4);
        assert_eq!(next_power_of_2_u64(1024), 1024);
        assert_eq!(next_power_of_2_u64(1025), 2048);
    }

    #[test]
    fn row_elems_bucket_is_b_times_slot_elems() {
        let p = FilterParams::recommended(Arity::Segmented4, 1000, 32).unwrap();
        let slot_elems = p.fp_elems() + p.val_elems();
        assert_eq!(p.row_elems_bucket(), p.slots_per_bucket() * slot_elems);
        assert_eq!(p.val_elems(), p.row_elems);
    }

    #[test]
    fn noise_budget_uses_seg_not_num_buckets() {
        // For seg=512, B should satisfy 2·seg·P² < 2^32.
        let p = FilterParams::recommended(Arity::Segmented4, 1000, 32).unwrap();
        let seg = p.seg() as u64;
        let peak = 2u64 * seg * (1u64 << (2 * p.mat_elem_bit_len));
        assert!(
            peak < (1u64 << 32),
            "noise budget violated: 2·{seg}·P² = {peak}"
        );
    }
}
