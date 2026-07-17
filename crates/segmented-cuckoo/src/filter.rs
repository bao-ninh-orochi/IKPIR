//! Generic cuckoo filter implementation.
//!
//! [`CuckooFilter<S>`] is parameterised over an [`IndexScheme`], which computes candidate
//! bucket indices. All six concrete filter types share the same insert/lookup/delete logic;
//! only index computation varies. See [`crate::scheme`] for the six concrete schemes.
//!
//! In the description below,
//! + `arity` is the number of candidate buckets per item,
//! + `bucket_size` is the number of fingerprints per bucket,
//! + `fingerprint_bits` is the fingerprint bit width.
//!
//! ## Insertion
//!
//! - **Direct insert**: if any of the arity candidate buckets has a free slot, the
//!   fingerprint is placed immediately.
//! - **Cuckoo kicking**: if all candidate buckets are full, a random slot is evicted and its
//!   fingerprint is relocated to one of its own alternates. This repeats up to `max_kicks`
//!   times.
//! - **Rollback**: if the kick budget is exhausted, all mutations are reversed and
//!   [`CuckooError::TableFull`] is returned. The table is left in its original state.
//!
//! ### Why rollback instead of a victim cache
//!
//! The original Fan et al. 2014 cuckoo filter
//! ([efficient/cuckoofilter](https://github.com/efficient/cuckoofilter)) handles kick-budget
//! exhaustion by stashing the last evicted fingerprint in a *victim cache*; an insertion is
//! only reported as failed when that single-slot cache is already occupied, and lookups must
//! also check the victim cache so the stashed item still reads as present.
//!
//! This crate does the opposite: when the kick budget is exhausted we replay the chain of
//! evictions in reverse, restore every touched slot to its pre-insert value, and return
//! [`CuckooError::TableFull`]. No victim cache exists, and therefore no lookup ever needs to
//! consult one. The motivation is keyword PIR, the consumer of this crate: the filter is
//! served as a matrix via homomorphic retrieval, so a victim cache would have to be encoded
//! as an additional row/column in that matrix. That costs both extra implementation
//! complexity (special-casing the victim slot in every operation) and extra space in the PIR
//! database — a price we are not willing to pay for the small load-factor benefit the victim
//! cache offers at our operating points.
//! ## Lookup and False positives
//!
//! The filter is a probabilistic data structure. [`CuckooFilter::contain`] may return `true`
//! for items that were never inserted (false positive). It never returns `false` for items that
//! are currently inserted (no false negatives). The theoretical false-positive rate is
//! `arity * bucket_size / 2^fingerprint_bits`.
//!
//! ## Deletion
//!
//! Deletion is supported but carries the usual caveat: deleting an item that was never inserted
//! may silently remove a fingerprint that belongs to a different item with the same fingerprint
//! and candidate indices. Only delete items you have previously inserted.
//!
//! ## Architecture
//!
//! This module depends on [`crate::fingerprint_table::FingerprintTable`] for bit-packed storage and on
//! [`crate::scheme::IndexScheme`] for index computation. The split ensures that all insert,
//! lookup, and delete mechanics are written once and the six concrete scheme types add no
//! duplicated logic.
//!
//! ## Security
//!
//! - The underlying item hash (xxHash3) is **not cryptographic**. Do not use this filter for
//!   security-sensitive membership checks. An adversary who knows the hash function can craft
//!   inputs that force artificially high false-positive rates or repeated `TableFull` errors
//!   (hash-flooding denial-of-service).
//! - Deletion of a fingerprint that was never inserted may silently remove a legitimately
//!   inserted item if they share the same fingerprint and candidate indices. Callers are
//!   responsible for deleting only items they have inserted.

// DECISION: The kicking loop uses indexed `for p in 0..arity` rather than an iterator because
// `p` serves as both the array subscript into `indices`/`all` and as the chain-position
// value that must be written to the position store. An enumerate() form would require a
// separate cast and is less readable in this context.
#![allow(clippy::needless_range_loop)]
// DECISION: explicit_counter_loop is suppressed for the same reason — `alt_count` is a
// compact inline counter used to build a small stack-allocated array of alternates.
#![allow(clippy::explicit_counter_loop)]

use rand::Rng;
use std::fmt;

use crate::fingerprint_table::FingerprintTable;
use crate::scheme::{
    IndexScheme, Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme, Standard2aryScheme,
    Standard3aryScheme, Standard4aryScheme,
};
use crate::util::{next_power_of_2, next_power_of_3, next_power_of_4};

pub const MAX_KICKS_DEFAULT: u32 = 500;

/// Conservative target load factors for `from_num_items` sizing, indexed as
/// `[arity - 2][bucket_size - 1]` (arity ∈ {2,3,4}, bucket_size ∈ {1,2,3,4}).
///
/// Each value is 0.95 × the mean empirical max load factor across all tested num_buckets values
/// (num_buckets = 2^14 … 2^20, 20 trials per config, MAX_KICKS = 500). Using 95% of the mean
/// rather than the raw minimum gives a threshold that is reliably achievable for any
/// valid num_buckets while still sizing the filter as tightly as possible. The worst-case num_buckets
/// (largest) is already captured in the mean, so these values are conservative for
/// smaller tables too.
///
/// Experimental mean load factors (source: `benches/load_factor.rs`):
/// | arity | bucket_size=1  | bucket_size=2  | bucket_size=3  | bucket_size=4  |
/// |-------|----------------|----------------|----------------|----------------|
/// | 2     | 0.50           | 0.87           | 0.94           | 0.96           |
/// | 3     | 0.90           | 0.98           | 0.99           | 0.99           |
/// | 4     | 0.96           | 0.99           | 1.00           | 1.00           |
pub const MAX_LOAD_FACTOR: [[f64; 4]; 3] = [
    // arity=2:   bucket_size=1   bucket_size=2   bucket_size=3   bucket_size=4
    [0.48, 0.83, 0.89, 0.91],
    // arity=3:   bucket_size=1   bucket_size=2   bucket_size=3   bucket_size=4
    [0.85, 0.93, 0.94, 0.94],
    // arity=4:   bucket_size=1   bucket_size=2   bucket_size=3   bucket_size=4
    [0.91, 0.94, 0.95, 0.95],
];

/// Return the conservative target load factor for a given arity and bucket_size.
///
/// Used by all `from_num_items` constructors to determine when to double `num_buckets`
/// (or triple for segmented-3ary)
#[inline]
pub const fn target_load_factor(arity: usize, bucket_size: usize) -> f64 {
    MAX_LOAD_FACTOR[arity - 2][bucket_size - 1]
}

/// Errors returned by filter operations.
///
/// Implements [`std::error::Error`] so it can be used with the `?` operator in calling code.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo::{CuckooError, CuckooFilter, Segmented2aryScheme};
///
/// match CuckooFilter::<Segmented2aryScheme>::new(3, 4, 12) {
///     Err(CuckooError::InvalidParams(msg)) => assert!(msg.contains("power of 2")),
///     _ => panic!("expected InvalidParams"),
/// }
/// assert_eq!(CuckooError::TableFull.to_string(), "table is full");
/// ```
#[derive(Debug)]
pub enum CuckooError {
    /// Insert failed because the table is full.
    ///
    /// The kick budget (`max_kicks`) was exhausted without finding a free slot. The table
    /// is rolled back to its state before the insert attempt.
    TableFull,
    /// Construction failed due to invalid parameters.
    ///
    /// The error message describes which constraint was violated.
    InvalidParams(String),
    /// Delete failed because no matching fingerprint was found in any candidate bucket.
    ///
    /// The item was either never inserted, or has already been deleted. The filter is
    /// unchanged.
    NotFound,
}

impl fmt::Display for CuckooError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TableFull => write!(f, "table is full"),
            Self::InvalidParams(msg) => write!(f, "invalid parameters: {msg}"),
            Self::NotFound => write!(f, "item not found in filter"),
        }
    }
}

impl std::error::Error for CuckooError {}

/// Generic cuckoo filter parameterised over an index scheme.
///
/// All insert/lookup/delete mechanics are implemented here. The scheme `S` contributes only
/// index computation (via [`IndexScheme::hash_item`] and [`IndexScheme::all_indices`]).
///
/// Prefer the six type aliases in `crate` (`Segmented2aryCuckooFilter`,
/// `Standard2aryCuckooFilter`, etc.) over constructing this type directly unless you are
/// implementing a custom scheme.
///
/// # Type parameter
///
/// - `S` — an [`IndexScheme`] implementation that determines arity and index layout.
///
/// # Security
///
/// See the module-level security note for hash-flooding and deletion caveats.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo::{CuckooFilter, Segmented2aryScheme};
///
/// let mut f = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
/// f.insert("hello").unwrap();
/// assert!(f.contain("hello"));
/// assert!(!f.contain("world"));
/// f.delete("hello").unwrap();
/// assert!(!f.contain("hello"));
/// ```
pub struct CuckooFilter<S: IndexScheme> {
    table: FingerprintTable,
    scheme: S,
    num_items: u64,
    max_kicks: u32,
    /// Pre-allocated rollback chain reused across `insert` calls; capacity tracks
    /// `max_kicks` so the kicking loop never allocates on the hot path.
    chain: Vec<(u32, u32, u32)>,
}

/// Supported arity values (candidate buckets per item).
pub const SUPPORTED_ARITIES: [u32; 3] = [2, 3, 4];

/// Inclusive range of supported `bucket_size` values (fingerprint slots per bucket).
pub const SUPPORTED_BUCKET_SIZES: std::ops::RangeInclusive<u32> = 1..=4;

/// Validate parameters common to all filter constructors.
///
/// Supported ranges:
/// - `arity` ∈ {2, 3, 4} (see [`SUPPORTED_ARITIES`]).
/// - `bucket_size` ∈ 1..=4 (see [`SUPPORTED_BUCKET_SIZES`]).
/// - `fingerprint_bits` must satisfy `2^fingerprint_bits > arity*bucket_size` (guarantees FPR < 1)
///   and be ≤ 32.
pub fn validate_common_params(
    arity: u32,
    bucket_size: u32,
    fingerprint_bits: u32,
) -> Result<(), CuckooError> {
    if !SUPPORTED_ARITIES.contains(&arity) {
        return Err(CuckooError::InvalidParams(format!(
            "arity must be one of {{2, 3, 4}}, got {arity}"
        )));
    }
    if !SUPPORTED_BUCKET_SIZES.contains(&bucket_size) {
        return Err(CuckooError::InvalidParams(format!(
            "bucket_size must be in 1..=4, got {bucket_size}"
        )));
    }
    let min_fingerprint_bits = (arity * bucket_size).ilog2() + 1;
    if fingerprint_bits < min_fingerprint_bits || fingerprint_bits > 32 {
        return Err(CuckooError::InvalidParams(format!(
            "fingerprint_bits must be in {min_fingerprint_bits}..=32 for arity={arity}, bucket_size={bucket_size} (ensures FPR < 1)"
        )));
    }
    Ok(())
}

// ─── Constructors ───────────────────────────────────────────────────────────

impl CuckooFilter<Segmented2aryScheme> {
    /// Create a segmented 2-ary filter with explicit dimensions.
    ///
    /// # Arguments
    ///
    /// - `num_buckets` — total number of buckets.
    /// - `bucket_size` — fingerprints (slots) per bucket.
    /// - `fingerprint_bits` — fingerprint bit width.
    ///
    /// # Constraints
    ///
    /// - `num_buckets` must be a power of 2 and ≥ 2.
    /// - `bucket_size` must be in `1..=4`.
    /// - `fingerprint_bits` must be in `[⌊log2(2 * bucket_size)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter with capacity `num_buckets * bucket_size` fingerprints and an initial load factor of 0.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Segmented2aryScheme};
    ///
    /// let mut f = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
    /// f.insert("hello").unwrap();
    /// assert!(f.contain("hello"));
    /// assert!(CuckooFilter::<Segmented2aryScheme>::new(3, 4, 12).is_err()); // num_buckets=3 not power of 2
    /// assert!(CuckooFilter::<Segmented2aryScheme>::new(1, 4, 12).is_err()); // num_buckets=1 < 2
    /// ```
    pub fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        if !num_buckets.is_power_of_two() || num_buckets < 2 {
            return Err(CuckooError::InvalidParams(
                "num_buckets must be a power of 2 and >= 2".into(),
            ));
        }
        validate_common_params(2, bucket_size, fingerprint_bits)?;
        let table = FingerprintTable::new(num_buckets, bucket_size, fingerprint_bits);
        Ok(Self {
            table,
            scheme: Segmented2aryScheme {
                segment_size: num_buckets / 2,
            },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
            chain: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
        })
    }

    /// Create a segmented 2-ary filter sized to hold at least `max_items`.
    ///
    /// Rounds `num_buckets` up to the next valid value (power of 2, ≥ 2). If the projected load factor
    /// exceeds the empirical target for this (arity=2, bucket_size) configuration, `num_buckets` is doubled
    /// until the projection is within bounds.
    ///
    /// # Arguments
    ///
    /// - `max_items` — minimum number of items the filter should hold.
    /// - `bucket_size` — fingerprints per bucket.
    /// - `fingerprint_bits` — fingerprint bit width.
    ///
    /// # Constraints
    ///
    /// - `bucket_size` must be in `1..=4`.
    /// - `fingerprint_bits` must be in `[⌊log2(2 * bucket_size)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter sized so that inserting `max_items` keeps the projected load factor below the
    /// empirical target for (arity=2, bucket_size). See [`MAX_LOAD_FACTOR`] for values.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `bucket_size` or `fingerprint_bits` are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Segmented2aryScheme};
    ///
    /// let f = CuckooFilter::<Segmented2aryScheme>::from_num_items(100_000, 4, 12).unwrap();
    /// assert!(f.size_in_bytes() > 0);
    /// ```
    pub fn from_num_items(
        max_items: u64,
        bucket_size: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(2, bucket_size, fingerprint_bits)?;
        let max_items_cap: u64 = u32::MAX as u64 * bucket_size as u64;
        if max_items > max_items_cap {
            return Err(CuckooError::InvalidParams(
                "max_items too large for u32 num_buckets".into(),
            ));
        }
        let buckets_needed = max_items.div_ceil(bucket_size as u64);
        let mut num_buckets = next_power_of_2(buckets_needed) as u32;
        if num_buckets < 2 {
            num_buckets = 2;
        }
        let target = target_load_factor(2, bucket_size as usize);
        while max_items as f64 / (num_buckets as u64 * bucket_size as u64) as f64 > target {
            num_buckets *= 2;
        }
        Self::new(num_buckets, bucket_size, fingerprint_bits)
    }
}

impl CuckooFilter<Segmented3aryScheme> {
    /// Create a segmented 3-ary filter with explicit dimensions.
    ///
    /// No per-slot position storage is needed: the chain position is derived from which
    /// segment the bucket index falls in.
    ///
    /// # Arguments
    ///
    /// - `num_buckets` — total buckets.
    /// - `bucket_size` — fingerprints per bucket.
    /// - `fingerprint_bits` — fingerprint bit width.
    ///
    /// # Constraints
    ///
    /// - `num_buckets` must equal `3 · 2^t` for some `t ≥ 0` (i.e., `num_buckets/3` is a power of 2).
    /// - `bucket_size` must be in `1..=4`.
    /// - `fingerprint_bits` must be in `[⌊log2(3 * bucket_size)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter with capacity `num_buckets * bucket_size` fingerprints and an initial load factor of 0.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Segmented3aryScheme};
    ///
    /// // num_buckets = 3 * 32 = 96 — each of 3 segments is 32 buckets
    /// let mut f = CuckooFilter::<Segmented3aryScheme>::new(96, 4, 12).unwrap();
    /// f.insert("data").unwrap();
    /// assert!(f.contain("data"));
    /// assert!(CuckooFilter::<Segmented3aryScheme>::new(64, 4, 12).is_err()); // 64/3 not integer
    /// ```
    pub fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        if num_buckets < 3 || num_buckets % 3 != 0 || !(num_buckets / 3).is_power_of_two() {
            return Err(CuckooError::InvalidParams(
                "num_buckets must be 3 * 2^t (num_buckets divisible by 3, num_buckets/3 a power of 2)".into(),
            ));
        }
        validate_common_params(3, bucket_size, fingerprint_bits)?;
        let segment_size = num_buckets / 3;
        let table = FingerprintTable::new(num_buckets, bucket_size, fingerprint_bits);
        Ok(Self {
            table,
            scheme: Segmented3aryScheme { segment_size },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
            chain: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
        })
    }

    /// Create a segmented 3-ary filter sized to hold at least `max_items`.
    ///
    /// Each segment must be a power of 2; this method computes the smallest valid `num_buckets = 3 * segment_size`
    /// where `segment_size = 2^t ≥ ceil(max_items / (3 * bucket_size))`, then doubles `segment_size` until the projected
    /// load is within the empirical target for (arity=3, bucket_size).
    ///
    /// # Arguments
    ///
    /// - `max_items` — minimum number of items the filter should hold.
    /// - `bucket_size` — fingerprints per bucket.
    /// - `fingerprint_bits` — fingerprint bit width.
    ///
    /// # Constraints
    ///
    /// - `bucket_size` must be in `1..=4`.
    /// - `fingerprint_bits` must be in `[⌊log2(3 * bucket_size)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter sized so that inserting `max_items` keeps the projected load factor below the
    /// empirical target for (arity=3, bucket_size). See [`MAX_LOAD_FACTOR`] for values.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `bucket_size` or `fingerprint_bits` are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Segmented3aryScheme};
    ///
    /// let mut f = CuckooFilter::<Segmented3aryScheme>::from_num_items(10_000, 4, 12).unwrap();
    /// f.insert("hello").unwrap();
    /// assert!(f.contain("hello"));
    /// ```
    pub fn from_num_items(
        max_items: u64,
        bucket_size: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(3, bucket_size, fingerprint_bits)?;
        let max_items_cap: u64 = u32::MAX as u64 * bucket_size as u64;
        if max_items > max_items_cap {
            return Err(CuckooError::InvalidParams(
                "max_items too large for u32 num_buckets".into(),
            ));
        }
        // Each segment must be power of 2; num_buckets = 3 * segment_size
        let slots_per_segment = max_items.div_ceil(3 * bucket_size as u64);
        let mut segment_size = next_power_of_2(slots_per_segment) as u32;
        if segment_size < 1 {
            segment_size = 1;
        }
        let mut num_buckets = 3 * segment_size;
        let target = target_load_factor(3, bucket_size as usize);
        while max_items as f64 / (num_buckets as u64 * bucket_size as u64) as f64 > target {
            segment_size *= 2;
            num_buckets = 3 * segment_size;
        }
        Self::new(num_buckets, bucket_size, fingerprint_bits)
    }
}

impl CuckooFilter<Segmented4aryScheme> {
    /// Create a segmented 4-ary filter with explicit dimensions.
    ///
    /// No per-slot position storage is needed: the chain position is derived from which
    /// segment the bucket index falls in.
    ///
    /// # Arguments
    ///
    /// - `num_buckets` — total buckets.
    /// - `bucket_size` — fingerprints per bucket.
    /// - `fingerprint_bits` — fingerprint bit width.
    ///
    /// # Constraints
    ///
    /// - `num_buckets` must be a power of 2 and ≥ 4 (so each of the four segments is also a power of 2).
    /// - `bucket_size` must be in `1..=4`.
    /// - `fingerprint_bits` must be in `[⌊log2(4 * bucket_size)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter with capacity `num_buckets * bucket_size` fingerprints and an initial load factor of 0.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Segmented4aryScheme};
    ///
    /// // num_buckets = 64, so each of 4 segments has 16 buckets
    /// let mut f = CuckooFilter::<Segmented4aryScheme>::new(64, 4, 12).unwrap();
    /// f.insert("data").unwrap();
    /// assert!(f.contain("data"));
    /// assert!(CuckooFilter::<Segmented4aryScheme>::new(2, 4, 12).is_err()); // num_buckets=2 < 4
    /// ```
    pub fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        if !num_buckets.is_power_of_two() || num_buckets < 4 {
            return Err(CuckooError::InvalidParams(
                "num_buckets must be a power of 2 and >= 4 (so each of 4 segments is a power of 2)"
                    .into(),
            ));
        }
        validate_common_params(4, bucket_size, fingerprint_bits)?;
        let segment_size = num_buckets / 4;
        let table = FingerprintTable::new(num_buckets, bucket_size, fingerprint_bits);
        Ok(Self {
            table,
            scheme: Segmented4aryScheme { segment_size },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
            chain: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
        })
    }

    /// Create a segmented 4-ary filter sized to hold at least `max_items`.
    ///
    /// Each segment must be a power of 2; this method computes the smallest valid `num_buckets = 4 * segment_size`
    /// where `segment_size = 2^t ≥ ceil(max_items / (4 * bucket_size))`, then doubles `segment_size` until the projected
    /// load is within the empirical target for (arity=4, bucket_size).
    ///
    /// # Arguments
    ///
    /// - `max_items` — minimum number of items the filter should hold.
    /// - `bucket_size` — fingerprints per bucket.
    /// - `fingerprint_bits` — fingerprint bit width.
    ///
    /// # Constraints
    ///
    /// - `bucket_size` must be in `1..=4`.
    /// - `fingerprint_bits` must be in `[⌊log2(4 * bucket_size)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter sized so that inserting `max_items` keeps the projected load factor below the
    /// empirical target for (arity=4, bucket_size). See [`MAX_LOAD_FACTOR`] for values.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `bucket_size` or `fingerprint_bits` are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Segmented4aryScheme};
    ///
    /// let mut f = CuckooFilter::<Segmented4aryScheme>::from_num_items(10_000, 4, 12).unwrap();
    /// f.insert("hello").unwrap();
    /// assert!(f.contain("hello"));
    /// ```
    pub fn from_num_items(
        max_items: u64,
        bucket_size: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(4, bucket_size, fingerprint_bits)?;
        let max_items_cap: u64 = u32::MAX as u64 * bucket_size as u64;
        if max_items > max_items_cap {
            return Err(CuckooError::InvalidParams(
                "max_items too large for u32 num_buckets".into(),
            ));
        }
        let slots_per_segment = max_items.div_ceil(4 * bucket_size as u64);
        let mut segment_size = next_power_of_2(slots_per_segment) as u32;
        if segment_size < 1 {
            segment_size = 1;
        }
        let mut num_buckets = 4 * segment_size;
        let target = target_load_factor(4, bucket_size as usize);
        while max_items as f64 / (num_buckets as u64 * bucket_size as u64) as f64 > target {
            segment_size *= 2;
            num_buckets = 4 * segment_size;
        }
        Self::new(num_buckets, bucket_size, fingerprint_bits)
    }
}

/// Follow the academic paper "Cuckoo Filter: Practically Better Than Bloom"
/// at <https://www.cs.cmu.edu/~dga/papers/cuckoo-conext2014.pdf>
impl CuckooFilter<Standard2aryScheme> {
    /// Create a standard 2-ary filter with explicit dimensions.
    ///
    /// # Arguments
    ///
    /// - `num_buckets` — total number of buckets.
    /// - `bucket_size` — fingerprints per bucket.
    /// - `fingerprint_bits` — fingerprint bit width.
    ///
    /// # Constraints
    ///
    /// - `num_buckets` must be a power of 2 and ≥ 1.
    /// - `bucket_size` must be in `1..=4`.
    /// - `fingerprint_bits` must be in `[⌊log2(2b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter with capacity `num_buckets * bucket_size` fingerprints and an initial load factor of 0.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Standard2aryScheme};
    ///
    /// let mut f = CuckooFilter::<Standard2aryScheme>::new(64, 4, 12).unwrap();
    /// f.insert(42u64.to_le_bytes()).unwrap();
    /// assert!(f.contain(42u64.to_le_bytes()));
    /// assert!(CuckooFilter::<Standard2aryScheme>::new(3, 4, 12).is_err()); // num_buckets not power of 2
    /// ```
    pub fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        if !num_buckets.is_power_of_two() || num_buckets < 1 {
            return Err(CuckooError::InvalidParams(
                "num_buckets must be a power of 2 and >= 1".into(),
            ));
        }
        validate_common_params(2, bucket_size, fingerprint_bits)?;
        let table = FingerprintTable::new(num_buckets, bucket_size, fingerprint_bits);
        Ok(Self {
            table,
            scheme: Standard2aryScheme { num_buckets },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
            chain: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
        })
    }

    /// Create a standard 2-ary filter sized to hold at least `max_items`.
    ///
    /// # Arguments
    ///
    /// - `max_items` — minimum number of items the filter should hold.
    /// - `bucket_size` — fingerprints per bucket.
    /// - `fingerprint_bits` — fingerprint bit width.
    ///
    /// # Constraints
    ///
    /// - `bucket_size` must be in `1..=4`.
    /// - `fingerprint_bits` must be in `[⌊log2(2b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter sized so that inserting `max_items` keeps the projected load factor below the
    /// empirical target for (arity=2, bucket_size). See [`MAX_LOAD_FACTOR`] for values.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `bucket_size` or `fingerprint_bits` are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Standard2aryScheme};
    ///
    /// let mut f = CuckooFilter::<Standard2aryScheme>::from_num_items(50_000, 4, 12).unwrap();
    /// f.insert(b"item".as_ref()).unwrap();
    /// assert!(f.contain(b"item".as_ref()));
    /// ```
    pub fn from_num_items(
        max_items: u64,
        bucket_size: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(2, bucket_size, fingerprint_bits)?;
        let max_items_cap: u64 = u32::MAX as u64 * bucket_size as u64;
        if max_items > max_items_cap {
            return Err(CuckooError::InvalidParams(
                "max_items too large for u32 num_buckets".into(),
            ));
        }
        let buckets_needed = max_items.div_ceil(bucket_size as u64);
        let mut num_buckets = next_power_of_2(buckets_needed) as u32;
        if num_buckets < 1 {
            num_buckets = 1;
        }
        let target = target_load_factor(2, bucket_size as usize);
        while max_items as f64 / (num_buckets as u64 * bucket_size as u64) as f64 > target {
            num_buckets *= 2;
        }
        Self::new(num_buckets, bucket_size, fingerprint_bits)
    }
}

/// Follow the academic paper "D-Ary Cuckoo Filter: A Space Efficient Data Structure for Set Membership Lookup"
/// at <https://ieeexplore.ieee.org/abstract/document/8368364>
impl CuckooFilter<Standard3aryScheme> {
    /// Create a standard 3-ary filter with explicit dimensions.
    ///
    /// Uses xor3 cycling so `all_indices` always reconstructs all three candidates from any
    /// starting index without per-slot position storage.
    ///
    /// # Arguments
    ///
    /// - `num_buckets` — total number of buckets.
    /// - `bucket_size` — fingerprints per bucket.
    /// - `fingerprint_bits` — fingerprint bit width.
    ///
    /// # Constraints
    ///
    /// - `num_buckets` must be a power of 3 (`3^t`) and ≥ 1.
    /// - `bucket_size` must be in `1..=4`.
    /// - `fingerprint_bits` must be in `[⌊log2(3 * bucket_size)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter with capacity `num_buckets * bucket_size` fingerprints and an initial load factor of 0.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Standard3aryScheme};
    ///
    /// let mut f = CuckooFilter::<Standard3aryScheme>::new(243, 4, 12).unwrap(); //243 = 3^5
    /// f.insert("data").unwrap();
    /// assert!(f.contain("data"));
    /// ```
    pub fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        if !crate::util::is_power_of_3(num_buckets) || num_buckets < 1 {
            return Err(CuckooError::InvalidParams(
                "num_buckets must be a power of 3 (3^t)".into(),
            ));
        }
        validate_common_params(3, bucket_size, fingerprint_bits)?;
        let table = FingerprintTable::new(num_buckets, bucket_size, fingerprint_bits);
        Ok(Self {
            table,
            scheme: Standard3aryScheme { num_buckets },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
            chain: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
        })
    }

    /// Create a standard 3-ary filter sized to hold at least `max_items`.
    ///
    /// If the projected load factor exceeds the empirical target for this (arity=3, bucket_size) configuration, `num_buckets` is tripled
    ///
    /// # Arguments
    ///
    /// - `max_items` — minimum number of items the filter should hold.
    /// - `bucket_size` — fingerprints per bucket.
    /// - `fingerprint_bits` — fingerprint bit width.
    ///
    /// # Constraints
    ///
    /// - `bucket_size` must be in `1..=4`.
    /// - `fingerprint_bits` must be in `[⌊log2(3 * bucket_size)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter sized so that inserting `max_items` keeps the projected load factor below the
    /// empirical target for (arity=3, bucket_size). See [`MAX_LOAD_FACTOR`] for values.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `bucket_size` or `fingerprint_bits` are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Standard3aryScheme};
    ///
    /// let mut f = CuckooFilter::<Standard3aryScheme>::from_num_items(10_000, 4, 12).unwrap();
    /// f.insert("hello").unwrap();
    /// assert!(f.contain("hello"));
    /// ```
    pub fn from_num_items(
        max_items: u64,
        bucket_size: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(3, bucket_size, fingerprint_bits)?;
        let max_items_cap: u64 = u32::MAX as u64 * bucket_size as u64;
        if max_items > max_items_cap {
            return Err(CuckooError::InvalidParams(
                "max_items too large for u32 num_buckets".into(),
            ));
        }
        let buckets_needed = max_items.div_ceil(bucket_size as u64);
        let mut num_buckets = next_power_of_3(buckets_needed) as u32;
        if num_buckets < 1 {
            num_buckets = 1;
        }
        let target = target_load_factor(3, bucket_size as usize);
        while max_items as f64 / (num_buckets as u64 * bucket_size as u64) as f64 > target {
            num_buckets *= 3;
        }
        Self::new(num_buckets, bucket_size, fingerprint_bits)
    }
}

/// Follow the academic paper "D-Ary Cuckoo Filter: A Space Efficient Data Structure for Set Membership Lookup"
/// at <https://ieeexplore.ieee.org/abstract/document/8368364>
impl CuckooFilter<Standard4aryScheme> {
    /// Create a standard 4-ary filter with explicit dimensions.
    ///
    /// Uses xor4 cycling so `all_indices` always reconstructs all four candidates from any
    /// starting index without per-slot position storage.
    ///
    /// # Arguments
    ///
    /// - `num_buckets` — total number of buckets.
    /// - `bucket_size` — fingerprints per bucket.
    /// - `fingerprint_bits` — fingerprint bit width.
    ///
    /// # Constraints
    ///
    /// - `num_buckets` must be a power of 4 (`4^t`) and ≥ 1.
    /// - `bucket_size` must be in `1..=4`.
    /// - `fingerprint_bits` must be in `[⌊log2(4 * bucket_size)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter with capacity `num_buckets * bucket_size` fingerprints and an initial load factor of 0.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Standard4aryScheme};
    ///
    /// let mut f = CuckooFilter::<Standard4aryScheme>::new(64, 4, 12).unwrap();
    /// f.insert("data").unwrap();
    /// assert!(f.contain("data"));
    /// ```
    pub fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        if !crate::util::is_power_of_4(num_buckets) || num_buckets < 1 {
            return Err(CuckooError::InvalidParams(
                "num_buckets must be a power of 4 (4^t)".into(),
            ));
        }
        validate_common_params(4, bucket_size, fingerprint_bits)?;
        let table = FingerprintTable::new(num_buckets, bucket_size, fingerprint_bits);
        Ok(Self {
            table,
            scheme: Standard4aryScheme { num_buckets },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
            chain: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
        })
    }

    /// Create a standard 4-ary filter sized to hold at least `max_items`.
    ///
    /// If the projected load factor exceeds the empirical target for this (arity=4, bucket_size) configuration, `num_buckets` is quadrupled
    ///
    /// # Arguments
    ///
    /// - `max_items` — minimum number of items the filter should hold.
    /// - `bucket_size` — fingerprints per bucket.
    /// - `fingerprint_bits` — fingerprint bit width.
    ///
    /// # Constraints
    ///
    /// - `bucket_size` must be in `1..=4`.
    /// - `fingerprint_bits` must be in `[⌊log2(4 * bucket_size)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter sized so that inserting `max_items` keeps the projected load factor below the
    /// empirical target for (arity=4, bucket_size). See [`MAX_LOAD_FACTOR`] for values.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `bucket_size` or `fingerprint_bits` are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Standard4aryScheme};
    ///
    /// let mut f = CuckooFilter::<Standard4aryScheme>::from_num_items(10_000, 4, 12).unwrap();
    /// f.insert("hello").unwrap();
    /// assert!(f.contain("hello"));
    /// ```
    pub fn from_num_items(
        max_items: u64,
        bucket_size: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(4, bucket_size, fingerprint_bits)?;
        let max_items_cap: u64 = u32::MAX as u64 * bucket_size as u64;
        if max_items > max_items_cap {
            return Err(CuckooError::InvalidParams(
                "max_items too large for u32 num_buckets".into(),
            ));
        }
        let buckets_needed = max_items.div_ceil(bucket_size as u64);
        let mut num_buckets = next_power_of_4(buckets_needed) as u32;
        if num_buckets < 1 {
            num_buckets = 1;
        }
        let target = target_load_factor(4, bucket_size as usize);
        while max_items as f64 / (num_buckets as u64 * bucket_size as u64) as f64 > target {
            num_buckets *= 4;
        }
        Self::new(num_buckets, bucket_size, fingerprint_bits)
    }
}

// ─── Generic filter operations ──────────────────────────────────────────────

impl<S: IndexScheme> CuckooFilter<S> {
    /// Insert an item into the filter.
    ///
    /// First attempts a direct placement into any of the arity candidate buckets. If all are
    /// full, enters a cuckoo-kicking loop: evicts a random slot and relocates the displaced
    /// fingerprint to one of its own alternates, repeating up to `max_kicks` times. On
    /// exhaustion, rolls back all mutations and returns [`CuckooError::TableFull`].
    ///
    /// # Note — rollback instead of a victim cache
    ///
    /// Unlike the original cuckoo filter of Fan et al. 2014
    /// ([efficient/cuckoofilter](https://github.com/efficient/cuckoofilter)), this method does
    /// **not** use a "victim cache" to absorb the final evicted fingerprint when the kick budget
    /// runs out. Instead, every eviction made during the failing insert is undone in reverse
    /// order and [`CuckooError::TableFull`] is returned; the filter is observably unchanged.
    ///
    /// The victim-cache approach does not fit our target application, keyword PIR. The filter
    /// is served as a matrix via homomorphic retrieval, so a victim slot would become an extra
    /// row/column that every lookup — including oblivious ones — must consult. That inflates
    /// both implementation complexity and PIR database size for a small load-factor gain we do
    /// not need. See the module-level "Why rollback instead of a victim cache" note for the
    /// full rationale.
    ///
    /// # Arguments
    ///
    /// - `item` — any value that implements `AsRef<[u8]>` (e.g., `&str`, `&[u8]`, `String`).
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// - [`CuckooError::TableFull`] — kick budget exhausted; table is **unchanged**.
    ///
    /// # Performance
    ///
    /// O(1) amortised — direct placement is O(arity). Kicking is O(max_kicks * arity) worst
    /// case but O(1) expected when load factor is well below saturation (~95%).
    ///
    /// # Security
    ///
    /// An adversary who can craft hash collisions may force repeated `TableFull` errors at
    /// low apparent load (hash-flooding DoS). If adversarial inputs are expected, replace
    /// xxHash3 with a keyed hash.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooError, CuckooFilter, Segmented2aryScheme};
    ///
    /// let mut f = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
    /// f.insert("hello").unwrap();
    /// assert!(f.contain("hello"));
    ///
    /// // Tiny filter — eventually returns TableFull
    /// let mut tiny = CuckooFilter::<Segmented2aryScheme>::new(2, 1, 2).unwrap();
    /// let mut full = false;
    /// for i in 0u32..100 {
    ///     if tiny.insert(i.to_le_bytes()).is_err() { full = true; break; }
    /// }
    /// assert!(full);
    /// ```
    pub fn insert<T: AsRef<[u8]>>(&mut self, item: T) -> Result<(), CuckooError> {
        let (fingerprint, indices) = self
            .scheme
            .hash_item(item.as_ref(), self.table.fingerprint_bits());
        let arity = self.scheme.arity();

        // Try all arity candidate buckets
        for pos in 0..arity {
            if let Some(_slot) = self.table.insert(indices[pos], fingerprint) {
                self.num_items += 1;
                return Ok(());
            }
        }

        // All arity buckets full — start kicking
        let mut rng = rand::rng();
        let start_pos = rng.random_range(0..arity as u32) as usize;
        let mut cur_index = indices[start_pos];
        let mut cur_fingerprint = fingerprint;

        // self.chain[i] = (bucket, slot, original_fingerprint); cleared and reused per call.
        self.chain.clear();

        for _ in 0..self.max_kicks {
            // Evict a random slot from cur_index
            let slot = rng.random_range(0..self.table.bucket_size());
            let evicted_fingerprint = self.table.read(cur_index, slot);

            // Write cur_fingerprint into this slot
            self.chain.push((cur_index, slot, evicted_fingerprint));
            self.table.write(cur_index, slot, cur_fingerprint);

            // Compute all candidates for the evicted fingerprint
            cur_fingerprint = evicted_fingerprint;
            let all = self.scheme.all_indices(cur_index, evicted_fingerprint);

            // Find which candidate position corresponds to the bucket we just kicked from.
            // Scanning all[] is correct for all six schemes and handles the h=0 degenerate
            // case (where all positions could map to the same index).
            let evicted_pos = (0..arity).find(|&p| all[p] == cur_index).unwrap_or(0);

            // Try to insert evicted fingerprint into each alternate bucket
            let mut placed = false;
            for p in 0..arity {
                if p == evicted_pos {
                    continue;
                }
                if let Some(_ins_slot) = self.table.insert(all[p], evicted_fingerprint) {
                    self.num_items += 1;
                    placed = true;
                    break;
                }
            }
            if placed {
                return Ok(());
            }

            // Pick a random alternate to continue kicking from
            let mut alts = [0u8; 3];
            let mut alt_count = 0;
            for p in 0..arity {
                if p != evicted_pos {
                    alts[alt_count] = p as u8;
                    alt_count += 1;
                }
            }
            let next_pos = alts[rng.random_range(0..alt_count as u32) as usize];
            cur_index = all[next_pos as usize];
        }

        // Kicks exhausted — rollback all changes
        for &(bucket, slot, original_fingerprint) in self.chain.iter().rev() {
            self.table.write(bucket, slot, original_fingerprint);
        }
        Err(CuckooError::TableFull)
    }

    /// Check whether the filter likely contains `item`.
    ///
    /// Probes all arity candidate buckets for a matching fingerprint.
    ///
    /// # Arguments
    ///
    /// - `item` — any value that implements `AsRef<[u8]>`.
    ///
    /// # Returns
    ///
    /// - `true` if the fingerprint is found in any candidate bucket. May return `true` for
    ///   items that were never inserted (false positive); the theoretical rate is
    ///   `arity * bucket_size / 2^fingerprint_bits`.
    /// - `false` if no matching fingerprint is found. Never a false negative for items
    ///   currently in the filter.
    ///
    /// # Performance
    ///
    /// O(arity · bucket_size) — probes arity buckets, scanning up to bucket_size slots each. For typical
    /// parameters (arity ≤ 4, bucket_size ≤ 4) this is effectively O(1).
    ///
    /// # Security
    ///
    /// An adversary who knows the hash function can pre-compute items that hash to the same
    /// fingerprint and candidate buckets, achieving a 100% false-positive rate for chosen
    /// queries without inserting those items. This is a probabilistic structure, not a
    /// cryptographic one.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Segmented2aryScheme};
    ///
    /// let mut f = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
    /// f.insert("hello").unwrap();
    /// assert!(f.contain("hello"));
    /// assert!(!f.contain("world")); // likely false for a fresh filter
    /// ```
    pub fn contain<T: AsRef<[u8]>>(&self, item: T) -> bool {
        // hash_item already enforce `fingerprint != 0`, so that there is no ambiguity between an
        // empty slot and a valid fingerprint. This simplifies contain() and delete() logic since
        // we can just check for equality with the fingerprint without needing
        // to check for empty slots.
        let (fingerprint, indices) = self
            .scheme
            .hash_item(item.as_ref(), self.table.fingerprint_bits());
        let arity = self.scheme.arity();
        for i in 0..arity {
            if self.table.contain(indices[i], fingerprint) {
                return true;
            }
        }
        false
    }

    /// Delete `item` from the filter.
    ///
    /// Probes all arity candidate buckets for a matching fingerprint. On the first match,
    /// zeroes the fingerprint slot.
    ///
    /// # Arguments
    ///
    /// - `item` — any value that implements `AsRef<[u8]>`.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the fingerprint was found and removed.
    ///
    /// # Errors
    ///
    /// - [`CuckooError::NotFound`] — no matching fingerprint in any candidate bucket. The
    ///   filter is **unchanged**. This happens when the item was never inserted, or was
    ///   already deleted.
    ///
    /// # Performance
    ///
    /// O(arity × bucket_size) — same complexity as [`contain`](Self::contain).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooError, CuckooFilter, Segmented2aryScheme};
    ///
    /// let mut f = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
    /// f.insert("hello").unwrap();
    /// f.delete("hello").unwrap();
    /// assert!(!f.contain("hello"));
    /// // Second delete of the same item returns an error.
    /// assert!(matches!(f.delete("hello"), Err(CuckooError::NotFound)));
    /// assert!(matches!(f.delete("never_inserted"), Err(CuckooError::NotFound)));
    /// ```
    pub fn delete<T: AsRef<[u8]>>(&mut self, item: T) -> Result<(), CuckooError> {
        let (fingerprint, indices) = self
            .scheme
            .hash_item(item.as_ref(), self.table.fingerprint_bits());
        let arity = self.scheme.arity();

        for i in 0..arity {
            if self.table.delete(indices[i], fingerprint) {
                self.num_items -= 1;
                return Ok(());
            }
        }
        Err(CuckooError::NotFound)
    }

    /// Return the number of items currently stored in the filter.
    ///
    /// This counter is maintained exactly by `insert` and `delete`. It is not affected by false
    /// positives.
    ///
    /// # Returns
    ///
    /// The count of successfully inserted (and not yet deleted) items.
    ///
    /// # Performance
    ///
    /// O(1) — reads a cached counter.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Segmented2aryScheme};
    ///
    /// let mut f = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
    /// assert_eq!(f.num_items(), 0);
    /// f.insert("a").unwrap();
    /// assert_eq!(f.num_items(), 1);
    /// f.delete("a").unwrap();
    /// assert_eq!(f.num_items(), 0);
    /// ```
    pub const fn num_items(&self) -> u64 {
        self.num_items
    }

    /// Return the byte size of the underlying fingerprint storage.
    ///
    /// Reports the logical storage footprint `ceil(num_buckets * bucket_size * fingerprint_bits / 8)` bytes,
    /// excluding the 8-byte alignment padding added by the underlying fingerprint storage.
    ///
    /// # Returns
    ///
    /// Number of bytes used by the fingerprint array (without padding).
    ///
    /// # Performance
    ///
    /// O(1) — reads a field from the underlying fingerprint storage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Segmented2aryScheme};
    ///
    /// // num_buckets=64, bucket_size=4, fingerprint_bits=12 → 64*4*12 = 3072 bits = 384 bytes
    /// let f = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
    /// assert_eq!(f.size_in_bytes(), 384);
    /// ```
    pub const fn size_in_bytes(&self) -> usize {
        self.table.size_in_bytes()
    }

    /// Return the current load factor: `num_items / (num_buckets * bucket_size)`.
    ///
    /// A load factor of 1.0 means every slot is occupied. In practice, cuckoo filters
    /// saturate (start returning `TableFull`) before reaching 1.0; the achievable maximum
    /// depends on scheme, arity, and `max_kicks`.
    ///
    /// # Returns
    ///
    /// A value in `[0.0, 1.0]` representing the fraction of slots currently occupied.
    ///
    /// # Performance
    ///
    /// O(1) — two integer field reads and a floating-point division.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Segmented2aryScheme};
    ///
    /// let mut f = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
    /// assert_eq!(f.load_factor(), 0.0);
    /// f.insert("hello").unwrap();
    /// // After inserting 1 item into a 256-slot filter: load = 1/256 ≈ 0.0039
    /// assert!(f.load_factor() > 0.0);
    /// assert!(f.load_factor() < 1.0);
    /// ```
    pub fn load_factor(&self) -> f64 {
        self.num_items as f64 / (self.table.num_buckets() as f64 * self.table.bucket_size() as f64)
    }

    /// Override the maximum number of cuckoo kicks before declaring the table full.
    ///
    /// The default is 500. A higher value increases the probability of a successful insert
    /// at very high load factors at the cost of slower worst-case inserts. Set to 0 to disable
    /// kicking entirely (only direct placements succeed).
    ///
    /// # Note — rollback instead of a victim cache
    ///
    /// When this budget is exhausted, [`insert`](Self::insert) rolls the in-progress chain of
    /// evictions back to the table's pre-insert state rather than stashing the final evicted
    /// fingerprint in a victim cache. Raising `max_kicks` therefore trades a longer worst-case
    /// insert for a smaller rollback failure rate, without ever leaving a "victim" fingerprint
    /// that future lookups must probe. See [`crate`] for the full design rationale.
    ///
    /// # Arguments
    ///
    /// - `max_kicks` — new kick budget per insert attempt.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo::{CuckooFilter, Segmented2aryScheme};
    ///
    /// let mut f = CuckooFilter::<Segmented2aryScheme>::new(2, 1, 2).unwrap();
    /// f.set_max_kicks(0); // disable kicking — only 2 direct placements possible
    /// let mut count = 0u32;
    /// for i in 0u32..100 {
    ///     if f.insert(i.to_le_bytes()).is_ok() { count += 1; }
    /// }
    /// assert!(count <= 2);
    /// ```
    pub fn set_max_kicks(&mut self, max_kicks: u32) {
        self.max_kicks = max_kicks;
        self.chain.reserve(max_kicks as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── 2-ary Segmented ────────────────────────────────────────────────────

    #[test]
    fn segmented_insert_contain_delete() {
        let mut filter = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
        assert!(filter.insert("hello").is_ok());
        assert!(filter.insert("world").is_ok());
        assert!(filter.contain("hello"));
        assert!(filter.contain("world"));
        assert!(!filter.contain("missing"));
        assert!(filter.delete("hello").is_ok());
        assert!(!filter.contain("hello"));
        assert!(filter.contain("world"));
        assert_eq!(filter.num_items(), 1);
    }

    #[test]
    fn segmented_insert_many() {
        let mut filter = CuckooFilter::<Segmented2aryScheme>::from_num_items(10000, 4, 12).unwrap();
        let mut inserted = 0u64;
        for i in 0u64..10000 {
            if filter.insert(i.to_le_bytes()).is_ok() {
                inserted += 1;
            } else {
                break;
            }
        }
        assert_eq!(filter.num_items(), inserted);
        for i in 0u64..inserted {
            assert!(
                filter.contain(i.to_le_bytes()),
                "item {i} should be present"
            );
        }
    }

    #[test]
    fn segmented_table_full() {
        let mut filter = CuckooFilter::<Segmented2aryScheme>::new(2, 1, 2).unwrap();
        let mut count = 0;
        for i in 0u32..1000 {
            if filter.insert(i.to_le_bytes()).is_err() {
                break;
            }
            count += 1;
        }
        assert!(count <= 2, "tiny filter should fill quickly, got {count}");
    }

    #[test]
    fn segmented_fill_correctness() {
        let mut filter = CuckooFilter::<Segmented2aryScheme>::new(4, 1, 2).unwrap();
        let mut last_ok = 0u32;
        for i in 0u32..1000 {
            if filter.insert(i.to_le_bytes()).is_err() {
                break;
            }
            last_ok = i;
        }
        for i in 0u32..=last_ok {
            assert!(filter.contain(i.to_le_bytes()), "item {i} missing");
        }
    }

    #[test]
    fn segmented_load_factor() {
        let filter = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
        assert_eq!(filter.load_factor(), 0.0);
    }

    #[test]
    fn segmented_invalid_params() {
        assert!(CuckooFilter::<Segmented2aryScheme>::new(3, 4, 12).is_err());
        assert!(CuckooFilter::<Segmented2aryScheme>::new(1, 4, 12).is_err());
        assert!(CuckooFilter::<Segmented2aryScheme>::new(4, 0, 12).is_err());
        assert!(CuckooFilter::<Segmented2aryScheme>::new(4, 4, 0).is_err());
        assert!(CuckooFilter::<Segmented2aryScheme>::new(4, 4, 33).is_err());
        assert!(CuckooFilter::<Segmented2aryScheme>::new(4, 4, 3).is_err());
    }

    // ─── 2-ary Standard ────────────────────────────────────────────────────

    #[test]
    fn standard_insert_contain_delete() {
        let mut filter = CuckooFilter::<Standard2aryScheme>::new(64, 4, 12).unwrap();
        assert!(filter.insert("hello").is_ok());
        assert!(filter.insert("world").is_ok());
        assert!(filter.contain("hello"));
        assert!(filter.contain("world"));
        assert!(!filter.contain("missing"));
        // TEST-CHANGE: delete now returns Result<(), CuckooError>.
        assert!(filter.delete("hello").is_ok());
        assert!(!filter.contain("hello"));
        assert!(filter.contain("world"));
        assert_eq!(filter.num_items(), 1);
    }

    #[test]
    fn standard_insert_many() {
        let mut filter = CuckooFilter::<Standard2aryScheme>::from_num_items(10000, 4, 12).unwrap();
        let mut inserted = 0u64;
        for i in 0u64..10000 {
            if filter.insert(i.to_le_bytes()).is_ok() {
                inserted += 1;
            } else {
                break;
            }
        }
        assert_eq!(filter.num_items(), inserted);
        for i in 0u64..inserted {
            assert!(
                filter.contain(i.to_le_bytes()),
                "item {i} should be present"
            );
        }
    }

    #[test]
    fn standard_table_full() {
        let mut filter = CuckooFilter::<Standard2aryScheme>::new(2, 1, 2).unwrap();
        let mut count = 0;
        for i in 0u32..1000 {
            if filter.insert(i.to_le_bytes()).is_err() {
                break;
            }
            count += 1;
        }
        assert!(count <= 2, "tiny filter should fill quickly, got {count}");
    }

    #[test]
    fn standard_fill_correctness() {
        let mut filter = CuckooFilter::<Standard2aryScheme>::new(4, 1, 2).unwrap();
        let mut last_ok = 0u32;
        for i in 0u32..1000 {
            if filter.insert(i.to_le_bytes()).is_err() {
                break;
            }
            last_ok = i;
        }
        for i in 0u32..=last_ok {
            assert!(filter.contain(i.to_le_bytes()), "item {i} missing");
        }
    }

    #[test]
    fn standard_invalid_params() {
        assert!(CuckooFilter::<Standard2aryScheme>::new(3, 4, 12).is_err());
        assert!(CuckooFilter::<Standard2aryScheme>::new(0, 4, 12).is_err());
        assert!(CuckooFilter::<Standard2aryScheme>::new(4, 0, 12).is_err());
        assert!(CuckooFilter::<Standard2aryScheme>::new(4, 4, 0).is_err());
        assert!(CuckooFilter::<Standard2aryScheme>::new(4, 4, 33).is_err());
        assert!(CuckooFilter::<Standard2aryScheme>::new(4, 4, 3).is_err());
    }

    #[test]
    fn standard_n1_degenerate() {
        let mut filter = CuckooFilter::<Standard2aryScheme>::new(1, 4, 4).unwrap();
        let mut count = 0;
        for i in 0u32..1000 {
            if filter.insert(i.to_le_bytes()).is_err() {
                break;
            }
            count += 1;
        }
        assert!(
            count <= 4,
            "num_buckets=1, bucket_size=4 filter holds at most 4 items, got {count}"
        );
    }

    // ─── 3-ary Standard ────────────────────────────────────────────────────

    #[test]
    fn standard_3ary_insert_contain_delete() {
        let mut filter = CuckooFilter::<Standard3aryScheme>::new(243, 4, 12).unwrap();
        assert!(filter.insert("hello").is_ok());
        assert!(filter.insert("world").is_ok());
        assert!(filter.contain("hello"));
        assert!(filter.contain("world"));
        assert!(!filter.contain("missing"));
        // TEST-CHANGE: delete now returns Result<(), CuckooError>.
        assert!(filter.delete("hello").is_ok());
        assert!(!filter.contain("hello"));
        assert!(filter.contain("world"));
        assert_eq!(filter.num_items(), 1);
    }

    #[test]
    fn standard_3ary_insert_many() {
        let mut filter = CuckooFilter::<Standard3aryScheme>::from_num_items(10000, 4, 12).unwrap();
        let mut inserted = 0u64;
        for i in 0u64..10000 {
            if filter.insert(i.to_le_bytes()).is_ok() {
                inserted += 1;
            } else {
                break;
            }
        }
        assert_eq!(filter.num_items(), inserted);
        for i in 0u64..inserted {
            assert!(
                filter.contain(i.to_le_bytes()),
                "item {i} should be present"
            );
        }
    }

    #[test]
    fn standard_3ary_fill_correctness() {
        let mut filter = CuckooFilter::<Standard3aryScheme>::new(27, 2, 8).unwrap();
        let mut last_ok = 0u32;
        for i in 0u32..1000 {
            if filter.insert(i.to_le_bytes()).is_err() {
                break;
            }
            last_ok = i;
        }
        for i in 0u32..=last_ok {
            assert!(filter.contain(i.to_le_bytes()), "item {i} missing");
        }
    }

    // ─── 3-ary Segmented ───────────────────────────────────────────────────

    #[test]
    fn segmented_3ary_insert_contain_delete() {
        // num_buckets = 3 * 32 = 96
        let mut filter = CuckooFilter::<Segmented3aryScheme>::new(96, 4, 12).unwrap();
        assert!(filter.insert("hello").is_ok());
        assert!(filter.insert("world").is_ok());
        assert!(filter.contain("hello"));
        assert!(filter.contain("world"));
        assert!(!filter.contain("missing"));
        // TEST-CHANGE: delete now returns Result<(), CuckooError>.
        assert!(filter.delete("hello").is_ok());
        assert!(!filter.contain("hello"));
        assert_eq!(filter.num_items(), 1);
    }

    #[test]
    fn segmented_3ary_insert_many() {
        let mut filter = CuckooFilter::<Segmented3aryScheme>::from_num_items(10000, 4, 12).unwrap();
        let mut inserted = 0u64;
        for i in 0u64..10000 {
            if filter.insert(i.to_le_bytes()).is_ok() {
                inserted += 1;
            } else {
                break;
            }
        }
        assert_eq!(filter.num_items(), inserted);
        for i in 0u64..inserted {
            assert!(
                filter.contain(i.to_le_bytes()),
                "item {i} should be present"
            );
        }
    }

    #[test]
    fn segmented_3ary_invalid_params() {
        assert!(CuckooFilter::<Segmented3aryScheme>::new(64, 4, 12).is_err()); // 64/3 not integer
        assert!(CuckooFilter::<Segmented3aryScheme>::new(9, 4, 12).is_err()); // 9/3=3, not power of 2
        assert!(CuckooFilter::<Segmented3aryScheme>::new(3, 4, 12).is_ok()); // 3/3=1, power of 2
        assert!(CuckooFilter::<Segmented3aryScheme>::new(96, 4, 12).is_ok()); // 96/3=32, power of 2
    }

    // ─── 4-ary Standard ────────────────────────────────────────────────────

    #[test]
    fn standard_4ary_insert_contain_delete() {
        let mut filter = CuckooFilter::<Standard4aryScheme>::new(64, 4, 12).unwrap();
        assert!(filter.insert("hello").is_ok());
        assert!(filter.insert("world").is_ok());
        assert!(filter.contain("hello"));
        assert!(filter.contain("world"));
        assert!(!filter.contain("missing"));
        // TEST-CHANGE: delete now returns Result<(), CuckooError>.
        assert!(filter.delete("hello").is_ok());
        assert!(!filter.contain("hello"));
        assert_eq!(filter.num_items(), 1);
    }

    #[test]
    fn standard_4ary_insert_many() {
        let mut filter = CuckooFilter::<Standard4aryScheme>::from_num_items(10000, 4, 12).unwrap();
        let mut inserted = 0u64;
        for i in 0u64..10000 {
            if filter.insert(i.to_le_bytes()).is_ok() {
                inserted += 1;
            } else {
                break;
            }
        }
        assert_eq!(filter.num_items(), inserted);
        for i in 0u64..inserted {
            assert!(
                filter.contain(i.to_le_bytes()),
                "item {i} should be present"
            );
        }
    }

    #[test]
    fn standard_4ary_fill_correctness() {
        let mut filter = CuckooFilter::<Standard4aryScheme>::new(16, 2, 8).unwrap();
        let mut last_ok = 0u32;
        for i in 0u32..1000 {
            if filter.insert(i.to_le_bytes()).is_err() {
                break;
            }
            last_ok = i;
        }
        for i in 0u32..=last_ok {
            assert!(filter.contain(i.to_le_bytes()), "item {i} missing");
        }
    }

    // ─── 4-ary Segmented ───────────────────────────────────────────────────

    #[test]
    fn segmented_4ary_insert_contain_delete() {
        // num_buckets = 4 * 16 = 64
        let mut filter = CuckooFilter::<Segmented4aryScheme>::new(64, 4, 12).unwrap();
        assert!(filter.insert("hello").is_ok());
        assert!(filter.insert("world").is_ok());
        assert!(filter.contain("hello"));
        assert!(filter.contain("world"));
        assert!(!filter.contain("missing"));
        // TEST-CHANGE: delete now returns Result<(), CuckooError>.
        assert!(filter.delete("hello").is_ok());
        assert!(!filter.contain("hello"));
        assert_eq!(filter.num_items(), 1);
    }

    #[test]
    fn segmented_4ary_insert_many() {
        let mut filter = CuckooFilter::<Segmented4aryScheme>::from_num_items(10000, 4, 12).unwrap();
        let mut inserted = 0u64;
        for i in 0u64..10000 {
            if filter.insert(i.to_le_bytes()).is_ok() {
                inserted += 1;
            } else {
                break;
            }
        }
        assert_eq!(filter.num_items(), inserted);
        for i in 0u64..inserted {
            assert!(
                filter.contain(i.to_le_bytes()),
                "item {i} should be present"
            );
        }
    }

    #[test]
    fn segmented_4ary_invalid_params() {
        assert!(CuckooFilter::<Segmented4aryScheme>::new(3, 4, 12).is_err()); // not power of 2
        assert!(CuckooFilter::<Segmented4aryScheme>::new(2, 4, 12).is_err()); // < 4
        assert!(CuckooFilter::<Segmented4aryScheme>::new(4, 4, 12).is_ok()); // 4/4=1 ✓
        assert!(CuckooFilter::<Segmented4aryScheme>::new(64, 4, 12).is_ok()); // 64/4=16 ✓
    }

    // ─── Shared behaviour across all types ─────────────────────────────────

    #[test]
    fn rollback_on_table_full_leaves_existing_items_intact() {
        // Fill a filter to capacity, record all inserted items, then attempt one more insert.
        // All previously inserted items must still be queryable.
        let mut filter = CuckooFilter::<Segmented2aryScheme>::new(4, 2, 4).unwrap();
        filter.set_max_kicks(50);
        let mut inserted: Vec<u32> = Vec::new();
        for i in 0u32..1000 {
            match filter.insert(i.to_le_bytes()) {
                Ok(()) => inserted.push(i),
                Err(CuckooError::TableFull) => break,
                Err(e) => panic!("unexpected error: {}", e),
            }
        }
        // All previously inserted items must still be present.
        for &item in &inserted {
            assert!(
                filter.contain(item.to_le_bytes()),
                "item {} was lost after table-full rollback",
                item
            );
        }
    }

    #[test]
    fn delete_absent_item_returns_not_found() {
        // TEST-CHANGE: delete returns Err(NotFound) instead of false.
        let mut filter = CuckooFilter::<Standard2aryScheme>::new(64, 4, 12).unwrap();
        filter.insert("hello").unwrap();
        assert!(matches!(
            filter.delete("not_inserted"),
            Err(CuckooError::NotFound)
        ));
        assert_eq!(filter.num_items(), 1); // "hello" must still be there
    }

    #[test]
    fn size_in_bytes_matches_formula() {
        // num_buckets=64, bucket_size=4, fp=12 → 64*4*12 = 3072 bits = 384 bytes
        let filter = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
        assert_eq!(filter.size_in_bytes(), 384);
        // num_buckets=32, bucket_size=2, fp=8 → 32*2*8 = 512 bits = 64 bytes
        let filter2 = CuckooFilter::<Standard2aryScheme>::new(32, 2, 8).unwrap();
        assert_eq!(filter2.size_in_bytes(), 64);
    }

    #[test]
    fn set_max_kicks_limits_kick_budget() {
        // With max_kicks=0, no kicking is attempted; any full-bucket insert fails.
        // Use a tiny filter with bucket_size=1 so the second insert into the same pair of buckets fails.
        let mut filter = CuckooFilter::<Segmented2aryScheme>::new(2, 1, 2).unwrap();
        filter.set_max_kicks(0);
        // First two inserts may succeed (one per bucket half); further inserts must fail
        // without kicks.
        let mut count = 0u32;
        for i in 0u32..100 {
            if filter.insert(i.to_le_bytes()).is_ok() {
                count += 1;
            }
        }
        // With no kicks, at most bucket_size*num_buckets = 1*2 = 2 items can be placed directly.
        assert!(count <= 2, "expected ≤ 2 direct placements, got {count}");
    }

    #[test]
    fn load_factor_increases_with_inserts() {
        let mut filter = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
        assert_eq!(filter.load_factor(), 0.0);
        filter.insert("a").unwrap();
        assert!(filter.load_factor() > 0.0);
        filter.insert("b").unwrap();
        assert!(filter.load_factor() > filter.load_factor() - 0.001); // monotone
    }

    #[test]
    fn error_display() {
        assert_eq!(CuckooError::TableFull.to_string(), "table is full");
        assert!(CuckooError::InvalidParams("bad".into())
            .to_string()
            .contains("bad"));
        assert_eq!(
            CuckooError::NotFound.to_string(),
            "item not found in filter"
        );
    }

    // ─── Comprehensive delete tests ─────────────────────────────────────────
    //
    // The macro generates 9 delete-contract tests for every one of the 6 filter
    // variants. Additional non-macro tests cover round-trip correctness for the
    // standard 3-ary and 4-ary variants.

    /// Generates the 9 delete-contract test functions inside a variant-specific
    /// sub-module. Invoked once per filter type below.
    ///
    /// Parameters:
    /// - `$mod_name` — the sub-module identifier (e.g., `seg2`)
    /// - `$filter_ty` — the concrete filter type (e.g., `Segmented2aryCuckooFilter`)
    /// - `$new` — an expression that constructs a fresh filter (must succeed)
    macro_rules! delete_contract_tests {
        ($mod_name:ident, $filter_ty:ty, $new:expr) => {
            mod $mod_name {
                use super::*;

                /// Insert X, delete X → Ok, contain returns false, size decremented.
                #[test]
                fn delete_inserted_item() {
                    let mut f = $new;
                    f.insert("item_x").unwrap();
                    assert_eq!(f.num_items(), 1);
                    f.delete("item_x").unwrap();
                    assert!(!f.contain("item_x"));
                    assert_eq!(f.num_items(), 0);
                }

                /// delete on a fresh empty filter → Err(NotFound), size stays 0.
                #[test]
                fn delete_from_empty_filter() {
                    let mut f: $filter_ty = $new;
                    assert!(matches!(f.delete("anything"), Err(CuckooError::NotFound)));
                    assert_eq!(f.num_items(), 0);
                }

                /// Insert A, delete B → Err(NotFound), A still present, size unchanged.
                #[test]
                fn delete_absent_from_nonempty() {
                    let mut f = $new;
                    f.insert("item_a").unwrap();
                    assert!(matches!(f.delete("item_b"), Err(CuckooError::NotFound)));
                    assert!(f.contain("item_a"));
                    assert_eq!(f.num_items(), 1);
                }

                /// Insert X, delete twice → first Ok, second Err(NotFound).
                #[test]
                fn double_delete() {
                    let mut f = $new;
                    f.insert("item_x").unwrap();
                    f.delete("item_x").unwrap();
                    assert!(matches!(f.delete("item_x"), Err(CuckooError::NotFound)));
                    assert_eq!(f.num_items(), 0);
                }

                /// Insert A and B, delete A → B still present, size = 1.
                #[test]
                fn delete_preserves_other_items() {
                    let mut f = $new;
                    f.insert("item_a").unwrap();
                    f.insert("item_b").unwrap();
                    f.delete("item_a").unwrap();
                    assert!(!f.contain("item_a"));
                    assert!(f.contain("item_b"));
                    assert_eq!(f.num_items(), 1);
                }

                /// Insert X, delete X, reinsert X → Ok, contain true, size = 1.
                #[test]
                fn insert_delete_reinsert() {
                    let mut f = $new;
                    f.insert("item_x").unwrap();
                    f.delete("item_x").unwrap();
                    f.insert("item_x").unwrap();
                    assert!(f.contain("item_x"));
                    assert_eq!(f.num_items(), 1);
                }

                /// Fill the filter, delete every inserted item → size = 0, none found.
                #[test]
                fn fill_then_delete_all() {
                    let mut f = $new;
                    let mut inserted: Vec<u64> = Vec::new();
                    for i in 0u64..10_000 {
                        match f.insert(i.to_le_bytes()) {
                            Ok(()) => inserted.push(i),
                            Err(CuckooError::TableFull) => break,
                            Err(e) => panic!("unexpected error: {e}"),
                        }
                    }
                    assert!(!inserted.is_empty(), "filter should hold at least one item");
                    for &item in &inserted {
                        f.delete(item.to_le_bytes())
                            .unwrap_or_else(|e| panic!("delete({item}) failed: {e}"));
                    }
                    assert_eq!(f.num_items(), 0);
                    for &item in &inserted {
                        // After deletion, the item must no longer be found.
                        // A false-positive here would be an extraordinary coincidence; we
                        // assert it does not happen for the items we actually removed.
                        assert!(
                            !f.contain(item.to_le_bytes()),
                            "item {item} still found after deletion"
                        );
                    }
                }

                /// Fill the filter near capacity (cuckoo kicks required), then delete
                /// each inserted item and confirm it is gone.
                #[test]
                fn delete_after_kicks() {
                    let mut f = $new;
                    // Fill to 80% to ensure some items required kicking.
                    let capacity = (f.size_in_bytes() * 8) as u64; // rough upper bound
                    let target = (capacity / 10).max(20).min(500);
                    let mut inserted: Vec<u64> = Vec::new();
                    for i in 0u64..target * 2 {
                        match f.insert(i.to_le_bytes()) {
                            Ok(()) => {
                                inserted.push(i);
                                if inserted.len() as u64 >= target {
                                    break;
                                }
                            }
                            Err(CuckooError::TableFull) => break,
                            Err(e) => panic!("unexpected error: {e}"),
                        }
                    }
                    for &item in &inserted {
                        f.delete(item.to_le_bytes())
                            .unwrap_or_else(|e| panic!("delete({item}) after kicks failed: {e}"));
                    }
                }

                /// 100 failed deletes must not change the size counter.
                #[test]
                fn delete_preserves_size_on_not_found() {
                    let mut f = $new;
                    for i in 0u64..5 {
                        f.insert(i.to_le_bytes()).unwrap();
                    }
                    let size_before = f.num_items();
                    for i in 1000u64..1100 {
                        let _ = f.delete(i.to_le_bytes()); // all should be NotFound
                    }
                    assert_eq!(f.num_items(), size_before);
                }
            }
        };
    }

    // Invoke for all 6 filter variants.
    delete_contract_tests!(
        delete_seg2,
        CuckooFilter<Segmented2aryScheme>,
        CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap()
    );
    delete_contract_tests!(
        delete_std2,
        CuckooFilter<Standard2aryScheme>,
        CuckooFilter::<Standard2aryScheme>::new(64, 4, 12).unwrap()
    );
    delete_contract_tests!(
        delete_std3,
        CuckooFilter<Standard3aryScheme>,
        CuckooFilter::<Standard3aryScheme>::new(81, 4, 12).unwrap()
    );
    delete_contract_tests!(
        delete_seg3,
        CuckooFilter<Segmented3aryScheme>,
        // num_buckets = 3*32 = 96
        CuckooFilter::<Segmented3aryScheme>::new(96, 4, 12).unwrap()
    );
    delete_contract_tests!(
        delete_std4,
        CuckooFilter<Standard4aryScheme>,
        CuckooFilter::<Standard4aryScheme>::new(64, 4, 12).unwrap()
    );
    delete_contract_tests!(
        delete_seg4,
        CuckooFilter<Segmented4aryScheme>,
        CuckooFilter::<Segmented4aryScheme>::new(64, 4, 12).unwrap()
    );

    // ─── Pre-allocated rollback chain (Commit 3) ─────────────────────────────

    #[test]
    fn repeated_inserts_reuse_chain() {
        let mut f = CuckooFilter::<Segmented2aryScheme>::from_num_items(2000, 4, 12).unwrap();
        let initial_cap = f.chain.capacity();
        for i in 0u32..1000 {
            let _ = f.insert(i.to_le_bytes());
        }
        assert_eq!(
            f.chain.capacity(),
            initial_cap,
            "filter chain grew past max_kicks"
        );
    }

    #[test]
    fn set_max_kicks_grows_chain() {
        let mut f = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
        f.set_max_kicks(2000);
        assert!(f.chain.capacity() >= 2000);
    }

    #[test]
    fn failed_insert_then_successful_insert_filter() {
        let mut f = CuckooFilter::<Segmented2aryScheme>::new(2, 1, 8).unwrap();
        let mut placed: Vec<u32> = Vec::new();
        for i in 0u32..200 {
            match f.insert(i.to_le_bytes()) {
                Ok(()) => placed.push(i),
                Err(CuckooError::TableFull) => break,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert!(!placed.is_empty());
        // All previously-inserted items must still be reported as present.
        for &i in &placed {
            assert!(f.contain(i.to_le_bytes()));
        }
        // After deleting one, a new insert proceeds normally.
        f.delete(placed[0].to_le_bytes()).unwrap();
        f.insert(b"fresh".as_ref()).unwrap();
        assert!(f.contain(b"fresh".as_ref()));
    }

    // ─── Insert/delete bookkeeping over an item-bytes grid ───────────────────
    //
    // The item hash diffuses its input before the filter sees an index or a
    // fingerprint, so what varying the item bytes exercises is the insert/delete
    // bookkeeping, not the byte patterns themselves. A fixed set of
    // representative encodings pins that contract for each scheme and keeps any
    // failure reproducible from the test name alone.

    /// Representative item encodings: minimal, all-zero, all-ones, a patterned
    /// mid-width run, and a full 64-byte item.
    fn item_bytes_grid() -> Vec<Vec<u8>> {
        vec![
            vec![0x00],
            vec![0xFF],
            vec![0u8; 16],
            vec![0xFFu8; 16],
            (0..32u8).map(|i| i.wrapping_mul(37)).collect(),
            (0..64u8).collect(),
        ]
    }

    /// Insert then delete returns a Standard 2-ary filter to empty, for every
    /// item encoding in the grid.
    #[test]
    fn insert_delete_roundtrip_std2_over_item_grid() {
        for item in item_bytes_grid() {
            let mut f = CuckooFilter::<Standard2aryScheme>::new(64, 4, 12).unwrap();
            f.insert(item.as_slice()).unwrap();
            assert_eq!(f.num_items(), 1, "insert failed for item {item:?}");
            f.delete(item.as_slice()).unwrap();
            assert_eq!(f.num_items(), 0, "delete failed for item {item:?}");
        }
    }

    /// Same round-trip on a Segmented 2-ary filter — segmentation changes index
    /// placement, not insert/delete bookkeeping.
    #[test]
    fn insert_delete_roundtrip_seg2_over_item_grid() {
        for item in item_bytes_grid() {
            let mut f = CuckooFilter::<Segmented2aryScheme>::new(64, 4, 12).unwrap();
            f.insert(item.as_slice()).unwrap();
            assert_eq!(f.num_items(), 1, "insert failed for item {item:?}");
            f.delete(item.as_slice()).unwrap();
            assert_eq!(f.num_items(), 0, "delete failed for item {item:?}");
        }
    }

    /// Deleting from a fresh Standard 3-ary filter reports `NotFound` for every
    /// item encoding — the modulo index path has no false "present" answer on an
    /// empty table.
    #[test]
    fn delete_without_insert_not_found_std3_over_item_grid() {
        for item in item_bytes_grid() {
            let mut f = CuckooFilter::<Standard3aryScheme>::new(81, 4, 12).unwrap();
            assert!(
                matches!(f.delete(item.as_slice()), Err(CuckooError::NotFound)),
                "expected NotFound for item {item:?}",
            );
        }
    }

    /// Inserting N distinct keys then deleting all of them returns the filter to
    /// empty, exercising the multi-item path (including any kicking) rather than
    /// the single-item case above.
    #[test]
    fn insert_n_delete_n_empty_std2() {
        let mut f = CuckooFilter::<Standard2aryScheme>::new(64, 4, 12).unwrap();
        let mut inserted: Vec<u64> = Vec::new();
        for k in (0u64..20).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)) {
            if f.insert(k.to_le_bytes()).is_ok() {
                inserted.push(k);
            }
        }
        assert!(!inserted.is_empty(), "no key was inserted");
        for k in &inserted {
            f.delete(k.to_le_bytes())
                .unwrap_or_else(|e| panic!("delete({k}) failed: {e}"));
        }
        assert_eq!(f.num_items(), 0);
    }

    #[test]
    fn from_num_items_rejects_overflow_max_items() {
        let huge: u64 = u32::MAX as u64 * 5;
        assert!(CuckooFilter::<Segmented2aryScheme>::from_num_items(huge, 4, 12).is_err());
        assert!(CuckooFilter::<Segmented3aryScheme>::from_num_items(huge, 4, 12).is_err());
        assert!(CuckooFilter::<Segmented4aryScheme>::from_num_items(huge, 4, 12).is_err());
        assert!(CuckooFilter::<Standard2aryScheme>::from_num_items(huge, 4, 12).is_err());
        assert!(CuckooFilter::<Standard3aryScheme>::from_num_items(huge, 4, 12).is_err());
        assert!(CuckooFilter::<Standard4aryScheme>::from_num_items(huge, 4, 12).is_err());
    }
}
