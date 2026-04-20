//! Generic cuckoo filter implementation.
//!
//! [`CuckooFilter<S>`] is parameterised over an [`IndexScheme`], which computes candidate
//! bucket indices. All six concrete filter types share the same insert/lookup/delete logic;
//! only index computation varies. See [`crate::scheme`] for the six concrete schemes.
//!
//! ## Insert semantics
//!
//! - **Direct insert**: if any of the k candidate buckets has a free slot, the tag is placed
//!   immediately.
//! - **Cuckoo kicking**: if all candidate buckets are full, a random slot is evicted and its
//!   tag is relocated to one of its own alternates. This repeats up to `max_kicks` times.
//! - **Rollback**: if the kick budget is exhausted, all mutations are reversed and
//!   [`CuckooError::TableFull`] is returned. The table is left in its original state.
//!
//! ## False positives
//!
//! The filter is a probabilistic data structure. [`CuckooFilter::contain`] may return `true`
//! for items that were never inserted (false positive). It never returns `false` for items that
//! are currently inserted (no false negatives). The theoretical false-positive rate is
//! `2b / 2^fp_bits`.
//!
//! ## Deletion
//!
//! Deletion is supported but carries the usual caveat: deleting an item that was never inserted
//! may silently remove a fingerprint that belongs to a different item with the same tag and
//! candidate indices. Only delete items you have previously inserted.
//!
//! ## Architecture
//!
//! This module depends on [`crate::bucket::TagTable`] for bit-packed storage and on
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
//!   inserted item if they share the same tag and candidate indices. Callers are responsible
//!   for deleting only items they have inserted.

// DECISION: The kicking loop uses indexed `for p in 0..k` rather than an iterator because
// `p` serves as both the array subscript into `indices`/`all` and as the chain-position
// value that must be written to the position store. An enumerate() form would require a
// separate cast and is less readable in this context.
#![allow(clippy::needless_range_loop)]
// DECISION: explicit_counter_loop is suppressed for the same reason — `alt_count` is a
// compact inline counter used to build a small stack-allocated array of alternates.
#![allow(clippy::explicit_counter_loop)]

use rand::Rng;
use std::fmt;

use crate::bucket::TagTable;
use crate::scheme::{
    IndexScheme, Segmented3aryScheme, Segmented4aryScheme, SegmentedScheme, Standard3aryScheme,
    Standard4aryScheme, StandardScheme,
};
use crate::util::{next_power_of_3, next_power_of_4, upper_power_of_2};

const MAX_KICKS_DEFAULT: u32 = 500;

/// Conservative target load factors for `from_num_items` sizing, indexed as
/// `[arity - 2][b - 1]` (arity ∈ {2,3,4}, b ∈ {1,2,3,4}).
///
/// Each value is 0.95 × the mean empirical max load factor across all tested n values
/// (n = 2^14 … 2^20, 20 trials per config, MAX_KICKS = 500). Using 95% of the mean
/// rather than the raw minimum gives a threshold that is reliably achievable for any
/// valid n while still sizing the filter as tightly as possible. The worst-case n
/// (largest) is already captured in the mean, so these values are conservative for
/// smaller tables too.
///
/// Experimental mean load factors (source: `benches/load_factor.rs`):
/// | arity | b=1  | b=2  | b=3  | b=4  |
/// |-------|------|------|------|------|
/// | 2     | 0.50 | 0.87 | 0.94 | 0.96 |
/// | 3     | 0.90 | 0.98 | 0.99 | 0.99 |
/// | 4     | 0.96 | 0.99 | 1.00 | 1.00 |
const MAX_LOAD_FACTOR: [[f64; 4]; 3] = [
    // arity=2:   b=1   b=2   b=3   b=4
    [0.48, 0.83, 0.89, 0.91],
    // arity=3:   b=1   b=2   b=3   b=4
    [0.85, 0.93, 0.94, 0.94],
    // arity=4:   b=1   b=2   b=3   b=4
    [0.91, 0.94, 0.95, 0.95],
];

/// Return the conservative target load factor for a given arity and b.
///
/// Used by all `from_num_items` constructors to determine when to double `n`.
#[inline]
fn target_load_factor(arity: usize, b: usize) -> f64 {
    MAX_LOAD_FACTOR[arity - 2][b.min(4) - 1]
}

/// Errors returned by filter operations.
///
/// Implements [`std::error::Error`] so it can be used with the `?` operator in calling code.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::{CuckooError, SegmentedCuckooFilter};
///
/// match SegmentedCuckooFilter::new(3, 4, 12) {
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
            CuckooError::TableFull => write!(f, "table is full"),
            CuckooError::InvalidParams(msg) => write!(f, "invalid parameters: {msg}"),
            CuckooError::NotFound => write!(f, "item not found in filter"),
        }
    }
}

impl std::error::Error for CuckooError {}

/// Statistics returned by [`CuckooFilter::add_with_stats`].
///
/// Provides insight into cuckoo kicking behaviour for a single insert operation. Useful when
/// analysing eviction chain length distributions across different filter configurations.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::SegmentedCuckooFilter;
///
/// let mut f = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
/// let stats = f.add_with_stats("hello").unwrap();
/// // A fresh filter always places the first item directly.
/// assert_eq!(stats.kicks, 0);
/// ```
pub struct InsertStats {
    /// Number of cuckoo kicks performed (0 means the item was placed directly without eviction).
    pub kicks: u32,
}

/// Generic cuckoo filter parameterised over an index scheme.
///
/// All insert/lookup/delete mechanics are implemented here. The scheme `S` contributes only
/// index computation (via [`IndexScheme::hash_item`] and [`IndexScheme::all_indices`]).
///
/// Prefer the six type aliases in `crate` (`SegmentedCuckooFilter`, `StandardCuckooFilter`,
/// etc.) over constructing this type directly unless you are implementing a custom scheme.
///
/// # Type parameter
///
/// - `S` — an [`IndexScheme`] implementation that determines arity and index layout.
///
/// # Security
///
/// See the [module-level security note](self#security) for hash-flooding and deletion caveats.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::SegmentedCuckooFilter;
///
/// let mut f = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
/// f.add("hello").unwrap();
/// assert!(f.contain("hello"));
/// assert!(!f.contain("world"));
/// f.delete("hello");
/// assert!(!f.contain("hello"));
/// ```
pub struct CuckooFilter<S: IndexScheme> {
    table: TagTable,
    scheme: S,
    num_items: u64,
    max_kicks: u32,
}

/// Validate parameters common to all filter constructors.
///
/// `fingerprint_bits` must satisfy `2^f > d*b` (guarantees FPR < 1).
fn validate_common_params(arity: u32, b: u32, fingerprint_bits: u32) -> Result<(), CuckooError> {
    if b < 1 {
        return Err(CuckooError::InvalidParams("b must be >= 1".into()));
    }
    let min_fp_bits = (arity * b).ilog2() + 1;
    if fingerprint_bits < min_fp_bits || fingerprint_bits > 32 {
        return Err(CuckooError::InvalidParams(format!(
            "fingerprint_bits must be >= {min_fp_bits} for arity={arity}, b={b} (ensures FPR < 1)"
        )));
    }
    Ok(())
}

// ─── Constructors ───────────────────────────────────────────────────────────

impl CuckooFilter<SegmentedScheme> {
    /// Create a segmented 2-ary filter with explicit dimensions.
    ///
    /// # Arguments
    ///
    /// - `n` — total number of buckets; must be a power of 2 and ≥ 2.
    /// - `b` — tags (slots) per bucket; must be ≥ 1.
    /// - `fingerprint_bits` — fingerprint bit width; must be in `[⌊log2(2b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter with capacity `n * b` tags and an initial load factor of 0.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::SegmentedCuckooFilter;
    ///
    /// let mut f = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
    /// f.add("hello").unwrap();
    /// assert!(f.contain("hello"));
    /// assert!(SegmentedCuckooFilter::new(3, 4, 12).is_err()); // n=3 not power of 2
    /// assert!(SegmentedCuckooFilter::new(1, 4, 12).is_err()); // n=1 < 2
    /// ```
    pub fn new(n: u32, b: u32, fingerprint_bits: u32) -> Result<Self, CuckooError> {
        if !n.is_power_of_two() || n < 2 {
            return Err(CuckooError::InvalidParams(
                "n must be a power of 2 and >= 2".into(),
            ));
        }
        validate_common_params(2, b, fingerprint_bits)?;
        let table = TagTable::new(n, b, fingerprint_bits);
        Ok(CuckooFilter {
            table,
            scheme: SegmentedScheme { half: n / 2 },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
        })
    }

    /// Create a segmented 2-ary filter sized to hold at least `max_items`.
    ///
    /// Rounds `n` up to the next valid value (power of 2, ≥ 2). If the projected load factor
    /// exceeds the empirical target for this (arity=2, b) configuration, `n` is doubled
    /// until the projection is within bounds.
    ///
    /// # Arguments
    ///
    /// - `max_items` — minimum number of items the filter should hold.
    /// - `b` — tags per bucket; must be ≥ 1.
    /// - `fingerprint_bits` — fingerprint bit width; must be in `[⌊log2(2b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter sized so that inserting `max_items` keeps the projected load factor below the
    /// empirical target for (arity=2, b). See [`MAX_LOAD_FACTOR`] for values.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `b` or `fingerprint_bits` are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::SegmentedCuckooFilter;
    ///
    /// let f = SegmentedCuckooFilter::from_num_items(100_000, 4, 12).unwrap();
    /// assert!(f.size_in_bytes() > 0);
    /// ```
    pub fn from_num_items(
        max_items: u64,
        b: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(2, b, fingerprint_bits)?;
        let buckets_needed = max_items.div_ceil(b as u64);
        let mut n = upper_power_of_2(buckets_needed) as u32;
        if n < 2 {
            n = 2;
        }
        let target = target_load_factor(2, b as usize);
        while max_items as f64 / (n as u64 * b as u64) as f64 > target {
            n *= 2;
        }
        Self::new(n, b, fingerprint_bits)
    }
}

impl CuckooFilter<StandardScheme> {
    /// Create a standard 2-ary filter with explicit dimensions.
    ///
    /// # Arguments
    ///
    /// - `n` — total number of buckets; must be a power of 2 and ≥ 1.
    /// - `b` — tags per bucket; must be ≥ 1.
    /// - `fingerprint_bits` — fingerprint bit width; must be in `[⌊log2(2b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter with capacity `n * b` tags and an initial load factor of 0.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::StandardCuckooFilter;
    ///
    /// let mut f = StandardCuckooFilter::new(64, 4, 12).unwrap();
    /// f.add(42u64.to_le_bytes()).unwrap();
    /// assert!(f.contain(42u64.to_le_bytes()));
    /// assert!(StandardCuckooFilter::new(3, 4, 12).is_err()); // n not power of 2
    /// ```
    pub fn new(n: u32, b: u32, fingerprint_bits: u32) -> Result<Self, CuckooError> {
        if !n.is_power_of_two() || n < 1 {
            return Err(CuckooError::InvalidParams(
                "n must be a power of 2 and >= 1".into(),
            ));
        }
        validate_common_params(2, b, fingerprint_bits)?;
        let table = TagTable::new(n, b, fingerprint_bits);
        Ok(CuckooFilter {
            table,
            scheme: StandardScheme { num_buckets: n },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
        })
    }

    /// Create a standard 2-ary filter sized to hold at least `max_items`.
    ///
    /// # Arguments
    ///
    /// - `max_items` — minimum number of items the filter should hold.
    /// - `b` — tags per bucket; must be ≥ 1.
    /// - `fingerprint_bits` — fingerprint bit width; must be in `[⌊log2(2b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter sized so that inserting `max_items` keeps the projected load factor below the
    /// empirical target for (arity=2, b). See [`MAX_LOAD_FACTOR`] for values.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `b` or `fingerprint_bits` are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::StandardCuckooFilter;
    ///
    /// let mut f = StandardCuckooFilter::from_num_items(50_000, 4, 12).unwrap();
    /// f.add(b"item".as_ref()).unwrap();
    /// assert!(f.contain(b"item".as_ref()));
    /// ```
    pub fn from_num_items(
        max_items: u64,
        b: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(2, b, fingerprint_bits)?;
        let buckets_needed = max_items.div_ceil(b as u64);
        let mut n = upper_power_of_2(buckets_needed) as u32;
        if n < 1 {
            n = 1;
        }
        let target = target_load_factor(2, b as usize);
        while max_items as f64 / (n as u64 * b as u64) as f64 > target {
            n *= 2;
        }
        Self::new(n, b, fingerprint_bits)
    }
}

impl CuckooFilter<Standard3aryScheme> {
    /// Create a standard 3-ary filter with explicit dimensions.
    ///
    /// Uses xor3 cycling so `all_indices` always reconstructs all three candidates from any
    /// starting index without per-slot position storage.
    ///
    /// # Arguments
    ///
    /// - `n` — total number of buckets; must be a power of 3 (3^k) and ≥ 1.
    /// - `b` — tags per bucket; must be ≥ 1.
    /// - `fingerprint_bits` — fingerprint bit width; must be in `[⌊log2(3b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter with capacity `n * b` tags and an initial load factor of 0.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::Standard3aryCuckooFilter;
    ///
    /// let mut f = Standard3aryCuckooFilter::new(243, 4, 12).unwrap();
    /// f.add("data").unwrap();
    /// assert!(f.contain("data"));
    /// ```
    pub fn new(n: u32, b: u32, fingerprint_bits: u32) -> Result<Self, CuckooError> {
        if !crate::util::is_power_of_3(n) || n < 1 {
            return Err(CuckooError::InvalidParams(
                "n must be a power of 3 (3^k)".into(),
            ));
        }
        validate_common_params(3, b, fingerprint_bits)?;
        let table = TagTable::new(n, b, fingerprint_bits);
        Ok(CuckooFilter {
            table,
            scheme: Standard3aryScheme { num_buckets: n },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
        })
    }

    /// Create a standard 3-ary filter sized to hold at least `max_items`.
    ///
    /// # Arguments
    ///
    /// - `max_items` — minimum number of items the filter should hold.
    /// - `b` — tags per bucket; must be ≥ 1.
    /// - `fingerprint_bits` — fingerprint bit width; must be in `[⌊log2(2b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter sized so that inserting `max_items` keeps the projected load factor below the
    /// empirical target for (arity=3, b). See [`MAX_LOAD_FACTOR`] for values.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `b` or `fingerprint_bits` are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::Standard3aryCuckooFilter;
    ///
    /// let mut f = Standard3aryCuckooFilter::from_num_items(10_000, 4, 12).unwrap();
    /// f.add("hello").unwrap();
    /// assert!(f.contain("hello"));
    /// ```
    pub fn from_num_items(
        max_items: u64,
        b: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(3, b, fingerprint_bits)?;
        let buckets_needed = max_items.div_ceil(b as u64);
        let mut n = next_power_of_3(buckets_needed) as u32;
        if n < 1 {
            n = 1;
        }
        let target = target_load_factor(3, b as usize);
        while max_items as f64 / (n as u64 * b as u64) as f64 > target {
            n *= 3;
        }
        Self::new(n, b, fingerprint_bits)
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
    /// - `n` — total buckets; must equal `3 · 2^m` for some m ≥ 0 (i.e., `n/3` is a power
    ///   of 2).
    /// - `b` — tags per bucket; must be ≥ 1.
    /// - `fingerprint_bits` — fingerprint bit width; must be in `[⌊log2(2b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter with capacity `n * b` tags and an initial load factor of 0.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::Segmented3aryCuckooFilter;
    ///
    /// // n = 3 * 32 = 96 — each of 3 segments is 32 buckets
    /// let mut f = Segmented3aryCuckooFilter::new(96, 4, 12).unwrap();
    /// f.add("data").unwrap();
    /// assert!(f.contain("data"));
    /// assert!(Segmented3aryCuckooFilter::new(64, 4, 12).is_err()); // 64/3 not integer
    /// ```
    pub fn new(n: u32, b: u32, fingerprint_bits: u32) -> Result<Self, CuckooError> {
        if n < 3 || n % 3 != 0 || !(n / 3).is_power_of_two() {
            return Err(CuckooError::InvalidParams(
                "n must be 3 * 2^m (n divisible by 3, n/3 a power of 2)".into(),
            ));
        }
        validate_common_params(3, b, fingerprint_bits)?;
        let seg = n / 3;
        let table = TagTable::new(n, b, fingerprint_bits);
        Ok(CuckooFilter {
            table,
            scheme: Segmented3aryScheme { segment_size: seg },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
        })
    }

    /// Create a segmented 3-ary filter sized to hold at least `max_items`.
    ///
    /// Each segment must be a power of 2; this method computes the smallest valid `n = 3 * seg`
    /// where `seg = 2^m ≥ ceil(max_items / (3 * b))`, then doubles `seg` until the projected
    /// load is within the empirical target for (arity=3, b).
    ///
    /// # Arguments
    ///
    /// - `max_items` — minimum number of items the filter should hold.
    /// - `b` — tags per bucket; must be ≥ 1.
    /// - `fingerprint_bits` — fingerprint bit width; must be in `[⌊log2(2b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter sized so that inserting `max_items` keeps the projected load factor below the
    /// empirical target for (arity=3, b). See [`MAX_LOAD_FACTOR`] for values.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `b` or `fingerprint_bits` are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::Segmented3aryCuckooFilter;
    ///
    /// let mut f = Segmented3aryCuckooFilter::from_num_items(10_000, 4, 12).unwrap();
    /// f.add("hello").unwrap();
    /// assert!(f.contain("hello"));
    /// ```
    pub fn from_num_items(
        max_items: u64,
        b: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(3, b, fingerprint_bits)?;
        // Each segment must be power of 2; n = 3 * segment_size
        let slots_per_segment = max_items.div_ceil(3 * b as u64);
        let mut seg = upper_power_of_2(slots_per_segment) as u32;
        if seg < 1 {
            seg = 1;
        }
        let mut n = 3 * seg;
        let target = target_load_factor(3, b as usize);
        while max_items as f64 / (n as u64 * b as u64) as f64 > target {
            seg *= 2;
            n = 3 * seg;
        }
        Self::new(n, b, fingerprint_bits)
    }
}

impl CuckooFilter<Standard4aryScheme> {
    /// Create a standard 4-ary filter with explicit dimensions.
    ///
    /// Uses xor4 cycling so `all_indices` always reconstructs all four candidates from any
    /// starting index without per-slot position storage.
    ///
    /// # Arguments
    ///
    /// - `n` — total number of buckets; must be a power of 4 (4^k) and ≥ 1.
    /// - `b` — tags per bucket; must be ≥ 1.
    /// - `fingerprint_bits` — fingerprint bit width; must be in `[⌊log2(4b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter with capacity `n * b` tags and an initial load factor of 0.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::Standard4aryCuckooFilter;
    ///
    /// let mut f = Standard4aryCuckooFilter::new(64, 4, 12).unwrap();
    /// f.add("data").unwrap();
    /// assert!(f.contain("data"));
    /// ```
    pub fn new(n: u32, b: u32, fingerprint_bits: u32) -> Result<Self, CuckooError> {
        if !crate::util::is_power_of_4(n) || n < 1 {
            return Err(CuckooError::InvalidParams(
                "n must be a power of 4 (4^k)".into(),
            ));
        }
        validate_common_params(4, b, fingerprint_bits)?;
        let table = TagTable::new(n, b, fingerprint_bits);
        Ok(CuckooFilter {
            table,
            scheme: Standard4aryScheme { num_buckets: n },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
        })
    }

    /// Create a standard 4-ary filter sized to hold at least `max_items`.
    ///
    /// # Arguments
    ///
    /// - `max_items` — minimum number of items the filter should hold.
    /// - `b` — tags per bucket; must be ≥ 1.
    /// - `fingerprint_bits` — fingerprint bit width; must be in `[⌊log2(2b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter sized so that inserting `max_items` keeps the projected load factor below the
    /// empirical target for (arity=4, b). See [`MAX_LOAD_FACTOR`] for values.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `b` or `fingerprint_bits` are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::Standard4aryCuckooFilter;
    ///
    /// let mut f = Standard4aryCuckooFilter::from_num_items(10_000, 4, 12).unwrap();
    /// f.add("hello").unwrap();
    /// assert!(f.contain("hello"));
    /// ```
    pub fn from_num_items(
        max_items: u64,
        b: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(4, b, fingerprint_bits)?;
        let buckets_needed = max_items.div_ceil(b as u64);
        let mut n = next_power_of_4(buckets_needed) as u32;
        if n < 1 {
            n = 1;
        }
        let target = target_load_factor(4, b as usize);
        while max_items as f64 / (n as u64 * b as u64) as f64 > target {
            n *= 4;
        }
        Self::new(n, b, fingerprint_bits)
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
    /// - `n` — total buckets; must be a power of 2 and ≥ 4 (so each of the four segments
    ///   is also a power of 2).
    /// - `b` — tags per bucket; must be ≥ 1.
    /// - `fingerprint_bits` — fingerprint bit width; must be in `[⌊log2(2b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter with capacity `n * b` tags and an initial load factor of 0.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::Segmented4aryCuckooFilter;
    ///
    /// // n = 64, so each of 4 segments has 16 buckets
    /// let mut f = Segmented4aryCuckooFilter::new(64, 4, 12).unwrap();
    /// f.add("data").unwrap();
    /// assert!(f.contain("data"));
    /// assert!(Segmented4aryCuckooFilter::new(2, 4, 12).is_err()); // n=2 < 4
    /// ```
    pub fn new(n: u32, b: u32, fingerprint_bits: u32) -> Result<Self, CuckooError> {
        if !n.is_power_of_two() || n < 4 {
            return Err(CuckooError::InvalidParams(
                "n must be a power of 2 and >= 4 (so each of 4 segments is a power of 2)".into(),
            ));
        }
        validate_common_params(4, b, fingerprint_bits)?;
        let seg = n / 4;
        let table = TagTable::new(n, b, fingerprint_bits);
        Ok(CuckooFilter {
            table,
            scheme: Segmented4aryScheme { segment_size: seg },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
        })
    }

    /// Create a segmented 4-ary filter sized to hold at least `max_items`.
    ///
    /// Each segment must be a power of 2; this method computes the smallest valid `n = 4 * seg`
    /// where `seg = 2^m ≥ ceil(max_items / (4 * b))`, then doubles `seg` until the projected
    /// load is within the empirical target for (arity=4, b).
    ///
    /// # Arguments
    ///
    /// - `max_items` — minimum number of items the filter should hold.
    /// - `b` — tags per bucket; must be ≥ 1.
    /// - `fingerprint_bits` — fingerprint bit width; must be in `[⌊log2(2b)⌋+1, 32]`.
    ///
    /// # Returns
    ///
    /// A filter sized so that inserting `max_items` keeps the projected load factor below the
    /// empirical target for (arity=4, b). See [`MAX_LOAD_FACTOR`] for values.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `b` or `fingerprint_bits` are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::Segmented4aryCuckooFilter;
    ///
    /// let mut f = Segmented4aryCuckooFilter::from_num_items(10_000, 4, 12).unwrap();
    /// f.add("hello").unwrap();
    /// assert!(f.contain("hello"));
    /// ```
    pub fn from_num_items(
        max_items: u64,
        b: u32,
        fingerprint_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(4, b, fingerprint_bits)?;
        let slots_per_segment = max_items.div_ceil(4 * b as u64);
        let mut seg = upper_power_of_2(slots_per_segment) as u32;
        if seg < 1 {
            seg = 1;
        }
        let mut n = 4 * seg;
        let target = target_load_factor(4, b as usize);
        while max_items as f64 / (n as u64 * b as u64) as f64 > target {
            seg *= 2;
            n = 4 * seg;
        }
        Self::new(n, b, fingerprint_bits)
    }
}

// ─── Generic filter operations ──────────────────────────────────────────────

impl<S: IndexScheme> CuckooFilter<S> {
    /// Insert an item into the filter.
    ///
    /// First attempts a direct placement into any of the k candidate buckets. If all are full,
    /// enters a cuckoo-kicking loop: evicts a random slot and relocates the displaced tag to
    /// one of its own alternates, repeating up to `max_kicks` times. On exhaustion, rolls back
    /// all mutations and returns [`CuckooError::TableFull`].
    ///
    /// This is the hot-path insert. It carries zero statistics overhead.
    /// Use [`add_with_stats`](Self::add_with_stats) when eviction chain analysis is needed.
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
    /// O(1) amortised — direct placement is O(k). Kicking is O(max_kicks * k) worst case
    /// but O(1) expected when load factor is well below saturation (~95%).
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
    /// use segmented_cuckoo_filter::{CuckooError, SegmentedCuckooFilter};
    ///
    /// let mut f = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
    /// f.add("hello").unwrap();
    /// assert!(f.contain("hello"));
    ///
    /// // Tiny filter — eventually returns TableFull
    /// let mut tiny = SegmentedCuckooFilter::new(2, 1, 2).unwrap();
    /// let mut full = false;
    /// for i in 0u32..100 {
    ///     if tiny.add(i.to_le_bytes()).is_err() { full = true; break; }
    /// }
    /// assert!(full);
    /// ```
    pub fn add<T: AsRef<[u8]>>(&mut self, item: T) -> Result<(), CuckooError> {
        let (tag, indices) = self
            .scheme
            .hash_item(item.as_ref(), self.table.bits_per_tag);
        let k = self.scheme.arity();

        // Try all k candidate buckets
        for pos in 0..k {
            if let Some(_slot) = self.table.insert_tag_to_bucket(indices[pos], tag) {
                self.num_items += 1;
                return Ok(());
            }
        }

        // All k buckets full — start kicking
        let mut rng = rand::rng();
        let start_pos = rng.random_range(0..k as u32) as usize;
        let mut cur_index = indices[start_pos];
        let mut cur_tag = tag;

        // chain[i] = (bucket, slot, original_tag)
        let mut chain: Vec<(u32, u32, u32)> = Vec::with_capacity(self.max_kicks as usize);

        for _ in 0..self.max_kicks {
            // Evict a random slot from cur_index
            let slot = rng.random_range(0..self.table.tags_per_bucket);
            let evicted_tag = self.table.read_tag(cur_index, slot);
            let evicted_pos = self.scheme.position_of(cur_index);

            // Write cur_tag into this slot
            chain.push((cur_index, slot, evicted_tag));
            self.table.write_tag(cur_index, slot, cur_tag);

            // Compute all candidates for the evicted tag
            cur_tag = evicted_tag;
            let all = self.scheme.all_indices(cur_index, evicted_tag, evicted_pos);

            // Try to insert evicted tag into each alternate bucket
            let mut placed = false;
            for p in 0..k {
                if p as u8 == evicted_pos {
                    continue;
                }
                if let Some(_ins_slot) = self.table.insert_tag_to_bucket(all[p], evicted_tag) {
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
            for p in 0..k {
                if p as u8 != evicted_pos {
                    alts[alt_count] = p as u8;
                    alt_count += 1;
                }
            }
            let next_pos = alts[rng.random_range(0..alt_count as u32) as usize];
            cur_index = all[next_pos as usize];
        }

        // Kicks exhausted — rollback all changes
        for &(bucket, slot, original_tag) in chain.iter().rev() {
            self.table.write_tag(bucket, slot, original_tag);
        }
        Err(CuckooError::TableFull)
    }

    /// Insert an item and return eviction statistics.
    ///
    /// Functionally identical to [`add`](Self::add) but returns [`InsertStats`] on success.
    /// Kept as a separate code path so that `add()` has zero statistics overhead — no branch
    /// or counter is paid when statistics are not needed.
    ///
    /// # Arguments
    ///
    /// - `item` — any value that implements `AsRef<[u8]>`.
    ///
    /// # Returns
    ///
    /// `Ok(InsertStats)` on success. `InsertStats::kicks` is 0 for direct placements.
    ///
    /// # Errors
    ///
    /// - [`CuckooError::TableFull`] — kick budget exhausted; table is **unchanged**.
    ///
    /// # Performance
    ///
    /// O(1) amortised; identical to [`add`](Self::add) plus one counter increment per kick.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::SegmentedCuckooFilter;
    ///
    /// let mut f = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
    /// let stats = f.add_with_stats("hello").unwrap();
    /// // First item in an empty filter is always placed directly.
    /// assert_eq!(stats.kicks, 0);
    /// assert!(f.contain("hello"));
    /// ```
    pub fn add_with_stats<T: AsRef<[u8]>>(&mut self, item: T) -> Result<InsertStats, CuckooError> {
        let (tag, indices) = self
            .scheme
            .hash_item(item.as_ref(), self.table.bits_per_tag);
        let k = self.scheme.arity();

        // Try all k candidate buckets
        for pos in 0..k {
            if let Some(_slot) = self.table.insert_tag_to_bucket(indices[pos], tag) {
                self.num_items += 1;
                return Ok(InsertStats { kicks: 0 });
            }
        }

        // All k buckets full — start kicking
        let mut rng = rand::rng();
        let start_pos = rng.random_range(0..k as u32) as usize;
        let mut cur_index = indices[start_pos];
        let mut cur_tag = tag;

        let mut chain: Vec<(u32, u32, u32)> = Vec::with_capacity(self.max_kicks as usize);
        let mut kicks: u32 = 0;

        for _ in 0..self.max_kicks {
            let slot = rng.random_range(0..self.table.tags_per_bucket);
            let evicted_tag = self.table.read_tag(cur_index, slot);
            let evicted_pos = self.scheme.position_of(cur_index);

            chain.push((cur_index, slot, evicted_tag));
            self.table.write_tag(cur_index, slot, cur_tag);

            cur_tag = evicted_tag;
            let all = self.scheme.all_indices(cur_index, evicted_tag, evicted_pos);
            kicks += 1;

            let mut placed = false;
            for p in 0..k {
                if p as u8 == evicted_pos {
                    continue;
                }
                if let Some(_ins_slot) = self.table.insert_tag_to_bucket(all[p], evicted_tag) {
                    self.num_items += 1;
                    placed = true;
                    break;
                }
            }
            if placed {
                return Ok(InsertStats { kicks });
            }

            let mut alts = [0u8; 3];
            let mut alt_count = 0;
            for p in 0..k {
                if p as u8 != evicted_pos {
                    alts[alt_count] = p as u8;
                    alt_count += 1;
                }
            }
            let next_pos = alts[rng.random_range(0..alt_count as u32) as usize];
            cur_index = all[next_pos as usize];
        }

        // Kicks exhausted — rollback all changes
        for &(bucket, slot, original_tag) in chain.iter().rev() {
            self.table.write_tag(bucket, slot, original_tag);
        }
        Err(CuckooError::TableFull)
    }

    /// Check whether the filter likely contains `item`.
    ///
    /// Probes all k candidate buckets for a matching fingerprint.
    ///
    /// # Arguments
    ///
    /// - `item` — any value that implements `AsRef<[u8]>`.
    ///
    /// # Returns
    ///
    /// - `true` if the fingerprint is found in any candidate bucket. May return `true` for
    ///   items that were never inserted (false positive); the theoretical rate is `2b / 2^f`.
    /// - `false` if no matching fingerprint is found. Never a false negative for items
    ///   currently in the filter.
    ///
    /// # Performance
    ///
    /// O(k * b) — probes k buckets, scanning up to b slots each. For typical parameters
    /// (k ≤ 4, b ≤ 4) this is effectively O(1).
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
    /// use segmented_cuckoo_filter::SegmentedCuckooFilter;
    ///
    /// let mut f = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
    /// f.add("hello").unwrap();
    /// assert!(f.contain("hello"));
    /// assert!(!f.contain("world")); // likely false for a fresh filter
    /// ```
    pub fn contain<T: AsRef<[u8]>>(&self, item: T) -> bool {
        let (tag, indices) = self
            .scheme
            .hash_item(item.as_ref(), self.table.bits_per_tag);
        let k = self.scheme.arity();
        for i in 0..k {
            if self.table.find_tag_in_bucket(indices[i], tag) {
                return true;
            }
        }
        false
    }

    /// Delete `item` from the filter.
    ///
    /// Probes all k candidate buckets for a matching fingerprint. On the first match,
    /// zeroes the tag slot and (for standard k > 2 schemes) clears the 2-bit chain-position
    /// entry to avoid stale position data influencing future kicks.
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
    /// O(k × b) — same complexity as [`contain`](Self::contain).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::{CuckooError, SegmentedCuckooFilter};
    ///
    /// let mut f = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
    /// f.add("hello").unwrap();
    /// f.delete("hello").unwrap();
    /// assert!(!f.contain("hello"));
    /// // Second delete of the same item returns an error.
    /// assert!(matches!(f.delete("hello"), Err(CuckooError::NotFound)));
    /// assert!(matches!(f.delete("never_inserted"), Err(CuckooError::NotFound)));
    /// ```
    pub fn delete<T: AsRef<[u8]>>(&mut self, item: T) -> Result<(), CuckooError> {
        let (tag, indices) = self
            .scheme
            .hash_item(item.as_ref(), self.table.bits_per_tag);
        let k = self.scheme.arity();

        for i in 0..k {
            if let Some(slot) = self.table.find_tag_in_bucket_slot(indices[i], tag) {
                self.table.write_tag(indices[i], slot, 0);
                self.num_items -= 1;
                return Ok(());
            }
        }
        Err(CuckooError::NotFound)
    }

    /// Return the number of items currently stored in the filter.
    ///
    /// This counter is maintained exactly by `add`, `add_with_stats`, and `delete`. It is not
    /// affected by false positives.
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
    /// use segmented_cuckoo_filter::SegmentedCuckooFilter;
    ///
    /// let mut f = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
    /// assert_eq!(f.size(), 0);
    /// f.add("a").unwrap();
    /// assert_eq!(f.size(), 1);
    /// f.delete("a").unwrap();
    /// assert_eq!(f.size(), 0);
    /// ```
    pub fn size(&self) -> u64 {
        self.num_items
    }

    /// Return the byte size of the underlying fingerprint storage.
    ///
    /// Reports the logical storage footprint `ceil(n * b * fp_bits / 8)` bytes, excluding
    /// the 8-byte alignment padding added by [`TagTable`].
    ///
    /// # Returns
    ///
    /// Number of bytes used by the fingerprint array (without padding).
    ///
    /// # Performance
    ///
    /// O(1) — reads a field from [`TagTable`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::SegmentedCuckooFilter;
    ///
    /// // n=64, b=4, fp_bits=12 → 64*4*12 = 3072 bits = 384 bytes
    /// let f = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
    /// assert_eq!(f.size_in_bytes(), 384);
    /// ```
    pub fn size_in_bytes(&self) -> usize {
        self.table.size_in_bytes()
    }

    /// Return the current load factor: `num_items / (n * b)`.
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
    /// use segmented_cuckoo_filter::SegmentedCuckooFilter;
    ///
    /// let mut f = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
    /// assert_eq!(f.load_factor(), 0.0);
    /// f.add("hello").unwrap();
    /// // After inserting 1 item into a 256-slot filter: load = 1/256 ≈ 0.0039
    /// assert!(f.load_factor() > 0.0);
    /// assert!(f.load_factor() < 1.0);
    /// ```
    pub fn load_factor(&self) -> f64 {
        self.num_items as f64 / (self.table.num_buckets as f64 * self.table.tags_per_bucket as f64)
    }

    /// Override the maximum number of cuckoo kicks before declaring the table full.
    ///
    /// The default is 500. A higher value increases the probability of a successful insert
    /// at very high load factors at the cost of slower worst-case inserts. Set to 0 to disable
    /// kicking entirely (only direct placements succeed).
    ///
    /// # Arguments
    ///
    /// - `max_kicks` — new kick budget per insert attempt.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::SegmentedCuckooFilter;
    ///
    /// let mut f = SegmentedCuckooFilter::new(2, 1, 2).unwrap();
    /// f.set_max_kicks(0); // disable kicking — only 2 direct placements possible
    /// let mut count = 0u32;
    /// for i in 0u32..100 {
    ///     if f.add(i.to_le_bytes()).is_ok() { count += 1; }
    /// }
    /// assert!(count <= 2);
    /// ```
    pub fn set_max_kicks(&mut self, max_kicks: u32) {
        self.max_kicks = max_kicks;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type SegmentedCuckooFilter = CuckooFilter<SegmentedScheme>;
    type StandardCuckooFilter = CuckooFilter<StandardScheme>;
    type Standard3aryCuckooFilter = CuckooFilter<Standard3aryScheme>;
    type Segmented3aryCuckooFilter = CuckooFilter<Segmented3aryScheme>;
    type Standard4aryCuckooFilter = CuckooFilter<Standard4aryScheme>;
    type Segmented4aryCuckooFilter = CuckooFilter<Segmented4aryScheme>;

    // ─── 2-ary Segmented ────────────────────────────────────────────────────

    #[test]
    fn segmented_insert_contain_delete() {
        let mut filter = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
        assert!(filter.add("hello").is_ok());
        assert!(filter.add("world").is_ok());
        assert!(filter.contain("hello"));
        assert!(filter.contain("world"));
        assert!(!filter.contain("missing"));
        // TEST-CHANGE: delete now returns Result<(), CuckooError> instead of bool.
        // Specification changed: API unified with add() for consistency.
        assert!(filter.delete("hello").is_ok());
        assert!(!filter.contain("hello"));
        assert!(filter.contain("world"));
        assert_eq!(filter.size(), 1);
    }

    #[test]
    fn segmented_insert_many() {
        let mut filter = SegmentedCuckooFilter::from_num_items(10000, 4, 12).unwrap();
        let mut inserted = 0u64;
        for i in 0u64..10000 {
            if filter.add(i.to_le_bytes()).is_ok() {
                inserted += 1;
            } else {
                break;
            }
        }
        assert_eq!(filter.size(), inserted);
        for i in 0u64..inserted {
            assert!(
                filter.contain(i.to_le_bytes()),
                "item {i} should be present"
            );
        }
    }

    #[test]
    fn segmented_table_full() {
        let mut filter = SegmentedCuckooFilter::new(2, 1, 2).unwrap();
        let mut count = 0;
        for i in 0u32..1000 {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            count += 1;
        }
        assert!(count <= 2, "tiny filter should fill quickly, got {count}");
    }

    #[test]
    fn segmented_fill_correctness() {
        let mut filter = SegmentedCuckooFilter::new(4, 1, 2).unwrap();
        let mut last_ok = 0u32;
        for i in 0u32..1000 {
            if filter.add(i.to_le_bytes()).is_err() {
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
        let filter = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
        assert_eq!(filter.load_factor(), 0.0);
    }

    #[test]
    fn segmented_invalid_params() {
        assert!(SegmentedCuckooFilter::new(3, 4, 12).is_err());
        assert!(SegmentedCuckooFilter::new(1, 4, 12).is_err());
        assert!(SegmentedCuckooFilter::new(4, 0, 12).is_err());
        assert!(SegmentedCuckooFilter::new(4, 4, 0).is_err());
        assert!(SegmentedCuckooFilter::new(4, 4, 33).is_err());
        assert!(SegmentedCuckooFilter::new(4, 4, 3).is_err());
    }

    // ─── 2-ary Standard ────────────────────────────────────────────────────

    #[test]
    fn standard_insert_contain_delete() {
        let mut filter = StandardCuckooFilter::new(64, 4, 12).unwrap();
        assert!(filter.add("hello").is_ok());
        assert!(filter.add("world").is_ok());
        assert!(filter.contain("hello"));
        assert!(filter.contain("world"));
        assert!(!filter.contain("missing"));
        // TEST-CHANGE: delete now returns Result<(), CuckooError>.
        assert!(filter.delete("hello").is_ok());
        assert!(!filter.contain("hello"));
        assert!(filter.contain("world"));
        assert_eq!(filter.size(), 1);
    }

    #[test]
    fn standard_insert_many() {
        let mut filter = StandardCuckooFilter::from_num_items(10000, 4, 12).unwrap();
        let mut inserted = 0u64;
        for i in 0u64..10000 {
            if filter.add(i.to_le_bytes()).is_ok() {
                inserted += 1;
            } else {
                break;
            }
        }
        assert_eq!(filter.size(), inserted);
        for i in 0u64..inserted {
            assert!(
                filter.contain(i.to_le_bytes()),
                "item {i} should be present"
            );
        }
    }

    #[test]
    fn standard_table_full() {
        let mut filter = StandardCuckooFilter::new(2, 1, 2).unwrap();
        let mut count = 0;
        for i in 0u32..1000 {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            count += 1;
        }
        assert!(count <= 2, "tiny filter should fill quickly, got {count}");
    }

    #[test]
    fn standard_fill_correctness() {
        let mut filter = StandardCuckooFilter::new(4, 1, 2).unwrap();
        let mut last_ok = 0u32;
        for i in 0u32..1000 {
            if filter.add(i.to_le_bytes()).is_err() {
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
        assert!(StandardCuckooFilter::new(3, 4, 12).is_err());
        assert!(StandardCuckooFilter::new(0, 4, 12).is_err());
        assert!(StandardCuckooFilter::new(4, 0, 12).is_err());
        assert!(StandardCuckooFilter::new(4, 4, 0).is_err());
        assert!(StandardCuckooFilter::new(4, 4, 33).is_err());
        assert!(StandardCuckooFilter::new(4, 4, 3).is_err());
    }

    #[test]
    fn standard_n1_degenerate() {
        let mut filter = StandardCuckooFilter::new(1, 4, 4).unwrap();
        let mut count = 0;
        for i in 0u32..1000 {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            count += 1;
        }
        assert!(
            count <= 4,
            "n=1, b=4 filter holds at most 4 items, got {count}"
        );
    }

    // ─── 3-ary Standard ────────────────────────────────────────────────────

    #[test]
    fn standard_3ary_insert_contain_delete() {
        let mut filter = Standard3aryCuckooFilter::new(243, 4, 12).unwrap();
        assert!(filter.add("hello").is_ok());
        assert!(filter.add("world").is_ok());
        assert!(filter.contain("hello"));
        assert!(filter.contain("world"));
        assert!(!filter.contain("missing"));
        // TEST-CHANGE: delete now returns Result<(), CuckooError>.
        assert!(filter.delete("hello").is_ok());
        assert!(!filter.contain("hello"));
        assert!(filter.contain("world"));
        assert_eq!(filter.size(), 1);
    }

    #[test]
    fn standard_3ary_insert_many() {
        let mut filter = Standard3aryCuckooFilter::from_num_items(10000, 4, 12).unwrap();
        let mut inserted = 0u64;
        for i in 0u64..10000 {
            if filter.add(i.to_le_bytes()).is_ok() {
                inserted += 1;
            } else {
                break;
            }
        }
        assert_eq!(filter.size(), inserted);
        for i in 0u64..inserted {
            assert!(
                filter.contain(i.to_le_bytes()),
                "item {i} should be present"
            );
        }
    }

    #[test]
    fn standard_3ary_fill_correctness() {
        let mut filter = Standard3aryCuckooFilter::new(27, 2, 8).unwrap();
        let mut last_ok = 0u32;
        for i in 0u32..1000 {
            if filter.add(i.to_le_bytes()).is_err() {
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
        // n = 3 * 32 = 96
        let mut filter = Segmented3aryCuckooFilter::new(96, 4, 12).unwrap();
        assert!(filter.add("hello").is_ok());
        assert!(filter.add("world").is_ok());
        assert!(filter.contain("hello"));
        assert!(filter.contain("world"));
        assert!(!filter.contain("missing"));
        // TEST-CHANGE: delete now returns Result<(), CuckooError>.
        assert!(filter.delete("hello").is_ok());
        assert!(!filter.contain("hello"));
        assert_eq!(filter.size(), 1);
    }

    #[test]
    fn segmented_3ary_insert_many() {
        let mut filter = Segmented3aryCuckooFilter::from_num_items(10000, 4, 12).unwrap();
        let mut inserted = 0u64;
        for i in 0u64..10000 {
            if filter.add(i.to_le_bytes()).is_ok() {
                inserted += 1;
            } else {
                break;
            }
        }
        assert_eq!(filter.size(), inserted);
        for i in 0u64..inserted {
            assert!(
                filter.contain(i.to_le_bytes()),
                "item {i} should be present"
            );
        }
    }

    #[test]
    fn segmented_3ary_invalid_params() {
        assert!(Segmented3aryCuckooFilter::new(64, 4, 12).is_err()); // 64/3 not integer
        assert!(Segmented3aryCuckooFilter::new(9, 4, 12).is_err()); // 9/3=3, not power of 2
        assert!(Segmented3aryCuckooFilter::new(3, 4, 12).is_ok()); // 3/3=1, power of 2
        assert!(Segmented3aryCuckooFilter::new(96, 4, 12).is_ok()); // 96/3=32, power of 2
    }

    // ─── 4-ary Standard ────────────────────────────────────────────────────

    #[test]
    fn standard_4ary_insert_contain_delete() {
        let mut filter = Standard4aryCuckooFilter::new(64, 4, 12).unwrap();
        assert!(filter.add("hello").is_ok());
        assert!(filter.add("world").is_ok());
        assert!(filter.contain("hello"));
        assert!(filter.contain("world"));
        assert!(!filter.contain("missing"));
        // TEST-CHANGE: delete now returns Result<(), CuckooError>.
        assert!(filter.delete("hello").is_ok());
        assert!(!filter.contain("hello"));
        assert_eq!(filter.size(), 1);
    }

    #[test]
    fn standard_4ary_insert_many() {
        let mut filter = Standard4aryCuckooFilter::from_num_items(10000, 4, 12).unwrap();
        let mut inserted = 0u64;
        for i in 0u64..10000 {
            if filter.add(i.to_le_bytes()).is_ok() {
                inserted += 1;
            } else {
                break;
            }
        }
        assert_eq!(filter.size(), inserted);
        for i in 0u64..inserted {
            assert!(
                filter.contain(i.to_le_bytes()),
                "item {i} should be present"
            );
        }
    }

    #[test]
    fn standard_4ary_fill_correctness() {
        let mut filter = Standard4aryCuckooFilter::new(16, 2, 8).unwrap();
        let mut last_ok = 0u32;
        for i in 0u32..1000 {
            if filter.add(i.to_le_bytes()).is_err() {
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
        // n = 4 * 16 = 64
        let mut filter = Segmented4aryCuckooFilter::new(64, 4, 12).unwrap();
        assert!(filter.add("hello").is_ok());
        assert!(filter.add("world").is_ok());
        assert!(filter.contain("hello"));
        assert!(filter.contain("world"));
        assert!(!filter.contain("missing"));
        // TEST-CHANGE: delete now returns Result<(), CuckooError>.
        assert!(filter.delete("hello").is_ok());
        assert!(!filter.contain("hello"));
        assert_eq!(filter.size(), 1);
    }

    #[test]
    fn segmented_4ary_insert_many() {
        let mut filter = Segmented4aryCuckooFilter::from_num_items(10000, 4, 12).unwrap();
        let mut inserted = 0u64;
        for i in 0u64..10000 {
            if filter.add(i.to_le_bytes()).is_ok() {
                inserted += 1;
            } else {
                break;
            }
        }
        assert_eq!(filter.size(), inserted);
        for i in 0u64..inserted {
            assert!(
                filter.contain(i.to_le_bytes()),
                "item {i} should be present"
            );
        }
    }

    #[test]
    fn segmented_4ary_invalid_params() {
        assert!(Segmented4aryCuckooFilter::new(3, 4, 12).is_err()); // not power of 2
        assert!(Segmented4aryCuckooFilter::new(2, 4, 12).is_err()); // < 4
        assert!(Segmented4aryCuckooFilter::new(4, 4, 12).is_ok()); // 4/4=1 ✓
        assert!(Segmented4aryCuckooFilter::new(64, 4, 12).is_ok()); // 64/4=16 ✓
    }

    // ─── Shared behaviour across all types ─────────────────────────────────

    #[test]
    fn add_with_stats_direct_placement_has_zero_kicks() {
        // Fresh filter — first item must be placed without kicking.
        let mut filter = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
        let stats = filter.add_with_stats("hello").unwrap();
        assert_eq!(stats.kicks, 0);
    }

    #[test]
    fn add_with_stats_tracks_kicks() {
        // Fill with add(), then use add_with_stats on a fresh item. If the filter is near full
        // some kicking will be required. We can only verify kicks >= 0 deterministically; what
        // we care about is that the item ends up in the filter.
        let mut filter = SegmentedCuckooFilter::from_num_items(500, 4, 12).unwrap();
        for i in 0u64..400 {
            filter.add(i.to_le_bytes()).ok();
        }
        if let Ok(stats) = filter.add_with_stats(9999u64.to_le_bytes()) {
            assert!(filter.contain(9999u64.to_le_bytes()));
            let _ = stats.kicks; // just verify the field is accessible
        }
    }

    #[test]
    fn add_with_stats_rollback_on_full() {
        // A tiny filter will fill quickly. Once full, add_with_stats must return TableFull.
        let mut filter = SegmentedCuckooFilter::new(2, 1, 2).unwrap();
        filter.set_max_kicks(10);
        let size_before_fail;
        loop {
            match filter.add_with_stats(42u32.to_le_bytes()) {
                Ok(_) => {}
                Err(CuckooError::TableFull) => {
                    size_before_fail = filter.size();
                    break;
                }
                Err(e) => panic!("unexpected error: {}", e),
            }
        }
        // After TableFull the size must not have changed.
        assert_eq!(filter.size(), size_before_fail);
    }

    #[test]
    fn rollback_on_table_full_leaves_existing_items_intact() {
        // Fill a filter to capacity, record all inserted items, then attempt one more insert.
        // All previously inserted items must still be queryable.
        let mut filter = SegmentedCuckooFilter::new(4, 2, 4).unwrap();
        filter.set_max_kicks(50);
        let mut inserted: Vec<u32> = Vec::new();
        for i in 0u32..1000 {
            match filter.add(i.to_le_bytes()) {
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
        let mut filter = StandardCuckooFilter::new(64, 4, 12).unwrap();
        filter.add("hello").unwrap();
        assert!(matches!(
            filter.delete("not_inserted"),
            Err(CuckooError::NotFound)
        ));
        assert_eq!(filter.size(), 1); // "hello" must still be there
    }

    #[test]
    fn size_in_bytes_matches_formula() {
        // n=64, b=4, fp=12 → 64*4*12 = 3072 bits = 384 bytes
        let filter = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
        assert_eq!(filter.size_in_bytes(), 384);
        // n=32, b=2, fp=8 → 32*2*8 = 512 bits = 64 bytes
        let filter2 = StandardCuckooFilter::new(32, 2, 8).unwrap();
        assert_eq!(filter2.size_in_bytes(), 64);
    }

    #[test]
    fn set_max_kicks_limits_kick_budget() {
        // With max_kicks=0, no kicking is attempted; any full-bucket insert fails.
        // Use a tiny filter with b=1 so the second insert into the same pair of buckets fails.
        let mut filter = SegmentedCuckooFilter::new(2, 1, 2).unwrap();
        filter.set_max_kicks(0);
        // First two inserts may succeed (one per bucket half); further inserts must fail
        // without kicks.
        let mut count = 0u32;
        for i in 0u32..100 {
            if filter.add(i.to_le_bytes()).is_ok() {
                count += 1;
            }
        }
        // With no kicks, at most b*n = 1*2 = 2 items can be placed directly.
        assert!(count <= 2, "expected ≤ 2 direct placements, got {count}");
    }

    #[test]
    fn load_factor_increases_with_inserts() {
        let mut filter = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
        assert_eq!(filter.load_factor(), 0.0);
        filter.add("a").unwrap();
        assert!(filter.load_factor() > 0.0);
        filter.add("b").unwrap();
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

    #[test]
    fn standard_3ary_add_with_stats() {
        let mut filter = Standard3aryCuckooFilter::new(243, 4, 12).unwrap();
        let stats = filter.add_with_stats("hello").unwrap();
        assert_eq!(stats.kicks, 0);
        assert!(filter.contain("hello"));
    }

    #[test]
    fn segmented_4ary_add_with_stats() {
        let mut filter = Segmented4aryCuckooFilter::new(64, 4, 12).unwrap();
        let stats = filter.add_with_stats("hello").unwrap();
        assert_eq!(stats.kicks, 0);
        assert!(filter.contain("hello"));
    }

    // ─── Comprehensive delete tests ─────────────────────────────────────────
    //
    // The macro generates 9 delete-contract tests for every one of the 6 filter
    // variants. Additional non-macro tests cover position-storage clearing for
    // the standard k>2 schemes, and proptest property tests cover the three
    // most representative variants.

    /// Generates the 9 delete-contract test functions inside a variant-specific
    /// sub-module. Invoked once per filter type below.
    ///
    /// Parameters:
    /// - `$mod_name` — the sub-module identifier (e.g., `seg2`)
    /// - `$filter_ty` — the concrete filter type (e.g., `SegmentedCuckooFilter`)
    /// - `$new` — an expression that constructs a fresh filter (must succeed)
    macro_rules! delete_contract_tests {
        ($mod_name:ident, $filter_ty:ty, $new:expr) => {
            mod $mod_name {
                use super::*;

                /// Insert X, delete X → Ok, contain returns false, size decremented.
                #[test]
                fn delete_inserted_item() {
                    let mut f = $new;
                    f.add("item_x").unwrap();
                    assert_eq!(f.size(), 1);
                    f.delete("item_x").unwrap();
                    assert!(!f.contain("item_x"));
                    assert_eq!(f.size(), 0);
                }

                /// delete on a fresh empty filter → Err(NotFound), size stays 0.
                #[test]
                fn delete_from_empty_filter() {
                    let mut f: $filter_ty = $new;
                    assert!(matches!(f.delete("anything"), Err(CuckooError::NotFound)));
                    assert_eq!(f.size(), 0);
                }

                /// Insert A, delete B → Err(NotFound), A still present, size unchanged.
                #[test]
                fn delete_absent_from_nonempty() {
                    let mut f = $new;
                    f.add("item_a").unwrap();
                    assert!(matches!(f.delete("item_b"), Err(CuckooError::NotFound)));
                    assert!(f.contain("item_a"));
                    assert_eq!(f.size(), 1);
                }

                /// Insert X, delete twice → first Ok, second Err(NotFound).
                #[test]
                fn double_delete() {
                    let mut f = $new;
                    f.add("item_x").unwrap();
                    f.delete("item_x").unwrap();
                    assert!(matches!(f.delete("item_x"), Err(CuckooError::NotFound)));
                    assert_eq!(f.size(), 0);
                }

                /// Insert A and B, delete A → B still present, size = 1.
                #[test]
                fn delete_preserves_other_items() {
                    let mut f = $new;
                    f.add("item_a").unwrap();
                    f.add("item_b").unwrap();
                    f.delete("item_a").unwrap();
                    assert!(!f.contain("item_a"));
                    assert!(f.contain("item_b"));
                    assert_eq!(f.size(), 1);
                }

                /// Insert X, delete X, reinsert X → Ok, contain true, size = 1.
                #[test]
                fn insert_delete_reinsert() {
                    let mut f = $new;
                    f.add("item_x").unwrap();
                    f.delete("item_x").unwrap();
                    f.add("item_x").unwrap();
                    assert!(f.contain("item_x"));
                    assert_eq!(f.size(), 1);
                }

                /// Fill the filter, delete every inserted item → size = 0, none found.
                #[test]
                fn fill_then_delete_all() {
                    let mut f = $new;
                    let mut inserted: Vec<u64> = Vec::new();
                    for i in 0u64..10_000 {
                        match f.add(i.to_le_bytes()) {
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
                    assert_eq!(f.size(), 0);
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
                        match f.add(i.to_le_bytes()) {
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
                        f.add(i.to_le_bytes()).unwrap();
                    }
                    let size_before = f.size();
                    for i in 1000u64..1100 {
                        let _ = f.delete(i.to_le_bytes()); // all should be NotFound
                    }
                    assert_eq!(f.size(), size_before);
                }
            }
        };
    }

    // Invoke for all 6 filter variants.
    delete_contract_tests!(
        delete_seg2,
        CuckooFilter<SegmentedScheme>,
        CuckooFilter::<SegmentedScheme>::new(64, 4, 12).unwrap()
    );
    delete_contract_tests!(
        delete_std2,
        CuckooFilter<StandardScheme>,
        CuckooFilter::<StandardScheme>::new(64, 4, 12).unwrap()
    );
    delete_contract_tests!(
        delete_std3,
        CuckooFilter<Standard3aryScheme>,
        CuckooFilter::<Standard3aryScheme>::new(81, 4, 12).unwrap()
    );
    delete_contract_tests!(
        delete_seg3,
        CuckooFilter<Segmented3aryScheme>,
        // n = 3*32 = 96
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

    // ─── Position-clearing tests (standard k>2 only) ─────────────────────────
    //
    // These verify that the 2-bit chain-position field is zeroed on deletion so that a
    // subsequent insert landing in the same slot reads a clean position, not stale data
    // from the previous occupant.

    #[test]
    fn delete_clears_position_standard_3ary() {
        let mut f = Standard3aryCuckooFilter::new(243, 4, 12).unwrap();
        // Insert many items to increase the likelihood of position storage being exercised.
        let mut inserted: Vec<u64> = Vec::new();
        for i in 0u64..200 {
            match f.add(i.to_le_bytes()) {
                Ok(()) => inserted.push(i),
                Err(CuckooError::TableFull) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }
        // Delete all inserted items.
        for &item in &inserted {
            f.delete(item.to_le_bytes()).unwrap();
        }
        assert_eq!(f.size(), 0);
        // Re-insert a fresh set of items. If stale position data remained after deletion,
        // lookup or eviction logic could become inconsistent, causing incorrect lookups.
        let mut reinserted: Vec<u64> = Vec::new();
        for i in 1000u64..1200 {
            match f.add(i.to_le_bytes()) {
                Ok(()) => reinserted.push(i),
                Err(CuckooError::TableFull) => break,
                Err(e) => panic!("unexpected error on reinsert: {e}"),
            }
        }
        for &item in &reinserted {
            assert!(
                f.contain(item.to_le_bytes()),
                "item {item} not found after reinsert into cleared filter"
            );
        }
    }

    #[test]
    fn delete_clears_position_standard_4ary() {
        let mut f = Standard4aryCuckooFilter::new(64, 4, 12).unwrap();
        let mut inserted: Vec<u64> = Vec::new();
        for i in 0u64..200 {
            match f.add(i.to_le_bytes()) {
                Ok(()) => inserted.push(i),
                Err(CuckooError::TableFull) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }
        for &item in &inserted {
            f.delete(item.to_le_bytes()).unwrap();
        }
        assert_eq!(f.size(), 0);
        let mut reinserted: Vec<u64> = Vec::new();
        for i in 1000u64..1200 {
            match f.add(i.to_le_bytes()) {
                Ok(()) => reinserted.push(i),
                Err(CuckooError::TableFull) => break,
                Err(e) => panic!("unexpected error on reinsert: {e}"),
            }
        }
        for &item in &reinserted {
            assert!(
                f.contain(item.to_le_bytes()),
                "item {item} not found after reinsert into cleared filter"
            );
        }
    }

    // ─── Property tests (proptest) ───────────────────────────────────────────
    //
    // Run for Standard 2-ary, Segmented 2-ary, and Standard 3-ary (the last covers
    // the position-storage code path under proptest randomisation).

    #[cfg(test)]
    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// For any random item bytes: insert then delete leaves the filter empty
            /// and size returns to 0.
            #[test]
            fn prop_insert_delete_roundtrip_std2(item in proptest::collection::vec(any::<u8>(), 1..=64)) {
                let mut f = CuckooFilter::<StandardScheme>::new(64, 4, 12).unwrap();
                f.add(item.as_slice()).unwrap();
                assert_eq!(f.size(), 1);
                f.delete(item.as_slice()).unwrap();
                assert_eq!(f.size(), 0);
            }

            /// For any random item bytes: insert then delete leaves the filter empty
            /// and size returns to 0 (Segmented 2-ary).
            #[test]
            fn prop_insert_delete_roundtrip_seg2(item in proptest::collection::vec(any::<u8>(), 1..=64)) {
                let mut f = CuckooFilter::<SegmentedScheme>::new(64, 4, 12).unwrap();
                f.add(item.as_slice()).unwrap();
                assert_eq!(f.size(), 1);
                f.delete(item.as_slice()).unwrap();
                assert_eq!(f.size(), 0);
            }

            /// For any random item bytes on a fresh Standard 3-ary filter:
            /// delete returns Err(NotFound).
            #[test]
            fn prop_delete_without_insert_not_found_std3(item in proptest::collection::vec(any::<u8>(), 1..=64)) {
                let mut f = CuckooFilter::<Standard3aryScheme>::new(81, 4, 12).unwrap();
                assert!(matches!(
                    f.delete(item.as_slice()),
                    Err(CuckooError::NotFound)
                ));
            }

            /// Insert N distinct items (encoded as u64 little-endian), delete all N,
            /// verify size = 0 (Standard 2-ary).
            #[test]
            fn prop_insert_n_delete_n_empty_std2(
                keys in proptest::collection::vec(any::<u64>(), 1..=20)
            ) {
                let mut f = CuckooFilter::<StandardScheme>::new(64, 4, 12).unwrap();
                let mut inserted: Vec<u64> = Vec::new();
                for k in &keys {
                    if f.add(k.to_le_bytes()).is_ok() {
                        inserted.push(*k);
                    }
                }
                for k in &inserted {
                    f.delete(k.to_le_bytes()).unwrap_or_else(|e| {
                        panic!("delete({k}) failed: {e}")
                    });
                }
                assert_eq!(f.size(), 0);
            }
        }
    }
}
