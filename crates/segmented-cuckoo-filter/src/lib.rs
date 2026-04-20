#![warn(missing_docs)]
//! Cuckoo filter variants across two dimensions: indexing strategy and arity.
//!
//! # Design overview
//!
//! A *cuckoo filter* is a space-efficient probabilistic membership structure. It stores
//! fingerprints (short hashes) of inserted items and supports insert, lookup, and delete
//! with a bounded false-positive rate. Unlike Bloom filters, deletes are exact.
//!
//! This crate explores two orthogonal axes of the design space:
//!
//! ## Indexing strategy
//!
//! | Strategy   | Candidate indices              | `n` constraint             |
//! |------------|-------------------------------|----------------------------|
//! | **Standard**   | All in `[0, n)`           | Power of 2                 |
//! | **Segmented**  | `i_j ∈ [j·seg, (j+1)·seg)` | 2^m (or k·2^m for 3-ary)  |
//!
//! Standard (original) partial-key cuckoo hashing places all candidate buckets anywhere
//! in the table. Segmented (bipartite) hashing confines each candidate to its own
//! contiguous segment, eliminating cross-segment interference and improving locality.
//!
//! ## Arity (candidate buckets per item)
//!
//! | Arity | Candidate buckets | Extra storage |
//! |-------|-------------------|---------------|
//! | 2     | 2                 | None          |
//! | 3     | 3                 | None          |
//! | 4     | 4                 | None          |
//!
//! Higher arity improves load factor at the cost of more candidate probes per lookup.
//! Standard k > 2 schemes use xor3/xor4 cycling so `all_indices` reconstructs all candidates
//! from any starting index — no per-slot position storage needed for any scheme.
//!
//! ## Module structure
//!
//! ```text
//! lib.rs      — public API, type aliases
//! filter.rs   — CuckooFilter<S> generic implementation
//! scheme.rs   — IndexScheme trait + 6 scheme structs
//! hash.rs     — tag hash functions, item hashing, index reconstruction
//! bucket.rs   — TagTable: bit-packed fingerprint storage
//! util.rs     — upper_power_of_2 helper
//! ```
//!
//! # Security considerations
//!
//! - The underlying item hash (xxHash3) is **not cryptographic**. Do not use this filter
//!   for security-sensitive membership checks (e.g., password lookups, CSRF token sets).
//!   An adversary who knows the hash function can craft inputs that all map to the same
//!   buckets, causing artificially high false-positive rates or denial-of-service via
//!   table-full conditions.
//! - False positives are bounded by `d·b / 2^fp_bits` where `d` is the arity. With
//!   `fp_bits = 12`, `b = 4`, `d = 2` the theoretical FPR is `≈ 0.2%`. This is a
//!   *probabilistic* guarantee, not a cryptographic one.
//! - Deletion of an item never inserted may silently remove a fingerprint that belongs to
//!   a different item sharing the same tag and candidate indices. Only delete items you
//!   have explicitly inserted.
//!
//! # Quick start
//!
//! ```rust
//! use segmented_cuckoo_filter::SegmentedCuckooFilter;
//!
//! // Create a filter: n=64 buckets, b=4 slots/bucket, 12-bit fingerprints.
//! let mut filter = SegmentedCuckooFilter::new(64, 4, 12).unwrap();
//!
//! filter.add("hello").unwrap();
//! assert!(filter.contain("hello"));
//! assert!(!filter.contain("world"));
//!
//! filter.delete("hello").unwrap();
//! assert!(!filter.contain("hello"));
//! ```
//!
//! To auto-size based on expected item count:
//!
//! ```rust
//! use segmented_cuckoo_filter::StandardCuckooFilter;
//!
//! let mut filter = StandardCuckooFilter::from_num_items(100_000, 4, 12).unwrap();
//! filter.add(b"item".as_ref()).unwrap();
//! assert!(filter.contain(b"item".as_ref()));
//! ```

pub mod bucket;
pub mod filter;
pub mod hash;
pub mod scheme;
pub mod util;

pub use filter::{CuckooError, CuckooFilter, InsertStats};
pub use scheme::{
    IndexScheme, Segmented3aryScheme, Segmented4aryScheme, SegmentedScheme, Standard3aryScheme,
    Standard4aryScheme, StandardScheme,
};

/// Segmented (bipartite) 2-ary cuckoo filter.
///
/// Partitions the table into two equal halves. The primary index `i1` is always in the
/// first half `[0, n/2)` and the alternate `i2` is always in the second half `[n/2, n)`.
/// This eliminates cross-half interference and typically achieves higher load factors than
/// the standard variant at the same parameters.
///
/// # Constraints
///
/// - `n` must be a power of 2 and ≥ 2.
/// - `b` (tags per bucket) must be ≥ 1.
/// - `fingerprint_bits` must satisfy `2^fp > 2b` (FPR < 1); minimum is `⌊log2(2b)⌋ + 1`.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::SegmentedCuckooFilter;
///
/// let mut f = SegmentedCuckooFilter::new(128, 4, 12).unwrap();
/// f.add("hello").unwrap();
/// assert!(f.contain("hello"));
/// ```
pub type SegmentedCuckooFilter = CuckooFilter<SegmentedScheme>;

/// Original (standard) 2-ary cuckoo filter.
///
/// The classic partial-key cuckoo filter design. Both candidate indices live anywhere
/// in `[0, n)`. Simple and cache-friendly for small tables.
///
/// # Constraints
///
/// - `n` must be a power of 2 and ≥ 1.
/// - `b` (tags per bucket) must be ≥ 1.
/// - `fingerprint_bits` must satisfy `2^fp > 2b`.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::StandardCuckooFilter;
///
/// let mut f = StandardCuckooFilter::from_num_items(50_000, 4, 12).unwrap();
/// f.add(42u64.to_le_bytes()).unwrap();
/// assert!(f.contain(42u64.to_le_bytes()));
/// ```
pub type StandardCuckooFilter = CuckooFilter<StandardScheme>;

/// Standard 3-ary cuckoo filter (three candidate buckets, all in `[0, n)`).
///
/// Each item has three candidate buckets linked by a xor3 chain: `i2 = xor3(i1, h)`,
/// `i3 = xor3(i2, h)`. Higher arity typically increases achievable load factor.
/// No per-slot position storage needed: `all_indices` cycles from `cur_index` using xor3.
///
/// # Constraints
///
/// - `n` must be a power of 3 (3^k) and ≥ 1.
/// - `b` must be ≥ 1.
/// - `fingerprint_bits` must satisfy `2^fp > 3b`.
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
pub type Standard3aryCuckooFilter = CuckooFilter<Standard3aryScheme>;

/// Segmented 3-ary cuckoo filter (three candidate buckets, one per segment).
///
/// Divides the table into three equal segments. `i_j ∈ [j·(n/3), (j+1)·(n/3))` for
/// j = 0, 1, 2. Chain position is derived from the segment number, so **no extra position
/// storage is needed** — a key advantage over [`Standard3aryCuckooFilter`].
///
/// # Constraints
///
/// - `n` must equal `3 · 2^m` for some m ≥ 0 (`n/3` must be a power of 2).
/// - `b` must be ≥ 1.
/// - `fingerprint_bits` must satisfy `2^fp > 3b`.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::Segmented3aryCuckooFilter;
///
/// // n = 3 * 32 = 96
/// let mut f = Segmented3aryCuckooFilter::new(96, 4, 12).unwrap();
/// f.add("data").unwrap();
/// assert!(f.contain("data"));
/// ```
pub type Segmented3aryCuckooFilter = CuckooFilter<Segmented3aryScheme>;

/// Standard 4-ary cuckoo filter (four candidate buckets, all in `[0, n)`).
///
/// Extends the xor4 chain to four indices: `i2 = xor4(i1, h)`, `i3 = xor4(i2, h)`,
/// `i4 = xor4(i3, h)`. The widest standard variant; highest potential load factor.
/// No per-slot position storage needed: `all_indices` cycles from `cur_index` using xor4.
///
/// # Constraints
///
/// - `n` must be a power of 4 (4^k) and ≥ 1.
/// - `b` must be ≥ 1.
/// - `fingerprint_bits` must satisfy `2^fp > 4b`.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::Standard4aryCuckooFilter;
///
/// let mut f = Standard4aryCuckooFilter::new(256, 4, 12).unwrap();
/// f.add("data").unwrap();
/// assert!(f.contain("data"));
/// ```
pub type Standard4aryCuckooFilter = CuckooFilter<Standard4aryScheme>;

/// Segmented 4-ary cuckoo filter (four candidate buckets, one per segment).
///
/// Divides the table into four equal segments. `i_j ∈ [j·(n/4), (j+1)·(n/4))` for
/// j = 0..3. Like [`Segmented3aryCuckooFilter`], position is derived from the segment
/// number, so **no extra position storage is needed**.
///
/// # Constraints
///
/// - `n` must be a power of 2 and ≥ 4 (ensures each segment is also a power of 2).
/// - `b` must be ≥ 1.
/// - `fingerprint_bits` must satisfy `2^fp > 4b`.
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
/// ```
pub type Segmented4aryCuckooFilter = CuckooFilter<Segmented4aryScheme>;
