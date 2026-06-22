#![warn(missing_docs)]
// Nursery-lint exceptions (intentional patterns; see the workspace Cargo.toml note):
#![allow(clippy::while_float)] // load-factor grow-loops compare an f64 ratio against an f64 target
#![allow(clippy::option_if_let_else)] // the explicit `mutation_log` match reads clearer than map_or_else
//! Cuckoo filter variants across two dimensions: indexing strategy and arity.
//!
//! # Design overview
//!
//! A *cuckoo filter* is a space-efficient probabilistic membership structure. It stores
//! fingerprints (short hashes) of inserted items and supports insert, lookup, and delete
//! with a bounded false-positive rate. Unlike Bloom filters, deletes are exact.
//!
//! This crate explores the segmented variant of cuckoo filters, which partitions the
//! table into equal segments and confines each candidate index to its own segment.
//! This design can achieve higher load factors and support one-round keyword PIR.
//!
//! # Notation
//!
//! In this crate, "2-ary", "3-ary", and "4-ary" refer to the number of candidate buckets per item.
//! "bucket_size" refers to the number of fingerprint slots per bucket, and "num_buckets" is the total number of buckets.
//! We support 3 arities (2, 3, 4) and 4 bucket sizes (1, 2, 3, 4) for both standard and segmented
//! indexing schemes. The letter `t` is used throughout these docs as the exponent variable when
//! stating power constraints on `num_buckets` — e.g. `2^t`, `3^t`, or `3 · 2^t`.
//!
//! # Design note — rollback vs. victim cache
//!
//! When a cuckoo insertion exhausts its kick budget, the original cuckoo filter of Fan et al. 2014
//! ([efficient/cuckoofilter](https://github.com/efficient/cuckoofilter)) stores the last evicted
//! fingerprint in a *victim cache* and only reports failure when that cache is already occupied;
//! lookups must also probe the victim cache, so an item living there still counts as present.
//!
//! **This crate deliberately does not use a victim cache.** When the kick budget is exhausted we
//! roll back every mutation made during the failing insert, leave the table in its pre-insert
//! state, and return [`CuckooError::TableFull`]. The motivation is keyword PIR: the filter is
//! materialised as a matrix and served obliviously, so a victim cache would have to be encoded as
//! an extra row/column. That inflates both implementation complexity and the PIR database size
//! for negligible gain at the load factors we target. See the filter module for the mechanics.
//!
//! # Module structure
//!
//! ```text
//! lib.rs                     — public API, type aliases
//! filter.rs                  — CuckooFilter<S> generic implementation
//! scheme.rs                  — IndexScheme trait + 6 scheme structs
//! hash.rs                    — fingerprint hash functions, item hashing, index reconstruction
//! fingerprint_table.rs       — FingerprintTable: bit-packed Vec<u8> fingerprint storage for the filter
//! fingerprint_value_table.rs — FingerprintValueTable: Vec<u32> cell-based storage for the KV store
//! store.rs                   — CuckooKVStore<S> generic implementation (insert/get/delete/update with kicking + rollback)
//! util.rs                    — helper compute next power of 2/3/4
//! ```
//!
//! # Security considerations
//!
//! - The underlying item hash (xxHash3) is **not cryptographic**. Do not use this filter
//!   for security-sensitive membership checks (e.g., password lookups, CSRF token sets).
//!   An adversary who knows the hash function can craft inputs that all map to the same
//!   buckets, causing artificially high false-positive rates or denial-of-service via
//!   table-full conditions.
//! - False positives are bounded by `d·bucket_size / 2^fingerprint_bits` where `d` is the arity. With
//!   `fingerprint_bits = 12`, `bucket_size = 4`, `d = 2` the theoretical FPR is `≈ 0.2%`. This is a
//!   *probabilistic* guarantee, not a cryptographic one.
//! - Deletion of an item never inserted may silently remove a fingerprint that belongs to
//!   a different item sharing the same fingerprint and candidate indices. Only delete items
//!   you have explicitly inserted.
//!
//! # Quick start
//!
//! ```rust
//! use segmented_cuckoo::Segmented2aryCuckooFilter;
//!
//! // Create a filter: num_buckets=64 buckets, bucket_size=4 slots/bucket, 12-bit fingerprints.
//! let mut filter = Segmented2aryCuckooFilter::new(64, 4, 12).unwrap();
//!
//! filter.insert("hello").unwrap();
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
//! use segmented_cuckoo::Standard2aryCuckooFilter;
//!
//! let mut filter = Standard2aryCuckooFilter::from_num_items(100_000, 4, 12).unwrap();
//! filter.insert(b"item".as_ref()).unwrap();
//! assert!(filter.contain(b"item".as_ref()));
//! ```

pub(crate) mod filter;
pub(crate) mod fingerprint_table;
pub(crate) mod fingerprint_value_table;
pub(crate) mod hash;
pub(crate) mod scheme;
pub(crate) mod store;
pub(crate) mod util;

pub use filter::{
    CuckooError, CuckooFilter, MAX_LOAD_FACTOR, SUPPORTED_ARITIES, SUPPORTED_BUCKET_SIZES,
};
pub use scheme::{
    IndexScheme, SchemeKind, SchemeMeta, Segmented2aryScheme, Segmented3aryScheme,
    Segmented4aryScheme, Standard2aryScheme, Standard3aryScheme, Standard4aryScheme,
};
pub use store::{
    pack_slot_cells, unpack_slot_cells, CuckooKVStore, CuckooParams, OccupiedSlot, SlotMutation,
};

// ════════════════════════════════════════════════════════════════════════════
// ░░ SEGMENTED SCHEMES ░░
// ════════════════════════════════════════════════════════════════════════════

/// Segmented 2-ary cuckoo filter.
///
/// Partitions the table into two equal halves. The primary index `i1` is always in the
/// first half `[0, num_buckets/2)` and the alternate `i2` is always in the second half `[num_buckets/2, num_buckets)`.
/// This eliminates cross-half interference and typically achieves higher load factors than
/// the standard variant at the same parameters.
///
/// # Constraints
///
/// - `num_buckets` must be a power of 2 and ≥ 2.
/// - `bucket_size` (fingerprints per bucket) must be in `1..=4`.
/// - `fingerprint_bits` must be in `[⌊log2(2 * bucket_size)⌋+1, 32]`.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo::Segmented2aryCuckooFilter;
///
/// let mut f = Segmented2aryCuckooFilter::new(128, 4, 12).unwrap();
/// f.insert("hello").unwrap();
/// assert!(f.contain("hello"));
/// ```
pub type Segmented2aryCuckooFilter = CuckooFilter<Segmented2aryScheme>;

/// Segmented 3-ary cuckoo filter (three candidate buckets, one per segment).
///
/// Divides the table into three equal segments. `i_j ∈ [j·(num_buckets/3), (j+1)·(num_buckets/3))` for
/// j = 0, 1, 2. Chain position is derived from the segment number, so **no extra position
/// storage is needed** — a key advantage over [`Standard3aryCuckooFilter`].
///
/// # Constraints
///
/// - `num_buckets` must equal `3 · 2^t` for some `t ≥ 0` (`num_buckets/3` must be a power of 2).
/// - `bucket_size` must be in `1..=4`.
/// - `fingerprint_bits` must be in `[⌊log2(3 * bucket_size)⌋+1, 32]`.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo::Segmented3aryCuckooFilter;
///
/// // num_buckets = 3 * 32 = 96
/// let mut f = Segmented3aryCuckooFilter::new(96, 4, 12).unwrap();
/// f.insert("data").unwrap();
/// assert!(f.contain("data"));
/// ```
pub type Segmented3aryCuckooFilter = CuckooFilter<Segmented3aryScheme>;

/// Segmented 4-ary cuckoo filter (four candidate buckets, one per segment).
///
/// Divides the table into four equal segments. `i_j ∈ [j·(num_buckets/4), (j+1)·(num_buckets/4))` for
/// j = 0..3. Like [`Segmented3aryCuckooFilter`], position is derived from the segment
/// number, so **no extra position storage is needed**.
///
/// # Constraints
///
/// - `num_buckets` must be a power of 2 and ≥ 4 (ensures each segment is also a power of 2).
/// - `bucket_size` must be in `1..=4`.
/// - `fingerprint_bits` must be in `[⌊log2(4 * bucket_size)⌋+1, 32]`.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo::Segmented4aryCuckooFilter;
///
/// // num_buckets = 64, so each of 4 segments has 16 buckets
/// let mut f = Segmented4aryCuckooFilter::new(64, 4, 12).unwrap();
/// f.insert("data").unwrap();
/// assert!(f.contain("data"));
/// ```
pub type Segmented4aryCuckooFilter = CuckooFilter<Segmented4aryScheme>;

// ════════════════════════════════════════════════════════════════════════════
// ░░ STANDARD SCHEMES ░░
// ════════════════════════════════════════════════════════════════════════════

/// Standard (original) 2-ary cuckoo filter.
///
/// Use *partial-key cuckoo hashing* technique. Both candidate indices live anywhere
/// in `[0, num_buckets)`. Simple and cache-friendly for small tables.
///
/// # Constraints
///
/// - `num_buckets` must be a power of 2 and ≥ 1.
/// - `bucket_size` (fingerprints per bucket) must be in `1..=4`.
/// - `fingerprint_bits` must be in `[⌊log2(2 * bucket_size)⌋+1, 32]`.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo::Standard2aryCuckooFilter;
///
/// let mut f = Standard2aryCuckooFilter::from_num_items(50_000, 4, 12).unwrap();
/// f.insert(42u64.to_le_bytes()).unwrap();
/// assert!(f.contain(42u64.to_le_bytes()));
/// ```
pub type Standard2aryCuckooFilter = CuckooFilter<Standard2aryScheme>;

/// Standard 3-ary cuckoo filter (three candidate buckets, all in `[0, num_buckets)`).
///
/// Each item has three candidate buckets linked by a xor3 chain: `i2 = xor3(i1, h)`,
/// `i3 = xor3(i2, h)`. Higher arity typically increases achievable load factor.
/// No per-slot position storage needed: `all_indices` cycles from `cur_index` using xor3.
///
/// # Constraints
///
/// - `num_buckets` must be a power of 3 (`3^t`) and ≥ 1.
/// - `bucket_size` must be in `1..=4`.
/// - `fingerprint_bits` must be in `[⌊log2(3 * bucket_size)⌋+1, 32]`.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo::Standard3aryCuckooFilter;
///
/// let mut f = Standard3aryCuckooFilter::new(243, 4, 12).unwrap();
/// f.insert("data").unwrap();
/// assert!(f.contain("data"));
/// ```
pub type Standard3aryCuckooFilter = CuckooFilter<Standard3aryScheme>;

/// Standard 4-ary cuckoo filter (four candidate buckets, all in `[0, num_buckets)`).
///
/// Extends the xor4 chain to four indices: `i2 = xor4(i1, h)`, `i3 = xor4(i2, h)`,
/// `i4 = xor4(i3, h)`. The widest standard variant; highest potential load factor.
/// No per-slot position storage needed: `all_indices` cycles from `cur_index` using xor4.
///
/// # Constraints
///
/// - `num_buckets` must be a power of 4 (`4^t`) and ≥ 1.
/// - `bucket_size` must be in `1..=4`.
/// - `fingerprint_bits` must be in `[⌊log2(4 * bucket_size)⌋+1, 32]`.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo::Standard4aryCuckooFilter;
///
/// let mut f = Standard4aryCuckooFilter::new(256, 4, 12).unwrap();
/// f.insert("data").unwrap();
/// assert!(f.contain("data"));
/// ```
pub type Standard4aryCuckooFilter = CuckooFilter<Standard4aryScheme>;

// ════════════════════════════════════════════════════════════════════════════
// ░░ SEGMENTED KV STORES ░░
// ════════════════════════════════════════════════════════════════════════════

/// Segmented 2-ary cuckoo key-value store.
///
/// Each slot stores `fingerprint ‖ value`; the segmented layout confines each candidate to
/// its own half of the table, giving every key a deterministic, fixed lookup-position set.
/// This is the property the IKPIR scheme relies on to retrieve a key with a single Index-PIR
/// query.
///
/// # Constraints
///
/// - `num_buckets` must be a power of 2 and ≥ 2.
/// - `bucket_size` must be in `1..=4`.
/// - `fingerprint_bits` must be in `[⌊log2(2 · bucket_size)⌋+1, 32]`.
/// - `value_bits` must be ≥ 1 (otherwise the store degenerates to a filter — use
///   [`Segmented2aryCuckooFilter`] instead).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo::Segmented2aryCuckooKVStore;
///
/// let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
/// store.insert("hello", &[0xAB]).unwrap();
/// assert_eq!(store.get("hello"), Some(vec![0xAB]));
///
/// // Zero-allocation read path: caller supplies a `value_size_in_bytes`-sized buffer.
/// let mut buf = vec![0u8; store.value_size_in_bytes()];
/// assert!(store.get_into("hello", &mut buf));
/// assert_eq!(buf, vec![0xAB]);
/// ```
pub type Segmented2aryCuckooKVStore = CuckooKVStore<Segmented2aryScheme>;

/// Segmented 3-ary cuckoo key-value store (three candidate buckets, one per segment).
///
/// Three equal segments; chain position is derived from segment number, so no per-slot
/// position storage is needed.
///
/// # Constraints
///
/// - `num_buckets` must equal `3 · 2^t` for some `t ≥ 0`.
/// - `bucket_size` must be in `1..=4`.
/// - `fingerprint_bits` must be in `[⌊log2(3 · bucket_size)⌋+1, 32]`.
/// - `value_bits` must be ≥ 1.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo::Segmented3aryCuckooKVStore;
///
/// // num_buckets = 3 * 32 = 96
/// let mut store = Segmented3aryCuckooKVStore::new(96, 4, 12, 8, 8).unwrap();
/// store.insert("data", &[0x42]).unwrap();
/// assert_eq!(store.get("data"), Some(vec![0x42]));
/// ```
pub type Segmented3aryCuckooKVStore = CuckooKVStore<Segmented3aryScheme>;

/// Segmented 4-ary cuckoo key-value store (four candidate buckets, one per segment).
///
/// Four equal segments; like the 3-ary variant, chain position is derived from segment
/// number.
///
/// # Constraints
///
/// - `num_buckets` must be a power of 2 and ≥ 4 (so each of the four segments is a power of 2).
/// - `bucket_size` must be in `1..=4`.
/// - `fingerprint_bits` must be in `[⌊log2(4 · bucket_size)⌋+1, 32]`.
/// - `value_bits` must be ≥ 1.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo::Segmented4aryCuckooKVStore;
///
/// let mut store = Segmented4aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
/// store.insert("data", &[0x42]).unwrap();
/// assert_eq!(store.get("data"), Some(vec![0x42]));
/// ```
pub type Segmented4aryCuckooKVStore = CuckooKVStore<Segmented4aryScheme>;
