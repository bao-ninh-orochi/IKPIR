//! Index scheme abstractions and their six concrete implementations.
//!
//! # Purpose
//!
//! The [`IndexScheme`] trait is the *sole* point of variation between all six filter
//! variants. Every `CuckooFilter<S>` uses identical insert/lookup/delete mechanics; only
//! the [`IndexScheme`] implementation changes how candidate bucket indices are computed.
//!
//! # Standard vs. Segmented
//!
//! | Scheme family | Index range           | `n` constraint              | Position storage  |
//! |---------------|-----------------------|-----------------------------|-------------------|
//! | Standard 2-ary  | Both in `[0, n)`    | Power of 2                  | None              |
//! | Segmented 2-ary | `i1 ∈ [0, n/2)`, `i2 ∈ [n/2, n)` | Power of 2 ≥ 2 | None |
//! | Standard 3-ary  | All in `[0, n)`     | Power of 3 (3^k)            | None              |
//! | Segmented 3-ary | `i_j ∈ [j·seg, (j+1)·seg)` | `n = 3·2^m`      | None              |
//! | Standard 4-ary  | All in `[0, n)`     | Power of 4 (4^k)            | None              |
//! | Segmented 4-ary | `i_j ∈ [j·seg, (j+1)·seg)` | Power of 2 ≥ 4   | None              |
//!
//! Standard k > 2 schemes use xor3/xor4 cycling so `all_indices` always starts from
//! `cur_index`. No per-slot position storage is needed for any scheme.
//!
//! # Security
//!
//! No secret data flows through this module. Index computations are simple arithmetic;
//! there are no data-dependent memory accesses that could leak timing information.

use crate::hash;

/// Abstraction over the index-computation strategy for a cuckoo filter.
///
/// All six concrete filter types are generic over this trait. The trait decouples index
/// computation from filter mechanics, enabling a single `CuckooFilter<S>` implementation
/// to cover every combination of arity and layout.
///
/// # Implementing this trait
///
/// All methods must be consistent with each other:
/// - `hash_item` and `all_indices` must be inverse: for any `(tag, indices)` returned by
///   `hash_item`, `all_indices(indices[p], tag, p)` must return the same set of indices.
/// - `position_of` must agree with `all_indices`: the position returned for an index must
///   be the one that `all_indices` expects to recover the full chain.
pub trait IndexScheme {
    /// Return the number of candidate bucket indices per item (2, 3, or 4).
    ///
    /// # Returns
    ///
    /// `2`, `3`, or `4`. The first `arity()` elements of any `indices` array returned by
    /// [`hash_item`](Self::hash_item) or [`all_indices`](Self::all_indices) are valid;
    /// the remaining elements are `0` (padding).
    fn arity(&self) -> usize;

    /// Hash an item to produce a fingerprint and k candidate bucket indices.
    ///
    /// This is the primary entry point for the filter's insert, lookup, and delete paths.
    /// The fingerprint `tag` is derived from the lower 32 bits of the xxh3 hash; the
    /// indices are derived from the upper 32 bits plus XOR chaining with tag-hash offsets.
    ///
    /// # Arguments
    ///
    /// - `item` — arbitrary byte slice representing the item to hash.
    /// - `bits_per_tag` — fingerprint bit width (1–32); must match the filter's `fp_bits`.
    ///
    /// # Returns
    ///
    /// `(tag, indices)` where:
    /// - `tag` is a non-zero fingerprint in `[1, 2^bits_per_tag]`.
    /// - `indices[0..arity()]` are the valid candidate bucket indices.
    /// - `indices[arity()..]` are `0` (unused padding).
    fn hash_item(&self, item: &[u8], bits_per_tag: u32) -> (u32, [u32; 4]);

    /// Reconstruct all k candidate indices from one known index, its fingerprint, and its
    /// chain position.
    ///
    /// Called during cuckoo kicking: after evicting a tag from a bucket, the filter needs
    /// to know all other buckets where it could be placed. Given one bucket index and the
    /// stored position, this function regenerates the full candidate set.
    ///
    /// # Arguments
    ///
    /// - `cur_index` — the bucket index currently holding (or being evicted from) the tag.
    /// - `tag` — the fingerprint stored at `cur_index`.
    /// - `position` — the 0-indexed chain position of `cur_index`:
    ///   `0` = primary index (`i1`), `1` = first alternate (`i2`), etc.
    ///   For segmented schemes this argument is ignored (position is derived from the index).
    ///
    /// # Returns
    ///
    /// `[i1, i2, …, 0, 0]` — the full candidate array in canonical chain order.
    /// `result[0..arity()]` are valid; remainder is `0`.
    fn all_indices(&self, cur_index: u32, tag: u32, position: u8) -> [u32; 4];

    /// Derive the chain position from a bucket index without extra storage.
    ///
    /// For **segmented** schemes the segment number encodes the position:
    /// `position = index / segment_size`. For **standard 2-ary** the XOR symmetry means
    /// position is irrelevant (returns `0`). For **standard k > 2** schemes using xor3/xor4
    /// cycling, `all_indices` always starts from `cur_index` at position 0, so this returns 0.
    ///
    /// # Arguments
    ///
    /// - `index` — a valid bucket index for this scheme.
    ///
    /// # Returns
    ///
    /// The 0-indexed chain position (`0` = primary, `1` = first alternate, …).
    fn position_of(&self, index: u32) -> u8;
}

// ─── 2-ary schemes ──────────────────────────────────────────────────────────

/// Original (standard) 2-ary cuckoo filter scheme.
///
/// Both candidate indices live anywhere in `[0, num_buckets)`. The XOR relationship
/// `i2 = i1 ^ h1(tag)` is self-inverse, so no per-slot position storage is needed: given
/// `(index, tag)` the alternate is always `index ^ h1(tag)`.
///
/// # Field
///
/// - `num_buckets` — total bucket count; must be a power of 2 and ≥ 1.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::scheme::{StandardScheme, IndexScheme};
///
/// let scheme = StandardScheme { num_buckets: 64 };
/// let (tag, idx) = scheme.hash_item(b"hello", 12);
/// assert_ne!(tag, 0);
/// assert!(idx[0] < 64 && idx[1] < 64);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct StandardScheme {
    /// Total bucket count; must be a power of 2 and ≥ 1.
    pub num_buckets: u32,
}

impl IndexScheme for StandardScheme {
    fn arity(&self) -> usize {
        2
    }
    fn hash_item(&self, item: &[u8], bits_per_tag: u32) -> (u32, [u32; 4]) {
        hash::hash_item_standard(item, self.num_buckets, bits_per_tag)
    }
    fn all_indices(&self, cur_index: u32, tag: u32, position: u8) -> [u32; 4] {
        hash::all_indices_standard(cur_index, tag, position, self.num_buckets)
    }
    fn position_of(&self, _index: u32) -> u8 {
        0 // irrelevant for 2-ary XOR
    }
}

/// Bipartite segmented 2-ary scheme.
///
/// The table is split into two equal halves. `i1` is always in `[0, half)` and `i2` is
/// always in `[half, 2·half)`. This confines each candidate to its own segment, preventing
/// cross-half interference. Position is determined by which half an index falls in, so no
/// per-slot position storage is required.
///
/// # Field
///
/// - `half` — segment size (`n / 2`); must be a power of 2 and ≥ 1 (so `n ≥ 2`).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::scheme::{SegmentedScheme, IndexScheme};
///
/// let scheme = SegmentedScheme { half: 32 };
/// let (tag, idx) = scheme.hash_item(b"hello", 12);
/// assert!(idx[0] < 32);
/// assert!(idx[1] >= 32 && idx[1] < 64);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SegmentedScheme {
    /// Segment size (`n / 2`); must be a power of 2 and ≥ 1 so `n ≥ 2`.
    pub half: u32,
}

impl IndexScheme for SegmentedScheme {
    fn arity(&self) -> usize {
        2
    }
    fn hash_item(&self, item: &[u8], bits_per_tag: u32) -> (u32, [u32; 4]) {
        hash::hash_item_segmented(item, self.half, bits_per_tag)
    }
    fn all_indices(&self, cur_index: u32, tag: u32, _position: u8) -> [u32; 4] {
        hash::all_indices_segmented(cur_index, tag, self.half)
    }
    fn position_of(&self, index: u32) -> u8 {
        if index < self.half {
            0
        } else {
            1
        }
    }
}

// ─── 3-ary schemes ──────────────────────────────────────────────────────────

/// Standard 3-ary scheme: all three candidate indices in `[0, num_buckets)`.
///
/// xor3 chain: `i2 = xor3(i1, h)`, `i3 = xor3(i2, h)` where `h = tag_hash_mod(tag, n)`.
/// All three indices share the same range. No per-slot position storage is needed because
/// `all_indices` always cycles from `cur_index` using xor3.
///
/// # Field
///
/// - `num_buckets` — total bucket count; must be a power of 3 (3^k) and ≥ 1.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::scheme::{Standard3aryScheme, IndexScheme};
///
/// let scheme = Standard3aryScheme { num_buckets: 243 };
/// let (tag, idx) = scheme.hash_item(b"hello", 12);
/// assert!(idx[0] < 243 && idx[1] < 243 && idx[2] < 243);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Standard3aryScheme {
    /// Total bucket count; must be a power of 3 (3^k) and ≥ 1.
    pub num_buckets: u32,
}

impl IndexScheme for Standard3aryScheme {
    fn arity(&self) -> usize {
        3
    }
    fn hash_item(&self, item: &[u8], bits_per_tag: u32) -> (u32, [u32; 4]) {
        hash::hash_item_standard_3ary(item, self.num_buckets, bits_per_tag)
    }
    fn all_indices(&self, cur_index: u32, tag: u32, position: u8) -> [u32; 4] {
        hash::all_indices_standard_3ary(cur_index, tag, position, self.num_buckets)
    }
    /// With xor3 cycling, `all_indices` always starts from `cur_index` at position 0.
    fn position_of(&self, _index: u32) -> u8 {
        0
    }
}

/// Segmented 3-ary scheme: `i_j ∈ [j·seg, (j+1)·seg)` for j = 0, 1, 2.
///
/// The table is divided into three equal segments. `n = 3 · seg` where `seg` must be a
/// power of 2. Chain position is derived from `index / seg` — no per-slot storage needed.
/// This is a key advantage over [`Standard3aryScheme`] at identical load capacity.
///
/// # Field
///
/// - `segment_size` — size of each segment (`n / 3`); must be a power of 2.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::scheme::{Segmented3aryScheme, IndexScheme};
///
/// let scheme = Segmented3aryScheme { segment_size: 32 }; // n = 96
/// let (_, idx) = scheme.hash_item(b"hello", 12);
/// assert!(idx[0] < 32);
/// assert!(idx[1] >= 32 && idx[1] < 64);
/// assert!(idx[2] >= 64 && idx[2] < 96);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Segmented3aryScheme {
    /// Size of each segment (`n / 3`); must be a power of 2.
    pub segment_size: u32,
}

impl IndexScheme for Segmented3aryScheme {
    fn arity(&self) -> usize {
        3
    }
    fn hash_item(&self, item: &[u8], bits_per_tag: u32) -> (u32, [u32; 4]) {
        hash::hash_item_segmented_3ary(item, self.segment_size, bits_per_tag)
    }
    fn all_indices(&self, cur_index: u32, tag: u32, _position: u8) -> [u32; 4] {
        hash::all_indices_segmented_3ary(cur_index, tag, self.segment_size)
    }
    fn position_of(&self, index: u32) -> u8 {
        (index / self.segment_size) as u8
    }
}

// ─── 4-ary schemes ──────────────────────────────────────────────────────────

/// Standard 4-ary scheme: all four candidate indices in `[0, num_buckets)`.
///
/// xor4 chain: `i2 = xor4(i1, h)`, `i3 = xor4(i2, h)`, `i4 = xor4(i3, h)` where
/// `h = tag_hash1(tag, n)`. The widest standard variant. No per-slot storage is needed
/// because `all_indices` always cycles from `cur_index` using xor4.
///
/// # Field
///
/// - `num_buckets` — total bucket count; must be a power of 4 (4^k) and ≥ 1.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::scheme::{Standard4aryScheme, IndexScheme};
///
/// let scheme = Standard4aryScheme { num_buckets: 256 };
/// let (_, idx) = scheme.hash_item(b"hello", 12);
/// assert!(idx.iter().all(|&i| i < 256));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Standard4aryScheme {
    /// Total bucket count; must be a power of 4 (4^k) and ≥ 1.
    pub num_buckets: u32,
}

impl IndexScheme for Standard4aryScheme {
    fn arity(&self) -> usize {
        4
    }
    fn hash_item(&self, item: &[u8], bits_per_tag: u32) -> (u32, [u32; 4]) {
        hash::hash_item_standard_4ary(item, self.num_buckets, bits_per_tag)
    }
    fn all_indices(&self, cur_index: u32, tag: u32, position: u8) -> [u32; 4] {
        hash::all_indices_standard_4ary(cur_index, tag, position, self.num_buckets)
    }
    /// With xor4 cycling, `all_indices` always starts from `cur_index` at position 0.
    fn position_of(&self, _index: u32) -> u8 {
        0
    }
}

/// Segmented 4-ary scheme: `i_j ∈ [j·seg, (j+1)·seg)` for j = 0, 1, 2, 3.
///
/// The table is divided into four equal segments. `n = 4 · seg` where `seg` must be a
/// power of 2 (equivalently, `n` is a power of 2 and ≥ 4). Position is derived from
/// `index / seg`, so **no per-slot position storage is needed**.
///
/// # Field
///
/// - `segment_size` — size of each segment (`n / 4`); must be a power of 2.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::scheme::{Segmented4aryScheme, IndexScheme};
///
/// let scheme = Segmented4aryScheme { segment_size: 16 }; // n = 64
/// let (_, idx) = scheme.hash_item(b"hello", 12);
/// assert!(idx[0] < 16);
/// assert!(idx[1] >= 16 && idx[1] < 32);
/// assert!(idx[2] >= 32 && idx[2] < 48);
/// assert!(idx[3] >= 48 && idx[3] < 64);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Segmented4aryScheme {
    /// Size of each segment (`n / 4`); must be a power of 2.
    pub segment_size: u32,
}

impl IndexScheme for Segmented4aryScheme {
    fn arity(&self) -> usize {
        4
    }
    fn hash_item(&self, item: &[u8], bits_per_tag: u32) -> (u32, [u32; 4]) {
        hash::hash_item_segmented_4ary(item, self.segment_size, bits_per_tag)
    }
    fn all_indices(&self, cur_index: u32, tag: u32, _position: u8) -> [u32; 4] {
        hash::all_indices_segmented_4ary(cur_index, tag, self.segment_size)
    }
    fn position_of(&self, index: u32) -> u8 {
        (index / self.segment_size) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── 2-ary ──

    #[test]
    fn segmented_fingerprint_non_zero() {
        let scheme = SegmentedScheme { half: 64 };
        for i in 0u32..1000 {
            let (tag, _) = scheme.hash_item(&i.to_le_bytes(), 12);
            assert_ne!(tag, 0);
        }
    }

    #[test]
    fn segmented_index_ranges() {
        let half = 128u32;
        let scheme = SegmentedScheme { half };
        for i in 0u32..1000 {
            let (_, idx) = scheme.hash_item(&i.to_le_bytes(), 12);
            assert!(idx[0] < half);
            assert!(idx[1] >= half && idx[1] < half * 2);
        }
    }

    #[test]
    fn segmented_all_indices_round_trip() {
        let half = 256u32;
        let scheme = SegmentedScheme { half };
        for i in 0u32..1000 {
            let (tag, idx) = scheme.hash_item(&i.to_le_bytes(), 16);
            for p in 0..2u8 {
                let a = scheme.all_indices(idx[p as usize], tag, p);
                assert_eq!(a[0], idx[0]);
                assert_eq!(a[1], idx[1]);
            }
        }
    }

    #[test]
    fn standard_fingerprint_non_zero() {
        let scheme = StandardScheme { num_buckets: 128 };
        for i in 0u32..1000 {
            let (tag, _) = scheme.hash_item(&i.to_le_bytes(), 12);
            assert_ne!(tag, 0);
        }
    }

    #[test]
    fn standard_all_indices_round_trip() {
        let n = 256u32;
        let scheme = StandardScheme { num_buckets: n };
        for i in 0u32..1000 {
            let (tag, idx) = scheme.hash_item(&i.to_le_bytes(), 16);
            for p in 0..2u8 {
                let a = scheme.all_indices(idx[p as usize], tag, p);
                assert_eq!(a[0], idx[0]);
                assert_eq!(a[1], idx[1]);
            }
        }
    }

    // ── 3-ary ──

    #[test]
    fn standard_3ary_round_trip() {
        let n = 243u32;
        let scheme = Standard3aryScheme { num_buckets: n };
        for i in 0u32..1000 {
            let (tag, idx) = scheme.hash_item(&i.to_le_bytes(), 16);
            for p in 0..3u8 {
                let a = scheme.all_indices(idx[p as usize], tag, p);
                let mut expected = [idx[0], idx[1], idx[2]];
                expected.sort();
                let mut got = [a[0], a[1], a[2]];
                got.sort();
                assert_eq!(got, expected, "i={} p={}", i, p);
            }
        }
    }

    #[test]
    fn standard_3ary_position_of_always_zero() {
        let scheme = Standard3aryScheme { num_buckets: 243 };
        assert_eq!(scheme.position_of(0), 0);
        assert_eq!(scheme.position_of(100), 0);
        assert_eq!(scheme.position_of(242), 0);
    }

    #[test]
    fn segmented_3ary_round_trip() {
        let seg = 128u32;
        let scheme = Segmented3aryScheme { segment_size: seg };
        for i in 0u32..1000 {
            let (tag, idx) = scheme.hash_item(&i.to_le_bytes(), 16);
            assert!(idx[0] < seg);
            assert!(idx[1] >= seg && idx[1] < 2 * seg);
            assert!(idx[2] >= 2 * seg && idx[2] < 3 * seg);
            for p in 0..3 {
                let a = scheme.all_indices(idx[p], tag, p as u8);
                assert_eq!(
                    [a[0], a[1], a[2]],
                    [idx[0], idx[1], idx[2]],
                    "i={} p={}",
                    i,
                    p
                );
            }
        }
    }

    // ── 4-ary ──

    #[test]
    fn standard_4ary_round_trip() {
        let n = 256u32;
        let scheme = Standard4aryScheme { num_buckets: n };
        for i in 0u32..1000 {
            let (tag, idx) = scheme.hash_item(&i.to_le_bytes(), 16);
            for p in 0..4u8 {
                let a = scheme.all_indices(idx[p as usize], tag, p);
                let mut expected = [idx[0], idx[1], idx[2], idx[3]];
                expected.sort();
                let mut got = [a[0], a[1], a[2], a[3]];
                got.sort();
                assert_eq!(got, expected, "i={} p={}", i, p);
            }
        }
    }

    #[test]
    fn standard_4ary_position_of_always_zero() {
        let scheme = Standard4aryScheme { num_buckets: 256 };
        assert_eq!(scheme.position_of(0), 0);
        assert_eq!(scheme.position_of(128), 0);
        assert_eq!(scheme.position_of(255), 0);
    }

    #[test]
    fn segmented_4ary_round_trip() {
        let seg = 64u32;
        let scheme = Segmented4aryScheme { segment_size: seg };
        for i in 0u32..1000 {
            let (tag, idx) = scheme.hash_item(&i.to_le_bytes(), 16);
            assert!(idx[0] < seg);
            assert!(idx[1] >= seg && idx[1] < 2 * seg);
            assert!(idx[2] >= 2 * seg && idx[2] < 3 * seg);
            assert!(idx[3] >= 3 * seg && idx[3] < 4 * seg);
            for p in 0..4u8 {
                let a = scheme.all_indices(idx[p as usize], tag, p);
                assert_eq!(a, idx, "i={} p={}", i, p);
            }
        }
    }

    // ── position_of ──

    #[test]
    fn segmented_position_of() {
        let scheme = SegmentedScheme { half: 64 };
        assert_eq!(scheme.position_of(0), 0);
        assert_eq!(scheme.position_of(63), 0);
        assert_eq!(scheme.position_of(64), 1);
        assert_eq!(scheme.position_of(127), 1);
    }

    #[test]
    fn segmented_3ary_position_of() {
        let scheme = Segmented3aryScheme { segment_size: 32 };
        assert_eq!(scheme.position_of(0), 0);
        assert_eq!(scheme.position_of(31), 0);
        assert_eq!(scheme.position_of(32), 1);
        assert_eq!(scheme.position_of(63), 1);
        assert_eq!(scheme.position_of(64), 2);
        assert_eq!(scheme.position_of(95), 2);
    }

    #[test]
    fn segmented_4ary_position_of() {
        let scheme = Segmented4aryScheme { segment_size: 16 };
        assert_eq!(scheme.position_of(0), 0);
        assert_eq!(scheme.position_of(15), 0);
        assert_eq!(scheme.position_of(16), 1);
        assert_eq!(scheme.position_of(32), 2);
        assert_eq!(scheme.position_of(48), 3);
    }

    #[test]
    fn standard_2ary_position_of_always_zero() {
        // For 2-ary standard, position_of is defined but effectively unused (XOR symmetry).
        let scheme = StandardScheme { num_buckets: 64 };
        assert_eq!(scheme.position_of(0), 0);
        assert_eq!(scheme.position_of(63), 0);
    }
}
