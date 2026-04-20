//! Hash primitives for fingerprint extraction and candidate-index computation.
//!
//! # Purpose
//!
//! This module provides three layers of hashing:
//!
//! 1. **Item hashing** — xxHash3 (64-bit, non-cryptographic) maps arbitrary byte slices
//!    to a 64-bit digest. The upper 32 bits seed the primary bucket index; the lower 32
//!    bits seed the fingerprint.
//!
//! 2. **Tag hash functions** — Three independent Murmur-constant mixers produce the XOR
//!    offsets that link candidate indices in the cuckoo chain. Each uses a distinct
//!    multiplier so their outputs are independently distributed for typical tag values.
//!
//! 3. **Index reconstruction** — The `all_indices_*` family recovers all k candidate
//!    indices from one known index, a tag, and a chain position. This is the critical
//!    path during cuckoo kicking.
//!
//! # Security
//!
//! xxHash3 is **not cryptographic**. Do not use this module for any security-sensitive
//! purpose (e.g., keyed authentication, HMAC). An adversary who can observe the hash
//! function outputs can craft items that collide on fingerprint or bucket index, leading
//! to elevated false-positive rates or targeted denial-of-service via table-full
//! conditions (hash-flooding attack). If adversarial inputs are expected, replace
//! `xxh3_64` with a keyed hash (e.g., SipHash-1-3).
//!
//! # Bit layout
//!
//! ```text
//! xxh3_64(item) → h: u64
//!   fingerprint (tag) = lower 32 bits, masked to bits_per_tag, non-zero
//!   primary index i1  = upper 32 bits, masked to (range - 1)
//! ```

use xxhash_rust::xxh3::xxh3_64;

// ── Tag hash functions ──────────────────────────────────────────────────────
// Each uses a different mixing constant to produce independent hashes.
// `range` must be a power of 2; output is in [0, range).

/// Compute the first XOR offset for a fingerprint tag.
///
/// Multiplies `tag` by the MurmurHash2 constant `0x5bd1e995` (with wrapping) and masks
/// the result to `[0, range)`. Used to derive `i2` from `i1` in all schemes.
///
/// # Arguments
///
/// - `tag` — non-zero fingerprint value.
/// - `range` — table size or segment size; **must be a power of 2**.
///
/// # Returns
///
/// An offset in `[0, range)`.
///
/// # Performance
///
/// O(1) — one multiply, one AND.
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::tag_hash1;
///
/// let offset = tag_hash1(0xABC, 256);
/// assert!(offset < 256);
/// ```
#[inline]
pub fn tag_hash1(tag: u32, range: u32) -> u32 {
    tag.wrapping_mul(0x5bd1e995) & (range - 1)
}

/// Compute the second XOR offset for a fingerprint tag.
///
/// Multiplies `tag` by the MurmurHash3 c1 constant `0xcc9e2d51` and masks to `[0, range)`.
/// Used to derive `i3` from `i2` in 3-ary and 4-ary schemes.
///
/// # Arguments
///
/// - `tag` — non-zero fingerprint value.
/// - `range` — table size or segment size; **must be a power of 2**.
///
/// # Returns
///
/// An offset in `[0, range)`, statistically independent of [`tag_hash1`] for most inputs.
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::tag_hash2;
///
/// let offset = tag_hash2(0xABC, 256);
/// assert!(offset < 256);
/// ```
#[inline]
pub fn tag_hash2(tag: u32, range: u32) -> u32 {
    tag.wrapping_mul(0xcc9e2d51) & (range - 1)
}

/// Compute the third XOR offset for a fingerprint tag.
///
/// Multiplies `tag` by the MurmurHash3 c2 constant `0x1b873593` and masks to `[0, range)`.
/// Used to derive `i4` from `i3` in 4-ary schemes.
///
/// # Arguments
///
/// - `tag` — non-zero fingerprint value.
/// - `range` — table size or segment size; **must be a power of 2**.
///
/// # Returns
///
/// An offset in `[0, range)`, statistically independent of `tag_hash1` and `tag_hash2`
/// for most inputs.
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::tag_hash3;
///
/// let offset = tag_hash3(0xABC, 256);
/// assert!(offset < 256);
/// ```
#[inline]
pub fn tag_hash3(tag: u32, range: u32) -> u32 {
    tag.wrapping_mul(0x1b873593) & (range - 1)
}

// ── Common helpers ──────────────────────────────────────────────────────────

/// Extract a non-zero fingerprint from the lower 32 bits of an xxh3 hash.
///
/// Fingerprint 0 is reserved to mean "slot empty" in [`TagTable`]. When the masked hash
/// value is 0, this function returns 1 instead, introducing a tiny bias toward tag = 1 in
/// exchange for keeping the empty-slot invariant universally valid.
///
/// # Arguments
///
/// - `h` — 64-bit xxh3 hash of the item.
/// - `bits_per_tag` — fingerprint bit width (1–32). Must be ≥ 1.
///
/// # Returns
///
/// A fingerprint in `[1, 2^bits_per_tag]` (never 0).
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::extract_tag;
///
/// // When the lower bits happen to be 0, tag is forced to 1.
/// let tag = extract_tag(0xFFFF_FFFF_0000_0000u64, 12);
/// assert_eq!(tag, 1);
///
/// // Normal case: mask to bits_per_tag.
/// let tag2 = extract_tag(0x0000_0000_0000_0FFFu64, 12);
/// assert_eq!(tag2, 0xFFF);
/// ```
///
/// [`TagTable`]: crate::bucket::TagTable
#[inline]
pub fn extract_tag(h: u64, bits_per_tag: u32) -> u32 {
    let mask = if bits_per_tag >= 32 {
        u32::MAX
    } else {
        (1u32 << bits_per_tag) - 1
    };
    let mut tag = (h as u32) & mask;
    if tag == 0 {
        tag = 1;
    }
    tag
}

/// Hash `item` with xxh3_64, extract the fingerprint, and derive the primary index `i1`.
///
/// This is the entry point for all `hash_item_*` functions. It wraps the xxh3 call and
/// the two extraction steps into a single reusable helper.
///
/// # Arguments
///
/// - `item` — arbitrary byte slice to hash.
/// - `bits_per_tag` — fingerprint bit width (1–32).
/// - `range` — table size or segment size; **must be a power of 2**. `i1` is masked to
///   `[0, range)`.
///
/// # Returns
///
/// `(h, tag, i1)` where:
/// - `h` — raw 64-bit xxh3 hash (exposed for tests / debugging).
/// - `tag` — non-zero fingerprint in `[1, 2^bits_per_tag]`.
/// - `i1` — primary bucket index in `[0, range)`.
///
/// # Performance
///
/// O(|item|) — dominated by the xxh3 call, which processes input in 256-bit SIMD blocks
/// on supported hardware (AVX2 / NEON).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::hash_and_i1;
///
/// let (_, tag, i1) = hash_and_i1(b"hello", 12, 128);
/// assert!(tag >= 1 && tag <= 0xFFF);
/// assert!(i1 < 128);
/// ```
#[inline]
pub fn hash_and_i1(item: &[u8], bits_per_tag: u32, range: u32) -> (u64, u32, u32) {
    let h = xxh3_64(item);
    let tag = extract_tag(h, bits_per_tag);
    let i1 = ((h >> 32) as u32) & (range - 1);
    (h, tag, i1)
}

// ── Standard (original) 2-ary ──────────────────────────────────────────────

/// Hash an item for the standard 2-ary scheme, yielding the fingerprint and two candidate indices.
///
/// Computes `i2 = i1 ^ tag_hash1(tag, n)`. Both indices are in `[0, num_buckets)`.
///
/// # Arguments
///
/// - `item` — byte slice to hash.
/// - `num_buckets` — total number of buckets; must be a power of 2.
/// - `bits_per_tag` — fingerprint bit width.
///
/// # Returns
///
/// `(tag, [i1, i2, 0, 0])` — `indices[0..2]` are valid; `indices[2..4]` are `0` (unused).
///
/// # Performance
///
/// O(|item|) — xxh3 dominates; the XOR step is O(1).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::hash_item_standard;
///
/// let (tag, idx) = hash_item_standard(b"hello", 128, 12);
/// assert_ne!(tag, 0);
/// assert!(idx[0] < 128);
/// assert!(idx[1] < 128);
/// ```
pub fn hash_item_standard(item: &[u8], num_buckets: u32, bits_per_tag: u32) -> (u32, [u32; 4]) {
    let (_h, tag, i1) = hash_and_i1(item, bits_per_tag, num_buckets);
    let i2 = i1 ^ tag_hash1(tag, num_buckets);
    (tag, [i1, i2, 0, 0])
}

// ── Segmented 2-ary ────────────────────────────────────────────────────────

/// Hash an item for the segmented 2-ary scheme.
///
/// `i1 ∈ [0, half)` and `i2 = half + (i1 ^ tag_hash1(tag, half))` so `i2 ∈ [half, n)`.
/// Each index is confined to its own segment, preventing cross-half bucket interference.
///
/// # Arguments
///
/// - `item` — byte slice to hash.
/// - `half` — segment size (`n / 2`); must be a power of 2.
/// - `bits_per_tag` — fingerprint bit width.
///
/// # Returns
///
/// `(tag, [i1, i2, 0, 0])` where `i1 < half` and `i2 ∈ [half, 2·half)`.
///
/// # Performance
///
/// O(|item|).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::hash_item_segmented;
///
/// let half = 64u32;
/// let (tag, idx) = hash_item_segmented(b"hello", half, 12);
/// assert!(idx[0] < half);
/// assert!(idx[1] >= half && idx[1] < 2 * half);
/// ```
pub fn hash_item_segmented(item: &[u8], half: u32, bits_per_tag: u32) -> (u32, [u32; 4]) {
    let (_h, tag, i1) = hash_and_i1(item, bits_per_tag, half);
    let i2 = half + (i1 ^ tag_hash1(tag, half));
    (tag, [i1, i2, 0, 0])
}

/// Base-3 digit-wise addition mod 3. Property: xor3(xor3(xor3(a,b),b),b) == a.
#[inline]
pub fn xor3(a: u32, b: u32) -> u32 {
    let mut result = 0u32;
    let mut aa = a;
    let mut bb = b;
    let mut place = 1u32;
    loop {
        result += ((aa % 3) + (bb % 3)) % 3 * place;
        aa /= 3;
        bb /= 3;
        if aa == 0 && bb == 0 {
            break;
        }
        // Prevent overflow: max u32 needs at most 21 base-3 digits
        if place > 1_000_000_000 {
            break;
        }
        place *= 3;
    }
    result
}

/// Base-4 digit-wise addition mod 4 using bitwise trick (O(1)).
/// Property: xor4(xor4(xor4(xor4(a,b),b),b),b) == a.
#[inline]
pub fn xor4(a: u32, b: u32) -> u32 {
    // Each base-4 digit is a 2-bit pair. Add low bits and high bits separately with carry.
    let lo_mask = 0x55555555u32; // bits 0,2,4,... (low bit of each 2-bit pair)
    let a_lo = a & lo_mask;
    let b_lo = b & lo_mask;
    let a_hi = (a >> 1) & lo_mask;
    let b_hi = (b >> 1) & lo_mask;
    let sum_lo = a_lo ^ b_lo;
    let carry = a_lo & b_lo;
    let sum_hi = a_hi ^ b_hi ^ carry;
    sum_lo | (sum_hi << 1)
}

/// Tag hash using modulo (for power-of-3 ranges where bitmask doesn't apply).
///
/// - `range` — table size; need not be a power of 2.
#[inline]
pub fn tag_hash_mod(tag: u32, range: u32) -> u32 {
    tag.wrapping_mul(0x5bd1e995) % range
}

// ── Standard 3-ary ─────────────────────────────────────────────────────────

/// Hash an item for the standard 3-ary scheme.
///
/// xor3 chain: `i2 = xor3(i1, h)`, `i3 = xor3(i2, h)` where `h = tag_hash_mod(tag, n)`.
/// All three indices are in `[0, num_buckets)`. `num_buckets` must be a power of 3.
///
/// # Arguments
///
/// - `item` — byte slice to hash.
/// - `num_buckets` — must be a power of 3.
/// - `bits_per_tag` — fingerprint bit width.
///
/// # Returns
///
/// `(tag, [i1, i2, i3, 0])` — `indices[0..3]` are valid.
///
/// # Performance
///
/// O(|item|).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::hash_item_standard_3ary;
///
/// let n = 243u32;
/// let (tag, idx) = hash_item_standard_3ary(b"hello", n, 12);
/// assert!(idx[0] < n && idx[1] < n && idx[2] < n);
/// ```
pub fn hash_item_standard_3ary(
    item: &[u8],
    num_buckets: u32,
    bits_per_tag: u32,
) -> (u32, [u32; 4]) {
    let raw = xxh3_64(item);
    let tag = extract_tag(raw, bits_per_tag);
    let i1 = ((raw >> 32) as u32) % num_buckets;
    let h = tag_hash_mod(tag, num_buckets);
    let i2 = xor3(i1, h);
    let i3 = xor3(i2, h);
    (tag, [i1, i2, i3, 0])
}

// ── Segmented 3-ary ────────────────────────────────────────────────────────

/// Hash an item for the segmented 3-ary scheme.
///
/// The XOR chain is computed in segment-local coordinates then offset by segment:
/// `i1 ∈ [0, seg)`, `i2 ∈ [seg, 2·seg)`, `i3 ∈ [2·seg, 3·seg)`.
///
/// # Arguments
///
/// - `item` — byte slice to hash.
/// - `segment_size` — `n / 3`; must be a power of 2.
/// - `bits_per_tag` — fingerprint bit width.
///
/// # Returns
///
/// `(tag, [i1, seg+i2_local, 2·seg+i3_local, 0])`.
///
/// # Performance
///
/// O(|item|).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::hash_item_segmented_3ary;
///
/// let seg = 32u32;
/// let (_, idx) = hash_item_segmented_3ary(b"hello", seg, 12);
/// assert!(idx[0] < seg);
/// assert!(idx[1] >= seg && idx[1] < 2 * seg);
/// assert!(idx[2] >= 2 * seg && idx[2] < 3 * seg);
/// ```
pub fn hash_item_segmented_3ary(
    item: &[u8],
    segment_size: u32,
    bits_per_tag: u32,
) -> (u32, [u32; 4]) {
    let (_h, tag, i1) = hash_and_i1(item, bits_per_tag, segment_size);
    let h1 = tag_hash1(tag, segment_size);
    let h2 = tag_hash2(tag, segment_size);
    let i2_local = i1 ^ h1;
    let i3_local = i2_local ^ h2;
    (
        tag,
        [i1, segment_size + i2_local, 2 * segment_size + i3_local, 0],
    )
}

// ── Standard 4-ary ─────────────────────────────────────────────────────────

/// Hash an item for the standard 4-ary scheme.
///
/// xor4 chain: `i2 = xor4(i1, h)`, `i3 = xor4(i2, h)`, `i4 = xor4(i3, h)` where
/// `h = tag_hash1(tag, n)`. All four indices are in `[0, num_buckets)`.
/// `num_buckets` must be a power of 4.
///
/// # Arguments
///
/// - `item` — byte slice to hash.
/// - `num_buckets` — must be a power of 4.
/// - `bits_per_tag` — fingerprint bit width.
///
/// # Returns
///
/// `(tag, [i1, i2, i3, i4])` — all four elements are valid.
///
/// # Performance
///
/// O(|item|).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::hash_item_standard_4ary;
///
/// let n = 256u32;
/// let (tag, idx) = hash_item_standard_4ary(b"hello", n, 12);
/// assert!(idx.iter().all(|&i| i < n));
/// ```
pub fn hash_item_standard_4ary(
    item: &[u8],
    num_buckets: u32,
    bits_per_tag: u32,
) -> (u32, [u32; 4]) {
    let (_h, tag, i1) = hash_and_i1(item, bits_per_tag, num_buckets);
    let h = tag_hash1(tag, num_buckets);
    let i2 = xor4(i1, h);
    let i3 = xor4(i2, h);
    let i4 = xor4(i3, h);
    (tag, [i1, i2, i3, i4])
}

// ── Segmented 4-ary ────────────────────────────────────────────────────────

/// Hash an item for the segmented 4-ary scheme.
///
/// `i_j ∈ [j·seg, (j+1)·seg)` for j = 0..3. Chain is computed in local coordinates then
/// offset per segment.
///
/// # Arguments
///
/// - `item` — byte slice to hash.
/// - `segment_size` — `n / 4`; must be a power of 2.
/// - `bits_per_tag` — fingerprint bit width.
///
/// # Returns
///
/// `(tag, [i1, seg+i2_local, 2·seg+i3_local, 3·seg+i4_local])`.
///
/// # Performance
///
/// O(|item|).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::hash_item_segmented_4ary;
///
/// let seg = 16u32;
/// let (_, idx) = hash_item_segmented_4ary(b"hello", seg, 12);
/// assert!(idx[0] < seg);
/// assert!(idx[1] >= seg   && idx[1] < 2 * seg);
/// assert!(idx[2] >= 2*seg && idx[2] < 3 * seg);
/// assert!(idx[3] >= 3*seg && idx[3] < 4 * seg);
/// ```
pub fn hash_item_segmented_4ary(
    item: &[u8],
    segment_size: u32,
    bits_per_tag: u32,
) -> (u32, [u32; 4]) {
    let (_h, tag, i1) = hash_and_i1(item, bits_per_tag, segment_size);
    let h1 = tag_hash1(tag, segment_size);
    let h2 = tag_hash2(tag, segment_size);
    let h3 = tag_hash3(tag, segment_size);
    let i2_local = i1 ^ h1;
    let i3_local = i2_local ^ h2;
    let i4_local = i3_local ^ h3;
    (
        tag,
        [
            i1,
            segment_size + i2_local,
            2 * segment_size + i3_local,
            3 * segment_size + i4_local,
        ],
    )
}

// ── all_indices reconstruction ──────────────────────────────────────────────
// Given one index + tag + chain position, reconstruct all k indices.

/// Reconstruct both candidate indices for the standard 2-ary scheme.
///
/// XOR symmetry means the chain can always be reconstructed from either endpoint plus the
/// tag. `position` 0 means `cur_index == i1`; position 1 means `cur_index == i2`.
///
/// # Arguments
///
/// - `cur_index` — the currently-occupied bucket index.
/// - `tag` — the fingerprint stored there.
/// - `position` — `0` if `cur_index` is `i1`, `1` if it is `i2`.
/// - `num_buckets` — table size; must be a power of 2.
///
/// # Returns
///
/// `[i1, i2, 0, 0]` — the two valid candidate indices.
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::{hash_item_standard, all_indices_standard};
///
/// let n = 128u32;
/// let (tag, idx) = hash_item_standard(b"hello", n, 12);
/// let recovered = all_indices_standard(idx[1], tag, 1, n);
/// assert_eq!(recovered[0], idx[0]);
/// assert_eq!(recovered[1], idx[1]);
/// ```
pub fn all_indices_standard(cur_index: u32, tag: u32, position: u8, num_buckets: u32) -> [u32; 4] {
    let h1 = tag_hash1(tag, num_buckets);
    // Reconstruct i1 by walking backward from position
    let i1 = match position {
        0 => cur_index,
        1 => cur_index ^ h1,
        _ => unreachable!("standard 2-ary position must be 0 or 1"),
    };
    let i2 = i1 ^ h1;
    [i1, i2, 0, 0]
}

/// Reconstruct both candidate indices for the segmented 2-ary scheme.
///
/// Position is inferred from which half `cur_index` falls in, so no explicit position
/// argument is needed (the `position` parameter to `IndexScheme::all_indices` is ignored
/// by the segmented scheme wrapper).
///
/// # Arguments
///
/// - `cur_index` — bucket index (in either half).
/// - `tag` — fingerprint stored at that index.
/// - `half` — segment size (`n / 2`); must be a power of 2.
///
/// # Returns
///
/// `[i1, i2, 0, 0]` where `i1 < half` and `i2 ∈ [half, 2·half)`.
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::{hash_item_segmented, all_indices_segmented};
///
/// let half = 64u32;
/// let (tag, idx) = hash_item_segmented(b"hello", half, 12);
/// let from_second = all_indices_segmented(idx[1], tag, half);
/// assert_eq!(from_second[0], idx[0]);
/// assert_eq!(from_second[1], idx[1]);
/// ```
pub fn all_indices_segmented(cur_index: u32, tag: u32, half: u32) -> [u32; 4] {
    let h1 = tag_hash1(tag, half);
    if cur_index < half {
        let i1 = cur_index;
        let i2 = half + (i1 ^ h1);
        [i1, i2, 0, 0]
    } else {
        let i2_local = cur_index - half;
        let i1 = i2_local ^ h1;
        [i1, cur_index, 0, 0]
    }
}

/// Reconstruct all three candidate indices for the standard 3-ary scheme.
///
/// xor3 cycling: applying h three times returns to start. Returns
/// `[cur_index, alt1, alt2, 0]` regardless of position.
///
/// # Arguments
///
/// - `cur_index` — one of the three candidate bucket indices.
/// - `tag` — fingerprint stored there.
/// - `_position` — ignored; position is determined by xor3 cycling.
/// - `num_buckets` — table size; must be a power of 3.
///
/// # Returns
///
/// `[cur_index, alt1, alt2, 0]`.
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::{hash_item_standard_3ary, all_indices_standard_3ary};
///
/// let n = 243u32;
/// let (tag, idx) = hash_item_standard_3ary(b"hello", n, 12);
/// for pos in 0..3u8 {
///     let r = all_indices_standard_3ary(idx[pos as usize], tag, pos, n);
///     let mut expected = [idx[0], idx[1], idx[2]];
///     expected.sort();
///     let mut got = [r[0], r[1], r[2]];
///     got.sort();
///     assert_eq!(got, expected);
/// }
/// ```
pub fn all_indices_standard_3ary(
    cur_index: u32,
    tag: u32,
    _position: u8,
    num_buckets: u32,
) -> [u32; 4] {
    // xor3 cycling: applying h three times returns to start.
    // all_indices always returns [cur_index, alt1, alt2, 0] regardless of position.
    let h = tag_hash_mod(tag, num_buckets);
    let alt1 = xor3(cur_index, h);
    let alt2 = xor3(alt1, h);
    [cur_index, alt1, alt2, 0]
}

/// Reconstruct all three candidate indices for the segmented 3-ary scheme.
///
/// Position is derived from which segment `cur_index` falls in (`cur_index / seg`).
/// The local index within the segment is used to recover `i2_local` (the hub), then
/// the full indices are assembled with segment offsets.
///
/// # Arguments
///
/// - `cur_index` — any one of the three candidate bucket indices.
/// - `tag` — fingerprint.
/// - `segment_size` — `n / 3`; must be a power of 2.
///
/// # Returns
///
/// `[i1, seg + i2_local, 2·seg + i3_local, 0]`.
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::{hash_item_segmented_3ary, all_indices_segmented_3ary};
///
/// let seg = 32u32;
/// let (tag, idx) = hash_item_segmented_3ary(b"hello", seg, 12);
/// let r = all_indices_segmented_3ary(idx[2], tag, seg);
/// assert_eq!(r[0], idx[0]);
/// ```
pub fn all_indices_segmented_3ary(cur_index: u32, tag: u32, segment_size: u32) -> [u32; 4] {
    let seg = segment_size;
    let h1 = tag_hash1(tag, seg);
    let h2 = tag_hash2(tag, seg);
    let position = cur_index / seg;
    let local = cur_index - position * seg;
    // Reconstruct i2_local (hub)
    let i2_local = match position {
        0 => local ^ h1,
        1 => local,
        2 => local ^ h2,
        _ => unreachable!(),
    };
    let i1 = i2_local ^ h1;
    let i3_local = i2_local ^ h2;
    [i1, seg + i2_local, 2 * seg + i3_local, 0]
}

/// Reconstruct all four candidate indices for the standard 4-ary scheme.
///
/// xor4 cycling: applying h four times returns to start. Returns
/// `[cur_index, alt1, alt2, alt3]` regardless of position.
///
/// # Arguments
///
/// - `cur_index` — any one of the four candidate bucket indices.
/// - `tag` — fingerprint.
/// - `_position` — ignored; position is determined by xor4 cycling.
/// - `num_buckets` — table size; must be a power of 4.
///
/// # Returns
///
/// `[cur_index, alt1, alt2, alt3]`.
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::{hash_item_standard_4ary, all_indices_standard_4ary};
///
/// let n = 256u32;
/// let (tag, idx) = hash_item_standard_4ary(b"hello", n, 12);
/// for pos in 0..4u8 {
///     let r = all_indices_standard_4ary(idx[pos as usize], tag, pos, n);
///     let mut expected = [idx[0], idx[1], idx[2], idx[3]];
///     expected.sort();
///     let mut got = [r[0], r[1], r[2], r[3]];
///     got.sort();
///     assert_eq!(got, expected);
/// }
/// ```
pub fn all_indices_standard_4ary(
    cur_index: u32,
    tag: u32,
    _position: u8,
    num_buckets: u32,
) -> [u32; 4] {
    // xor4 cycling: applying h four times returns to start.
    let h = tag_hash1(tag, num_buckets);
    let alt1 = xor4(cur_index, h);
    let alt2 = xor4(alt1, h);
    let alt3 = xor4(alt2, h);
    [cur_index, alt1, alt2, alt3]
}

/// Reconstruct all four candidate indices for the segmented 4-ary scheme.
///
/// Position is derived from which segment `cur_index` falls in. The local offset is
/// walked back to `i1` then the chain is rebuilt forward with segment offsets applied.
///
/// # Arguments
///
/// - `cur_index` — any one of the four candidate bucket indices.
/// - `tag` — fingerprint.
/// - `segment_size` — `n / 4`; must be a power of 2.
///
/// # Returns
///
/// `[i1, seg+i2_local, 2·seg+i3_local, 3·seg+i4_local]`.
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust
/// use segmented_cuckoo_filter::hash::{hash_item_segmented_4ary, all_indices_segmented_4ary};
///
/// let seg = 16u32;
/// let (tag, idx) = hash_item_segmented_4ary(b"hello", seg, 12);
/// let r = all_indices_segmented_4ary(idx[2], tag, seg);
/// assert_eq!(r, idx);
/// ```
pub fn all_indices_segmented_4ary(cur_index: u32, tag: u32, segment_size: u32) -> [u32; 4] {
    let seg = segment_size;
    let h1 = tag_hash1(tag, seg);
    let h2 = tag_hash2(tag, seg);
    let h3 = tag_hash3(tag, seg);
    let position = cur_index / seg;
    let local = cur_index - position * seg;
    // Walk to i1 (position 0)
    let i1 = match position {
        0 => local,
        1 => local ^ h1,
        2 => (local ^ h2) ^ h1,
        3 => ((local ^ h3) ^ h2) ^ h1,
        _ => unreachable!(),
    };
    let i2_local = i1 ^ h1;
    let i3_local = i2_local ^ h2;
    let i4_local = i3_local ^ h3;
    [i1, seg + i2_local, 2 * seg + i3_local, 3 * seg + i4_local]
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    // ── 2-ary standard ──

    #[test]
    fn standard_fingerprint_non_zero() {
        for i in 0u32..1000 {
            let (tag, _) = hash_item_standard(&i.to_le_bytes(), 128, 12);
            assert_ne!(tag, 0);
        }
    }

    #[test]
    fn standard_index_ranges() {
        let n = 128u32;
        for i in 0u32..1000 {
            let (_, idx) = hash_item_standard(&i.to_le_bytes(), n, 12);
            assert!(idx[0] < n);
            assert!(idx[1] < n);
        }
    }

    #[test]
    fn standard_all_indices_round_trip() {
        let n = 256u32;
        for i in 0u32..1000 {
            let (tag, idx) = hash_item_standard(&i.to_le_bytes(), n, 16);
            // From position 0
            let a = all_indices_standard(idx[0], tag, 0, n);
            assert_eq!(a[0], idx[0]);
            assert_eq!(a[1], idx[1]);
            // From position 1
            let b = all_indices_standard(idx[1], tag, 1, n);
            assert_eq!(b[0], idx[0]);
            assert_eq!(b[1], idx[1]);
        }
    }

    // ── 2-ary segmented ──

    #[test]
    fn segmented_fingerprint_non_zero() {
        for i in 0u32..1000 {
            let (tag, _) = hash_item_segmented(&i.to_le_bytes(), 64, 12);
            assert_ne!(tag, 0);
        }
    }

    #[test]
    fn segmented_index_ranges() {
        let half = 128u32;
        for i in 0u32..1000 {
            let (_, idx) = hash_item_segmented(&i.to_le_bytes(), half, 12);
            assert!(idx[0] < half);
            assert!(idx[1] >= half && idx[1] < half * 2);
        }
    }

    #[test]
    fn segmented_all_indices_round_trip() {
        let half = 256u32;
        for i in 0u32..1000 {
            let (tag, idx) = hash_item_segmented(&i.to_le_bytes(), half, 16);
            let a = all_indices_segmented(idx[0], tag, half);
            assert_eq!(a[0], idx[0]);
            assert_eq!(a[1], idx[1]);
            let b = all_indices_segmented(idx[1], tag, half);
            assert_eq!(b[0], idx[0]);
            assert_eq!(b[1], idx[1]);
        }
    }

    // ── 3-ary standard ──

    #[test]
    fn standard_3ary_index_ranges() {
        let n = 243u32;
        for i in 0u32..1000 {
            let (_, idx) = hash_item_standard_3ary(&i.to_le_bytes(), n, 12);
            assert!(idx[0] < n);
            assert!(idx[1] < n);
            assert!(idx[2] < n);
        }
    }

    #[test]
    fn standard_3ary_all_indices_round_trip() {
        let n = 243u32;
        for i in 0u32..1000 {
            let (tag, idx) = hash_item_standard_3ary(&i.to_le_bytes(), n, 16);
            for pos in 0..3u8 {
                let a = all_indices_standard_3ary(idx[pos as usize], tag, pos, n);
                let mut expected = [idx[0], idx[1], idx[2]];
                expected.sort();
                let mut got = [a[0], a[1], a[2]];
                got.sort();
                assert_eq!(got, expected, "i={} pos={}", i, pos);
            }
        }
    }

    #[test]
    fn xor3_identity_property() {
        // xor3(xor3(xor3(a, b), b), b) == a
        for a in [0u32, 1, 7, 100, 242] {
            for b in [0u32, 1, 5, 99, 241] {
                assert_eq!(xor3(xor3(xor3(a, b), b), b), a, "a={} b={}", a, b);
            }
        }
    }

    #[test]
    fn xor4_identity_property() {
        // xor4 applied 4 times returns to start
        for a in [0u32, 1, 5, 255, 65535] {
            for b in [0u32, 1, 3, 170, 65534] {
                let r = xor4(xor4(xor4(xor4(a, b), b), b), b);
                assert_eq!(r, a, "a={} b={}", a, b);
            }
        }
    }

    #[test]
    fn tag_hash_mod_range() {
        for range in [3u32, 9, 27, 243] {
            for tag in 1u32..=100 {
                assert!(tag_hash_mod(tag, range) < range);
            }
        }
    }

    // ── 3-ary segmented ──

    #[test]
    fn segmented_3ary_index_ranges() {
        let seg = 64u32;
        for i in 0u32..1000 {
            let (_, idx) = hash_item_segmented_3ary(&i.to_le_bytes(), seg, 12);
            assert!(idx[0] < seg, "i1={} must be < seg={}", idx[0], seg);
            assert!(idx[1] >= seg && idx[1] < 2 * seg);
            assert!(idx[2] >= 2 * seg && idx[2] < 3 * seg);
        }
    }

    #[test]
    fn segmented_3ary_all_indices_round_trip() {
        let seg = 128u32;
        for i in 0u32..1000 {
            let (tag, idx) = hash_item_segmented_3ary(&i.to_le_bytes(), seg, 16);
            for pos in 0..3 {
                let a = all_indices_segmented_3ary(idx[pos], tag, seg);
                assert_eq!(a[0], idx[0], "i={} pos={}", i, pos);
                assert_eq!(a[1], idx[1], "i={} pos={}", i, pos);
                assert_eq!(a[2], idx[2], "i={} pos={}", i, pos);
            }
        }
    }

    // ── 4-ary standard ──

    #[test]
    fn standard_4ary_index_ranges() {
        let n = 256u32;
        for i in 0u32..1000 {
            let (_, idx) = hash_item_standard_4ary(&i.to_le_bytes(), n, 12);
            for j in 0..4 {
                assert!(idx[j] < n);
            }
        }
    }

    #[test]
    fn standard_4ary_all_indices_round_trip() {
        let n = 256u32;
        for i in 0u32..1000 {
            let (tag, idx) = hash_item_standard_4ary(&i.to_le_bytes(), n, 16);
            for pos in 0..4u8 {
                let a = all_indices_standard_4ary(idx[pos as usize], tag, pos, n);
                let mut expected = [idx[0], idx[1], idx[2], idx[3]];
                expected.sort();
                let mut got = [a[0], a[1], a[2], a[3]];
                got.sort();
                assert_eq!(got, expected, "i={} pos={}", i, pos);
            }
        }
    }

    // ── 4-ary segmented ──

    #[test]
    fn segmented_4ary_index_ranges() {
        let seg = 64u32;
        for i in 0u32..1000 {
            let (_, idx) = hash_item_segmented_4ary(&i.to_le_bytes(), seg, 12);
            assert!(idx[0] < seg);
            assert!(idx[1] >= seg && idx[1] < 2 * seg);
            assert!(idx[2] >= 2 * seg && idx[2] < 3 * seg);
            assert!(idx[3] >= 3 * seg && idx[3] < 4 * seg);
        }
    }

    #[test]
    fn segmented_4ary_all_indices_round_trip() {
        let seg = 128u32;
        for i in 0u32..1000 {
            let (tag, idx) = hash_item_segmented_4ary(&i.to_le_bytes(), seg, 16);
            for pos in 0..4 {
                let a = all_indices_segmented_4ary(idx[pos], tag, seg);
                assert_eq!(a[0], idx[0], "i={} pos={}", i, pos);
                assert_eq!(a[1], idx[1], "i={} pos={}", i, pos);
                assert_eq!(a[2], idx[2], "i={} pos={}", i, pos);
                assert_eq!(a[3], idx[3], "i={} pos={}", i, pos);
            }
        }
    }

    // ── Tag hash independence ──

    #[test]
    fn tag_hashes_are_different() {
        // Ensure h1, h2, h3 produce different outputs for typical tags
        let range = 256u32;
        let mut same_count = 0;
        for tag in 1u32..1000 {
            let h1 = tag_hash1(tag, range);
            let h2 = tag_hash2(tag, range);
            let h3 = tag_hash3(tag, range);
            if h1 == h2 || h2 == h3 || h1 == h3 {
                same_count += 1;
            }
        }
        // Some collisions are expected, but the vast majority should differ
        assert!(
            same_count < 100,
            "too many hash collisions: {}/999",
            same_count
        );
    }

    // ── extract_tag ──

    #[test]
    fn extract_tag_never_zero() {
        // Craft a hash whose lower bits are 0 with various masks
        let h_zero_low: u64 = 0xDEAD_BEEF_0000_0000;
        assert_eq!(extract_tag(h_zero_low, 12), 1);
        assert_eq!(extract_tag(h_zero_low, 32), 1);
    }

    #[test]
    fn extract_tag_masks_correctly() {
        let h: u64 = 0x0000_0000_FFFF_FFFF;
        assert_eq!(extract_tag(h, 8), 0xFF);
        assert_eq!(extract_tag(h, 12), 0xFFF);
        assert_eq!(extract_tag(h, 32), 0xFFFF_FFFF);
    }
}
