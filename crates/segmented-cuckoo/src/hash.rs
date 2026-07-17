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
//! 2. **Fingerprint hash functions** — Three independent Murmur-constant mixers produce the
//!    XOR offsets that link candidate indices in the cuckoo chain. Each uses a distinct
//!    multiplier so their outputs are independently distributed for typical fingerprint values.
//!
//! 3. **Index reconstruction** — The `all_indices_*` family recovers all candidate bucket
//!    indices from one known index and a fingerprint store at this index. This is the critical
//!    path during cuckoo kicking. (this is called "partial-key cuckoo hashing" in the cuckoo filter paper).
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
//!   xxh3_64(item) → h: u64
//!   fingerprint        = lower 32 bits, masked to fingerprint_bits, non-zero
//!   primary index i1   = upper 32 bits, masked to (range - 1)
//! ```
//!
//! # File layout
//!
//! The segmented variants come first (our primary focus), followed by the standard
//! variants. Within each section, scheme order is 2-ary → 3-ary → 4-ary, and each arity
//! groups its `hash_item_*` entry point with its `all_indices_*` reconstruction helper.

use xxhash_rust::xxh3::xxh3_64;

// ── Fingerprint hash functions ──────────────────────────────────────────────
// Each uses a different mixing constant to produce independent hashes.
// `range` must be a power of 2; output is in [0, range).

/// Compute the first XOR offset for a fingerprint.
///
/// Multiplies `fingerprint` by the MurmurHash2 constant `0x5bd1e995` (with wrapping) and
/// masks the result to `[0, range)`. Used to derive `i2` from `i1` in all schemes.
///
/// # Arguments
///
/// - `fingerprint` — non-zero fingerprint value.
/// - `range` — table size or segment size.
///
/// # Constraints
///
/// - `range` **must be a power of 2**.
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
/// ```rust,ignore
/// use segmented_cuckoo::hash::fingerprint_hash1;
///
/// let offset = fingerprint_hash1(0xABC, 256);
/// assert!(offset < 256);
/// ```
#[inline]
pub const fn fingerprint_hash1(fingerprint: u32, range: u32) -> u32 {
    fingerprint.wrapping_mul(0x5bd1e995) & (range - 1)
}

/// Compute the second XOR offset for a fingerprint.
///
/// Multiplies `fingerprint` by the MurmurHash3 c1 constant `0xcc9e2d51` and masks to
/// `[0, range)`. Used to derive `i3` from `i2` in 3-ary and 4-ary schemes.
///
/// # Arguments
///
/// - `fingerprint` — non-zero fingerprint value.
/// - `range` — table size or segment size.
///
/// # Constraints
///
/// - `range` **must be a power of 2**.
///
/// # Returns
///
/// An offset in `[0, range)`, statistically independent of [`fingerprint_hash1`] for most
/// inputs.
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::fingerprint_hash2;
///
/// let offset = fingerprint_hash2(0xABC, 256);
/// assert!(offset < 256);
/// ```
#[inline]
pub const fn fingerprint_hash2(fingerprint: u32, range: u32) -> u32 {
    fingerprint.wrapping_mul(0xcc9e2d51) & (range - 1)
}

/// Compute the third XOR offset for a fingerprint.
///
/// Multiplies `fingerprint` by the MurmurHash3 c2 constant `0x1b873593` and masks to
/// `[0, range)`. Used to derive `i4` from `i3` in 4-ary schemes.
///
/// # Arguments
///
/// - `fingerprint` — non-zero fingerprint value.
/// - `range` — table size or segment size.
///
/// # Constraints
///
/// - `range` **must be a power of 2**.
///
/// # Returns
///
/// An offset in `[0, range)`, statistically independent of `fingerprint_hash1` and
/// `fingerprint_hash2` for most inputs.
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::fingerprint_hash3;
///
/// let offset = fingerprint_hash3(0xABC, 256);
/// assert!(offset < 256);
/// ```
#[inline]
pub const fn fingerprint_hash3(fingerprint: u32, range: u32) -> u32 {
    fingerprint.wrapping_mul(0x1b873593) & (range - 1)
}

// ── Common helpers ──────────────────────────────────────────────────────────

/// Extract a non-zero fingerprint from the lower 32 bits of an xxh3 hash.
///
/// Fingerprint 0 is reserved to mean "slot empty" in [`FingerprintTable`]. When the masked
/// hash value is 0, this function returns 1 instead, introducing a tiny bias toward
/// fingerprint = 1 in exchange for keeping the empty-slot invariant universally valid.
///
/// # Arguments
///
/// - `h` — 64-bit xxh3 hash of the item.
/// - `fingerprint_bits` — fingerprint bit width (1–32). Must be ≥ 1.
///
/// # Returns
///
/// A fingerprint in `[1, 2^fingerprint_bits]` (never 0).
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::extract_fingerprint;
///
/// // When the lower bits happen to be 0, fingerprint is forced to 1.
/// let fp = extract_fingerprint(0xFFFF_FFFF_0000_0000u64, 12);
/// assert_eq!(fp, 1);
///
/// // Normal case: mask to fingerprint_bits.
/// let fp2 = extract_fingerprint(0x0000_0000_0000_0FFFu64, 12);
/// assert_eq!(fp2, 0xFFF);
/// ```
///
/// [`FingerprintTable`]: crate::fingerprint_table::FingerprintTable
#[inline]
pub const fn extract_fingerprint(h: u64, fingerprint_bits: u32) -> u32 {
    let mask = if fingerprint_bits >= 32 {
        u32::MAX
    } else {
        (1u32 << fingerprint_bits) - 1
    };
    let mut fingerprint = (h as u32) & mask;
    if fingerprint == 0 {
        fingerprint = 1;
    }
    fingerprint
}

/// Hash `item` with xxh3_64, extract the fingerprint, and derive the primary index `i1`.
///
/// This is the entry point for all `hash_item_*` functions. It wraps the xxh3 call and
/// the two extraction steps into a single reusable helper.
///
/// # Arguments
///
/// - `item` — arbitrary byte slice to hash.
/// - `fingerprint_bits` — fingerprint bit width (1–32).
/// - `range` — table size or segment size; **must be a power of 2**. `i1` is masked to
///   `[0, range)`.
///
/// # Returns
///
/// `(h, fingerprint, i1)` where:
/// - `h` — raw 64-bit xxh3 hash (exposed for tests / debugging).
/// - `fingerprint` — non-zero fingerprint in `[1, 2^fingerprint_bits]`.
/// - `i1` — primary bucket index in `[0, range)`.
///
/// # Performance
///
/// O(|item|) — dominated by the xxh3 call, which processes input in 256-bit SIMD blocks
/// on supported hardware (AVX2 / NEON).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::hash_and_i1;
///
/// let (_, fingerprint, i1) = hash_and_i1(b"hello", 12, 128);
/// assert!(fingerprint >= 1 && fingerprint <= 0xFFF);
/// assert!(i1 < 128);
/// ```
#[inline]
pub fn hash_and_i1(item: &[u8], fingerprint_bits: u32, range: u32) -> (u64, u32, u32) {
    let h = xxh3_64(item);
    let fingerprint = extract_fingerprint(h, fingerprint_bits);
    let i1 = ((h >> 32) as u32) & (range - 1);
    (h, fingerprint, i1)
}

// ── xor3 and xor4 implementation ────────────────────────────────────────────────────────

/// Base-3 digits per [`XOR3_T81`] chunk. `3^4 = 81` fits a `u8`, keeping the
/// table at `81 * 81` = 6.4 KiB — L1-resident on every target we build for.
const XOR3_CHUNK_DIGITS: u32 = 4;

/// `XOR3_T81[x][y]` = base-3 digit-wise sum mod 3 of `x` and `y`, each read as a
/// [`XOR3_CHUNK_DIGITS`]-digit base-3 number.
///
/// Built at compile time by the same digit loop [`xor3`] used to run at
/// runtime, so the table *is* the reference semantics, evaluated once.
const fn build_xor3_t81() -> [[u8; 81]; 81] {
    let mut t = [[0u8; 81]; 81];
    let mut x = 0usize;
    while x < 81 {
        let mut y = 0usize;
        while y < 81 {
            let mut r = 0u32;
            let mut place = 1u32;
            let mut d = 0;
            while d < XOR3_CHUNK_DIGITS {
                r += ((x as u32 / place) % 3 + (y as u32 / place) % 3) % 3 * place;
                place *= 3;
                d += 1;
            }
            t[x][y] = r as u8;
            y += 1;
        }
        x += 1;
    }
    t
}

static XOR3_T81: [[u8; 81]; 81] = build_xor3_t81();

/// Follow the academic paper "D-Ary Cuckoo Filter: A Space Efficient Data Structure for Set Membership Lookup"
/// at https://ieeexplore.ieee.org/abstract/document/8368364
///
/// Base-3 digit-wise addition mod 3. Property: xor3(xor3(xor3(a,b),b),b) == a.
///
/// That cycling property is what lets [`all_indices_standard_3ary`] rebuild all
/// three candidate buckets from whichever one it is standing on, so no per-slot
/// position needs storing.
///
/// # Why a table
///
/// Digit `k` of the result depends only on digit `k` of each input — mod-3
/// addition carries nothing. The digits are therefore independent, and the
/// obvious `while aa != 0 { aa /= 3 }` digit loop is the wrong shape: each
/// iteration's divide feeds the next, so a `3^13` table serialises ~13 divides
/// (~26 ns) while the core sits idle. Chunking into [`XOR3_CHUNK_DIGITS`]-digit
/// groups turns that chain into five *independent* lookups the core issues at
/// once (~3.7 ns): every divisor below is a literal, so each becomes a
/// multiply-shift rather than a division.
///
/// # Digit range
///
/// Digits `0..=19` only — digit 20 is deliberately **not**
/// processed, matching the digit loop this replaced (which stopped once
/// `place > 1_000_000_000`, and `3^19 = 1_162_261_467` trips that). Callers pass
/// indices `< num_buckets <= 3^20`, whose digit 20 is always 0, so the cutoff is
/// unobservable through the filter; it is preserved so the two agree on every
/// `u32`, including the `>= 3^20` values only a direct call can produce. The
/// trailing `% 81` on each chunk is what discards it.
///
/// # Returns
///
/// The digit-wise mod-3 sum, at most `3^20 - 1 = 3_486_784_400` — no overflow.
///
/// # Performance
///
/// O(1): five independent table lookups, no division, no branches.
#[inline]
pub const fn xor3(a: u32, b: u32) -> u32 {
    // Chunk j covers digits 4j..=4j+3; `/ 3^(4j)` shifts it down, `% 81` keeps 4 digits.
    let c0 = XOR3_T81[(a % 81) as usize][(b % 81) as usize] as u32;
    let c1 = XOR3_T81[((a / 81) % 81) as usize][((b / 81) % 81) as usize] as u32;
    let c2 = XOR3_T81[((a / 6561) % 81) as usize][((b / 6561) % 81) as usize] as u32;
    let c3 = XOR3_T81[((a / 531441) % 81) as usize][((b / 531441) % 81) as usize] as u32;
    let c4 = XOR3_T81[((a / 43046721) % 81) as usize][((b / 43046721) % 81) as usize] as u32;
    // Reassemble at each chunk's base-3 weight 3^(4j).
    c0 + c1 * 81 + c2 * 6561 + c3 * 531441 + c4 * 43046721
}

/// Base-4 digit-wise addition mod 4 using bitwise trick (O(1)).
/// Property: xor4(xor4(xor4(xor4(a,b),b),b),b) == a.
#[inline]
pub const fn xor4(a: u32, b: u32) -> u32 {
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

/// Fingerprint hash using modulo (for power-of-3 ranges where bitmask doesn't apply).
///
/// - `range` — table size; need not be a power of 2.
#[inline]
pub const fn fingerprint_hash_mod(fingerprint: u32, range: u32) -> u32 {
    fingerprint.wrapping_mul(0x5bd1e995) % range
}

// ════════════════════════════════════════════════════════════════════════════
// ░░ SEGMENTED SCHEMES (primary focus) ░░
// ════════════════════════════════════════════════════════════════════════════

// ── Segmented 2-ary ────────────────────────────────────────────────────────

/// Hash an item for the segmented 2-ary scheme.
///
/// `i1 ∈ [0, half)` and `i2 = half + (i1 ^ fingerprint_hash1(fingerprint, half))` so
/// `i2 ∈ [half, num_buckets)`. Each index is confined to its own segment.
///
/// # Arguments
///
/// - `item` — byte slice to hash.
/// - `segment_size` — segment size (`num_buckets / 2`); must be a power of 2.
/// - `fingerprint_bits` — fingerprint bit width.
///
/// # Returns
///
/// `(fingerprint, [i1, i2, 0, 0])` where `i1 < half` and `i2 ∈ [half, 2·half)`.
///
/// # Performance
///
/// O(|item|).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::hash_item_segmented_2ary;
///
/// let segment_size = 64u32;
/// let (fingerprint, idx) = hash_item_segmented_2ary(b"hello", segment_size, 12);
/// assert!(idx[0] < segment_size);
/// assert!(idx[1] >= segment_size && idx[1] < 2 * segment_size);
/// ```
pub fn hash_item_segmented_2ary(
    item: &[u8],
    segment_size: u32,
    fingerprint_bits: u32,
) -> (u32, [u32; 4]) {
    let (_h, fingerprint, i1) = hash_and_i1(item, fingerprint_bits, segment_size);
    let i2 = segment_size + (i1 ^ fingerprint_hash1(fingerprint, segment_size));
    (fingerprint, [i1, i2, 0, 0])
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
/// - `fingerprint` — fingerprint stored at that index.
/// - `segment_size` — segment size (`num_buckets / 2`); must be a power of 2.
///
/// # Returns
///
/// `[i1, i2, 0, 0]` where `i1 < segment_size` and `i2 ∈ [segment_size, 2·segment_size)`.
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::{hash_item_segmented_2ary, all_indices_segmented_2ary};
///
/// let segment_size = 64u32;
/// let (fingerprint, idx) = hash_item_segmented_2ary(b"hello", segment_size, 12);
/// let from_second = all_indices_segmented_2ary(idx[1], fingerprint, segment_size);
/// assert_eq!(from_second[0], idx[0]);
/// assert_eq!(from_second[1], idx[1]);
/// ```
pub const fn all_indices_segmented_2ary(
    cur_index: u32,
    fingerprint: u32,
    segment_size: u32,
) -> [u32; 4] {
    let h1 = fingerprint_hash1(fingerprint, segment_size);
    if cur_index < segment_size {
        let i1 = cur_index;
        let i2 = segment_size + (i1 ^ h1);
        [i1, i2, 0, 0]
    } else {
        let i2_local = cur_index - segment_size;
        let i1 = i2_local ^ h1;
        [i1, cur_index, 0, 0]
    }
}

// ── Segmented 3-ary ────────────────────────────────────────────────────────

/// Hash an item for the segmented 3-ary scheme.
///
/// The XOR chain is computed in segment-local coordinates then offset by segment:
/// `i1 ∈ [0, segment_size)`, `i2 ∈ [segment_size, 2·segment_size)`, `i3 ∈ [2·segment_size, 3·segment_size)`.
///
/// # Arguments
///
/// - `item` — byte slice to hash.
/// - `segment_size` — `num_buckets / 3`; must be a power of 2.
/// - `fingerprint_bits` — fingerprint bit width.
///
/// # Returns
///
/// `(fingerprint, [i1, segment_size+i2_local, 2·segment_size+i3_local, 0])`.
///
/// # Performance
///
/// O(|item|).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::hash_item_segmented_3ary;
///
/// let segment_size = 32u32;
/// let (_, idx) = hash_item_segmented_3ary(b"hello", segment_size, 12);
/// assert!(idx[0] < segment_size);
/// assert!(idx[1] >= segment_size && idx[1] < 2 * segment_size);
/// assert!(idx[2] >= 2 * segment_size && idx[2] < 3 * segment_size);
/// ```
pub fn hash_item_segmented_3ary(
    item: &[u8],
    segment_size: u32,
    fingerprint_bits: u32,
) -> (u32, [u32; 4]) {
    let (_h, fingerprint, i1) = hash_and_i1(item, fingerprint_bits, segment_size);
    let h1 = fingerprint_hash1(fingerprint, segment_size);
    let h2 = fingerprint_hash2(fingerprint, segment_size);
    let i2_local = i1 ^ h1;
    let i3_local = i2_local ^ h2;
    (
        fingerprint,
        [i1, segment_size + i2_local, 2 * segment_size + i3_local, 0],
    )
}

/// Reconstruct all three candidate indices for the segmented 3-ary scheme.
///
/// Position is derived from which segment `cur_index` falls in (`cur_index / segment_size`).
/// The local index within the segment is used to recover `i2_local` (the hub), then
/// the full indices are assembled with segment offsets.
///
/// # Arguments
///
/// - `cur_index` — any one of the three candidate bucket indices.
/// - `fingerprint` — fingerprint.
/// - `segment_size` — `num_buckets / 3`; must be a power of 2.
///
/// # Returns
///
/// `[i1, segment_size + i2_local, 2·segment_size + i3_local, 0]`.
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::{hash_item_segmented_3ary, all_indices_segmented_3ary};
///
/// let segment_size = 32u32;
/// let (fingerprint, idx) = hash_item_segmented_3ary(b"hello", segment_size, 12);
/// let r = all_indices_segmented_3ary(idx[2], fingerprint, segment_size);
/// assert_eq!(r[0], idx[0]);
/// ```
pub fn all_indices_segmented_3ary(cur_index: u32, fingerprint: u32, segment_size: u32) -> [u32; 4] {
    let h1 = fingerprint_hash1(fingerprint, segment_size);
    let h2 = fingerprint_hash2(fingerprint, segment_size);
    let position = cur_index / segment_size;
    let local = cur_index - position * segment_size;
    // Reconstruct i2_local (hub)
    let i2_local = match position {
        0 => local ^ h1,
        1 => local,
        2 => local ^ h2,
        _ => unreachable!(),
    };
    let i1 = i2_local ^ h1;
    let i3_local = i2_local ^ h2;
    [i1, segment_size + i2_local, 2 * segment_size + i3_local, 0]
}

// ── Segmented 4-ary ────────────────────────────────────────────────────────

/// Hash an item for the segmented 4-ary scheme.
///
/// `i_j ∈ [j·segment_size, (j+1)·segment_size)` for j = 0..3. Chain is computed in local coordinates then
/// offset per segment.
///
/// # Arguments
///
/// - `item` — byte slice to hash.
/// - `segment_size` — `num_buckets / 4`; must be a power of 2.
/// - `fingerprint_bits` — fingerprint bit width.
///
/// # Returns
///
/// `(fingerprint, [i1, segment_size+i2_local, 2·segment_size+i3_local, 3·segment_size+i4_local])`.
///
/// # Performance
///
/// O(|item|).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::hash_item_segmented_4ary;
///
/// let segment_size = 16u32;
/// let (_, idx) = hash_item_segmented_4ary(b"hello", segment_size, 12);
/// assert!(idx[0] < segment_size);
/// assert!(idx[1] >= segment_size   && idx[1] < 2 * segment_size);
/// assert!(idx[2] >= 2*segment_size && idx[2] < 3 * segment_size);
/// assert!(idx[3] >= 3*segment_size && idx[3] < 4 * segment_size);
/// ```
pub fn hash_item_segmented_4ary(
    item: &[u8],
    segment_size: u32,
    fingerprint_bits: u32,
) -> (u32, [u32; 4]) {
    let (_h, fingerprint, i1) = hash_and_i1(item, fingerprint_bits, segment_size);
    let h1 = fingerprint_hash1(fingerprint, segment_size);
    let h2 = fingerprint_hash2(fingerprint, segment_size);
    let h3 = fingerprint_hash3(fingerprint, segment_size);
    let i2_local = i1 ^ h1;
    let i3_local = i2_local ^ h2;
    let i4_local = i3_local ^ h3;
    (
        fingerprint,
        [
            i1,
            segment_size + i2_local,
            2 * segment_size + i3_local,
            3 * segment_size + i4_local,
        ],
    )
}

/// Reconstruct all four candidate indices for the segmented 4-ary scheme.
///
/// Position is derived from which segment `cur_index` falls in. The local offset is
/// walked back to `i1` then the chain is rebuilt forward with segment offsets applied.
///
/// # Arguments
///
/// - `cur_index` — any one of the four candidate bucket indices.
/// - `fingerprint` — fingerprint.
/// - `segment_size` — `num_buckets / 4`; must be a power of 2.
///
/// # Returns
///
/// `[i1, segment_size+i2_local, 2·segment_size+i3_local, 3·segment_size+i4_local]`.
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::{hash_item_segmented_4ary, all_indices_segmented_4ary};
///
/// let segment_size = 16u32;
/// let (fingerprint, idx) = hash_item_segmented_4ary(b"hello", segment_size, 12);
/// let r = all_indices_segmented_4ary(idx[2], fingerprint, segment_size);
/// assert_eq!(r, idx);
/// ```
pub fn all_indices_segmented_4ary(cur_index: u32, fingerprint: u32, segment_size: u32) -> [u32; 4] {
    let h1 = fingerprint_hash1(fingerprint, segment_size);
    let h2 = fingerprint_hash2(fingerprint, segment_size);
    let h3 = fingerprint_hash3(fingerprint, segment_size);
    let position = cur_index / segment_size;
    let local = cur_index - position * segment_size;
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
    [
        i1,
        segment_size + i2_local,
        2 * segment_size + i3_local,
        3 * segment_size + i4_local,
    ]
}

// ════════════════════════════════════════════════════════════════════════════
// ░░ STANDARD SCHEMES ░░
// ════════════════════════════════════════════════════════════════════════════

// ── Standard 2-ary (original) ──────────────────────────────────────────────

/// Follow the academic paper "Cuckoo Filter: Practically Better Than Bloom"
/// at https://www.cs.cmu.edu/~dga/papers/cuckoo-conext2014.pdf
///
/// Hash an item for the standard 2-ary scheme, yielding the fingerprint and two candidate indices.
///
/// Computes `i2 = i1 ^ fingerprint_hash1(fingerprint, num_buckets)`. Both indices are in
/// `[0, num_buckets)`.
///
/// # Arguments
///
/// - `item` — byte slice to hash.
/// - `num_buckets` — total number of buckets; must be a power of 2.
/// - `fingerprint_bits` — fingerprint bit width.
///
/// # Returns
///
/// `(fingerprint, [i1, i2, 0, 0])` — `indices[0..2]` are valid; `indices[2..4]` are `0`.
///
/// # Performance
///
/// O(|item|) — xxh3 dominates; the XOR step is O(1).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::hash_item_standard_2ary;
///
/// let (fingerprint, idx) = hash_item_standard_2ary(b"hello", 128, 12);
/// assert_ne!(fingerprint, 0);
/// assert!(idx[0] < 128);
/// assert!(idx[1] < 128);
/// ```
pub fn hash_item_standard_2ary(
    item: &[u8],
    num_buckets: u32,
    fingerprint_bits: u32,
) -> (u32, [u32; 4]) {
    let (_h, fingerprint, i1) = hash_and_i1(item, fingerprint_bits, num_buckets);
    let i2 = i1 ^ fingerprint_hash1(fingerprint, num_buckets);
    (fingerprint, [i1, i2, 0, 0])
}

/// Reconstruct both candidate indices for the standard 2-ary scheme.
///
/// XOR symmetry: `i2 = i1 ^ h1` and `i1 = i2 ^ h1`, so the pair can always be
/// reconstructed from either endpoint. Returns `[cur_index, cur_index ^ h1, 0, 0]`.
///
/// # Arguments
///
/// - `cur_index` — either `i1` or `i2`; the XOR relationship is symmetric.
/// - `fingerprint` — the fingerprint stored there.
/// - `num_buckets` — table size; must be a power of 2.
///
/// # Returns
///
/// `[cur_index, cur_index ^ h1, 0, 0]` — the two valid candidate indices
/// (note: the pair `{result[0], result[1]}` equals `{i1, i2}` but the order
/// depends on which endpoint `cur_index` is).
///
/// # Performance
///
/// O(1).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::{hash_item_standard_2ary, all_indices_standard_2ary};
///
/// let num_buckets = 128u32;
/// let (fingerprint, idx) = hash_item_standard_2ary(b"hello", num_buckets, 12);
/// // The pair {a[0], a[1]} must equal {idx[0], idx[1]} regardless of which endpoint we start from.
/// for &start in &[idx[0], idx[1]] {
///     let a = all_indices_standard_2ary(start, fingerprint, num_buckets);
///     let mut got = [a[0], a[1]]; got.sort();
///     let mut exp = [idx[0], idx[1]]; exp.sort();
///     assert_eq!(got, exp);
/// }
/// ```
pub const fn all_indices_standard_2ary(
    cur_index: u32,
    fingerprint: u32,
    num_buckets: u32,
) -> [u32; 4] {
    let h1 = fingerprint_hash1(fingerprint, num_buckets);
    [cur_index, cur_index ^ h1, 0, 0]
}

// ── Standard 3-ary ─────────────────────────────────────────────────────────

/// Follow the academic paper "D-Ary Cuckoo Filter: A Space Efficient Data Structure for Set Membership Lookup"
/// at https://ieeexplore.ieee.org/abstract/document/8368364
///
/// Hash an item for the standard 3-ary scheme.
///
/// xor3 chain: `i2 = xor3(i1, h)`, `i3 = xor3(i2, h)` where
/// `h = fingerprint_hash_mod(fingerprint, num_buckets)`. All three indices are in `[0, num_buckets)`.
/// `num_buckets` must be a power of 3.
///
/// # Arguments
///
/// - `item` — byte slice to hash.
/// - `num_buckets` — must be a power of 3.
/// - `fingerprint_bits` — fingerprint bit width.
///
/// # Returns
///
/// `(fingerprint, [i1, i2, i3, 0])` — `indices[0..3]` are valid.
///
/// # Performance
///
/// O(|item|).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::hash_item_standard_3ary;
///
/// let num_buckets = 243u32;
/// let (fingerprint, idx) = hash_item_standard_3ary(b"hello", num_buckets, 12);
/// assert!(idx[0] < num_buckets && idx[1] < num_buckets && idx[2] < num_buckets);
/// ```
pub fn hash_item_standard_3ary(
    item: &[u8],
    num_buckets: u32,
    fingerprint_bits: u32,
) -> (u32, [u32; 4]) {
    let raw = xxh3_64(item);
    let fingerprint = extract_fingerprint(raw, fingerprint_bits);
    let i1 = ((raw >> 32) as u32) % num_buckets;
    let h = fingerprint_hash_mod(fingerprint, num_buckets);
    let i2 = xor3(i1, h);
    let i3 = xor3(i2, h);
    (fingerprint, [i1, i2, i3, 0])
}

/// Reconstruct all three candidate indices for the standard 3-ary scheme.
///
/// xor3 cycling: applying h three times returns to start. Returns
/// `[cur_index, alt1, alt2, 0]` regardless of position.
///
/// # Arguments
///
/// - `cur_index` — one of the three candidate bucket indices.
/// - `fingerprint` — fingerprint stored there.
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
/// ```rust,ignore
/// use segmented_cuckoo::hash::{hash_item_standard_3ary, all_indices_standard_3ary};
///
/// let num_buckets = 243u32;
/// let (fingerprint, idx) = hash_item_standard_3ary(b"hello", num_buckets, 12);
/// for pos in 0..3 {
///     let r = all_indices_standard_3ary(idx[pos], fingerprint, num_buckets);
///     let mut expected = [idx[0], idx[1], idx[2]];
///     expected.sort();
///     let mut got = [r[0], r[1], r[2]];
///     got.sort();
///     assert_eq!(got, expected);
/// }
/// ```
pub const fn all_indices_standard_3ary(
    cur_index: u32,
    fingerprint: u32,
    num_buckets: u32,
) -> [u32; 4] {
    // xor3 cycling: applying h three times returns to start.
    // all_indices always returns [cur_index, alt1, alt2, 0] regardless of position.
    let h = fingerprint_hash_mod(fingerprint, num_buckets);
    let alt1 = xor3(cur_index, h);
    let alt2 = xor3(alt1, h);
    [cur_index, alt1, alt2, 0]
}

// ── Standard 4-ary ─────────────────────────────────────────────────────────

/// Follow the academic paper "D-Ary Cuckoo Filter: A Space Efficient Data Structure for Set Membership Lookup"
/// at https://ieeexplore.ieee.org/abstract/document/8368364
///
/// Hash an item for the standard 4-ary scheme.
///
/// xor4 chain: `i2 = xor4(i1, h)`, `i3 = xor4(i2, h)`, `i4 = xor4(i3, h)` where
/// `h = fingerprint_hash1(fingerprint, num_buckets)`. All four indices are in `[0, num_buckets)`.
/// `num_buckets` must be a power of 4.
///
/// # Arguments
///
/// - `item` — byte slice to hash.
/// - `num_buckets` — must be a power of 4.
/// - `fingerprint_bits` — fingerprint bit width.
///
/// # Returns
///
/// `(fingerprint, [i1, i2, i3, i4])` — all four elements are valid.
///
/// # Performance
///
/// O(|item|).
///
/// # Examples
///
/// ```rust,ignore
/// use segmented_cuckoo::hash::hash_item_standard_4ary;
///
/// let num_buckets = 256u32;
/// let (fingerprint, idx) = hash_item_standard_4ary(b"hello", num_buckets, 12);
/// assert!(idx.iter().all(|&i| i < num_buckets));
/// ```
pub fn hash_item_standard_4ary(
    item: &[u8],
    num_buckets: u32,
    fingerprint_bits: u32,
) -> (u32, [u32; 4]) {
    let (_h, fingerprint, i1) = hash_and_i1(item, fingerprint_bits, num_buckets);
    let h = fingerprint_hash1(fingerprint, num_buckets);
    let i2 = xor4(i1, h);
    let i3 = xor4(i2, h);
    let i4 = xor4(i3, h);
    (fingerprint, [i1, i2, i3, i4])
}

/// Reconstruct all four candidate indices for the standard 4-ary scheme.
///
/// xor4 cycling: applying h four times returns to start. Returns
/// `[cur_index, alt1, alt2, alt3]` regardless of position.
///
/// # Arguments
///
/// - `cur_index` — any one of the four candidate bucket indices.
/// - `fingerprint` — fingerprint.
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
/// ```rust,ignore
/// use segmented_cuckoo::hash::{hash_item_standard_4ary, all_indices_standard_4ary};
///
/// let num_buckets = 256u32;
/// let (fingerprint, idx) = hash_item_standard_4ary(b"hello", num_buckets, 12);
/// for pos in 0..4 {
///     let r = all_indices_standard_4ary(idx[pos], fingerprint, num_buckets);
///     let mut expected = [idx[0], idx[1], idx[2], idx[3]];
///     expected.sort();
///     let mut got = [r[0], r[1], r[2], r[3]];
///     got.sort();
///     assert_eq!(got, expected);
/// }
/// ```
pub const fn all_indices_standard_4ary(
    cur_index: u32,
    fingerprint: u32,
    num_buckets: u32,
) -> [u32; 4] {
    // xor4 cycling: applying h four times returns to start.
    let h = fingerprint_hash1(fingerprint, num_buckets);
    let alt1 = xor4(cur_index, h);
    let alt2 = xor4(alt1, h);
    let alt3 = xor4(alt2, h);
    [cur_index, alt1, alt2, alt3]
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    // ── 2-ary segmented ──

    #[test]
    fn segmented_2ary_fingerprint_non_zero() {
        for i in 0u32..1000 {
            let (fingerprint, _) = hash_item_segmented_2ary(&i.to_le_bytes(), 64, 12);
            assert_ne!(fingerprint, 0);
        }
    }

    #[test]
    fn segmented_2ary_index_ranges() {
        let segment_size = 128u32;
        for i in 0u32..1000 {
            let (_, idx) = hash_item_segmented_2ary(&i.to_le_bytes(), segment_size, 12);
            assert!(idx[0] < segment_size);
            assert!(idx[1] >= segment_size && idx[1] < segment_size * 2);
        }
    }

    #[test]
    fn segmented_2ary_all_indices_round_trip() {
        let segment_size = 256u32;
        for i in 0u32..1000 {
            let (fingerprint, idx) = hash_item_segmented_2ary(&i.to_le_bytes(), segment_size, 16);
            let a = all_indices_segmented_2ary(idx[0], fingerprint, segment_size);
            assert_eq!(a[0], idx[0]);
            assert_eq!(a[1], idx[1]);
            let b = all_indices_segmented_2ary(idx[1], fingerprint, segment_size);
            assert_eq!(b[0], idx[0]);
            assert_eq!(b[1], idx[1]);
        }
    }

    // ── 3-ary segmented ──

    #[test]
    fn segmented_3ary_index_ranges() {
        let segment_size = 64u32;
        for i in 0u32..1000 {
            let (_, idx) = hash_item_segmented_3ary(&i.to_le_bytes(), segment_size, 12);
            assert!(
                idx[0] < segment_size,
                "i1={} must be < segment_size={}",
                idx[0],
                segment_size
            );
            assert!(idx[1] >= segment_size && idx[1] < 2 * segment_size);
            assert!(idx[2] >= 2 * segment_size && idx[2] < 3 * segment_size);
        }
    }

    #[test]
    fn segmented_3ary_all_indices_round_trip() {
        let segment_size = 128u32;
        for i in 0u32..1000 {
            let (fingerprint, idx) = hash_item_segmented_3ary(&i.to_le_bytes(), segment_size, 16);
            for pos in 0..3 {
                let a = all_indices_segmented_3ary(idx[pos], fingerprint, segment_size);
                assert_eq!(a[0], idx[0], "i={} pos={}", i, pos);
                assert_eq!(a[1], idx[1], "i={} pos={}", i, pos);
                assert_eq!(a[2], idx[2], "i={} pos={}", i, pos);
            }
        }
    }

    // ── 4-ary segmented ──

    #[test]
    fn segmented_4ary_index_ranges() {
        let segment_size = 64u32;
        for i in 0u32..1000 {
            let (_, idx) = hash_item_segmented_4ary(&i.to_le_bytes(), segment_size, 12);
            assert!(idx[0] < segment_size);
            assert!(idx[1] >= segment_size && idx[1] < 2 * segment_size);
            assert!(idx[2] >= 2 * segment_size && idx[2] < 3 * segment_size);
            assert!(idx[3] >= 3 * segment_size && idx[3] < 4 * segment_size);
        }
    }

    #[test]
    fn segmented_4ary_all_indices_round_trip() {
        let segment_size = 128u32;
        for i in 0u32..1000 {
            let (fingerprint, idx) = hash_item_segmented_4ary(&i.to_le_bytes(), segment_size, 16);
            for pos in 0..4 {
                let a = all_indices_segmented_4ary(idx[pos], fingerprint, segment_size);
                assert_eq!(a[0], idx[0], "i={} pos={}", i, pos);
                assert_eq!(a[1], idx[1], "i={} pos={}", i, pos);
                assert_eq!(a[2], idx[2], "i={} pos={}", i, pos);
                assert_eq!(a[3], idx[3], "i={} pos={}", i, pos);
            }
        }
    }

    // ── 2-ary standard ──

    #[test]
    fn standard_2ary_fingerprint_non_zero() {
        for i in 0u32..1000 {
            let (fingerprint, _) = hash_item_standard_2ary(&i.to_le_bytes(), 128, 12);
            assert_ne!(fingerprint, 0);
        }
    }

    #[test]
    fn standard_2ary_index_ranges() {
        let num_buckets = 128u32;
        for i in 0u32..1000 {
            let (_, idx) = hash_item_standard_2ary(&i.to_le_bytes(), num_buckets, 12);
            assert!(idx[0] < num_buckets);
            assert!(idx[1] < num_buckets);
        }
    }

    #[test]
    fn standard_2ary_all_indices_round_trip() {
        let num_buckets = 256u32;
        for i in 0u32..1000 {
            let (fingerprint, idx) = hash_item_standard_2ary(&i.to_le_bytes(), num_buckets, 16);
            // TEST-CHANGE: position param removed; result order depends on which endpoint
            // cur_index is, so use sort-and-compare instead of positional assertions.
            for &start in &[idx[0], idx[1]] {
                let a = all_indices_standard_2ary(start, fingerprint, num_buckets);
                let mut got = [a[0], a[1]];
                got.sort();
                let mut exp = [idx[0], idx[1]];
                exp.sort();
                assert_eq!(got, exp, "i={}", i);
            }
        }
    }

    // ── 3-ary standard ──

    #[test]
    fn standard_3ary_index_ranges() {
        let num_buckets = 243u32;
        for i in 0u32..1000 {
            let (_, idx) = hash_item_standard_3ary(&i.to_le_bytes(), num_buckets, 12);
            assert!(idx[0] < num_buckets);
            assert!(idx[1] < num_buckets);
            assert!(idx[2] < num_buckets);
        }
    }

    #[test]
    fn standard_3ary_all_indices_round_trip() {
        let num_buckets = 243u32;
        for i in 0u32..1000 {
            let (fingerprint, idx) = hash_item_standard_3ary(&i.to_le_bytes(), num_buckets, 16);
            for pos in 0..3 {
                let a = all_indices_standard_3ary(idx[pos], fingerprint, num_buckets);
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
    fn fingerprint_hash_mod_range() {
        for range in [3u32, 9, 27, 243] {
            for fp in 1u32..=100 {
                assert!(fingerprint_hash_mod(fp, range) < range);
            }
        }
    }

    // ── 4-ary standard ──

    #[test]
    fn standard_4ary_index_ranges() {
        let num_buckets = 256u32;
        for i in 0u32..1000 {
            let (_, idx) = hash_item_standard_4ary(&i.to_le_bytes(), num_buckets, 12);
            for j in 0..4 {
                assert!(idx[j] < num_buckets);
            }
        }
    }

    #[test]
    fn standard_4ary_all_indices_round_trip() {
        let num_buckets = 256u32;
        for i in 0u32..1000 {
            let (fingerprint, idx) = hash_item_standard_4ary(&i.to_le_bytes(), num_buckets, 16);
            for pos in 0..4 {
                let a = all_indices_standard_4ary(idx[pos], fingerprint, num_buckets);
                let mut expected = [idx[0], idx[1], idx[2], idx[3]];
                expected.sort();
                let mut got = [a[0], a[1], a[2], a[3]];
                got.sort();
                assert_eq!(got, expected, "i={} pos={}", i, pos);
            }
        }
    }

    // ── Fingerprint hash independence ──

    #[test]
    fn fingerprint_hashes_are_different() {
        // Ensure h1, h2, h3 produce different outputs for typical fingerprints
        let range = 256u32;
        let mut same_count = 0;
        for fp in 1u32..1000 {
            let h1 = fingerprint_hash1(fp, range);
            let h2 = fingerprint_hash2(fp, range);
            let h3 = fingerprint_hash3(fp, range);
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

    // ── extract_fingerprint ──

    #[test]
    fn extract_fingerprint_never_zero() {
        // Craft a hash whose lower bits are 0 with various masks
        let h_zero_low: u64 = 0xDEAD_BEEF_0000_0000;
        assert_eq!(extract_fingerprint(h_zero_low, 12), 1);
        assert_eq!(extract_fingerprint(h_zero_low, 32), 1);
    }

    #[test]
    fn extract_fingerprint_masks_correctly() {
        let h: u64 = 0x0000_0000_FFFF_FFFF;
        assert_eq!(extract_fingerprint(h, 8), 0xFF);
        assert_eq!(extract_fingerprint(h, 12), 0xFFF);
        assert_eq!(extract_fingerprint(h, 32), 0xFFFF_FFFF);
    }

    // ── xor3 ──
    //
    // `xor3` is a chunked-table rewrite of a base-3 digit loop. The loop is kept
    // here as the oracle: every test below asserts the table agrees with it, so
    // the two can never silently diverge. Note the `place > 1_000_000_000` cutoff
    // — the oracle stops after digit 19, and `xor3` must match that on every u32
    // (see `xor3`'s `# Digit range`).

    /// The original digit loop, verbatim. Reference semantics for `xor3`.
    fn xor3_reference(a: u32, b: u32) -> u32 {
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
            if place > 1_000_000_000 {
                break;
            }
            place *= 3;
        }
        result
    }

    /// Deterministic xorshift64 — a fixed seed keeps a failure reproducible.
    fn xorshift(state: &mut u64) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state as u32
    }

    #[test]
    fn xor3_matches_reference_exhaustively_over_low_digits() {
        // Every (digit_a, digit_b) pair in digits 0..=5, which spans the first
        // chunk boundary (digits 0..=3 | 4..=7) where a chunked rewrite would break.
        let n = 3u32.pow(6);
        for a in 0..n {
            for b in 0..n {
                assert_eq!(xor3(a, b), xor3_reference(a, b), "a={a} b={b}");
            }
        }
    }

    #[test]
    fn xor3_matches_reference_over_standard_3ary_range() {
        // Every index the standard 3-ary filter can hold, against the `h` values
        // `fingerprint_hash_mod` actually produces for it.
        let num_buckets = 3u32.pow(13);
        for a in 0..num_buckets {
            let h = fingerprint_hash_mod(a | 1, num_buckets);
            assert_eq!(xor3(a, h), xor3_reference(a, h), "a={a} h={h}");
        }
    }

    #[test]
    fn xor3_matches_reference_over_full_u32_range() {
        // Reaches the >= 3^20 inputs where digit 20 exists and both must drop it.
        let mut state = 0x243F_6A88_85A3_08D3;
        for _ in 0..200_000 {
            let (a, b) = (xorshift(&mut state), xorshift(&mut state));
            assert_eq!(xor3(a, b), xor3_reference(a, b), "a={a} b={b}");
        }
    }

    #[test]
    fn xor3_matches_reference_at_boundaries() {
        let corners = [
            0,
            1,
            2,
            80,
            81,
            242,
            243,
            3u32.pow(13) - 1,
            3u32.pow(13),
            3u32.pow(19),
            3u32.pow(20) - 1,
            3u32.pow(20),
            u32::MAX - 1,
            u32::MAX,
        ];
        for &a in &corners {
            for &b in &corners {
                assert_eq!(xor3(a, b), xor3_reference(a, b), "a={a} b={b}");
            }
        }
    }

    #[test]
    fn xor3_result_never_exceeds_3_pow_20() {
        // Digits 0..=19 each contribute at most 2 * 3^k, so the sum is 3^20 - 1.
        // Guards the `c0 + c1 * 81 + ...` reassembly against u32 overflow.
        let mut state = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..100_000 {
            let (a, b) = (xorshift(&mut state), xorshift(&mut state));
            assert!(xor3(a, b) < 3u32.pow(20), "a={a} b={b}");
        }
    }

    #[test]
    fn xor3_cycles_after_three_applications() {
        // The property the d-ary construction rests on, and the reason
        // `all_indices_standard_3ary` can rebuild the candidate set from any of them.
        let num_buckets = 3u32.pow(13);
        for a in (0..num_buckets).step_by(37) {
            let h = fingerprint_hash_mod(a | 1, num_buckets);
            assert_eq!(xor3(xor3(xor3(a, h), h), h), a, "a={a} h={h}");
        }
    }
}
