//! Bit-packed slot storage primitive and the fingerprint-typed wrappers.
//!
//! # Overview
//!
//! Three types live here:
//!
//! - [`DataLayout`] is the raw bit-packed array. Each slot holds a value of
//!   `data_size_in_bit` bits, packed contiguously LSB-first into a flat `Vec<u8>`
//!   buffer. The slot width is uncapped — it can be a single bit, 32 bits, or
//!   thousands of bits — so the same primitive backs both the cuckoo filter (which
//!   uses ≤ 32-bit fingerprints) and the `CuckooKVStore` (which stores
//!   `fingerprint ‖ value` per slot, with value potentially very wide).
//!
//! - [`FingerprintTable`] is a thin wrapper around `DataLayout` that fixes the slot
//!   width at `fingerprint_bits ∈ 1..=32`, exposes a `u32`-typed read/write API, and
//!   implements the bucket-scan operations (`find`, `insert`, `delete`) on top of
//!   `DataLayout`'s single-slot read/write.
//!
//! - [`FingerprintValueTable`] is a sibling wrapper that widens each slot to
//!   `fingerprint_bits + value_bits` and stores `fingerprint ‖ value` per slot.
//!   Used by `CuckooKVStore` as the backing array for the PIR scheme.
//!
//! # Memory layout
//!
//! Slots are packed contiguously, LSB-first within each byte and across byte
//! boundaries. The bit offset of slot `(bucket, slot)` is
//! `(bucket * slots_per_bucket + slot) * data_size_in_bit`. Eight bytes plus
//! `slot_size_in_bytes()` of trailing padding are appended to the allocation so the
//! per-byte shift-and-OR loop can always read one byte past the slot's last byte
//! (needed for misaligned reads at the very last slot).
//!
//! # Empty-slot convention
//!
//! `DataLayout` itself does not impose any rule on slot values — a slot of all-zero
//! bytes is just a slot of all-zero bytes. The "all-zeros means empty" convention
//! lives in the wrappers: `FingerprintTable` checks `read == 0` to find an empty
//! slot, and `FingerprintValueTable` checks `read_fingerprint == 0`. The wrappers
//! also enforce their own non-zero invariants (e.g., `FingerprintTable` panics on
//! `fingerprint == 0` in `insert` / `delete`).
//!
//! # Security
//!
//! No secret data flows through this module. Reads and writes do not branch on slot
//! contents — only on the bucket/slot indices and the bit-shift derived from them
//! — so there is no value-dependent timing leak.
//!
//! [`CuckooFilter`]: crate::CuckooFilter

// DECISION: the bytewise read/write loops use `for i in 0..n` because `i` indexes
// both `data[byte_pos + i]` *and* `out[i]` / `src[i]` (and conditionally
// `data[byte_pos + i + 1]`) per iteration. `enumerate()` would only yield one of
// these, so explicit indexing reads more clearly than a zipped-iterator chain.
#![allow(clippy::needless_range_loop)]

// ─── DataLayout ─────────────────────────────────────────────────────────────

/// Bit-packed storage for `num_buckets * slots_per_bucket` slots, each
/// `data_size_in_bit` bits wide.
///
/// The slot width is unbounded (subject only to memory). Read and write operate on
/// one slot at a time using caller-supplied byte buffers of length
/// [`slot_size_in_bytes`](Self::slot_size_in_bytes); the internal algorithm is a
/// per-byte shift-and-OR that works for any width without a `u64` / `u128` cap.
///
/// `DataLayout` deliberately exposes no "empty value", "non-zero", or bucket-scan
/// helpers; those belong to wrappers (see [`FingerprintTable`] for the
/// fingerprint-typed wrapper).
///
/// # Layout
///
/// Slots are LSB-first packed across bytes. With `data_size_in_bit = 12`,
/// `slots_per_bucket = 4`:
///
/// ```text
/// byte:   |   0    |   1    |   2    |   3    |   4    | ...
/// bits:   |76543210|76543210|76543210|76543210|76543210|
/// slot 0:  <----- 12 ----->                              bits  0..11
/// slot 1:                       <----- 12 ----->          bits 12..23
/// slot 2:                                        <-- 12   bits 24..35 (crosses byte 3 → 4)
/// ```
///
/// # Allocation
///
/// `data.len() == ceil(num_buckets * slots_per_bucket * data_size_in_bit / 8)
///                + slot_size_in_bytes() + 8`.
///
/// The trailing padding lets the misaligned bytewise loop read one byte past each
/// slot's nominal last byte (needed at the very last slot in the buffer).
#[derive(Clone)]
pub(crate) struct DataLayout {
    /// Raw packed-bit byte buffer (with trailing padding).
    data: Vec<u8>,
    /// Total number of buckets.
    num_buckets: u32,
    /// Slots per bucket.
    slots_per_bucket: u32,
    /// Width of each slot in bits.
    data_size_in_bit: u32,
}

impl DataLayout {
    /// Construct a zero-initialised layout.
    ///
    /// # Arguments
    ///
    /// - `num_buckets`     — total number of buckets.
    /// - `slots_per_bucket` — slots per bucket.
    /// - `data_size_in_bit` — bit width of every slot. Must be ≥ 1.
    ///
    /// # Panics
    ///
    /// Panics if `data_size_in_bit == 0`, or if the product
    /// `num_buckets * slots_per_bucket * data_size_in_bit` overflows `u64`.
    pub fn new(num_buckets: u32, slots_per_bucket: u32, data_size_in_bit: u32) -> Self {
        assert!(data_size_in_bit > 0, "data_size_in_bit must be >= 1");
        let total_bits = (num_buckets as u64)
            .checked_mul(slots_per_bucket as u64)
            .and_then(|x| x.checked_mul(data_size_in_bit as u64))
            .expect("num_buckets * slots_per_bucket * data_size_in_bit overflows u64");
        let total_bytes = total_bits.div_ceil(8) as usize;
        let slot_bytes = (data_size_in_bit as usize).div_ceil(8);
        DataLayout {
            data: vec![0u8; total_bytes + slot_bytes + 8],
            num_buckets,
            slots_per_bucket,
            data_size_in_bit,
        }
    }

    /// Total number of buckets.
    #[inline]
    pub fn num_buckets(&self) -> u32 {
        self.num_buckets
    }

    /// Slots per bucket.
    #[inline]
    pub fn slots_per_bucket(&self) -> u32 {
        self.slots_per_bucket
    }

    /// Width of each slot in bits.
    #[inline]
    pub fn data_size_in_bit(&self) -> u32 {
        self.data_size_in_bit
    }

    /// Logical byte size of the packed storage, *excluding* the trailing padding.
    ///
    /// `ceil(num_buckets * slots_per_bucket * data_size_in_bit / 8)`. Used by callers
    /// that want to report memory footprint without counting implementation overhead.
    pub fn size_in_bytes(&self) -> usize {
        let total_bits =
            self.num_buckets as u64 * self.slots_per_bucket as u64 * self.data_size_in_bit as u64;
        total_bits.div_ceil(8) as usize
    }

    /// Number of bytes needed to hold one slot's value: `ceil(data_size_in_bit / 8)`.
    ///
    /// Caller-supplied buffers passed to [`read`](Self::read) and [`write`](Self::write)
    /// must have exactly this length.
    #[inline]
    pub fn slot_size_in_bytes(&self) -> usize {
        (self.data_size_in_bit as usize).div_ceil(8)
    }

    /// Compute the bit offset of slot `(bucket, slot)` from the start of `data`.
    #[inline]
    fn bit_offset(&self, bucket: u32, slot: u32) -> u64 {
        (bucket as u64 * self.slots_per_bucket as u64 + slot as u64) * self.data_size_in_bit as u64
    }

    /// Read the value at `(bucket, slot)` into `out`.
    ///
    /// `out` must have length [`slot_size_in_bytes`](Self::slot_size_in_bytes). The
    /// returned bytes are LSB-first packed; bits above `data_size_in_bit` in the last
    /// output byte are zeroed.
    ///
    /// Thin wrapper over [`read_partial`](Self::read_partial) covering the full slot.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, `slot >= slots_per_bucket`, or
    /// `out.len() != slot_size_in_bytes()`.
    pub fn read(&self, bucket: u32, slot: u32, out: &mut [u8]) {
        let n = self.slot_size_in_bytes();
        assert_eq!(out.len(), n, "out buffer length must equal slot_size_in_bytes ({n})");
        self.read_partial(bucket, slot, 0, self.data_size_in_bit, out);
    }

    /// Write the value in `src` to slot `(bucket, slot)`.
    ///
    /// `src` must have length [`slot_size_in_bytes`](Self::slot_size_in_bytes). Bits
    /// above `data_size_in_bit` in `src[n - 1]` are masked off before writing, so they
    /// cannot bleed into the next slot.
    ///
    /// Thin wrapper over [`write_partial`](Self::write_partial) covering the full slot.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, `slot >= slots_per_bucket`, or
    /// `src.len() != slot_size_in_bytes()`.
    pub fn write(&mut self, bucket: u32, slot: u32, src: &[u8]) {
        let n = self.slot_size_in_bytes();
        assert_eq!(src.len(), n, "src buffer length must equal slot_size_in_bytes ({n})");
        self.write_partial(bucket, slot, 0, self.data_size_in_bit, src);
    }

    /// Read `n_bits` bits starting at bit `bit_offset` within slot `(bucket, slot)`.
    ///
    /// `out` must have length `ceil(n_bits / 8)`. Output is LSB-first packed; bits
    /// above `n_bits` in the last byte are zeroed.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, `slot >= slots_per_bucket`,
    /// `bit_offset + n_bits > data_size_in_bit`, `n_bits == 0`, or
    /// `out.len() != ceil(n_bits / 8)`.
    pub fn read_partial(
        &self,
        bucket: u32,
        slot: u32,
        bit_offset: u32,
        n_bits: u32,
        out: &mut [u8],
    ) {
        assert!(
            bucket < self.num_buckets && slot < self.slots_per_bucket,
            "(bucket, slot) out of range: ({bucket}, {slot}) not in [0, {}) x [0, {})",
            self.num_buckets,
            self.slots_per_bucket,
        );
        assert!(n_bits >= 1, "n_bits must be >= 1");
        assert!(
            bit_offset + n_bits <= self.data_size_in_bit,
            "bit_offset + n_bits ({}) exceeds data_size_in_bit ({})",
            bit_offset + n_bits,
            self.data_size_in_bit,
        );
        let n = (n_bits as usize).div_ceil(8);
        assert_eq!(out.len(), n, "out buffer length must equal ceil(n_bits / 8) ({n})");

        let start_bit = self.bit_offset(bucket, slot) + bit_offset as u64;
        let byte_pos = (start_bit / 8) as usize;
        let bit_shift = (start_bit % 8) as u32;
        let n_bits_us = n_bits as usize;
        let touched = (bit_shift as usize + n_bits_us).div_ceil(8);

        if bit_shift == 0 {
            out.copy_from_slice(&self.data[byte_pos..byte_pos + n]);
        } else {
            let inv_shift = 8 - bit_shift;
            for i in 0..n {
                let lo = self.data[byte_pos + i] >> bit_shift;
                let hi = if i + 1 < touched {
                    self.data[byte_pos + i + 1] << inv_shift
                } else {
                    0
                };
                out[i] = lo | hi;
            }
        }

        // Zero bits above `n_bits` in the last output byte.
        let last_bits = n_bits_us - (n - 1) * 8;
        if last_bits < 8 {
            out[n - 1] &= (1u8 << last_bits) - 1;
        }
    }

    /// Write `n_bits` bits from `src` into slot `(bucket, slot)` starting at bit
    /// `bit_offset` within the slot.
    ///
    /// Only the `n_bits` range is touched: bits before and after that range within
    /// the slot are preserved. Bits above `n_bits` in `src[n - 1]` are masked off
    /// before writing, so they cannot bleed past the written range.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, `slot >= slots_per_bucket`,
    /// `bit_offset + n_bits > data_size_in_bit`, `n_bits == 0`, or
    /// `src.len() != ceil(n_bits / 8)`.
    pub fn write_partial(
        &mut self,
        bucket: u32,
        slot: u32,
        bit_offset: u32,
        n_bits: u32,
        src: &[u8],
    ) {
        assert!(
            bucket < self.num_buckets && slot < self.slots_per_bucket,
            "(bucket, slot) out of range: ({bucket}, {slot}) not in [0, {}) x [0, {})",
            self.num_buckets,
            self.slots_per_bucket,
        );
        assert!(n_bits >= 1, "n_bits must be >= 1");
        assert!(
            bit_offset + n_bits <= self.data_size_in_bit,
            "bit_offset + n_bits ({}) exceeds data_size_in_bit ({})",
            bit_offset + n_bits,
            self.data_size_in_bit,
        );
        let n = (n_bits as usize).div_ceil(8);
        assert_eq!(src.len(), n, "src buffer length must equal ceil(n_bits / 8) ({n})");

        let start_bit = self.bit_offset(bucket, slot) + bit_offset as u64;
        let byte_pos = (start_bit / 8) as usize;
        let bit_shift = (start_bit % 8) as u32;
        let n_bits_us = n_bits as usize;
        let touched = (bit_shift as usize + n_bits_us).div_ceil(8);

        // Mask `src[n - 1]` so bits above `n_bits` cannot bleed past the written range.
        let last_bits = n_bits_us - (n - 1) * 8;
        let src_last_mask: u8 = if last_bits == 8 { 0xFF } else { (1u8 << last_bits) - 1 };

        // Step 1 — clear the destination bits within [bit_shift, bit_shift + n_bits).
        if touched == 1 {
            let mask = (((1u16 << n_bits_us) - 1) << bit_shift) as u8;
            self.data[byte_pos] &= !mask;
        } else {
            // First byte: keep low `bit_shift` bits, clear the high bits.
            self.data[byte_pos] &= (1u8 << bit_shift).wrapping_sub(1);
            // Middle bytes: clear entirely.
            for i in 1..touched - 1 {
                self.data[byte_pos + i] = 0;
            }
            // Last byte: clear the low bits the range writes into.
            let last_end = bit_shift as usize + n_bits_us - (touched - 1) * 8; // 1..=8
            let clear_mask: u8 =
                if last_end == 8 { 0xFF } else { (1u8 << last_end) - 1 };
            self.data[byte_pos + touched - 1] &= !clear_mask;
        }

        // Step 2 — OR in the source bits.
        for i in 0..n {
            let s = if i == n - 1 { src[i] & src_last_mask } else { src[i] };
            self.data[byte_pos + i] |= s.wrapping_shl(bit_shift);
            if bit_shift > 0 && i + 1 < touched {
                self.data[byte_pos + i + 1] |= s.wrapping_shr(8 - bit_shift);
            }
        }
    }
}

// ─── FingerprintTable ───────────────────────────────────────────────────────

/// Bit-packed fingerprint storage for a cuckoo filter.
///
/// Thin wrapper over [`DataLayout`] that fixes the slot value type to `u32`
/// (with widths in `1..=32` bits) and provides the bucket-scoped operations
/// (`contain`, `find`, `insert`, `delete`) the filter needs.
///
/// # Invariants (caller-enforced)
///
/// - `fingerprint_bits ∈ 1..=32` — asserted in [`new`](Self::new) at the storage
///   level. The filter-level `validate_common_params` (in `filter.rs`) enforces a
///   *higher* lower bound (`⌊log2(arity·bucket_size)⌋+1`, ≥ 2) for false-positive-
///   rate reasons. The two checks are intentionally different: the storage minimum
///   of 1 is the structural requirement; the filter minimum is a FPR policy.
/// - Fingerprint value `0` is reserved to mean "empty slot"; the hash layer must
///   never produce `0` for a real key.
///   [`insert`](Self::insert) and [`delete`](Self::delete) panic on a zero input.
///   [`read`](Self::read) and [`write`](Self::write) accept `0` so callers can
///   initialise / clear slots.
/// - `bucket < num_buckets` and `slot < slots_per_bucket` are checked by the
///   underlying [`DataLayout::read`] / [`DataLayout::write`], which panic out of
///   range with an `out of range` message.
#[derive(Clone)]
pub(crate) struct FingerprintTable {
    layout: DataLayout,
}

impl FingerprintTable {
    /// Create a fingerprint table.
    ///
    /// # Arguments
    ///
    /// - `num_buckets`      — total number of buckets.
    /// - `slots_per_bucket` — slots per bucket.
    /// - `fingerprint_bits` — fingerprint bit width. Must be in `1..=32` (storage-level
    ///   minimum). The filter-level `validate_common_params` enforces a higher lower
    ///   bound for FPR reasons; these two checks are complementary, not redundant.
    ///
    /// # Panics
    ///
    /// Panics if `fingerprint_bits` is not in `1..=32`.
    pub fn new(num_buckets: u32, slots_per_bucket: u32, fingerprint_bits: u32) -> Self {
        assert!(
            (1..=32).contains(&fingerprint_bits),
            "fingerprint_bits must be in 1..=32, got {fingerprint_bits}"
        );
        FingerprintTable {
            layout: DataLayout::new(num_buckets, slots_per_bucket, fingerprint_bits),
        }
    }

    /// Total number of buckets.
    #[inline]
    pub fn num_buckets(&self) -> u32 {
        self.layout.num_buckets()
    }

    /// Slots per bucket.
    #[inline]
    pub fn slots_per_bucket(&self) -> u32 {
        self.layout.slots_per_bucket()
    }

    /// Fingerprint bit width.
    #[inline]
    pub fn fingerprint_bits(&self) -> u32 {
        self.layout.data_size_in_bit()
    }

    /// Logical byte size of the underlying storage, excluding alignment padding.
    pub fn size_in_bytes(&self) -> usize {
        self.layout.size_in_bytes()
    }

    /// Read the fingerprint stored at `(bucket, slot)`.
    ///
    /// A return value of `0` means the slot is empty (the "empty = 0" invariant is
    /// enforced by [`insert`](Self::insert)).
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets` or `slot >= slots_per_bucket`.
    #[inline]
    pub fn read(&self, bucket: u32, slot: u32) -> u32 {
        let n = self.layout.slot_size_in_bytes();
        let mut buf = [0u8; 4];
        self.layout.read(bucket, slot, &mut buf[..n]);
        u32::from_le_bytes(buf)
    }

    /// Write `fingerprint` into the slot at `(bucket, slot)`.
    ///
    /// Only the lower `fingerprint_bits` bits of `fingerprint` are stored; higher
    /// bits are silently dropped. Pass `0` to mark the slot as empty.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets` or `slot >= slots_per_bucket`.
    #[inline]
    pub fn write(&mut self, bucket: u32, slot: u32, fingerprint: u32) {
        let n = self.layout.slot_size_in_bytes();
        let bytes = fingerprint.to_le_bytes();
        self.layout.write(bucket, slot, &bytes[..n]);
    }

    /// Return `true` if `fingerprint` is present in any slot of `bucket`.
    ///
    /// Used by `CuckooFilter::contain` to test membership without needing the slot
    /// index. Scans every slot in the bucket linearly.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`.
    pub fn contain(&self, bucket: u32, fingerprint: u32) -> bool {
        for slot in 0..self.layout.slots_per_bucket() {
            if self.read(bucket, slot) == fingerprint {
                return true;
            }
        }
        false
    }

    /// Find the slot index of `fingerprint` within `bucket`.
    ///
    /// Returns the index of the first matching slot, or `None` if not found.
    /// Kept as a public helper symmetric with [`contain`](Self::contain);
    /// `FingerprintValueTable::find` is the analogous active call site.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`.
    #[allow(dead_code)]
    pub fn find(&self, bucket: u32, fingerprint: u32) -> Option<u32> {
        (0..self.layout.slots_per_bucket())
            .find(|&slot| self.read(bucket, slot) == fingerprint)
    }

    /// Delete the first occurrence of `fingerprint` from `bucket` by zeroing its slot.
    ///
    /// Returns `true` if `fingerprint` was found and cleared, `false` otherwise.
    /// Only the first matching slot is cleared; duplicates (from multiple insertions
    /// of the same item) must be deleted one at a time.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, or if `fingerprint == 0` (which would be
    /// indistinguishable from an empty slot under the "empty = 0" invariant).
    pub fn delete(&mut self, bucket: u32, fingerprint: u32) -> bool {
        assert!(fingerprint != 0, "fingerprint cannot be zero");
        for slot in 0..self.layout.slots_per_bucket() {
            if self.read(bucket, slot) == fingerprint {
                self.write(bucket, slot, 0);
                return true;
            }
        }
        false
    }

    /// Insert `fingerprint` into the first empty slot of `bucket`.
    ///
    /// Returns `Some(slot)` with the slot index that was used, or `None` if all slots
    /// in `bucket` are occupied.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, or if `fingerprint == 0` (zero is reserved
    /// for empty slots).
    pub fn insert(&mut self, bucket: u32, fingerprint: u32) -> Option<u32> {
        assert!(fingerprint != 0, "fingerprint cannot be zero");
        for slot in 0..self.layout.slots_per_bucket() {
            if self.read(bucket, slot) == 0 {
                self.write(bucket, slot, fingerprint);
                return Some(slot);
            }
        }
        None
    }
}

// ─── FingerprintValueTable ──────────────────────────────────────────────────

/// Bit-packed storage for `fingerprint ‖ value` entries in a cuckoo KV store.
///
/// Sibling wrapper to [`FingerprintTable`], both backed by [`DataLayout`]. Each slot
/// stores `fingerprint_bits + value_bits` bits laid out as:
///
/// ```text
/// bit:  0 ............ F-1   F .................. F+V-1
///       [ fingerprint     ] [ value                   ]
/// ```
///
/// where `F = fingerprint_bits`, `V = value_bits`. The fingerprint occupies the low
/// bits and the value occupies the high bits within each `DataLayout` slot.
///
/// # Invariants
///
/// - `fingerprint_bits ∈ 1..=32` — enforced in [`new`](Self::new).
/// - `value_bits ≥ 1` — enforced in [`new`](Self::new). A zero-*content* value is
///   permitted; only a zero-*width* value is rejected.
/// - Fingerprint value `0` means "slot empty". [`insert`](Self::insert) and
///   [`delete`](Self::delete) panic on `fingerprint == 0`. [`write`](Self::write)
///   accepts `0` so callers can zero out a slot directly.
/// - Value buffers must have length `ceil(value_bits / 8)` everywhere a value is
///   passed in or out.
/// - [`insert`](Self::insert) does **not** deduplicate: it inserts into the first
///   empty slot regardless of whether the fingerprint already exists in the bucket.
///   Caller-level dedup (in `CuckooKVStore`) is left for the insertion layer.
pub(crate) struct FingerprintValueTable {
    layout: DataLayout,
    fingerprint_bits: u32,
    value_bits: u32,
    /// Pre-allocated all-zero buffer of `slot_size_in_bytes` bytes, used by `delete`
    /// to clear an entry without allocating per call.
    zero_buf: Vec<u8>,
}

impl FingerprintValueTable {
    /// Create a fingerprint-value table.
    ///
    /// # Arguments
    ///
    /// - `num_buckets`      — total number of buckets.
    /// - `slots_per_bucket` — slots per bucket.
    /// - `fingerprint_bits` — fingerprint bit width. Must be in `1..=32`.
    /// - `value_bits`       — value bit width. Must be ≥ 1.
    ///
    /// # Panics
    ///
    /// Panics if `fingerprint_bits` is not in `1..=32`, or if `value_bits == 0`.
    pub fn new(
        num_buckets: u32,
        slots_per_bucket: u32,
        fingerprint_bits: u32,
        value_bits: u32,
    ) -> Self {
        assert!(
            (1..=32).contains(&fingerprint_bits),
            "fingerprint_bits must be in 1..=32, got {fingerprint_bits}"
        );
        assert!(value_bits >= 1, "value_bits must be >= 1, got {value_bits}");
        let entry_bits = fingerprint_bits + value_bits;
        let layout = DataLayout::new(num_buckets, slots_per_bucket, entry_bits);
        let zero_buf = vec![0u8; layout.slot_size_in_bytes()];
        FingerprintValueTable {
            layout,
            fingerprint_bits,
            value_bits,
            zero_buf,
        }
    }

    /// Total number of buckets.
    #[inline]
    pub fn num_buckets(&self) -> u32 {
        self.layout.num_buckets()
    }

    /// Slots per bucket.
    #[inline]
    pub fn slots_per_bucket(&self) -> u32 {
        self.layout.slots_per_bucket()
    }

    /// Fingerprint bit width.
    #[inline]
    pub fn fingerprint_bits(&self) -> u32 {
        self.fingerprint_bits
    }

    /// Value bit width.
    ///
    /// Inspection helper for callers (e.g., `ikpir-server`) that need to size value
    /// buffers without knowing the slot layout. Not used internally by [`CuckooKVStore`].
    ///
    /// [`CuckooKVStore`]: crate::CuckooKVStore
    #[inline]
    #[allow(dead_code)]
    pub fn value_bits(&self) -> u32 {
        self.value_bits
    }

    /// Entry bit width: `fingerprint_bits + value_bits`.
    ///
    /// Inspection helper symmetric with [`fingerprint_bits`](Self::fingerprint_bits) and
    /// [`value_bits`](Self::value_bits). Not used internally.
    #[inline]
    #[allow(dead_code)]
    pub fn entry_bits(&self) -> u32 {
        self.fingerprint_bits + self.value_bits
    }

    /// Logical byte size of the underlying storage, excluding alignment padding.
    pub fn size_in_bytes(&self) -> usize {
        self.layout.size_in_bytes()
    }

    /// Number of bytes needed to hold one value: `ceil(value_bits / 8)`.
    #[inline]
    pub fn value_size_in_bytes(&self) -> usize {
        (self.value_bits as usize).div_ceil(8)
    }

    /// Read the fingerprint stored at `(bucket, slot)`.
    ///
    /// Returns `0` if the slot is empty.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets` or `slot >= slots_per_bucket`.
    pub fn read_fingerprint(&self, bucket: u32, slot: u32) -> u32 {
        let fp_bytes = (self.fingerprint_bits as usize).div_ceil(8);
        let mut buf = [0u8; 4];
        self.layout
            .read_partial(bucket, slot, 0, self.fingerprint_bits, &mut buf[..fp_bytes]);
        u32::from_le_bytes(buf)
    }

    /// Read the value stored at `(bucket, slot)` into `out`.
    ///
    /// `out` must have length [`value_size_in_bytes`](Self::value_size_in_bytes).
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, `slot >= slots_per_bucket`, or
    /// `out.len() != value_size_in_bytes()`.
    pub fn read_value(&self, bucket: u32, slot: u32, out: &mut [u8]) {
        let vsize = self.value_size_in_bytes();
        assert_eq!(
            out.len(),
            vsize,
            "out buffer length must equal value_size_in_bytes ({vsize})"
        );
        self.layout
            .read_partial(bucket, slot, self.fingerprint_bits, self.value_bits, out);
    }

    /// Write `fingerprint ‖ value` into slot `(bucket, slot)`.
    ///
    /// Pass `fingerprint = 0` and an all-zero value to mark a slot empty.
    ///
    /// `value` must have length [`value_size_in_bytes`](Self::value_size_in_bytes).
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, `slot >= slots_per_bucket`, or
    /// `value.len() != value_size_in_bytes()`.
    pub fn write(&mut self, bucket: u32, slot: u32, fingerprint: u32, value: &[u8]) {
        let vsize = self.value_size_in_bytes();
        assert_eq!(
            value.len(),
            vsize,
            "value buffer length must equal value_size_in_bytes ({vsize})"
        );
        let fp_bytes = (self.fingerprint_bits as usize).div_ceil(8);
        let fp_le = fingerprint.to_le_bytes();
        self.layout
            .write_partial(bucket, slot, 0, self.fingerprint_bits, &fp_le[..fp_bytes]);
        self.layout
            .write_partial(bucket, slot, self.fingerprint_bits, self.value_bits, value);
    }

    /// Return `true` if `fingerprint` is present in any slot of `bucket`.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`.
    pub fn contain(&self, bucket: u32, fingerprint: u32) -> bool {
        for slot in 0..self.layout.slots_per_bucket() {
            if self.read_fingerprint(bucket, slot) == fingerprint {
                return true;
            }
        }
        false
    }

    /// Find the slot index of `fingerprint` within `bucket`.
    ///
    /// Returns `Some(slot)` for the first matching slot. If the same fingerprint appears
    /// in multiple slots, the lowest slot index is returned.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`.
    pub fn find(&self, bucket: u32, fingerprint: u32) -> Option<u32> {
        (0..self.layout.slots_per_bucket())
            .find(|&slot| self.read_fingerprint(bucket, slot) == fingerprint)
    }

    /// Insert `fingerprint ‖ value` into the first empty slot of `bucket`.
    ///
    /// Returns `Some(slot)` with the slot index used, or `None` if the bucket is full.
    ///
    /// Does **not** deduplicate: inserts into the first empty slot regardless of whether
    /// `fingerprint` already exists in the bucket. Caller-level dedup (in `CuckooKVStore`)
    /// is expected before calling this method.
    ///
    /// `value` must have length [`value_size_in_bytes`](Self::value_size_in_bytes).
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, `fingerprint == 0`, or
    /// `value.len() != value_size_in_bytes()`.
    pub fn insert(&mut self, bucket: u32, fingerprint: u32, value: &[u8]) -> Option<u32> {
        assert!(fingerprint != 0, "fingerprint cannot be zero");
        let vsize = self.value_size_in_bytes();
        assert_eq!(
            value.len(),
            vsize,
            "value buffer length must equal value_size_in_bytes ({vsize})"
        );
        for slot in 0..self.layout.slots_per_bucket() {
            if self.read_fingerprint(bucket, slot) == 0 {
                self.write(bucket, slot, fingerprint, value);
                return Some(slot);
            }
        }
        None
    }

    /// Delete the first occurrence of `fingerprint` from `bucket` by zeroing its slot.
    ///
    /// Returns `true` if found and cleared, `false` if not found.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, or if `fingerprint == 0`.
    pub fn delete(&mut self, bucket: u32, fingerprint: u32) -> bool {
        assert!(fingerprint != 0, "fingerprint cannot be zero");
        for slot in 0..self.layout.slots_per_bucket() {
            if self.read_fingerprint(bucket, slot) == fingerprint {
                self.layout.write_partial(
                    bucket,
                    slot,
                    0,
                    self.fingerprint_bits + self.value_bits,
                    &self.zero_buf,
                );
                return true;
            }
        }
        false
    }

    /// Update the value for the first slot in `bucket` matching `fingerprint`.
    ///
    /// Returns `true` if found and updated, `false` if not found.
    ///
    /// Re-writes the fingerprint bits alongside the new value.
    ///
    /// `new_value` must have length [`value_size_in_bytes`](Self::value_size_in_bytes).
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets` or `new_value.len() != value_size_in_bytes()`.
    pub fn update_value(&mut self, bucket: u32, fingerprint: u32, new_value: &[u8]) -> bool {
        let vsize = self.value_size_in_bytes();
        assert_eq!(
            new_value.len(),
            vsize,
            "new_value buffer length must equal value_size_in_bytes ({vsize})"
        );
        if let Some(slot) = self.find(bucket, fingerprint) {
            self.layout
                .write_partial(bucket, slot, self.fingerprint_bits, self.value_bits, new_value);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── DataLayout direct tests ─────────────────────────────────────────

    fn roundtrip(num_buckets: u32, slots_per_bucket: u32, data_size_in_bit: u32, value: &[u8]) {
        let mut layout = DataLayout::new(num_buckets, slots_per_bucket, data_size_in_bit);
        let n = layout.slot_size_in_bytes();
        assert_eq!(value.len(), n);

        // Write to one slot, read back.
        layout.write(1, 0, value);
        let mut out = vec![0u8; n];
        layout.read(1, 0, &mut out);
        // Compare against `value` with the high bits of the last byte zeroed
        // (the read normalises that, so the write must produce the same).
        let mut expected = value.to_vec();
        let last_bits = data_size_in_bit as usize - (n - 1) * 8;
        if last_bits < 8 {
            expected[n - 1] &= (1u8 << last_bits) - 1;
        }
        assert_eq!(out, expected);

        // Neighbouring slots must remain zero.
        let mut neighbour = vec![0u8; n];
        layout.read(0, 0, &mut neighbour);
        assert!(neighbour.iter().all(|&b| b == 0), "neighbour slot was disturbed");
        if slots_per_bucket > 1 {
            layout.read(1, 1, &mut neighbour);
            assert!(neighbour.iter().all(|&b| b == 0), "next slot in bucket disturbed");
        }
    }

    #[test]
    fn datalayout_roundtrip_8bit() {
        roundtrip(4, 4, 8, &[0xAB]);
    }

    #[test]
    fn datalayout_roundtrip_12bit() {
        roundtrip(4, 4, 12, &[0xCD, 0x0A]);
    }

    #[test]
    fn datalayout_roundtrip_32bit() {
        roundtrip(4, 4, 32, &[0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn datalayout_roundtrip_64bit() {
        roundtrip(4, 4, 64, &[0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF]);
    }

    #[test]
    fn datalayout_roundtrip_100bit() {
        let mut v = vec![0u8; 13];
        for (i, b) in v.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        roundtrip(4, 4, 100, &v);
    }

    #[test]
    fn datalayout_roundtrip_512bit() {
        let v: Vec<u8> = (0u8..64).collect();
        roundtrip(4, 2, 512, &v);
    }

    #[test]
    fn datalayout_roundtrip_1024bit() {
        let v: Vec<u8> = (0..128).map(|i: usize| (i.wrapping_mul(31) ^ 0xA5) as u8).collect();
        roundtrip(4, 2, 1024, &v);
    }

    #[test]
    fn datalayout_all_misalignments_12bit() {
        // 12-bit slots in 1 bucket of 8 slots. The 8 slots cycle through every bit_shift
        // value (0, 12, 24, 36, 48, 60, 72, 84) mod 8 → 0, 4, 0, 4, 0, 4, 0, 4. Combined
        // with bit_shift values produced by inserting at neighbouring buckets, this
        // exercises both aligned and misaligned cases.
        let mut layout = DataLayout::new(2, 8, 12);
        let values: [u16; 16] = [
            0x000, 0x001, 0x002, 0xFFE, 0xFFF, 0xABC, 0x123, 0x7DE,
            0x000, 0x001, 0xAAA, 0x555, 0x800, 0x7FF, 0xC0D, 0x0E0,
        ];
        for (idx, &val) in values.iter().enumerate() {
            let bucket = (idx / 8) as u32;
            let slot = (idx % 8) as u32;
            let bytes = (val as u32).to_le_bytes();
            layout.write(bucket, slot, &bytes[..2]);
        }
        for (idx, &val) in values.iter().enumerate() {
            let bucket = (idx / 8) as u32;
            let slot = (idx % 8) as u32;
            let mut out = [0u8; 2];
            layout.read(bucket, slot, &mut out);
            let read = u16::from_le_bytes(out);
            assert_eq!(read, val, "mismatch at bucket={bucket}, slot={slot}");
        }
    }

    #[test]
    fn datalayout_value_zero_is_allowed() {
        // DataLayout has no "non-zero" rule. Writing then reading 0 must work.
        let mut layout = DataLayout::new(2, 2, 12);
        layout.write(0, 0, &[0u8, 0u8]);
        let mut out = [0u8; 2];
        layout.read(0, 0, &mut out);
        assert_eq!(out, [0u8, 0u8]);
    }

    #[test]
    fn datalayout_high_bits_in_src_are_masked() {
        // Write a value whose last byte has bits set above `data_size_in_bit`; verify
        // (a) the read returns only the lower `data_size_in_bit` bits, and (b) the
        // adjacent slot is untouched.
        let mut layout = DataLayout::new(1, 2, 12);
        // src has top 4 bits set in the second byte; only bits [8..12) are valid.
        layout.write(0, 0, &[0x55, 0xFF]);
        let mut out = [0u8; 2];
        layout.read(0, 0, &mut out);
        assert_eq!(out, [0x55, 0x0F]);
        // Slot 1 must remain zero — we must not have leaked the masked-off bits into it.
        layout.read(0, 1, &mut out);
        assert_eq!(out, [0u8, 0u8]);
    }

    #[test]
    fn datalayout_slot_size_in_bytes() {
        assert_eq!(DataLayout::new(1, 1, 1).slot_size_in_bytes(), 1);
        assert_eq!(DataLayout::new(1, 1, 8).slot_size_in_bytes(), 1);
        assert_eq!(DataLayout::new(1, 1, 9).slot_size_in_bytes(), 2);
        assert_eq!(DataLayout::new(1, 1, 12).slot_size_in_bytes(), 2);
        assert_eq!(DataLayout::new(1, 1, 32).slot_size_in_bytes(), 4);
        assert_eq!(DataLayout::new(1, 1, 1024).slot_size_in_bytes(), 128);
    }

    #[test]
    fn datalayout_size_in_bytes() {
        // 16 buckets × 4 slots × 12 bits = 768 bits = 96 bytes
        assert_eq!(DataLayout::new(16, 4, 12).size_in_bytes(), 96);
        // 8 × 2 × 16 = 256 bits = 32 bytes
        assert_eq!(DataLayout::new(8, 2, 16).size_in_bytes(), 32);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn datalayout_read_panics_on_oob() {
        let layout = DataLayout::new(2, 2, 8);
        let mut out = [0u8; 1];
        layout.read(2, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn datalayout_write_panics_on_oob() {
        let mut layout = DataLayout::new(2, 2, 8);
        layout.write(0, 2, &[1]);
    }

    #[test]
    #[should_panic(expected = "slot_size_in_bytes")]
    fn datalayout_read_panics_on_wrong_buffer_length() {
        let layout = DataLayout::new(2, 2, 12);
        let mut out = [0u8; 1]; // should be 2
        layout.read(0, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "slot_size_in_bytes")]
    fn datalayout_write_panics_on_wrong_buffer_length() {
        let mut layout = DataLayout::new(2, 2, 12);
        layout.write(0, 0, &[1u8]); // should be 2 bytes
    }

    #[test]
    #[should_panic(expected = "data_size_in_bit must be >= 1")]
    fn datalayout_new_panics_on_zero_width() {
        let _ = DataLayout::new(1, 1, 0);
    }

    // ─── FingerprintTable tests ──────────────────────────────────────────

    #[test]
    fn fingerprint_table_read_write_8bit() {
        let mut table = FingerprintTable::new(16, 4, 8);
        table.write(3, 2, 0xAB);
        assert_eq!(table.read(3, 2), 0xAB);
        assert_eq!(table.read(3, 1), 0);
    }

    #[test]
    fn fingerprint_table_read_write_12bit() {
        let mut table = FingerprintTable::new(16, 4, 12);
        table.write(5, 1, 0xFED);
        assert_eq!(table.read(5, 1), 0xFED);
        table.write(5, 2, 0x123);
        assert_eq!(table.read(5, 2), 0x123);
        assert_eq!(table.read(5, 1), 0xFED); // unchanged
    }

    #[test]
    fn fingerprint_table_read_write_16bit() {
        let mut table = FingerprintTable::new(8, 4, 16);
        table.write(7, 3, 0xBEEF);
        assert_eq!(table.read(7, 3), 0xBEEF);
    }

    #[test]
    fn fingerprint_table_read_write_32bit() {
        let mut table = FingerprintTable::new(4, 2, 32);
        table.write(1, 0, 0xDEADBEEF);
        assert_eq!(table.read(1, 0), 0xDEADBEEF);
    }

    #[test]
    fn fingerprint_table_contain_and_delete() {
        let mut table = FingerprintTable::new(8, 4, 12);
        table.write(2, 0, 100);
        table.write(2, 1, 200);
        table.write(2, 2, 300);
        assert!(table.contain(2, 200));
        assert!(!table.contain(2, 999));
        assert!(table.delete(2, 200));
        assert!(!table.contain(2, 200));
    }

    #[test]
    fn fingerprint_table_insert() {
        let mut table = FingerprintTable::new(4, 2, 8);
        assert_eq!(table.insert(0, 10), Some(0));
        assert_eq!(table.insert(0, 20), Some(1));
        assert_eq!(table.insert(0, 30), None); // bucket full
        assert!(table.contain(0, 10));
        assert!(table.contain(0, 20));
    }

    #[test]
    fn fingerprint_table_find() {
        let mut table = FingerprintTable::new(4, 4, 8);
        table.write(1, 0, 10);
        table.write(1, 2, 30);
        assert_eq!(table.find(1, 10), Some(0));
        assert_eq!(table.find(1, 30), Some(2));
        assert_eq!(table.find(1, 99), None);
    }

    #[test]
    fn fingerprint_table_delete_absent() {
        let mut table = FingerprintTable::new(4, 2, 8);
        assert!(!table.delete(0, 42));
    }

    #[test]
    fn fingerprint_table_size_in_bytes() {
        // 16 buckets × 4 slots × 12 bits = 768 bits = 96 bytes
        let table = FingerprintTable::new(16, 4, 12);
        assert_eq!(table.size_in_bytes(), 96);
        // 8 * 2 * 16 = 256 bits = 32 bytes
        let table2 = FingerprintTable::new(8, 2, 16);
        assert_eq!(table2.size_in_bytes(), 32);
    }

    // -------- FingerprintTable panic tests --------

    #[test]
    #[should_panic(expected = "out of range")]
    fn fingerprint_table_read_panics_on_bucket_oob() {
        let table = FingerprintTable::new(4, 2, 8);
        let _ = table.read(4, 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fingerprint_table_read_panics_on_slot_oob() {
        let table = FingerprintTable::new(4, 2, 8);
        let _ = table.read(0, 2);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fingerprint_table_read_panics_on_far_oob_bucket() {
        let table = FingerprintTable::new(4, 2, 8);
        let _ = table.read(u32::MAX, 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fingerprint_table_write_panics_on_bucket_oob() {
        let mut table = FingerprintTable::new(4, 2, 8);
        table.write(4, 0, 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fingerprint_table_write_panics_on_slot_oob() {
        let mut table = FingerprintTable::new(4, 2, 8);
        table.write(0, 2, 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fingerprint_table_contain_panics_on_bucket_oob() {
        let table = FingerprintTable::new(4, 2, 8);
        let _ = table.contain(4, 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fingerprint_table_find_panics_on_bucket_oob() {
        let table = FingerprintTable::new(4, 2, 8);
        let _ = table.find(4, 1);
    }

    #[test]
    #[should_panic(expected = "fingerprint cannot be zero")]
    fn fingerprint_table_delete_panics_on_zero_fingerprint() {
        let mut table = FingerprintTable::new(4, 2, 8);
        let _ = table.delete(0, 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fingerprint_table_delete_panics_on_bucket_oob() {
        let mut table = FingerprintTable::new(4, 2, 8);
        let _ = table.delete(4, 1);
    }

    #[test]
    #[should_panic(expected = "fingerprint cannot be zero")]
    fn fingerprint_table_insert_panics_on_zero_fingerprint() {
        let mut table = FingerprintTable::new(4, 2, 8);
        let _ = table.insert(0, 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fingerprint_table_insert_panics_on_bucket_oob() {
        let mut table = FingerprintTable::new(4, 2, 8);
        let _ = table.insert(4, 1);
    }

    #[test]
    #[should_panic(expected = "fingerprint_bits must be in 1..=32")]
    fn fingerprint_table_new_panics_on_zero_fp_bits() {
        let _ = FingerprintTable::new(4, 2, 0);
    }

    // ─── FingerprintValueTable tests ─────────────────────────────────────

    fn make_value(n: usize, seed: u8) -> Vec<u8> {
        (0..n).map(|i| (i as u8).wrapping_mul(37).wrapping_add(seed)).collect()
    }

    fn fvt_roundtrip(fp_bits: u32, value_bits: u32, fingerprint: u32, value: &[u8]) {
        let vsize = (value_bits as usize).div_ceil(8);
        assert_eq!(value.len(), vsize);

        let mut t = FingerprintValueTable::new(4, 4, fp_bits, value_bits);
        t.write(0, 0, fingerprint, value);

        assert_eq!(t.read_fingerprint(0, 0), fingerprint);
        let mut out = vec![0u8; vsize];
        t.read_value(0, 0, &mut out);

        // Mask the last value byte for comparison (same as DataLayout::read does).
        let mut expected = value.to_vec();
        let last_bits = (value_bits % 8) as usize;
        if last_bits != 0 {
            expected[vsize - 1] &= (1u8 << last_bits) - 1;
        }
        assert_eq!(out, expected, "fp_bits={fp_bits}, value_bits={value_bits}");

        // Neighbour slot must be untouched.
        assert_eq!(t.read_fingerprint(0, 1), 0);
        let mut neighbour = vec![0u8; vsize];
        t.read_value(0, 1, &mut neighbour);
        assert!(neighbour.iter().all(|&b| b == 0));
    }

    #[test]
    fn fvt_roundtrip_8_8() {
        fvt_roundtrip(8, 8, 0xAB, &[0xCD]);
    }

    #[test]
    fn fvt_roundtrip_12_64() {
        let v = make_value(8, 7);
        fvt_roundtrip(12, 64, 0xFED, &v);
    }

    #[test]
    fn fvt_roundtrip_16_1024() {
        let v = make_value(128, 13);
        fvt_roundtrip(16, 1024, 0xBEEF, &v);
    }

    #[test]
    fn fvt_roundtrip_12_100() {
        let v = make_value(13, 5);
        fvt_roundtrip(12, 100, 0xA5C, &v);
    }

    #[test]
    fn fvt_insert_contain_find_read_delete_lifecycle() {
        let vsize = 2usize; // value_bits=16
        let mut t = FingerprintValueTable::new(8, 4, 12, 16);
        let fp = 0x123u32;
        let val = [0xAB_u8, 0xCD];

        let slot = t.insert(0, fp, &val).expect("insert into empty bucket");
        assert!(t.contain(0, fp));
        assert_eq!(t.find(0, fp), Some(slot));

        let mut out = vec![0u8; vsize];
        t.read_value(0, slot, &mut out);
        assert_eq!(out, val);

        assert!(t.delete(0, fp));
        assert!(!t.contain(0, fp));
        assert_eq!(t.read_fingerprint(0, slot), 0);
    }

    #[test]
    fn fvt_update_value_preserves_fingerprint() {
        let mut t = FingerprintValueTable::new(8, 4, 12, 16);
        let fp = 0x456u32;
        let val1 = [0x11_u8, 0x22];
        let val2 = [0x33_u8, 0x44];

        t.insert(0, fp, &val1).unwrap();
        assert!(t.update_value(0, fp, &val2));

        assert_eq!(t.read_fingerprint(0, 0), fp);
        let mut out = vec![0u8; 2];
        t.read_value(0, 0, &mut out);
        assert_eq!(out, val2);
    }

    #[test]
    fn fvt_two_slots_same_fingerprint_find_returns_first() {
        let mut t = FingerprintValueTable::new(8, 4, 12, 16);
        let fp = 0x789u32;
        let val1 = [0x01_u8, 0x02];
        let val2 = [0x03_u8, 0x04];

        t.write(0, 0, fp, &val1);
        t.write(0, 2, fp, &val2);

        // find returns the lowest matching slot
        assert_eq!(t.find(0, fp), Some(0));
    }

    #[test]
    #[should_panic(expected = "fingerprint cannot be zero")]
    fn fvt_insert_panics_on_zero_fingerprint() {
        let mut t = FingerprintValueTable::new(4, 2, 8, 8);
        t.insert(0, 0, &[0u8]).unwrap();
    }

    #[test]
    #[should_panic(expected = "fingerprint cannot be zero")]
    fn fvt_delete_panics_on_zero_fingerprint() {
        let mut t = FingerprintValueTable::new(4, 2, 8, 8);
        t.delete(0, 0);
    }

    #[test]
    #[should_panic(expected = "value buffer length")]
    fn fvt_write_panics_on_wrong_value_length() {
        let mut t = FingerprintValueTable::new(4, 2, 8, 16); // value_size=2 bytes
        t.write(0, 0, 1, &[0u8]); // wrong: only 1 byte
    }

    #[test]
    #[should_panic(expected = "value_bits must be >= 1")]
    fn fvt_new_panics_on_zero_value_bits() {
        let _ = FingerprintValueTable::new(4, 2, 8, 0);
    }

    #[test]
    #[should_panic(expected = "fingerprint_bits must be in 1..=32")]
    fn fvt_new_panics_on_zero_fp_bits() {
        let _ = FingerprintValueTable::new(4, 2, 0, 8);
    }

    // ─── Functional gaps ─────────────────────────────────────────────────

    #[test]
    fn fvt_insert_returns_none_when_bucket_full() {
        let mut t = FingerprintValueTable::new(4, 2, 8, 8);
        assert_eq!(t.insert(0, 0x11, &[0xAA]), Some(0));
        assert_eq!(t.insert(0, 0x22, &[0xBB]), Some(1));
        assert_eq!(t.insert(0, 0x33, &[0xCC]), None);
        // Bucket 1 still empty.
        assert_eq!(t.insert(1, 0x44, &[0xDD]), Some(0));
    }

    #[test]
    fn fvt_delete_returns_false_on_absent() {
        let mut t = FingerprintValueTable::new(4, 2, 8, 8);
        t.insert(0, 0x11, &[0xAA]).unwrap();
        assert!(!t.delete(0, 0x99));
        // Original entry still present and intact.
        assert_eq!(t.read_fingerprint(0, 0), 0x11);
        let mut out = [0u8; 1];
        t.read_value(0, 0, &mut out);
        assert_eq!(out, [0xAA]);
    }

    #[test]
    fn fvt_update_value_returns_false_on_absent() {
        let mut t = FingerprintValueTable::new(4, 2, 8, 8);
        t.insert(0, 0x11, &[0xAA]).unwrap();
        assert!(!t.update_value(0, 0x99, &[0xBB]));
        // Original value unchanged.
        let mut out = [0u8; 1];
        t.read_value(0, 0, &mut out);
        assert_eq!(out, [0xAA]);
    }

    #[test]
    fn fvt_high_bits_in_value_are_masked() {
        // value_bits=12; the second value byte has bits set above bit 12.
        let mut t = FingerprintValueTable::new(2, 2, 8, 12);
        t.write(0, 0, 0xAB, &[0x55, 0xFF]);
        let mut out = [0u8; 2];
        t.read_value(0, 0, &mut out);
        assert_eq!(out, [0x55, 0x0F]);
        // Adjacent slot fingerprint must remain 0 — masked bits did not bleed in.
        assert_eq!(t.read_fingerprint(0, 1), 0);
    }

    #[test]
    fn fvt_multi_bucket_independence() {
        let mut t = FingerprintValueTable::new(4, 2, 12, 16);
        t.write(0, 0, 0x123, &[0xAA, 0xBB]);
        // Other buckets must remain zero.
        for b in 1..4 {
            for s in 0..2 {
                assert_eq!(t.read_fingerprint(b, s), 0);
                let mut out = [0u8; 2];
                t.read_value(b, s, &mut out);
                assert_eq!(out, [0u8; 2]);
            }
        }
    }

    #[test]
    fn fvt_full_bucket_then_delete_then_insert() {
        let mut t = FingerprintValueTable::new(2, 2, 8, 8);
        t.insert(0, 0x11, &[0xAA]).unwrap();
        t.insert(0, 0x22, &[0xBB]).unwrap();
        assert_eq!(t.insert(0, 0x33, &[0xCC]), None);
        assert!(t.delete(0, 0x11));
        // Slot 0 reused on next insert (lowest-empty-slot rule).
        assert_eq!(t.insert(0, 0x33, &[0xCC]), Some(0));
        assert_eq!(t.read_fingerprint(0, 0), 0x33);
    }

    #[test]
    fn fvt_size_in_bytes() {
        // 16 buckets × 4 slots × (12 + 64) bits = 16 * 4 * 76 = 4864 bits = 608 bytes.
        let t = FingerprintValueTable::new(16, 4, 12, 64);
        assert_eq!(t.size_in_bytes(), 608);
    }

    // ─── Out-of-range bucket panics ──────────────────────────────────────

    #[test]
    #[should_panic(expected = "out of range")]
    fn fvt_read_fingerprint_panics_on_bucket_oob() {
        let t = FingerprintValueTable::new(4, 2, 8, 8);
        let _ = t.read_fingerprint(4, 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fvt_read_value_panics_on_bucket_oob() {
        let t = FingerprintValueTable::new(4, 2, 8, 8);
        let mut out = [0u8; 1];
        t.read_value(4, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fvt_write_panics_on_bucket_oob() {
        let mut t = FingerprintValueTable::new(4, 2, 8, 8);
        t.write(4, 0, 1, &[0u8]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fvt_contain_panics_on_bucket_oob() {
        let t = FingerprintValueTable::new(4, 2, 8, 8);
        let _ = t.contain(4, 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fvt_find_panics_on_bucket_oob() {
        let t = FingerprintValueTable::new(4, 2, 8, 8);
        let _ = t.find(4, 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fvt_insert_panics_on_bucket_oob() {
        let mut t = FingerprintValueTable::new(4, 2, 8, 8);
        let _ = t.insert(4, 1, &[0u8]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fvt_delete_panics_on_bucket_oob() {
        let mut t = FingerprintValueTable::new(4, 2, 8, 8);
        let _ = t.delete(4, 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn fvt_update_value_panics_on_bucket_oob() {
        let mut t = FingerprintValueTable::new(4, 2, 8, 8);
        let _ = t.update_value(4, 1, &[0u8]);
    }

    // ─── Wrong-length buffer panics ──────────────────────────────────────

    #[test]
    #[should_panic(expected = "value_size_in_bytes")]
    fn fvt_read_value_panics_on_wrong_out_length() {
        let t = FingerprintValueTable::new(4, 2, 8, 16); // value_size=2 bytes
        let mut out = [0u8; 1]; // wrong
        t.read_value(0, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "value_size_in_bytes")]
    fn fvt_update_value_panics_on_wrong_value_length() {
        let mut t = FingerprintValueTable::new(4, 2, 8, 16); // value_size=2 bytes
        t.update_value(0, 1, &[0u8]); // wrong: 1 byte
    }

    #[test]
    #[should_panic(expected = "value_size_in_bytes")]
    fn fvt_insert_panics_on_wrong_value_length() {
        let mut t = FingerprintValueTable::new(4, 2, 8, 16); // value_size=2 bytes
        let _ = t.insert(0, 1, &[0u8]); // wrong: 1 byte
    }

    // ─── DataLayout::read_partial / write_partial ────────────────────────

    #[test]
    fn read_partial_aligned_offset() {
        // Slot is 16 bits; read the high 8 bits.
        let mut layout = DataLayout::new(2, 2, 16);
        layout.write(0, 0, &[0xAB, 0xCD]);
        let mut out = [0u8; 1];
        layout.read_partial(0, 0, 8, 8, &mut out);
        assert_eq!(out, [0xCD]);
    }

    #[test]
    fn read_partial_misaligned_within_byte() {
        // Slot is 12 bits; read 4 bits starting at bit_offset 4.
        let mut layout = DataLayout::new(2, 2, 12);
        layout.write(0, 0, &[0x5A, 0x0C]); // bits = 0xC5A
        let mut out = [0u8; 1];
        layout.read_partial(0, 0, 4, 4, &mut out);
        assert_eq!(out, [0x5]); // bits 4..8 of 0xC5A = 0x5
    }

    #[test]
    fn read_partial_crosses_byte_boundary() {
        // Slot is 24 bits; read 12 bits starting at bit_offset 6 (crosses byte boundary).
        let mut layout = DataLayout::new(2, 1, 24);
        layout.write(0, 0, &[0xAA, 0xBB, 0xCC]);
        let mut out = [0u8; 2];
        layout.read_partial(0, 0, 6, 12, &mut out);
        // Read bits 6..18 of 0xCCBBAA. Layout LSB-first: bit i of slot = data bit at start_bit+i.
        // bits[6..18] = 12 bits starting at bit 6 of the 24-bit value.
        let full: u32 = 0xCCBBAA;
        let extracted: u32 = (full >> 6) & 0xFFF;
        let mut expected = [0u8; 2];
        expected[0] = (extracted & 0xFF) as u8;
        expected[1] = ((extracted >> 8) & 0xF) as u8;
        assert_eq!(out, expected);
    }

    #[test]
    fn read_partial_at_last_slot() {
        // Exercise the trailing tail-padding region — the very last slot.
        let mut layout = DataLayout::new(4, 4, 12);
        // Last slot is (3, 3). Write a value whose high bits straddle the padding region.
        layout.write(3, 3, &[0xFE, 0x0F]); // 0xFFE
        let mut out = [0u8; 2];
        layout.read_partial(3, 3, 0, 12, &mut out);
        assert_eq!(out, [0xFE, 0x0F]);
        // A 1-bit partial read at the highest bit of the last slot.
        let mut bit = [0u8; 1];
        layout.read_partial(3, 3, 11, 1, &mut bit);
        assert_eq!(bit, [0x1]);
    }

    #[test]
    fn write_partial_doesnt_disturb_remainder_of_slot() {
        // 24-bit slot. Pre-fill, then write 8 bits at offset 8; bits [0..8) and [16..24)
        // must be preserved.
        let mut layout = DataLayout::new(2, 2, 24);
        layout.write(0, 0, &[0x11, 0x22, 0x33]);
        layout.write_partial(0, 0, 8, 8, &[0x99]);
        let mut out = [0u8; 3];
        layout.read(0, 0, &mut out);
        assert_eq!(out, [0x11, 0x99, 0x33]);
    }

    #[test]
    fn write_partial_doesnt_disturb_neighbour_slot() {
        // Adjacent slots: writing a partial range in slot (0,0) must not touch slot (0,1).
        let mut layout = DataLayout::new(1, 2, 12);
        layout.write(0, 1, &[0xCD, 0x0A]); // neighbour value 0xACD
        layout.write_partial(0, 0, 0, 8, &[0xFF]); // overwrite low 8 bits of slot 0
        let mut nb = [0u8; 2];
        layout.read(0, 1, &mut nb);
        assert_eq!(nb, [0xCD, 0x0A]);
    }

    #[test]
    fn write_partial_then_read_partial_roundtrip() {
        // Sweep widths covering: aligned, sub-byte, just-past-byte, multi-byte,
        // just-past-2-bytes, near 32 bits, past 32 bits, and very wide.
        let widths: &[u32] = &[1, 7, 8, 9, 12, 16, 31, 33, 64, 100];
        for &n_bits in widths {
            let slot_bits = n_bits + 5; // pad so partial doesn't span the entire slot
            let mut layout = DataLayout::new(2, 2, slot_bits);
            let n = (n_bits as usize).div_ceil(8);
            // Generate a deterministic value with the high bits of the last byte set
            // (so the masking path is exercised).
            let mut src: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(53).wrapping_add(7)).collect();
            // Ensure high bits in last byte are set above n_bits, so masking matters.
            src[n - 1] |= 0x80;

            // Write at bit_offset=2 inside the slot (forces misaligned start within slot).
            layout.write_partial(0, 1, 2, n_bits, &src);
            let mut out = vec![0u8; n];
            layout.read_partial(0, 1, 2, n_bits, &mut out);
            // Expected: src masked to n_bits.
            let mut expected = src.clone();
            let last_bits = n_bits as usize - (n - 1) * 8;
            if last_bits < 8 {
                expected[n - 1] &= (1u8 << last_bits) - 1;
            }
            assert_eq!(out, expected, "roundtrip failed at n_bits={n_bits}");
        }
    }

    #[test]
    fn write_partial_high_bits_in_src_are_masked() {
        // Write 4 bits with high bits of src set; verify masking and that adjacent
        // bit positions within the slot are not disturbed.
        let mut layout = DataLayout::new(1, 1, 16);
        layout.write(0, 0, &[0x00, 0xFF]); // pre-fill: high byte all set
        layout.write_partial(0, 0, 4, 4, &[0xF5]); // only low 4 bits should be written
        let mut out = [0u8; 2];
        layout.read(0, 0, &mut out);
        // Bits 0..4 untouched (0), bits 4..8 = 0x5, bits 8..16 untouched (0xFF).
        assert_eq!(out, [0x50, 0xFF]);
    }

    #[test]
    #[should_panic(expected = "exceeds data_size_in_bit")]
    fn read_partial_panics_on_range_oob() {
        let layout = DataLayout::new(2, 2, 12);
        let mut out = [0u8; 1];
        layout.read_partial(0, 0, 8, 8, &mut out); // 8 + 8 = 16 > 12
    }

    #[test]
    #[should_panic(expected = "exceeds data_size_in_bit")]
    fn write_partial_panics_on_range_oob() {
        let mut layout = DataLayout::new(2, 2, 12);
        layout.write_partial(0, 0, 8, 8, &[0u8]);
    }

    #[test]
    #[should_panic(expected = "out buffer length")]
    fn read_partial_panics_on_wrong_buffer_length() {
        let layout = DataLayout::new(2, 2, 16);
        let mut out = [0u8; 1]; // n_bits=12 needs 2 bytes
        layout.read_partial(0, 0, 0, 12, &mut out);
    }

    #[test]
    #[should_panic(expected = "src buffer length")]
    fn write_partial_panics_on_wrong_buffer_length() {
        let mut layout = DataLayout::new(2, 2, 16);
        layout.write_partial(0, 0, 0, 12, &[0u8]); // need 2 bytes
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn read_partial_panics_on_bucket_oob() {
        let layout = DataLayout::new(2, 2, 8);
        let mut out = [0u8; 1];
        layout.read_partial(2, 0, 0, 8, &mut out);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn write_partial_panics_on_slot_oob() {
        let mut layout = DataLayout::new(2, 2, 8);
        layout.write_partial(0, 2, 0, 8, &[0u8]);
    }

    #[test]
    #[should_panic(expected = "n_bits must be >= 1")]
    fn read_partial_panics_on_zero_n_bits() {
        let layout = DataLayout::new(2, 2, 8);
        let mut out: [u8; 0] = [];
        layout.read_partial(0, 0, 0, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "n_bits must be >= 1")]
    fn write_partial_panics_on_zero_n_bits() {
        let mut layout = DataLayout::new(2, 2, 8);
        let src: [u8; 0] = [];
        layout.write_partial(0, 0, 0, 0, &src);
    }

    // ─── FingerprintValueTable on the new partial path ───────────────────

    #[test]
    fn delete_zeros_value_bits_too() {
        // After delete, both fingerprint and value bits must read back as zero.
        let mut t = FingerprintValueTable::new(4, 2, 12, 64);
        let val: Vec<u8> = (1u8..=8).collect();
        t.insert(0, 0xAB, &val).unwrap();
        assert!(t.delete(0, 0xAB));
        assert_eq!(t.read_fingerprint(0, 0), 0);
        let mut out = vec![0u8; 8];
        t.read_value(0, 0, &mut out);
        assert!(out.iter().all(|&b| b == 0), "value bits not zeroed: {out:?}");
    }

    #[test]
    fn update_value_doesnt_touch_fingerprint_bits() {
        // Re-affirm against the partial-write path: fingerprint stays intact.
        let mut t = FingerprintValueTable::new(4, 2, 12, 16);
        let fp = 0xABC;
        t.insert(0, fp, &[0x11, 0x22]).unwrap();
        // Pre-populate slot 1 with a different fingerprint to detect cross-slot bleed.
        t.insert(0, 0x123, &[0xAA, 0xBB]).unwrap();
        assert!(t.update_value(0, fp, &[0x33, 0x44]));
        assert_eq!(t.read_fingerprint(0, 0), fp);
        let mut out = [0u8; 2];
        t.read_value(0, 0, &mut out);
        assert_eq!(out, [0x33, 0x44]);
        // Adjacent slot untouched.
        assert_eq!(t.read_fingerprint(0, 1), 0x123);
        t.read_value(0, 1, &mut out);
        assert_eq!(out, [0xAA, 0xBB]);
    }
}
