//! Bit-packed fingerprint storage.
//!
//! # Purpose
//!
//! [`FingerprintTable`] is the physical storage layer of every [`CuckooFilter`]. It stores
//! fingerprints at arbitrary bit widths (1–32 bits) in a flat byte buffer,
//! without byte alignment. This allows, for example, 12-bit fingerprints to be packed at
//! 66% of the space that 16-bit (byte-aligned) storage would consume.
//!
//! # Memory layout
//!
//! Fingerprints are packed contiguously, LSB-first. To read or write a fingerprint, the code
//! loads an aligned 8-byte window, shifts, and masks:
//!
//! ```text
//! bit offset = (bucket * fingerprints_per_bucket + slot) * fingerprint_bits
//! byte index = bit_offset / 8
//! shift      = bit_offset % 8
//! value      = (u64_at(byte_index) >> shift) & mask
//! ```
//!
//! Eight bytes of padding are appended to the allocation so that every 8-byte load stays
//! within bounds regardless of alignment.
//!
//! # Security
//!
//! No secret data flows through this module. All reads and writes are unconditional; there
//! are no data-dependent branches that could leak timing information about fingerprint values.
//!
//! [`CuckooFilter`]: crate::CuckooFilter

/// Bit-packed fingerprint storage for a cuckoo filter.
///
/// All fingerprint widths from 1 to 32 bits are supported. Fingerprints are stored without
/// byte alignment; the internal representation loads/stores u64 windows at the byte level.
#[derive(Clone)]
pub struct FingerprintTable {
    /// Raw fingerprint data: `⌈num_buckets · bucket_size · fingerprint_bits / 8⌉ + 8` bytes (8 bytes padding).
    data: Vec<u8>,
    /// Total number of buckets.
    pub num_buckets: u32,
    /// Slots (fingerprints) per bucket.
    pub fingerprints_per_bucket: u32,
    /// Bit width of each fingerprint (1–32).
    pub fingerprint_bits: u32,
}

impl FingerprintTable {
    /// Create a fingerprint table.
    ///
    /// # Arguments
    ///
    /// - `num_buckets` — total number of buckets `num_buckets`.
    /// - `fingerprints_per_bucket` — slots per bucket `bucket_size`.
    /// - `fingerprint_bits` — fingerprint bit width.
    ///
    /// # Constraints
    ///
    /// - `fingerprint_bits` must be in `1..=32`.
    ///
    /// Constraints on these parameters are not enforced by this constructor, but must be upheld by callers to ensure correct filter behavior.
    ///
    /// # Returns
    ///
    /// A zero-initialised [`FingerprintTable`] with `⌈num_buckets·bucket_size·fingerprint_bits/8⌉ + 8` bytes allocated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::FingerprintTable;
    ///
    /// let mut table = FingerprintTable::new(64, 4, 12);
    /// assert_eq!(table.num_buckets, 64);
    /// assert_eq!(table.fingerprints_per_bucket, 4);
    /// assert_eq!(table.fingerprint_bits, 12);
    /// ```
    pub fn new(num_buckets: u32, fingerprints_per_bucket: u32, fingerprint_bits: u32) -> Self {
        let total_bits =
            num_buckets as u64 * fingerprints_per_bucket as u64 * fingerprint_bits as u64;
        let total_bytes = total_bits.div_ceil(8) as usize;
        FingerprintTable {
            data: vec![0u8; total_bytes + 8],
            num_buckets,
            fingerprints_per_bucket,
            fingerprint_bits,
        }
    }

    /// Return the byte size of the fingerprint storage, excluding the trailing 8-byte padding.
    ///
    /// This is the *logical* storage cost: `⌈num_buckets · bucket_size · fingerprint_bits / 8⌉`. Callers use this to
    /// report memory usage without counting implementation overhead.
    ///
    /// # Returns
    ///
    /// Byte count of fingerprint data, rounded up to the nearest byte.
    ///
    /// # Performance
    ///
    /// O(1) — arithmetic only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::FingerprintTable;
    ///
    /// // 64 buckets × 4 slots × 12 bits = 3072 bits = 384 bytes
    /// let table = FingerprintTable::new(64, 4, 12);
    /// assert_eq!(table.size_in_bytes(), 384);
    /// ```
    pub fn size_in_bytes(&self) -> usize {
        let total_bits = self.num_buckets as u64
            * self.fingerprints_per_bucket as u64
            * self.fingerprint_bits as u64;
        total_bits.div_ceil(8) as usize
    }

    #[inline]
    fn slot_index(&self, bucket: u32, slot: u32) -> usize {
        bucket as usize * self.fingerprints_per_bucket as usize + slot as usize
    }

    #[inline]
    fn bit_offset(&self, bucket: u32, slot: u32) -> usize {
        self.slot_index(bucket, slot) * self.fingerprint_bits as usize
    }

    /// Read the fingerprint stored at `(bucket, slot)`.
    ///
    /// Loads an 8-byte little-endian word from the byte boundary at or before the
    /// fingerprint's bit offset, then shifts and masks to extract exactly `fingerprint_bits` bits.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index.
    /// - `slot` — slot index within the bucket.
    ///
    /// # Constraints
    ///
    /// - `bucket` must be in `[0, num_buckets)`.
    /// - `slot` must be in `[0, fingerprints_per_bucket)`.
    ///
    /// # Returns
    ///
    /// The fingerprint value, zero-padded to `u32`. A return value of `0` means the slot
    /// is empty (the "empty = 0" invariant is enforced by [`insert_fingerprint_to_bucket`]).
    ///
    /// # Performance
    ///
    /// O(1) — one unaligned `u64` load, shift, mask.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets` or `slot >= fingerprints_per_bucket`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::FingerprintTable;
    ///
    /// let mut table = FingerprintTable::new(4, 2, 12);
    /// table.write_fingerprint(1, 0, 0xABC);
    /// assert_eq!(table.read_fingerprint(1, 0), 0xABC);
    /// assert_eq!(table.read_fingerprint(1, 1), 0); // slot still empty
    /// ```
    ///
    /// [`insert_fingerprint_to_bucket`]: Self::insert_fingerprint_to_bucket
    ///
    /// The `expect` is an invariant: the 8-byte padding allocation guarantees the slice is
    /// always exactly 8 bytes, so `try_into()` cannot fail.
    #[inline]
    #[allow(clippy::expect_used)]
    pub fn read_fingerprint(&self, bucket: u32, slot: u32) -> u32 {
        assert!(
            bucket < self.num_buckets && slot < self.fingerprints_per_bucket,
            "(bucket, slot) out of range: ({bucket}, {slot}) not in [0, {}) x [0, {})",
            self.num_buckets,
            self.fingerprints_per_bucket,
        );
        let bit_pos = self.bit_offset(bucket, slot);
        let byte_pos = bit_pos / 8;
        let bit_shift = bit_pos % 8;
        let mask = (1u64 << self.fingerprint_bits) - 1;
        // SAFETY-equivalent: the 8-byte padding at the end of `data` guarantees that loading
        // 8 bytes at `byte_pos` never exceeds the allocation.
        let val = u64::from_le_bytes(
            self.data[byte_pos..byte_pos + 8]
                .try_into()
                .expect("invariant: 8-byte slice always converts to [u8; 8]"),
        );
        ((val >> bit_shift) & mask) as u32
    }

    /// Write `fingerprint` into the slot at `(bucket, slot)`.
    ///
    /// Performs a read-modify-write on the 8-byte word that covers the fingerprint's bit
    /// range: clears the existing `fingerprint_bits` bits, then OR-in the new value.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index.
    /// - `slot` — slot index within the bucket.
    /// - `fingerprint` — the fingerprint to write; only the lower `fingerprint_bits` bits are
    ///   stored. Pass `0` to mark the slot as empty.
    ///
    /// # Constraints
    ///
    /// - `bucket` must be in `[0, num_buckets)`.
    /// - `slot` must be in `[0, fingerprints_per_bucket)`.
    ///
    /// # Performance
    ///
    /// O(1) — one unaligned `u64` load, mask, OR, store.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets` or `slot >= fingerprints_per_bucket`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::FingerprintTable;
    ///
    /// let mut table = FingerprintTable::new(4, 4, 12);
    /// table.write_fingerprint(0, 3, 0xFFF);
    /// assert_eq!(table.read_fingerprint(0, 3), 0xFFF);
    ///
    /// // Overwrite with 0 to mark as empty.
    /// table.write_fingerprint(0, 3, 0);
    /// assert_eq!(table.read_fingerprint(0, 3), 0);
    /// ```
    ///
    /// See [`read_fingerprint`](Self::read_fingerprint) for the `expect` invariant rationale.
    #[inline]
    #[allow(clippy::expect_used)]
    pub fn write_fingerprint(&mut self, bucket: u32, slot: u32, fingerprint: u32) {
        assert!(
            bucket < self.num_buckets && slot < self.fingerprints_per_bucket,
            "(bucket, slot) out of range: ({bucket}, {slot}) not in [0, {}) x [0, {})",
            self.num_buckets,
            self.fingerprints_per_bucket,
        );
        let bit_pos = self.bit_offset(bucket, slot);
        let byte_pos = bit_pos / 8;
        let bit_shift = bit_pos % 8;
        let mask = (1u64 << self.fingerprint_bits) - 1;
        // SAFETY-equivalent: see read_fingerprint — 8-byte padding prevents out-of-bounds.
        let mut val = u64::from_le_bytes(
            self.data[byte_pos..byte_pos + 8]
                .try_into()
                .expect("invariant: 8-byte slice always converts to [u8; 8]"),
        );
        val &= !(mask << bit_shift);
        val |= (fingerprint as u64 & mask) << bit_shift;
        self.data[byte_pos..byte_pos + 8].copy_from_slice(&val.to_le_bytes());
    }

    /// Return `true` if `fingerprint` is present in any slot of `bucket`.
    ///
    /// Scans each slot in the bucket linearly and compares against `fingerprint`. Used by
    /// [`CuckooFilter::contain`] to check membership without needing the slot index.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index.
    /// - `fingerprint` — the fingerprint to search for (0 means empty).
    ///
    /// # Constraints
    ///
    /// - `bucket` must be in `[0, num_buckets)`.
    /// - `fingerprint` must be non-zero for normal lookups; `0` is deliberately allowed to
    ///   support finding empty slots.
    ///
    /// # Returns
    ///
    /// `true` if any slot in `bucket` holds `fingerprint`, `false` otherwise.
    ///
    /// # Performance
    ///
    /// O(bucket_size) — scans all `fingerprints_per_bucket` slots. In practice bucket_size ≤ 8, so this is a
    /// tight fixed-size loop.
    ///
    /// /// # Panics
    ///
    /// Panics if `bucket >= num_buckets` (already enforced in read_fingerprint) 
    /// `fingerprint == 0` is use deliberately allowed here to support finding empty slots, so no panic for zero fingerprints.
    /// 
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::FingerprintTable;
    ///
    /// let mut table = FingerprintTable::new(4, 4, 12);
    /// table.write_fingerprint(1, 2, 0x42);
    /// assert!(table.find_fingerprint_in_bucket(1, 0x42));
    /// assert!(!table.find_fingerprint_in_bucket(1, 0x99));
    /// ```
    ///
    /// [`CuckooFilter::contain`]: crate::CuckooFilter::contain
    pub fn find_fingerprint_in_bucket(&self, bucket: u32, fingerprint: u32) -> bool {
        for slot in 0..self.fingerprints_per_bucket {
            if self.read_fingerprint(bucket, slot) == fingerprint {
                return true;
            }
        }
        false
    }

    /// Find the slot index of `fingerprint` within `bucket`.
    ///
    /// Like [`find_fingerprint_in_bucket`] but also returns the matching slot index. Used
    /// internally during deletion when the matching slot must be zeroed.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index.
    /// - `fingerprint` — the fingerprint to find.
    ///
    /// # Constraints
    ///
    /// - `bucket` must be in `[0, num_buckets)`.
    ///
    /// # Returns
    ///
    /// `Some(slot)` if `fingerprint` is present, `None` otherwise.
    ///
    /// # Performance
    ///
    /// O(bucket_size) — linear scan over slots.
    ///
    /// # Panics
    /// 
    /// Panics if `bucket >= num_buckets` (already enforced in read_fingerprint).
    /// `fingerprint == 0` is use deliberately allowed here to support finding empty slots, so no panic for zero fingerprints.
    /// 
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::FingerprintTable;
    ///
    /// let mut table = FingerprintTable::new(4, 4, 12);
    /// table.write_fingerprint(0, 2, 0xFF);
    /// assert_eq!(table.find_fingerprint_in_bucket_slot(0, 0xFF), Some(2));
    /// assert_eq!(table.find_fingerprint_in_bucket_slot(0, 0xAB), None);
    /// ```
    ///
    /// [`find_fingerprint_in_bucket`]: Self::find_fingerprint_in_bucket
    pub fn find_fingerprint_in_bucket_slot(
        &self,
        bucket: u32,
        fingerprint: u32,
    ) -> Option<u32> {
        (0..self.fingerprints_per_bucket)
            .find(|&slot| self.read_fingerprint(bucket, slot) == fingerprint)
    }

    /// Delete the first occurrence of `fingerprint` from `bucket` by writing `0` to that slot.
    ///
    /// Writing `0` marks the slot as empty, consistent with the "empty = 0" invariant.
    /// Only the first matching slot is cleared; duplicate fingerprints (from multiple
    /// insertions of the same item) are deleted one at a time.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index.
    /// - `fingerprint` — the fingerprint to remove.
    ///
    /// # Returns
    ///
    /// `true` if `fingerprint` was found and cleared, `false` if it was not in `bucket`.
    ///
    /// # Performance
    ///
    /// O(bucket_size) — linear scan; clears at most one slot.
    ///
    /// # Panics
    /// 
    /// Panics if `bucket >= num_buckets` or if `fingerprint == 0` (0 is reserved for empty slots and cannot be a valid fingerprint).
    /// 
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::FingerprintTable;
    ///
    /// let mut table = FingerprintTable::new(4, 4, 8);
    /// table.write_fingerprint(0, 0, 42);
    /// assert!(table.delete_fingerprint_from_bucket(0, 42));
    /// assert!(!table.find_fingerprint_in_bucket(0, 42)); // gone
    /// assert!(!table.delete_fingerprint_from_bucket(0, 99)); // not present
    /// ```
    pub fn delete_fingerprint_from_bucket(&mut self, bucket: u32, fingerprint: u32) -> bool {
        assert!(fingerprint != 0, "fingerprint cannot be zero");
        for slot in 0..self.fingerprints_per_bucket {
            if self.read_fingerprint(bucket, slot) == fingerprint {
                self.write_fingerprint(bucket, slot, 0);
                return true;
            }
        }
        false
    }

    /// Insert `fingerprint` into the first empty (zero) slot of `bucket`.
    ///
    /// Scans for the first slot where `read_fingerprint == 0` and writes `fingerprint` there.
    /// Empty slots are identified by the value 0, which is why fingerprint 0 is forbidden for
    /// real items.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index.
    /// - `fingerprint` — the fingerprint to insert.
    ///
    /// # Constraints
    ///
    /// - `fingerprint` must be non-zero.
    ///
    /// # Returns
    ///
    /// `Some(slot)` with the slot index that was used, or `None` if all `bucket_size` slots in
    /// `bucket` are occupied.
    ///
    /// # Performance
    ///
    /// O(bucket_size) — linear scan; writes at most one slot.
    ///
    /// # Panics
    /// 
    /// Panics if `bucket >= num_buckets` (already enforced in read_fingerprint) 
    /// or if `fingerprint == 0` (0 is reserved for empty slots and cannot be a valid fingerprint).
    /// 
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::FingerprintTable;
    ///
    /// let mut table = FingerprintTable::new(4, 2, 8);
    /// assert_eq!(table.insert_fingerprint_to_bucket(0, 10), Some(0));
    /// assert_eq!(table.insert_fingerprint_to_bucket(0, 20), Some(1));
    /// assert_eq!(table.insert_fingerprint_to_bucket(0, 30), None);  // bucket full (bucket_size=2)
    /// ```
    pub fn insert_fingerprint_to_bucket(&mut self, bucket: u32, fingerprint: u32) -> Option<u32> {
        assert!(fingerprint != 0, "fingerprint cannot be zero");
        for slot in 0..self.fingerprints_per_bucket {
            if self.read_fingerprint(bucket, slot) == 0 {
                self.write_fingerprint(bucket, slot, fingerprint);
                return Some(slot);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_write_8bit() {
        let mut table = FingerprintTable::new(16, 4, 8);
        table.write_fingerprint(3, 2, 0xAB);
        assert_eq!(table.read_fingerprint(3, 2), 0xAB);
        assert_eq!(table.read_fingerprint(3, 1), 0);
    }

    #[test]
    fn test_read_write_12bit() {
        let mut table = FingerprintTable::new(16, 4, 12);
        table.write_fingerprint(5, 1, 0xFED);
        assert_eq!(table.read_fingerprint(5, 1), 0xFED);
        table.write_fingerprint(5, 2, 0x123);
        assert_eq!(table.read_fingerprint(5, 2), 0x123);
        assert_eq!(table.read_fingerprint(5, 1), 0xFED); // unchanged
    }

    #[test]
    fn test_read_write_16bit() {
        let mut table = FingerprintTable::new(8, 4, 16);
        table.write_fingerprint(7, 3, 0xBEEF);
        assert_eq!(table.read_fingerprint(7, 3), 0xBEEF);
    }

    #[test]
    fn test_read_write_32bit() {
        let mut table = FingerprintTable::new(4, 2, 32);
        table.write_fingerprint(1, 0, 0xDEADBEEF);
        assert_eq!(table.read_fingerprint(1, 0), 0xDEADBEEF);
    }

    #[test]
    fn test_find_and_delete() {
        let mut table = FingerprintTable::new(8, 4, 12);
        table.write_fingerprint(2, 0, 100);
        table.write_fingerprint(2, 1, 200);
        table.write_fingerprint(2, 2, 300);
        assert!(table.find_fingerprint_in_bucket(2, 200));
        assert!(!table.find_fingerprint_in_bucket(2, 999));
        assert!(table.delete_fingerprint_from_bucket(2, 200));
        assert!(!table.find_fingerprint_in_bucket(2, 200));
    }

    #[test]
    fn test_insert_fingerprint() {
        let mut table = FingerprintTable::new(4, 2, 8);
        assert_eq!(table.insert_fingerprint_to_bucket(0, 10), Some(0));
        assert_eq!(table.insert_fingerprint_to_bucket(0, 20), Some(1));
        assert_eq!(table.insert_fingerprint_to_bucket(0, 30), None); // bucket full
        assert!(table.find_fingerprint_in_bucket(0, 10));
        assert!(table.find_fingerprint_in_bucket(0, 20));
    }

    #[test]
    fn test_find_fingerprint_in_bucket_slot() {
        let mut table = FingerprintTable::new(4, 4, 8);
        table.write_fingerprint(1, 0, 10);
        table.write_fingerprint(1, 2, 30);
        assert_eq!(table.find_fingerprint_in_bucket_slot(1, 10), Some(0));
        assert_eq!(table.find_fingerprint_in_bucket_slot(1, 30), Some(2));
        assert_eq!(table.find_fingerprint_in_bucket_slot(1, 99), None);
    }

    #[test]
    fn test_delete_absent_fingerprint() {
        let mut table = FingerprintTable::new(4, 2, 8);
        assert!(!table.delete_fingerprint_from_bucket(0, 42));
    }

    #[test]
    fn test_size_in_bytes() {
        // 16 buckets * 4 slots * 12 bits = 768 bits = 96 bytes
        let table = FingerprintTable::new(16, 4, 12);
        assert_eq!(table.size_in_bytes(), 96);
        // 8 * 2 * 16 = 256 bits = 32 bytes
        let table2 = FingerprintTable::new(8, 2, 16);
        assert_eq!(table2.size_in_bytes(), 32);
    }

    // -------- Input-validation panic tests --------

    #[test]
    #[should_panic(expected = "out of range")]
    fn read_fingerprint_panics_on_bucket_oob() {
        let table = FingerprintTable::new(4, 2, 8);
        let _ = table.read_fingerprint(4, 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn read_fingerprint_panics_on_slot_oob() {
        let table = FingerprintTable::new(4, 2, 8);
        let _ = table.read_fingerprint(0, 2);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn read_fingerprint_panics_on_far_oob_bucket() {
        let table = FingerprintTable::new(4, 2, 8);
        let _ = table.read_fingerprint(u32::MAX, 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn write_fingerprint_panics_on_bucket_oob() {
        let mut table = FingerprintTable::new(4, 2, 8);
        table.write_fingerprint(4, 0, 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn write_fingerprint_panics_on_slot_oob() {
        let mut table = FingerprintTable::new(4, 2, 8);
        table.write_fingerprint(0, 2, 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn find_fingerprint_in_bucket_panics_on_bucket_oob() {
        let table = FingerprintTable::new(4, 2, 8);
        let _ = table.find_fingerprint_in_bucket(4, 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn find_fingerprint_in_bucket_slot_panics_on_bucket_oob() {
        let table = FingerprintTable::new(4, 2, 8);
        let _ = table.find_fingerprint_in_bucket_slot(4, 1);
    }

    #[test]
    #[should_panic(expected = "fingerprint cannot be zero")]
    fn delete_fingerprint_panics_on_zero_fingerprint() {
        let mut table = FingerprintTable::new(4, 2, 8);
        let _ = table.delete_fingerprint_from_bucket(0, 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn delete_fingerprint_panics_on_bucket_oob() {
        let mut table = FingerprintTable::new(4, 2, 8);
        let _ = table.delete_fingerprint_from_bucket(4, 1);
    }

    #[test]
    #[should_panic(expected = "fingerprint cannot be zero")]
    fn insert_fingerprint_panics_on_zero_fingerprint() {
        let mut table = FingerprintTable::new(4, 2, 8);
        let _ = table.insert_fingerprint_to_bucket(0, 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn insert_fingerprint_panics_on_bucket_oob() {
        let mut table = FingerprintTable::new(4, 2, 8);
        let _ = table.insert_fingerprint_to_bucket(4, 1);
    }
}
