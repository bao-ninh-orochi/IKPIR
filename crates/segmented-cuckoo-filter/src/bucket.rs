//! Bit-packed fingerprint storage.
//!
//! # Purpose
//!
//! [`TagTable`] is the physical storage layer of every [`CuckooFilter`]. It stores
//! fingerprints (*tags*) at arbitrary bit widths (1–32 bits) in a flat byte buffer,
//! without byte alignment. This allows, for example, 12-bit fingerprints to be packed at
//! 66% of the space that 16-bit (byte-aligned) storage would consume.
//!
//! # Memory layout
//!
//! Tags are packed contiguously, LSB-first. To read or write a tag, the code loads an
//! aligned 8-byte window, shifts, and masks:
//!
//! ```text
//! bit offset = (bucket * tags_per_bucket + slot) * bits_per_tag
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
/// All fingerprint widths from 1 to 32 bits are supported. Tags are stored without byte
/// alignment; the internal representation loads/stores u64 windows at the byte level.
#[derive(Clone)]
pub struct TagTable {
    /// Raw fingerprint data: `⌈n · b · fp_bits / 8⌉ + 8` bytes (8 bytes padding).
    data: Vec<u8>,
    /// Total number of buckets.
    pub num_buckets: u32,
    /// Slots (fingerprints) per bucket.
    pub tags_per_bucket: u32,
    /// Bit width of each fingerprint (1–32).
    pub bits_per_tag: u32,
}

impl TagTable {
    /// Create a tag table without chain-position storage.
    ///
    /// Suitable for 2-ary schemes and all segmented k > 2 schemes, where chain position
    /// can be inferred from the bucket index without extra storage.
    ///
    /// # Arguments
    ///
    /// - `num_buckets` — total number of buckets `n`.
    /// - `tags_per_bucket` — slots per bucket `b`.
    /// - `bits_per_tag` — fingerprint bit width (1–32).
    ///
    /// # Returns
    ///
    /// A zero-initialised [`TagTable`] with `⌈n·b·fp_bits/8⌉ + 8` bytes allocated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::TagTable;
    ///
    /// let mut table = TagTable::new(64, 4, 12);
    /// assert_eq!(table.num_buckets, 64);
    /// assert_eq!(table.tags_per_bucket, 4);
    /// assert_eq!(table.bits_per_tag, 12);
    /// ```
    pub fn new(num_buckets: u32, tags_per_bucket: u32, bits_per_tag: u32) -> Self {
        let total_bits = num_buckets as u64 * tags_per_bucket as u64 * bits_per_tag as u64;
        let total_bytes = total_bits.div_ceil(8) as usize;
        TagTable {
            data: vec![0u8; total_bytes + 8],
            num_buckets,
            tags_per_bucket,
            bits_per_tag,
        }
    }

    /// Return the byte size of the fingerprint storage, excluding padding and position bytes.
    ///
    /// This is the *logical* storage cost: `⌈n · b · fp_bits / 8⌉`. Callers use this to
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
    /// use segmented_cuckoo_filter::bucket::TagTable;
    ///
    /// // 64 buckets × 4 slots × 12 bits = 3072 bits = 384 bytes
    /// let table = TagTable::new(64, 4, 12);
    /// assert_eq!(table.size_in_bytes(), 384);
    /// ```
    pub fn size_in_bytes(&self) -> usize {
        let total_bits =
            self.num_buckets as u64 * self.tags_per_bucket as u64 * self.bits_per_tag as u64;
        total_bits.div_ceil(8) as usize
    }

    #[inline]
    fn slot_index(&self, bucket: u32, slot: u32) -> usize {
        bucket as usize * self.tags_per_bucket as usize + slot as usize
    }

    #[inline]
    fn bit_offset(&self, bucket: u32, slot: u32) -> usize {
        self.slot_index(bucket, slot) * self.bits_per_tag as usize
    }

    /// Read the fingerprint stored at `(bucket, slot)`.
    ///
    /// Loads an 8-byte little-endian word from the byte boundary at or before the tag's
    /// bit offset, then shifts and masks to extract exactly `bits_per_tag` bits.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index; must be in `[0, num_buckets)`.
    /// - `slot` — slot index within the bucket; must be in `[0, tags_per_bucket)`.
    ///
    /// # Returns
    ///
    /// The fingerprint value, zero-padded to `u32`. A return value of `0` means the slot
    /// is empty (the "empty = 0" invariant is enforced by [`insert_tag_to_bucket`]).
    ///
    /// # Performance
    ///
    /// O(1) — one unaligned `u64` load, shift, mask.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::TagTable;
    ///
    /// let mut table = TagTable::new(4, 2, 12);
    /// table.write_tag(1, 0, 0xABC);
    /// assert_eq!(table.read_tag(1, 0), 0xABC);
    /// assert_eq!(table.read_tag(1, 1), 0); // slot still empty
    /// ```
    ///
    /// [`insert_tag_to_bucket`]: Self::insert_tag_to_bucket
    ///
    /// The `expect` is an invariant: the 8-byte padding allocation guarantees the slice is
    /// always exactly 8 bytes, so `try_into()` cannot fail.
    #[inline]
    #[allow(clippy::expect_used)]
    pub fn read_tag(&self, bucket: u32, slot: u32) -> u32 {
        let bit_pos = self.bit_offset(bucket, slot);
        let byte_pos = bit_pos / 8;
        let bit_shift = bit_pos % 8;
        let mask = (1u64 << self.bits_per_tag) - 1;
        // SAFETY-equivalent: the 8-byte padding at the end of `data` guarantees that loading
        // 8 bytes at `byte_pos` never exceeds the allocation.
        let val = u64::from_le_bytes(
            self.data[byte_pos..byte_pos + 8]
                .try_into()
                .expect("invariant: 8-byte slice always converts to [u8; 8]"),
        );
        ((val >> bit_shift) & mask) as u32
    }

    /// Write `tag` into the slot at `(bucket, slot)`.
    ///
    /// Performs a read-modify-write on the 8-byte word that covers the tag's bit range:
    /// clears the existing `bits_per_tag` bits, then OR-in the new value.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index; must be in `[0, num_buckets)`.
    /// - `slot` — slot index within the bucket; must be in `[0, tags_per_bucket)`.
    /// - `tag` — the fingerprint to write; only the lower `bits_per_tag` bits are stored.
    ///   Pass `0` to mark the slot as empty.
    ///
    /// # Performance
    ///
    /// O(1) — one unaligned `u64` load, mask, OR, store.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::TagTable;
    ///
    /// let mut table = TagTable::new(4, 4, 12);
    /// table.write_tag(0, 3, 0xFFF);
    /// assert_eq!(table.read_tag(0, 3), 0xFFF);
    ///
    /// // Overwrite with 0 to mark as empty.
    /// table.write_tag(0, 3, 0);
    /// assert_eq!(table.read_tag(0, 3), 0);
    /// ```
    ///
    /// See [`read_tag`](Self::read_tag) for the `expect` invariant rationale.
    #[inline]
    #[allow(clippy::expect_used)]
    pub fn write_tag(&mut self, bucket: u32, slot: u32, tag: u32) {
        let bit_pos = self.bit_offset(bucket, slot);
        let byte_pos = bit_pos / 8;
        let bit_shift = bit_pos % 8;
        let mask = (1u64 << self.bits_per_tag) - 1;
        // SAFETY-equivalent: see read_tag — 8-byte padding prevents out-of-bounds.
        let mut val = u64::from_le_bytes(
            self.data[byte_pos..byte_pos + 8]
                .try_into()
                .expect("invariant: 8-byte slice always converts to [u8; 8]"),
        );
        val &= !(mask << bit_shift);
        val |= (tag as u64 & mask) << bit_shift;
        self.data[byte_pos..byte_pos + 8].copy_from_slice(&val.to_le_bytes());
    }

    /// Return `true` if `tag` is present in any slot of `bucket`.
    ///
    /// Scans each slot in the bucket linearly and compares against `tag`. Used by
    /// [`CuckooFilter::contain`] to check membership without needing the slot index.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index in `[0, num_buckets)`.
    /// - `tag` — the fingerprint to search for; must be non-zero (0 means empty).
    ///
    /// # Returns
    ///
    /// `true` if any slot in `bucket` holds `tag`, `false` otherwise.
    ///
    /// # Performance
    ///
    /// O(b) — scans all `tags_per_bucket` slots. In practice b ≤ 8, so this is a tight
    /// fixed-size loop.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::TagTable;
    ///
    /// let mut table = TagTable::new(4, 4, 12);
    /// table.write_tag(1, 2, 0x42);
    /// assert!(table.find_tag_in_bucket(1, 0x42));
    /// assert!(!table.find_tag_in_bucket(1, 0x99));
    /// ```
    ///
    /// [`CuckooFilter::contain`]: crate::CuckooFilter::contain
    pub fn find_tag_in_bucket(&self, bucket: u32, tag: u32) -> bool {
        for slot in 0..self.tags_per_bucket {
            if self.read_tag(bucket, slot) == tag {
                return true;
            }
        }
        false
    }

    /// Find the slot index of `tag` within `bucket`.
    ///
    /// Like [`find_tag_in_bucket`] but also returns the matching slot index. Used
    /// internally during deletion when the position must be zeroed.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index in `[0, num_buckets)`.
    /// - `tag` — the fingerprint to find.
    ///
    /// # Returns
    ///
    /// `Some(slot)` if `tag` is present, `None` otherwise.
    ///
    /// # Performance
    ///
    /// O(b) — linear scan over slots.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::TagTable;
    ///
    /// let mut table = TagTable::new(4, 4, 12);
    /// table.write_tag(0, 2, 0xFF);
    /// assert_eq!(table.find_tag_in_bucket_slot(0, 0xFF), Some(2));
    /// assert_eq!(table.find_tag_in_bucket_slot(0, 0xAB), None); // 0xAB was never written
    /// ```
    ///
    /// [`find_tag_in_bucket`]: Self::find_tag_in_bucket
    pub fn find_tag_in_bucket_slot(&self, bucket: u32, tag: u32) -> Option<u32> {
        (0..self.tags_per_bucket).find(|&slot| self.read_tag(bucket, slot) == tag)
    }

    /// Delete the first occurrence of `tag` from `bucket` by writing `0` to that slot.
    ///
    /// Writing `0` marks the slot as empty, consistent with the "empty = 0" invariant.
    /// Only the first matching slot is cleared; duplicate fingerprints (from multiple
    /// insertions of the same item) are deleted one at a time.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index.
    /// - `tag` — the fingerprint to remove.
    ///
    /// # Returns
    ///
    /// `true` if `tag` was found and cleared, `false` if `tag` was not in `bucket`.
    ///
    /// # Performance
    ///
    /// O(b) — linear scan; clears at most one slot.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::TagTable;
    ///
    /// let mut table = TagTable::new(4, 4, 8);
    /// table.write_tag(0, 0, 42);
    /// assert!(table.delete_tag_from_bucket(0, 42));
    /// assert!(!table.find_tag_in_bucket(0, 42)); // gone
    /// assert!(!table.delete_tag_from_bucket(0, 99)); // not present
    /// ```
    pub fn delete_tag_from_bucket(&mut self, bucket: u32, tag: u32) -> bool {
        for slot in 0..self.tags_per_bucket {
            if self.read_tag(bucket, slot) == tag {
                self.write_tag(bucket, slot, 0);
                return true;
            }
        }
        false
    }

    /// Insert `tag` into the first empty (zero) slot of `bucket`.
    ///
    /// Scans for the first slot where `read_tag == 0` and writes `tag` there. Empty slots
    /// are identified by the value 0, which is why fingerprint 0 is forbidden for real items.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index.
    /// - `tag` — the fingerprint to insert; must be non-zero.
    ///
    /// # Returns
    ///
    /// `Some(slot)` with the slot index that was used, or `None` if all `b` slots in
    /// `bucket` are occupied. The caller uses the returned slot index to write the
    /// chain position (for standard k > 2 schemes).
    ///
    /// # Performance
    ///
    /// O(b) — linear scan; writes at most one slot.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use segmented_cuckoo_filter::bucket::TagTable;
    ///
    /// let mut table = TagTable::new(4, 2, 8);
    /// assert_eq!(table.insert_tag_to_bucket(0, 10), Some(0));
    /// assert_eq!(table.insert_tag_to_bucket(0, 20), Some(1));
    /// assert_eq!(table.insert_tag_to_bucket(0, 30), None);  // bucket full (b=2)
    /// ```
    pub fn insert_tag_to_bucket(&mut self, bucket: u32, tag: u32) -> Option<u32> {
        for slot in 0..self.tags_per_bucket {
            if self.read_tag(bucket, slot) == 0 {
                self.write_tag(bucket, slot, tag);
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
        let mut table = TagTable::new(16, 4, 8);
        table.write_tag(3, 2, 0xAB);
        assert_eq!(table.read_tag(3, 2), 0xAB);
        assert_eq!(table.read_tag(3, 1), 0);
    }

    #[test]
    fn test_read_write_12bit() {
        let mut table = TagTable::new(16, 4, 12);
        table.write_tag(5, 1, 0xFED);
        assert_eq!(table.read_tag(5, 1), 0xFED);
        table.write_tag(5, 2, 0x123);
        assert_eq!(table.read_tag(5, 2), 0x123);
        assert_eq!(table.read_tag(5, 1), 0xFED); // unchanged
    }

    #[test]
    fn test_read_write_16bit() {
        let mut table = TagTable::new(8, 4, 16);
        table.write_tag(7, 3, 0xBEEF);
        assert_eq!(table.read_tag(7, 3), 0xBEEF);
    }

    #[test]
    fn test_read_write_32bit() {
        let mut table = TagTable::new(4, 2, 32);
        table.write_tag(1, 0, 0xDEADBEEF);
        assert_eq!(table.read_tag(1, 0), 0xDEADBEEF);
    }

    #[test]
    fn test_find_and_delete() {
        let mut table = TagTable::new(8, 4, 12);
        table.write_tag(2, 0, 100);
        table.write_tag(2, 1, 200);
        table.write_tag(2, 2, 300);
        assert!(table.find_tag_in_bucket(2, 200));
        assert!(!table.find_tag_in_bucket(2, 999));
        assert!(table.delete_tag_from_bucket(2, 200));
        assert!(!table.find_tag_in_bucket(2, 200));
    }

    #[test]
    fn test_insert_tag() {
        let mut table = TagTable::new(4, 2, 8);
        assert_eq!(table.insert_tag_to_bucket(0, 10), Some(0));
        assert_eq!(table.insert_tag_to_bucket(0, 20), Some(1));
        assert_eq!(table.insert_tag_to_bucket(0, 30), None); // bucket full
        assert!(table.find_tag_in_bucket(0, 10));
        assert!(table.find_tag_in_bucket(0, 20));
    }

    #[test]
    fn test_find_tag_in_bucket_slot() {
        let mut table = TagTable::new(4, 4, 8);
        table.write_tag(1, 0, 10);
        table.write_tag(1, 2, 30);
        assert_eq!(table.find_tag_in_bucket_slot(1, 10), Some(0));
        assert_eq!(table.find_tag_in_bucket_slot(1, 30), Some(2));
        assert_eq!(table.find_tag_in_bucket_slot(1, 99), None);
    }

    #[test]
    fn test_delete_absent_tag() {
        let mut table = TagTable::new(4, 2, 8);
        assert!(!table.delete_tag_from_bucket(0, 42));
    }

    #[test]
    fn test_size_in_bytes() {
        // 16 buckets * 4 slots * 12 bits = 768 bits = 96 bytes
        let table = TagTable::new(16, 4, 12);
        assert_eq!(table.size_in_bytes(), 96);
        // 8 * 2 * 16 = 256 bits = 32 bytes
        let table2 = TagTable::new(8, 2, 16);
        assert_eq!(table2.size_in_bytes(), 32);
    }
}
