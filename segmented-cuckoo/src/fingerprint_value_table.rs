//! Cell-based fingerprint-and-value storage for the IKPIR KV store.
//!
//! # Overview
//!
//! [`FingerprintValueTable`] is the physical storage layer for the `CuckooKVStore`. Unlike
//! [`FingerprintTable`](crate::fingerprint_table::FingerprintTable), which uses bit-packed
//! `Vec<u8>` storage optimised for minimal memory, this type uses a flat `Vec<u32>` of
//! *cells*, where each cell holds at most `plaintext_bits` payload bits in its low bits
//! with the high `(32 - plaintext_bits)` bits always zero.
//!
//! This layout matches ChalametPIR's `construct_row` convention: the PIR matvec operates
//! over `u32` cells and assumes `cell & ((1 << plaintext_bits) - 1) == cell` for every
//! cell. Aligned `u32` reads amortise the cost of the matvec inner loop and allow LLVM to
//! auto-vectorise it, at the cost of a `⌈32 / plaintext_bits⌉`× space overhead relative
//! to the bit-packed representation.
//!
//! # Layout invariants
//!
//! - `cells_per_slot = ⌈(fingerprint_bits + value_bits) / plaintext_bits⌉`. The last cell
//!   of every slot may carry fewer than `plaintext_bits` payload bits (ragged tail).
//! - Each cell carries up to `plaintext_bits` payload bits in its low bits; the high
//!   `(32 - plaintext_bits)` bits are **always zero**. This invariant is what makes the
//!   PIR matvec correct — ChalametPIR's `vec_mult_u32_u32` (utils.rs:72–87) assumes it.
//! - Slot `s` occupies cells `[s · cells_per_slot, (s + 1) · cells_per_slot)`,
//!   contiguously in `cells`.
//! - Within a slot, the fingerprint occupies bits `[0, fingerprint_bits)` of the
//!   bit-stream (across the leading cells); the value occupies bits
//!   `[fingerprint_bits, fingerprint_bits + value_bits)`. Either may straddle cell
//!   boundaries.
//! - Empty-slot convention: `fingerprint == 0` means the slot is empty; value bits in
//!   that slot are meaningless.
//! - **No 8-byte tail padding** — every cell read is an aligned `u32` load; no unaligned
//!   wide-load trick is needed (contrast with `FingerprintTable`'s 8-byte tail padding).

/// Cell-based fingerprint-and-value storage for the IKPIR cuckoo KV store.
///
/// Each entry `(fingerprint, value)` occupies `cells_per_slot` consecutive cells in a flat
/// `Vec<u32>`. Cell `c` of the flat array corresponds to slot `c / cells_per_slot`, at
/// cell-within-slot position `c % cells_per_slot`.
///
/// See the module-level documentation for the full layout invariants.
pub(crate) struct FingerprintValueTable {
    /// Flat cell array. Length = `num_slots * cells_per_slot`.
    cells: Vec<u32>,
    /// Total number of buckets.
    num_buckets: u32,
    /// Slots per bucket.
    bucket_size: u32,
    /// Bit width of each fingerprint (1–32).
    fingerprint_bits: u32,
    /// Bit width of each value (≥ 1).
    value_bits: u32,
    /// Bit width of each PIR plaintext cell (1–32).
    plaintext_bits: u32,
    /// `⌈(fingerprint_bits + value_bits) / plaintext_bits⌉`; cached at construction.
    cells_per_slot: u32,
}

impl FingerprintValueTable {
    /// Create a cell-based fingerprint-value table.
    ///
    /// # Arguments
    ///
    /// - `num_buckets`      — total number of buckets.
    /// - `bucket_size`      — slots per bucket.
    /// - `fingerprint_bits` — fingerprint bit width. Must be in `1..=32`.
    /// - `value_bits`       — value bit width. Must be ≥ 1.
    /// - `plaintext_bits`   — PIR plaintext cell width. Must be in `1..=32`.
    ///
    /// # Panics
    ///
    /// Panics if `fingerprint_bits` is not in `1..=32`, `value_bits == 0`,
    /// `plaintext_bits` is not in `1..=32`, or if
    /// `num_buckets * bucket_size * cells_per_slot` overflows `usize`.
    pub(crate) fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        plaintext_bits: u32,
    ) -> Self {
        assert!(
            (1..=32).contains(&fingerprint_bits),
            "fingerprint_bits must be in 1..=32, got {fingerprint_bits}"
        );
        assert!(value_bits >= 1, "value_bits must be >= 1, got {value_bits}");
        assert!(
            (1..=32).contains(&plaintext_bits),
            "plaintext_bits must be in 1..=32, got {plaintext_bits}"
        );
        let entry_bits = fingerprint_bits + value_bits;
        let cells_per_slot = entry_bits.div_ceil(plaintext_bits);
        let num_slots = (num_buckets as u64)
            .checked_mul(bucket_size as u64)
            .expect("num_buckets * bucket_size overflows u64");
        let total_cells = num_slots
            .checked_mul(cells_per_slot as u64)
            .and_then(|n| usize::try_from(n).ok())
            .expect("num_buckets * bucket_size * cells_per_slot overflows usize");
        FingerprintValueTable {
            cells: vec![0u32; total_cells],
            num_buckets,
            bucket_size,
            fingerprint_bits,
            value_bits,
            plaintext_bits,
            cells_per_slot,
        }
    }

    // ── Dimension accessors ───────────────────────────────────────────────────

    /// Total number of buckets.
    #[inline]
    pub(crate) fn num_buckets(&self) -> u32 {
        self.num_buckets
    }

    /// Slots per bucket.
    #[inline]
    pub(crate) fn bucket_size(&self) -> u32 {
        self.bucket_size
    }

    /// Bit width of each fingerprint (1–32).
    #[inline]
    pub(crate) fn fingerprint_bits(&self) -> u32 {
        self.fingerprint_bits
    }

    /// Bit width of each value (≥ 1).
    #[inline]
    pub(crate) fn value_bits(&self) -> u32 {
        self.value_bits
    }

    /// Bit width of each PIR plaintext cell (1–32).
    #[inline]
    pub(crate) fn plaintext_bits(&self) -> u32 {
        self.plaintext_bits
    }

    /// Cells per slot: `⌈(fingerprint_bits + value_bits) / plaintext_bits⌉`.
    #[inline]
    pub(crate) fn cells_per_slot(&self) -> u32 {
        self.cells_per_slot
    }

    /// Total number of slots: `num_buckets * bucket_size`.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn num_slots(&self) -> u64 {
        self.num_buckets as u64 * self.bucket_size as u64
    }

    /// Number of cells needed to hold the fingerprint: `⌈fingerprint_bits / plaintext_bits⌉`.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn fingerprint_size_in_cells(&self) -> u32 {
        self.fingerprint_bits.div_ceil(self.plaintext_bits)
    }

    /// Number of cells needed to hold the value: `⌈value_bits / plaintext_bits⌉`.
    #[inline]
    pub(crate) fn value_size_in_cells(&self) -> u32 {
        self.value_bits.div_ceil(self.plaintext_bits)
    }

    /// Number of cells per slot — same as `cells_per_slot()`.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn entry_size_in_cells(&self) -> u32 {
        self.cells_per_slot
    }

    /// Total number of cells in the flat array: `num_slots * cells_per_slot`.
    #[inline]
    pub(crate) fn size_in_cells(&self) -> usize {
        self.cells.len()
    }

    /// Byte size of the flat cell array: `4 * size_in_cells()`.
    ///
    /// This reflects the actual `Vec<u32>` allocation. It is larger than the bit-packed
    /// equivalent by a factor of `⌈32 / plaintext_bits⌉ · (plaintext_bits / 8)`.
    #[inline]
    pub(crate) fn size_in_bytes(&self) -> usize {
        4 * self.size_in_cells()
    }

    // ── Private bit-stream helpers ────────────────────────────────────────────

    /// Bit offset of the first cell of slot `(bucket, slot)` in the cell stream.
    #[inline]
    fn slot_bit_offset(&self, bucket: u32, slot: u32) -> u64 {
        (bucket as u64 * self.bucket_size as u64 + slot as u64)
            * self.cells_per_slot as u64
            * self.plaintext_bits as u64
    }

    #[inline]
    fn assert_in_range(&self, bucket: u32, slot: u32) {
        assert!(
            bucket < self.num_buckets && slot < self.bucket_size,
            "(bucket, slot) out of range: ({bucket}, {slot}) not in [0, {}) x [0, {})",
            self.num_buckets,
            self.bucket_size,
        );
    }

    /// Read up to 32 bits from the cell stream starting at `bit_offset`.
    ///
    /// `n_bits` must be in `1..=32`. Reads at most `⌈n_bits / plaintext_bits⌉ + 1` cells.
    #[inline]
    fn read_bits(&self, bit_offset: u64, n_bits: u32) -> u32 {
        let pb = self.plaintext_bits as u64;
        let mut cell_idx = bit_offset / pb;
        let intra_start = (bit_offset % pb) as u32;
        let pb32 = self.plaintext_bits;

        let mut acc: u64 = 0;

        // First cell: shift right by intra_start to align to bit 0 of acc.
        let v = self.cells[cell_idx as usize] as u64;
        acc |= v >> intra_start;
        let mut filled = pb32 - intra_start;
        cell_idx += 1;

        // Load subsequent cells until we have n_bits.
        while filled < n_bits {
            let v = self.cells[cell_idx as usize] as u64;
            acc |= v << filled;
            filled += pb32;
            cell_idx += 1;
        }

        (acc & ((1u64 << n_bits) - 1)) as u32
    }

    /// Write `n_bits` low bits of `value` into the cell stream at `bit_offset`.
    ///
    /// `n_bits` must be in `1..=32`. Preserves bits outside the `[bit_offset, bit_offset+n_bits)`
    /// window. High `(32 − plaintext_bits)` bits of touched cells remain zero.
    #[inline]
    fn write_bits(&mut self, bit_offset: u64, n_bits: u32, value: u32) {
        let pb = self.plaintext_bits as u64;
        let first_cell = bit_offset / pb;
        // Exclusive end cell index (ceiling division).
        let last_cell = (bit_offset + n_bits as u64).div_ceil(pb);

        let mut value_bit_pos: u32 = 0;

        for cell_idx in first_cell..last_cell {
            let cell_base_bit = cell_idx * pb;

            // Intersection of [bit_offset, bit_offset+n_bits) with [cell_base_bit, cell_base_bit+pb).
            let lo = if bit_offset > cell_base_bit {
                (bit_offset - cell_base_bit) as u32
            } else {
                0
            };
            let hi = {
                let end_bit = bit_offset + n_bits as u64;
                let cell_end = cell_base_bit + pb;
                (end_bit.min(cell_end) - cell_base_bit) as u32
            };
            let n = hi - lo;

            let mask = ((1u64 << n) - 1) as u32;
            let v_bits = (value >> value_bit_pos) & mask;
            let placed = v_bits << lo;
            let cell_mask = mask << lo;

            let idx = cell_idx as usize;
            self.cells[idx] = (self.cells[idx] & !cell_mask) | placed;

            value_bit_pos += n;
        }
    }

    // ── Public cell-based API ─────────────────────────────────────────────────

    /// Return a shared reference to the flat cell array for PIR backend access.
    ///
    /// Length = `num_slots * cells_per_slot`. Each cell holds at most `plaintext_bits`
    /// payload bits in its low bits; high bits are always zero.
    pub(crate) fn as_cells(&self) -> &[u32] {
        &self.cells
    }

    /// Read the fingerprint stored at `(bucket, slot)`.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets` or `slot >= bucket_size`.
    pub(crate) fn read_fingerprint(&self, bucket: u32, slot: u32) -> u32 {
        self.assert_in_range(bucket, slot);
        self.read_bits(self.slot_bit_offset(bucket, slot), self.fingerprint_bits)
    }

    /// Write `fingerprint` into the cell-stream at `(bucket, slot)`.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets` or `slot >= bucket_size`.
    pub(crate) fn write_fingerprint(&mut self, bucket: u32, slot: u32, fp: u32) {
        self.assert_in_range(bucket, slot);
        self.write_bits(self.slot_bit_offset(bucket, slot), self.fingerprint_bits, fp);
    }

    /// Read `value_size_in_cells()` cells from `(bucket, slot)` into `out`.
    ///
    /// `out` must have length `value_size_in_cells()`.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, `slot >= bucket_size`, or
    /// `out.len() != value_size_in_cells()`.
    pub(crate) fn read_value(&self, bucket: u32, slot: u32, out: &mut [u32]) {
        self.assert_in_range(bucket, slot);
        let n_cells = self.value_size_in_cells();
        assert_eq!(
            out.len(),
            n_cells as usize,
            "out length must equal value_size_in_cells ({n_cells})"
        );
        let pb = self.plaintext_bits;
        let off0 = self.slot_bit_offset(bucket, slot) + self.fingerprint_bits as u64;
        for i in 0..n_cells {
            let n = (self.value_bits - i * pb).min(pb);
            out[i as usize] = self.read_bits(off0 + (i as u64) * pb as u64, n);
        }
    }

    /// Write `value` cells into `(bucket, slot)`, starting at bit offset `fingerprint_bits`.
    ///
    /// `value` must have length `value_size_in_cells()`. Input cells are masked to
    /// `plaintext_bits` low bits before writing.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, `slot >= bucket_size`, or
    /// `value.len() != value_size_in_cells()`.
    pub(crate) fn write_value(&mut self, bucket: u32, slot: u32, value: &[u32]) {
        self.assert_in_range(bucket, slot);
        let n_cells = self.value_size_in_cells();
        assert_eq!(
            value.len(),
            n_cells as usize,
            "value length must equal value_size_in_cells ({n_cells})"
        );
        let pb = self.plaintext_bits;
        let off0 = self.slot_bit_offset(bucket, slot) + self.fingerprint_bits as u64;
        for i in 0..n_cells {
            let n = (self.value_bits - i * pb).min(pb);
            // Mask stray high bits defensively.
            let mask_n = ((1u64 << n) - 1) as u32;
            let masked = value[i as usize] & mask_n;
            self.write_bits(off0 + (i as u64) * pb as u64, n, masked);
        }
    }

    /// Write `fingerprint ‖ value` into `(bucket, slot)` in one call.
    ///
    /// Equivalent to `write_fingerprint` followed by `write_value`, but avoids
    /// a second bounds check.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, `slot >= bucket_size`, or
    /// `value.len() != value_size_in_cells()`.
    pub(crate) fn write(&mut self, bucket: u32, slot: u32, fp: u32, value: &[u32]) {
        self.write_fingerprint(bucket, slot, fp);
        self.write_value(bucket, slot, value);
    }

    /// Return `true` if `fingerprint` is present in any slot of `bucket`.
    ///
    /// `fingerprint == 0` is allowed (matches empty slots).
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`.
    pub(crate) fn contain(&self, bucket: u32, fp: u32) -> bool {
        (0..self.bucket_size).any(|s| self.read_fingerprint(bucket, s) == fp)
    }

    /// Return the slot index of `fingerprint` within `bucket`, or `None` if absent.
    ///
    /// `fingerprint == 0` is allowed (matches empty slots).
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`.
    pub(crate) fn find(&self, bucket: u32, fp: u32) -> Option<u32> {
        (0..self.bucket_size).find(|&s| self.read_fingerprint(bucket, s) == fp)
    }

    /// Insert `fingerprint ‖ value` into the first empty slot of `bucket`.
    ///
    /// Returns `Some(slot)` on success, `None` if all slots are occupied.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets`, `fingerprint == 0`, or
    /// `value.len() != value_size_in_cells()`.
    pub(crate) fn insert(&mut self, bucket: u32, fp: u32, value: &[u32]) -> Option<u32> {
        assert!(fp != 0, "fingerprint cannot be zero");
        assert_eq!(
            value.len(),
            self.value_size_in_cells() as usize,
            "value length must equal value_size_in_cells ({})",
            self.value_size_in_cells()
        );
        for s in 0..self.bucket_size {
            if self.read_fingerprint(bucket, s) == 0 {
                self.write(bucket, s, fp, value);
                return Some(s);
            }
        }
        None
    }

    /// Delete the first occurrence of `fingerprint` from `bucket` by zeroing its slot.
    ///
    /// Zeros both fingerprint and value bits so subsequent `read_value` on the empty slot
    /// returns all zeros.
    ///
    /// # Panics
    ///
    /// Panics if `bucket >= num_buckets` or `fingerprint == 0`.
    #[allow(dead_code)]
    pub(crate) fn delete(&mut self, bucket: u32, fp: u32) -> bool {
        assert!(fp != 0, "fingerprint cannot be zero");
        for s in 0..self.bucket_size {
            if self.read_fingerprint(bucket, s) == fp {
                self.write_fingerprint(bucket, s, 0);
                let zeros = vec![0u32; self.value_size_in_cells() as usize];
                self.write_value(bucket, s, &zeros);
                return true;
            }
        }
        false
    }

    // ── IKPIR-bridge helpers ──────────────────────────────────────────────────

    /// Reconstruct a table from a previously-snapshotted cell array.
    ///
    /// Validates that `cells.len() == expected size_in_cells` and that every cell
    /// satisfies the ChalametPIR high-bits-zero invariant. O(N) cost; designed for
    /// snapshot-restore only, not the hot path.
    ///
    /// # Errors
    ///
    /// Returns a static error string on length mismatch or invariant violation.
    pub(crate) fn from_cells(
        cells: Vec<u32>,
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        plaintext_bits: u32,
    ) -> Result<Self, &'static str> {
        let entry_bits = fingerprint_bits + value_bits;
        let cells_per_slot = entry_bits.div_ceil(plaintext_bits);
        let expected = (num_buckets as u64)
            .saturating_mul(bucket_size as u64)
            .saturating_mul(cells_per_slot as u64) as usize;
        if cells.len() != expected {
            return Err("cells length does not match expected size_in_cells");
        }
        if plaintext_bits < 32 {
            let hi_mask = !((1u32 << plaintext_bits) - 1);
            if cells.iter().any(|&c| c & hi_mask != 0) {
                return Err("cell has high bits set (ChalametPIR high-bits-zero invariant violated)");
            }
        }
        Ok(FingerprintValueTable {
            cells,
            num_buckets,
            bucket_size,
            fingerprint_bits,
            value_bits,
            plaintext_bits,
            cells_per_slot,
        })
    }

    /// Read `out.len()` consecutive value cells starting at `cell_start` within `(bucket, slot)`.
    ///
    /// Each output cell carries `min(plaintext_bits, value_bits − cell_start·plaintext_bits)` bits.
    /// Used by the streaming `get_into` path to avoid a full `Vec` allocation.
    pub(crate) fn read_value_cells_chunk(
        &self,
        bucket: u32,
        slot: u32,
        cell_start: usize,
        out: &mut [u32],
    ) {
        let pb = self.plaintext_bits;
        let off0 = self.slot_bit_offset(bucket, slot) + self.fingerprint_bits as u64;
        for (i, c) in out.iter_mut().enumerate() {
            let ci = cell_start + i;
            let n = (self.value_bits - ci as u32 * pb).min(pb);
            *c = self.read_bits(off0 + ci as u64 * pb as u64, n);
        }
    }

    /// Read all value cells from `(bucket, slot)` into a heap-allocated boxed slice.
    ///
    /// One allocation per call. Used by the mutation log to snapshot old/new cell state.
    pub(crate) fn read_value_to_box(&self, bucket: u32, slot: u32) -> Box<[u32]> {
        let n = self.value_size_in_cells() as usize;
        let mut buf = vec![0u32; n];
        self.read_value(bucket, slot, &mut buf);
        buf.into_boxed_slice()
    }

    /// Update the value for the first slot in `bucket` matching `fingerprint`.
    ///
    /// Returns `true` if found and updated, `false` if `fingerprint` was not in `bucket`.
    ///
    /// # Panics
    ///
    /// Panics if `new_value.len() != value_size_in_cells()`.
    #[allow(dead_code)]
    pub(crate) fn update_value(&mut self, bucket: u32, fp: u32, new_value: &[u32]) -> bool {
        assert_eq!(
            new_value.len(),
            self.value_size_in_cells() as usize,
            "new_value length must equal value_size_in_cells ({})",
            self.value_size_in_cells()
        );
        if let Some(s) = self.find(bucket, fp) {
            self.write_value(bucket, s, new_value);
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fvt(nb: u32, bs: u32, fb: u32, vb: u32, pb: u32) -> FingerprintValueTable {
        FingerprintValueTable::new(nb, bs, fb, vb, pb)
    }

    // ── Round-trip sweep ──────────────────────────────────────────────────────

    fn roundtrip_param(pb: u32, fp_bits: u32, vb: u32) {
        let nb = 8u32;
        let bs = 4u32;
        let mut fvt = make_fvt(nb, bs, fp_bits, vb, pb);
        let vcells = fvt.value_size_in_cells() as usize;

        let fp_mask = ((1u64 << fp_bits) - 1) as u32;
        let cell_mask = if pb >= 32 { u32::MAX } else { (1u32 << pb) - 1 };

        for b in 0..nb {
            for s in 0..bs {
                let fp = ((b * bs + s + 1) & fp_mask).max(1);
                let value: Vec<u32> = (0..vcells)
                    .map(|i| ((b * 7 + s * 3 + i as u32 + 1) & cell_mask))
                    .collect();

                fvt.write(b, s, fp, &value);

                assert_eq!(fvt.read_fingerprint(b, s), fp, "pb={pb} fp_bits={fp_bits} vb={vb} b={b} s={s}");

                let mut got = vec![0u32; vcells];
                fvt.read_value(b, s, &mut got);
                for (i, (&g, &e)) in got.iter().zip(value.iter()).enumerate() {
                    let n = (vb - i as u32 * pb).min(pb);
                    let mask = ((1u64 << n) - 1) as u32;
                    assert_eq!(g & mask, e & mask, "pb={pb} fp_bits={fp_bits} vb={vb} b={b} s={s} cell={i}");
                }

                // High-bit invariant
                if pb < 32 {
                    let hi_mask = !((1u32 << pb) - 1);
                    for &c in fvt.as_cells() {
                        assert_eq!(c & hi_mask, 0, "high bits non-zero at pb={pb}");
                    }
                }
            }
        }
    }

    #[test]
    fn cell_api_roundtrip_param() {
        for &pb in &[8u32, 9, 10, 12, 16, 32] {
            for &fp_bits in &[12u32, 16, 32] {
                for &vb in &[1u32, 8, 32, 64, 1024] {
                    if fp_bits > 32 || vb == 0 { continue; }
                    roundtrip_param(pb, fp_bits, vb);
                }
            }
        }
    }

    #[test]
    fn empty_slot_semantics() {
        let fvt = make_fvt(4, 4, 12, 8, 8);
        // All slots start empty (fingerprint == 0).
        for b in 0..4 {
            for s in 0..4 {
                assert_eq!(fvt.read_fingerprint(b, s), 0);
            }
        }
    }

    #[test]
    fn insert_then_find() {
        let mut fvt = make_fvt(4, 4, 12, 8, 8);
        let value = [0xABu32];
        assert_eq!(fvt.insert(0, 0x42, &value), Some(0));
        assert_eq!(fvt.find(0, 0x42), Some(0));
        assert_eq!(fvt.find(0, 0x99), None);
    }

    #[test]
    fn insert_into_full_bucket_returns_none() {
        let mut fvt = make_fvt(4, 2, 12, 8, 8);
        let v = [1u32];
        assert!(fvt.insert(0, 1, &v).is_some());
        assert!(fvt.insert(0, 2, &v).is_some());
        assert_eq!(fvt.insert(0, 3, &v), None); // full
    }

    #[test]
    fn delete_clears_slot() {
        let mut fvt = make_fvt(4, 4, 12, 8, 8);
        let value = [0xFFu32];
        fvt.insert(0, 5, &value).unwrap();
        assert!(fvt.delete(0, 5));
        assert_eq!(fvt.read_fingerprint(0, 0), 0);
        let mut out = [0u32; 1];
        fvt.read_value(0, 0, &mut out);
        assert_eq!(out[0], 0);
    }

    #[test]
    fn update_value_preserves_fingerprint() {
        let mut fvt = make_fvt(4, 4, 12, 8, 8);
        let v1 = [0xAAu32];
        let v2 = [0xBBu32];
        fvt.insert(0, 7, &v1).unwrap();
        assert!(fvt.update_value(0, 7, &v2));
        assert_eq!(fvt.read_fingerprint(0, 0), 7);
        let mut out = [0u32; 1];
        fvt.read_value(0, 0, &mut out);
        assert_eq!(out[0], 0xBB);
    }

    #[test]
    fn as_cells_high_bit_invariant_random_writes() {
        let pb = 9u32;
        let mut fvt = make_fvt(8, 4, 12, 64, pb);
        let vcells = fvt.value_size_in_cells() as usize;
        let hi_mask = !((1u32 << pb) - 1);
        for b in 0..8u32 {
            for s in 0..4u32 {
                let fp = (b * 4 + s + 1) % ((1 << 12) - 1) + 1;
                let value: Vec<u32> = (0..vcells).map(|i| (i as u32 * 13 + 7) & ((1 << pb) - 1)).collect();
                fvt.write(b, s, fp, &value);
            }
        }
        for &c in fvt.as_cells() {
            assert_eq!(c & hi_mask, 0);
        }
    }

    #[test]
    fn as_cells_dimensions() {
        let nb = 16u32;
        let bs = 4u32;
        let pb = 10u32;
        let fvt = make_fvt(nb, bs, 12, 64, pb);
        let expected = nb as usize * bs as usize * fvt.cells_per_slot() as usize;
        assert_eq!(fvt.as_cells().len(), expected);
    }

    #[test]
    fn cell_boundary_straddle_fp() {
        // pb=9, fp_bits=12 — fp straddles cells 0 and 1
        let mut fvt = make_fvt(4, 4, 12, 8, 9);
        fvt.write_fingerprint(0, 0, 0xFFF);
        assert_eq!(fvt.read_fingerprint(0, 0), 0xFFF);
        // High bits must remain zero
        let hi_mask = !((1u32 << 9) - 1);
        for &c in fvt.as_cells() {
            assert_eq!(c & hi_mask, 0);
        }
    }

    #[test]
    fn cell_boundary_straddle_value() {
        // pb=10, fp_bits=4, vb=14 — value straddles two cells
        let mut fvt = make_fvt(4, 4, 4, 14, 10);
        let value = [0x3FFu32, 0xFu32]; // 10 + 4 bits = 14 bits
        fvt.write(0, 0, 5, &value);
        assert_eq!(fvt.read_fingerprint(0, 0), 5);
        let mut out = [0u32; 2];
        fvt.read_value(0, 0, &mut out);
        assert_eq!(out[0] & 0x3FF, value[0] & 0x3FF);
        assert_eq!(out[1] & 0xF, value[1] & 0xF);
    }

    // ── Wide value (vb=1024) ──────────────────────────────────────────────────

    #[test]
    fn wide_value_roundtrip_at_pb_8_vb_1024() {
        let pb = 8u32;
        let vb = 1024u32;
        let mut fvt = make_fvt(4, 4, 12, vb, pb);
        let vcells = fvt.value_size_in_cells() as usize;
        let value: Vec<u32> = (0..vcells).map(|i| (i as u32 * 37 + 13) & 0xFF).collect();
        fvt.write(0, 0, 0xABC, &value);
        let mut out = vec![0u32; vcells];
        fvt.read_value(0, 0, &mut out);
        assert_eq!(out, value);
    }

    #[test]
    fn wide_value_roundtrip_at_pb_9_vb_1024() {
        let pb = 9u32;
        let vb = 1024u32;
        let mut fvt = make_fvt(4, 4, 12, vb, pb);
        let vcells = fvt.value_size_in_cells() as usize;
        let hi_mask = !((1u32 << pb) - 1);
        let value: Vec<u32> = (0..vcells).map(|i| ((i as u32 * 17 + 5) & ((1 << pb) - 1))).collect();
        fvt.write(0, 0, 0x42, &value);
        let mut out = vec![0u32; vcells];
        fvt.read_value(0, 0, &mut out);
        // Check last cell separately (ragged tail)
        let last = vcells - 1;
        let tail_bits = vb - last as u32 * pb;
        let tail_mask = ((1u64 << tail_bits) - 1) as u32;
        for i in 0..last {
            assert_eq!(out[i], value[i], "cell {i}");
        }
        assert_eq!(out[last] & tail_mask, value[last] & tail_mask, "last cell");
        // High-bit invariant
        for &c in fvt.as_cells() {
            assert_eq!(c & hi_mask, 0);
        }
    }

    // ── Panic tests ───────────────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "fingerprint cannot be zero")]
    fn insert_panics_on_zero_fp() {
        let mut fvt = make_fvt(4, 4, 12, 8, 8);
        let _ = fvt.insert(0, 0, &[0u32]);
    }

    #[test]
    #[should_panic(expected = "fingerprint cannot be zero")]
    fn delete_panics_on_zero_fp() {
        let mut fvt = make_fvt(4, 4, 12, 8, 8);
        let _ = fvt.delete(0, 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn read_fingerprint_panics_on_oob_bucket() {
        let fvt = make_fvt(4, 4, 12, 8, 8);
        let _ = fvt.read_fingerprint(4, 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn read_fingerprint_panics_on_oob_slot() {
        let fvt = make_fvt(4, 4, 12, 8, 8);
        let _ = fvt.read_fingerprint(0, 4);
    }

    #[test]
    #[should_panic(expected = "out length must equal value_size_in_cells")]
    fn read_value_panics_on_wrong_length() {
        let fvt = make_fvt(4, 4, 12, 8, 8);
        let mut out = [0u32; 5];
        fvt.read_value(0, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "value length must equal value_size_in_cells")]
    fn write_value_panics_on_wrong_length() {
        let mut fvt = make_fvt(4, 4, 12, 8, 8);
        let value = [0u32; 5];
        fvt.write_value(0, 0, &value);
    }
}
