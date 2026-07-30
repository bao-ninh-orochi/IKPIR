//! Cell-based fingerprint-and-value storage for the IKPIR KV store.
//!
//! # Purpose
//!
//! Provides [`FingerprintValueTable`], the physical storage layer underneath
//! [`CuckooKVStore`](crate::CuckooKVStore). Holds one
//! `(fingerprint, value)` entry per slot in a flat `Vec<u32>` of *cells*,
//! laid out so the ChalametPIR / FrodoPIR `vec_mult_u32_u32` matvec can
//! consume the array directly.
//!
//! # Design / architecture
//!
//! - **Cells, not bytes.** Unlike
//!   [`FingerprintTable`](crate::fingerprint_table::FingerprintTable) — which
//!   bit-packs into `Vec<u8>` to minimise filter footprint — this type uses a
//!   flat `Vec<u32>` where each cell carries at most `plaintext_bits` payload
//!   bits in its low bits, with the high `(32 − plaintext_bits)` bits always
//!   zero. The PIR matvec accumulates cells into a `u64`; the high-bit-zero
//!   invariant is what prevents overflow inside that accumulator.
//! - **Slot layout.** `cells_per_slot = ⌈(fingerprint_bits + value_bits)
//!   / plaintext_bits⌉`. Slot `s` occupies cells
//!   `[s · cells_per_slot, (s + 1) · cells_per_slot)`. Within a slot, the
//!   fingerprint occupies the leading `fingerprint_bits` of the bit-stream
//!   and the value occupies the next `value_bits`; either may straddle a
//!   cell boundary. The trailing cell may carry fewer than `plaintext_bits`
//!   bits (ragged tail).
//! - **Empty-slot convention.** `fingerprint == 0` flags an empty slot. The
//!   non-zero fingerprint invariant lives in the hash layer
//!   ([`crate::hash`]).
//! - **No tail padding.** Every cell read is an aligned `u32` load; the
//!   unaligned-wide-load trick that `FingerprintTable` uses is not needed
//!   here.
//! - **Trade-off.** Cell-based storage costs `⌈32 / plaintext_bits⌉`× the
//!   bit-packed representation but lets LLVM auto-vectorise the matvec inner
//!   loop, which is the IKPIR hot path.
//!
//! # Related files
//!
//! - `store.rs` — sole consumer; wraps this table inside `CuckooKVStore`
//!   and drives all reads/writes.
//! - `fingerprint_table.rs` — bit-packed counterpart for the
//!   filter-only `CuckooFilter`.
//! - `lib.rs` — module-level docs that point users at the type aliases.

/// Cell-based fingerprint-and-value storage for the IKPIR cuckoo KV store.
///
/// # Purpose
///
/// Physical storage for the SCF-backed key-value store. Each `(fp, value)`
/// entry occupies `cells_per_slot` consecutive `u32` cells; cell `c` maps to
/// slot `c / cells_per_slot` at intra-slot position `c % cells_per_slot`.
///
/// # Rationale
///
/// The flat `Vec<u32>` layout, combined with the high-bit-zero invariant on
/// each cell, lets the PIR backend (FrodoPIR's `vec_mult_u32_u32`) treat the
/// array as a row-major plaintext matrix without any decoding. See the
/// module-level docs for the full layout invariants.
pub struct FingerprintValueTable {
    /// Flat cell array. Length = `num_slots * cells_per_slot`.
    cells: Vec<u32>,
    /// Total number of buckets.
    num_buckets: u32,
    /// Slots per bucket.
    bucket_size: u32,
    /// Bit width of each fingerprint (1–64).
    fingerprint_bits: u32,
    /// Bit width of each value (≥ 1).
    value_bits: u32,
    /// Bit width of each PIR plaintext cell (1–32).
    plaintext_bits: u32,
    /// `⌈(fingerprint_bits + value_bits) / plaintext_bits⌉`; cached at construction.
    cells_per_slot: u32,
}

impl FingerprintValueTable {
    /// Create an empty cell-based fingerprint-value table of the given shape.
    ///
    /// # Purpose
    ///
    /// Allocate the flat cell array and cache the per-slot cell count so the
    /// hot-path `read_bits` / `write_bits` helpers do not have to recompute
    /// `⌈entry_bits / plaintext_bits⌉` on every call.
    ///
    /// # Arguments
    ///
    /// - `num_buckets`      — total number of buckets in the table.
    /// - `bucket_size`      — slots per bucket.
    /// - `fingerprint_bits` — fingerprint width in bits.
    /// - `value_bits`       — value width in bits.
    /// - `plaintext_bits`   — PIR plaintext cell width in bits (each cell
    ///   carries this many low-order payload bits).
    ///
    /// # Constraints
    ///
    /// - `fingerprint_bits` must be in `1..=64`.
    /// - `value_bits` must be ≥ 1.
    /// - `plaintext_bits` must be in `1..=32`.
    /// - `num_buckets · bucket_size · cells_per_slot` must fit in `usize`.
    ///
    /// # Returns
    ///
    /// A zero-initialised table; every slot is empty (`fingerprint == 0`).
    ///
    /// # Complexity
    ///
    /// `O(num_buckets · bucket_size · cells_per_slot)` time and memory
    /// (one `vec![0u32; …]` allocation).
    pub(crate) fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        plaintext_bits: u32,
    ) -> Self {
        assert!(
            (1..=64).contains(&fingerprint_bits),
            "fingerprint_bits must be in 1..=64, got {fingerprint_bits}"
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
        Self {
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
    pub(crate) const fn num_buckets(&self) -> u32 {
        self.num_buckets
    }

    /// Slots per bucket.
    #[inline]
    pub(crate) const fn bucket_size(&self) -> u32 {
        self.bucket_size
    }

    /// Bit width of each fingerprint (1–64).
    #[inline]
    pub(crate) const fn fingerprint_bits(&self) -> u32 {
        self.fingerprint_bits
    }

    /// Bit width of each value (≥ 1).
    #[inline]
    pub(crate) const fn value_bits(&self) -> u32 {
        self.value_bits
    }

    /// Bit width of each PIR plaintext cell (1–32).
    #[inline]
    pub(crate) const fn plaintext_bits(&self) -> u32 {
        self.plaintext_bits
    }

    /// Cells per slot: `⌈(fingerprint_bits + value_bits) / plaintext_bits⌉`.
    #[inline]
    pub(crate) const fn cells_per_slot(&self) -> u32 {
        self.cells_per_slot
    }

    /// Total number of slots: `num_buckets · bucket_size`.
    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn num_slots(&self) -> u64 {
        self.num_buckets as u64 * self.bucket_size as u64
    }

    /// Cells needed for one fingerprint: `⌈fingerprint_bits / plaintext_bits⌉`.
    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn fingerprint_size_in_cells(&self) -> u32 {
        self.fingerprint_bits.div_ceil(self.plaintext_bits)
    }

    /// Cells needed for one value: `⌈value_bits / plaintext_bits⌉`.
    #[inline]
    pub(crate) const fn value_size_in_cells(&self) -> u32 {
        self.value_bits.div_ceil(self.plaintext_bits)
    }

    /// Cells per slot — same as [`cells_per_slot`](Self::cells_per_slot).
    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn entry_size_in_cells(&self) -> u32 {
        self.cells_per_slot
    }

    /// Total length of the flat cell array: `num_slots · cells_per_slot`.
    #[inline]
    pub(crate) fn size_in_cells(&self) -> usize {
        self.cells.len()
    }

    /// Heap footprint of the flat cell array, in bytes: `4 · size_in_cells()`.
    ///
    /// Larger than the bit-packed equivalent by a factor of
    /// `⌈32 / plaintext_bits⌉ · (plaintext_bits / 8)` — the price paid for
    /// PIR-matvec-friendly alignment.
    #[inline]
    pub(crate) fn size_in_bytes(&self) -> usize {
        4 * self.size_in_cells()
    }

    // ── Private bit-stream helpers ────────────────────────────────────────────

    /// Bit offset of the first cell of slot `(bucket, slot)` in the cell stream.
    #[inline]
    const fn slot_bit_offset(&self, bucket: u32, slot: u32) -> u64 {
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

    /// Read up to 64 bits from the cell stream starting at `bit_offset`.
    ///
    /// `n_bits` must be in `1..=64`. Reads at most
    /// `⌈n_bits / plaintext_bits⌉ + 1` cells.
    #[inline]
    fn read_bits(&self, bit_offset: u64, n_bits: u32) -> u64 {
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

        // Load subsequent cells until we have n_bits. The loop guard keeps
        // `filled < n_bits <= 64`, so the shift amount never reaches 64; the
        // debug_assert catches an invariant break in test builds and
        // compiles out of release (this runs on benched decode paths).
        while filled < n_bits {
            let v = self.cells[cell_idx as usize] as u64;
            debug_assert!(filled < 64, "shift amount must stay below 64");
            acc |= v << filled;
            filled += pb32;
            cell_idx += 1;
        }

        // `u64::MAX >> (64 - n_bits)` rather than `(1u64 << n_bits) - 1`: the
        // latter overflows (shift amount 64) when n_bits == 64.
        acc & (u64::MAX >> (64 - n_bits))
    }

    /// Write `n_bits` low bits of `value` into the cell stream at `bit_offset`.
    ///
    /// `n_bits` must be in `1..=64`. Preserves bits outside
    /// `[bit_offset, bit_offset + n_bits)`. High `(32 − plaintext_bits)` bits
    /// of touched cells remain zero.
    #[inline]
    fn write_bits(&mut self, bit_offset: u64, n_bits: u32, value: u64) {
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

            let mask = (1u64 << n) - 1;
            // The loop consumes exactly `n_bits <= 64` bits across all cells,
            // so `value_bit_pos` is below 64 whenever a cell still needs bits;
            // the debug_assert catches an invariant break in test builds and
            // compiles out of release.
            debug_assert!(value_bit_pos < 64, "shift amount must stay below 64");
            let v_bits = ((value >> value_bit_pos) & mask) as u32;
            let placed = v_bits << lo;
            let cell_mask = (mask as u32) << lo;

            let idx = cell_idx as usize;
            self.cells[idx] = (self.cells[idx] & !cell_mask) | placed;

            value_bit_pos += n;
        }
    }

    // ── Public cell-based API ─────────────────────────────────────────────────

    /// Borrow the flat cell array for PIR backend access.
    ///
    /// # Purpose
    ///
    /// Hand the PIR matvec a contiguous, alignment-friendly slice without a
    /// copy. The caller treats this as a row-major plaintext matrix of shape
    /// `num_slots × cells_per_slot`.
    ///
    /// # Returns
    ///
    /// A shared slice of length `num_slots · cells_per_slot`. Every cell
    /// carries at most `plaintext_bits` payload bits in its low bits; the
    /// high `(32 − plaintext_bits)` bits are guaranteed zero.
    pub(crate) fn as_cells(&self) -> &[u32] {
        &self.cells
    }

    /// Read the fingerprint stored at `(bucket, slot)`.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index, `< num_buckets`.
    /// - `slot`   — slot index within the bucket, `< bucket_size`.
    ///
    /// # Constraints
    ///
    /// Panics if `bucket >= num_buckets` or `slot >= bucket_size`.
    ///
    /// # Returns
    ///
    /// The stored fingerprint (`0` if the slot is empty).
    ///
    /// # Complexity
    ///
    /// `O(⌈fingerprint_bits / plaintext_bits⌉)` cell reads.
    pub(crate) fn read_fingerprint(&self, bucket: u32, slot: u32) -> u64 {
        self.assert_in_range(bucket, slot);
        self.read_bits(self.slot_bit_offset(bucket, slot), self.fingerprint_bits)
    }

    /// Overwrite the fingerprint at `(bucket, slot)`.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index, `< num_buckets`.
    /// - `slot`   — slot index within the bucket, `< bucket_size`.
    /// - `fp`     — new fingerprint (`0` marks the slot empty).
    ///
    /// # Constraints
    ///
    /// Panics if `bucket >= num_buckets` or `slot >= bucket_size`. The value
    /// portion of the slot is **not** touched.
    ///
    /// # Complexity
    ///
    /// `O(⌈fingerprint_bits / plaintext_bits⌉)` cell read-modify-writes.
    pub(crate) fn write_fingerprint(&mut self, bucket: u32, slot: u32, fp: u64) {
        self.assert_in_range(bucket, slot);
        self.write_bits(
            self.slot_bit_offset(bucket, slot),
            self.fingerprint_bits,
            fp,
        );
    }

    /// Read all value cells of `(bucket, slot)` into `out`.
    ///
    /// # Purpose
    ///
    /// Zero-allocation reader used by the streaming `CuckooKVStore::get_into`
    /// path; the caller owns `out`.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index, `< num_buckets`.
    /// - `slot`   — slot index within the bucket, `< bucket_size`.
    /// - `out`    — destination slice; one entry per value cell. Each entry
    ///   is masked to its tail-bit width (the last cell may be ragged).
    ///
    /// # Constraints
    ///
    /// Panics if `bucket >= num_buckets`, `slot >= bucket_size`, or
    /// `out.len() != value_size_in_cells()`.
    ///
    /// # Complexity
    ///
    /// `O(value_size_in_cells())` cell reads.
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
            // Value cells are always <= plaintext_bits (<= 32) wide, so the
            // truncation to u32 is lossless — only fingerprint reads use the
            // full u64 range `read_bits` now supports.
            out[i as usize] = self.read_bits(off0 + (i as u64) * pb as u64, n) as u32;
        }
    }

    /// Overwrite the value cells of `(bucket, slot)`.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index, `< num_buckets`.
    /// - `slot`   — slot index within the bucket, `< bucket_size`.
    /// - `value`  — `value_size_in_cells()` cells; each is masked to its
    ///   tail-bit width before write so stray high bits never escape into
    ///   the cell stream.
    ///
    /// # Constraints
    ///
    /// Panics if `bucket >= num_buckets`, `slot >= bucket_size`, or
    /// `value.len() != value_size_in_cells()`. The fingerprint portion of the
    /// slot is **not** touched.
    ///
    /// # Complexity
    ///
    /// `O(value_size_in_cells())` cell read-modify-writes.
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
            self.write_bits(off0 + (i as u64) * pb as u64, n, masked as u64);
        }
    }

    /// Write `fingerprint ‖ value` into `(bucket, slot)` in one call.
    ///
    /// # Purpose
    ///
    /// Convenience wrapper for the common "drop a complete entry into a
    /// slot" path, used by both `insert` and the cuckoo kick chain.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index, `< num_buckets`.
    /// - `slot`   — slot index within the bucket, `< bucket_size`.
    /// - `fp`     — fingerprint (non-zero for occupied slots).
    /// - `value`  — `value_size_in_cells()` cells.
    ///
    /// # Constraints
    ///
    /// Panics if `bucket >= num_buckets`, `slot >= bucket_size`, or
    /// `value.len() != value_size_in_cells()`.
    ///
    /// # Rationale
    ///
    /// Equivalent to [`write_fingerprint`](Self::write_fingerprint) followed
    /// by [`write_value`](Self::write_value), but inlined to avoid a second
    /// bounds check.
    ///
    /// # Complexity
    ///
    /// `O(cells_per_slot)` cell read-modify-writes.
    pub(crate) fn write(&mut self, bucket: u32, slot: u32, fp: u64, value: &[u32]) {
        self.write_fingerprint(bucket, slot, fp);
        self.write_value(bucket, slot, value);
    }

    /// Test whether `fp` is present in any slot of `bucket`.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index, `< num_buckets`.
    /// - `fp`     — fingerprint to probe. `fp == 0` is allowed and matches
    ///   any empty slot (the caller must filter out the empty case if that
    ///   is undesirable).
    ///
    /// # Constraints
    ///
    /// Panics if `bucket >= num_buckets`.
    ///
    /// # Returns
    ///
    /// `true` if some slot in `bucket` has fingerprint `fp`, else `false`.
    ///
    /// # Complexity
    ///
    /// `O(bucket_size · ⌈fingerprint_bits / plaintext_bits⌉)` cell reads.
    pub(crate) fn contain(&self, bucket: u32, fp: u64) -> bool {
        (0..self.bucket_size).any(|s| self.read_fingerprint(bucket, s) == fp)
    }

    /// Locate the first slot in `bucket` carrying fingerprint `fp`.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index, `< num_buckets`.
    /// - `fp`     — fingerprint to locate. `fp == 0` matches empty slots.
    ///
    /// # Constraints
    ///
    /// Panics if `bucket >= num_buckets`.
    ///
    /// # Returns
    ///
    /// `Some(slot)` for the first matching slot in scan order, else `None`.
    ///
    /// # Complexity
    ///
    /// `O(bucket_size · ⌈fingerprint_bits / plaintext_bits⌉)` worst case;
    /// short-circuits on first match.
    pub(crate) fn find(&self, bucket: u32, fp: u64) -> Option<u32> {
        (0..self.bucket_size).find(|&s| self.read_fingerprint(bucket, s) == fp)
    }

    /// Insert `fingerprint ‖ value` into the first empty slot of `bucket`.
    ///
    /// # Purpose
    ///
    /// Direct insertion — does **not** perform cuckoo kicking. The caller
    /// (`CuckooKVStore::insert`) drives the kick chain when this returns
    /// `None`.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index, `< num_buckets`.
    /// - `fp`     — fingerprint to write. Must be non-zero.
    /// - `value`  — value cells; length must equal `value_size_in_cells()`.
    ///
    /// # Constraints
    ///
    /// Panics if `bucket >= num_buckets`, `fp == 0`, or
    /// `value.len() != value_size_in_cells()`.
    ///
    /// # Returns
    ///
    /// `Some(slot)` if an empty slot was found and written, `None` if every
    /// slot in `bucket` is already occupied.
    ///
    /// # Complexity
    ///
    /// `O(bucket_size + cells_per_slot)` — scan for empty slot, then one
    /// `write`.
    pub(crate) fn insert(&mut self, bucket: u32, fp: u64, value: &[u32]) -> Option<u32> {
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

    /// Delete the first occurrence of `fp` in `bucket` and zero its slot.
    ///
    /// # Purpose
    ///
    /// Removes both the fingerprint and value bits so a subsequent
    /// [`read_value`](Self::read_value) on the now-empty slot returns all
    /// zeros — required for PIR matvec correctness on freed slots.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index, `< num_buckets`.
    /// - `fp`     — fingerprint to delete. Must be non-zero.
    ///
    /// # Constraints
    ///
    /// Panics if `bucket >= num_buckets` or `fp == 0`.
    ///
    /// # Returns
    ///
    /// `true` if a matching slot was found and cleared, `false` otherwise.
    ///
    /// # Complexity
    ///
    /// `O(bucket_size + cells_per_slot)` on hit, `O(bucket_size)` on miss.
    #[allow(dead_code)]
    pub(crate) fn delete(&mut self, bucket: u32, fp: u64) -> bool {
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
    /// # Purpose
    ///
    /// Snapshot-restore path used by `CuckooKVStore::from_cells`: lets a
    /// caller persist `as_cells().to_vec()`, then later rebuild the table
    /// without replaying every insert.
    ///
    /// # Arguments
    ///
    /// - `cells`            — flat cell array previously obtained from
    ///   [`as_cells`](Self::as_cells); length must equal
    ///   `num_buckets · bucket_size · cells_per_slot`.
    /// - `num_buckets`      — table shape (must match the snapshot).
    /// - `bucket_size`      — table shape (must match the snapshot).
    /// - `fingerprint_bits` — table shape (must match the snapshot).
    /// - `value_bits`       — table shape (must match the snapshot).
    /// - `plaintext_bits`   — table shape (must match the snapshot).
    ///
    /// # Constraints
    ///
    /// Validates `fingerprint_bits ∈ 1..=64`, `cells.len()`, and the
    /// high-bits-zero invariant
    /// (`cell & ~((1 << plaintext_bits) - 1) == 0`) on every cell. Cheap
    /// integrity guard against a corrupt or wrong-shape snapshot.
    ///
    /// # Returns
    ///
    /// `Ok(table)` on success, `Err(&'static str)` on a width out of range,
    /// length mismatch, or invariant violation.
    ///
    /// # Complexity
    ///
    /// `O(cells.len())` — a single linear scan; designed for snapshot-restore
    /// only, not the hot path.
    pub(crate) fn from_cells(
        cells: Vec<u32>,
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        plaintext_bits: u32,
    ) -> Result<Self, &'static str> {
        if !(1..=64).contains(&fingerprint_bits) {
            return Err("fingerprint_bits must be in 1..=64");
        }
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
                return Err(
                    "cell has high bits set (ChalametPIR high-bits-zero invariant violated)",
                );
            }
        }
        Ok(Self {
            cells,
            num_buckets,
            bucket_size,
            fingerprint_bits,
            value_bits,
            plaintext_bits,
            cells_per_slot,
        })
    }

    /// Read a contiguous chunk of value cells starting at intra-slot index
    /// `cell_start`.
    ///
    /// # Purpose
    ///
    /// Streaming primitive behind `CuckooKVStore::get_into`: lets the caller
    /// pull the value out one chunk at a time into a caller-owned buffer
    /// (no heap allocation per call).
    ///
    /// # Arguments
    ///
    /// - `bucket`     — bucket index, `< num_buckets`.
    /// - `slot`       — slot index within the bucket, `< bucket_size`.
    /// - `cell_start` — intra-slot cell index (counting from the start of
    ///   the value portion, i.e. cell `0` is the cell containing
    ///   `value_bits[0..plaintext_bits]`).
    /// - `out`        — destination slice, written cell-by-cell. Each
    ///   produced cell holds
    ///   `min(plaintext_bits, value_bits − cell_start · plaintext_bits)` bits.
    ///
    /// # Complexity
    ///
    /// `O(out.len())` cell reads.
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
            // Value cells are always <= plaintext_bits (<= 32) wide.
            *c = self.read_bits(off0 + ci as u64 * pb as u64, n) as u32;
        }
    }

    /// Read every value cell of `(bucket, slot)` into a freshly allocated
    /// boxed slice.
    ///
    /// # Purpose
    ///
    /// Used by the mutation log to capture the *old* cell state of a slot
    /// before a mutation overwrites it.
    ///
    /// # Arguments
    ///
    /// - `bucket` — bucket index, `< num_buckets`.
    /// - `slot`   — slot index within the bucket, `< bucket_size`.
    ///
    /// # Returns
    ///
    /// `Box<[u32]>` of length `value_size_in_cells()`. One allocation per
    /// call; mutation-log frequency is amortised by the
    /// `enable_mutation_log` opt-in.
    ///
    /// # Complexity
    ///
    /// `O(value_size_in_cells())` cell reads + one allocation.
    pub(crate) fn read_value_to_box(&self, bucket: u32, slot: u32) -> Box<[u32]> {
        let n = self.value_size_in_cells() as usize;
        let mut buf = vec![0u32; n];
        self.read_value(bucket, slot, &mut buf);
        buf.into_boxed_slice()
    }

    /// Replace the value of the first slot in `bucket` matching `fp`.
    ///
    /// # Purpose
    ///
    /// Direct in-place update path for `CuckooKVStore::update`; does **not**
    /// touch the fingerprint or move the entry.
    ///
    /// # Arguments
    ///
    /// - `bucket`    — bucket index, `< num_buckets`.
    /// - `fp`        — fingerprint of the target entry.
    /// - `new_value` — replacement value cells; length must equal
    ///   `value_size_in_cells()`.
    ///
    /// # Constraints
    ///
    /// Panics if `new_value.len() != value_size_in_cells()`.
    ///
    /// # Returns
    ///
    /// `true` on a successful update, `false` if `fp` was not in `bucket`.
    ///
    /// # Complexity
    ///
    /// `O(bucket_size + value_size_in_cells())` on hit, `O(bucket_size)` on
    /// miss.
    #[allow(dead_code)]
    pub(crate) fn update_value(&mut self, bucket: u32, fp: u64, new_value: &[u32]) -> bool {
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
    //! Unit tests for [`FingerprintValueTable`].
    //!
    //! Covers four invariants this storage layer must hold for the PIR
    //! matvec to remain correct:
    //!
    //! 1. **Roundtrip** — `write(fp, v); read_fingerprint() == fp;
    //!    read_value() == v` for every `(plaintext_bits, fingerprint_bits,
    //!    value_bits)` combination the IKPIR config matrix uses.
    //! 2. **High-bit-zero** — after any write, every cell of `as_cells()`
    //!    has zero in its top `(32 − plaintext_bits)` bits.
    //! 3. **Empty-slot semantics** — fresh tables have all-zero fingerprints;
    //!    `delete` returns to that state.
    //! 4. **Panic contracts** — every bounds / length / non-zero-fp
    //!    precondition is checked.

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

        // `u64` throughout (not `(1u64 << fp_bits) as u32`): fp_bits now
        // ranges up to 64, where `1u64 << 64` would overflow.
        let fp_mask: u64 = if fp_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << fp_bits) - 1
        };
        let cell_mask = if pb >= 32 { u32::MAX } else { (1u32 << pb) - 1 };

        for b in 0..nb {
            for s in 0..bs {
                let fp = (u64::from(b * bs + s + 1) & fp_mask).max(1);
                let value: Vec<u32> = (0..vcells)
                    .map(|i| ((b * 7 + s * 3 + i as u32 + 1) & cell_mask))
                    .collect();

                fvt.write(b, s, fp, &value);

                assert_eq!(
                    fvt.read_fingerprint(b, s),
                    fp,
                    "pb={pb} fp_bits={fp_bits} vb={vb} b={b} s={s}"
                );

                let mut got = vec![0u32; vcells];
                fvt.read_value(b, s, &mut got);
                for (i, (&g, &e)) in got.iter().zip(value.iter()).enumerate() {
                    let n = (vb - i as u32 * pb).min(pb);
                    let mask = ((1u64 << n) - 1) as u32;
                    assert_eq!(
                        g & mask,
                        e & mask,
                        "pb={pb} fp_bits={fp_bits} vb={vb} b={b} s={s} cell={i}"
                    );
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

    /// Sweeps a representative cross-product of `(plaintext_bits,
    /// fingerprint_bits, value_bits)` and asserts roundtrip + high-bit-zero
    /// after every write. The cross-product covers the IKPIR config matrix.
    #[test]
    fn cell_api_roundtrip_param() {
        for &pb in &[8u32, 9, 10, 12, 16, 32] {
            for &fp_bits in &[12u32, 16, 32] {
                for &vb in &[1u32, 8, 32, 64, 1024] {
                    if fp_bits > 32 || vb == 0 {
                        continue;
                    }
                    roundtrip_param(pb, fp_bits, vb);
                }
            }
        }
    }

    /// `fingerprint_bits = 64` with `plaintext_bits = 8`: the fingerprint
    /// alone occupies `⌈(64+8)/8⌉ = 9` cells — the widened "9-cell
    /// straddle" case `roundtrip_param` is already generic enough to cover.
    #[test]
    fn roundtrip_fp64_pb8_nine_cell_straddle() {
        roundtrip_param(8, 64, 8);
    }

    /// `fingerprint_bits = 33` — one bit past the old 32-bit cap.
    #[test]
    fn roundtrip_fp33() {
        roundtrip_param(8, 33, 8);
    }

    /// `fingerprint_bits = 64`, fingerprint value = all-ones (`u64::MAX`):
    /// exercises the exact top-of-range mask with no truncation.
    #[test]
    fn fp64_all_ones_value_roundtrips() {
        let mut fvt = make_fvt(4, 4, 64, 8, 8);
        let value = [0x11u32];
        fvt.write(0, 0, u64::MAX, &value);
        assert_eq!(fvt.read_fingerprint(0, 0), u64::MAX);
    }

    /// Even at `fingerprint_bits = 64`, an empty slot still reads back
    /// exactly `0` and stays distinguishable from every non-zero
    /// fingerprint — including the all-ones one — across an insert/delete
    /// cycle.
    #[test]
    fn empty_slot_still_distinguishable_at_f64() {
        let mut fvt = make_fvt(4, 4, 64, 8, 8);
        assert_eq!(fvt.read_fingerprint(0, 0), 0);
        let value = [0x22u32];
        assert_eq!(fvt.insert(0, u64::MAX, &value), Some(0));
        assert_eq!(fvt.read_fingerprint(0, 0), u64::MAX);
        assert!(fvt.delete(0, u64::MAX));
        assert_eq!(fvt.read_fingerprint(0, 0), 0);
    }

    /// A freshly constructed table has all slots empty (`fingerprint == 0`).
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

    /// `insert` followed by `find` returns the slot index; an absent
    /// fingerprint returns `None`.
    #[test]
    fn insert_then_find() {
        let mut fvt = make_fvt(4, 4, 12, 8, 8);
        let value = [0xABu32];
        assert_eq!(fvt.insert(0, 0x42, &value), Some(0));
        assert_eq!(fvt.find(0, 0x42), Some(0));
        assert_eq!(fvt.find(0, 0x99), None);
    }

    /// `insert` into a fully occupied bucket returns `None` (caller is
    /// expected to drive the cuckoo kick chain at the `CuckooKVStore` level).
    #[test]
    fn insert_into_full_bucket_returns_none() {
        let mut fvt = make_fvt(4, 2, 12, 8, 8);
        let v = [1u32];
        assert!(fvt.insert(0, 1, &v).is_some());
        assert!(fvt.insert(0, 2, &v).is_some());
        assert_eq!(fvt.insert(0, 3, &v), None); // full
    }

    /// `delete` zeros both the fingerprint and the value bits — important for
    /// PIR matvec correctness on freed slots.
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

    /// `update_value` overwrites only the value cells, leaving the
    /// fingerprint unchanged.
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

    /// The high-bit-zero invariant survives a representative random write
    /// pattern at `plaintext_bits = 9` (a value chosen because it forces
    /// straddling and ragged-tail bits in every slot).
    #[test]
    fn as_cells_high_bit_invariant_random_writes() {
        let pb = 9u32;
        let mut fvt = make_fvt(8, 4, 12, 64, pb);
        let vcells = fvt.value_size_in_cells() as usize;
        let hi_mask = !((1u32 << pb) - 1);
        for b in 0..8u32 {
            for s in 0..4u32 {
                let fp = u64::from((b * 4 + s + 1) % ((1 << 12) - 1) + 1);
                let value: Vec<u32> = (0..vcells)
                    .map(|i| (i as u32 * 13 + 7) & ((1 << pb) - 1))
                    .collect();
                fvt.write(b, s, fp, &value);
            }
        }
        for &c in fvt.as_cells() {
            assert_eq!(c & hi_mask, 0);
        }
    }

    /// `as_cells().len() == num_buckets · bucket_size · cells_per_slot` —
    /// pins the shape contract the PIR matvec relies on.
    #[test]
    fn as_cells_dimensions() {
        let nb = 16u32;
        let bs = 4u32;
        let pb = 10u32;
        let fvt = make_fvt(nb, bs, 12, 64, pb);
        let expected = nb as usize * bs as usize * fvt.cells_per_slot() as usize;
        assert_eq!(fvt.as_cells().len(), expected);
    }

    /// `read_bits` / `write_bits` correctly handle a fingerprint that
    /// straddles a cell boundary (`plaintext_bits=9`, `fingerprint_bits=12`).
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

    /// `read_bits` / `write_bits` correctly handle a value that straddles a
    /// cell boundary (`plaintext_bits=10`, `value_bits=14`).
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

    /// Wide values (`value_bits = 1024`) at `plaintext_bits = 8` roundtrip
    /// without truncation. Pins the byte-aligned hot path.
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

    /// Wide values at `plaintext_bits = 9` roundtrip — verifies ragged-tail
    /// masking in the last value cell and the high-bit-zero invariant
    /// simultaneously.
    #[test]
    fn wide_value_roundtrip_at_pb_9_vb_1024() {
        let pb = 9u32;
        let vb = 1024u32;
        let mut fvt = make_fvt(4, 4, 12, vb, pb);
        let vcells = fvt.value_size_in_cells() as usize;
        let hi_mask = !((1u32 << pb) - 1);
        let value: Vec<u32> = (0..vcells)
            .map(|i| ((i as u32 * 17 + 5) & ((1 << pb) - 1)))
            .collect();
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

    /// `insert` rejects a zero fingerprint (would collide with the
    /// empty-slot sentinel).
    #[test]
    #[should_panic(expected = "fingerprint cannot be zero")]
    fn insert_panics_on_zero_fp() {
        let mut fvt = make_fvt(4, 4, 12, 8, 8);
        let _ = fvt.insert(0, 0, &[0u32]);
    }

    /// `delete` rejects a zero fingerprint (matches the `insert` contract).
    #[test]
    #[should_panic(expected = "fingerprint cannot be zero")]
    fn delete_panics_on_zero_fp() {
        let mut fvt = make_fvt(4, 4, 12, 8, 8);
        let _ = fvt.delete(0, 0);
    }

    /// `read_fingerprint` panics on out-of-range bucket index.
    #[test]
    #[should_panic(expected = "out of range")]
    fn read_fingerprint_panics_on_oob_bucket() {
        let fvt = make_fvt(4, 4, 12, 8, 8);
        let _ = fvt.read_fingerprint(4, 0);
    }

    /// `read_fingerprint` panics on out-of-range slot index.
    #[test]
    #[should_panic(expected = "out of range")]
    fn read_fingerprint_panics_on_oob_slot() {
        let fvt = make_fvt(4, 4, 12, 8, 8);
        let _ = fvt.read_fingerprint(0, 4);
    }

    /// `read_value` panics if the destination slice has the wrong length.
    #[test]
    #[should_panic(expected = "out length must equal value_size_in_cells")]
    fn read_value_panics_on_wrong_length() {
        let fvt = make_fvt(4, 4, 12, 8, 8);
        let mut out = [0u32; 5];
        fvt.read_value(0, 0, &mut out);
    }

    /// `write_value` panics if the source slice has the wrong length.
    #[test]
    #[should_panic(expected = "value length must equal value_size_in_cells")]
    fn write_value_panics_on_wrong_length() {
        let mut fvt = make_fvt(4, 4, 12, 8, 8);
        let value = [0u32; 5];
        fvt.write_value(0, 0, &value);
    }
}
