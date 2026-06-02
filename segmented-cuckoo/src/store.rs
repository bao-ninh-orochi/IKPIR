//! `CuckooKVStore<S>` — cuckoo-filter-backed key-value store.
//!
//! # Purpose
//!
//! Upgrades a Cuckoo Filter into a key-value store: each slot carries
//! `fingerprint ‖ value` (laid out by
//! [`FingerprintValueTable`](crate::fingerprint_value_table)). The scheme
//! `S` determines arity and index layout (same [`IndexScheme`] variants as
//! [`CuckooFilter`](crate::CuckooFilter)). The store is the array that the
//! IKPIR server publishes to PIR queries.
//!
//! # Design / architecture
//!
//! - **Same kick + rollback machinery as `CuckooFilter`**, but the chain
//!   carries value cells too. On `TableFull`, every touched slot is
//!   restored to its pre-call state; on success, mutations are emitted to
//!   the (opt-in) mutation log.
//! - **Mutation log** records every committed slot write
//!   ([`SlotMutation`]) for IKPIR incremental hint patching. Disabled by
//!   default — opt in with [`CuckooKVStore::enable_mutation_log`].
//!   Rolled-back inserts emit nothing.
//! - **Zero-alloc hot path.** `get_into` streams the value through a stack
//!   buffer; the kick chain reuses `chain_meta` / `chain_values` /
//!   `cur_value` scratch.
//! - **Duplicate-key contract.** [`CuckooKVStore::insert`] does **not**
//!   deduplicate. [`get`](CuckooKVStore::get) /
//!   [`delete`](CuckooKVStore::delete) / [`update`](CuckooKVStore::update)
//!   all use **first-match-wins** semantics: candidate-bucket index
//!   `0..arity`, then lowest-index slot within that bucket. Purge
//!   duplicates by calling `delete` until it returns
//!   [`CuckooError::NotFound`].
//!
//! # Related files
//!
//! - `fingerprint_value_table.rs` — owned by this store; provides the
//!   cell-stream read/write primitives.
//! - `scheme.rs` / `hash.rs` — index arithmetic for each segmented scheme.
//! - `filter.rs` — sibling type that uses [`FingerprintTable`] instead and
//!   shares the kick / rollback strategy.
//! - `ikpir-server` — wraps this store in per-segment Index-PIR
//!   sub-databases via `as_cells()` + `drain_mutations()`.

// DECISION: indexed `for p in 0..arity` is used in the kicking loop because `p` indexes
// both the candidate-bucket array `all` and the chain-position bookkeeping. Mirrors the
// pattern in `filter.rs::insert`.
#![allow(clippy::needless_range_loop)]

use rand::Rng;

use crate::filter::{target_load_factor, validate_common_params, CuckooError, MAX_KICKS_DEFAULT};
use crate::fingerprint_value_table::FingerprintValueTable;
use crate::scheme::{
    IndexScheme, SchemeKind, SchemeMeta, Segmented2aryScheme, Segmented3aryScheme,
    Segmented4aryScheme,
};
use crate::util::next_power_of_2;

// ─── Public params bundle ────────────────────────────────────────────────────

/// Geometry parameters that fully describe a [`CuckooKVStore`]'s layout.
///
/// # Purpose
///
/// Server-to-client setup material. The IKPIR client receives this bundle
/// once and re-derives lookup positions on every query without holding a
/// live store. `Copy + Eq` and contains no heap data, so it ships across
/// process boundaries trivially.
///
/// # Rationale
///
/// Bundling the six shape parameters into one type lets the client compute
/// `candidate_buckets` and `slot_cell_range` from a single immutable
/// snapshot — there is no privacy reason to hide any of these values from
/// the client, and exposing them avoids a round-trip per query.
///
/// # Example
///
/// ```rust
/// use segmented_cuckoo::{Segmented2aryCuckooKVStore, CuckooParams, SchemeKind};
///
/// let store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
/// let p = store.params();
/// assert_eq!(p.scheme_kind, SchemeKind::Segmented2ary);
/// assert_eq!(p.arity(), 2);
/// assert_eq!(p.cells_per_slot(), (12 + 8_u32).div_ceil(8));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CuckooParams {
    /// Which segmented scheme is in use.
    pub scheme_kind: SchemeKind,
    /// Total number of buckets.
    pub num_buckets: u32,
    /// Slots per bucket.
    pub bucket_size: u32,
    /// Fingerprint bit width.
    pub fingerprint_bits: u32,
    /// Value bit width.
    pub value_bits: u32,
    /// PIR plaintext cell width.
    pub plaintext_bits: u32,
}

impl CuckooParams {
    /// Number of candidate buckets per item (2, 3, or 4).
    #[inline]
    pub const fn arity(&self) -> usize {
        match self.scheme_kind {
            SchemeKind::Segmented2ary => 2,
            SchemeKind::Segmented3ary => 3,
            SchemeKind::Segmented4ary => 4,
        }
    }

    /// Size of each segment: `num_buckets / arity()`.
    #[inline]
    pub const fn segment_size(&self) -> u32 {
        self.num_buckets / self.arity() as u32
    }

    /// Cells per slot: `⌈(fingerprint_bits + value_bits) / plaintext_bits⌉`.
    #[inline]
    pub const fn cells_per_slot(&self) -> u32 {
        (self.fingerprint_bits + self.value_bits).div_ceil(self.plaintext_bits)
    }

    /// Cells per value: `⌈value_bits / plaintext_bits⌉`.
    #[inline]
    pub const fn value_size_in_cells(&self) -> u32 {
        self.value_bits.div_ceil(self.plaintext_bits)
    }

    /// Bytes per stored value: `⌈value_bits / 8⌉`. Matches
    /// [`CuckooKVStore::value_size_in_bytes`] and the `Vec<u8>` length
    /// returned by [`unpack_slot_cells`].
    #[inline]
    pub const fn value_size_in_bytes(&self) -> usize {
        (self.value_bits as usize).div_ceil(8)
    }

    /// Total cells in the flat cell array: `num_buckets · bucket_size · cells_per_slot`.
    pub const fn size_in_cells(&self) -> usize {
        self.num_buckets as usize * self.bucket_size as usize * self.cells_per_slot() as usize
    }

    /// Flat-cell-array range covering slot `(bucket, slot)`.
    ///
    /// # Purpose
    ///
    /// Translate a `(bucket, slot)` coordinate into a cell-index range so
    /// the IKPIR PIR client can pull exactly the slot it needs after
    /// matvec.
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
    /// A `Range<usize>` of length `cells_per_slot`. Fingerprint occupies
    /// the leading `fingerprint_bits` bits of that cell stream; value
    /// occupies the next `value_bits` bits.
    pub fn slot_cell_range(&self, bucket: u32, slot: u32) -> std::ops::Range<usize> {
        assert!(
            bucket < self.num_buckets && slot < self.bucket_size,
            "(bucket, slot) = ({bucket}, {slot}) out of range [{}, {})",
            self.num_buckets,
            self.bucket_size,
        );
        let cps = self.cells_per_slot() as usize;
        let off = (bucket as usize * self.bucket_size as usize + slot as usize) * cps;
        off..off + cps
    }

    /// Re-derive `(fingerprint, indices)` for a key without holding the
    /// store.
    ///
    /// # Purpose
    ///
    /// IKPIR client entry point: given a key, recover the same fingerprint
    /// and candidate-bucket set the server would derive on
    /// `CuckooKVStore::insert(key, …)`.
    ///
    /// # Arguments
    ///
    /// - `key` — arbitrary bytes.
    ///
    /// # Returns
    ///
    /// `(fingerprint, indices)` where `indices[..arity()]` are the
    /// candidate buckets in `0..num_buckets` and `indices[arity()..]` are
    /// zero-padded and must be ignored.
    ///
    /// # Complexity
    ///
    /// One xxh3 hash of `key` plus `O(arity)` index arithmetic.
    pub fn candidate_buckets(&self, key: &[u8]) -> (u32, [u32; 4]) {
        match self.scheme_kind {
            SchemeKind::Segmented2ary => Segmented2aryScheme {
                segment_size: self.segment_size(),
            }
            .hash_item(key, self.fingerprint_bits),
            SchemeKind::Segmented3ary => Segmented3aryScheme {
                segment_size: self.segment_size(),
            }
            .hash_item(key, self.fingerprint_bits),
            SchemeKind::Segmented4ary => Segmented4aryScheme {
                segment_size: self.segment_size(),
            }
            .hash_item(key, self.fingerprint_bits),
        }
    }
}

// ─── Mutation log ────────────────────────────────────────────────────────────

/// A single committed slot write, recorded by the mutation log.
///
/// # Purpose
///
/// IKPIR server input to `fold_mutations_into_row_deltas` — captures
/// before/after state for one slot write so the PIR hint can be patched
/// incrementally instead of being rebuilt from scratch.
///
/// # Rationale
///
/// `old_value_cells` and `new_value_cells` each incur one heap allocation
/// per committed slot write. The log is opt-in (disabled by default), so
/// this cost only applies on the server steady-state path. Storing both
/// "before" and "after" cells (rather than just the diff) lets the patcher
/// compute `Δ = new − old` without re-reading the table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotMutation {
    /// Bucket index of the mutated slot.
    pub bucket: u32,
    /// Slot index within the bucket.
    pub slot: u32,
    /// Fingerprint before mutation (`0` = slot was empty).
    pub old_fingerprint: u32,
    /// Fingerprint after mutation (`0` = slot is now empty).
    pub new_fingerprint: u32,
    /// Value cells before mutation (`value_size_in_cells()` elements).
    pub old_value_cells: Box<[u32]>,
    /// Value cells after mutation (`value_size_in_cells()` elements).
    pub new_value_cells: Box<[u32]>,
}

// ─── Iteration surface ───────────────────────────────────────────────────────

/// An occupied slot, as yielded by [`CuckooKVStore::iter_occupied_slots`].
///
/// # Purpose
///
/// Diagnostic / batch-rebuild view over the store. The borrowed
/// `value_cells` slice lets the consumer decode without paying a re-read
/// through the table API.
pub struct OccupiedSlot<'a> {
    /// Bucket index.
    pub bucket: u32,
    /// Slot index within the bucket.
    pub slot: u32,
    /// Fingerprint stored in this slot (always non-zero — empty slots are
    /// skipped by the iterator).
    pub fingerprint: u32,
    /// Raw `cells_per_slot` cells for this slot, borrowed from the store's
    /// cell array.
    ///
    /// Fingerprint occupies the leading `fingerprint_bits` bits; value
    /// occupies the next `value_bits` bits. Pass to the same decode path
    /// as the PIR client (i.e. [`unpack_slot_cells`]).
    pub value_cells: &'a [u32],
}

/// Generic cuckoo key-value store parameterised over an index scheme.
///
/// # Purpose
///
/// Storage layer used by the IKPIR server: writes `(key, value)` pairs
/// into a cell-packed fingerprint-and-value array and exposes that array
/// directly via [`as_cells`](Self::as_cells) for PIR matvec.
///
/// # Rationale
///
/// Prefer the type aliases
/// [`Segmented2aryCuckooKVStore`](crate::Segmented2aryCuckooKVStore),
/// [`Segmented3aryCuckooKVStore`](crate::Segmented3aryCuckooKVStore), and
/// [`Segmented4aryCuckooKVStore`](crate::Segmented4aryCuckooKVStore) over
/// instantiating this type directly unless you are implementing a custom
/// scheme.
pub struct CuckooKVStore<S: IndexScheme> {
    table: FingerprintValueTable,
    scheme: S,
    num_items: u64,
    max_kicks: u32,
    /// Per-kick metadata reused across `insert` calls; capacity tracks `max_kicks`.
    chain_meta: Vec<(u32, u32, u32)>,
    /// Slab of `max_kicks * value_size_in_cells` u32 cells; kick k holds the original
    /// (evicted) value at the k-th kick, used for rollback.
    chain_values: Vec<u32>,
    /// Reusable buffer for the in-flight (displaced) value during kicking;
    /// length equals `value_size_in_cells`.
    cur_value: Vec<u32>,
    /// When `Some`, records every committed slot write for incremental PIR-hint patching.
    mutation_log: Option<Vec<SlotMutation>>,
    /// Scratch buffer for mutations during an `insert` kick chain; flushed to
    /// `mutation_log` on commit, dropped on rollback. No-alloc when log is disabled.
    pending_mutations: Vec<SlotMutation>,
}

// ─── Streaming value unpack (zero-allocation read path) ─────────────────────

const STREAM_CHUNK_CELLS: usize = 16;

/// Unpack value cells from `(bucket, slot)` directly into `out` without a
/// heap allocation.
///
/// Reads value cells in chunks of [`STREAM_CHUNK_CELLS`] onto the stack,
/// accumulating bits into a `u64` accumulator and flushing complete bytes
/// into `out`. Replaces the older approach of materialising an
/// intermediate `Vec<u32>` per call. Fast path is `plaintext_bits == 8` and
/// `value_bits % 8 == 0` — each cell is one byte.
fn unpack_value_streaming(
    table: &FingerprintValueTable,
    bucket: u32,
    slot: u32,
    out: &mut [u8],
    value_bits: u32,
    pb: u32,
    total_cells: usize,
) {
    if pb == 8 && value_bits % 8 == 0 {
        // Fast path: each cell is one byte.
        let mut scratch = [0u32; STREAM_CHUNK_CELLS];
        let mut remaining = total_cells;
        let mut cell_off = 0usize;
        let mut byte_off = 0usize;
        while remaining > 0 {
            let take = remaining.min(STREAM_CHUNK_CELLS);
            table.read_value_cells_chunk(bucket, slot, cell_off, &mut scratch[..take]);
            for &c in &scratch[..take] {
                out[byte_off] = c as u8;
                byte_off += 1;
            }
            cell_off += take;
            remaining -= take;
        }
        return;
    }

    // General path: accumulate bits across cells into bytes.
    let mut scratch = [0u32; STREAM_CHUNK_CELLS];
    let mut remaining = total_cells;
    let mut cell_off = 0usize;
    let mut acc: u64 = 0;
    let mut bits_in_acc: u32 = 0;
    let mut byte_off = 0usize;

    while remaining > 0 {
        let take = remaining.min(STREAM_CHUNK_CELLS);
        table.read_value_cells_chunk(bucket, slot, cell_off, &mut scratch[..take]);
        for (i, &c) in scratch[..take].iter().enumerate() {
            let ci = cell_off + i;
            let n = (value_bits - ci as u32 * pb).min(pb);
            acc |= (c as u64 & ((1u64 << n) - 1)) << bits_in_acc;
            bits_in_acc += n;
            while bits_in_acc >= 8 {
                if byte_off < out.len() {
                    out[byte_off] = acc as u8;
                    byte_off += 1;
                }
                acc >>= 8;
                bits_in_acc -= 8;
            }
        }
        cell_off += take;
        remaining -= take;
    }

    // Flush any remaining partial byte (when value_bits is not a multiple of 8 and
    // the last accumulator fill didn't trigger the ≥8 threshold).
    if bits_in_acc > 0 && byte_off < out.len() {
        out[byte_off] = acc as u8;
    }

    if value_bits % 8 != 0 {
        if let Some(last) = out.last_mut() {
            *last &= (1u8 << (value_bits % 8)) - 1;
        }
    }
}

// ─── Private byte↔cell conversion helpers ───────────────────────────────────

/// Pack `value_bits` bits from `bytes` into `cells` using `plaintext_bits`-wide cells.
///
/// Little-endian byte order, LSB-first within each byte. Fast path is
/// `pb == 8` and `value_bits % 8 == 0`: each cell is a single byte cast to
/// `u32`. General path streams bytes through an accumulator and emits
/// `pb`-bit cells.
fn pack_value_bytes_to_cells(bytes: &[u8], cells: &mut [u32], value_bits: u32, pb: u32) {
    if pb == 8 && value_bits % 8 == 0 {
        for (c, &b) in cells.iter_mut().zip(bytes.iter()) {
            *c = b as u32;
        }
        return;
    }
    // General path: stream bytes through an accumulator, emit pb-bit cells.
    let mut acc: u64 = 0u64;
    let mut bits_in_acc: u32 = 0;
    let mut byte_iter = bytes.iter().copied();
    for (i, c) in cells.iter_mut().enumerate() {
        // Fill accumulator until we have at least pb bits (or bytes are exhausted).
        while bits_in_acc < pb {
            if let Some(b) = byte_iter.next() {
                acc |= (b as u64) << bits_in_acc;
                bits_in_acc += 8;
            } else {
                break;
            }
        }
        let n = (value_bits - i as u32 * pb).min(pb);
        let mask = (1u64 << n) - 1;
        *c = (acc & mask) as u32;
        acc >>= n;
        bits_in_acc = bits_in_acc.saturating_sub(n);
    }
}

// ─── Public slot pack / unpack ───────────────────────────────────────────────

/// Read up to 32 bits from a standalone cell slice at `bit_offset` using
/// `pb`-bit cells.
#[inline]
const fn slot_read_bits(cells: &[u32], bit_offset: u64, n_bits: u32, pb: u32) -> u32 {
    let pb64 = pb as u64;
    let mut cell_idx = (bit_offset / pb64) as usize;
    let intra_start = (bit_offset % pb64) as u32;

    let mut acc: u64 = 0;
    let v = cells[cell_idx] as u64;
    acc |= v >> intra_start;
    let mut filled = pb - intra_start;
    cell_idx += 1;

    while filled < n_bits {
        let v = cells[cell_idx] as u64;
        acc |= v << filled;
        filled += pb;
        cell_idx += 1;
    }

    (acc & ((1u64 << n_bits) - 1)) as u32
}

/// Write `n_bits` low bits of `value` into a standalone cell slice at
/// `bit_offset`.
#[inline]
fn slot_write_bits(cells: &mut [u32], bit_offset: u64, n_bits: u32, value: u32, pb: u32) {
    let pb64 = pb as u64;
    let first_cell = (bit_offset / pb64) as usize;
    let last_cell = ((bit_offset + n_bits as u64).div_ceil(pb64)) as usize;

    let mut value_bit_pos: u32 = 0;

    for cell_idx in first_cell..last_cell {
        let cell_base_bit = cell_idx as u64 * pb64;

        let lo = if bit_offset > cell_base_bit {
            (bit_offset - cell_base_bit) as u32
        } else {
            0
        };
        let hi = {
            let end_bit = bit_offset + n_bits as u64;
            let cell_end = cell_base_bit + pb64;
            (end_bit.min(cell_end) - cell_base_bit) as u32
        };
        let n = hi - lo;

        let mask = ((1u64 << n) - 1) as u32;
        let v_bits = (value >> value_bit_pos) & mask;
        let placed = v_bits << lo;
        let cell_mask = mask << lo;

        cells[cell_idx] = (cells[cell_idx] & !cell_mask) | placed;

        value_bit_pos += n;
    }
}

/// Pack `fingerprint ‖ value_cells` into `out`, byte-identical to the slot
/// payload that [`CuckooKVStore::insert`] would write at any
/// `(bucket, slot)`.
///
/// # Purpose
///
/// Lets a caller construct a slot payload off-store (e.g. when constructing
/// a synthetic `cells` array for `from_cells`, or when testing the PIR
/// decode path without a live store).
///
/// # Arguments
///
/// - `params`      — geometry of the target store.
/// - `fingerprint` — fingerprint to place in the leading bits.
/// - `value_cells` — value cells; length must equal
///   `params.value_size_in_cells()`.
/// - `out`         — destination cell slice; length must equal
///   `params.cells_per_slot()`. Zeroed by this call before write.
///
/// # Constraints
///
/// Panics if `out.len() != params.cells_per_slot()` or
/// `value_cells.len() != params.value_size_in_cells()`.
///
/// # Complexity
///
/// `O(params.cells_per_slot())` cell writes.
pub fn pack_slot_cells(
    params: &CuckooParams,
    fingerprint: u32,
    value_cells: &[u32],
    out: &mut [u32],
) {
    let cps = params.cells_per_slot() as usize;
    assert_eq!(out.len(), cps, "out.len() must equal cells_per_slot()");
    let v_cells = params.value_size_in_cells() as usize;
    assert_eq!(
        value_cells.len(),
        v_cells,
        "value_cells.len() must equal value_size_in_cells()"
    );

    out.fill(0);
    let pb = params.plaintext_bits;
    slot_write_bits(out, 0, params.fingerprint_bits, fingerprint, pb);

    let off0 = params.fingerprint_bits as u64;
    for i in 0..v_cells {
        let n = (params.value_bits - i as u32 * pb).min(pb);
        let mask = ((1u64 << n) - 1) as u32;
        let masked = value_cells[i] & mask;
        slot_write_bits(out, off0 + (i as u64) * pb as u64, n, masked, pb);
    }
}

/// Inverse of [`pack_slot_cells`].
///
/// # Purpose
///
/// Used by the IKPIR client (`ikpir-client::IkpirClient::decode`) to
/// extract `(fingerprint, value_bytes)` from a PIR-decoded row slice.
///
/// # Arguments
///
/// - `params`     — geometry of the source store.
/// - `slot_cells` — `cells_per_slot()` cells worth of one slot's payload.
///
/// # Returns
///
/// `(fingerprint, value_bytes)` where `value_bytes.len() == ⌈value_bits / 8⌉`.
///
/// # Complexity
///
/// `O(params.cells_per_slot())` cell reads.
pub fn unpack_slot_cells(params: &CuckooParams, slot_cells: &[u32]) -> (u32, Vec<u8>) {
    let pb = params.plaintext_bits;
    let fp_bits = params.fingerprint_bits;
    let vb = params.value_bits;
    let v_cells = params.value_size_in_cells() as usize;
    let vbytes = vb.div_ceil(8) as usize;

    let fp = slot_read_bits(slot_cells, 0, fp_bits, pb);

    let val_start_bit = fp_bits as u64;
    let mut acc: u64 = 0;
    let mut bits_in_acc: u32 = 0;
    let mut out = vec![0u8; vbytes];
    let mut byte_off = 0usize;

    for ci in 0..v_cells {
        let n = (vb - ci as u32 * pb).min(pb);
        let bit_off = val_start_bit + ci as u64 * pb as u64;
        let c = slot_read_bits(slot_cells, bit_off, n, pb);
        acc |= (c as u64) << bits_in_acc;
        bits_in_acc += n;
        while bits_in_acc >= 8 && byte_off < vbytes {
            out[byte_off] = acc as u8;
            byte_off += 1;
            acc >>= 8;
            bits_in_acc -= 8;
        }
    }

    if bits_in_acc > 0 && byte_off < vbytes {
        out[byte_off] = acc as u8;
    }

    if vb % 8 != 0 {
        if let Some(last) = out.last_mut() {
            *last &= (1u8 << (vb % 8)) - 1;
        }
    }

    (fp, out)
}

// ─── Constructors ───────────────────────────────────────────────────────────

impl CuckooKVStore<Segmented2aryScheme> {
    /// Create a segmented 2-ary KV store with explicit dimensions.
    ///
    /// # Arguments
    ///
    /// - `num_buckets`      — total number of buckets. Must be a power of 2 and ≥ 2.
    /// - `bucket_size`      — slots per bucket. Must be in `1..=4`.
    /// - `fingerprint_bits` — fingerprint bit width. Must be in
    ///   `[⌊log2(2·bucket_size)⌋+1, 32]`.
    /// - `value_bits`       — value bit width. Must be ≥ 1.
    /// - `plaintext_bits`   — PIR plaintext cell width (1–32). Recommended
    ///   `8` for byte-typed values (byte↔cell becomes a no-op when
    ///   `value_bits % 8 == 0`); `9`/`10` for tighter PIR matvec when
    ///   caller pre-packs.
    ///
    /// # Constraints
    ///
    /// Each parameter must satisfy the bound above; returns
    /// [`CuckooError::InvalidParams`] otherwise. **Degenerate mode**:
    /// `num_buckets = 2` produces `segment_size = 1`, in which case every
    /// key maps to fixed indices `[0, 1]` and FPR ≈ 1 once past
    /// `bucket_size` items. Functionally correct but only useful for
    /// minimal-table tests.
    ///
    /// # Returns
    ///
    /// `Ok(store)` with `num_items = 0` and mutation log disabled, or
    /// `Err(CuckooError::InvalidParams)` if any constraint is violated.
    ///
    /// # Complexity
    ///
    /// `O(num_buckets · bucket_size · cells_per_slot)` time and memory
    /// (one zero-initialised `Vec<u32>` allocation).
    pub fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        plaintext_bits: u32,
    ) -> Result<Self, CuckooError> {
        if !num_buckets.is_power_of_two() || num_buckets < 2 {
            return Err(CuckooError::InvalidParams(
                "num_buckets must be a power of 2 and >= 2".into(),
            ));
        }
        validate_common_params(2, bucket_size, fingerprint_bits)?;
        if value_bits == 0 {
            return Err(CuckooError::InvalidParams("value_bits must be >= 1".into()));
        }
        if !(1..=32).contains(&plaintext_bits) {
            return Err(CuckooError::InvalidParams(
                "plaintext_bits must be in 1..=32".into(),
            ));
        }
        let table = FingerprintValueTable::new(
            num_buckets,
            bucket_size,
            fingerprint_bits,
            value_bits,
            plaintext_bits,
        );
        let vcells = table.value_size_in_cells() as usize;
        Ok(Self {
            table,
            scheme: Segmented2aryScheme {
                segment_size: num_buckets / 2,
            },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
            chain_meta: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
            chain_values: vec![
                0u32;
                (MAX_KICKS_DEFAULT as usize)
                    .checked_mul(vcells)
                    .expect("max_kicks * value_size_in_cells overflows usize")
            ],
            cur_value: vec![0u32; vcells],
            mutation_log: None,
            pending_mutations: Vec::new(),
        })
    }

    /// Create a segmented 2-ary KV store sized to hold at least `max_items`.
    ///
    /// # Purpose
    ///
    /// Capacity-driven constructor: lets the caller specify a target item
    /// count instead of computing `num_buckets` manually.
    ///
    /// # Arguments
    ///
    /// - `max_items`        — target item count.
    /// - `bucket_size`      — slots per bucket; same constraints as
    ///   [`new`](Self::new).
    /// - `fingerprint_bits` — same constraints as [`new`](Self::new).
    /// - `value_bits`       — same constraints as [`new`](Self::new).
    /// - `plaintext_bits`   — same constraints as [`new`](Self::new).
    ///
    /// # Rationale
    ///
    /// Rounds `num_buckets` up to the next valid value (power of 2, ≥ 2).
    /// If the projected load factor exceeds the empirical target for this
    /// `(arity=2, bucket_size)` configuration (see
    /// [`crate::MAX_LOAD_FACTOR`]), `num_buckets` is doubled until the
    /// projection is within bounds.
    ///
    /// # Returns
    ///
    /// `Ok(store)` on success, `Err(CuckooError::InvalidParams)` if any
    /// scalar constraint is violated or `max_items` exceeds the `u32`
    /// num-buckets cap.
    pub fn from_num_items(
        max_items: u64,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        plaintext_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(2, bucket_size, fingerprint_bits)?;
        if value_bits == 0 {
            return Err(CuckooError::InvalidParams("value_bits must be >= 1".into()));
        }
        let max_items_cap: u64 = u32::MAX as u64 * bucket_size as u64;
        if max_items > max_items_cap {
            return Err(CuckooError::InvalidParams(
                "max_items too large for u32 num_buckets".into(),
            ));
        }
        let buckets_needed = max_items.div_ceil(bucket_size as u64);
        let mut num_buckets = next_power_of_2(buckets_needed) as u32;
        if num_buckets < 2 {
            num_buckets = 2;
        }
        let target = target_load_factor(2, bucket_size as usize);
        while max_items as f64 / (num_buckets as u64 * bucket_size as u64) as f64 > target {
            num_buckets *= 2;
        }
        Self::new(
            num_buckets,
            bucket_size,
            fingerprint_bits,
            value_bits,
            plaintext_bits,
        )
    }
}

impl CuckooKVStore<Segmented3aryScheme> {
    /// Create a segmented 3-ary KV store with explicit dimensions.
    ///
    /// # Arguments
    ///
    /// - `num_buckets`      — total buckets. Must equal `3 · 2^t` for some
    ///   `t ≥ 0` (so each of the 3 segments is a power of 2).
    /// - `bucket_size`      — slots per bucket. Must be in `1..=4`.
    /// - `fingerprint_bits` — fingerprint bit width. Must be in
    ///   `[⌊log2(3·bucket_size)⌋+1, 32]`.
    /// - `value_bits`       — value bit width. Must be ≥ 1.
    /// - `plaintext_bits`   — PIR plaintext cell width (1–32).
    ///
    /// # Returns
    ///
    /// `Ok(store)` with `num_items = 0` and mutation log disabled, or
    /// `Err(CuckooError::InvalidParams)` if any constraint is violated.
    ///
    /// # Complexity
    ///
    /// `O(num_buckets · bucket_size · cells_per_slot)` time and memory.
    pub fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        plaintext_bits: u32,
    ) -> Result<Self, CuckooError> {
        if num_buckets < 3 || num_buckets % 3 != 0 || !(num_buckets / 3).is_power_of_two() {
            return Err(CuckooError::InvalidParams(
                "num_buckets must be 3 * 2^t (num_buckets divisible by 3, num_buckets/3 a power of 2)".into(),
            ));
        }
        validate_common_params(3, bucket_size, fingerprint_bits)?;
        if value_bits == 0 {
            return Err(CuckooError::InvalidParams("value_bits must be >= 1".into()));
        }
        if !(1..=32).contains(&plaintext_bits) {
            return Err(CuckooError::InvalidParams(
                "plaintext_bits must be in 1..=32".into(),
            ));
        }
        let segment_size = num_buckets / 3;
        let table = FingerprintValueTable::new(
            num_buckets,
            bucket_size,
            fingerprint_bits,
            value_bits,
            plaintext_bits,
        );
        let vcells = table.value_size_in_cells() as usize;
        Ok(Self {
            table,
            scheme: Segmented3aryScheme { segment_size },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
            chain_meta: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
            chain_values: vec![
                0u32;
                (MAX_KICKS_DEFAULT as usize)
                    .checked_mul(vcells)
                    .expect("max_kicks * value_size_in_cells overflows usize")
            ],
            cur_value: vec![0u32; vcells],
            mutation_log: None,
            pending_mutations: Vec::new(),
        })
    }

    /// Create a segmented 3-ary KV store sized to hold at least `max_items`.
    ///
    /// # Rationale
    ///
    /// Each segment must be a power of 2; this method picks the smallest
    /// valid `num_buckets = 3 · segment_size` with
    /// `segment_size = 2^t ≥ ⌈max_items / (3 · bucket_size)⌉`, then doubles
    /// `segment_size` until the projected load factor is within the
    /// empirical target for `(arity=3, bucket_size)`.
    ///
    /// # Arguments / Returns
    ///
    /// Same shape as [`from_num_items`](CuckooKVStore::<Segmented2aryScheme>::from_num_items).
    pub fn from_num_items(
        max_items: u64,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        plaintext_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(3, bucket_size, fingerprint_bits)?;
        if value_bits == 0 {
            return Err(CuckooError::InvalidParams("value_bits must be >= 1".into()));
        }
        let max_items_cap: u64 = u32::MAX as u64 * bucket_size as u64;
        if max_items > max_items_cap {
            return Err(CuckooError::InvalidParams(
                "max_items too large for u32 num_buckets".into(),
            ));
        }
        let slots_per_segment = max_items.div_ceil(3 * bucket_size as u64);
        let mut segment_size = next_power_of_2(slots_per_segment) as u32;
        if segment_size < 1 {
            segment_size = 1;
        }
        let mut num_buckets = 3 * segment_size;
        let target = target_load_factor(3, bucket_size as usize);
        while max_items as f64 / (num_buckets as u64 * bucket_size as u64) as f64 > target {
            segment_size *= 2;
            num_buckets = 3 * segment_size;
        }
        Self::new(
            num_buckets,
            bucket_size,
            fingerprint_bits,
            value_bits,
            plaintext_bits,
        )
    }
}

impl CuckooKVStore<Segmented4aryScheme> {
    /// Create a segmented 4-ary KV store with explicit dimensions.
    ///
    /// # Arguments
    ///
    /// - `num_buckets`      — total buckets. Must be a power of 2 and ≥ 4
    ///   (so each of the 4 segments is a power of 2).
    /// - `bucket_size`      — slots per bucket. Must be in `1..=4`.
    /// - `fingerprint_bits` — fingerprint bit width. Must be in
    ///   `[⌊log2(4·bucket_size)⌋+1, 32]`.
    /// - `value_bits`       — value bit width. Must be ≥ 1.
    /// - `plaintext_bits`   — PIR plaintext cell width (1–32).
    ///
    /// # Returns
    ///
    /// `Ok(store)` with `num_items = 0` and mutation log disabled, or
    /// `Err(CuckooError::InvalidParams)` if any constraint is violated.
    pub fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        plaintext_bits: u32,
    ) -> Result<Self, CuckooError> {
        if !num_buckets.is_power_of_two() || num_buckets < 4 {
            return Err(CuckooError::InvalidParams(
                "num_buckets must be a power of 2 and >= 4 (so each of 4 segments is a power of 2)"
                    .into(),
            ));
        }
        validate_common_params(4, bucket_size, fingerprint_bits)?;
        if value_bits == 0 {
            return Err(CuckooError::InvalidParams("value_bits must be >= 1".into()));
        }
        if !(1..=32).contains(&plaintext_bits) {
            return Err(CuckooError::InvalidParams(
                "plaintext_bits must be in 1..=32".into(),
            ));
        }
        let segment_size = num_buckets / 4;
        let table = FingerprintValueTable::new(
            num_buckets,
            bucket_size,
            fingerprint_bits,
            value_bits,
            plaintext_bits,
        );
        let vcells = table.value_size_in_cells() as usize;
        Ok(Self {
            table,
            scheme: Segmented4aryScheme { segment_size },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
            chain_meta: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
            chain_values: vec![
                0u32;
                (MAX_KICKS_DEFAULT as usize)
                    .checked_mul(vcells)
                    .expect("max_kicks * value_size_in_cells overflows usize")
            ],
            cur_value: vec![0u32; vcells],
            mutation_log: None,
            pending_mutations: Vec::new(),
        })
    }

    /// Create a segmented 4-ary KV store sized to hold at least `max_items`.
    ///
    /// # Rationale
    ///
    /// Each segment must be a power of 2; this method picks the smallest
    /// valid `num_buckets = 4 · segment_size`, then doubles `segment_size`
    /// until the projected load factor is within the empirical target for
    /// `(arity=4, bucket_size)`.
    ///
    /// # Arguments / Returns
    ///
    /// Same shape as [`from_num_items`](CuckooKVStore::<Segmented2aryScheme>::from_num_items).
    pub fn from_num_items(
        max_items: u64,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        plaintext_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(4, bucket_size, fingerprint_bits)?;
        if value_bits == 0 {
            return Err(CuckooError::InvalidParams("value_bits must be >= 1".into()));
        }
        let max_items_cap: u64 = u32::MAX as u64 * bucket_size as u64;
        if max_items > max_items_cap {
            return Err(CuckooError::InvalidParams(
                "max_items too large for u32 num_buckets".into(),
            ));
        }
        let slots_per_segment = max_items.div_ceil(4 * bucket_size as u64);
        let mut segment_size = next_power_of_2(slots_per_segment) as u32;
        if segment_size < 1 {
            segment_size = 1;
        }
        let mut num_buckets = 4 * segment_size;
        let target = target_load_factor(4, bucket_size as usize);
        while max_items as f64 / (num_buckets as u64 * bucket_size as u64) as f64 > target {
            segment_size *= 2;
            num_buckets = 4 * segment_size;
        }
        Self::new(
            num_buckets,
            bucket_size,
            fingerprint_bits,
            value_bits,
            plaintext_bits,
        )
    }
}

// ─── Generic operations ─────────────────────────────────────────────────────

impl<S: IndexScheme> CuckooKVStore<S> {
    #[inline]
    const fn value_size_in_cells(&self) -> usize {
        self.table.value_size_in_cells() as usize
    }

    /// Bytes needed to hold one stored value: `⌈value_bits / 8⌉`.
    #[inline]
    pub const fn value_size_in_bytes(&self) -> usize {
        (self.table.value_bits() as usize).div_ceil(8)
    }

    /// Insert `(key, value)` into the store.
    ///
    /// # Purpose
    ///
    /// Primary write path. Tries direct placement in any of the arity
    /// candidate buckets; if all are full, drives a cuckoo-kicking loop
    /// that evicts and relocates up to `max_kicks` `(fingerprint, value)`
    /// pairs. On exhaustion every touched slot is rolled back and
    /// [`CuckooError::TableFull`] is returned (the store is observably
    /// unchanged).
    ///
    /// # Arguments
    ///
    /// - `key`   — arbitrary bytes identifying the entry.
    /// - `value` — bytes to store. Length must equal
    ///   [`value_size_in_bytes()`](Self::value_size_in_bytes).
    ///
    /// # Constraints
    ///
    /// Panics if `value.len() != value_size_in_bytes()`. Once the kicking
    /// loop has begun, a panic from any source leaves the store in an
    /// inconsistent state (matches
    /// [`CuckooFilter::insert`](crate::CuckooFilter::insert) behaviour).
    ///
    /// # Rationale
    ///
    /// **Duplicates are allowed** — re-inserting the same key keeps both
    /// copies. [`get`](Self::get) returns the lowest-index match, so the
    /// most recently inserted value is observable only after deleting the
    /// earlier copy. See crate-level rollback-vs-victim-cache decision in
    /// `CLAUDE.md`.
    ///
    /// # Returns
    ///
    /// - `Ok(())` on successful direct placement or kick relocation;
    ///   `num_items += 1`. If the mutation log is enabled, every committed
    ///   slot write is appended (kicks first, final placement last).
    /// - `Err(CuckooError::TableFull)` if the kick budget is exhausted;
    ///   store is unchanged and no mutations are logged.
    ///
    /// # Complexity
    ///
    /// Amortised `O(value_size_in_cells())` on direct placement; worst
    /// case `O(max_kicks · value_size_in_cells())` on a long kick chain.
    pub fn insert<K: AsRef<[u8]>>(&mut self, key: K, value: &[u8]) -> Result<(), CuckooError> {
        let vbytes = self.value_size_in_bytes();
        assert_eq!(
            value.len(),
            vbytes,
            "value buffer length must equal value_size_in_bytes ({vbytes})"
        );

        let vcells = self.value_size_in_cells();
        let vbits = self.table.value_bits();
        let pb = self.table.plaintext_bits();

        let (fingerprint, indices) = self
            .scheme
            .hash_item(key.as_ref(), self.table.fingerprint_bits());
        let arity = self.scheme.arity();

        // Pack bytes to cells once before any placement attempt.
        pack_value_bytes_to_cells(value, &mut self.cur_value, vbits, pb);

        // Direct placement (no kicking).
        for pos in 0..arity {
            if let Some(slot) = self
                .table
                .insert(indices[pos], fingerprint, &self.cur_value)
            {
                self.num_items += 1;
                if let Some(ref mut log) = self.mutation_log {
                    let new_val = self.cur_value.clone().into_boxed_slice();
                    log.push(SlotMutation {
                        bucket: indices[pos],
                        slot,
                        old_fingerprint: 0,
                        new_fingerprint: fingerprint,
                        old_value_cells: vec![0u32; vcells].into_boxed_slice(),
                        new_value_cells: new_val,
                    });
                }
                return Ok(());
            }
        }

        let mut rng = rand::rng();
        let start_pos = rng.random_range(0..arity as u32) as usize;
        let mut cur_index = indices[start_pos];
        let mut cur_fingerprint = fingerprint;
        // chain_meta[k] = (bucket, slot, original_fingerprint).
        // chain_values[k*vcells..(k+1)*vcells] = original (evicted) value at kick k.
        self.chain_meta.clear();
        // pending_mutations accumulates mutations for the kick chain; flushed on commit only.
        self.pending_mutations.clear();

        for _ in 0..self.max_kicks {
            let kick_idx = self.chain_meta.len();
            let slab_off = kick_idx * vcells;
            let slot = rng.random_range(0..self.table.bucket_size());
            let evicted_fp = self.table.read_fingerprint(cur_index, slot);

            // Stash the slot's original value in chain_values for rollback.
            self.table.read_value(
                cur_index,
                slot,
                &mut self.chain_values[slab_off..slab_off + vcells],
            );
            self.chain_meta.push((cur_index, slot, evicted_fp));

            // Write the in-flight value into the slot.
            self.table
                .write(cur_index, slot, cur_fingerprint, &self.cur_value);

            // Buffer the kick mutation (commit order: first kick first).
            if self.mutation_log.is_some() {
                let old_val = self.chain_values[slab_off..slab_off + vcells]
                    .to_vec()
                    .into_boxed_slice();
                let new_val = self.cur_value.clone().into_boxed_slice();
                self.pending_mutations.push(SlotMutation {
                    bucket: cur_index,
                    slot,
                    old_fingerprint: evicted_fp,
                    new_fingerprint: cur_fingerprint,
                    old_value_cells: old_val,
                    new_value_cells: new_val,
                });
            }

            cur_fingerprint = evicted_fp;
            // The just-evicted value (now in chain_values[k]) becomes the next cur_value.
            self.cur_value
                .copy_from_slice(&self.chain_values[slab_off..slab_off + vcells]);

            let all = self.scheme.all_indices(cur_index, cur_fingerprint);

            // Find which candidate position corresponds to the bucket we just kicked from.
            // `unwrap_or(0)` mirrors `CuckooFilter::insert` defensiveness; segmented schemes
            // can't actually collapse but the impl is generic over `IndexScheme`.
            let evicted_pos = (0..arity).find(|&p| all[p] == cur_index).unwrap_or(0);

            let mut placed = false;
            for p in 0..arity {
                if p == evicted_pos {
                    continue;
                }
                if let Some(final_slot) =
                    self.table.insert(all[p], cur_fingerprint, &self.cur_value)
                {
                    self.num_items += 1;
                    // Add the final placement mutation and flush all pending to the log.
                    if self.mutation_log.is_some() {
                        let new_val = self.cur_value.clone().into_boxed_slice();
                        self.pending_mutations.push(SlotMutation {
                            bucket: all[p],
                            slot: final_slot,
                            old_fingerprint: 0,
                            new_fingerprint: cur_fingerprint,
                            old_value_cells: vec![0u32; vcells].into_boxed_slice(),
                            new_value_cells: new_val,
                        });
                        let pending = std::mem::take(&mut self.pending_mutations);
                        self.mutation_log.as_mut().unwrap().extend(pending);
                    }
                    placed = true;
                    break;
                }
            }
            if placed {
                return Ok(());
            }

            let mut alts = [0u8; 3];
            let mut alt_count = 0;
            for p in 0..arity {
                if p != evicted_pos {
                    alts[alt_count] = p as u8;
                    alt_count += 1;
                }
            }
            let next_pos = alts[rng.random_range(0..alt_count as u32) as usize];
            cur_index = all[next_pos as usize];
        }

        // Kicks exhausted — restore each touched slot to its original (evicted) state.
        // Rolled-back inserts emit nothing to the log.
        for kick_idx in (0..self.chain_meta.len()).rev() {
            let (bucket, slot, original_fp) = self.chain_meta[kick_idx];
            let slab_off = kick_idx * vcells;
            self.table.write(
                bucket,
                slot,
                original_fp,
                &self.chain_values[slab_off..slab_off + vcells],
            );
        }
        self.pending_mutations.clear();
        Err(CuckooError::TableFull)
    }

    /// Test whether `key` is (probably) present.
    ///
    /// # Arguments
    ///
    /// - `key` — arbitrary bytes.
    ///
    /// # Returns
    ///
    /// `true` if some candidate bucket carries a matching fingerprint;
    /// `false` otherwise. Subject to the Cuckoo Filter false-positive rate
    /// — may return `true` for a key never inserted. Never returns `false`
    /// for a key currently in the store.
    ///
    /// # Complexity
    ///
    /// `O(arity · bucket_size · ⌈fingerprint_bits / plaintext_bits⌉)`
    /// cell reads.
    pub fn contain<K: AsRef<[u8]>>(&self, key: K) -> bool {
        let (fingerprint, indices) = self
            .scheme
            .hash_item(key.as_ref(), self.table.fingerprint_bits());
        let arity = self.scheme.arity();
        (0..arity).any(|i| self.table.contain(indices[i], fingerprint))
    }

    /// Retrieve the value stored for `key`, if present.
    ///
    /// # Arguments
    ///
    /// - `key` — arbitrary bytes.
    ///
    /// # Returns
    ///
    /// `Some(value_bytes)` (length `value_size_in_bytes()`) for the first
    /// matching `(bucket, slot)` in probe order
    /// (candidate-bucket index `0..arity`, then lowest-index slot within
    /// that bucket), `None` if no match is found. Subject to the same
    /// false-positive caveat as [`contain`](Self::contain).
    ///
    /// # Complexity
    ///
    /// `O(arity · bucket_size · cells_per_slot)` cell reads + one
    /// `Vec<u8>` allocation. For the alloc-free variant see
    /// [`get_into`](Self::get_into).
    pub fn get<K: AsRef<[u8]>>(&self, key: K) -> Option<Vec<u8>> {
        let mut out = vec![0u8; self.value_size_in_bytes()];
        if self.get_into(key, &mut out) {
            Some(out)
        } else {
            None
        }
    }

    /// Read the value for `key` into `out`, returning `true` on hit.
    ///
    /// # Purpose
    ///
    /// Zero-allocation read path. The caller owns the destination buffer.
    ///
    /// # Arguments
    ///
    /// - `key` — arbitrary bytes.
    /// - `out` — destination buffer; length must equal
    ///   [`value_size_in_bytes()`](Self::value_size_in_bytes).
    ///
    /// # Constraints
    ///
    /// Panics if `out.len() != value_size_in_bytes()`.
    ///
    /// # Rationale
    ///
    /// Streams cells through a stack accumulator instead of
    /// materialising an intermediate `Vec<u32>` like the older code
    /// path; on a miss `out` is left untouched.
    ///
    /// # Returns
    ///
    /// `true` if a slot matched (`out` filled), `false` otherwise.
    /// Subject to the same false-positive caveat as
    /// [`contain`](Self::contain).
    ///
    /// # Complexity
    ///
    /// `O(arity · bucket_size · cells_per_slot)` cell reads, no heap
    /// allocations.
    pub fn get_into<K: AsRef<[u8]>>(&self, key: K, out: &mut [u8]) -> bool {
        assert_eq!(
            out.len(),
            self.value_size_in_bytes(),
            "out buffer length must equal value_size_in_bytes"
        );
        let (fingerprint, indices) = self
            .scheme
            .hash_item(key.as_ref(), self.table.fingerprint_bits());
        let arity = self.scheme.arity();
        let vcells = self.value_size_in_cells();
        let vbits = self.table.value_bits();
        let pb = self.table.plaintext_bits();
        for i in 0..arity {
            if let Some(slot) = self.table.find(indices[i], fingerprint) {
                unpack_value_streaming(&self.table, indices[i], slot, out, vbits, pb, vcells);
                return true;
            }
        }
        false
    }

    /// Delete the entry for `key` from the store.
    ///
    /// # Purpose
    ///
    /// Removes the **first** matching slot in probe order — same probe
    /// order as [`get`](Self::get) — and zeros both the fingerprint and
    /// the value bits in that slot. Decrements `num_items`.
    ///
    /// # Arguments
    ///
    /// - `key` — arbitrary bytes.
    ///
    /// # Rationale
    ///
    /// First-match-wins matches the duplicate-key contract in the
    /// file-level docs. To purge all duplicates of the same key, call this
    /// method repeatedly until it returns
    /// [`CuckooError::NotFound`]. The value bits are zeroed so a freed
    /// slot returns all-zero on subsequent reads — required for PIR matvec
    /// correctness.
    ///
    /// # Returns
    ///
    /// - `Ok(())` on a successful deletion. If the mutation log is enabled,
    ///   one [`SlotMutation`] with `new_fingerprint == 0` is appended.
    /// - `Err(CuckooError::NotFound)` if no matching fingerprint is found;
    ///   store is unchanged and no mutation is logged.
    ///
    /// # Complexity
    ///
    /// `O(arity · bucket_size · ⌈fingerprint_bits / plaintext_bits⌉)`
    /// worst case on the fingerprint scan; `O(value_size_in_cells())` on
    /// the clear write.
    pub fn delete<K: AsRef<[u8]>>(&mut self, key: K) -> Result<(), CuckooError> {
        let (fingerprint, indices) = self
            .scheme
            .hash_item(key.as_ref(), self.table.fingerprint_bits());
        let arity = self.scheme.arity();
        for i in 0..arity {
            if let Some(slot) = self.table.find(indices[i], fingerprint) {
                if let Some(ref mut log) = self.mutation_log {
                    let old_val = self.table.read_value_to_box(indices[i], slot);
                    let vcells = self.table.value_size_in_cells() as usize;
                    log.push(SlotMutation {
                        bucket: indices[i],
                        slot,
                        old_fingerprint: fingerprint,
                        new_fingerprint: 0,
                        old_value_cells: old_val,
                        new_value_cells: vec![0u32; vcells].into_boxed_slice(),
                    });
                }
                // Zero the slot: fp=0 means empty; clear value cells.
                self.cur_value.fill(0);
                self.table.write(indices[i], slot, 0, &self.cur_value);
                self.num_items -= 1;
                return Ok(());
            }
        }
        Err(CuckooError::NotFound)
    }

    /// Replace the value for `key` with `new_value`.
    ///
    /// # Purpose
    ///
    /// In-place value mutation that preserves the fingerprint and the
    /// slot location — never triggers cuckoo kicking.
    ///
    /// # Arguments
    ///
    /// - `key`       — arbitrary bytes.
    /// - `new_value` — replacement bytes. Length must equal
    ///   [`value_size_in_bytes()`](Self::value_size_in_bytes).
    ///
    /// # Constraints
    ///
    /// Panics if `new_value.len() != value_size_in_bytes()`.
    ///
    /// # Rationale
    ///
    /// First-match-wins (matches [`get`](Self::get) /
    /// [`delete`](Self::delete) semantics). Does **not** modify
    /// [`num_items`](Self::num_items). To update all duplicates, call this
    /// method repeatedly.
    ///
    /// # Returns
    ///
    /// - `Ok(())` on a successful update. If the mutation log is enabled,
    ///   one [`SlotMutation`] with `old_fingerprint == new_fingerprint` is
    ///   appended.
    /// - `Err(CuckooError::NotFound)` if no matching fingerprint is found;
    ///   store is unchanged and no mutation is logged.
    ///
    /// # Complexity
    ///
    /// `O(arity · bucket_size · ⌈fingerprint_bits / plaintext_bits⌉)`
    /// worst case on the fingerprint scan; `O(value_size_in_cells())` on
    /// the value-only write.
    pub fn update<K: AsRef<[u8]>>(&mut self, key: K, new_value: &[u8]) -> Result<(), CuckooError> {
        let vbytes = self.value_size_in_bytes();
        assert_eq!(
            new_value.len(),
            vbytes,
            "new_value buffer length must equal value_size_in_bytes ({vbytes})"
        );
        let vbits = self.table.value_bits();
        let pb = self.table.plaintext_bits();
        pack_value_bytes_to_cells(new_value, &mut self.cur_value, vbits, pb);

        let (fingerprint, indices) = self
            .scheme
            .hash_item(key.as_ref(), self.table.fingerprint_bits());
        let arity = self.scheme.arity();
        for i in 0..arity {
            if let Some(slot) = self.table.find(indices[i], fingerprint) {
                if let Some(ref mut log) = self.mutation_log {
                    let old_val = self.table.read_value_to_box(indices[i], slot);
                    let new_val = self.cur_value.clone().into_boxed_slice();
                    log.push(SlotMutation {
                        bucket: indices[i],
                        slot,
                        old_fingerprint: fingerprint,
                        new_fingerprint: fingerprint,
                        old_value_cells: old_val,
                        new_value_cells: new_val,
                    });
                }
                self.table.write_value(indices[i], slot, &self.cur_value);
                return Ok(());
            }
        }
        Err(CuckooError::NotFound)
    }

    /// Number of items currently stored. Maintained exactly by `insert`
    /// and `delete`; unaffected by false positives.
    pub const fn num_items(&self) -> u64 {
        self.num_items
    }

    /// Byte size of the underlying `Vec<u32>` cell array:
    /// `4 · num_slots · cells_per_slot`.
    ///
    /// Larger than the bit-packed equivalent by a factor of
    /// `⌈32 / plaintext_bits⌉ · (plaintext_bits / 8)` — the ChalametPIR
    /// cell layout pays this overhead to enable matvec auto-vectorisation.
    pub fn size_in_bytes(&self) -> usize {
        self.table.size_in_bytes()
    }

    /// Current load factor: `num_items / (num_buckets · bucket_size)`.
    pub fn load_factor(&self) -> f64 {
        self.num_items as f64 / (self.table.num_buckets() as f64 * self.table.bucket_size() as f64)
    }

    /// Override the maximum number of cuckoo kicks before declaring the
    /// store full.
    ///
    /// # Arguments
    ///
    /// - `max_kicks` — new kick budget. `0` disables kicking entirely;
    ///   default is `500`.
    ///
    /// # Rationale
    ///
    /// Higher values increase insertion success at high load factor at the
    /// cost of slower worst-case inserts. Grows `chain_meta`'s capacity
    /// (never shrinks it) and resizes `chain_values` exactly to the new
    /// budget so subsequent `insert` calls remain alloc-free.
    pub fn set_max_kicks(&mut self, max_kicks: u32) {
        self.max_kicks = max_kicks;
        let vcells = self.value_size_in_cells();
        self.chain_meta.reserve(max_kicks as usize);
        self.chain_values.resize(
            (max_kicks as usize)
                .checked_mul(vcells)
                .expect("max_kicks * value_size_in_cells overflows usize"),
            0u32,
        );
    }

    // ─── Mutation log ─────────────────────────────────────────────────────────

    /// Enable the mutation log (idempotent).
    ///
    /// # Rationale
    ///
    /// Once enabled, every committed insert, delete, and update appends a
    /// [`SlotMutation`] to the log. Rolled-back inserts emit nothing. The
    /// IKPIR server enables the log at construction time and drains it
    /// after every mutation to drive incremental hint patching.
    pub fn enable_mutation_log(&mut self) {
        if self.mutation_log.is_none() {
            self.mutation_log = Some(Vec::new());
        }
    }

    /// Disable the mutation log and free its memory.
    ///
    /// # Rationale
    ///
    /// Re-enabling via [`enable_mutation_log`](Self::enable_mutation_log)
    /// allocates a fresh buffer.
    ///
    /// # IKPIR invariant
    ///
    /// The IKPIR server (`ikpir-server` crate) enables the mutation log
    /// at construction time and depends on it to emit `HintDeltaBundle`s
    /// for every `insert` / `update` / `delete`. Disabling the log on a
    /// store wrapped by `IkpirServer` **breaks incremental hint patching**
    /// — subsequent mutations will silently drop their deltas and the
    /// client's hint will diverge from the server's. Standalone
    /// `CuckooKVStore` users (without IKPIR) may disable freely.
    pub fn disable_mutation_log(&mut self) {
        self.mutation_log = None;
    }

    /// `true` iff the mutation log is currently enabled.
    pub const fn mutation_log_enabled(&self) -> bool {
        self.mutation_log.is_some()
    }

    /// Drain all buffered mutations.
    ///
    /// # Returns
    ///
    /// All mutations since the last drain in commit order. The log remains
    /// enabled and empty after this call; returns an empty `Vec` if the
    /// log is disabled.
    ///
    /// # Complexity
    ///
    /// `O(1)` (`std::mem::take` on the internal `Vec`).
    pub fn drain_mutations(&mut self) -> Vec<SlotMutation> {
        match self.mutation_log {
            Some(ref mut v) => std::mem::take(v),
            None => Vec::new(),
        }
    }

    // ─── Snapshot / restore ───────────────────────────────────────────────────

    /// Clone the entire cell array into a new `Vec<u32>`.
    ///
    /// # Rationale
    ///
    /// Pairs with
    /// [`from_cells`](CuckooKVStore::<Segmented2aryScheme>::from_cells) for
    /// snapshot-restore. The clone preserves the high-bits-zero invariant
    /// because the source already satisfies it.
    ///
    /// # Complexity
    ///
    /// `O(size_in_cells)` time and memory (one allocation).
    pub fn snapshot_cells(&self) -> Vec<u32> {
        self.table.as_cells().to_vec()
    }

    // ─── Iteration ───────────────────────────────────────────────────────────

    /// Iterate over every occupied slot in the store.
    ///
    /// # Returns
    ///
    /// One [`OccupiedSlot`] per non-empty `(bucket, slot)` pair, in
    /// bucket-major, slot-minor order. The borrowed `value_cells` slice is
    /// valid for the lifetime of the iterator.
    ///
    /// # Complexity
    ///
    /// `O(num_buckets · bucket_size)` fingerprint reads to enumerate;
    /// each yielded item is constant-time to materialise.
    pub fn iter_occupied_slots(&self) -> impl Iterator<Item = OccupiedSlot<'_>> {
        let nb = self.table.num_buckets();
        let bs = self.table.bucket_size();
        let cps = self.table.cells_per_slot() as usize;
        let cells = self.table.as_cells();
        let table = &self.table;

        (0..nb).flat_map(move |b| {
            (0..bs).filter_map(move |s| {
                let fp = table.read_fingerprint(b, s);
                if fp == 0 {
                    return None;
                }
                let off = (b as usize * bs as usize + s as usize) * cps;
                Some(OccupiedSlot {
                    bucket: b,
                    slot: s,
                    fingerprint: fp,
                    value_cells: &cells[off..off + cps],
                })
            })
        })
    }

    // ─── Mutation replay ──────────────────────────────────────────────────────

    /// Apply a [`SlotMutation`] as a raw cell write.
    ///
    /// # Purpose
    ///
    /// Replay-style write that bypasses the insert/delete/update logic:
    /// writes `new_fingerprint` and `new_value_cells` directly into the
    /// slot. Used by the IKPIR server to patch the PIR hint and by the
    /// mutation-log replay tests.
    ///
    /// # Arguments
    ///
    /// - `m` — mutation to replay.
    ///
    /// # Constraints
    ///
    /// `m.bucket < num_buckets`, `m.slot < bucket_size`, and
    /// `m.new_value_cells.len() == value_size_in_cells()` — same panic
    /// contract as the underlying storage writes.
    ///
    /// # Rationale
    ///
    /// `num_items` is **not** updated; this is a raw storage write
    /// intended for incremental PIR hint patching, not a semantic
    /// key-value mutation.
    pub fn apply_mutation(&mut self, m: &SlotMutation) {
        self.table
            .write_fingerprint(m.bucket, m.slot, m.new_fingerprint);
        self.table.write_value(m.bucket, m.slot, &m.new_value_cells);
    }

    // ─── PIR-parameter accessors ─────────────────────────────────────────────

    /// Borrow the flat cell array for PIR backend access.
    ///
    /// # Returns
    ///
    /// A shared slice of length `num_buckets · bucket_size · cells_per_slot`
    /// satisfying the ChalametPIR high-bits-zero invariant.
    pub fn as_cells(&self) -> &[u32] {
        self.table.as_cells()
    }

    /// Total number of buckets.
    #[inline]
    pub const fn num_buckets(&self) -> u32 {
        self.table.num_buckets()
    }

    /// Slots per bucket.
    #[inline]
    pub const fn bucket_size(&self) -> u32 {
        self.table.bucket_size()
    }

    /// Cells per slot: `⌈(fingerprint_bits + value_bits) / plaintext_bits⌉`.
    #[inline]
    pub const fn cells_per_slot(&self) -> u32 {
        self.table.cells_per_slot()
    }

    /// PIR plaintext cell width.
    #[inline]
    pub const fn plaintext_bits(&self) -> u32 {
        self.table.plaintext_bits()
    }

    /// Fingerprint bit width.
    #[inline]
    pub const fn fingerprint_bits(&self) -> u32 {
        self.table.fingerprint_bits()
    }

    /// Value bit width.
    #[inline]
    pub const fn value_bits(&self) -> u32 {
        self.table.value_bits()
    }
}

// ─── Params accessor (segmented schemes only) ────────────────────────────────

impl<S: IndexScheme + SchemeMeta> CuckooKVStore<S> {
    /// Public geometry parameters for this store.
    ///
    /// # Purpose
    ///
    /// The bundle shipped to the IKPIR client at setup time. `Copy` and
    /// heap-free, so it crosses process boundaries trivially.
    ///
    /// # Returns
    ///
    /// A [`CuckooParams`] mirroring `(scheme_kind, num_buckets,
    /// bucket_size, fingerprint_bits, value_bits, plaintext_bits)`.
    pub const fn params(&self) -> CuckooParams {
        CuckooParams {
            scheme_kind: S::KIND,
            num_buckets: self.table.num_buckets(),
            bucket_size: self.table.bucket_size(),
            fingerprint_bits: self.table.fingerprint_bits(),
            value_bits: self.table.value_bits(),
            plaintext_bits: self.table.plaintext_bits(),
        }
    }

    /// Flat-cell-array range covering slot `(bucket, slot)`.
    ///
    /// Equivalent to `self.params().slot_cell_range(bucket, slot)`.
    ///
    /// # Constraints
    ///
    /// Panics if `bucket >= num_buckets` or `slot >= bucket_size`.
    pub fn slot_cell_range(&self, bucket: u32, slot: u32) -> std::ops::Range<usize> {
        self.params().slot_cell_range(bucket, slot)
    }
}

// ─── `from_cells` constructors ───────────────────────────────────────────────

impl CuckooKVStore<Segmented2aryScheme> {
    /// Reconstruct a 2-ary store from a previously-snapshotted cell array.
    ///
    /// # Purpose
    ///
    /// Snapshot-restore primitive: pairs with
    /// [`snapshot_cells`](CuckooKVStore::snapshot_cells) to persist and
    /// reload a store without replaying every insert.
    ///
    /// # Arguments
    ///
    /// - `cells`     — previously snapshotted cell array; length must equal
    ///   `params.size_in_cells()`.
    /// - `params`    — geometry of the source store; `scheme_kind` must be
    ///   `Segmented2ary`.
    /// - `num_items` — item count carried over from the source store
    ///   (not validated against `cells`; trust-on-restore).
    ///
    /// # Constraints
    ///
    /// Validates `params.scheme_kind == Segmented2ary`, `params.num_buckets`
    /// is a power of 2 and ≥ 2, and the cell array passes the
    /// ChalametPIR high-bits-zero invariant. Mutation log starts disabled.
    ///
    /// # Returns
    ///
    /// `Ok(store)` on success, `Err(CuckooError::InvalidParams)` on
    /// scheme-kind mismatch, invalid `num_buckets`, or cell-array
    /// invariant violation.
    pub fn from_cells(
        cells: Vec<u32>,
        params: CuckooParams,
        num_items: u64,
    ) -> Result<Self, CuckooError> {
        if params.scheme_kind != SchemeKind::Segmented2ary {
            return Err(CuckooError::InvalidParams(
                "scheme_kind mismatch: expected Segmented2ary".into(),
            ));
        }
        if !params.num_buckets.is_power_of_two() || params.num_buckets < 2 {
            return Err(CuckooError::InvalidParams(
                "num_buckets must be a power of 2 and >= 2".into(),
            ));
        }
        let table = FingerprintValueTable::from_cells(
            cells,
            params.num_buckets,
            params.bucket_size,
            params.fingerprint_bits,
            params.value_bits,
            params.plaintext_bits,
        )
        .map_err(|e| CuckooError::InvalidParams(e.to_string()))?;
        let vcells = table.value_size_in_cells() as usize;
        Ok(Self {
            table,
            scheme: Segmented2aryScheme {
                segment_size: params.num_buckets / 2,
            },
            num_items,
            max_kicks: MAX_KICKS_DEFAULT,
            chain_meta: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
            chain_values: vec![
                0u32;
                (MAX_KICKS_DEFAULT as usize)
                    .checked_mul(vcells)
                    .expect("max_kicks * value_size_in_cells overflows usize")
            ],
            cur_value: vec![0u32; vcells],
            mutation_log: None,
            pending_mutations: Vec::new(),
        })
    }
}

impl CuckooKVStore<Segmented3aryScheme> {
    /// Reconstruct a 3-ary store from a previously-snapshotted cell array.
    ///
    /// # Constraints
    ///
    /// Validates `params.scheme_kind == Segmented3ary` and
    /// `params.num_buckets = 3 · 2^t`. Same cell-array invariant check as
    /// the 2-ary variant.
    ///
    /// # Returns
    ///
    /// `Ok(store)` on success, `Err(CuckooError::InvalidParams)` on any
    /// validation failure.
    pub fn from_cells(
        cells: Vec<u32>,
        params: CuckooParams,
        num_items: u64,
    ) -> Result<Self, CuckooError> {
        if params.scheme_kind != SchemeKind::Segmented3ary {
            return Err(CuckooError::InvalidParams(
                "scheme_kind mismatch: expected Segmented3ary".into(),
            ));
        }
        if params.num_buckets < 3
            || params.num_buckets % 3 != 0
            || !(params.num_buckets / 3).is_power_of_two()
        {
            return Err(CuckooError::InvalidParams(
                "num_buckets must be 3 * 2^t".into(),
            ));
        }
        let table = FingerprintValueTable::from_cells(
            cells,
            params.num_buckets,
            params.bucket_size,
            params.fingerprint_bits,
            params.value_bits,
            params.plaintext_bits,
        )
        .map_err(|e| CuckooError::InvalidParams(e.to_string()))?;
        let vcells = table.value_size_in_cells() as usize;
        Ok(Self {
            table,
            scheme: Segmented3aryScheme {
                segment_size: params.num_buckets / 3,
            },
            num_items,
            max_kicks: MAX_KICKS_DEFAULT,
            chain_meta: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
            chain_values: vec![
                0u32;
                (MAX_KICKS_DEFAULT as usize)
                    .checked_mul(vcells)
                    .expect("max_kicks * value_size_in_cells overflows usize")
            ],
            cur_value: vec![0u32; vcells],
            mutation_log: None,
            pending_mutations: Vec::new(),
        })
    }
}

impl CuckooKVStore<Segmented4aryScheme> {
    /// Reconstruct a 4-ary store from a previously-snapshotted cell array.
    ///
    /// # Constraints
    ///
    /// Validates `params.scheme_kind == Segmented4ary` and
    /// `params.num_buckets` is a power of 2 and ≥ 4. Same cell-array
    /// invariant check as the 2-ary variant.
    ///
    /// # Returns
    ///
    /// `Ok(store)` on success, `Err(CuckooError::InvalidParams)` on any
    /// validation failure.
    pub fn from_cells(
        cells: Vec<u32>,
        params: CuckooParams,
        num_items: u64,
    ) -> Result<Self, CuckooError> {
        if params.scheme_kind != SchemeKind::Segmented4ary {
            return Err(CuckooError::InvalidParams(
                "scheme_kind mismatch: expected Segmented4ary".into(),
            ));
        }
        if !params.num_buckets.is_power_of_two() || params.num_buckets < 4 {
            return Err(CuckooError::InvalidParams(
                "num_buckets must be a power of 2 and >= 4".into(),
            ));
        }
        let table = FingerprintValueTable::from_cells(
            cells,
            params.num_buckets,
            params.bucket_size,
            params.fingerprint_bits,
            params.value_bits,
            params.plaintext_bits,
        )
        .map_err(|e| CuckooError::InvalidParams(e.to_string()))?;
        let vcells = table.value_size_in_cells() as usize;
        Ok(Self {
            table,
            scheme: Segmented4aryScheme {
                segment_size: params.num_buckets / 4,
            },
            num_items,
            max_kicks: MAX_KICKS_DEFAULT,
            chain_meta: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
            chain_values: vec![
                0u32;
                (MAX_KICKS_DEFAULT as usize)
                    .checked_mul(vcells)
                    .expect("max_kicks * value_size_in_cells overflows usize")
            ],
            cur_value: vec![0u32; vcells],
            mutation_log: None,
            pending_mutations: Vec::new(),
        })
    }
}

// ─── Send + Sync static assertions ──────────────────────────────────────────

const _: fn() = || {
    const fn assert_send<T: Send>() {}
    const fn assert_sync<T: Sync>() {}
    assert_send::<CuckooKVStore<Segmented2aryScheme>>();
    assert_sync::<CuckooKVStore<Segmented2aryScheme>>();
    assert_send::<CuckooKVStore<Segmented3aryScheme>>();
    assert_sync::<CuckooKVStore<Segmented3aryScheme>>();
    assert_send::<CuckooKVStore<Segmented4aryScheme>>();
    assert_sync::<CuckooKVStore<Segmented4aryScheme>>();
};

#[cfg(test)]
mod tests {
    //! Unit tests for [`CuckooKVStore`].
    //!
    //! Grouped by concern (smoke / end-to-end / load-factor / params /
    //! mutation log / snapshot-restore / iteration). Together they pin the
    //! public contract that the IKPIR server relies on:
    //!
    //! 1. `insert` / `get` / `update` / `delete` round-trip without
    //!    distortion across all arities and a representative
    //!    `(plaintext_bits, value_bits)` matrix.
    //! 2. `TableFull` rolls back to the pre-call cell array.
    //! 3. `as_cells()` length and the ChalametPIR high-bits-zero invariant
    //!    hold after arbitrary mutation sequences.
    //! 4. The mutation log emits exactly the writes a replayer needs to
    //!    reconstruct the cell array from an empty store.
    //! 5. `snapshot_cells` / `from_cells` round-trip lossless.

    use super::*;

    fn make_value(n: usize, seed: u8) -> Vec<u8> {
        (0..n)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(seed))
            .collect()
    }

    // ─── Original smoke tests ────────────────────────────────────────────

    /// Default 2-ary construction succeeds; `size_in_bytes` reflects the
    /// cell-aligned layout (`64 · 4 · 10 · 4 = 10240` for `cells_per_slot = 10`).
    #[test]
    fn new_works() {
        let store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64, 8).unwrap();
        assert_eq!(store.num_items(), 0);
        // cells_per_slot = ceil((12+64)/8) = ceil(9.5) = 10
        // size_in_bytes = 64 * 4 * 10 * 4 = 10240
        assert_eq!(store.size_in_bytes(), 10240);
    }

    /// Same as `new_works` but for the 3-ary scheme.
    #[test]
    fn new_3ary_works() {
        let store = CuckooKVStore::<Segmented3aryScheme>::new(96, 4, 12, 64, 8).unwrap();
        assert_eq!(store.num_items(), 0);
        assert!(store.size_in_bytes() > 0);
    }

    /// Same as `new_works` but for the 4-ary scheme.
    #[test]
    fn new_4ary_works() {
        let store = CuckooKVStore::<Segmented4aryScheme>::new(64, 4, 12, 64, 8).unwrap();
        assert_eq!(store.num_items(), 0);
        assert!(store.size_in_bytes() > 0);
    }

    /// `new` rejects a `num_buckets` that violates the 2-ary
    /// power-of-two-and-≥-2 constraint.
    #[test]
    fn new_returns_err_on_invalid_num_buckets() {
        assert!(CuckooKVStore::<Segmented2aryScheme>::new(3, 4, 12, 64, 8).is_err());
    }

    /// `new` rejects `value_bits == 0` for every arity.
    #[test]
    fn new_returns_err_on_zero_value_bits() {
        assert!(CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 0, 8).is_err());
    }

    // ─── End-to-end behaviour ────────────────────────────────────────────

    /// Basic insert+get round-trip: stored value is observable verbatim.
    #[test]
    fn insert_get_roundtrip() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64, 8).unwrap();
        let value = make_value(8, 1);
        store.insert("hello", &value).unwrap();
        assert_eq!(store.get("hello"), Some(value));
        assert_eq!(store.num_items(), 1);
    }

    /// `update` makes the new value observable through `get`.
    #[test]
    fn insert_update_get_returns_new_value() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64, 8).unwrap();
        let v1 = make_value(8, 1);
        let v2 = make_value(8, 2);
        store.insert("k", &v1).unwrap();
        store.update("k", &v2).unwrap();
        assert_eq!(store.get("k"), Some(v2));
    }

    /// After `delete`, `contain` returns false and a second `delete`
    /// returns `NotFound`; `num_items` returns to 0.
    #[test]
    fn insert_delete_contain_returns_false() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64, 8).unwrap();
        let v = make_value(8, 7);
        store.insert("k", &v).unwrap();
        store.delete("k").unwrap();
        assert!(!store.contain("k"));
        assert!(matches!(store.delete("k"), Err(CuckooError::NotFound)));
        assert_eq!(store.num_items(), 0);
    }

    /// `delete` on a never-inserted key returns `NotFound`.
    #[test]
    fn delete_of_never_inserted_returns_not_found() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64, 8).unwrap();
        assert!(matches!(
            store.delete("missing"),
            Err(CuckooError::NotFound)
        ));
    }

    /// `update` on a never-inserted key returns `NotFound`.
    #[test]
    fn update_of_never_inserted_returns_not_found() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64, 8).unwrap();
        let v = make_value(8, 1);
        assert!(matches!(
            store.update("missing", &v),
            Err(CuckooError::NotFound)
        ));
    }

    /// `update` does **not** change `num_items` (in-place value mutation,
    /// matches the documented contract).
    #[test]
    fn update_does_not_change_num_items() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64, 8).unwrap();
        let v1 = make_value(8, 1);
        let v2 = make_value(8, 2);
        store.insert("k", &v1).unwrap();
        let before = store.num_items();
        store.update("k", &v2).unwrap();
        assert_eq!(store.num_items(), before);
    }

    /// On `TableFull`, every successful insert before the failure is
    /// still retrievable — i.e. the kick chain rolled back cleanly.
    #[test]
    fn table_full_triggers_rollback() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(2, 1, 24, 8, 8).unwrap();
        let mut inserted: Vec<(u32, [u8; 1])> = Vec::new();
        let mut full_seen = false;
        for i in 0u32..200 {
            let v = [(i & 0xFF) as u8];
            match store.insert(i.to_le_bytes(), &v) {
                Ok(()) => inserted.push((i, v)),
                Err(CuckooError::TableFull) => {
                    full_seen = true;
                    break;
                }
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert!(full_seen, "should have hit TableFull on tiny store");
        for (i, v) in &inserted {
            assert_eq!(
                store.get(i.to_le_bytes()),
                Some(v.to_vec()),
                "rollback failed for key {i}"
            );
        }
        assert_eq!(store.num_items(), inserted.len() as u64);
    }

    /// 2-ary kick chain under load — keys keep round-tripping after kicks.
    #[test]
    fn kicking_under_load_2ary() {
        let mut store =
            CuckooKVStore::<Segmented2aryScheme>::from_num_items(1000, 4, 32, 64, 8).unwrap();
        let mut keys: Vec<u64> = Vec::new();
        for i in 0u64..1000 {
            let v = make_value(8, i as u8);
            if store.insert(i.to_le_bytes(), &v).is_ok() {
                keys.push(i);
            } else {
                break;
            }
        }
        assert!(!keys.is_empty());
        for &i in &keys {
            assert_eq!(store.get(i.to_le_bytes()), Some(make_value(8, i as u8)));
        }
    }

    /// 3-ary kick chain under load.
    #[test]
    fn kicking_under_load_3ary() {
        let mut store =
            CuckooKVStore::<Segmented3aryScheme>::from_num_items(1000, 4, 32, 64, 8).unwrap();
        let mut keys: Vec<u64> = Vec::new();
        for i in 0u64..1000 {
            let v = make_value(8, i as u8);
            if store.insert(i.to_le_bytes(), &v).is_ok() {
                keys.push(i);
            } else {
                break;
            }
        }
        assert!(!keys.is_empty());
        for &i in &keys {
            assert_eq!(store.get(i.to_le_bytes()), Some(make_value(8, i as u8)));
        }
    }

    /// 4-ary kick chain under load.
    #[test]
    fn kicking_under_load_4ary() {
        let mut store =
            CuckooKVStore::<Segmented4aryScheme>::from_num_items(1000, 4, 32, 64, 8).unwrap();
        let mut keys: Vec<u64> = Vec::new();
        for i in 0u64..1000 {
            let v = make_value(8, i as u8);
            if store.insert(i.to_le_bytes(), &v).is_ok() {
                keys.push(i);
            } else {
                break;
            }
        }
        assert!(!keys.is_empty());
        for &i in &keys {
            assert_eq!(store.get(i.to_le_bytes()), Some(make_value(8, i as u8)));
        }
    }

    /// Wide values (`value_bits = 1024`) survive kicking — the chain
    /// correctly shuffles all `cells_per_slot` cells per slot, not just
    /// the fingerprint.
    #[test]
    fn wide_value_roundtrip_through_kicking() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(8, 2, 32, 1024, 8).unwrap();
        let vsize = 128;
        let mut inserted: Vec<(u32, Vec<u8>)> = Vec::new();
        for i in 0u32..100 {
            let v = make_value(vsize, i as u8);
            match store.insert(i.to_le_bytes(), &v) {
                Ok(()) => inserted.push((i, v)),
                Err(CuckooError::TableFull) => break,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert!(!inserted.is_empty());
        for (i, v) in &inserted {
            assert_eq!(store.get(i.to_le_bytes()).as_ref(), Some(v));
        }
    }

    /// Duplicate insertion: `get` returns the first inserted value;
    /// deleting it exposes the second; deleting again leaves the slot
    /// empty.
    #[test]
    fn duplicate_insert_first_match_wins() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        let v1 = [0xAAu8];
        let v2 = [0xBBu8];
        store.insert("k", &v1).unwrap();
        store.insert("k", &v2).unwrap();
        assert_eq!(store.get("k"), Some(v1.to_vec()));
        store.delete("k").unwrap();
        assert_eq!(store.get("k"), Some(v2.to_vec()));
        store.delete("k").unwrap();
        assert_eq!(store.get("k"), None);
        assert!(matches!(store.delete("k"), Err(CuckooError::NotFound)));
    }

    /// `set_max_kicks(0)` disables kicking so only direct placements
    /// succeed.
    #[test]
    fn set_max_kicks_zero_disables_kicking() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(2, 1, 2, 8, 8).unwrap();
        store.set_max_kicks(0);
        let mut count = 0u32;
        for i in 0u32..100 {
            if store.insert(i.to_le_bytes(), &[i as u8]).is_ok() {
                count += 1;
            } else {
                break;
            }
        }
        assert!(count <= 2, "with no kicking, at most 2 direct placements");
    }

    /// `load_factor` tracks `num_items` exactly: each insert adds one
    /// slot's worth, each delete subtracts one.
    #[test]
    fn load_factor_invariant() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        let unit = 1.0 / (64.0 * 4.0);
        assert_eq!(store.load_factor(), 0.0);
        store.insert("a", &[1]).unwrap();
        assert!((store.load_factor() - unit).abs() < f64::EPSILON);
        store.insert("b", &[2]).unwrap();
        assert!(2.0f64.mul_add(-unit, store.load_factor()).abs() < f64::EPSILON);
        store.delete("a").unwrap();
        assert!((store.load_factor() - unit).abs() < f64::EPSILON);
    }

    /// `insert` panics if the value buffer is the wrong length.
    #[test]
    #[should_panic(expected = "value_size_in_bytes")]
    fn insert_panics_on_wrong_value_length() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 16, 8).unwrap();
        let _ = store.insert("k", &[0u8]); // value_size = 2 bytes; 1 byte is wrong
    }

    /// `update` panics if the value buffer is the wrong length.
    #[test]
    #[should_panic(expected = "value_size_in_bytes")]
    fn update_panics_on_wrong_value_length() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 16, 8).unwrap();
        store.insert("k", &[0u8, 0]).unwrap();
        let _ = store.update("k", &[0u8]); // wrong length
    }

    /// `from_num_items` succeeds for the 2-ary scheme at a realistic size.
    #[test]
    fn from_num_items_2ary_works() {
        let store =
            CuckooKVStore::<Segmented2aryScheme>::from_num_items(10_000, 4, 12, 64, 8).unwrap();
        assert!(store.size_in_bytes() > 0);
    }

    /// `from_num_items` succeeds for the 3-ary scheme.
    #[test]
    fn from_num_items_3ary_works() {
        let store =
            CuckooKVStore::<Segmented3aryScheme>::from_num_items(10_000, 4, 12, 64, 8).unwrap();
        assert!(store.size_in_bytes() > 0);
    }

    /// `from_num_items` succeeds for the 4-ary scheme.
    #[test]
    fn from_num_items_4ary_works() {
        let store =
            CuckooKVStore::<Segmented4aryScheme>::from_num_items(10_000, 4, 12, 64, 8).unwrap();
        assert!(store.size_in_bytes() > 0);
    }

    /// `from_num_items` rejects `value_bits == 0` for every arity.
    #[test]
    fn from_num_items_returns_err_on_zero_value_bits() {
        assert!(CuckooKVStore::<Segmented2aryScheme>::from_num_items(100, 4, 12, 0, 8).is_err());
        assert!(CuckooKVStore::<Segmented3aryScheme>::from_num_items(100, 4, 12, 0, 8).is_err());
        assert!(CuckooKVStore::<Segmented4aryScheme>::from_num_items(100, 4, 12, 0, 8).is_err());
    }

    // ─── get_into / value_size_in_bytes ──────────────────────────────────

    /// `get_into` returns `true` and fills `out` on hit.
    #[test]
    fn get_into_hit_writes_value_returns_true() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64, 8).unwrap();
        let v = make_value(8, 9);
        store.insert("k", &v).unwrap();
        let mut out = vec![0u8; store.value_size_in_bytes()];
        assert!(store.get_into("k", &mut out));
        assert_eq!(out, v);
    }

    /// `get_into` returns `false` and leaves `out` untouched on miss.
    #[test]
    fn get_into_miss_returns_false() {
        let store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64, 8).unwrap();
        let mut out = vec![0u8; store.value_size_in_bytes()];
        assert!(!store.get_into("missing", &mut out));
        assert!(
            out.iter().all(|&b| b == 0),
            "out should remain untouched on miss"
        );
    }

    /// `get_into` panics on a wrong-length destination buffer.
    #[test]
    #[should_panic(expected = "value_size_in_bytes")]
    fn get_into_panics_on_wrong_length() {
        let store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 16, 8).unwrap();
        let mut out = [0u8; 1]; // value_size = 2 bytes
        store.get_into("k", &mut out);
    }

    // ─── Pre-allocated rollback buffers ──────────────────────────────────

    /// Repeated inserts do not grow `chain_meta` or `chain_values` —
    /// the kick-chain scratch is reused across calls.
    #[test]
    fn repeated_inserts_reuse_buffers() {
        let mut store =
            CuckooKVStore::<Segmented2aryScheme>::from_num_items(2000, 4, 12, 64, 8).unwrap();
        let initial_meta_cap = store.chain_meta.capacity();
        let initial_values_len = store.chain_values.len();
        for i in 0u32..1000 {
            let v = make_value(8, i as u8);
            let _ = store.insert(i.to_le_bytes(), &v);
        }
        assert_eq!(
            store.chain_meta.capacity(),
            initial_meta_cap,
            "chain_meta grew past max_kicks across repeated inserts"
        );
        assert_eq!(
            store.chain_values.len(),
            initial_values_len,
            "chain_values length changed across repeated inserts"
        );
    }

    /// `set_max_kicks` grows / shrinks `chain_values` exactly to the new
    /// budget and reserves capacity on `chain_meta`.
    #[test]
    fn set_max_kicks_grows_buffers() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64, 8).unwrap();
        let vcells = store.value_size_in_cells();
        store.set_max_kicks(2000);
        assert!(store.chain_meta.capacity() >= 2000);
        assert_eq!(store.chain_values.len(), 2000 * vcells);
        // Shrink: chain_values resizes down. chain_meta keeps capacity.
        store.set_max_kicks(100);
        assert_eq!(store.chain_values.len(), 100 * vcells);
    }

    /// After `TableFull`, the store keeps accepting unrelated inserts —
    /// the failed kick chain doesn't poison the state.
    #[test]
    fn failed_insert_then_successful_insert() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(2, 1, 24, 8, 8).unwrap();
        let mut placed: Vec<(u32, [u8; 1])> = Vec::new();
        for i in 0u32..200 {
            let v = [(i & 0xFF) as u8];
            match store.insert(i.to_le_bytes(), &v) {
                Ok(()) => placed.push((i, v)),
                Err(CuckooError::TableFull) => break,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert!(
            !placed.is_empty(),
            "should have placed something before failing"
        );
        for (i, v) in &placed {
            assert_eq!(store.get(i.to_le_bytes()), Some(v.to_vec()));
        }
        let (k, _) = placed[0];
        store.delete(k.to_le_bytes()).unwrap();
        let new_v = [0xEEu8];
        store.insert(b"fresh".as_ref(), &new_v).unwrap();
        assert_eq!(store.get(b"fresh".as_ref()), Some(new_v.to_vec()));
    }

    /// `from_num_items` rejects `max_items` exceeding `u32::MAX · bucket_size`.
    #[test]
    fn from_num_items_rejects_overflow_max_items() {
        let huge: u64 = u32::MAX as u64 * 5;
        assert!(CuckooKVStore::<Segmented2aryScheme>::from_num_items(huge, 4, 12, 8, 8).is_err());
        assert!(CuckooKVStore::<Segmented3aryScheme>::from_num_items(huge, 4, 12, 8, 8).is_err());
        assert!(CuckooKVStore::<Segmented4aryScheme>::from_num_items(huge, 4, 12, 8, 8).is_err());
    }

    /// On 32-bit targets, `set_max_kicks(u32::MAX)` panics on `usize`
    /// overflow (verifying the overflow guard documented on `set_max_kicks`).
    #[test]
    #[cfg(target_pointer_width = "32")]
    #[should_panic(expected = "overflows usize")]
    fn set_max_kicks_panics_on_overflow_value_bits() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        store.set_max_kicks(u32::MAX);
    }

    // ─── New Phase 2 tests ───────────────────────────────────────────────

    /// `plaintext_bits = 9`, `value_bits = 8` — fingerprint straddles a cell
    /// boundary; verify roundtrip.
    #[test]
    fn roundtrip_plaintext_9_value_8() {
        // pb=9, vb=8, fp=12 — fingerprint straddles a cell boundary
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 9).unwrap();
        let v = [0xABu8];
        store.insert("key", &v).unwrap();
        assert_eq!(store.get("key"), Some(v.to_vec()));
    }

    /// `plaintext_bits = 10`, `value_bits = 64` — both straddle cell
    /// boundaries; verify roundtrip.
    #[test]
    fn roundtrip_plaintext_10_value_64() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64, 10).unwrap();
        let v = make_value(8, 42);
        store.insert("k64", &v).unwrap();
        assert_eq!(store.get("k64"), Some(v));
    }

    /// Wide value (`value_bits = 256` = 32 bytes) at `plaintext_bits = 12`.
    #[test]
    fn roundtrip_plaintext_12_wide_value() {
        // value_bits=256 → 32 bytes
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 256, 12).unwrap();
        let v = make_value(32, 7);
        store.insert("wide", &v).unwrap();
        assert_eq!(store.get("wide"), Some(v));
    }

    /// At `plaintext_bits = 9`, every cell must keep its high 23 bits
    /// zero after 50 inserts (ChalametPIR invariant).
    #[test]
    fn as_cells_high_bits_invariant() {
        // At pb=9, the high 23 bits of every cell must remain zero after many inserts.
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 9).unwrap();
        let hi_mask = !((1u32 << 9) - 1);
        for i in 0u32..50 {
            let _ = store.insert(i.to_le_bytes(), &[(i & 0xFF) as u8]);
        }
        for &c in store.as_cells() {
            assert_eq!(c & hi_mask, 0, "high bits non-zero in cell");
        }
    }

    /// `as_cells().len() == num_buckets · bucket_size · cells_per_slot`.
    #[test]
    fn as_cells_dimensions() {
        let store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64, 8).unwrap();
        let expected = store.num_buckets() as usize
            * store.bucket_size() as usize
            * store.cells_per_slot() as usize;
        assert_eq!(store.as_cells().len(), expected);
    }

    // ─── CuckooParams / params() accessor ───────────────────────────────────

    /// `CuckooParams::candidate_buckets` matches the store's internal
    /// hash — pins the IKPIR client's ability to derive lookup positions
    /// from `params()` alone.
    #[test]
    fn params_round_trip_candidate_buckets() {
        let store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        let p = store.params();
        assert_eq!(p.scheme_kind, SchemeKind::Segmented2ary);
        assert_eq!(p.arity(), 2);
        // candidate_buckets must return the same fp+indices as the store's internal hash.
        for i in 0u32..100 {
            let key = i.to_le_bytes();
            let (fp_params, idx_params) = p.candidate_buckets(&key);
            let (fp_store, idx_store) = store.scheme.hash_item(&key, p.fingerprint_bits);
            assert_eq!(fp_params, fp_store, "fp mismatch for key {i}");
            assert_eq!(
                &idx_params[..2],
                &idx_store[..2],
                "indices mismatch for key {i}"
            );
        }
    }

    /// `slot_cell_range` covers exactly `cells_per_slot` cells inside
    /// `as_cells()`, and the cells at the returned range carry the
    /// expected fingerprint after an insert.
    #[test]
    fn params_slot_cell_range_matches_cells() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        let v = [0xABu8];
        store.insert("key", &v).unwrap();
        let p = store.params();
        let (fp, indices) = p.candidate_buckets(b"key");
        // At least one slot must carry the fingerprint we expect.
        let mut found = false;
        'outer: for &b in &indices[..2] {
            for s in 0..p.bucket_size {
                let r = p.slot_cell_range(b, s);
                let raw_fp = store.table.read_fingerprint(b, s);
                if raw_fp == fp {
                    // Verify the range covers cells_per_slot cells.
                    assert_eq!(r.len(), p.cells_per_slot() as usize);
                    // Verify the range is within as_cells().
                    assert!(r.end <= store.as_cells().len());
                    found = true;
                    break 'outer;
                }
            }
        }
        assert!(found, "inserted key not found via params");
    }

    // ─── Mutation log ────────────────────────────────────────────────────────

    /// Mutation log is **off** by default — the IKPIR server has to
    /// explicitly opt in, and standalone `CuckooKVStore` users pay nothing.
    #[test]
    fn mutation_log_disabled_by_default() {
        let store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        assert!(!store.mutation_log_enabled());
    }

    /// `enable_mutation_log` is idempotent; `disable_mutation_log` clears
    /// the buffered log.
    #[test]
    fn enable_disable_idempotent() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        store.enable_mutation_log();
        store.enable_mutation_log(); // idempotent
        assert!(store.mutation_log_enabled());
        assert_eq!(store.drain_mutations().len(), 0);
        store.disable_mutation_log();
        assert!(!store.mutation_log_enabled());
    }

    /// A successful insert emits exactly one mutation with
    /// `old_fingerprint == 0` and zeroed `old_value_cells`.
    #[test]
    fn insert_emits_mutation() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        store.enable_mutation_log();
        store.insert("k", &[0xABu8]).unwrap();
        let mutations = store.drain_mutations();
        assert_eq!(mutations.len(), 1);
        assert_eq!(mutations[0].old_fingerprint, 0);
        assert_ne!(mutations[0].new_fingerprint, 0);
        assert!(mutations[0].old_value_cells.iter().all(|&c| c == 0));
    }

    /// A successful delete emits exactly one mutation with
    /// `new_fingerprint == 0` and zeroed `new_value_cells`.
    #[test]
    fn delete_emits_mutation() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        store.insert("k", &[0xABu8]).unwrap();
        store.enable_mutation_log();
        store.delete("k").unwrap();
        let mutations = store.drain_mutations();
        assert_eq!(mutations.len(), 1);
        assert_ne!(mutations[0].old_fingerprint, 0);
        assert_eq!(mutations[0].new_fingerprint, 0);
        assert!(mutations[0].new_value_cells.iter().all(|&c| c == 0));
    }

    /// A successful update emits exactly one mutation with
    /// `old_fingerprint == new_fingerprint` and differing value cells.
    #[test]
    fn update_emits_mutation_same_fp() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        store.insert("k", &[0x11u8]).unwrap();
        store.enable_mutation_log();
        store.update("k", &[0x22u8]).unwrap();
        let mutations = store.drain_mutations();
        assert_eq!(mutations.len(), 1);
        assert_eq!(mutations[0].old_fingerprint, mutations[0].new_fingerprint);
        assert_ne!(mutations[0].old_value_cells, mutations[0].new_value_cells);
    }

    /// `TableFull` rollback emits **nothing** — only successful inserts
    /// produce mutations.
    #[test]
    fn table_full_emits_nothing() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(2, 1, 24, 8, 8).unwrap();
        store.enable_mutation_log();
        let mut full_seen = false;
        for i in 0u32..200 {
            match store.insert(i.to_le_bytes(), &[(i & 0xFF) as u8]) {
                Ok(()) => {}
                Err(CuckooError::TableFull) => {
                    full_seen = true;
                    break;
                }
                Err(e) => panic!("unexpected: {e:?}"),
            }
        }
        assert!(full_seen);
        // TableFull rollback must not emit mutations.
        let mutations = store.drain_mutations();
        // Only the successful inserts should have emitted mutations.
        assert_eq!(mutations.len() as u64, store.num_items());
    }

    /// Replaying a drained mutation log against an empty store reproduces
    /// the original cell array byte-for-byte — the core invariant the
    /// IKPIR server's incremental hint patcher relies on.
    #[test]
    fn mutation_log_replay_property() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        store.enable_mutation_log();
        for i in 0u32..30 {
            let _ = store.insert(i.to_le_bytes(), &[(i & 0xFF) as u8]);
        }
        let _ = store.delete(5u32.to_le_bytes());
        let _ = store.update(3u32.to_le_bytes(), &[0xFFu8]);

        let mutations = store.drain_mutations();
        let original_cells = store.snapshot_cells();
        let p = store.params();

        // Replay on a fresh store.
        let mut fresh = CuckooKVStore::<Segmented2aryScheme>::new(
            p.num_buckets,
            p.bucket_size,
            p.fingerprint_bits,
            p.value_bits,
            p.plaintext_bits,
        )
        .unwrap();
        for m in &mutations {
            fresh.apply_mutation(m);
        }
        assert_eq!(
            fresh.as_cells(),
            original_cells.as_slice(),
            "replayed mutations must reproduce the original cell array"
        );
    }

    /// `delete` on an absent key emits no mutation.
    #[test]
    fn delete_not_found_emits_nothing() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        store.enable_mutation_log();
        let _ = store.delete("missing");
        assert!(store.drain_mutations().is_empty());
    }

    /// `update` on an absent key emits no mutation.
    #[test]
    fn update_not_found_emits_nothing() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        store.enable_mutation_log();
        let _ = store.update("missing", &[0u8]);
        assert!(store.drain_mutations().is_empty());
    }

    // ─── Snapshot / restore ──────────────────────────────────────────────────

    /// `snapshot_cells` → `from_cells` round-trips every inserted key.
    #[test]
    fn snapshot_restore_roundtrip() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        let mut inserted = Vec::new();
        for i in 0u32..30 {
            let v = [(i & 0xFF) as u8];
            if store.insert(i.to_le_bytes(), &v).is_ok() {
                inserted.push((i, v));
            }
        }
        let cells = store.snapshot_cells();
        let p = store.params();
        let n = store.num_items();

        let restored = CuckooKVStore::<Segmented2aryScheme>::from_cells(cells, p, n).unwrap();
        assert_eq!(restored.num_items(), n);
        for (i, v) in &inserted {
            assert_eq!(
                restored.get(i.to_le_bytes()),
                Some(v.to_vec()),
                "restored store missing key {i}"
            );
        }
    }

    /// `from_cells` rejects a cell array of the wrong length.
    #[test]
    fn from_cells_rejects_wrong_length() {
        let p = CuckooParams {
            scheme_kind: SchemeKind::Segmented2ary,
            num_buckets: 64,
            bucket_size: 4,
            fingerprint_bits: 12,
            value_bits: 8,
            plaintext_bits: 8,
        };
        let wrong = vec![0u32; 1]; // way too short
        assert!(CuckooKVStore::<Segmented2aryScheme>::from_cells(wrong, p, 0).is_err());
    }

    /// `from_cells` rejects a cell array whose high bits are non-zero
    /// (ChalametPIR invariant violation).
    #[test]
    fn from_cells_rejects_high_bits_set() {
        let p = CuckooParams {
            scheme_kind: SchemeKind::Segmented2ary,
            num_buckets: 64,
            bucket_size: 4,
            fingerprint_bits: 12,
            value_bits: 8,
            plaintext_bits: 8,
        };
        let mut cells = vec![0u32; p.size_in_cells()];
        cells[0] = u32::MAX; // high bits set → violation
        assert!(CuckooKVStore::<Segmented2aryScheme>::from_cells(cells, p, 0).is_err());
    }

    /// `from_cells` rejects a `params.scheme_kind` that doesn't match the
    /// generic parameter (a fresh `Segmented2aryScheme` store rejects a
    /// `Segmented3ary` snapshot).
    #[test]
    fn from_cells_rejects_scheme_kind_mismatch() {
        let p = CuckooParams {
            scheme_kind: SchemeKind::Segmented3ary, // wrong for 2ary store
            num_buckets: 96,
            bucket_size: 4,
            fingerprint_bits: 12,
            value_bits: 8,
            plaintext_bits: 8,
        };
        let cells = vec![0u32; p.size_in_cells()];
        assert!(CuckooKVStore::<Segmented2aryScheme>::from_cells(cells, p, 0).is_err());
    }

    // ─── iter_occupied_slots ─────────────────────────────────────────────────

    /// `iter_occupied_slots().count() == num_items` exactly.
    #[test]
    fn iter_occupied_slots_count_matches_num_items() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        for i in 0u32..20 {
            let _ = store.insert(i.to_le_bytes(), &[(i & 0xFF) as u8]);
        }
        let count = store.iter_occupied_slots().count();
        assert_eq!(count as u64, store.num_items());
    }

    /// Every yielded `OccupiedSlot` has non-zero fingerprint and a
    /// `value_cells` slice of the right length.
    #[test]
    fn iter_occupied_slots_fingerprints_non_zero() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8, 8).unwrap();
        for i in 0u32..15 {
            let _ = store.insert(i.to_le_bytes(), &[(i & 0xFF) as u8]);
        }
        for occ in store.iter_occupied_slots() {
            assert_ne!(occ.fingerprint, 0);
            assert_eq!(occ.value_cells.len(), store.cells_per_slot() as usize);
        }
    }
}
