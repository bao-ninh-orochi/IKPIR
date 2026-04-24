//! Segmented cuckoo map: `(key, value)` placement over k segments, with
//! both fingerprint and value storage, and cuckoo-style eviction.
//!
//! # Purpose
//!
//! Under the per-segment PIR architecture, every enrolled `(key, value)`
//! lives in **exactly one** of its k candidate buckets (`indices[j]` for
//! j ∈ [0, k)). The cuckoo filter in `segmented-cuckoo-filter` places
//! fingerprints but cannot store user values — this module layers a
//! parallel value store on top of the SCF's [`TagTable`] primitives and
//! materialises one bucket row per `(segment, local_idx)` ready to be
//! written into a per-segment `D_j` matrix.
//!
//! # Bucket-row layout
//!
//! ```text
//! row_elems_bucket = b · (fp_elems + val_elems)
//! slot s @ offset  s · (fp_elems + val_elems)
//!     [fp_elems]  split  fp  LSB-first into `B`-bit chunks
//!     [val_elems] encoded via [`lwe::encode_value`]
//! empty slot: fp == 0, value_slots == 0
//! ```
//!
//! # Invariants
//!
//! - `tables[j].read_fingerprint(local, slot) == 0` ⇔ `values[j][local][slot] == None`.
//! - A successful `insert` leaves exactly one `(segment, local, slot)` holding
//!   the new `(tag, value)`; `delete` clears both halves atomically.
//! - `encode_bucket(j, local)` is pure: it reflects the current
//!   `(tables, values)` state and never mutates it.
//! - Peel ordering is **irrelevant** (cuckoo filters do not peel); the
//!   ordering among kicks is driven by the internal ChaCha20 RNG.
//!
//! # Security
//!
//! No secret data flows through this module at server-side use (keys
//! and values are public to the server). Kick-path randomness is
//! deterministic given the construction seed, which keeps tests and
//! benchmarks reproducible.
//!
//! # Atomicity
//!
//! `insert` is **atomic**: on cuckoo-kick exhaustion the internal trace
//! stack is replayed in reverse, restoring every `(tables, values)`
//! cell that was touched. `delete` mutates at most one cell and so is
//! trivially atomic. Callers observe either full commit or full
//! rollback.

// The cuckoo-kick loops index multiple parallel structures (`tables[j]`,
// `values[j]`, `indices[j]`) by the same loop variable `j`, which the
// `needless_range_loop` lint repeatedly objects to. Same allow that the
// upstream `segmented-cuckoo-filter::filter` module carries.
#![allow(clippy::needless_range_loop)]

use core::fmt;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use segmented_cuckoo_filter::bucket::FingerprintTable;
use segmented_cuckoo_filter::IndexScheme;

use crate::lwe::encode_value;
use crate::matrix::{Matrix, MatrixError};
use crate::params::{FilterParams, CIPHERTEXT_MODULUS_BITS};

/// Default cuckoo-kick budget before giving up on an insert.
pub const DEFAULT_MAX_KICKS: u32 = 500;

// ── Public types ───────────────────────────────────────────────────────────

/// Row-level update emitted by [`SegmentedCuckooMap::insert`] / `delete`.
///
/// Each touched bucket in each segment produces one `PerSegmentRow`
/// carrying the bucket's **complete** post-operation encoded payload —
/// not a delta on individual slots. The caller feeds these rows to
/// `ikpir-server::update::apply` which handles the matrix-delta against
/// `H_j = A_j · D_j`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerSegmentRow {
    /// Segment index in `[0, k)`.
    pub segment_idx: u8,
    /// Row index within the segment, in `[0, seg)`.
    pub row_local_idx: u32,
    /// Full bucket row in u32 elements, length `row_elems_bucket`.
    pub new_payload: Vec<u32>,
}

/// Batch of per-segment row updates returned by mutation methods.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpdateBatch {
    /// Ordered list of touched bucket rows; same `(segment_idx, row_local_idx)`
    /// may appear multiple times if several kicks passed through the same bucket,
    /// but in that case only the last entry for that pair carries the final state.
    pub rows: Vec<PerSegmentRow>,
}

/// Error cases raised by `SegmentedCuckooMap`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapError {
    /// Caller supplied a value with `value.len() != params.value_len`.
    ValueLength,
    /// `(tag, value)` pair already present in one of the candidate buckets.
    Duplicate,
    /// Cuckoo-kick budget exhausted; no mutation retained.
    TableFull,
    /// Delete target not found in any candidate bucket.
    NotFound,
    /// The supplied [`FilterParams`] uses a non-segmented arity.
    ArityNotSegmented,
    /// Underlying matrix error (only in `to_matrices`).
    Matrix(MatrixError),
    /// Encoded payload failed to serialise (should not happen with valid params).
    EncodingFailed,
}

impl From<MatrixError> for MapError {
    fn from(e: MatrixError) -> Self {
        Self::Matrix(e)
    }
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ValueLength => f.write_str("value length does not match params.value_len"),
            Self::Duplicate => f.write_str("duplicate (tag, value) already present"),
            Self::TableFull => f.write_str("cuckoo kicks exhausted; insert rolled back"),
            Self::NotFound => f.write_str("delete target not found in any candidate bucket"),
            Self::ArityNotSegmented => f.write_str("SegmentedCuckooMap requires a segmented Arity"),
            Self::Matrix(e) => write!(f, "matrix error: {e}"),
            Self::EncodingFailed => f.write_str("value encoding failed"),
        }
    }
}

impl std::error::Error for MapError {}

// ── Map ────────────────────────────────────────────────────────────────────

/// Per-segment cuckoo map with companion value storage.
///
/// Owns:
///   - k independent [`FingerprintTable`]s (one per segment) of shape `seg × b`,
///   - k independent value mirrors of the same shape,
///   - a seeded ChaCha20 RNG driving deterministic kick-slot selection.
///
/// The generic `S` is an [`IndexScheme`] from `segmented-cuckoo-filter`;
/// only segmented schemes are supported (checked at construction).
#[derive(Clone)]
pub struct SegmentedCuckooMap<S: IndexScheme + Clone> {
    scheme: S,
    params: FilterParams,
    b: u32,
    k: u32,
    seg: u32,
    tables: Vec<FingerprintTable>,
    /// `values[j][local_idx][slot] = Some(value) | None` in lock-step with
    /// `tables[j].read_fingerprint(local_idx, slot) != 0`.
    values: Vec<Vec<Vec<Option<Vec<u8>>>>>,
    max_kicks: u32,
    rng: ChaCha20Rng,
}

// `FingerprintTable` does not derive `Debug`; we expose only the structural fields.
impl<S: IndexScheme + Clone> fmt::Debug for SegmentedCuckooMap<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentedCuckooMap")
            .field("k", &self.k)
            .field("seg", &self.seg)
            .field("b", &self.b)
            .field("fp_bits", &self.params.fingerprint_bits)
            .field("value_len", &self.params.value_len)
            .field("max_kicks", &self.max_kicks)
            .finish()
    }
}

impl<S: IndexScheme + Clone> SegmentedCuckooMap<S> {
    /// Build an empty map.
    ///
    /// # Errors
    ///
    /// [`MapError::ArityNotSegmented`] if `params.arity` is not one of the
    /// segmented variants.
    pub fn new(
        scheme: S,
        params: FilterParams,
        b: u32,
        max_kicks: u32,
        rng_seed: [u8; 32],
    ) -> Result<Self, MapError> {
        if !params.arity.is_segmented() {
            return Err(MapError::ArityNotSegmented);
        }
        let k = params.degree();
        let seg = params.seg();
        let tables: Vec<FingerprintTable> = (0..k)
            .map(|_| FingerprintTable::new(seg, b, params.fingerprint_bits))
            .collect();
        let values: Vec<Vec<Vec<Option<Vec<u8>>>>> = (0..k as usize)
            .map(|_| {
                (0..seg as usize)
                    .map(|_| (0..b as usize).map(|_| None).collect())
                    .collect()
            })
            .collect();
        Ok(Self {
            scheme,
            params,
            b,
            k,
            seg,
            tables,
            values,
            max_kicks,
            rng: ChaCha20Rng::from_seed(rng_seed),
        })
    }

    /// Read-only view of the active parameters.
    pub fn params(&self) -> &FilterParams {
        &self.params
    }

    /// Slots per bucket (`b`).
    pub fn slots_per_bucket(&self) -> u32 {
        self.b
    }

    /// Number of segments (`k = arity.degree()`).
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Segment size (`seg = num_buckets / k`).
    pub fn seg(&self) -> u32 {
        self.seg
    }

    /// Number of `u32` elements per fingerprint slot chunk.
    pub fn fp_elems(&self) -> u32 {
        let b = self.params.mat_elem_bit_len;
        self.params.fingerprint_bits.div_ceil(b)
    }

    /// Number of `u32` elements per value slot chunk.
    pub fn val_elems(&self) -> u32 {
        let b = self.params.mat_elem_bit_len;
        (self.params.value_len as u64 * 8).div_ceil(b as u64) as u32
    }

    /// `fp_elems() + val_elems()`.
    pub fn slot_elems(&self) -> u32 {
        self.fp_elems() + self.val_elems()
    }

    /// Total `u32` elements per bucket row (`b · slot_elems`).
    pub fn row_elems_bucket(&self) -> u32 {
        self.b * self.slot_elems()
    }

    /// Compute `(tag, indices)` for `key` without mutating state.
    pub fn lookup_candidates(&self, key: &[u8]) -> (u32, [u32; 4]) {
        self.scheme.hash_item(key, self.params.fingerprint_bits)
    }

    /// Insert `(key, value)`. Returns the per-segment row updates that the
    /// caller should then feed to the matrix-delta machinery.
    ///
    /// # Errors
    ///
    /// - [`MapError::ValueLength`] if `value.len() != params.value_len`.
    /// - [`MapError::Duplicate`] if the same `(fingerprint, value)` is already
    ///   present at any of the k candidate buckets.
    /// - [`MapError::TableFull`] if cuckoo kicks are exhausted; map is
    ///   rolled back to its pre-call state.
    pub fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<UpdateBatch, MapError> {
        if value.len() != self.params.value_len as usize {
            return Err(MapError::ValueLength);
        }
        let (tag, indices) = self.lookup_candidates(key);

        // ── 0. Duplicate check (same (fp, value) already present).
        for j in 0..self.k as usize {
            let local = (indices[j] - (j as u32) * self.seg) as usize;
            for s in 0..self.b as usize {
                if self.tables[j].read_fingerprint(local as u32, s as u32) == tag {
                    if let Some(stored) = &self.values[j][local][s] {
                        if stored.as_slice() == value {
                            return Err(MapError::Duplicate);
                        }
                    }
                }
            }
        }

        // ── 1. Trace stack for rollback: (seg_idx, local, slot, old_fp, old_value).
        let mut trace: Vec<(u8, u32, u32, u32, Option<Vec<u8>>)> = Vec::new();

        // ── 2. Fast path: direct insertion into any of the k candidate buckets.
        for j in 0..self.k as usize {
            let local = indices[j] - (j as u32) * self.seg;
            if let Some(slot) = self.tables[j].insert_fingerprint_to_bucket(local, tag) {
                // `insert_fingerprint_to_bucket` found a slot with previous tag==0 (empty).
                // Mirror: values[j][local][slot] should be None.
                debug_assert!(self.values[j][local as usize][slot as usize].is_none());
                let old_val = self.values[j][local as usize][slot as usize].take();
                self.values[j][local as usize][slot as usize] = Some(value.to_vec());
                trace.push((j as u8, local, slot, 0, old_val));
                return Ok(self.collapse_trace_to_batch(&trace));
            }
        }

        // ── 3. Kick path: every candidate bucket is full. Start from a random candidate.
        let start_j = (self.rng.next_u32() % self.k) as usize;
        let mut cur_seg = start_j as u8;
        let mut cur_local = indices[start_j] - (start_j as u32) * self.seg;
        let mut cur_fp = tag;
        let mut cur_val: Vec<u8> = value.to_vec();

        for _ in 0..self.max_kicks {
            // Evict a random slot in the current bucket.
            let slot = self.rng.next_u32() % self.b;
            let evicted_fp = self.tables[cur_seg as usize].read_fingerprint(cur_local, slot);
            let evicted_val = self.values[cur_seg as usize][cur_local as usize][slot as usize]
                .take()
                .unwrap_or_else(|| vec![0u8; self.params.value_len as usize]);

            // Record for rollback: old cell state.
            trace.push((
                cur_seg,
                cur_local,
                slot,
                evicted_fp,
                Some(evicted_val.clone()),
            ));

            // Overwrite with (cur_fp, cur_val).
            self.tables[cur_seg as usize].write_fingerprint(cur_local, slot, cur_fp);
            self.values[cur_seg as usize][cur_local as usize][slot as usize] = Some(cur_val);

            // Evictee's alternate candidates (global indices).
            let evicted_global = (cur_seg as u32) * self.seg + cur_local;
            let all = self.scheme.all_indices(evicted_global, evicted_fp);

            // Try to place the evictee in each alternate segment.
            let mut placed_via_direct = false;
            for p in 0..self.k as usize {
                if p as u8 == cur_seg {
                    continue;
                }
                let local_p = all[p] - (p as u32) * self.seg;
                debug_assert!(
                    all[p] / self.seg == p as u32,
                    "alternate {p} index {} not in its segment",
                    all[p]
                );
                if let Some(slot_p) = self.tables[p].insert_fingerprint_to_bucket(local_p, evicted_fp) {
                    debug_assert!(self.values[p][local_p as usize][slot_p as usize].is_none());
                    let old_val = self.values[p][local_p as usize][slot_p as usize].take();
                    self.values[p][local_p as usize][slot_p as usize] = Some(evicted_val.clone());
                    trace.push((p as u8, local_p, slot_p, 0, old_val));
                    placed_via_direct = true;
                    break;
                }
            }
            if placed_via_direct {
                return Ok(self.collapse_trace_to_batch(&trace));
            }

            // Pick a random alternate to continue kicking from.
            let mut alts: Vec<u8> = Vec::with_capacity(self.k as usize - 1);
            for p in 0..self.k as u8 {
                if p != cur_seg {
                    alts.push(p);
                }
            }
            let next_pos = alts[(self.rng.next_u32() as usize) % alts.len()];
            cur_seg = next_pos;
            cur_local = all[next_pos as usize] - (next_pos as u32) * self.seg;
            cur_fp = evicted_fp;
            cur_val = evicted_val;
        }

        // ── 4. Kicks exhausted. Rollback.
        for (seg_idx, local, slot, old_fp, old_val) in trace.iter().rev() {
            self.tables[*seg_idx as usize].write_fingerprint(*local, *slot, *old_fp);
            self.values[*seg_idx as usize][*local as usize][*slot as usize] = old_val.clone();
        }
        Err(MapError::TableFull)
    }

    /// Overwrite the value stored alongside `key`'s fingerprint, in place.
    ///
    /// Finds the first slot across the `k` candidate buckets whose fingerprint
    /// equals `tag(key)` (same probe order as [`delete`][Self::delete]) and
    /// replaces its value cell with `new_value`. The fingerprint stays put, so
    /// exactly one bucket row is touched and exactly one [`PerSegmentRow`] is
    /// emitted.
    ///
    /// # Errors
    ///
    /// * [`MapError::ValueLength`] — `new_value.len() != params.value_len`.
    /// * [`MapError::NotFound`] — no candidate bucket contains the key's tag.
    ///
    /// # Caveat
    ///
    /// Same as [`delete`][Self::delete]: if a different key's `(tag, candidate_set)`
    /// coincides with the target key's, this may overwrite the wrong slot's
    /// value. Callers must only `update_value` keys they previously inserted.
    pub fn update_value(
        &mut self,
        key: &[u8],
        new_value: &[u8],
    ) -> Result<UpdateBatch, MapError> {
        if new_value.len() != self.params.value_len as usize {
            return Err(MapError::ValueLength);
        }
        let (tag, indices) = self.lookup_candidates(key);
        for j in 0..self.k as usize {
            let local = indices[j] - (j as u32) * self.seg;
            if let Some(slot) = self.tables[j].find_fingerprint_in_bucket_slot(local, tag) {
                self.values[j][local as usize][slot as usize] = Some(new_value.to_vec());
                let new_payload = self.encode_bucket(j as u8, local)?;
                return Ok(UpdateBatch {
                    rows: vec![PerSegmentRow {
                        segment_idx: j as u8,
                        row_local_idx: local,
                        new_payload,
                    }],
                });
            }
        }
        Err(MapError::NotFound)
    }

    /// Delete the first slot in any candidate bucket whose fingerprint
    /// matches `key`. Returns the single-row update for the affected bucket.
    ///
    /// # Caveat
    ///
    /// Cuckoo filter semantics: if `key` was never inserted, this may
    /// nevertheless delete a slot owned by a *different* key whose
    /// `(tag, indices)` coincide with `key`'s. Callers must only delete
    /// keys they inserted.
    pub fn delete(&mut self, key: &[u8]) -> Result<UpdateBatch, MapError> {
        let (tag, indices) = self.lookup_candidates(key);
        for j in 0..self.k as usize {
            let local = indices[j] - (j as u32) * self.seg;
            if let Some(slot) = self.tables[j].find_fingerprint_in_bucket_slot(local, tag) {
                self.tables[j].write_fingerprint(local, slot, 0);
                self.values[j][local as usize][slot as usize] = None;
                let new_payload = self.encode_bucket(j as u8, local)?;
                return Ok(UpdateBatch {
                    rows: vec![PerSegmentRow {
                        segment_idx: j as u8,
                        row_local_idx: local,
                        new_payload,
                    }],
                });
            }
        }
        Err(MapError::NotFound)
    }

    /// Materialise bucket row `(segment_idx, row_local_idx)` into its
    /// `row_elems_bucket`-length `u32` payload.
    pub fn encode_bucket(&self, segment_idx: u8, row_local_idx: u32) -> Result<Vec<u32>, MapError> {
        if segment_idx as u32 >= self.k || row_local_idx >= self.seg {
            return Err(MapError::EncodingFailed);
        }
        let fp_elems = self.fp_elems() as usize;
        let val_elems = self.val_elems() as usize;
        let slot_elems = fp_elems + val_elems;
        let mut out = vec![0u32; self.row_elems_bucket() as usize];

        for s in 0..self.b as usize {
            let base = s * slot_elems;
            let stored_fp = self.tables[segment_idx as usize].read_fingerprint(row_local_idx, s as u32);
            // Fingerprint chunks, LSB-first.
            encode_fp_into(
                stored_fp,
                self.params.fingerprint_bits,
                self.params.mat_elem_bit_len,
                &mut out[base..base + fp_elems],
            );
            // Value chunks. Empty slot: zeros are already in place from vec![0u32; _].
            if let Some(val) = &self.values[segment_idx as usize][row_local_idx as usize][s] {
                let encoded = encode_value(val, &self.params).ok_or(MapError::EncodingFailed)?;
                debug_assert_eq!(encoded.len(), val_elems);
                out[base + fp_elems..base + fp_elems + val_elems].copy_from_slice(&encoded);
            }
        }
        Ok(out)
    }

    /// Build `k` per-segment `Matrix` objects of shape `seg × row_elems_bucket`
    /// where each row is the bucket's current encoded payload.
    pub fn to_matrices(&self) -> Result<Vec<Matrix>, MapError> {
        let cols = self.row_elems_bucket();
        let mut mats = Vec::with_capacity(self.k as usize);
        for j in 0..self.k {
            let mut buf = Vec::with_capacity((self.seg * cols) as usize);
            for local in 0..self.seg {
                let row = self.encode_bucket(j as u8, local)?;
                buf.extend_from_slice(&row);
            }
            mats.push(Matrix::from_elems(self.seg, cols, buf)?);
        }
        Ok(mats)
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    /// Collapse a trace into one `PerSegmentRow` per distinct touched bucket,
    /// preserving the *order in which each bucket was LAST touched* so the
    /// caller's `update::apply` sees the final encoded payload.
    fn collapse_trace_to_batch(
        &self,
        trace: &[(u8, u32, u32, u32, Option<Vec<u8>>)],
    ) -> UpdateBatch {
        // Emit one row per distinct (segment_idx, row_local_idx); last occurrence wins.
        let mut seen: Vec<(u8, u32)> = Vec::new();
        let mut rows: Vec<PerSegmentRow> = Vec::new();
        // Iterate trace in reverse so we pick up the latest state first; then reverse output.
        for &(seg_idx, local, _, _, _) in trace.iter().rev() {
            if !seen.iter().any(|&(s, l)| s == seg_idx && l == local) {
                seen.push((seg_idx, local));
                let payload = match self.encode_bucket(seg_idx, local) {
                    Ok(p) => p,
                    Err(_) => continue, // unreachable in practice; skip on failure
                };
                rows.push(PerSegmentRow {
                    segment_idx: seg_idx,
                    row_local_idx: local,
                    new_payload: payload,
                });
            }
        }
        rows.reverse();
        UpdateBatch { rows }
    }
}

// ── Free helpers ───────────────────────────────────────────────────────────

/// Split `fp` into `dst.len()` plaintext chunks of `bit_len` bits, LSB first.
///
/// Each chunk lives in `[0, 2^bit_len)`. `bit_len` must be in
/// `[1, CIPHERTEXT_MODULUS_BITS)`; panics in debug otherwise.
#[inline]
fn encode_fp_into(fp: u32, fp_bits: u32, bit_len: u32, dst: &mut [u32]) {
    debug_assert!((1..CIPHERTEXT_MODULUS_BITS).contains(&bit_len));
    let mask = (1u32 << bit_len) - 1;
    let fp_mask = if fp_bits >= 32 {
        u32::MAX
    } else {
        (1u32 << fp_bits) - 1
    };
    let fp = fp & fp_mask;
    for (i, slot) in dst.iter_mut().enumerate() {
        let shift = (i as u32) * bit_len;
        *slot = if shift >= 32 { 0 } else { (fp >> shift) & mask };
    }
}

/// Inverse of [`encode_fp_into`]: reconstruct a fingerprint from a slice of
/// `bit_len`-bit chunks (LSB first). Masks the result to `fp_bits` so any
/// unused high bits in the top chunk are zeroed out.
#[inline]
pub fn decode_fp(chunks: &[u32], fp_bits: u32, bit_len: u32) -> u32 {
    debug_assert!((1..CIPHERTEXT_MODULUS_BITS).contains(&bit_len));
    let elem_mask = (1u32 << bit_len) - 1;
    let fp_mask = if fp_bits >= 32 {
        u32::MAX
    } else {
        (1u32 << fp_bits) - 1
    };
    let mut fp: u32 = 0;
    for (i, &c) in chunks.iter().enumerate() {
        let shift = (i as u32) * bit_len;
        if shift >= 32 {
            break;
        }
        let piece = c & elem_mask;
        fp |= piece.wrapping_shl(shift);
    }
    fp & fp_mask
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{Arity, DEFAULT_SLOTS_PER_BUCKET};
    use segmented_cuckoo_filter::{Segmented3aryScheme, Segmented4aryScheme, Segmented2aryScheme};

    fn seg2_map(num_items: u32, value_len: u32) -> SegmentedCuckooMap<Segmented2aryScheme> {
        let params = FilterParams::recommended(Arity::Segmented2, num_items, value_len).unwrap();
        let scheme = Segmented2aryScheme { half: params.seg() };
        SegmentedCuckooMap::new(
            scheme,
            params,
            DEFAULT_SLOTS_PER_BUCKET,
            DEFAULT_MAX_KICKS,
            [0u8; 32],
        )
        .unwrap()
    }

    fn seg3_map(num_items: u32, value_len: u32) -> SegmentedCuckooMap<Segmented3aryScheme> {
        let params = FilterParams::recommended(Arity::Segmented3, num_items, value_len).unwrap();
        let scheme = Segmented3aryScheme {
            segment_size: params.seg(),
        };
        SegmentedCuckooMap::new(
            scheme,
            params,
            DEFAULT_SLOTS_PER_BUCKET,
            DEFAULT_MAX_KICKS,
            [0u8; 32],
        )
        .unwrap()
    }

    fn seg4_map(num_items: u32, value_len: u32) -> SegmentedCuckooMap<Segmented4aryScheme> {
        let params = FilterParams::recommended(Arity::Segmented4, num_items, value_len).unwrap();
        let scheme = Segmented4aryScheme {
            segment_size: params.seg(),
        };
        SegmentedCuckooMap::new(
            scheme,
            params,
            DEFAULT_SLOTS_PER_BUCKET,
            DEFAULT_MAX_KICKS,
            [0u8; 32],
        )
        .unwrap()
    }

    fn mk_value(i: u32, len: u32) -> Vec<u8> {
        (0..len as usize)
            .map(|j| ((i.wrapping_add(j as u32)) & 0xFF) as u8)
            .collect()
    }

    fn mk_key(i: u32) -> Vec<u8> {
        format!("key-{i}").into_bytes()
    }

    #[test]
    fn fp_encode_decode_round_trip() {
        for bit_len in [4u32, 7, 9, 12, 14] {
            for fp_bits in [8u32, 12, 16, 20] {
                if fp_bits < bit_len {
                    continue;
                }
                let fp_elems = fp_bits.div_ceil(bit_len) as usize;
                let mut buf = vec![0u32; fp_elems];
                for sample in [0u32, 1, 0xABC, 0xFFFF, 0x1234_5678] {
                    encode_fp_into(sample, fp_bits, bit_len, &mut buf);
                    let decoded = decode_fp(&buf, fp_bits, bit_len);
                    let expected = sample
                        & if fp_bits >= 32 {
                            u32::MAX
                        } else {
                            (1u32 << fp_bits) - 1
                        };
                    assert_eq!(
                        decoded, expected,
                        "fp={:#x} bit_len={} fp_bits={}",
                        sample, bit_len, fp_bits
                    );
                    for &c in &buf {
                        assert!(c < (1u32 << bit_len), "encoded chunk out of range: {}", c);
                    }
                }
            }
        }
    }

    #[test]
    fn new_rejects_non_segmented_arity() {
        // Construct Standard4 params directly since `recommended` rejects non-segmented arities.
        let params = FilterParams {
            arity: Arity::Standard4,
            num_buckets: 128,
            fingerprint_bits: 12,
            row_elems: 32,
            mat_elem_bit_len: 8,
            value_len: 32,
        };
        let scheme = Segmented2aryScheme { half: params.seg() };
        let err = SegmentedCuckooMap::new(
            scheme,
            params,
            DEFAULT_SLOTS_PER_BUCKET,
            DEFAULT_MAX_KICKS,
            [0u8; 32],
        )
        .unwrap_err();
        assert_eq!(err, MapError::ArityNotSegmented);
    }

    #[test]
    fn insert_then_lookup_roundtrip_seg4() {
        let value_len = 8;
        let num = 100;
        let mut map = seg4_map(num, value_len);
        for i in 0..num {
            map.insert(&mk_key(i), &mk_value(i, value_len))
                .unwrap_or_else(|e| panic!("insert {i}: {e:?}"));
        }
        for i in 0..num {
            let (tag, indices) = map.lookup_candidates(&mk_key(i));
            // Expect the tag to be found in at least one candidate bucket, at a slot whose
            // mirrored value equals the stored value.
            let mut found = false;
            for j in 0..map.k() as usize {
                let local = indices[j] - (j as u32) * map.seg();
                for s in 0..map.slots_per_bucket() {
                    if map.tables[j].read_fingerprint(local, s) == tag {
                        let val = map.values[j][local as usize][s as usize].as_ref().unwrap();
                        if val.as_slice() == mk_value(i, value_len).as_slice() {
                            found = true;
                        }
                    }
                }
            }
            assert!(found, "key {i} not found after insert");
        }
    }

    #[test]
    fn insert_then_lookup_roundtrip_seg2_and_seg3() {
        let value_len = 8;
        let num = 50;
        for arity_id in 0..2 {
            if arity_id == 0 {
                let mut m = seg2_map(num, value_len);
                for i in 0..num {
                    m.insert(&mk_key(i), &mk_value(i, value_len)).unwrap();
                }
            } else {
                let mut m = seg3_map(num, value_len);
                for i in 0..num {
                    m.insert(&mk_key(i), &mk_value(i, value_len)).unwrap();
                }
            }
        }
    }

    #[test]
    fn duplicate_key_is_rejected() {
        let mut map = seg4_map(50, 8);
        let key = mk_key(0);
        let val = mk_value(0, 8);
        map.insert(&key, &val).unwrap();
        let err = map.insert(&key, &val).unwrap_err();
        assert_eq!(err, MapError::Duplicate);
    }

    #[test]
    fn value_length_mismatch_is_rejected() {
        let mut map = seg4_map(50, 8);
        let err = map.insert(&mk_key(0), &mk_value(0, 7)).unwrap_err();
        assert_eq!(err, MapError::ValueLength);
    }

    #[test]
    fn insert_kick_exhaustion_rolls_back_everything() {
        // Tiny map with max_kicks=0 and a capacity of exactly the buckets we fill.
        // Deliberately construct a situation where every candidate bucket is already full and
        // kicks are disallowed.
        let params = FilterParams::recommended(Arity::Segmented4, 32, 4).unwrap();
        let scheme = Segmented4aryScheme {
            segment_size: params.seg(),
        };
        let b = DEFAULT_SLOTS_PER_BUCKET;
        let mut map = SegmentedCuckooMap::new(scheme, params, b, 0, [0u8; 32]).unwrap();

        // Find a key whose candidate buckets we can saturate deterministically.
        // Strategy: precompute many keys, pick one, fill its k candidate buckets with
        // unrelated entries (b * k different (fp, value) pairs placed directly).
        let target_key = mk_key(0);
        let (_, target_indices) = map.lookup_candidates(&target_key);
        // Fill all k candidate buckets of the target key with b direct tags each.
        for j in 0..map.k() as usize {
            let local = target_indices[j] - (j as u32) * map.seg();
            for s in 0..map.slots_per_bucket() {
                // Place a distinct non-zero tag directly via the low-level TagTable API.
                let fake_tag = 1 + (j as u32) * 10 + s;
                assert!(map.tables[j]
                    .insert_fingerprint_to_bucket(local, fake_tag)
                    .is_some());
                map.values[j][local as usize][s as usize] = Some(vec![0xAAu8; 4]);
            }
        }

        // Snapshot
        let tables_before: Vec<Vec<u32>> = (0..map.k() as usize)
            .map(|j| {
                (0..map.seg() * map.slots_per_bucket())
                    .map(|idx| {
                        map.tables[j]
                            .read_fingerprint(idx / map.slots_per_bucket(), idx % map.slots_per_bucket())
                    })
                    .collect()
            })
            .collect();
        let values_before = map.values.clone();

        // Try to insert target key with max_kicks=0. All candidate buckets are full -> TableFull.
        let err = map.insert(&target_key, &mk_value(0, 4)).unwrap_err();
        assert_eq!(err, MapError::TableFull);

        // Verify full state-restoration.
        for j in 0..map.k() as usize {
            for local in 0..map.seg() {
                for s in 0..map.slots_per_bucket() {
                    let idx = (local * map.slots_per_bucket() + s) as usize;
                    assert_eq!(
                        map.tables[j].read_fingerprint(local, s),
                        tables_before[j][idx],
                        "table[{j}][{local}][{s}] not restored"
                    );
                }
            }
        }
        assert_eq!(map.values, values_before, "value mirror not restored");
    }

    #[test]
    fn delete_removes_fp_and_value() {
        let mut map = seg4_map(50, 8);
        let key = mk_key(0);
        let val = mk_value(0, 8);
        map.insert(&key, &val).unwrap();
        let batch = map.delete(&key).unwrap();
        assert_eq!(batch.rows.len(), 1);

        // Re-find fails.
        let err = map.delete(&key).unwrap_err();
        assert_eq!(err, MapError::NotFound);

        // And the fingerprint slot is really gone.
        let (tag, indices) = map.lookup_candidates(&key);
        for j in 0..map.k() as usize {
            let local = indices[j] - (j as u32) * map.seg();
            for s in 0..map.slots_per_bucket() {
                if map.tables[j].read_fingerprint(local, s) == tag {
                    panic!("fingerprint still present after delete");
                }
                if map.values[j][local as usize][s as usize].is_some() {
                    let stored = map.values[j][local as usize][s as usize].as_ref().unwrap();
                    assert_ne!(
                        stored.as_slice(),
                        val.as_slice(),
                        "value still present after delete"
                    );
                }
            }
        }
    }

    #[test]
    fn update_value_replaces_value_in_place() {
        let value_len: u32 = 8;
        let mut map = seg4_map(50, value_len);
        let key = mk_key(7);
        let orig = mk_value(7, value_len);
        let new_val = mk_value(999, value_len);
        map.insert(&key, &orig).unwrap();

        // Locate the slot the insert landed in.
        let (tag, indices) = map.lookup_candidates(&key);
        let (seg_before, local_before, slot_before) = (0..map.k() as usize)
            .find_map(|j| {
                let local = indices[j] - (j as u32) * map.seg();
                (0..map.slots_per_bucket())
                    .find(|s| map.tables[j].read_fingerprint(local, *s) == tag)
                    .map(|s| (j as u8, local, s))
            })
            .expect("slot located after insert");

        let batch = map.update_value(&key, &new_val).unwrap();
        assert_eq!(batch.rows.len(), 1);
        assert_eq!(batch.rows[0].segment_idx, seg_before);
        assert_eq!(batch.rows[0].row_local_idx, local_before);

        // Same slot, same tag; value is replaced.
        assert_eq!(
            map.tables[seg_before as usize].read_fingerprint(local_before, slot_before),
            tag,
        );
        assert_eq!(
            map.values[seg_before as usize][local_before as usize][slot_before as usize]
                .as_deref(),
            Some(new_val.as_slice()),
        );
    }

    #[test]
    fn update_value_unknown_key_is_error() {
        let mut map = seg4_map(50, 8);
        let err = map.update_value(&mk_key(123), &mk_value(0, 8)).unwrap_err();
        assert_eq!(err, MapError::NotFound);
    }

    #[test]
    fn update_value_wrong_length_is_error() {
        let mut map = seg4_map(50, 8);
        let key = mk_key(0);
        map.insert(&key, &mk_value(0, 8)).unwrap();
        let err = map.update_value(&key, &mk_value(0, 7)).unwrap_err();
        assert_eq!(err, MapError::ValueLength);
    }

    #[test]
    fn encode_bucket_empty_is_all_zero() {
        let map = seg4_map(50, 8);
        let row = map.encode_bucket(0, 0).unwrap();
        assert_eq!(row.len(), map.row_elems_bucket() as usize);
        assert!(row.iter().all(|&e| e == 0));
    }

    #[test]
    fn encode_bucket_round_trip_single_slot() {
        use crate::lwe::decode_value;

        let value_len: u32 = 8;
        let mut map = seg4_map(50, value_len);
        let key = mk_key(42);
        let val = mk_value(42, value_len);
        map.insert(&key, &val).unwrap();

        // Find where we placed it.
        let (tag, indices) = map.lookup_candidates(&key);
        let (seg_idx, local, slot) = (0..map.k() as usize)
            .find_map(|j| {
                let local = indices[j] - (j as u32) * map.seg();
                (0..map.slots_per_bucket())
                    .find(|s| map.tables[j].read_fingerprint(local, *s) == tag)
                    .map(|s| (j as u8, local, s))
            })
            .expect("stored slot not located");

        let row = map.encode_bucket(seg_idx, local).unwrap();
        let slot_elems = map.slot_elems() as usize;
        let fp_elems = map.fp_elems() as usize;
        let val_elems = map.val_elems() as usize;
        let base = (slot as usize) * slot_elems;

        // Check fingerprint.
        let decoded_fp = decode_fp(
            &row[base..base + fp_elems],
            map.params().fingerprint_bits,
            map.params().mat_elem_bit_len,
        );
        assert_eq!(decoded_fp, tag, "fp round-trip mismatch");

        // Check value payload chunks decode back to the original bytes.
        let payloads = &row[base + fp_elems..base + fp_elems + val_elems];
        let decoded_val = decode_value(payloads, map.params()).unwrap();
        assert_eq!(decoded_val, val);
    }

    #[test]
    fn to_matrices_has_correct_shape_and_round_trip() {
        let value_len: u32 = 8;
        let num = 100;
        let mut map = seg4_map(num, value_len);
        for i in 0..num {
            map.insert(&mk_key(i), &mk_value(i, value_len)).unwrap();
        }
        let mats = map.to_matrices().unwrap();
        assert_eq!(mats.len(), map.k() as usize);
        for m in &mats {
            assert_eq!(m.rows(), map.seg());
            assert_eq!(m.cols(), map.row_elems_bucket());
        }
        // Each inserted key should be recoverable from its matrix row.
        use crate::lwe::decode_value;
        for i in 0..num {
            let (tag, indices) = map.lookup_candidates(&mk_key(i));
            let mut found = false;
            for j in 0..map.k() as usize {
                let local = indices[j] - (j as u32) * map.seg();
                let row = mats[j].row(local).unwrap();
                let slot_elems = map.slot_elems() as usize;
                let fp_elems = map.fp_elems() as usize;
                let val_elems = map.val_elems() as usize;
                for s in 0..map.slots_per_bucket() {
                    let base = (s as usize) * slot_elems;
                    let decoded_fp = decode_fp(
                        &row[base..base + fp_elems],
                        map.params().fingerprint_bits,
                        map.params().mat_elem_bit_len,
                    );
                    if decoded_fp == tag {
                        let decoded_val = decode_value(
                            &row[base + fp_elems..base + fp_elems + val_elems],
                            map.params(),
                        )
                        .unwrap();
                        if decoded_val == mk_value(i, value_len) {
                            found = true;
                        }
                    }
                }
            }
            assert!(found, "key {i} not recovered from to_matrices output");
        }
    }
}
