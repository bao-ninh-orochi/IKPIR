//! `CuckooKVStore<S>` — cuckoo-filter-backed key-value store.
//!
//! Each slot stores `fingerprint ‖ value` via [`FingerprintValueTable`]. The scheme
//! `S` determines arity and index layout (same [`IndexScheme`] variants as
//! [`CuckooFilter`](crate::CuckooFilter)).
//!
//! # Duplicate-key contract
//!
//! [`CuckooKVStore::insert`] does **not** deduplicate. Re-inserting the same key keeps
//! both copies in the store. [`get`](CuckooKVStore::get), [`delete`](CuckooKVStore::delete),
//! and [`update`](CuckooKVStore::update) all use **first-match-wins** semantics: the probe
//! order is candidate-bucket index `0..arity`, then lowest-index slot within that bucket.
//! To purge duplicates, call `delete` repeatedly until it returns
//! [`CuckooError::NotFound`].

// DECISION: indexed `for p in 0..arity` is used in the kicking loop because `p` indexes
// both the candidate-bucket array `all` and the chain-position bookkeeping. Mirrors the
// pattern in `filter.rs::insert`.
#![allow(clippy::needless_range_loop)]

use rand::Rng;

use crate::data_layout::FingerprintValueTable;
use crate::filter::{target_load_factor, validate_common_params, CuckooError, MAX_KICKS_DEFAULT};
use crate::scheme::{IndexScheme, Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};
use crate::util::next_power_of_2;

/// Generic cuckoo key-value store parameterised over an index scheme.
///
/// Stores `(key, value)` pairs in a bit-packed fingerprint-and-value array. The server
/// exposes this array to PIR queries; the client recovers `(fingerprint, value)` for a
/// target key using a single Index-PIR query over the fixed slot positions derived by
/// the segmented scheme.
///
/// Prefer the type aliases [`Segmented2aryCuckooKVStore`](crate::Segmented2aryCuckooKVStore),
/// [`Segmented3aryCuckooKVStore`](crate::Segmented3aryCuckooKVStore), and
/// [`Segmented4aryCuckooKVStore`](crate::Segmented4aryCuckooKVStore) over using this type
/// directly unless you are implementing a custom scheme.
pub struct CuckooKVStore<S: IndexScheme> {
    table: FingerprintValueTable,
    scheme: S,
    num_items: u64,
    max_kicks: u32,
    /// Per-kick metadata reused across `insert` calls; capacity tracks `max_kicks`.
    chain_meta: Vec<(u32, u32, u32)>,
    /// Slab of `max_kicks * value_size_in_bytes` bytes; slot k holds the original
    /// (evicted) value at the k-th kick, used for rollback.
    chain_values: Vec<u8>,
    /// Reusable buffer for the in-flight (displaced) value during kicking;
    /// length equals `value_size_in_bytes`.
    cur_value: Vec<u8>,
}

// ─── Constructors ───────────────────────────────────────────────────────────

impl CuckooKVStore<Segmented2aryScheme> {
    /// Create a segmented 2-ary KV store with explicit dimensions.
    ///
    /// # Arguments
    ///
    /// - `num_buckets`      — total number of buckets. Must be a power of 2 and ≥ 2.
    /// - `bucket_size`      — slots per bucket. Must be in `1..=4`.
    /// - `fingerprint_bits` — fingerprint bit width. Must be in `[⌊log2(2·bucket_size)⌋+1, 32]`.
    /// - `value_bits`       — value bit width. Must be ≥ 1.
    ///
    /// # Degenerate mode
    ///
    /// `num_buckets = 2` (the minimum) produces `segment_size = 1`. With `segment_size == 1`,
    /// every key maps to the same fixed indices `[0, 1]`, giving an FPR ≈ 1 once past
    /// `bucket_size` items. Functionally correct but only useful for minimal-table tests.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    pub fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
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
        let table = FingerprintValueTable::new(num_buckets, bucket_size, fingerprint_bits, value_bits);
        let vsize = table.value_size_in_bytes();
        Ok(CuckooKVStore {
            table,
            scheme: Segmented2aryScheme { segment_size: num_buckets / 2 },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
            chain_meta: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
            chain_values: vec![0u8; (MAX_KICKS_DEFAULT as usize).checked_mul(vsize).expect("max_kicks * value_size_in_bytes overflows usize")],
            cur_value: vec![0u8; vsize],
        })
    }

    /// Create a segmented 2-ary KV store sized to hold at least `max_items`.
    ///
    /// Rounds `num_buckets` up to the next valid value (power of 2, ≥ 2). If the projected
    /// load factor exceeds the empirical target for this `(arity=2, bucket_size)` configuration,
    /// `num_buckets` is doubled until the projection is within bounds.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `bucket_size`, `fingerprint_bits`, or
    /// `value_bits` are invalid.
    pub fn from_num_items(
        max_items: u64,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(2, bucket_size, fingerprint_bits)?;
        if value_bits == 0 {
            return Err(CuckooError::InvalidParams("value_bits must be >= 1".into()));
        }
        let max_items_cap: u64 = u32::MAX as u64 * bucket_size as u64;
        if max_items > max_items_cap {
            return Err(CuckooError::InvalidParams("max_items too large for u32 num_buckets".into()));
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
        Self::new(num_buckets, bucket_size, fingerprint_bits, value_bits)
    }
}

impl CuckooKVStore<Segmented3aryScheme> {
    /// Create a segmented 3-ary KV store with explicit dimensions.
    ///
    /// # Arguments
    ///
    /// - `num_buckets`      — total buckets. Must equal `3 · 2^t` for some `t ≥ 0`.
    /// - `bucket_size`      — slots per bucket. Must be in `1..=4`.
    /// - `fingerprint_bits` — fingerprint bit width. Must be in `[⌊log2(3·bucket_size)⌋+1, 32]`.
    /// - `value_bits`       — value bit width. Must be ≥ 1.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    pub fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
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
        let segment_size = num_buckets / 3;
        let table = FingerprintValueTable::new(num_buckets, bucket_size, fingerprint_bits, value_bits);
        let vsize = table.value_size_in_bytes();
        Ok(CuckooKVStore {
            table,
            scheme: Segmented3aryScheme { segment_size },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
            chain_meta: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
            chain_values: vec![0u8; (MAX_KICKS_DEFAULT as usize).checked_mul(vsize).expect("max_kicks * value_size_in_bytes overflows usize")],
            cur_value: vec![0u8; vsize],
        })
    }

    /// Create a segmented 3-ary KV store sized to hold at least `max_items`.
    ///
    /// Each segment must be a power of 2; this method computes the smallest valid
    /// `num_buckets = 3 · segment_size` where
    /// `segment_size = 2^t ≥ ceil(max_items / (3 · bucket_size))`, then doubles `segment_size`
    /// until the projected load factor is within the empirical target for `(arity=3, bucket_size)`.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `bucket_size`, `fingerprint_bits`, or
    /// `value_bits` are invalid.
    pub fn from_num_items(
        max_items: u64,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(3, bucket_size, fingerprint_bits)?;
        if value_bits == 0 {
            return Err(CuckooError::InvalidParams("value_bits must be >= 1".into()));
        }
        let max_items_cap: u64 = u32::MAX as u64 * bucket_size as u64;
        if max_items > max_items_cap {
            return Err(CuckooError::InvalidParams("max_items too large for u32 num_buckets".into()));
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
        Self::new(num_buckets, bucket_size, fingerprint_bits, value_bits)
    }
}

impl CuckooKVStore<Segmented4aryScheme> {
    /// Create a segmented 4-ary KV store with explicit dimensions.
    ///
    /// # Arguments
    ///
    /// - `num_buckets`      — total buckets. Must be a power of 2 and ≥ 4.
    /// - `bucket_size`      — slots per bucket. Must be in `1..=4`.
    /// - `fingerprint_bits` — fingerprint bit width. Must be in `[⌊log2(4·bucket_size)⌋+1, 32]`.
    /// - `value_bits`       — value bit width. Must be ≥ 1.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if any constraint is violated.
    pub fn new(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
    ) -> Result<Self, CuckooError> {
        if !num_buckets.is_power_of_two() || num_buckets < 4 {
            return Err(CuckooError::InvalidParams(
                "num_buckets must be a power of 2 and >= 4 (so each of 4 segments is a power of 2)".into(),
            ));
        }
        validate_common_params(4, bucket_size, fingerprint_bits)?;
        if value_bits == 0 {
            return Err(CuckooError::InvalidParams("value_bits must be >= 1".into()));
        }
        let segment_size = num_buckets / 4;
        let table = FingerprintValueTable::new(num_buckets, bucket_size, fingerprint_bits, value_bits);
        let vsize = table.value_size_in_bytes();
        Ok(CuckooKVStore {
            table,
            scheme: Segmented4aryScheme { segment_size },
            num_items: 0,
            max_kicks: MAX_KICKS_DEFAULT,
            chain_meta: Vec::with_capacity(MAX_KICKS_DEFAULT as usize),
            chain_values: vec![0u8; (MAX_KICKS_DEFAULT as usize).checked_mul(vsize).expect("max_kicks * value_size_in_bytes overflows usize")],
            cur_value: vec![0u8; vsize],
        })
    }

    /// Create a segmented 4-ary KV store sized to hold at least `max_items`.
    ///
    /// Each segment must be a power of 2; this method computes the smallest valid
    /// `num_buckets = 4 · segment_size`, then doubles `segment_size` until the projected
    /// load factor is within the empirical target for `(arity=4, bucket_size)`.
    ///
    /// # Errors
    ///
    /// Returns [`CuckooError::InvalidParams`] if `bucket_size`, `fingerprint_bits`, or
    /// `value_bits` are invalid.
    pub fn from_num_items(
        max_items: u64,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
    ) -> Result<Self, CuckooError> {
        validate_common_params(4, bucket_size, fingerprint_bits)?;
        if value_bits == 0 {
            return Err(CuckooError::InvalidParams("value_bits must be >= 1".into()));
        }
        let max_items_cap: u64 = u32::MAX as u64 * bucket_size as u64;
        if max_items > max_items_cap {
            return Err(CuckooError::InvalidParams("max_items too large for u32 num_buckets".into()));
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
        Self::new(num_buckets, bucket_size, fingerprint_bits, value_bits)
    }
}

// ─── Generic operations ─────────────────────────────────────────────────────

impl<S: IndexScheme> CuckooKVStore<S> {
    /// Insert `(key, value)` into the store.
    ///
    /// Tries direct insertion into any of the arity candidate buckets. If all are full,
    /// runs a cuckoo-kicking loop that evicts and relocates `(fingerprint, value)` pairs
    /// up to `max_kicks` times. On exhaustion, every mutation is rolled back and
    /// [`CuckooError::TableFull`] is returned (the store is observably unchanged).
    ///
    /// # Duplicate keys
    ///
    /// Duplicates are **allowed**: re-inserting the same key keeps both copies. [`get`](Self::get)
    /// returns the lowest-index match, so the most recently inserted value is observable
    /// only after deleting earlier copies.
    ///
    /// # Arguments
    ///
    /// - `key`   — arbitrary bytes identifying the entry.
    /// - `value` — bytes to store. Must have length `ceil(value_bits / 8)`.
    ///
    /// # Errors
    ///
    /// - [`CuckooError::TableFull`] — kick budget exhausted; store is unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `value.len() != ceil(value_bits / 8)`. Once the kicking loop has begun, a
    /// panic from any source leaves the store in an inconsistent state (matches
    /// [`CuckooFilter::insert`](crate::CuckooFilter::insert) behaviour).
    pub fn insert<K: AsRef<[u8]>>(&mut self, key: K, value: &[u8]) -> Result<(), CuckooError> {
        let vsize = self.table.value_size_in_bytes();
        assert_eq!(
            value.len(),
            vsize,
            "value buffer length must equal value_size_in_bytes ({vsize})"
        );

        let (fingerprint, indices) = self
            .scheme
            .hash_item(key.as_ref(), self.table.fingerprint_bits());
        let arity = self.scheme.arity();

        for pos in 0..arity {
            if self.table.insert(indices[pos], fingerprint, value).is_some() {
                self.num_items += 1;
                return Ok(());
            }
        }

        let mut rng = rand::rng();
        let start_pos = rng.random_range(0..arity as u32) as usize;
        let mut cur_index = indices[start_pos];
        let mut cur_fingerprint = fingerprint;
        // Stage the in-flight value into the pre-allocated cur_value buffer.
        self.cur_value.copy_from_slice(value);
        // chain_meta[k] = (bucket, slot, original_fingerprint).
        // chain_values[k*vsize..(k+1)*vsize] = original (evicted) value at kick k.
        self.chain_meta.clear();

        for _ in 0..self.max_kicks {
            let kick_idx = self.chain_meta.len();
            let slab_off = kick_idx * vsize;
            let slot = rng.random_range(0..self.table.slots_per_bucket());
            let evicted_fp = self.table.read_fingerprint(cur_index, slot);

            // Stash the slot's original value in chain_values for rollback.
            self.table
                .read_value(cur_index, slot, &mut self.chain_values[slab_off..slab_off + vsize]);
            self.chain_meta.push((cur_index, slot, evicted_fp));

            // Write the in-flight value into the slot.
            self.table
                .write(cur_index, slot, cur_fingerprint, &self.cur_value);

            cur_fingerprint = evicted_fp;
            // The just-evicted value (now in chain_values[k]) becomes the next cur_value.
            self.cur_value
                .copy_from_slice(&self.chain_values[slab_off..slab_off + vsize]);

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
                if self
                    .table
                    .insert(all[p], cur_fingerprint, &self.cur_value)
                    .is_some()
                {
                    self.num_items += 1;
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
        for kick_idx in (0..self.chain_meta.len()).rev() {
            let (bucket, slot, original_fp) = self.chain_meta[kick_idx];
            let slab_off = kick_idx * vsize;
            self.table.write(
                bucket,
                slot,
                original_fp,
                &self.chain_values[slab_off..slab_off + vsize],
            );
        }
        Err(CuckooError::TableFull)
    }

    /// Return `true` if `key` is likely present in the store.
    ///
    /// Probes all arity candidate buckets for a matching fingerprint. May return `true`
    /// for keys that were never inserted (false positive). Never returns `false` for keys
    /// currently in the store.
    pub fn contain<K: AsRef<[u8]>>(&self, key: K) -> bool {
        let (fingerprint, indices) = self
            .scheme
            .hash_item(key.as_ref(), self.table.fingerprint_bits());
        let arity = self.scheme.arity();
        (0..arity).any(|i| self.table.contain(indices[i], fingerprint))
    }

    /// Retrieve the value stored for `key`, if present.
    ///
    /// Returns `Some(value_bytes)` for the first matching `(bucket, slot)` in probe order
    /// (candidate-bucket index `0..arity`, then lowest-index slot within that bucket).
    /// `None` if no match is found. Subject to the same false-positive caveat as
    /// [`contain`](Self::contain).
    pub fn get<K: AsRef<[u8]>>(&self, key: K) -> Option<Vec<u8>> {
        let mut out = vec![0u8; self.table.value_size_in_bytes()];
        if self.get_into(key, &mut out) { Some(out) } else { None }
    }

    /// Number of bytes needed to hold one stored value: `ceil(value_bits / 8)`.
    ///
    /// Inspection helper for callers (e.g., the `ikpir-server` crate) that need to size
    /// value buffers without knowing the slot layout.
    #[inline]
    pub fn value_size_in_bytes(&self) -> usize {
        self.table.value_size_in_bytes()
    }

    /// Read the value for `key` into `out`, returning `true` on hit.
    ///
    /// Zero-allocation read path: the caller supplies a buffer of exactly
    /// [`value_size_in_bytes`](Self::value_size_in_bytes) bytes; on a hit the value is
    /// written into it. On a miss `out` is left untouched and `false` is returned.
    ///
    /// Subject to the same false-positive caveat as [`contain`](Self::contain).
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != value_size_in_bytes()`.
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
        for i in 0..arity {
            if let Some(slot) = self.table.find(indices[i], fingerprint) {
                self.table.read_value(indices[i], slot, out);
                return true;
            }
        }
        false
    }

    /// Delete the entry for `key` from the store.
    ///
    /// Removes only the **first** matching slot in probe order. To purge duplicates of
    /// the same key, call this method repeatedly until it returns
    /// [`CuckooError::NotFound`].
    ///
    /// # Errors
    ///
    /// - [`CuckooError::NotFound`] — no matching fingerprint found; store is unchanged.
    pub fn delete<K: AsRef<[u8]>>(&mut self, key: K) -> Result<(), CuckooError> {
        let (fingerprint, indices) = self
            .scheme
            .hash_item(key.as_ref(), self.table.fingerprint_bits());
        let arity = self.scheme.arity();
        for i in 0..arity {
            if self.table.delete(indices[i], fingerprint) {
                self.num_items -= 1;
                return Ok(());
            }
        }
        Err(CuckooError::NotFound)
    }

    /// Update the value for `key` to `new_value`.
    ///
    /// Mutates only the **first** matching slot in probe order. To update all duplicates of
    /// the same key, call this method repeatedly. Does not change [`num_items`](Self::num_items).
    ///
    /// # Arguments
    ///
    /// - `key`       — key to update.
    /// - `new_value` — replacement value bytes. Must have length `ceil(value_bits / 8)`.
    ///
    /// # Errors
    ///
    /// - [`CuckooError::NotFound`] — no matching fingerprint found; store is unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `new_value.len() != ceil(value_bits / 8)`.
    pub fn update<K: AsRef<[u8]>>(&mut self, key: K, new_value: &[u8]) -> Result<(), CuckooError> {
        let vsize = self.table.value_size_in_bytes();
        assert_eq!(
            new_value.len(),
            vsize,
            "new_value buffer length must equal value_size_in_bytes ({vsize})"
        );
        let (fingerprint, indices) = self
            .scheme
            .hash_item(key.as_ref(), self.table.fingerprint_bits());
        let arity = self.scheme.arity();
        for i in 0..arity {
            if self.table.update_value(indices[i], fingerprint, new_value) {
                return Ok(());
            }
        }
        Err(CuckooError::NotFound)
    }

    /// Return the number of items currently stored.
    ///
    /// Maintained exactly by `insert` and `delete`; not affected by false positives.
    pub fn num_items(&self) -> u64 {
        self.num_items
    }

    /// Return the logical byte size of the underlying fingerprint-value storage.
    ///
    /// Reports `ceil(num_buckets · bucket_size · (fingerprint_bits + value_bits) / 8)` bytes,
    /// excluding alignment padding.
    pub fn size_in_bytes(&self) -> usize {
        self.table.size_in_bytes()
    }

    /// Return the current load factor: `num_items / (num_buckets · bucket_size)`.
    pub fn load_factor(&self) -> f64 {
        self.num_items as f64
            / (self.table.num_buckets() as f64 * self.table.slots_per_bucket() as f64)
    }

    /// Override the maximum number of cuckoo kicks before declaring the store full.
    ///
    /// The default is `500`. A higher value increases the probability of a successful insert
    /// at high load factors at the cost of slower worst-case inserts. Set to `0` to disable
    /// kicking entirely (only direct placements succeed).
    ///
    /// Grows `chain_meta`'s capacity (never shrinks it) and resizes `chain_values` exactly
    /// to the new budget so subsequent `insert` calls remain alloc-free.
    pub fn set_max_kicks(&mut self, max_kicks: u32) {
        self.max_kicks = max_kicks;
        let vsize = self.table.value_size_in_bytes();
        self.chain_meta.reserve(max_kicks as usize);
        self.chain_values.resize((max_kicks as usize).checked_mul(vsize).expect("max_kicks * value_size_in_bytes overflows usize"), 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_value(n: usize, seed: u8) -> Vec<u8> {
        (0..n).map(|i| (i as u8).wrapping_mul(37).wrapping_add(seed)).collect()
    }

    // ─── Original smoke tests (kept) ─────────────────────────────────────

    #[test]
    fn new_works() {
        let store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64).unwrap();
        assert_eq!(store.num_items(), 0);
        // 64 buckets × 4 slots × (12 + 64) bits = 64*4*76 = 19456 bits = 2432 bytes
        assert_eq!(store.size_in_bytes(), 2432);
    }

    #[test]
    fn new_3ary_works() {
        let store = CuckooKVStore::<Segmented3aryScheme>::new(96, 4, 12, 64).unwrap();
        assert_eq!(store.num_items(), 0);
        assert!(store.size_in_bytes() > 0);
    }

    #[test]
    fn new_4ary_works() {
        let store = CuckooKVStore::<Segmented4aryScheme>::new(64, 4, 12, 64).unwrap();
        assert_eq!(store.num_items(), 0);
        assert!(store.size_in_bytes() > 0);
    }

    #[test]
    fn new_returns_err_on_invalid_num_buckets() {
        assert!(CuckooKVStore::<Segmented2aryScheme>::new(3, 4, 12, 64).is_err());
    }

    #[test]
    fn new_returns_err_on_zero_value_bits() {
        assert!(CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 0).is_err());
    }

    // ─── End-to-end behaviour ────────────────────────────────────────────

    #[test]
    fn insert_get_roundtrip() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64).unwrap();
        let value = make_value(8, 1);
        store.insert("hello", &value).unwrap();
        assert_eq!(store.get("hello"), Some(value));
        assert_eq!(store.num_items(), 1);
    }

    #[test]
    fn insert_update_get_returns_new_value() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64).unwrap();
        let v1 = make_value(8, 1);
        let v2 = make_value(8, 2);
        store.insert("k", &v1).unwrap();
        store.update("k", &v2).unwrap();
        assert_eq!(store.get("k"), Some(v2));
    }

    #[test]
    fn insert_delete_contain_returns_false() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64).unwrap();
        let v = make_value(8, 7);
        store.insert("k", &v).unwrap();
        store.delete("k").unwrap();
        assert!(!store.contain("k"));
        assert!(matches!(store.delete("k"), Err(CuckooError::NotFound)));
        assert_eq!(store.num_items(), 0);
    }

    #[test]
    fn delete_of_never_inserted_returns_not_found() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64).unwrap();
        assert!(matches!(store.delete("missing"), Err(CuckooError::NotFound)));
    }

    #[test]
    fn update_of_never_inserted_returns_not_found() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64).unwrap();
        let v = make_value(8, 1);
        assert!(matches!(store.update("missing", &v), Err(CuckooError::NotFound)));
    }

    #[test]
    fn update_does_not_change_num_items() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64).unwrap();
        let v1 = make_value(8, 1);
        let v2 = make_value(8, 2);
        store.insert("k", &v1).unwrap();
        let before = store.num_items();
        store.update("k", &v2).unwrap();
        assert_eq!(store.num_items(), before);
    }

    #[test]
    fn table_full_triggers_rollback() {
        // Tiny store: 2 buckets, 1 slot/bucket. fp_bits=24 keeps fingerprint collisions
        // vanishingly rare so the rollback assertion below isn't undermined by FPR.
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(2, 1, 24, 8).unwrap();
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
        // Every previously-successful insert must still be retrievable byte-for-byte.
        for (i, v) in &inserted {
            assert_eq!(
                store.get(i.to_le_bytes()),
                Some(v.to_vec()),
                "rollback failed for key {i}"
            );
        }
        assert_eq!(store.num_items(), inserted.len() as u64);
    }

    #[test]
    fn kicking_under_load_2ary() {
        // fp_bits=32 makes inter-key fingerprint collisions effectively impossible, so
        // the get-by-key assertion is sound for all 1000 keys.
        let mut store = CuckooKVStore::<Segmented2aryScheme>::from_num_items(1000, 4, 32, 64).unwrap();
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

    #[test]
    fn kicking_under_load_3ary() {
        let mut store = CuckooKVStore::<Segmented3aryScheme>::from_num_items(1000, 4, 32, 64).unwrap();
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

    #[test]
    fn kicking_under_load_4ary() {
        let mut store = CuckooKVStore::<Segmented4aryScheme>::from_num_items(1000, 4, 32, 64).unwrap();
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

    #[test]
    fn wide_value_roundtrip_through_kicking() {
        // value_bits = 1024 → 128 bytes per value. fp_bits=32 to avoid FPR-driven false
        // matches that would invalidate the byte-for-byte assertion.
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(8, 2, 32, 1024).unwrap();
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

    #[test]
    fn duplicate_insert_first_match_wins() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8).unwrap();
        let v1 = [0xAA];
        let v2 = [0xBB];
        store.insert("k", &v1).unwrap();
        store.insert("k", &v2).unwrap();
        // First insert sits in the lowest-index candidate's lowest-index slot;
        // get returns it.
        assert_eq!(store.get("k"), Some(v1.to_vec()));
        store.delete("k").unwrap();
        // After deleting the first match, get returns the second copy.
        assert_eq!(store.get("k"), Some(v2.to_vec()));
        store.delete("k").unwrap();
        // Both copies purged.
        assert_eq!(store.get("k"), None);
        assert!(matches!(store.delete("k"), Err(CuckooError::NotFound)));
    }

    #[test]
    fn set_max_kicks_zero_disables_kicking() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(2, 1, 2, 8).unwrap();
        store.set_max_kicks(0);
        // Capacity = 2 buckets × 1 slot = 2 entries; the third overflowing insert must
        // hit `TableFull` immediately because kicking is disabled.
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

    #[test]
    fn load_factor_invariant() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8).unwrap();
        let unit = 1.0 / (64.0 * 4.0);
        assert_eq!(store.load_factor(), 0.0);
        store.insert("a", &[1]).unwrap();
        assert!((store.load_factor() - unit).abs() < f64::EPSILON);
        store.insert("b", &[2]).unwrap();
        assert!((store.load_factor() - 2.0 * unit).abs() < f64::EPSILON);
        store.delete("a").unwrap();
        assert!((store.load_factor() - unit).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "value_size_in_bytes")]
    fn insert_panics_on_wrong_value_length() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 16).unwrap();
        let _ = store.insert("k", &[0u8]); // value_size = 2 bytes; 1 byte is wrong
    }

    #[test]
    #[should_panic(expected = "value_size_in_bytes")]
    fn update_panics_on_wrong_value_length() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 16).unwrap();
        store.insert("k", &[0u8, 0]).unwrap();
        let _ = store.update("k", &[0u8]); // wrong length
    }

    #[test]
    fn from_num_items_2ary_works() {
        let store =
            CuckooKVStore::<Segmented2aryScheme>::from_num_items(10_000, 4, 12, 64).unwrap();
        assert!(store.size_in_bytes() > 0);
    }

    #[test]
    fn from_num_items_3ary_works() {
        let store =
            CuckooKVStore::<Segmented3aryScheme>::from_num_items(10_000, 4, 12, 64).unwrap();
        assert!(store.size_in_bytes() > 0);
    }

    #[test]
    fn from_num_items_4ary_works() {
        let store =
            CuckooKVStore::<Segmented4aryScheme>::from_num_items(10_000, 4, 12, 64).unwrap();
        assert!(store.size_in_bytes() > 0);
    }

    #[test]
    fn from_num_items_returns_err_on_zero_value_bits() {
        assert!(CuckooKVStore::<Segmented2aryScheme>::from_num_items(100, 4, 12, 0).is_err());
        assert!(CuckooKVStore::<Segmented3aryScheme>::from_num_items(100, 4, 12, 0).is_err());
        assert!(CuckooKVStore::<Segmented4aryScheme>::from_num_items(100, 4, 12, 0).is_err());
    }

    // ─── get_into / value_size_in_bytes (Commit 4) ───────────────────────

    #[test]
    fn get_into_hit_writes_value_returns_true() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64).unwrap();
        let v = make_value(8, 9);
        store.insert("k", &v).unwrap();
        let mut out = vec![0u8; store.value_size_in_bytes()];
        assert!(store.get_into("k", &mut out));
        assert_eq!(out, v);
    }

    #[test]
    fn get_into_miss_returns_false() {
        let store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64).unwrap();
        let mut out = vec![0u8; store.value_size_in_bytes()];
        assert!(!store.get_into("missing", &mut out));
        assert!(out.iter().all(|&b| b == 0), "out should remain untouched on miss");
    }

    #[test]
    #[should_panic(expected = "value_size_in_bytes")]
    fn get_into_panics_on_wrong_length() {
        let store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 16).unwrap();
        let mut out = [0u8; 1]; // value_size = 2 bytes
        store.get_into("k", &mut out);
    }

    // ─── Pre-allocated rollback buffers (Commit 3) ───────────────────────

    #[test]
    fn repeated_inserts_reuse_buffers() {
        // Inserting many items must not grow the rollback buffers past max_kicks.
        let mut store =
            CuckooKVStore::<Segmented2aryScheme>::from_num_items(2000, 4, 12, 64).unwrap();
        let initial_meta_cap = store.chain_meta.capacity();
        let initial_values_len = store.chain_values.len();
        for i in 0u32..1000 {
            let v = make_value(8, i as u8);
            // Don't care about TableFull at this scale; just make sure we exercise kicking.
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

    #[test]
    fn set_max_kicks_grows_buffers() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 64).unwrap();
        let vsize = store.table.value_size_in_bytes();
        store.set_max_kicks(2000);
        assert!(store.chain_meta.capacity() >= 2000);
        assert_eq!(store.chain_values.len(), 2000 * vsize);
        // Shrink: chain_values resizes down (Vec::resize). chain_meta keeps capacity.
        store.set_max_kicks(100);
        assert_eq!(store.chain_values.len(), 100 * vsize);
    }

    #[test]
    fn failed_insert_then_successful_insert() {
        // Force a TableFull (rollback path), then verify the next insert succeeds and
        // prior keys are untouched.
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(2, 1, 24, 8).unwrap();
        // Fill until we get a TableFull.
        let mut placed: Vec<(u32, [u8; 1])> = Vec::new();
        for i in 0u32..200 {
            let v = [(i & 0xFF) as u8];
            match store.insert(i.to_le_bytes(), &v) {
                Ok(()) => placed.push((i, v)),
                Err(CuckooError::TableFull) => break,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert!(!placed.is_empty(), "should have placed something before failing");
        // After the failure, every prior key must still resolve correctly.
        for (i, v) in &placed {
            assert_eq!(store.get(i.to_le_bytes()), Some(v.to_vec()));
        }
        // Delete one to free a slot, then insert a new key — must succeed cleanly.
        let (k, _) = placed[0];
        store.delete(k.to_le_bytes()).unwrap();
        let new_v = [0xEE];
        store.insert(b"fresh".as_ref(), &new_v).unwrap();
        assert_eq!(store.get(b"fresh".as_ref()), Some(new_v.to_vec()));
    }

    #[test]
    fn from_num_items_rejects_overflow_max_items() {
        let huge: u64 = u32::MAX as u64 * 5;
        assert!(CuckooKVStore::<Segmented2aryScheme>::from_num_items(huge, 4, 12, 8).is_err());
        assert!(CuckooKVStore::<Segmented3aryScheme>::from_num_items(huge, 4, 12, 8).is_err());
        assert!(CuckooKVStore::<Segmented4aryScheme>::from_num_items(huge, 4, 12, 8).is_err());
    }

    #[test]
    #[cfg(target_pointer_width = "32")]
    #[should_panic(expected = "overflows usize")]
    fn set_max_kicks_panics_on_overflow_value_bits() {
        let mut store = CuckooKVStore::<Segmented2aryScheme>::new(64, 4, 12, 8).unwrap();
        store.set_max_kicks(u32::MAX);
    }
}
