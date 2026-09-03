//! `fold_mutations_into_row_deltas` — slot mutations → sparse per-segment
//! row deltas.
//!
//! # Purpose
//!
//! Bridge layer between the SCF-level mutation log
//! ([`segmented_cuckoo::SlotMutation`]) and the backend-level hint
//! patcher
//! ([`crate::IncrementalPirBackend::server_patch_hint`]). Turns
//! per-slot before/after cell snapshots into the sparse
//! `(row, [(offset, Δ), …])` format every incremental backend consumes.
//!
//! # Design / architecture
//!
//! Two-pass algorithm:
//!
//! 1. **Accumulate.** Walk `muts` once, packing each `(fp, value)` pair
//!    via [`pack_slot_cells`] and summing the cell deltas into a
//!    per-segment `BTreeMap<row, BTreeMap<offset, i64>>` accumulator.
//!    Multiple mutations on the same `(row, offset)` sum naturally.
//! 2. **Filter and emit.** Materialise the accumulator into the wire
//!    shape, dropping zero deltas and empty rows so the output stays
//!    sparse.
//!
//! `BTreeMap` (rather than `HashMap`) so emitted rows and offsets are in
//! sorted order — important for deterministic on-wire output and easier
//! diffing in tests.
//!
//! # Related files
//!
//! - `server.rs::commit_mutations` — sole caller.
//! - `ikpir_common::SegmentRowDeltas` — output type alias.
//! - `ikpir_common::IncrementalPirBackend::server_patch_hint` —
//!   consumes the output.

use std::collections::BTreeMap;

use segmented_cuckoo::{pack_slot_cells, CuckooParams, SlotMutation};

use ikpir_common::SegmentRowDeltas;

/// Fold a batch of slot-level mutations into per-segment, per-row sparse
/// cell deltas suitable for
/// [`IncrementalPirBackend::server_patch_hint`](crate::IncrementalPirBackend::server_patch_hint).
///
/// # Arguments
///
/// - `muts`   — slot mutations drained from the SCF mutation log.
/// - `params` — current store geometry; supplies arity, segment size,
///   `cells_per_slot`, and `pack_slot_cells` reuses its bit-layout
///   constants.
///
/// # Returns
///
/// `Vec<SegmentRowDeltas>` of length `arity`. `out[seg]` is the sparse
/// edit list `Vec<(row_in_segment, Vec<(cell_offset_in_row, delta)>)>`
/// for segment `seg`. Multiple mutations on the same row are summed;
/// zero deltas and empty rows are dropped before emission.
///
/// # Complexity
///
/// `O(|muts| · cells_per_slot)` for the accumulation pass plus
/// `O(touched_rows · log + touched_cells · log)` for the BTreeMap
/// operations (one batch per drain).
pub fn fold_mutations_into_row_deltas(
    muts: &[SlotMutation],
    params: &CuckooParams,
) -> Vec<SegmentRowDeltas> {
    let arity = params.arity();
    let segment_size = params.segment_size();
    let bucket_size = params.bucket_size;
    let cps = params.cells_per_slot() as usize;

    let mut acc: Vec<BTreeMap<u32, BTreeMap<u16, i64>>> =
        (0..arity).map(|_| BTreeMap::new()).collect();

    let mut old_buf = vec![0u32; cps];
    let mut new_buf = vec![0u32; cps];

    for m in muts {
        let seg_idx = (m.bucket / segment_size) as usize;
        let bucket_in_seg = m.bucket % segment_size;
        let slot = m.slot as usize;
        debug_assert!(seg_idx < arity);
        debug_assert!((slot as u32) < bucket_size);

        pack_slot_cells(params, m.old_fingerprint, &m.old_value_cells, &mut old_buf);
        pack_slot_cells(params, m.new_fingerprint, &m.new_value_cells, &mut new_buf);

        let row_entry = acc[seg_idx].entry(bucket_in_seg).or_default();
        for c in 0..cps {
            let delta = new_buf[c] as i64 - old_buf[c] as i64;
            if delta == 0 {
                continue;
            }
            let cell_offset: u16 = u16::try_from(slot * cps + c).expect("row width fits in u16");
            *row_entry.entry(cell_offset).or_insert(0) += delta;
        }
    }

    acc.into_iter()
        .map(|seg| {
            seg.into_iter()
                .filter_map(|(row, cells)| {
                    let v: Vec<(u16, i64)> = cells.into_iter().filter(|(_, d)| *d != 0).collect();
                    if v.is_empty() {
                        None
                    } else {
                        Some((row, v))
                    }
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`fold_mutations_into_row_deltas`].
    //!
    //! Pins the four behaviours every incremental hint patcher relies on:
    //! correct seg/row routing, insert/delete duality, mutation
    //! summation, and aggressive zero-pruning.

    use super::*;
    use segmented_cuckoo::CuckooParams;

    /// 2-ary fixture: 64 buckets (32 per segment), `bucket_size = 4`,
    /// `fingerprint_bits = 12`, `value_bits = 8`, `plaintext_bits = 8`.
    /// `cells_per_slot = ⌈(12+8)/8⌉ = 3`, `value_size_in_cells = 1`.
    fn params_2ary() -> CuckooParams {
        let store = segmented_cuckoo::Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
        store.params()
    }

    /// Build a `value_size_in_cells()`-long zeroed cell vector.
    fn zero_value_cells(params: &CuckooParams) -> Box<[u32]> {
        vec![0u32; params.value_size_in_cells() as usize].into_boxed_slice()
    }

    /// Build a `value_size_in_cells()`-long cell vector with `val` in
    /// cell `0` and zeros elsewhere — keeps the test deltas easy to
    /// reason about.
    fn nonzero_value_cells(params: &CuckooParams, val: u32) -> Box<[u32]> {
        let mut v = vec![0u32; params.value_size_in_cells() as usize];
        v[0] = val;
        v.into_boxed_slice()
    }

    /// A mutation on bucket 33 (= segment 1, bucket-in-seg 1) must land
    /// in `out[1]` only.
    #[test]
    fn single_insert_lands_in_correct_segment() {
        let p = params_2ary();
        // Bucket 33 → seg_idx = 33 / 32 = 1, bucket_in_seg = 1
        let m = SlotMutation {
            bucket: 33,
            slot: 0,
            old_fingerprint: 0,
            new_fingerprint: 7,
            old_value_cells: zero_value_cells(&p),
            new_value_cells: nonzero_value_cells(&p, 42),
        };
        let out = fold_mutations_into_row_deltas(&[m], &p);
        assert!(out[0].is_empty(), "seg 0 must be untouched");
        assert!(!out[1].is_empty(), "seg 1 must have an entry");
        assert_eq!(out[1][0].0, 1, "bucket_in_seg == 1");
    }

    /// Delete must produce the exact negation of the corresponding
    /// insert's deltas (cell-by-cell).
    #[test]
    fn single_delete_produces_negated_deltas() {
        let p = params_2ary();
        // Insert first, then delete — deltas should be negations of each other.
        let insert_m = SlotMutation {
            bucket: 5,
            slot: 1,
            old_fingerprint: 0,
            new_fingerprint: 99,
            old_value_cells: zero_value_cells(&p),
            new_value_cells: nonzero_value_cells(&p, 77),
        };
        let delete_m = SlotMutation {
            bucket: 5,
            slot: 1,
            old_fingerprint: 99,
            new_fingerprint: 0,
            old_value_cells: nonzero_value_cells(&p, 77),
            new_value_cells: zero_value_cells(&p),
        };
        let ins = fold_mutations_into_row_deltas(&[insert_m], &p);
        let del = fold_mutations_into_row_deltas(&[delete_m], &p);
        assert!(!ins[0].is_empty());
        assert!(!del[0].is_empty());
        for ((ir, ic), (dr, dc)) in ins[0].iter().zip(del[0].iter()) {
            assert_eq!(ir, dr);
            for ((io, id), (d_o, d_d)) in ic.iter().zip(dc.iter()) {
                assert_eq!(io, d_o);
                assert_eq!(*id, -*d_d, "delete delta must negate insert delta");
            }
        }
    }

    /// Two mutations on the same slot must sum: insert(val=10) then
    /// update(val=10 → val=20) must produce the same net delta as a
    /// single insert(val=20) from empty.
    #[test]
    fn two_mutations_same_slot_sum_correctly() {
        let p = params_2ary();
        // Insert (empty→fp=3, val=10), then update (fp=3,val=10 → fp=3,val=20).
        // Net should equal insert-from-empty with val=20.
        let insert_m = SlotMutation {
            bucket: 10,
            slot: 2,
            old_fingerprint: 0,
            new_fingerprint: 3,
            old_value_cells: zero_value_cells(&p),
            new_value_cells: nonzero_value_cells(&p, 10),
        };
        let update_m = SlotMutation {
            bucket: 10,
            slot: 2,
            old_fingerprint: 3,
            new_fingerprint: 3,
            old_value_cells: nonzero_value_cells(&p, 10),
            new_value_cells: nonzero_value_cells(&p, 20),
        };
        let net_expected = SlotMutation {
            bucket: 10,
            slot: 2,
            old_fingerprint: 0,
            new_fingerprint: 3,
            old_value_cells: zero_value_cells(&p),
            new_value_cells: nonzero_value_cells(&p, 20),
        };
        let chained = fold_mutations_into_row_deltas(&[insert_m, update_m], &p);
        let expected = fold_mutations_into_row_deltas(&[net_expected], &p);
        assert_eq!(
            chained, expected,
            "chained insert+update must equal single-step net delta"
        );
    }

    /// A no-op mutation (old == new) must emit nothing — the
    /// zero-delta filter drops the row entirely.
    #[test]
    fn noop_mutation_emits_nothing() {
        let p = params_2ary();
        // old == new → delta is zero → row entry must be dropped.
        let m = SlotMutation {
            bucket: 7,
            slot: 0,
            old_fingerprint: 5,
            new_fingerprint: 5,
            old_value_cells: nonzero_value_cells(&p, 99),
            new_value_cells: nonzero_value_cells(&p, 99),
        };
        let out = fold_mutations_into_row_deltas(&[m], &p);
        assert!(
            out.iter().all(|seg| seg.is_empty()),
            "no-op mutation must produce empty output"
        );
    }

    /// Mutations on different buckets within the same segment land in
    /// distinct row entries, sorted by row index (BTreeMap ordering).
    #[test]
    fn multiple_buckets_same_segment_preserve_sparsity() {
        let p = params_2ary();
        // Touch buckets 2 and 5 (both in seg 0) at different slots.
        let m1 = SlotMutation {
            bucket: 2,
            slot: 0,
            old_fingerprint: 0,
            new_fingerprint: 11,
            old_value_cells: zero_value_cells(&p),
            new_value_cells: nonzero_value_cells(&p, 1),
        };
        let m2 = SlotMutation {
            bucket: 5,
            slot: 1,
            old_fingerprint: 0,
            new_fingerprint: 22,
            old_value_cells: zero_value_cells(&p),
            new_value_cells: nonzero_value_cells(&p, 2),
        };
        let out = fold_mutations_into_row_deltas(&[m1, m2], &p);
        assert!(out[1].is_empty(), "seg 1 untouched");
        assert_eq!(out[0].len(), 2, "two rows in seg 0");
        // BTreeMap order: row 2 < row 5
        assert_eq!(out[0][0].0, 2);
        assert_eq!(out[0][1].0, 5);
    }

    /// Two writes to the SAME `(bucket, slot)` within one batch must
    /// telescope to `final − initial` exactly, and the result must
    /// satisfy `|delta| < p` for every cell -- even when each *individual*
    /// mutation's delta is itself near the `±(p−1)` extreme, so their
    /// naive (non-cancelling) sum could look like it exceeds `(−p, p)`.
    /// This pins the invariant `docs/hint-delta-wire-format.md` §8 relies
    /// on to justify `|γ| < p` as a protocol invariant for the wire
    /// encoder: it would catch a fold that clamped or reduced each
    /// mutation's delta independently (instead of summing exactly and
    /// letting the intermediate term cancel), which would corrupt the
    /// telescoped total whenever a slot is written more than once in a
    /// batch.
    #[test]
    fn repeated_writes_to_one_slot_telescope() {
        let p = params_2ary();

        // fp/value chosen to push every packed cell toward its bit-packed
        // maximum ("hits p-1 where possible" -- fp and value bits straddle
        // cell boundaries, so not every cell lands exactly on p-1, but
        // every cell is pushed to the top of its packed range).
        let old1_fp = 0u64;
        let old1_val = zero_value_cells(&p);
        let new1_fp = 0xFFFu64; // fingerprint_bits = 12, all bits set
        let new1_val = nonzero_value_cells(&p, 0xFF); // value_bits = 8, all bits set

        let new2_fp = 0x001u64;
        let new2_val = nonzero_value_cells(&p, 0x01);

        let m1 = SlotMutation {
            bucket: 9,
            slot: 2,
            old_fingerprint: old1_fp,
            new_fingerprint: new1_fp,
            old_value_cells: old1_val.clone(),
            new_value_cells: new1_val.clone(),
        };
        let m2 = SlotMutation {
            bucket: 9,
            slot: 2,
            old_fingerprint: new1_fp,
            new_fingerprint: new2_fp,
            old_value_cells: new1_val.clone(),
            new_value_cells: new2_val.clone(),
        };

        let batched = fold_mutations_into_row_deltas(&[m1, m2], &p);

        let net = SlotMutation {
            bucket: 9,
            slot: 2,
            old_fingerprint: old1_fp,
            new_fingerprint: new2_fp,
            old_value_cells: old1_val,
            new_value_cells: new2_val,
        };
        let expected = fold_mutations_into_row_deltas(&[net], &p);

        assert_eq!(
            batched, expected,
            "two writes to one slot in a batch must telescope to final - initial"
        );

        let p_bound = 1i64 << p.plaintext_bits;
        for seg in &batched {
            for (_row, cells) in seg {
                for (_off, delta) in cells {
                    assert!(
                        delta.abs() < p_bound,
                        "telescoped delta {delta} must satisfy |delta| < p ({p_bound})"
                    );
                }
            }
        }

        // Second case: the second mutation exactly restores the slot's
        // original (empty) contents, so the net delta is all-zero and the
        // row must be dropped entirely, even though each individual
        // mutation was itself a large, non-trivial write.
        let m1b = SlotMutation {
            bucket: 11,
            slot: 1,
            old_fingerprint: 0,
            new_fingerprint: new1_fp,
            old_value_cells: zero_value_cells(&p),
            new_value_cells: new1_val.clone(),
        };
        let m2b = SlotMutation {
            bucket: 11,
            slot: 1,
            old_fingerprint: new1_fp,
            new_fingerprint: 0,
            old_value_cells: new1_val,
            new_value_cells: zero_value_cells(&p),
        };
        let restored = fold_mutations_into_row_deltas(&[m1b, m2b], &p);
        assert!(
            restored.iter().all(|seg| seg.is_empty()),
            "a batch that restores a slot's original contents must emit nothing, \
             even though it is two large individual writes, not a single no-op"
        );
    }
}
