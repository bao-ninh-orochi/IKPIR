use std::collections::BTreeMap;

use segmented_cuckoo::{pack_slot_cells, CuckooParams, SlotMutation};

use crate::wire::SegmentRowDeltas;

/// Fold a batch of slot-level mutations into per-segment, per-row sparse cell
/// deltas suitable for `IncrementalPirBackend::server_patch_hint`.
///
/// Output: `out[seg_idx]` is `Vec<(row_idx_in_segment, Vec<(cell_offset_in_row, delta)>)>`.
/// Multiple mutations on the same row are summed; zero deltas and empty rows are dropped.
pub(crate) fn fold_mutations_into_row_deltas(
    muts: &[SlotMutation],
    params: &CuckooParams,
) -> Vec<SegmentRowDeltas> {
    let arity        = params.arity();
    let segment_size = params.segment_size();
    let bucket_size  = params.bucket_size;
    let cps          = params.cells_per_slot() as usize;

    let mut acc: Vec<BTreeMap<u32, BTreeMap<u16, i64>>> =
        (0..arity).map(|_| BTreeMap::new()).collect();

    let mut old_buf = vec![0u32; cps];
    let mut new_buf = vec![0u32; cps];

    for m in muts {
        let seg_idx       = (m.bucket / segment_size) as usize;
        let bucket_in_seg = m.bucket % segment_size;
        let slot          = m.slot as usize;
        debug_assert!(seg_idx < arity);
        debug_assert!((slot as u32) < bucket_size);

        pack_slot_cells(params, m.old_fingerprint, &m.old_value_cells, &mut old_buf);
        pack_slot_cells(params, m.new_fingerprint, &m.new_value_cells, &mut new_buf);

        let row_entry = acc[seg_idx].entry(bucket_in_seg).or_default();
        for c in 0..cps {
            let delta = new_buf[c] as i64 - old_buf[c] as i64;
            if delta == 0 { continue; }
            let cell_offset: u16 =
                u16::try_from(slot * cps + c).expect("row width fits in u16");
            *row_entry.entry(cell_offset).or_insert(0) += delta;
        }
    }

    acc.into_iter()
        .map(|seg| {
            seg.into_iter()
                .filter_map(|(row, cells)| {
                    let v: Vec<(u16, i64)> = cells.into_iter()
                        .filter(|(_, d)| *d != 0)
                        .collect();
                    if v.is_empty() { None } else { Some((row, v)) }
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use segmented_cuckoo::CuckooParams;

    fn params_2ary() -> CuckooParams {
        // 64 buckets, arity 2 → 32 per segment, b=4, fp=12, vb=8, pb=8
        // cells_per_slot = ceil((12+8)/8) = 3; value_size_in_cells = ceil(8/8) = 1
        let store = segmented_cuckoo::Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
        store.params()
    }

    fn zero_value_cells(params: &CuckooParams) -> Box<[u32]> {
        vec![0u32; params.value_size_in_cells() as usize].into_boxed_slice()
    }

    fn nonzero_value_cells(params: &CuckooParams, val: u32) -> Box<[u32]> {
        let mut v = vec![0u32; params.value_size_in_cells() as usize];
        v[0] = val;
        v.into_boxed_slice()
    }

    #[test]
    fn single_insert_lands_in_correct_segment() {
        let p = params_2ary();
        // Bucket 33 → seg_idx = 33 / 32 = 1, bucket_in_seg = 1
        let m = SlotMutation {
            bucket:         33,
            slot:           0,
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

    #[test]
    fn single_delete_produces_negated_deltas() {
        let p = params_2ary();
        // Insert first, then delete — deltas should be negations of each other.
        let insert_m = SlotMutation {
            bucket:         5,
            slot:           1,
            old_fingerprint: 0,
            new_fingerprint: 99,
            old_value_cells: zero_value_cells(&p),
            new_value_cells: nonzero_value_cells(&p, 77),
        };
        let delete_m = SlotMutation {
            bucket:         5,
            slot:           1,
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

    #[test]
    fn two_mutations_same_slot_sum_correctly() {
        let p = params_2ary();
        // Insert (empty→fp=3, val=10), then update (fp=3,val=10 → fp=3,val=20).
        // Net should equal insert-from-empty with val=20.
        let insert_m = SlotMutation {
            bucket:         10,
            slot:           2,
            old_fingerprint: 0,
            new_fingerprint: 3,
            old_value_cells: zero_value_cells(&p),
            new_value_cells: nonzero_value_cells(&p, 10),
        };
        let update_m = SlotMutation {
            bucket:         10,
            slot:           2,
            old_fingerprint: 3,
            new_fingerprint: 3,
            old_value_cells: nonzero_value_cells(&p, 10),
            new_value_cells: nonzero_value_cells(&p, 20),
        };
        let net_expected = SlotMutation {
            bucket:         10,
            slot:           2,
            old_fingerprint: 0,
            new_fingerprint: 3,
            old_value_cells: zero_value_cells(&p),
            new_value_cells: nonzero_value_cells(&p, 20),
        };
        let chained  = fold_mutations_into_row_deltas(&[insert_m, update_m], &p);
        let expected = fold_mutations_into_row_deltas(&[net_expected], &p);
        assert_eq!(chained, expected, "chained insert+update must equal single-step net delta");
    }

    #[test]
    fn noop_mutation_emits_nothing() {
        let p = params_2ary();
        // old == new → delta is zero → row entry must be dropped.
        let m = SlotMutation {
            bucket:         7,
            slot:           0,
            old_fingerprint: 5,
            new_fingerprint: 5,
            old_value_cells: nonzero_value_cells(&p, 99),
            new_value_cells: nonzero_value_cells(&p, 99),
        };
        let out = fold_mutations_into_row_deltas(&[m], &p);
        assert!(out.iter().all(|seg| seg.is_empty()), "no-op mutation must produce empty output");
    }

    #[test]
    fn multiple_buckets_same_segment_preserve_sparsity() {
        let p = params_2ary();
        // Touch buckets 2 and 5 (both in seg 0) at different slots.
        let m1 = SlotMutation {
            bucket:         2,
            slot:           0,
            old_fingerprint: 0,
            new_fingerprint: 11,
            old_value_cells: zero_value_cells(&p),
            new_value_cells: nonzero_value_cells(&p, 1),
        };
        let m2 = SlotMutation {
            bucket:         5,
            slot:           1,
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
}
