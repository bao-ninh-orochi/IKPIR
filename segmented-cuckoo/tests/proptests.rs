//! Property-based tests for `segmented-cuckoo`.
//!
//! Uses `proptest` (already a dev-dep). Each test runs 64 cases by default.

use proptest::prelude::*;
use segmented_cuckoo::{pack_slot_cells, unpack_slot_cells, CuckooParams, SchemeKind, Segmented2aryCuckooKVStore};

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn make_store(
    num_buckets: u32,
    bucket_size: u32,
    fp_bits: u32,
    value_bits: u32,
    pb: u32,
) -> Segmented2aryCuckooKVStore {
    Segmented2aryCuckooKVStore::new(num_buckets, bucket_size, fp_bits, value_bits, pb)
        .expect("constructor failed with valid params")
}

// ─── FVT cell-layout property ────────────────────────────────────────────────
//
// Random valid (pb, fp_bits, value_bits) triple. After writing a key, the value
// must round-trip and no high bits should be set in any cell.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn prop_fvt_cell_layout_roundtrip(
        pb in 1u32..=32u32,
        fp_bits in 5u32..=32u32,
        value_bits in 1u32..=256u32,
        seed in 0u8..=255u8,
    ) {
        // bucket_size=4, arity=2: fp_bits must be > floor(log2(8)) = 3.
        // We use fp_bits starting from 5 to always satisfy this.
        let mut store = make_store(8, 4, fp_bits, value_bits, pb);
        let vbytes = store.value_size_in_bytes();

        let mut v = vec![0u8; vbytes];
        for (i, b) in v.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(seed);
        }
        // Mask the last byte if value_bits is not byte-aligned.
        if value_bits % 8 != 0 {
            if let Some(last) = v.last_mut() {
                *last &= (1u8 << (value_bits % 8)) - 1;
            }
        }

        // A failed insert (TableFull) is fine; just skip the roundtrip check.
        if store.insert(b"prop_key" as &[u8], &v).is_ok() {
            if let Some(got) = store.get(b"prop_key" as &[u8]) {
                prop_assert_eq!(got, v,
                    "value round-trip failed at pb={} fp={} vb={}", pb, fp_bits, value_bits);
            }
        }

        // High-bits-zero invariant must hold regardless.
        if pb < 32 {
            let hi_mask = !((1u32 << pb) - 1);
            for &c in store.as_cells() {
                prop_assert_eq!(c & hi_mask, 0, "high bits set in cell (pb={})", pb);
            }
        }
    }

    // ─── Mutation-log replay property ────────────────────────────────────────
    //
    // Perform a random mix of inserts and deletes with the log enabled.
    // Drain all mutations, replay them onto a fresh empty store via `apply_mutation`,
    // and assert `as_cells()` matches.

    #[test]
    fn prop_mutation_log_replay(
        ops in proptest::collection::vec(
            (0u32..30u32, 0u8..=255u8, 0u8..2u8),  // (key_idx, value_byte, op: 0=insert 1=delete)
            1..20,
        ),
    ) {
        let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
        store.enable_mutation_log();

        for (k, v, op) in &ops {
            match op {
                0 => { let _ = store.insert(k.to_le_bytes(), &[*v]); }
                _ => { let _ = store.delete(k.to_le_bytes()); }
            }
        }

        let mutations = store.drain_mutations();
        let original_cells = store.snapshot_cells();
        let p = store.params();

        // Replay on a fresh empty store.
        let mut fresh = Segmented2aryCuckooKVStore::new(
            p.num_buckets, p.bucket_size, p.fingerprint_bits, p.value_bits, p.plaintext_bits,
        )
        .unwrap();
        for m in &mutations {
            fresh.apply_mutation(m);
        }

        prop_assert_eq!(
            fresh.as_cells(),
            original_cells.as_slice(),
            "replayed mutations did not reproduce the original cell array",
        );
    }

    // ─── Snapshot / restore property ─────────────────────────────────────────

    #[test]
    fn prop_snapshot_restore(
        keys in proptest::collection::vec(0u32..100u32, 1..20),
        seed in 0u8..=255u8,
    ) {
        let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
        let v = [seed];
        let mut inserted = vec![];
        for k in &keys {
            if store.insert(k.to_le_bytes(), &v).is_ok() {
                inserted.push(*k);
            }
        }

        let cells = store.snapshot_cells();
        let p = store.params();
        let n = store.num_items();

        let restored = Segmented2aryCuckooKVStore::from_cells(cells, p, n).unwrap();
        prop_assert_eq!(restored.num_items(), n);
        for k in &inserted {
            prop_assert!(
                restored.get(k.to_le_bytes()).is_some(),
                "key {k} missing after restore",
            );
        }
    }

    // ─── candidate_buckets property ──────────────────────────────────────────
    //
    // params().candidate_buckets(key) must return non-zero fp and indices
    // within the expected segment ranges.

    #[test]
    fn prop_candidate_buckets_in_range(
        key in proptest::collection::vec(any::<u8>(), 0..32),
    ) {
        let store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
        let p = store.params();
        let (fp, indices) = p.candidate_buckets(&key);
        prop_assert_ne!(fp, 0, "fingerprint must never be zero");
        // 2-ary: indices[0] in [0, segment_size), indices[1] in [segment_size, num_buckets)
        let seg = p.segment_size();
        prop_assert!(indices[0] < seg, "i0={} must be < segment_size={}", indices[0], seg);
        prop_assert!(
            indices[1] >= seg && indices[1] < p.num_buckets,
            "i1={} must be in [{}, {})", indices[1], seg, p.num_buckets,
        );
    }

    // ─── pack_slot_cells / unpack_slot_cells ─────────────────────────────────

    #[test]
    fn prop_pack_unpack_roundtrip(
        pb        in 1u32..=32u32,
        fp_bits   in 5u32..=32u32,
        value_bits in 1u32..=256u32,
        fp_raw    in 0u32..=u32::MAX,
        seed      in 0u8..=255u8,
    ) {
        let store = make_store(8, 4, fp_bits, value_bits, pb);
        let params = store.params();

        // Build value_bytes matching value_bits length.
        let vbytes = value_bits.div_ceil(8) as usize;
        let mut value_bytes: Vec<u8> = (0..vbytes)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(seed))
            .collect();
        if value_bits % 8 != 0 {
            if let Some(last) = value_bytes.last_mut() {
                *last &= (1u8 << (value_bits % 8)) - 1;
            }
        }

        // Convert bytes → value cells using the same logic as `pack_value_bytes_to_cells`.
        let vcells = params.value_size_in_cells() as usize;
        let mut value_cells = vec![0u32; vcells];
        {
            let mut acc: u64 = 0u64;
            let mut bits_in_acc: u32 = 0;
            let mut byte_iter = value_bytes.iter().copied();
            for (i, c) in value_cells.iter_mut().enumerate() {
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

        // Mask fingerprint to fp_bits.
        let fp_mask = if fp_bits < 32 { (1u32 << fp_bits) - 1 } else { u32::MAX };
        let fp = (fp_raw & fp_mask).max(1); // keep non-zero

        let mut out = vec![0u32; params.cells_per_slot() as usize];
        pack_slot_cells(&params, fp, &value_cells, &mut out);

        let (got_fp, got_bytes) = unpack_slot_cells(&params, &out);
        prop_assert_eq!(got_fp, fp & fp_mask, "fingerprint mismatch");
        prop_assert_eq!(got_bytes, value_bytes, "value round-trip failed");
    }

    #[test]
    fn prop_pack_matches_fvt_write(
        pb        in 1u32..=32u32,
        fp_bits   in 5u32..=32u32,
        value_bits in 1u32..=256u32,
        seed      in 0u8..=255u8,
    ) {
        let mut store = make_store(8, 4, fp_bits, value_bits, pb);
        let params = store.params();

        let vbytes = value_bits.div_ceil(8) as usize;
        let mut v: Vec<u8> = (0..vbytes)
            .map(|i| (i as u8).wrapping_mul(13).wrapping_add(seed))
            .collect();
        if value_bits % 8 != 0 {
            if let Some(last) = v.last_mut() {
                *last &= (1u8 << (value_bits % 8)) - 1;
            }
        }

        if store.insert(b"test_key" as &[u8], &v).is_err() {
            return Ok(()); // table full; skip
        }

        // Find the occupied slot.
        let occupied: Vec<_> = store.iter_occupied_slots().collect();
        let slot = match occupied.first() {
            Some(s) => (s.bucket, s.slot, s.fingerprint),
            None => return Ok(()),
        };
        let (bucket, s, fp) = slot;

        // Build value_cells from the same value bytes.
        let vcells = params.value_size_in_cells() as usize;
        let mut value_cells = vec![0u32; vcells];
        {
            let mut acc: u64 = 0u64;
            let mut bits_in_acc: u32 = 0;
            let mut byte_iter = v.iter().copied();
            for (i, c) in value_cells.iter_mut().enumerate() {
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

        let mut packed = vec![0u32; params.cells_per_slot() as usize];
        pack_slot_cells(&params, fp, &value_cells, &mut packed);

        let ground_truth = &store.as_cells()[params.slot_cell_range(bucket, s)];
        prop_assert_eq!(packed.as_slice(), ground_truth,
            "pack_slot_cells output differs from FVT cell layout at pb={} fp_bits={} vb={}", pb, fp_bits, value_bits);
    }

    #[test]
    fn prop_unpack_matches_fvt_read(
        pb        in 1u32..=32u32,
        fp_bits   in 5u32..=32u32,
        value_bits in 1u32..=256u32,
        seed      in 0u8..=255u8,
    ) {
        let mut store = make_store(8, 4, fp_bits, value_bits, pb);
        let params = store.params();

        let vbytes = value_bits.div_ceil(8) as usize;
        let mut v: Vec<u8> = (0..vbytes)
            .map(|i| (i as u8).wrapping_mul(19).wrapping_add(seed))
            .collect();
        if value_bits % 8 != 0 {
            if let Some(last) = v.last_mut() {
                *last &= (1u8 << (value_bits % 8)) - 1;
            }
        }

        if store.insert(b"test_key2" as &[u8], &v).is_err() {
            return Ok(()); // table full; skip
        }

        let occupied: Vec<_> = store.iter_occupied_slots().collect();
        let slot = match occupied.first() {
            Some(s) => (s.bucket, s.slot, s.fingerprint),
            None => return Ok(()),
        };
        let (bucket, s, store_fp) = slot;

        let slot_cells = &store.as_cells()[params.slot_cell_range(bucket, s)];
        let (got_fp, got_bytes) = unpack_slot_cells(&params, slot_cells);

        prop_assert_eq!(got_fp, store_fp,
            "unpack_slot_cells fingerprint mismatch at pb={} fp_bits={} vb={}", pb, fp_bits, value_bits);
        prop_assert_eq!(got_bytes, v,
            "unpack_slot_cells value mismatch at pb={} fp_bits={} vb={}", pb, fp_bits, value_bits);
    }

    // ─── from_cells rejects wrong-size array ─────────────────────────────────

    #[test]
    fn prop_from_cells_rejects_wrong_size(
        extra in 1usize..100usize,
    ) {
        let p = CuckooParams {
            scheme_kind: SchemeKind::Segmented2ary,
            num_buckets: 8,
            bucket_size: 2,
            fingerprint_bits: 12,
            value_bits: 8,
            plaintext_bits: 8,
        };
        // Too long
        let cells = vec![0u32; p.size_in_cells() + extra];
        prop_assert!(
            Segmented2aryCuckooKVStore::from_cells(cells, p, 0).is_err(),
            "from_cells should reject oversized array",
        );
        // Too short
        if p.size_in_cells() > extra {
            let cells = vec![0u32; p.size_in_cells() - extra];
            prop_assert!(
                Segmented2aryCuckooKVStore::from_cells(cells, p, 0).is_err(),
                "from_cells should reject undersized array",
            );
        }
    }
}
