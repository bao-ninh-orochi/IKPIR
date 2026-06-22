//! Demonstrates the client-side PIR recovery path without any LWE or oblivious
//! network transfer.
//!
//! The real IKPIR client would replace the direct `cells[r]` read with a single
//! Index-PIR query. This example shows all the surrounding logic (params bundle,
//! candidate bucket derivation, cell-range mapping, fp+value decode) that stays
//! identical whether the read is plaintext or oblivious.
//!
//! Run with:
//! ```text
//! cargo run -p segmented-cuckoo --example pir_plaintext_recovery
//! ```

use segmented_cuckoo::{unpack_slot_cells, CuckooParams, Segmented2aryCuckooKVStore};

fn main() {
    // 1. Server builds and populates the store.
    let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    let dataset: &[(&[u8], &[u8])] = &[
        (b"alice", &[0x01]),
        (b"bob", &[0x02]),
        (b"charlie", &[0x03]),
        (b"dave", &[0x04]),
        (b"eve", &[0x05]),
    ];
    for &(k, v) in dataset {
        store.insert(k, v).expect("insert failed");
    }

    // 2. Server sends params + cell snapshot to the client.
    let params: CuckooParams = store.params();
    let cells: Vec<u32> = store.snapshot_cells();
    println!("Store geometry: {params:?}");
    println!("Total cells sent to client: {}", cells.len());

    // 3. Client recovers values without holding the store.
    for &(key, expected_value) in dataset {
        // a. Derive candidate buckets and fingerprint.
        let (fp, indices) = params.candidate_buckets(key);

        // b. For each candidate bucket and slot, check if fp matches.
        let mut recovered: Option<Vec<u8>> = None;
        'search: for &b in &indices[..params.arity()] {
            for s in 0..params.bucket_size {
                let r = params.slot_cell_range(b, s);
                // In a real PIR scheme this read would be oblivious (Index-PIR query).
                let slot_cells = &cells[r];

                let (decoded_fp, value_bytes) = unpack_slot_cells(&params, slot_cells);
                if decoded_fp == fp {
                    recovered = Some(value_bytes);
                    break 'search;
                }
            }
        }

        let recovered = recovered.expect("key not found in cell array");
        assert_eq!(
            recovered,
            expected_value,
            "value mismatch for key {:?}: got {recovered:?}, expected {expected_value:?}",
            std::str::from_utf8(key).unwrap_or("<binary>"),
        );
        println!(
            "  {:8} → recovered {:?}  ✓",
            std::str::from_utf8(key).unwrap_or("<binary>"),
            recovered,
        );
    }

    println!("\nAll values recovered correctly via plaintext PIR simulation.");
}
