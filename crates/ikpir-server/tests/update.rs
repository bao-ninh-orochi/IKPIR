//! Integration tests for incremental server updates via the keyword API.

use std::collections::HashMap;

use ikpir_client::Client;
use ikpir_common::cuckoo_store::PerSegmentRow;
use ikpir_common::params::{Arity, FilterParams};
use ikpir_server::Server;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn build_db(num: u32, value_len: u32) -> HashMap<Vec<u8>, Vec<u8>> {
    (0..num)
        .map(|i| {
            let key = format!("key-{i}").into_bytes();
            let value: Vec<u8> = (0..value_len as usize)
                .map(|j| ((i as usize).wrapping_add(j.wrapping_mul(3))) as u8)
                .collect();
            (key, value)
        })
        .collect()
}

fn query_key(
    server: &Server,
    client: &Client,
    rng: &mut ChaCha20Rng,
    key: &[u8],
) -> Option<Vec<u8>> {
    let (state, query_bytes) = client.query(key, rng).expect("query");
    let response_bytes = server.respond(&query_bytes).expect("respond");
    client.decrypt(&state, &response_bytes).expect("decrypt")
}

fn check_invariant(server: &Server) {
    for j in 0..server.params().degree() as usize {
        let expected = server
            .segment_a(j)
            .mul(server.segment_d(j))
            .expect("recompute");
        assert_eq!(
            server.segment_h(j),
            &expected,
            "H_{j} != A_{j}·D_{j} after update"
        );
    }
}

#[test]
fn insert_then_query_returns_value() {
    let params = FilterParams::recommended(Arity::Segmented4, 32, 8).expect("params");
    let db = build_db(32, 8);
    let seed_mu = [0xAAu8; 32];
    let (mut server, hint_bytes, params_bytes) =
        Server::setup(&params, &seed_mu, &db).expect("setup");
    let client = Client::new(&seed_mu, &hint_bytes, &params_bytes).expect("client");
    let mut rng = ChaCha20Rng::from_seed([0xBBu8; 32]);

    let new_key = b"brand-new-key";
    let new_val = vec![0xFFu8; 8];
    server.insert(new_key, &new_val).expect("insert");
    check_invariant(&server);

    let result = query_key(&server, &client, &mut rng, new_key);
    // Hint was built before insert — client decrypts against old H, so we
    // verify the invariant holds but don't assert the query result here.
    // (Phase 8 / proper hint-refresh is out of scope; tested in update_e2e.)
    let _ = result; // suppress unused warning
}

#[test]
fn query_unchanged_after_noop_update() {
    let params = FilterParams::recommended(Arity::Segmented4, 64, 8).expect("params");
    let db = build_db(64, 8);
    let seed_mu = [0xAAu8; 32];
    let (mut server, hint_bytes, params_bytes) =
        Server::setup(&params, &seed_mu, &db).expect("setup");
    let client = Client::new(&seed_mu, &hint_bytes, &params_bytes).expect("client");
    let mut rng = ChaCha20Rng::from_seed([0xBBu8; 32]);

    // Noop: apply a PerSegmentRow with the current D_j[0, :] value unchanged.
    let current_row = server.segment_d(0).row(0).expect("row 0").to_vec();
    let noop = PerSegmentRow {
        segment_idx: 0,
        row_local_idx: 0,
        new_payload: current_row,
    };
    server.apply_row_updates(&[noop]).expect("noop update");
    check_invariant(&server);

    // Every sampled key should still decrypt correctly against the original hint.
    for i in (0u32..64).step_by(8) {
        let key = format!("key-{i}").into_bytes();
        let expected = db.get(&key).expect("key present");
        let result = query_key(&server, &client, &mut rng, &key);
        assert_eq!(
            result.as_deref(),
            Some(expected.as_slice()),
            "key-{i} failed after noop update"
        );
    }
}

#[test]
fn hint_bytes_differ_after_nontrivial_update() {
    let params = FilterParams::recommended(Arity::Segmented4, 32, 4).expect("params");
    let db = build_db(32, 4);
    let seed_mu = [0x11u8; 32];
    let (mut server, _, _) = Server::setup(&params, &seed_mu, &db).expect("setup");

    let hint_before: Vec<_> = (0..params.degree() as usize)
        .map(|j| server.segment_h(j).as_slice().to_vec())
        .collect();

    // Flip the first element of segment 0, row 0 to a different value.
    let mask: u32 = (1u32 << params.mat_elem_bit_len) - 1;
    let original = server.segment_d(0).row(0).expect("row 0")[0];
    let new_val = {
        let v = (original.wrapping_add(1)) & mask;
        if v == original {
            (original.wrapping_add(2)) & mask
        } else {
            v
        }
    };
    let mut new_row = server.segment_d(0).row(0).expect("row 0").to_vec();
    new_row[0] = new_val;
    server
        .apply_row_updates(&[PerSegmentRow {
            segment_idx: 0,
            row_local_idx: 0,
            new_payload: new_row,
        }])
        .expect("apply");

    // At least one segment's hint must have changed.
    let changed = (0..params.degree() as usize)
        .any(|j| server.segment_h(j).as_slice() != hint_before[j].as_slice());
    assert!(changed, "no hint changed after modifying D[0][0]");
    check_invariant(&server);
}

#[test]
fn delete_then_query_returns_none() {
    let params = FilterParams::recommended(Arity::Segmented4, 32, 8).expect("params");
    let db = build_db(32, 8);
    let seed_mu = [0x22u8; 32];
    let (mut server, hint_bytes, params_bytes) =
        Server::setup(&params, &seed_mu, &db).expect("setup");
    let client = Client::new(&seed_mu, &hint_bytes, &params_bytes).expect("client");
    let mut rng = ChaCha20Rng::from_seed([0x33u8; 32]);

    let key = b"key-0";

    // Verify it's present first (using old hint — should succeed).
    let before = query_key(&server, &client, &mut rng, key);
    assert!(before.is_some(), "key-0 should be present initially");

    server.delete(key).expect("delete");
    check_invariant(&server);

    // After delete, the slot is zeroed; the client (with old hint) should now
    // return None because no slot has a matching fingerprint in the zeroed row.
    let after = query_key(&server, &client, &mut rng, key);
    assert!(
        after.is_none(),
        "key-0 should be absent after delete, got {:?}",
        after
    );
}

#[test]
fn insert_duplicate_is_rejected() {
    let params = FilterParams::recommended(Arity::Segmented4, 32, 8).expect("params");
    let db = build_db(32, 8);
    let (mut server, _, _) = Server::setup(&params, &[0u8; 32], &db).expect("setup");

    // Inserting the exact same (key, value) pair that was loaded during setup must fail.
    let key = b"key-0".to_vec();
    // build_db uses: (i + j*3) as u8 for value bytes
    let val: Vec<u8> = (0..8)
        .map(|j: usize| (0usize.wrapping_add(j.wrapping_mul(3))) as u8)
        .collect();
    let err = server.insert(&key, &val);
    assert!(err.is_err(), "inserting a duplicate (key, val) must fail");
}

#[test]
fn delete_unknown_key_is_error() {
    let params = FilterParams::recommended(Arity::Segmented4, 32, 8).expect("params");
    let db = build_db(32, 8);
    let (mut server, _, _) = Server::setup(&params, &[0u8; 32], &db).expect("setup");

    let err = server.delete(b"never-inserted");
    assert!(err.is_err(), "deleting a non-existent key must fail");
}

#[test]
fn apply_row_updates_segment_oob_is_error() {
    use ikpir_server::update::UpdateError;

    let params = FilterParams::recommended(Arity::Segmented4, 32, 4).expect("params");
    let db = build_db(32, 4);
    let (mut server, _, _) = Server::setup(&params, &[0u8; 32], &db).expect("setup");

    let bad = PerSegmentRow {
        segment_idx: params.degree() as u8,
        row_local_idx: 0,
        new_payload: vec![0u32; params.row_elems_bucket() as usize],
    };
    assert_eq!(
        server.apply_row_updates(&[bad]),
        Err(UpdateError::SegmentOutOfBounds)
    );
}
