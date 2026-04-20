//! Mixed-workload end-to-end tests for incremental update operations.
//!
//! These tests simulate realistic usage patterns: repeated insert / delete /
//! update_value / query sequences, tracking expected state in a shadow map
//! and verifying that every query result matches.
//!
//! Note: these tests use a fresh server hint for each query sub-sequence
//! (via a re-setup with the current state) to decouple the hint-staleness
//! issue (tracked as a future phase) from the correctness of the per-segment
//! matrix update logic.

use std::collections::HashMap;

use ikpir_client::Client;
use ikpir_common::params::{Arity, FilterParams};
use ikpir_server::Server;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn build_db(num: u32, value_len: u32) -> HashMap<Vec<u8>, Vec<u8>> {
    (0..num)
        .map(|i| {
            let key = format!("key-{i:04}").into_bytes();
            let value: Vec<u8> = (0..value_len as usize)
                .map(|j| ((i as usize * 7).wrapping_add(j)) as u8)
                .collect();
            (key, value)
        })
        .collect()
}

/// Rebuild client from the server's current segments so the hint is fresh.
fn fresh_client(server: &Server) -> Client {
    let seed_mu = server.seed_mu();
    let hints: Vec<_> = (0..server.params().degree() as usize)
        .map(|j| server.segment_h(j).clone())
        .collect();
    let hint_bytes = ikpir_common::serialization::batch_hint_to_bytes(&hints);
    let params_bytes = ikpir_common::serialization::params_to_bytes(server.params());
    Client::new(seed_mu, &hint_bytes, &params_bytes).expect("fresh_client")
}

fn run_mixed_workload(arity: Arity) {
    let value_len = 8u32;
    let initial_n = 32u32;
    let params = FilterParams::recommended(arity, initial_n * 2, value_len).expect("params");

    // Build an initial database with `initial_n` entries.
    let db = build_db(initial_n, value_len);
    let seed_mu = [0xCAu8; 32];
    let (mut server, _, _) = Server::setup(&params, &seed_mu, &db).expect("setup");

    // Shadow map tracks the expected server state.
    let mut shadow: HashMap<Vec<u8>, Vec<u8>> = db.clone();
    let mut rng = ChaCha20Rng::from_seed([0xFEu8; 32]);
    let mut query_rng = ChaCha20Rng::from_seed([0xEFu8; 32]);

    // 100 mixed operations.
    let mut next_key_id = initial_n;
    for op in 0..100u32 {
        let op_class = op % 4;
        match op_class {
            0 => {
                // Insert a new key.
                let key = format!("key-{next_key_id:04}").into_bytes();
                let value: Vec<u8> = (0..value_len as usize)
                    .map(|j| ((next_key_id as usize).wrapping_add(j)) as u8)
                    .collect();
                if server.insert(&key, &value).is_ok() {
                    shadow.insert(key, value);
                    next_key_id += 1;
                }
            }
            1 => {
                // Delete a random existing key.
                if let Some((k, _)) = shadow.iter().next().map(|(k, v)| (k.clone(), v.clone())) {
                    if server.delete(&k).is_ok() {
                        shadow.remove(&k);
                    }
                }
            }
            2 => {
                // Update a random existing key.
                if let Some((k, _)) = shadow.iter().next().map(|(k, v)| (k.clone(), v.clone())) {
                    let new_val: Vec<u8> = (0..value_len as usize)
                        .map(|j| ((op as usize).wrapping_add(j)) as u8)
                        .collect();
                    if server.update_value(&k, &new_val).is_ok() {
                        shadow.insert(k, new_val);
                    }
                }
            }
            _ => {
                // Query: rebuild client with fresh hint, then verify a few keys.
                let client = fresh_client(&server);
                let keys: Vec<_> = shadow.keys().take(4).cloned().collect();
                for key in keys {
                    let expected = shadow.get(&key).cloned();
                    let (state, query_bytes) = client.query(&key, &mut query_rng).expect("query");
                    let response_bytes = server.respond(&query_bytes).expect("respond");
                    let result = client.decrypt(&state, &response_bytes).expect("decrypt");
                    assert_eq!(
                        result,
                        expected,
                        "op={op} arity={arity} key={:?}: expected {expected:?}, got {result:?}",
                        String::from_utf8_lossy(&key)
                    );
                }
                // Also verify a key that should be absent.
                let absent_key = b"never-in-db-0000";
                if !shadow.contains_key(absent_key.as_slice()) {
                    let (state, qb) = client.query(absent_key, &mut query_rng).expect("query");
                    let rb = server.respond(&qb).expect("respond");
                    let result = client.decrypt(&state, &rb).expect("decrypt");
                    assert!(result.is_none(), "absent key returned Some(_) at op={op}");
                }
            }
        }

        // Check H_j = A_j · D_j invariant after every operation.
        for j in 0..server.params().degree() as usize {
            let expected = server
                .segment_a(j)
                .mul(server.segment_d(j))
                .expect("recompute");
            assert_eq!(
                server.segment_h(j),
                &expected,
                "invariant broken at op={op} segment={j} arity={arity}"
            );
        }

        let _ = &mut rng; // keep rng alive
    }
}

#[test]
fn mixed_workload_segmented4() {
    run_mixed_workload(Arity::Segmented4);
}

#[test]
fn mixed_workload_segmented2() {
    run_mixed_workload(Arity::Segmented2);
}

#[test]
fn mixed_workload_segmented3() {
    run_mixed_workload(Arity::Segmented3);
}
