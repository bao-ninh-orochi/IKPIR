//! End-to-end integration tests: `Server::setup → Client::new → query →
//! respond → decrypt → verify`.
//!
//! Exercises all three segmented SCF arities at multiple database sizes.

use std::collections::HashMap;

use ikpir_client::Client;
use ikpir_common::params::{Arity, FilterParams};
use ikpir_server::Server;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn build_db(num: u32, value_len: u32) -> HashMap<Vec<u8>, Vec<u8>> {
    (0..num)
        .map(|i| {
            let key = format!("key-{i}").into_bytes();
            let mut value = vec![0u8; value_len as usize];
            for (j, b) in value.iter_mut().enumerate() {
                *b = ((i.wrapping_add(j as u32)) & 0xFF) as u8;
            }
            (key, value)
        })
        .collect()
}

fn run_roundtrip(arity: Arity, num: u32, value_len: u32) {
    let params = FilterParams::recommended(arity, num, value_len).expect("params");
    let db = build_db(num, value_len);

    let seed_mu = [17u8; 32];
    let (server, hint_bytes, params_bytes) = Server::setup(&params, &seed_mu, &db).expect("setup");

    let client = Client::new(&seed_mu, &hint_bytes, &params_bytes).expect("client setup");

    let mut rng = ChaCha20Rng::from_seed([42u8; 32]);

    let step = (num / 8).max(1);
    let sample_keys: Vec<u32> = (0..num).step_by(step as usize).collect();
    for k in sample_keys {
        let key = format!("key-{k}").into_bytes();
        let expected = db.get(&key).expect("key in db");

        let (state, query_bytes) = client.query(&key, &mut rng).expect("query");
        let response_bytes = server.respond(&query_bytes).expect("respond");
        let recovered = client
            .decrypt(&state, &response_bytes)
            .expect("decrypt")
            .unwrap_or_else(|| {
                panic!(
                    "key {:?} not found under arity {arity}",
                    String::from_utf8_lossy(&key)
                )
            });

        assert_eq!(
            recovered,
            *expected,
            "round trip failed for {:?} under {arity}",
            String::from_utf8_lossy(&key)
        );
    }
}

#[test]
fn segmented2_end_to_end() {
    run_roundtrip(Arity::Segmented2, 128, 32);
}

#[test]
fn segmented3_end_to_end() {
    run_roundtrip(Arity::Segmented3, 128, 32);
}

#[test]
fn segmented4_end_to_end() {
    run_roundtrip(Arity::Segmented4, 128, 32);
}

#[test]
fn segmented4_end_to_end_larger_db() {
    run_roundtrip(Arity::Segmented4, 512, 32);
}

#[test]
fn segmented4_short_values() {
    run_roundtrip(Arity::Segmented4, 64, 4);
}

#[test]
fn decrypt_returns_none_for_never_inserted_key() {
    let params = FilterParams::recommended(Arity::Segmented4, 64, 8).expect("params");
    let db = build_db(64, 8);
    let seed_mu = [0xABu8; 32];
    let (server, hint_bytes, params_bytes) = Server::setup(&params, &seed_mu, &db).expect("setup");
    let client = Client::new(&seed_mu, &hint_bytes, &params_bytes).expect("client");
    let mut rng = ChaCha20Rng::from_seed([0xCDu8; 32]);

    // This key was never inserted.
    let (state, query_bytes) = client
        .query(b"never-inserted-key", &mut rng)
        .expect("query");
    let response_bytes = server.respond(&query_bytes).expect("respond");
    let result = client.decrypt(&state, &response_bytes).expect("decrypt");

    // The vast majority of the time this is None; a false positive is possible
    // (~0.4 %) but in practice never fires with a deterministic test key.
    assert!(
        result.is_none(),
        "expected None for never-inserted key, got {:?}",
        result
    );
}
