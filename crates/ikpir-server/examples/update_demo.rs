//! Demonstration of incremental insert / delete / update_value with timing.
//!
//! Run with:
//!   cargo run --release --example update_demo -p ikpir-server

use std::collections::HashMap;
use std::time::Instant;

use ikpir_client::Client;
use ikpir_common::params::{Arity, FilterParams};
use ikpir_server::Server;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn fresh_client(server: &Server) -> Client {
    let hints: Vec<_> = (0..server.params().degree() as usize)
        .map(|j| server.segment_h(j).clone())
        .collect();
    let hint_bytes = ikpir_common::serialization::batch_hint_to_bytes(&hints);
    let params_bytes = ikpir_common::serialization::params_to_bytes(server.params());
    Client::new(server.seed_mu(), &hint_bytes, &params_bytes).expect("fresh_client")
}

fn main() {
    let value_len = 32u32;
    let n = 128u32;
    let params = FilterParams::recommended(Arity::Segmented4, n, value_len).expect("params");

    let db: HashMap<Vec<u8>, Vec<u8>> = (0..n)
        .map(|i| {
            let key = format!("key-{i:04}").into_bytes();
            let value: Vec<u8> = (0..value_len as usize)
                .map(|j| (i as u8).wrapping_add(j as u8))
                .collect();
            (key, value)
        })
        .collect();

    let seed_mu = [0x42u8; 32];
    let t0 = Instant::now();
    let (mut server, _, _) = Server::setup(&params, &seed_mu, &db).expect("setup");
    eprintln!(
        "setup:         {:>8.2} ms  ({n} keys, {value_len} B values)",
        t0.elapsed().as_secs_f64() * 1e3
    );

    let mut rng = ChaCha20Rng::from_seed([0x11u8; 32]);

    // Insert a new key.
    let new_key = b"new-key-demo";
    let new_val = vec![0xABu8; value_len as usize];
    let t1 = Instant::now();
    server.insert(new_key, &new_val).expect("insert");
    eprintln!(
        "insert:        {:>8.2} µs",
        t1.elapsed().as_secs_f64() * 1e6
    );

    let client = fresh_client(&server);
    let (state, qb) = client.query(new_key, &mut rng).expect("query");
    let rb = server.respond(&qb).expect("respond");
    let t2 = Instant::now();
    let result = client.decrypt(&state, &rb).expect("decrypt");
    eprintln!(
        "query+decrypt: {:>8.2} µs",
        t2.elapsed().as_secs_f64() * 1e6
    );
    assert_eq!(result, Some(new_val.clone()), "inserted key not found");
    eprintln!("  → Ok(Some([0xAB; {}]))", value_len);

    // Update the key's value.
    let updated_val = vec![0xCDu8; value_len as usize];
    let t3 = Instant::now();
    server
        .update_value(new_key, &updated_val)
        .expect("update_value");
    eprintln!(
        "update_value:  {:>8.2} µs",
        t3.elapsed().as_secs_f64() * 1e6
    );

    let client = fresh_client(&server);
    let (state, qb) = client.query(new_key, &mut rng).expect("query");
    let rb = server.respond(&qb).expect("respond");
    let result = client.decrypt(&state, &rb).expect("decrypt");
    assert_eq!(result, Some(updated_val.clone()), "updated key mismatch");
    eprintln!("  → Ok(Some([0xCD; {}]))", value_len);

    // Delete the key.
    let t4 = Instant::now();
    server.delete(new_key).expect("delete");
    eprintln!(
        "delete:        {:>8.2} µs",
        t4.elapsed().as_secs_f64() * 1e6
    );

    let client = fresh_client(&server);
    let (state, qb) = client.query(new_key, &mut rng).expect("query");
    let rb = server.respond(&qb).expect("respond");
    let result = client.decrypt(&state, &rb).expect("decrypt");
    assert_eq!(result, None, "deleted key should return None");
    eprintln!("  → Ok(None)  (key absent after delete)");

    // Verify invariant on all segments.
    for j in 0..server.params().degree() as usize {
        let expected = server.segment_a(j).mul(server.segment_d(j)).expect("mul");
        assert_eq!(server.segment_h(j), &expected, "H_{j} invariant violated");
    }
    eprintln!(
        "H_j = A_j·D_j invariant holds for all {} segments",
        server.params().degree()
    );
}
