//! Incremental-update demo: insert, delete, and update_value with per-segment
//! hint maintenance, plus a full-recompute speedup comparison.
//!
//! Usage:
//!
//! ```bash
//! cargo run --release --example pir_update_demo -p ikpir-server
//! ```

use std::collections::HashMap;
use std::time::Instant;

use ikpir_client::Client;
use ikpir_common::params::{Arity, FilterParams};
use ikpir_server::Server;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const NUM_ITEMS: u32 = 128;
const VALUE_LEN: u32 = 32;

fn build_db() -> HashMap<Vec<u8>, Vec<u8>> {
    (0..NUM_ITEMS)
        .map(|i| {
            let key = format!("key-{i:04}").into_bytes();
            let value: Vec<u8> = (0..VALUE_LEN as usize)
                .map(|j| ((i as usize).wrapping_add(j.wrapping_mul(13))) as u8)
                .collect();
            (key, value)
        })
        .collect()
}

fn fresh_client(server: &Server) -> Client {
    let hints: Vec<_> = (0..server.params().degree() as usize)
        .map(|j| server.segment_h(j).clone())
        .collect();
    let hint_bytes = ikpir_common::serialization::batch_hint_to_bytes(&hints);
    let params_bytes = ikpir_common::serialization::params_to_bytes(server.params());
    Client::new(server.seed_mu(), &hint_bytes, &params_bytes).expect("fresh_client")
}

fn main() {
    let params =
        FilterParams::recommended(Arity::Segmented4, NUM_ITEMS, VALUE_LEN).expect("params");
    let db = build_db();
    let seed_mu = [0x42u8; 32];

    println!("=== ikpir per-segment incremental update demo ===");
    println!(
        "N={NUM_ITEMS}, value_len={VALUE_LEN} B, arity=Segmented4\n\
         num_buckets={}, row_elems_bucket={}, mat_elem_bit_len={}",
        params.num_buckets,
        params.row_elems_bucket(),
        params.mat_elem_bit_len
    );

    let t0 = Instant::now();
    let (mut server, _, _) = Server::setup(&params, &seed_mu, &db).expect("Server::setup");
    println!("\nSetup: {:.2?}", t0.elapsed());

    let mut rng = ChaCha20Rng::from_seed([0x55u8; 32]);

    // Baseline query for key-0042.
    let probe_key = b"key-0042" as &[u8];
    let expected = db.get(probe_key).expect("probe key in db");
    let client = fresh_client(&server);
    let (state, qb) = client.query(probe_key, &mut rng).expect("query");
    let rb = server.respond(&qb).expect("respond");
    let recovered = client
        .decrypt(&state, &rb)
        .expect("decrypt")
        .unwrap_or_else(|| panic!("key-0042 not found after setup"));
    assert_eq!(&recovered, expected, "baseline query mismatch");
    println!("[OK] Baseline query for key-0042 verified.");

    // Insert a new key.
    let new_key = b"key-new-0001";
    let new_val = vec![0xABu8; VALUE_LEN as usize];
    let t_ins = Instant::now();
    server.insert(new_key, &new_val).expect("insert");
    println!("insert:        {:.2?}", t_ins.elapsed());

    let client = fresh_client(&server);
    let (state, qb) = client.query(new_key, &mut rng).expect("query");
    let rb = server.respond(&qb).expect("respond");
    let result = client.decrypt(&state, &rb).expect("decrypt");
    assert_eq!(
        result,
        Some(new_val.clone()),
        "new key not found after insert"
    );
    println!("[OK] Inserted key found.");

    // Update the new key's value.
    let upd_val = vec![0xCDu8; VALUE_LEN as usize];
    let t_upd = Instant::now();
    server
        .update_value(new_key, &upd_val)
        .expect("update_value");
    println!("update_value:  {:.2?}", t_upd.elapsed());

    let client = fresh_client(&server);
    let (state, qb) = client.query(new_key, &mut rng).expect("query");
    let rb = server.respond(&qb).expect("respond");
    let result = client.decrypt(&state, &rb).expect("decrypt");
    assert_eq!(result, Some(upd_val.clone()), "updated value mismatch");
    println!("[OK] Updated key found with new value.");

    // Delete the new key.
    let t_del = Instant::now();
    server.delete(new_key).expect("delete");
    println!("delete:        {:.2?}", t_del.elapsed());

    let client = fresh_client(&server);
    let (state, qb) = client.query(new_key, &mut rng).expect("query");
    let rb = server.respond(&qb).expect("respond");
    let result = client.decrypt(&state, &rb).expect("decrypt");
    assert!(result.is_none(), "deleted key should be absent");
    println!("[OK] Deleted key returns None.");

    // Measure incremental update vs full recompute.
    let t_full = Instant::now();
    let mut match_count = 0usize;
    for j in 0..server.params().degree() as usize {
        let expected_h = server.segment_a(j).mul(server.segment_d(j)).expect("mul");
        if server.segment_h(j) == &expected_h {
            match_count += 1;
        }
    }
    let full_elapsed = t_full.elapsed();
    assert_eq!(
        match_count,
        server.params().degree() as usize,
        "invariant violated"
    );
    println!("\nFull recompute H_j = A_j·D_j check: {:.2?}", full_elapsed);
    println!(
        "H_j invariant holds for all {} segments.",
        server.params().degree()
    );
    println!("\nDemo complete.");
}
