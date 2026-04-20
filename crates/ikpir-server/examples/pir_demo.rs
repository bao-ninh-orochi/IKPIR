//! Static-database demo across multiple arities.
//!
//! Builds a small `(keyword → value)` database, runs the full PIR round-trip
//! for each of the three Segmented SCF arities, and prints the recovered
//! values so the demo doubles as a correctness eyeball-test.
//!
//! Usage:
//!
//! ```bash
//! cargo run --release --example pir_demo -p ikpir-server
//! ```

use std::collections::HashMap;
use std::time::Instant;

use ikpir_client::Client;
use ikpir_common::params::{Arity, FilterParams};
use ikpir_server::Server;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const NUM_ITEMS: u32 = 256;
const VALUE_LEN: u32 = 32;

fn build_db(num: u32, value_len: u32) -> HashMap<Vec<u8>, Vec<u8>> {
    (0..num)
        .map(|i| {
            let key = format!("user:{i:04}").into_bytes();
            let value: Vec<u8> = format!("payload-for-user-{i:04}")
                .bytes()
                .chain(std::iter::repeat(0))
                .take(value_len as usize)
                .collect();
            (key, value)
        })
        .collect()
}

fn demo_arity(arity: Arity, db: &HashMap<Vec<u8>, Vec<u8>>) {
    println!("\n=== {arity:?} — {NUM_ITEMS} records × {VALUE_LEN}-byte values ===");
    let params = FilterParams::recommended(arity, NUM_ITEMS, VALUE_LEN).expect("params");
    println!(
        "  num_buckets={}  mat_elem_bit_len={}  row_elems={}",
        params.num_buckets, params.mat_elem_bit_len, params.row_elems
    );

    let seed_mu = [0xAAu8; 32];

    let t0 = Instant::now();
    let (server, hint_bytes, params_bytes) =
        Server::setup(&params, &seed_mu, db).expect("server setup");
    println!(
        "  Server setup: {:.1?}  |  hint bytes: {}  |  params bytes: {}",
        t0.elapsed(),
        hint_bytes.len(),
        params_bytes.len()
    );

    let t0 = Instant::now();
    let client = Client::new(&seed_mu, &hint_bytes, &params_bytes).expect("client setup");
    println!("  Client setup: {:.1?}", t0.elapsed());

    let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
    let probe_keys = [0u32, 1, 42, NUM_ITEMS / 2, NUM_ITEMS - 1];
    for k in probe_keys {
        let key = format!("user:{k:04}").into_bytes();
        let expected = db.get(&key).expect("key in db");

        let t_q = Instant::now();
        let (state, q_bytes) = client.query(&key, &mut rng).expect("query");
        let t_r = Instant::now();
        let resp_bytes = server.respond(&q_bytes).expect("respond");
        let t_d = Instant::now();
        let recovered = client
            .decrypt(&state, &resp_bytes)
            .expect("decrypt")
            .unwrap_or_else(|| panic!("key not found for k={k}"));
        let t_end = Instant::now();

        let match_note = if recovered == *expected {
            "match"
        } else {
            "MISMATCH"
        };
        println!(
            "  k={k:4}: query={:>7.2?}, respond={:>8.2?}, decrypt={:>7.2?}  | {} bytes in / {} bytes back | {match_note}",
            t_r.duration_since(t_q),
            t_d.duration_since(t_r),
            t_end.duration_since(t_d),
            q_bytes.len(),
            resp_bytes.len(),
        );
        assert_eq!(recovered, *expected, "round-trip mismatch at k={k}");
    }
}

fn main() {
    let db = build_db(NUM_ITEMS, VALUE_LEN);
    println!("Building a PIR demo over {NUM_ITEMS} × {VALUE_LEN}-byte records.");
    for arity in [Arity::Segmented2, Arity::Segmented3, Arity::Segmented4] {
        demo_arity(arity, &db);
    }
    println!("\nAll round-trips verified.");
}
