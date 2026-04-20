//! **Intent:** Per-query end-to-end wall-clock (client query + server
//! respond + client decrypt) across DB sizes and Segmented-SCF arities.
//!
//! **Output:** `results/query_latency.csv`
//! Columns: `arity,num_items,value_len,num_buckets,stage,mean_ns,min_ns,max_ns,stddev_ns`
//!
//! `stage ∈ {query, respond, decrypt, total}` so downstream plots can stack
//! the three phases.

mod helpers;

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use ikpir_client::Client;
use ikpir_common::params::{Arity, FilterParams};
use ikpir_server::Server;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn build_db(num: u32, value_len: u32) -> HashMap<Vec<u8>, Vec<u8>> {
    (0..num)
        .map(|i| {
            let key = format!("k-{i:08}").into_bytes();
            let value: Vec<u8> = (0..value_len as usize)
                .map(|j| ((i as usize).wrapping_mul(37).wrapping_add(j) & 0xFF) as u8)
                .collect();
            (key, value)
        })
        .collect()
}

fn bench_one(
    arity: Arity,
    num_items: u32,
    value_len: u32,
    queries_per_trial: usize,
) -> (Vec<u128>, Vec<u128>, Vec<u128>, FilterParams) {
    let params = FilterParams::recommended(arity, num_items, value_len).expect("params");
    let db = build_db(num_items, value_len);
    let seed_mu = [0xBBu8; 32];
    let (server, hint_bytes, params_bytes) = Server::setup(&params, &seed_mu, &db).expect("setup");
    let client = Client::new(&seed_mu, &hint_bytes, &params_bytes).expect("client setup");
    let mut rng = ChaCha20Rng::from_seed([0x55u8; 32]);

    let mut qs = Vec::with_capacity(queries_per_trial);
    let mut rs = Vec::with_capacity(queries_per_trial);
    let mut ds = Vec::with_capacity(queries_per_trial);

    // Warm-up.
    for i in 0..4 {
        let key = format!("k-{:08}", i % num_items).into_bytes();
        let (state, qb) = client.query(&key, &mut rng).unwrap();
        let rb = server.respond(&qb).unwrap();
        let _ = client.decrypt(&state, &rb).unwrap();
    }

    for trial in 0..queries_per_trial {
        let pick = (trial as u32).wrapping_mul(2654435761) % num_items;
        let key = format!("k-{pick:08}").into_bytes();

        let t0 = Instant::now();
        let (state, qb) = client.query(&key, &mut rng).unwrap();
        let t1 = Instant::now();
        let rb = server.respond(&qb).unwrap();
        let t2 = Instant::now();
        let recovered = client.decrypt(&state, &rb).unwrap().unwrap();
        let t3 = Instant::now();

        assert_eq!(recovered, *db.get(&key).unwrap());
        qs.push(t1.duration_since(t0).as_nanos());
        rs.push(t2.duration_since(t1).as_nanos());
        ds.push(t3.duration_since(t2).as_nanos());
    }

    (qs, rs, ds, params)
}

fn write_row<W: Write>(
    w: &mut W,
    arity: Arity,
    num_items: u32,
    value_len: u32,
    num_buckets: u32,
    stage: &str,
    samples: &[u128],
) {
    let ns: Vec<f64> = samples.iter().map(|&x| x as f64).collect();
    let s = helpers::compute_stats(&ns);
    writeln!(
        w,
        "{arity},{num_items},{value_len},{num_buckets},{stage},{:.0},{:.0},{:.0},{:.0}",
        s.mean, s.min, s.max, s.stddev
    )
    .unwrap();
}

fn main() {
    let mut w = helpers::csv_writer(
        "query_latency.csv",
        "arity,num_items,value_len,num_buckets,stage,mean_ns,min_ns,max_ns,stddev_ns",
    );

    let num_items_set: Vec<u32> = std::env::var("IKPIR_BENCH_SIZES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<u32>().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![128, 512, 2048]);
    let value_len = 32u32;
    let queries_per_trial = 32usize;

    for &num_items in &num_items_set {
        for arity in [Arity::Segmented2, Arity::Segmented3, Arity::Segmented4] {
            let (qs, rs, ds, params) = bench_one(arity, num_items, value_len, queries_per_trial);
            let totals: Vec<u128> = qs
                .iter()
                .zip(rs.iter())
                .zip(ds.iter())
                .map(|((&a, &b), &c)| a + b + c)
                .collect();
            write_row(
                &mut w,
                arity,
                num_items,
                value_len,
                params.num_buckets,
                "query",
                &qs,
            );
            write_row(
                &mut w,
                arity,
                num_items,
                value_len,
                params.num_buckets,
                "respond",
                &rs,
            );
            write_row(
                &mut w,
                arity,
                num_items,
                value_len,
                params.num_buckets,
                "decrypt",
                &ds,
            );
            write_row(
                &mut w,
                arity,
                num_items,
                value_len,
                params.num_buckets,
                "total",
                &totals,
            );
            let t_mean = totals.iter().sum::<u128>() as f64 / totals.len() as f64;
            eprintln!("{arity} n={num_items} total_mean={:.1}µs", t_mean / 1e3);
        }
    }
}
