//! **Intent:** Measure initial server-setup wall-clock across DB sizes and
//! Segmented-SCF arities.
//!
//! **Output:** `results/setup_latency.csv`
//! Columns: `arity,num_items,value_len,num_buckets,mat_elem_bit_len,mean_ns,min_ns,max_ns,stddev_ns,hint_bytes,db_bytes`
//!
//! `hint_bytes` and `db_bytes` are recorded alongside timings so plots can
//! cross-correlate space cost with setup cost.

mod helpers;

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use ikpir_common::params::{Arity, FilterParams};
use ikpir_server::Server;

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
    trials: usize,
) -> (Vec<u128>, FilterParams, usize, usize) {
    let params = FilterParams::recommended(arity, num_items, value_len).expect("params");
    let db = build_db(num_items, value_len);
    let mut durations = Vec::with_capacity(trials);
    let mut last_hint_bytes = 0usize;
    let mut last_db_bytes = 0usize;
    for t in 0..trials {
        let seed_mu = [(t as u8).wrapping_add(1); 32];
        let start = Instant::now();
        let (server, hint_bytes, _p_bytes) = Server::setup(&params, &seed_mu, &db).expect("setup");
        let elapsed = start.elapsed().as_nanos();
        last_hint_bytes = hint_bytes.len();
        last_db_bytes = (server.params().num_buckets as usize)
            * (server.params().row_elems_bucket() as usize)
            * 4;
        durations.push(elapsed);
        drop(server);
    }
    (durations, params, last_hint_bytes, last_db_bytes)
}

fn main() {
    let mut w = helpers::csv_writer(
        "setup_latency.csv",
        "arity,num_items,value_len,num_buckets,mat_elem_bit_len,mean_ns,min_ns,max_ns,stddev_ns,hint_bytes,db_bytes",
    );

    // DB sizes: keep within `just repro-smoke` budget (~1 min). Paper-grade
    // runs can extend this via env var sweep.
    let num_items_set: Vec<u32> = std::env::var("IKPIR_BENCH_SIZES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<u32>().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![128, 512, 2048]);
    let value_len = 32u32;
    let trials = 3usize;

    for &num_items in &num_items_set {
        for arity in [Arity::Segmented2, Arity::Segmented3, Arity::Segmented4] {
            let (ds, params, hint_bytes, db_bytes) = bench_one(arity, num_items, value_len, trials);
            let stats = helpers::compute_stats(&ds.iter().map(|&x| x as f64).collect::<Vec<_>>());
            writeln!(
                w,
                "{},{},{},{},{},{:.0},{:.0},{:.0},{:.0},{},{}",
                arity,
                num_items,
                value_len,
                params.num_buckets,
                params.mat_elem_bit_len,
                stats.mean,
                stats.min,
                stats.max,
                stats.stddev,
                hint_bytes,
                db_bytes,
            )
            .unwrap();
            eprintln!(
                "{} n={} mean={:.1}ms hint={}B db={}B",
                arity,
                num_items,
                stats.mean / 1e6,
                hint_bytes,
                db_bytes,
            );
        }
    }
}
