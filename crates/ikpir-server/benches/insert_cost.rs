//! **Intent:** Per-insert wall-clock across batch sizes and DB sizes.
//!
//! **Output:** `results/insert_cost.csv`
//! Columns: `arity,num_items,num_buckets,t,mean_us,min_us,max_us,stddev_us`

mod helpers;

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use ikpir_common::params::{Arity, FilterParams};
use ikpir_server::Server;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const T_VALUES: &[usize] = &[1, 4, 16, 64, 256];
const VALUE_LEN: u32 = 32;
const TRIALS: usize = 5;

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

fn bench_one(arity: Arity, num_items: u32, t: usize, trials: usize) -> (Vec<f64>, u32) {
    let params =
        FilterParams::recommended(arity, num_items + t as u32 * 2, VALUE_LEN).expect("params");
    let db = build_db(num_items, VALUE_LEN);
    let seed_mu = [0x42u8; 32];
    let num_buckets = params.num_buckets;

    let mut us = Vec::with_capacity(trials);

    for trial in 0..trials {
        let mut rng = ChaCha20Rng::from_seed([(trial as u8).wrapping_add(1); 32]);
        let (mut server, _, _) = Server::setup(&params, &seed_mu, &db).expect("setup");

        // Build t distinct insert keys not already in db.
        let insert_keys: Vec<Vec<u8>> = (0..t)
            .map(|k| format!("ins-{trial:02}-{k:04}").into_bytes())
            .collect();
        let insert_vals: Vec<Vec<u8>> = (0..t)
            .map(|k| {
                (0..VALUE_LEN as usize)
                    .map(|j| {
                        (rand::RngCore::next_u32(&mut rng) as usize)
                            .wrapping_add(j)
                            .wrapping_add(k) as u8
                    })
                    .collect()
            })
            .collect();

        let start = Instant::now();
        for (key, val) in insert_keys.iter().zip(insert_vals.iter()) {
            server.insert(key, val).expect("insert");
        }
        us.push(start.elapsed().as_nanos() as f64 / 1_000.0);
    }

    (us, num_buckets)
}

fn main() {
    let mut w = helpers::csv_writer(
        "insert_cost.csv",
        "arity,num_items,num_buckets,t,mean_us,min_us,max_us,stddev_us",
    );

    let num_items_set: Vec<u32> = std::env::var("IKPIR_BENCH_SIZES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse::<u32>().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![128, 512, 2048]);

    for &num_items in &num_items_set {
        for arity in [Arity::Segmented2, Arity::Segmented3, Arity::Segmented4] {
            for &t in T_VALUES {
                let (us, num_buckets) = bench_one(arity, num_items, t, TRIALS);
                let stats = helpers::compute_stats(&us);
                writeln!(
                    w,
                    "{},{},{},{},{:.3},{:.3},{:.3},{:.3}",
                    arity,
                    num_items,
                    num_buckets,
                    t,
                    stats.mean,
                    stats.min,
                    stats.max,
                    stats.stddev,
                )
                .unwrap();
                eprintln!("{} N={} t={} mean={:.1}µs", arity, num_items, t, stats.mean);
            }
        }
    }
    println!("insert_cost: wrote results/insert_cost.csv");
}
