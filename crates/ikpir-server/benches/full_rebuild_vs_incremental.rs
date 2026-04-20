//! **Intent:** Break-even analysis — for which `t` (inserts) does a full
//! per-segment rebuild become cheaper than the incremental keyword insert?
//!
//! **Output:** `results/full_rebuild_vs_incremental.csv`
//! Columns: `arity,num_items,num_buckets,t,incremental_us,full_rebuild_us`

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

fn bench_one(arity: Arity, num_items: u32, t: usize, trials: usize) -> (Vec<f64>, Vec<f64>, u32) {
    let params =
        FilterParams::recommended(arity, num_items + t as u32 * 2, VALUE_LEN).expect("params");
    let db = build_db(num_items, VALUE_LEN);
    let seed_mu = [0x42u8; 32];
    let num_buckets = params.num_buckets;

    let mut inc_us = Vec::with_capacity(trials);
    let mut full_us = Vec::with_capacity(trials);

    for trial in 0..trials {
        let mut rng = ChaCha20Rng::from_seed([(trial as u8).wrapping_add(1); 32]);
        let (mut server, _, _) = Server::setup(&params, &seed_mu, &db).expect("setup");

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

        // Incremental: t keyword inserts.
        let t0 = Instant::now();
        for (key, val) in insert_keys.iter().zip(insert_vals.iter()) {
            server.insert(key, val).expect("insert");
        }
        inc_us.push(t0.elapsed().as_nanos() as f64 / 1_000.0);

        // Full rebuild: recompute A_j · D_j for each segment.
        let t1 = Instant::now();
        let k = server.params().degree() as usize;
        for j in 0..k {
            let _ = server.segment_a(j).mul(server.segment_d(j)).expect("mul");
        }
        full_us.push(t1.elapsed().as_nanos() as f64 / 1_000.0);
    }

    (inc_us, full_us, num_buckets)
}

fn main() {
    let mut w = helpers::csv_writer(
        "full_rebuild_vs_incremental.csv",
        "arity,num_items,num_buckets,t,incremental_us,full_rebuild_us",
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
                let (inc, full, num_buckets) = bench_one(arity, num_items, t, TRIALS);
                let inc_stats = helpers::compute_stats(&inc);
                let full_stats = helpers::compute_stats(&full);
                writeln!(
                    w,
                    "{},{},{},{},{:.3},{:.3}",
                    arity, num_items, num_buckets, t, inc_stats.mean, full_stats.mean,
                )
                .unwrap();
                let crossover = if full_stats.mean <= inc_stats.mean {
                    " <-- crossover"
                } else {
                    ""
                };
                eprintln!(
                    "{} N={} t={} incr={:.1}µs full={:.1}µs{}",
                    arity, num_items, t, inc_stats.mean, full_stats.mean, crossover
                );
            }
        }
    }
    println!("full_rebuild_vs_incremental: wrote results/full_rebuild_vs_incremental.csv");
}
