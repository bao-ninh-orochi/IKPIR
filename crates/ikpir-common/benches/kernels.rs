//! Kernel-level timing harness for the `perf/optimized` branch.
//!
//! # Purpose
//!
//! Times each backend hot path in isolation, at the public trait API,
//! so every optimization commit on this branch can cite before/after
//! numbers from the exact call sites the protocol exercises:
//!
//! | Label        | Trait call                     | Dominant kernel               |
//! |--------------|--------------------------------|-------------------------------|
//! | `setup`      | `server_setup` (one segment)   | `sample_a` + `compute_hint`   |
//! | `expand_a`   | `expand_hint_material`         | `sample_a` (ChaCha20)         |
//! | `answer`     | `server_answer`                | `qᵀ·D` matvec                 |
//! | `query_cold` | `client_query`, empty queue    | `A·s + e` matvec              |
//! | `decode_cold`| `client_decode`, no Phase C    | `sᵀ·H` matvec                 |
//! | `patch_entry`| `server_patch_hint` EntryLevel | sparse column patch           |
//! | `patch_row`  | `server_patch_hint` RowLevel   | dense rank-one row patch      |
//!
//! # Usage
//!
//! ```text
//! cargo bench -p ikpir-common --bench kernels            # default shape
//! cargo bench -p ikpir-common --bench kernels -- --heavy # + wide-value shape
//! ```
//!
//! Deliberately plain `Instant` timing (no criterion): the goal is a
//! stable, low-noise relative comparison between commits on one
//! machine, not statistical archival. Trial counts are fixed per op so
//! two runs of the same build are directly comparable; `min` is the
//! headline (least scheduler noise), `mean` is the sanity check.

use std::time::Instant;

use ikpir_common::{
    FrodoConfig, FrodoPirBackend, HintPatchMode, IncrementalPirBackend, IndexPirBackend,
    SimpleConfig, SimplePirBackend,
};

/// Per-segment shape to measure: `n_rows` DB rows of `row_width` cells.
#[derive(Clone, Copy)]
struct Shape {
    n_rows: u32,
    row_width: u32,
    plaintext_bits: u32,
}

/// Deterministic xorshift-filled DB so runs are comparable across
/// commits without a rand dependency in the timed region.
fn make_db(n_rows: u32, row_width: u32, plaintext_bits: u32) -> Vec<u32> {
    let mask = (1u32 << plaintext_bits) - 1;
    let mut state = 0x2545_F491u32;
    (0..(n_rows as usize) * (row_width as usize))
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state & mask
        })
        .collect()
}

/// Synthetic sparse mutation burst: `n_rows_touched` rows, each with
/// `cells_per_row` cell deltas — the shape `server_mutation` produces.
fn make_row_deltas(
    n_rows_touched: u32,
    cells_per_row: u16,
    shape: Shape,
) -> Vec<(u32, Vec<(u16, i64)>)> {
    (0..n_rows_touched)
        .map(|r| {
            let row = (r * 97) % shape.n_rows;
            let cells = (0..cells_per_row)
                .map(|c| (((c * 31) % shape.row_width as u16), i64::from(c) - 3))
                .collect();
            (row, cells)
        })
        .collect()
}

/// Time `iters` runs of `f`, reporting `(min, mean)` in seconds.
fn time_op<F: FnMut()>(iters: u32, mut f: F) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut total = 0.0;
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed().as_secs_f64();
        min = min.min(dt);
        total += dt;
    }
    (min, total / f64::from(iters))
}

fn report(backend: &str, shape: Shape, label: &str, iters: u32, min: f64, mean: f64) {
    println!(
        "{backend:<7} n={:<6} w={:<4} {label:<12} iters={iters:<4} min={:>12.3?}  mean={:>12.3?}",
        shape.n_rows,
        shape.row_width,
        std::time::Duration::from_secs_f64(min),
        std::time::Duration::from_secs_f64(mean),
    );
}

/// Run the full op suite for one backend at one shape.
fn run_backend<B: IndexPirBackend + IncrementalPirBackend>(
    backend: &str,
    config: &B::Config,
    shape: Shape,
) {
    let Shape {
        n_rows,
        row_width,
        plaintext_bits,
    } = shape;
    let db = make_db(n_rows, row_width, plaintext_bits);

    // setup: full per-segment preprocessing (sample_a + compute_hint).
    let (min, mean) = time_op(2, || {
        let out = B::server_setup(config, &db, n_rows, row_width, plaintext_bits);
        std::hint::black_box(&out);
    });
    report(backend, shape, "setup", 2, min, mean);

    let (params, material, mut hint) =
        B::server_setup(config, &db, n_rows, row_width, plaintext_bits);

    // expand_a: seed → A expansion alone (the ChaCha20 share of setup).
    let (min, mean) = time_op(3, || {
        let m = B::expand_hint_material(&params);
        std::hint::black_box(&m);
    });
    report(backend, shape, "expand_a", 3, min, mean);

    // query_cold: empty prepared queue → sample_slot (A·s + e).
    let mut state = B::client_setup(&params, &hint);
    let (min, mean) = time_op(20, || {
        let q = B::client_query(&mut state, 3);
        std::hint::black_box(&q);
    });
    report(backend, shape, "query_cold", 20, min, mean);

    // answer: the online qᵀ·D matvec.
    let query = B::client_query(&mut state, 5);
    let (min, mean) = time_op(50, || {
        let r = B::server_answer(&params, &db, n_rows, row_width, &query);
        std::hint::black_box(&r);
    });
    report(backend, shape, "answer", 50, min, mean);

    // decode_cold: no Phase C material → sᵀ·H matvec.
    let response = B::server_answer(&params, &db, n_rows, row_width, &query);
    let (min, mean) = time_op(200, || {
        let v = B::client_decode(&state, &response);
        std::hint::black_box(&v);
    });
    report(backend, shape, "decode_cold", 200, min, mean);

    // patch_entry / patch_row: 64-row × 8-cell burst, both realizations.
    // Repeated application drifts the hint values, but the cost is
    // data-independent so the timing stays valid.
    let deltas = make_row_deltas(64, 8, shape);
    let (min, mean) = time_op(50, || {
        B::server_patch_hint(
            &params,
            &material,
            &mut hint,
            &deltas,
            HintPatchMode::EntryLevel,
        );
    });
    report(backend, shape, "patch_entry", 50, min, mean);
    let (min, mean) = time_op(50, || {
        B::server_patch_hint(
            &params,
            &material,
            &mut hint,
            &deltas,
            HintPatchMode::RowLevel,
        );
    });
    report(backend, shape, "patch_row", 50, min, mean);
}

fn main() {
    let heavy = std::env::args().any(|a| a == "--heavy");

    // Default: the paper's mid-size per-segment shape (num_buckets 65536,
    // arity 4 → 16384 rows; bucket_size 4 × ~28 cells/slot → width 112).
    // --heavy adds the wide-value shape (1 KiB values → width ~832).
    let mut shapes = vec![Shape {
        n_rows: 16_384,
        row_width: 112,
        plaintext_bits: 10,
    }];
    if heavy {
        shapes.push(Shape {
            n_rows: 16_384,
            row_width: 832,
            plaintext_bits: 10,
        });
    }

    println!("kernels: per-op wall times (min over iters is the headline number)");
    for &shape in &shapes {
        run_backend::<FrodoPirBackend>("frodo", &FrodoConfig::default(), shape);
        run_backend::<SimplePirBackend>("simple", &SimpleConfig::default(), shape);
    }
}
