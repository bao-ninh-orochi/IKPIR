//! Invariants that must hold under any random workload.
//!
//! These are looser than the round-trip correctness tests in `correctness.rs`
//! — they check structural properties (load factor, FPR bounds) rather than
//! exact per-item outcomes.

use segmented_cuckoo_filter::{CuckooError, Segmented2aryCuckooFilter, Standard2aryCuckooFilter};

const FINGERPRINT_BITS: u32 = 12;
const N: u32 = 1 << 14;
const B: u32 = 4;

#[test]
fn load_factor_never_exceeds_one() {
    let mut f = Standard2aryCuckooFilter::new(N, B, FINGERPRINT_BITS).unwrap();
    let mut i: u64 = 0;
    loop {
        match f.add(i.to_le_bytes()) {
            Ok(()) => i += 1,
            Err(CuckooError::TableFull) => break,
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
    let lf = f.load_factor();
    assert!(lf > 0.0 && lf <= 1.0, "load factor out of [0,1]: {lf}");
}

#[test]
fn segmented_load_factor_at_least_half() {
    let mut f = Segmented2aryCuckooFilter::new(N, B, FINGERPRINT_BITS).unwrap();
    let mut i: u64 = 0;
    loop {
        match f.add(i.to_le_bytes()) {
            Ok(()) => i += 1,
            Err(CuckooError::TableFull) => break,
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
    let lf = f.load_factor();
    assert!(
        lf > 0.5,
        "segmented 2-ary should easily exceed lf=0.5; got {lf}"
    );
}

#[test]
fn empty_filter_contains_nothing() {
    let f = Standard2aryCuckooFilter::new(N, B, FINGERPRINT_BITS).unwrap();
    for i in 0u64..1000 {
        assert!(
            !f.contain(i.to_le_bytes()),
            "empty filter claimed to contain {i}"
        );
    }
}

#[test]
fn false_positive_rate_within_bound() {
    // Theoretical bound for standard 2-ary: d*bucket_size / 2^fingerprint_bits.
    // With d=2, bucket_size=4, fingerprint_bits=12 → 8/4096 ≈ 0.2%.
    // We tolerate 3× the bound to account for finite-sample variance.
    let mut f = Standard2aryCuckooFilter::new(N, B, FINGERPRINT_BITS).unwrap();
    let mut inserted = 0u64;
    for i in 0u64..(N as u64 * B as u64 / 2) {
        if f.add(i.to_le_bytes()).is_ok() {
            inserted += 1;
        }
    }

    let trials = 100_000u64;
    let mut fps = 0u64;
    for i in (1u64 << 40)..((1u64 << 40) + trials) {
        // Items past the high watermark were never inserted.
        if f.contain(i.to_le_bytes()) {
            fps += 1;
        }
    }
    let observed = fps as f64 / trials as f64;
    let bound = 2.0 * B as f64 / (1u64 << FINGERPRINT_BITS) as f64;
    assert!(
        observed < 3.0 * bound,
        "observed FPR {observed:.5} exceeds 3× theoretical bound {bound:.5} (inserted {inserted})"
    );
}
