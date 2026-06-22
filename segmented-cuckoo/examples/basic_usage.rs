//! Minimal usage demo for every `CuckooFilter` variant.
//!
//! # Purpose
//!
//! Sanity-check / first-touch example: shows `insert` / `contain` /
//! `delete` on all six filter aliases (Standard / Segmented × 2/3/4-ary)
//! using the same input set so the reader can compare outputs side by
//! side.
//!
//! # Run
//!
//! ```text
//! cargo run -p segmented-cuckoo --example basic_usage
//! ```
//!
//! # What to look for
//!
//! - The same keys yield the same `contain` answers across all six
//!   variants (false-positive aside).
//! - `delete` is exact for keys that were genuinely inserted.

use segmented_cuckoo::{
    Segmented2aryCuckooFilter, Segmented3aryCuckooFilter, Segmented4aryCuckooFilter,
    Standard2aryCuckooFilter, Standard3aryCuckooFilter, Standard4aryCuckooFilter,
};

macro_rules! demo {
    ($label:expr, $filter:expr) => {{
        let mut f = $filter;
        println!("=== {} ===", $label);
        f.insert("apple").unwrap();
        f.insert("banana").unwrap();
        f.insert("cherry").unwrap();
        f.insert(42u64.to_le_bytes()).unwrap();
        println!("contains 'apple':  {}", f.contain("apple"));
        println!("contains 'banana': {}", f.contain("banana"));
        println!("contains 'grape':  {}", f.contain("grape"));
        println!("contains 42:       {}", f.contain(42u64.to_le_bytes()));
        println!("Deleting 'banana'...");
        let _ = f.delete("banana");
        println!("contains 'banana': {}", f.contain("banana"));
        println!(
            "items: {}  bytes: {}  load: {:.4}",
            f.num_items(),
            f.size_in_bytes(),
            f.load_factor()
        );
        println!();
    }};
}

fn main() {
    // ==== USING `new` CONSTRUCTOR =====================================================

    // ── 2-ary ─────────────────────────────────────────────────────────────────
    // Segmented 2-ary: num_buckets must be a power of 2 and ≥ 2.
    demo!(
        "Segmented 2-ary Cuckoo Filter",
        Segmented2aryCuckooFilter::new(128, 4, 12).unwrap()
    );
    // Standard 2-ary: num_buckets must be a power of 2 and ≥ 2.
    demo!(
        "Standard 2-ary Cuckoo Filter",
        Standard2aryCuckooFilter::new(128, 4, 12).unwrap()
    );

    // ── 3-ary ─────────────────────────────────────────────────────────────────
    // Segmented 3-ary: num_buckets must be 3·2^m (num_buckets/3 is a power of 2).
    demo!(
        "Segmented 3-ary Cuckoo Filter",
        Segmented3aryCuckooFilter::from_num_items(96, 4, 12).unwrap() // 3 * 2^5 = 96 buckets
    );
    // Standard 3-ary: num_buckets must be a power of 3 and ≥ 1.
    demo!(
        "Standard 3-ary Cuckoo Filter",
        Standard3aryCuckooFilter::from_num_items(81, 4, 12).unwrap() // 3^4 = 81 buckets
    );

    // ── 4-ary ─────────────────────────────────────────────────────────────────
    // Segmented 4-ary: num_buckets must be a power of 2 and ≥ 4 (ensures each segment is also a power of 2).
    demo!(
        "Segmented 4-ary Cuckoo Filter",
        Segmented4aryCuckooFilter::from_num_items(128, 4, 12).unwrap()
    );
    // Standard 4-ary: num_buckets must be a power of 4 and ≥ 1.
    demo!(
        "Standard 4-ary Cuckoo Filter",
        Standard4aryCuckooFilter::from_num_items(64, 4, 12).unwrap() // 4^3 = 64 buckets
    );

    // ==== USING `from_num_items` CONSTRUCTOR =====================================================

    // ── 2-ary ─────────────────────────────────────────────────────────────────
    demo!(
        "Segmented 2-ary Cuckoo Filter",
        Segmented2aryCuckooFilter::from_num_items(100, 4, 12).unwrap()
    );
    demo!(
        "Standard 2-ary Cuckoo Filter",
        Standard2aryCuckooFilter::from_num_items(100, 4, 12).unwrap()
    );

    // ── 3-ary ─────────────────────────────────────────────────────────────────
    demo!(
        "Standard 3-ary Cuckoo Filter",
        Standard3aryCuckooFilter::from_num_items(100, 4, 12).unwrap()
    );
    demo!(
        "Segmented 3-ary Cuckoo Filter",
        Segmented3aryCuckooFilter::from_num_items(100, 4, 12).unwrap()
    );

    // ── 4-ary ─────────────────────────────────────────────────────────────────
    demo!(
        "Standard 4-ary Cuckoo Filter",
        Standard4aryCuckooFilter::from_num_items(100, 4, 12).unwrap()
    );
    demo!(
        "Segmented 4-ary Cuckoo Filter",
        Segmented4aryCuckooFilter::from_num_items(100, 4, 12).unwrap()
    );
}
