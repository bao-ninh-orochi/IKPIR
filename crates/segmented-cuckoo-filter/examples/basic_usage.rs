use segmented_cuckoo_filter::{
    Segmented3aryCuckooFilter, Segmented4aryCuckooFilter, SegmentedCuckooFilter,
    Standard3aryCuckooFilter, Standard4aryCuckooFilter, StandardCuckooFilter,
};

macro_rules! demo {
    ($label:expr, $filter:expr) => {{
        let mut f = $filter;
        println!("=== {} ===", $label);
        f.add("apple").unwrap();
        f.add("banana").unwrap();
        f.add("cherry").unwrap();
        f.add(42u64.to_le_bytes()).unwrap();
        println!("contains 'apple':  {}", f.contain("apple"));
        println!("contains 'banana': {}", f.contain("banana"));
        println!("contains 'grape':  {}", f.contain("grape"));
        println!("contains 42:       {}", f.contain(42u64.to_le_bytes()));
        println!("Deleting 'banana'...");
        let _ = f.delete("banana");
        println!("contains 'banana': {}", f.contain("banana"));
        println!(
            "items: {}  bytes: {}  load: {:.4}",
            f.size(),
            f.size_in_bytes(),
            f.load_factor()
        );
        println!();
    }};
}

fn main() {
    // ── 2-ary ─────────────────────────────────────────────────────────────────
    demo!(
        "Segmented 2-ary Cuckoo Filter",
        SegmentedCuckooFilter::from_num_items(100_000, 4, 12).unwrap()
    );
    demo!(
        "Standard 2-ary Cuckoo Filter",
        StandardCuckooFilter::from_num_items(100_000, 4, 12).unwrap()
    );

    // ── 3-ary ─────────────────────────────────────────────────────────────────
    // Standard 3-ary: n must be power of 2; from_num_items auto-sizes.
    demo!(
        "Standard 3-ary Cuckoo Filter",
        Standard3aryCuckooFilter::from_num_items(100_000, 4, 12).unwrap()
    );
    // Segmented 3-ary: n must be 3·2^m; from_num_items auto-sizes.
    demo!(
        "Segmented 3-ary Cuckoo Filter",
        Segmented3aryCuckooFilter::from_num_items(100_000, 4, 12).unwrap()
    );

    // ── 4-ary ─────────────────────────────────────────────────────────────────
    // Standard 4-ary: n must be power of 2; from_num_items auto-sizes.
    demo!(
        "Standard 4-ary Cuckoo Filter",
        Standard4aryCuckooFilter::from_num_items(100_000, 4, 12).unwrap()
    );
    // Segmented 4-ary: n must be power of 2 and ≥ 4; from_num_items auto-sizes.
    demo!(
        "Segmented 4-ary Cuckoo Filter",
        Segmented4aryCuckooFilter::from_num_items(100_000, 4, 12).unwrap()
    );
}
