use segmented_cuckoo_filter::{
    Segmented3aryCuckooFilter, Segmented4aryCuckooFilter, SegmentedCuckooFilter,
    Standard3aryCuckooFilter, Standard4aryCuckooFilter, StandardCuckooFilter,
};

fn print_header() {
    println!(
        "{:<12} {:<6} {:<6} {:<12} {:<12}",
        "n", "b", "fp", "inserted", "load_factor"
    );
    println!("{}", "-".repeat(50));
}

fn main() {
    // Configurations: (n, b, fingerprint_bits)
    // Standard / Segmented 2-ary / Standard 3-ary / Segmented 4-ary: n must be power of 2
    let configs_pow2: Vec<(u32, u32, u32)> = vec![
        (1 << 14, 2, 12),
        (1 << 14, 4, 12),
        (1 << 14, 8, 12),
        (1 << 16, 2, 16),
        (1 << 16, 4, 16),
        (1 << 16, 8, 16),
    ];

    // Segmented 3-ary: n must be 3·2^m (n/3 is a power of 2).
    // Use segment sizes 2^12=4096 and 2^14=16384, giving n=12288 and n=49152.
    let configs_3ary_seg: Vec<(u32, u32, u32)> = vec![
        (3 * (1 << 12), 2, 12),
        (3 * (1 << 12), 4, 12),
        (3 * (1 << 12), 8, 12),
        (3 * (1 << 14), 2, 16),
        (3 * (1 << 14), 4, 16),
        (3 * (1 << 14), 8, 16),
    ];

    // ── Segmented 2-ary ───────────────────────────────────────────────────────
    println!("=== Segmented 2-ary Cuckoo Filter ===");
    print_header();
    for &(n, b, fp) in &configs_pow2 {
        let mut filter = SegmentedCuckooFilter::new(n, b, fp).unwrap();
        let mut inserted = 0u64;
        for i in 0u64.. {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            inserted += 1;
        }
        println!(
            "{:<12} {:<6} {:<6} {:<12} {:<12.6}",
            n,
            b,
            fp,
            inserted,
            filter.load_factor()
        );
    }

    // ── Standard 2-ary ───────────────────────────────────────────────────────
    println!("\n=== Standard 2-ary Cuckoo Filter ===");
    print_header();
    for &(n, b, fp) in &configs_pow2 {
        let mut filter = StandardCuckooFilter::new(n, b, fp).unwrap();
        let mut inserted = 0u64;
        for i in 0u64.. {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            inserted += 1;
        }
        println!(
            "{:<12} {:<6} {:<6} {:<12} {:<12.6}",
            n,
            b,
            fp,
            inserted,
            filter.load_factor()
        );
    }

    // Pow3: n values that are powers of 3 (3^k).
    let configs_pow3: Vec<(u32, u32, u32)> = vec![
        (3u32.pow(8), 2, 12), // 6561
        (3u32.pow(8), 4, 12),
        (3u32.pow(8), 8, 12),
        (3u32.pow(9), 2, 16), // 19683
        (3u32.pow(9), 4, 16),
        (3u32.pow(9), 8, 16),
    ];

    // ── Standard 3-ary ───────────────────────────────────────────────────────
    println!("\n=== Standard 3-ary Cuckoo Filter ===");
    print_header();
    for &(n, b, fp) in &configs_pow3 {
        let mut filter = Standard3aryCuckooFilter::new(n, b, fp).unwrap();
        let mut inserted = 0u64;
        for i in 0u64.. {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            inserted += 1;
        }
        println!(
            "{:<12} {:<6} {:<6} {:<12} {:<12.6}",
            n,
            b,
            fp,
            inserted,
            filter.load_factor()
        );
    }

    // ── Segmented 3-ary ──────────────────────────────────────────────────────
    println!("\n=== Segmented 3-ary Cuckoo Filter ===");
    print_header();
    for &(n, b, fp) in &configs_3ary_seg {
        let mut filter = Segmented3aryCuckooFilter::new(n, b, fp).unwrap();
        let mut inserted = 0u64;
        for i in 0u64.. {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            inserted += 1;
        }
        println!(
            "{:<12} {:<6} {:<6} {:<12} {:<12.6}",
            n,
            b,
            fp,
            inserted,
            filter.load_factor()
        );
    }

    // Pow4: n values that are powers of 4 (4^k).
    let configs_pow4: Vec<(u32, u32, u32)> = vec![
        (4u32.pow(7), 2, 12), // 16384
        (4u32.pow(7), 4, 12),
        (4u32.pow(7), 8, 12),
        (4u32.pow(8), 2, 16), // 65536
        (4u32.pow(8), 4, 16),
        (4u32.pow(8), 8, 16),
    ];

    // ── Standard 4-ary ───────────────────────────────────────────────────────
    println!("\n=== Standard 4-ary Cuckoo Filter ===");
    print_header();
    for &(n, b, fp) in &configs_pow4 {
        let mut filter = Standard4aryCuckooFilter::new(n, b, fp).unwrap();
        let mut inserted = 0u64;
        for i in 0u64.. {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            inserted += 1;
        }
        println!(
            "{:<12} {:<6} {:<6} {:<12} {:<12.6}",
            n,
            b,
            fp,
            inserted,
            filter.load_factor()
        );
    }

    // ── Segmented 4-ary ──────────────────────────────────────────────────────
    // n must be a power of 2 and >= 4 (same as 2-ary standard/segmented)
    println!("\n=== Segmented 4-ary Cuckoo Filter ===");
    print_header();
    for &(n, b, fp) in &configs_pow2 {
        let mut filter = Segmented4aryCuckooFilter::new(n, b, fp).unwrap();
        let mut inserted = 0u64;
        for i in 0u64.. {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            inserted += 1;
        }
        println!(
            "{:<12} {:<6} {:<6} {:<12} {:<12.6}",
            n,
            b,
            fp,
            inserted,
            filter.load_factor()
        );
    }
}
