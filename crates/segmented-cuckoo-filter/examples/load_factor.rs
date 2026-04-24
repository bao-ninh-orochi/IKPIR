use segmented_cuckoo_filter::{
    Segmented3aryCuckooFilter, Segmented4aryCuckooFilter, Segmented2aryCuckooFilter,
    Standard3aryCuckooFilter, Standard4aryCuckooFilter, Standard2aryCuckooFilter,
};

fn print_header() {
    println!(
        "{:<16} {:<16} {:<16} {:<16} {:<16}",
        "num_buckets", "bucket_size", "fingerprint_bits", "inserted", "load_factor"
    );
    println!("{}", "-".repeat(120));
}

fn main() {
    // Configurations: (num_buckets, bucket_size, fingerprint_bits)
    // Standard 2-ary / Segmented 2-ary / Segmented 4-ary: num_buckets must be power of 2
    let configs_pow2: Vec<(u32, u32, u32)> = vec![
        (1 << 14, 1, 12),
        (1 << 14, 2, 12),
        (1 << 14, 3, 12),
        (1 << 14, 4, 12),
        (1 << 16, 1, 16),
        (1 << 16, 2, 16),
        (1 << 16, 3, 16),
        (1 << 16, 4, 16),
    ];

    // Segmented 3-ary: num_buckets must be 3·2^m (num_buckets/3 is a power of 2).
    // Use segment sizes 2^12=4096 and 2^14=16384, giving num_buckets=12288 and num_buckets=49152.
    let configs_3ary_seg: Vec<(u32, u32, u32)> = vec![
        (3 * (1 << 12), 1, 12),
        (3 * (1 << 12), 2, 12),
        (3 * (1 << 12), 3, 12),
        (3 * (1 << 12), 4, 12),
        (3 * (1 << 14), 1, 16),
        (3 * (1 << 14), 2, 16),
        (3 * (1 << 14), 3, 16),
        (3 * (1 << 14), 4, 16),
    ];

    // Pow3: num_buckets values that are powers of 3 (3^t).
    let configs_pow3: Vec<(u32, u32, u32)> = vec![
        (3u32.pow(8), 1, 12), // 6561
        (3u32.pow(8), 2, 12),
        (3u32.pow(8), 3, 12),
        (3u32.pow(8), 4, 12),
        (3u32.pow(9), 1, 16), // 19683
        (3u32.pow(9), 2, 16),
        (3u32.pow(9), 3, 16),
        (3u32.pow(9), 4, 16),
    ];

        // Pow4: num_buckets values that are powers of 4 (4^t).
    let configs_pow4: Vec<(u32, u32, u32)> = vec![
        (4u32.pow(7), 1, 12), // 16384
        (4u32.pow(7), 2, 12),
        (4u32.pow(7), 3, 12),
        (4u32.pow(7), 4, 12),
        (4u32.pow(8), 1, 16), // 65536
        (4u32.pow(8), 2, 16),
        (4u32.pow(8), 3, 16),
        (4u32.pow(8), 4, 16),
    ];


    // ── Segmented 2-ary ───────────────────────────────────────────────────────
    println!("=== Segmented 2-ary Cuckoo Filter ===");
    print_header();
    for &(num_buckets, bucket_size, fp) in &configs_pow2 {
        let mut filter = Segmented2aryCuckooFilter::new(num_buckets, bucket_size, fp).unwrap();
        let mut inserted = 0u64;
        for i in 0u64.. {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            inserted += 1;
        }
        println!(
            "{:<16} {:<16} {:<16} {:<16} {:<16}",
            num_buckets,
            bucket_size,
            fp,
            inserted,
            filter.load_factor()
        );
    }

    // ── Standard 2-ary ───────────────────────────────────────────────────────
    println!("\n=== Standard 2-ary Cuckoo Filter ===");
    print_header();
    for &(num_buckets, bucket_size, fp) in &configs_pow2 {
        let mut filter = Standard2aryCuckooFilter::new(num_buckets, bucket_size, fp).unwrap();
        let mut inserted = 0u64;
        for i in 0u64.. {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            inserted += 1;
        }
        println!(
            "{:<16} {:<16} {:<16} {:<16} {:<16}",
            num_buckets,
            bucket_size,
            fp,
            inserted,
            filter.load_factor()
        );
    }

    // ── Segmented 3-ary ──────────────────────────────────────────────────────
    println!("\n=== Segmented 3-ary Cuckoo Filter ===");
    print_header();
    for &(num_buckets, bucket_size, fp) in &configs_3ary_seg {
        let mut filter = Segmented3aryCuckooFilter::new(num_buckets, bucket_size, fp).unwrap();
        let mut inserted = 0u64;
        for i in 0u64.. {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            inserted += 1;
        }
        println!(
            "{:<16} {:<16} {:<16} {:<16} {:<16}",
            num_buckets,
            bucket_size,
            fp,
            inserted,
            filter.load_factor()
        );
    }

    // ── Standard 3-ary ───────────────────────────────────────────────────────
    println!("\n=== Standard 3-ary Cuckoo Filter ===");
    print_header();
    for &(num_buckets, bucket_size, fp) in &configs_pow3 {
        let mut filter = Standard3aryCuckooFilter::new(num_buckets, bucket_size, fp).unwrap();
        let mut inserted = 0u64;
        for i in 0u64.. {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            inserted += 1;
        }
        println!(
            "{:<16} {:<16} {:<16} {:<16} {:<16}",
            num_buckets,
            bucket_size,
            fp,
            inserted,
            filter.load_factor()
        );
    }

    // ── Segmented 4-ary ──────────────────────────────────────────────────────
    // num_buckets must be a power of 2 and >= 4 (same as 2-ary standard/segmented)
    println!("\n=== Segmented 4-ary Cuckoo Filter ===");
    print_header();
    for &(num_buckets, bucket_size, fp) in &configs_pow2 {
        let mut filter = Segmented4aryCuckooFilter::new(num_buckets, bucket_size, fp).unwrap();
        let mut inserted = 0u64;
        for i in 0u64.. {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            inserted += 1;
        }
        println!(
            "{:<16} {:<16} {:<16} {:<16} {:<16}",
            num_buckets,
            bucket_size,
            fp,
            inserted,
            filter.load_factor()
        );
    }

    // ── Standard 4-ary ───────────────────────────────────────────────────────
    println!("\n=== Standard 4-ary Cuckoo Filter ===");
    print_header();
    for &(num_buckets, bucket_size, fp) in &configs_pow4 {
        let mut filter = Standard4aryCuckooFilter::new(num_buckets, bucket_size, fp).unwrap();
        let mut inserted = 0u64;
        for i in 0u64.. {
            if filter.add(i.to_le_bytes()).is_err() {
                break;
            }
            inserted += 1;
        }
        println!(
            "{:<16} {:<16} {:<16} {:<16} {:<16}",
            num_buckets,
            bucket_size,
            fp,
            inserted,
            filter.load_factor()
        );
    }
}
