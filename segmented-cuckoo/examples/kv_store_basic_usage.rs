use segmented_cuckoo::{
    Segmented2aryCuckooKVStore, Segmented3aryCuckooKVStore, Segmented4aryCuckooKVStore,
};

macro_rules! demo {
    ($label:expr, $store:expr) => {{
        let mut s = $store;
        println!("=== {} ===", $label);
        s.insert("apple", &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]).unwrap();
        s.insert("banana", &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]).unwrap();
        s.insert("cherry", &[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x99]).unwrap();
        s.insert(42u64.to_le_bytes(), &[0x42; 8]).unwrap();

        println!("get 'apple':  {:?}", s.get("apple"));
        println!("get 'grape':  {:?}", s.get("grape"));

        // Zero-allocation read into a caller-owned buffer.
        let mut buf = vec![0u8; s.value_size_in_bytes()];
        let hit = s.get_into("banana", &mut buf);
        println!("get_into 'banana': hit={hit} buf={buf:?}");

        // Update an existing key's value.
        s.update("apple", &[0xFE; 8]).unwrap();
        println!("after update, get 'apple': {:?}", s.get("apple"));

        // Delete a key.
        s.delete("cherry").unwrap();
        println!("after delete, get 'cherry': {:?}", s.get("cherry"));

        println!(
            "items: {}  bytes: {}  load: {:.4}",
            s.num_items(),
            s.size_in_bytes(),
            s.load_factor()
        );
        println!();
    }};
}

fn main() {
    // Each segmented variant: 64-bit values, 12-bit fingerprints.
    demo!(
        "Segmented 2-ary KV store",
        Segmented2aryCuckooKVStore::new(128, 4, 12, 64).unwrap()
    );
    demo!(
        "Segmented 3-ary KV store",
        Segmented3aryCuckooKVStore::new(96, 4, 12, 64).unwrap() // 3 * 32 = 96
    );
    demo!(
        "Segmented 4-ary KV store",
        Segmented4aryCuckooKVStore::new(64, 4, 12, 64).unwrap()
    );

    // from_num_items: auto-size for an expected workload.
    demo!(
        "Segmented 2-ary KV store (from_num_items=10_000)",
        Segmented2aryCuckooKVStore::from_num_items(10_000, 4, 12, 64).unwrap()
    );
}
