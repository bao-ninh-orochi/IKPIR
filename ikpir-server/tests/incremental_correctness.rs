use ikpir_client::IkpirClient;
use ikpir_server::{FrodoConfig, FrodoPirBackend, HintDeltaBundle, IkpirError, IkpirServer};
use segmented_cuckoo::{
    IndexScheme, SchemeMeta,
    Segmented2aryCuckooKVStore, Segmented2aryScheme,
    Segmented3aryCuckooKVStore, Segmented3aryScheme,
    Segmented4aryCuckooKVStore, Segmented4aryScheme,
};

fn build_empty_2() -> IkpirServer<Segmented2aryScheme, FrodoPirBackend> {
    let store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}
fn build_empty_3() -> IkpirServer<Segmented3aryScheme, FrodoPirBackend> {
    // 96 = 3 * 32: same per-segment row count as arity-2 with 64 buckets.
    let store = Segmented3aryCuckooKVStore::new(96, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}
fn build_empty_4() -> IkpirServer<Segmented4aryScheme, FrodoPirBackend> {
    let store = Segmented4aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}

fn build_tiny_2() -> IkpirServer<Segmented2aryScheme, FrodoPirBackend> {
    // 16 buckets * 4 slots = 64 capacity. Easy to fill.
    let store = Segmented2aryCuckooKVStore::new(16, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}
fn build_tiny_3() -> IkpirServer<Segmented3aryScheme, FrodoPirBackend> {
    // 12 = 3 * 4 buckets * 4 slots = 48 capacity.
    let store = Segmented3aryCuckooKVStore::new(12, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}
fn build_tiny_4() -> IkpirServer<Segmented4aryScheme, FrodoPirBackend> {
    let store = Segmented4aryCuckooKVStore::new(16, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}

fn mutation_log_drained_inner<S>(
    mut server: IkpirServer<S, FrodoPirBackend>,
    mut fresh:  IkpirServer<S, FrodoPirBackend>,
)
where S: IndexScheme + SchemeMeta + 'static {
    for k in 0u32..2000 {
        match server.insert(&k.to_le_bytes(), &[k as u8]) {
            Ok(_)                      => {}
            Err(IkpirError::TableFull) => break,
            Err(other)                 => panic!("unexpected {other:?}"),
        }
    }
    let epoch_after_fill = server.epoch();

    // A second failed insert must NOT advance the epoch.
    let err = match server.insert(&999_999u32.to_le_bytes(), &[0]) {
        Err(e) => e,
        Ok(_)  => panic!("expected TableFull, got Ok"),
    };
    assert!(matches!(err, IkpirError::TableFull));
    assert_eq!(server.epoch(), epoch_after_fill,
        "failed insert must not advance epoch (mutation log leaked)");

    // Verify: hints before and after a failing insert must be identical.
    for k in 0u32..2000 {
        match fresh.insert(&k.to_le_bytes(), &[k as u8]) {
            Ok(_)                      => {}
            Err(IkpirError::TableFull) => break,
            Err(other)                 => panic!("unexpected {other:?}"),
        }
    }
    let hints_before = fresh.setup().hints.clone();
    let _ = fresh.insert(&999_999u32.to_le_bytes(), &[0]); // intentional failure
    let hints_after = fresh.setup().hints;
    assert_eq!(hints_before, hints_after,
        "failed insert must not corrupt hint (mutation log leaked)");
}

fn sparse_delta_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
where S: IndexScheme + SchemeMeta + 'static {
    let bundle: HintDeltaBundle<FrodoPirBackend> = server.insert(b"k", b"v").unwrap();

    let cps = server.params().cells_per_slot() as usize;

    let total: usize = bundle.per_segment_row_deltas.iter()
        .flat_map(|seg| seg.iter())
        .map(|(_, deltas)| deltas.len())
        .sum();

    assert!(total >= 1,   "fp is non-zero, so ≥1 cell delta required");
    assert!(total <= cps, "delta count {total} exceeds cells_per_slot {cps}");

    let touched_segments = bundle.per_segment_row_deltas.iter()
        .filter(|seg| !seg.is_empty())
        .count();
    assert_eq!(touched_segments, 1,
        "a single insert lands in one bucket → one segment touched");
}

#[test]
fn mutation_log_drained_on_failure() {
    mutation_log_drained_inner(build_tiny_2(), build_tiny_2());
}
#[test]
fn mutation_log_drained_on_failure_3ary() {
    mutation_log_drained_inner(build_tiny_3(), build_tiny_3());
}
#[test]
fn mutation_log_drained_on_failure_4ary() {
    mutation_log_drained_inner(build_tiny_4(), build_tiny_4());
}

#[test]
fn sparse_delta_correct()       { sparse_delta_inner(build_empty_2()); }
#[test]
fn sparse_delta_correct_3ary()  { sparse_delta_inner(build_empty_3()); }
#[test]
fn sparse_delta_correct_4ary()  { sparse_delta_inner(build_empty_4()); }

// Stress test: many mutations × many precomputed slots. The c-patching loop
// is the load-bearing piece — if its math is wrong, this surfaces a decode
// divergence quickly across many distinct (slot, hint) pairs.
fn many_mutations_with_warm_queue_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
where S: IndexScheme + SchemeMeta + 'static {
    // Seed the database before snapshotting the client.
    for k in 0u32..20 {
        server.insert(&k.to_le_bytes(), &[k as u8 ^ 0x33]).unwrap();
    }
    let mut client: IkpirClient<FrodoPirBackend> = IkpirClient::from_setup(server.setup());
    client.precompute_queries(60); // > number of probes below to keep cheap path active
    client.precompute_decodes();

    // Apply 30 mutations, never re-warming.
    for k in 20u32..50 {
        let delta: HintDeltaBundle<FrodoPirBackend> =
            server.insert(&k.to_le_bytes(), &[k as u8 ^ 0x33]).unwrap();
        client.apply_delta(delta).unwrap();
    }
    for k in 0u32..10 {
        let delta = server.update(&k.to_le_bytes(), &[k as u8 ^ 0x77]).unwrap();
        client.apply_delta(delta).unwrap();
    }
    for k in 30u32..40 {
        let delta = server.delete(&k.to_le_bytes()).unwrap();
        client.apply_delta(delta).unwrap();
    }

    // After 50 patches, the warm queue is still consistent with the patched
    // hint. Probe a mix of present/absent keys against an oracle.
    let mut oracle: IkpirClient<FrodoPirBackend> = IkpirClient::from_setup(server.setup());
    let probes: &[u32] = &[0, 5, 12, 19, 25, 30, 35, 40, 45, 999];
    for &k in probes {
        let key = k.to_le_bytes();
        let q_w = client.build_query(&key);
        let r_w = server.answer(&q_w).unwrap();
        let v_w = client.decode(&key, &r_w).unwrap();

        let q_o = oracle.build_query(&key);
        let r_o = server.answer(&q_o).unwrap();
        let v_o = oracle.decode(&key, &r_o).unwrap();
        assert_eq!(v_w, v_o, "warm queue diverged from oracle on key {k}");
    }
}

#[test]
fn many_mutations_with_warm_queue()       { many_mutations_with_warm_queue_inner(build_empty_2()); }
#[test]
fn many_mutations_with_warm_queue_3ary()  { many_mutations_with_warm_queue_inner(build_empty_3()); }
#[test]
fn many_mutations_with_warm_queue_4ary()  { many_mutations_with_warm_queue_inner(build_empty_4()); }
