//! Integration tests for IKPIR incremental-correctness invariants under
//! the `SimplePirBackend`.
//!
//! # Purpose
//!
//! Mirror of `incremental_correctness.rs`: pins the same three failure-
//! mode behaviours, but against `SimplePirBackend`. This is the
//! cross-cutting safety net for the SimplePIR reshape + sparse-patch
//! math; if the reshape coordinate translation or the `patch_slot_c`
//! `dot` shortcut is subtly off, `many_mutations_with_warm_queue` will
//! surface it as a decode divergence.

use ikpir_client::IkpirClient;
use ikpir_server::{HintDeltaBundle, IkpirError, IkpirServer, SimpleConfig, SimplePirBackend};
use segmented_cuckoo::{
    IndexScheme, SchemeMeta, Segmented2aryCuckooKVStore, Segmented2aryScheme,
    Segmented3aryCuckooKVStore, Segmented3aryScheme, Segmented4aryCuckooKVStore,
    Segmented4aryScheme,
};

/// Reduced LWE dimension keeps the per-segment matvec fast in tests.
const fn test_config() -> SimpleConfig {
    SimpleConfig {
        lwe_dim: 256,
        sigma: 6.4,
    }
}

/// Build an empty 2-ary `IkpirServer` (64 buckets × 4 slots = 256
/// capacity).
fn build_empty_2() -> IkpirServer<Segmented2aryScheme, SimplePirBackend> {
    let store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, test_config())
}
/// Build an empty 3-ary `IkpirServer` (`num_buckets = 3 · 32 = 96`).
fn build_empty_3() -> IkpirServer<Segmented3aryScheme, SimplePirBackend> {
    let store = Segmented3aryCuckooKVStore::new(96, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, test_config())
}
/// Build an empty 4-ary `IkpirServer` (64 buckets × 4 slots).
fn build_empty_4() -> IkpirServer<Segmented4aryScheme, SimplePirBackend> {
    let store = Segmented4aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, test_config())
}

/// Build a tiny 2-ary `IkpirServer` (16 buckets × 4 slots = 64
/// capacity); easy to fill.
fn build_tiny_2() -> IkpirServer<Segmented2aryScheme, SimplePirBackend> {
    let store = Segmented2aryCuckooKVStore::new(16, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, test_config())
}
/// Build a tiny 3-ary `IkpirServer` (12 buckets × 4 slots = 48
/// capacity); easy to fill.
fn build_tiny_3() -> IkpirServer<Segmented3aryScheme, SimplePirBackend> {
    let store = Segmented3aryCuckooKVStore::new(12, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, test_config())
}
/// Build a tiny 4-ary `IkpirServer` (16 buckets × 4 slots = 64
/// capacity); easy to fill.
fn build_tiny_4() -> IkpirServer<Segmented4aryScheme, SimplePirBackend> {
    let store = Segmented4aryCuckooKVStore::new(16, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, test_config())
}

/// Fill `server` to `TableFull`; then verify that a second failed
/// `insert` does **not** advance the epoch and does **not** corrupt
/// the hint. `fresh` is a second copy used to check the hint identity.
fn mutation_log_drained_inner<S>(
    mut server: IkpirServer<S, SimplePirBackend>,
    mut fresh: IkpirServer<S, SimplePirBackend>,
) where
    S: IndexScheme + SchemeMeta + 'static,
{
    for k in 0u32..2000 {
        match server.insert(&k.to_le_bytes(), &[k as u8]) {
            Ok(_) => {}
            Err(IkpirError::TableFull) => break,
            Err(other) => panic!("unexpected {other:?}"),
        }
    }
    let epoch_after_fill = server.epoch();

    let err = match server.insert(&999_999u32.to_le_bytes(), &[0]) {
        Err(e) => e,
        Ok(_) => panic!("expected TableFull, got Ok"),
    };
    assert!(matches!(err, IkpirError::TableFull));
    assert_eq!(
        server.epoch(),
        epoch_after_fill,
        "failed insert must not advance epoch (mutation log leaked)"
    );

    for k in 0u32..2000 {
        match fresh.insert(&k.to_le_bytes(), &[k as u8]) {
            Ok(_) => {}
            Err(IkpirError::TableFull) => break,
            Err(other) => panic!("unexpected {other:?}"),
        }
    }
    let hints_before = fresh.setup().hints;
    let _ = fresh.insert(&999_999u32.to_le_bytes(), &[0]); // intentional failure
    let hints_after = fresh.setup().hints;
    assert_eq!(
        hints_before, hints_after,
        "failed insert must not corrupt hint (mutation log leaked)"
    );
}

/// A single `insert` must produce a sparse delta bundle: exactly one
/// segment touched (because each key lands in one bucket per scheme)
/// and at most `cells_per_slot` cell deltas in that segment.
fn sparse_delta_inner<S>(mut server: IkpirServer<S, SimplePirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    let bundle: HintDeltaBundle<SimplePirBackend> = server.insert(b"k", b"v").unwrap();

    let cps = server.params().cells_per_slot() as usize;

    let total: usize = bundle
        .per_segment_row_deltas
        .iter()
        .flat_map(|seg| seg.iter())
        .map(|(_, deltas)| deltas.len())
        .sum();

    assert!(total >= 1, "fp is non-zero, so ≥1 cell delta required");
    assert!(
        total <= cps,
        "delta count {total} exceeds cells_per_slot {cps}"
    );

    let touched_segments = bundle
        .per_segment_row_deltas
        .iter()
        .filter(|seg| !seg.is_empty())
        .count();
    assert_eq!(
        touched_segments, 1,
        "a single insert lands in one bucket → one segment touched"
    );
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
fn sparse_delta_correct() {
    sparse_delta_inner(build_empty_2());
}
#[test]
fn sparse_delta_correct_3ary() {
    sparse_delta_inner(build_empty_3());
}
#[test]
fn sparse_delta_correct_4ary() {
    sparse_delta_inner(build_empty_4());
}

/// Stress: many mutations against a client with warm Phase-B/Phase-C
/// queues. The `c`-patching loop in `client_patch_state` is the
/// load-bearing piece — if the reshape-coordinate translation or the
/// SimplePIR `dot` shortcut is wrong, this surfaces a decode divergence
/// quickly across many distinct `(slot, hint)` pairs.
fn many_mutations_with_warm_queue_inner<S>(mut server: IkpirServer<S, SimplePirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    for k in 0u32..20 {
        server.insert(&k.to_le_bytes(), &[k as u8 ^ 0x33]).unwrap();
    }
    let mut client: IkpirClient<SimplePirBackend> = IkpirClient::from_setup(server.setup());
    client.precompute_queries(60);
    client.precompute_decodes();

    for k in 20u32..50 {
        let delta: HintDeltaBundle<SimplePirBackend> =
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

    let mut oracle: IkpirClient<SimplePirBackend> = IkpirClient::from_setup(server.setup());
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
fn many_mutations_with_warm_queue() {
    many_mutations_with_warm_queue_inner(build_empty_2());
}
#[test]
fn many_mutations_with_warm_queue_3ary() {
    many_mutations_with_warm_queue_inner(build_empty_3());
}
#[test]
fn many_mutations_with_warm_queue_4ary() {
    many_mutations_with_warm_queue_inner(build_empty_4());
}
