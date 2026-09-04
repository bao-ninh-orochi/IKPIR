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
use ikpir_server::{
    HintDeltaBundle, HintPatchMode, IkpirError, IkpirServer, SimpleConfig, SimplePirBackend,
};
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
    // The insert that *reports* `TableFull` is the one under test — a
    // table that rejected one key may still accept another (cuckoo kick
    // victims are chosen at random), so probing with some extra key would
    // be a coin flip. Snapshot the epoch after each success instead, and
    // let the fill loop's own failure be the failed insert.
    let mut epoch_before_fail = server.epoch();
    let mut hit_full = false;
    for k in 0u32..2000 {
        match server.insert(&k.to_le_bytes(), &[k as u8]) {
            Ok(_) => epoch_before_fail = server.epoch(),
            Err(IkpirError::TableFull) => {
                hit_full = true;
                break;
            }
            Err(other) => panic!("unexpected {other:?}"),
        }
    }
    assert!(hit_full, "fixture must reach TableFull within 2000 inserts");
    assert_eq!(
        server.epoch(),
        epoch_before_fail,
        "failed insert must not advance epoch (mutation log leaked)"
    );

    // Same shape — the snapshot tracks the last success, so the compare
    // straddles exactly the failing insert.
    let mut hints_before = fresh.setup().hints;
    let mut hit_full = false;
    for k in 0u32..2000 {
        match fresh.insert(&k.to_le_bytes(), &[k as u8]) {
            Ok(_) => hints_before = fresh.setup().hints,
            Err(IkpirError::TableFull) => {
                hit_full = true;
                break;
            }
            Err(other) => panic!("unexpected {other:?}"),
        }
    }
    assert!(hit_full, "fixture must reach TableFull within 2000 inserts");
    let hints_after = fresh.setup().hints;
    assert_eq!(
        hints_before, hints_after,
        "failed insert must not corrupt hint (mutation log leaked)"
    );
}

/// Success-after-failure leakage guard. Fill to `TableFull`; confirm a
/// further failed `insert` does not advance the epoch; then `delete` an
/// existing key and re-`insert` it. Each successful mutation must advance
/// the epoch by exactly one, proving the mutation log was drained on the
/// `TableFull` failure and that neither the delete nor the
/// success-after-failure insert leaks rolled-back deltas.
fn success_after_failure_advances_once_inner<S>(mut server: IkpirServer<S, SimplePirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    // Fill to TableFull, remembering the last key that was inserted OK.
    // The failing insert is the fill loop's own — see
    // `mutation_log_drained_inner` for why probing with an extra key
    // would be a coin flip.
    let mut last_ok: Option<u32> = None;
    let mut epoch_full = server.epoch();
    let mut hit_full = false;
    for k in 0u32..2000 {
        match server.insert(&k.to_le_bytes(), &[k as u8]) {
            Ok(_) => {
                last_ok = Some(k);
                epoch_full = server.epoch();
            }
            Err(IkpirError::TableFull) => {
                hit_full = true;
                break;
            }
            Err(other) => panic!("unexpected {other:?}"),
        }
    }
    assert!(hit_full, "fixture must reach TableFull within 2000 inserts");
    let key = last_ok.expect("at least one insert must succeed before TableFull");

    // That failed insert must NOT have advanced the epoch.
    assert_eq!(
        server.epoch(),
        epoch_full,
        "failed insert must not advance epoch"
    );

    // Delete a key that is provably present: epoch advances exactly once.
    let del = server
        .delete(&key.to_le_bytes())
        .expect("delete of a present key must succeed");
    assert_eq!(
        del.epoch,
        epoch_full + 1,
        "delete must advance epoch by exactly one"
    );

    // Re-insert the SAME key. Its candidate slot was just freed, so the
    // insert is guaranteed to fit; success-after-failure must advance the
    // epoch by exactly one more.
    let ins = server
        .insert(&key.to_le_bytes(), &[key as u8])
        .expect("re-insert into the freed slot must succeed");
    assert_eq!(
        ins.epoch,
        epoch_full + 2,
        "success-after-failure insert must advance epoch by exactly one"
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

/// A delete then a success-after-failure insert each advance the epoch
/// exactly once on a 2-ary server.
#[test]
fn success_after_failure_advances_once() {
    success_after_failure_advances_once_inner(build_tiny_2());
}
/// Same, on the 3-ary server.
#[test]
fn success_after_failure_advances_once_3ary() {
    success_after_failure_advances_once_inner(build_tiny_3());
}
/// Same, on the 4-ary server.
#[test]
fn success_after_failure_advances_once_4ary() {
    success_after_failure_advances_once_inner(build_tiny_4());
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
        client.accumulate_delta(delta).unwrap();
    }
    for k in 0u32..10 {
        let delta = server.update(&k.to_le_bytes(), &[k as u8 ^ 0x77]).unwrap();
        client.accumulate_delta(delta).unwrap();
    }
    for k in 30u32..40 {
        let delta = server.delete(&k.to_le_bytes()).unwrap();
        client.accumulate_delta(delta).unwrap();
    }

    let mut oracle: IkpirClient<SimplePirBackend> = IkpirClient::from_setup(server.setup());
    let probes: &[u32] = &[0, 5, 12, 19, 25, 30, 35, 40, 45, 999];
    for &k in probes {
        let key = k.to_le_bytes();
        let q_w = client.build_query(&key);
        let r_w = server.answer(&q_w).unwrap();
        let v_w = client.decode(&key, &q_w, &r_w).unwrap();

        let q_o = oracle.build_query(&key);
        let r_o = server.answer(&q_o).unwrap();
        let v_o = oracle.decode(&key, &q_o, &r_o).unwrap();
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

/// Server-side hint-patch mode switch under the SimplePIR reshape: the
/// server switches `HintPatchMode` mid-stream while a rewind client, which
/// never selects a realization of its own, stays consistent throughout.
/// This is the backend where the two realizations differ most (the
/// row-level pass spans the whole `reshape_row_width`), so a grouping or
/// translation bug in the row-level path surfaces here as a decode
/// divergence.
fn server_patch_mode_switch_inner<S>(mut server: IkpirServer<S, SimplePirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    assert_eq!(
        server.hint_patch_mode(),
        HintPatchMode::EntryLevel,
        "server default mode must be entry-level"
    );

    for k in 0u32..20 {
        server.insert(&k.to_le_bytes(), &[k as u8]).unwrap();
    }

    // Server realizes patches row-level from here on.
    server.set_hint_patch_mode(HintPatchMode::RowLevel);
    let mut client: IkpirClient<SimplePirBackend> = IkpirClient::from_setup(server.setup());

    for k in 20u32..25 {
        let d = server.insert(&k.to_le_bytes(), &[k as u8]).unwrap();
        client.accumulate_delta(d).unwrap();
    }

    // Swap back mid-stream.
    server.set_hint_patch_mode(HintPatchMode::EntryLevel);
    for k in 0u32..10 {
        let d = server.update(&k.to_le_bytes(), &[k as u8 ^ 0x5A]).unwrap();
        client.accumulate_delta(d).unwrap();
    }
    for k in 15u32..20 {
        let d = server.delete(&k.to_le_bytes()).unwrap();
        client.accumulate_delta(d).unwrap();
    }

    // Behavioural check across every mutation class the stream produced.
    let expect = |k: u32| -> Option<Vec<u8>> {
        match k {
            0..=9 => Some(vec![k as u8 ^ 0x5A]), // updated
            10..=14 => Some(vec![k as u8]),      // untouched
            15..=19 => None,                     // deleted
            20..=24 => Some(vec![k as u8]),      // inserted post-setup
            _ => None,                           // never inserted
        }
    };
    for k in [0u32, 4, 9, 10, 14, 15, 19, 20, 24, 999] {
        let key = k.to_le_bytes();
        let q = client.build_query(&key);
        let r = server.answer(&q).unwrap();
        let v = client.decode(&key, &q, &r).unwrap();
        assert_eq!(
            v,
            expect(k),
            "server-patch-mode-switch decode mismatch on key {k}"
        );
    }
}

/// A server that switches `HintPatchMode` mid-stream stays consistent with
/// a rewind client on a 2-ary server.
#[test]
fn server_patch_mode_switch() {
    server_patch_mode_switch_inner(build_empty_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn server_patch_mode_switch_3ary() {
    server_patch_mode_switch_inner(build_empty_3());
}
/// Same as 2-ary, on the 4-ary server.
#[test]
fn server_patch_mode_switch_4ary() {
    server_patch_mode_switch_inner(build_empty_4());
}
