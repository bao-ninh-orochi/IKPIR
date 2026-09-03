//! Integration tests for IKPIR incremental-correctness invariants.
//!
//! # Purpose
//!
//! Pins three failure-mode behaviours that are too cross-cutting to
//! live in a single src/ unit test:
//!
//! 1. **`mutation_log_drained_on_failure`** — a failed `insert`
//!    (`TableFull`) must drain the SCF mutation log so the next
//!    successful mutation emits a `HintDeltaBundle` containing only its
//!    own deltas, not the leaked rolled-back ones.
//! 2. **`sparse_delta_correct`** — the bundle of deltas a single
//!    mutation emits must, when applied via `client.apply_delta`,
//!    advance the client's hint to match a freshly-rebuilt client.
//! 3. **`many_mutations_with_warm_queue`** — a long stream of
//!    mutations against a client that has Phase-B/Phase-C warm queues
//!    keeps every prepared slot's decode material consistent with the
//!    patched hint.
//!
//! Each test is parameterised by arity (2 / 3 / 4) so the diagnostic on
//! failure pins which arity broke.

use ikpir_client::ClientUpdateMode;
use ikpir_client::IkpirClient;
use ikpir_server::{
    FrodoConfig, FrodoPirBackend, HintDeltaBundle, HintPatchMode, IkpirError, IkpirServer,
};
use segmented_cuckoo::{
    IndexScheme, SchemeMeta, Segmented2aryCuckooKVStore, Segmented2aryScheme,
    Segmented3aryCuckooKVStore, Segmented3aryScheme, Segmented4aryCuckooKVStore,
    Segmented4aryScheme,
};

/// Build an empty 2-ary `IkpirServer` (64 buckets × 4 slots = 256
/// capacity).
fn build_empty_2() -> IkpirServer<Segmented2aryScheme, FrodoPirBackend> {
    let store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}
/// Build an empty 3-ary `IkpirServer` (`num_buckets = 3 · 32 = 96`,
/// same per-segment row count as the 2-ary fixture).
fn build_empty_3() -> IkpirServer<Segmented3aryScheme, FrodoPirBackend> {
    let store = Segmented3aryCuckooKVStore::new(96, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}
/// Build an empty 4-ary `IkpirServer` (64 buckets × 4 slots).
fn build_empty_4() -> IkpirServer<Segmented4aryScheme, FrodoPirBackend> {
    let store = Segmented4aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}

/// Build a tiny 2-ary `IkpirServer` (16 buckets × 4 slots = 64
/// capacity); easy to fill.
fn build_tiny_2() -> IkpirServer<Segmented2aryScheme, FrodoPirBackend> {
    let store = Segmented2aryCuckooKVStore::new(16, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}
/// Build a tiny 3-ary `IkpirServer` (12 buckets × 4 slots = 48
/// capacity); easy to fill.
fn build_tiny_3() -> IkpirServer<Segmented3aryScheme, FrodoPirBackend> {
    let store = Segmented3aryCuckooKVStore::new(12, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}
/// Build a tiny 4-ary `IkpirServer` (16 buckets × 4 slots = 64
/// capacity); easy to fill.
fn build_tiny_4() -> IkpirServer<Segmented4aryScheme, FrodoPirBackend> {
    let store = Segmented4aryCuckooKVStore::new(16, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}

/// Fill `server` to `TableFull`; then verify that a second failed
/// `insert` does **not** advance the epoch and does **not** corrupt
/// the hint. `fresh` is a second copy used to check the hint identity.
fn mutation_log_drained_inner<S>(
    mut server: IkpirServer<S, FrodoPirBackend>,
    mut fresh: IkpirServer<S, FrodoPirBackend>,
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

    // Verify: hints before and after a failing insert must be identical.
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
fn success_after_failure_advances_once_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
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
fn sparse_delta_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    let bundle: HintDeltaBundle<FrodoPirBackend> = server.insert(b"k", b"v").unwrap();

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

/// Failed `insert` on a 2-ary server leaves epoch and hint unchanged.
#[test]
fn mutation_log_drained_on_failure() {
    mutation_log_drained_inner(build_tiny_2(), build_tiny_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn mutation_log_drained_on_failure_3ary() {
    mutation_log_drained_inner(build_tiny_3(), build_tiny_3());
}
/// Same as 2-ary, on the 4-ary server.
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

/// A single `insert` on a 2-ary server produces a sparse delta bundle
/// touching exactly one segment.
#[test]
fn sparse_delta_correct() {
    sparse_delta_inner(build_empty_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn sparse_delta_correct_3ary() {
    sparse_delta_inner(build_empty_3());
}
/// Same as 2-ary, on the 4-ary server.
#[test]
fn sparse_delta_correct_4ary() {
    sparse_delta_inner(build_empty_4());
}

/// Stress: many mutations against a client with warm Phase-B/Phase-C
/// queues. The `c`-patching loop in `client_patch_state` is the
/// load-bearing piece — if its math is wrong, this surfaces a decode
/// divergence quickly across many distinct `(slot, hint)` pairs.
fn many_mutations_with_warm_queue_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    // Seed the database before snapshotting the client.
    for k in 0u32..20 {
        server.insert(&k.to_le_bytes(), &[k as u8 ^ 0x33]).unwrap();
    }
    let mut client: IkpirClient<FrodoPirBackend> = IkpirClient::from_setup(server.setup());
    client.set_update_mode(ClientUpdateMode::HintPatch);
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
    oracle.set_update_mode(ClientUpdateMode::HintPatch);
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

/// Many mutations with a warm precomputation queue still yield the
/// same decodes as a freshly-rebuilt oracle client (2-ary).
#[test]
fn many_mutations_with_warm_queue() {
    many_mutations_with_warm_queue_inner(build_empty_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn many_mutations_with_warm_queue_3ary() {
    many_mutations_with_warm_queue_inner(build_empty_3());
}
/// Same as 2-ary, on the 4-ary server.
#[test]
fn many_mutations_with_warm_queue_4ary() {
    many_mutations_with_warm_queue_inner(build_empty_4());
}

/// Cross-mode lock-step: a server realizing its hint patches with one
/// [`HintPatchMode`] stays consistent with a client realizing them with
/// the other — including a mid-stream swap of both sides. Either
/// realization leaves the hint equal to `A·D` for the post-mutation
/// database, so the mode is a purely local choice and every
/// insert / update / delete class must decode correctly afterwards.
fn cross_mode_lock_step_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
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

    // Server realizes patches row-level; client entry-level (its default).
    server.set_hint_patch_mode(HintPatchMode::RowLevel);
    let mut client: IkpirClient<FrodoPirBackend> = IkpirClient::from_setup(server.setup());
    client.set_update_mode(ClientUpdateMode::HintPatch);
    assert_eq!(
        client.hint_patch_mode(),
        HintPatchMode::EntryLevel,
        "client default mode must be entry-level"
    );

    for k in 20u32..25 {
        let d = server.insert(&k.to_le_bytes(), &[k as u8]).unwrap();
        client.apply_delta(d).unwrap();
    }

    // Swap both sides mid-stream: server entry-level, client row-level.
    server.set_hint_patch_mode(HintPatchMode::EntryLevel);
    client.set_hint_patch_mode(HintPatchMode::RowLevel);
    for k in 0u32..10 {
        let d = server.update(&k.to_le_bytes(), &[k as u8 ^ 0x5A]).unwrap();
        client.apply_delta(d).unwrap();
    }
    for k in 15u32..20 {
        let d = server.delete(&k.to_le_bytes()).unwrap();
        client.apply_delta(d).unwrap();
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
        let v = client.decode(&key, &r).unwrap();
        assert_eq!(v, expect(k), "cross-mode decode mismatch on key {k}");
    }
}

/// Server and client in opposite patch modes (with a mid-stream swap)
/// stay in lock-step on a 2-ary server.
#[test]
fn cross_mode_lock_step() {
    cross_mode_lock_step_inner(build_empty_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn cross_mode_lock_step_3ary() {
    cross_mode_lock_step_inner(build_empty_3());
}
/// Same as 2-ary, on the 4-ary server.
#[test]
fn cross_mode_lock_step_4ary() {
    cross_mode_lock_step_inner(build_empty_4());
}
