//! Integration tests for the IKPIR server's setup / answer / rebuild
//! protocol against a real `IkpirClient` and `FrodoPirBackend`.
//!
//! # Purpose
//!
//! Pins three end-to-end behaviours that the unit tests can't cover
//! because they cross the server/client boundary:
//!
//! 1. `setup → build_query → answer → decode` returns the originally
//!    inserted value, for every arity (2/3/4) and every populated key.
//! 2. `answer` rejects stale-epoch queries with `IkpirError::StaleEpoch`.
//! 3. `full_rebuild` increments the epoch and emits a setup bundle that
//!    matches a fresh `IkpirServer::new` from the same store.
//!
//! Each behaviour is parameterised by arity via the `inner` helper
//! functions, then exercised by three thin `#[test]` wrappers (one per
//! arity) so the diagnostic on failure pins which arity broke.

use ikpir_client::IkpirClient;
use ikpir_server::{FrodoConfig, FrodoPirBackend, IkpirError, IkpirServer, ServerSetupBundle};
use segmented_cuckoo::{
    IndexScheme, SchemeMeta, Segmented2aryCuckooKVStore, Segmented2aryScheme,
    Segmented3aryCuckooKVStore, Segmented3aryScheme, Segmented4aryCuckooKVStore,
    Segmented4aryScheme,
};

/// Build a 2-ary `IkpirServer` populated with 16 keys (`0..16`).
fn build_server_2() -> IkpirServer<Segmented2aryScheme, FrodoPirBackend> {
    let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    for k in 0u32..16 {
        store.insert(k.to_le_bytes(), &[k as u8 ^ 0xA5]).unwrap();
    }
    IkpirServer::new(store, FrodoConfig::default())
}

/// Build a 3-ary `IkpirServer` populated with 16 keys (`0..16`).
fn build_server_3() -> IkpirServer<Segmented3aryScheme, FrodoPirBackend> {
    // num_buckets must be 3 * 2^t. 12 = 3*4, capacity = 48 slots.
    let mut store = Segmented3aryCuckooKVStore::new(12, 4, 12, 8, 8).unwrap();
    for k in 0u32..16 {
        store.insert(k.to_le_bytes(), &[k as u8 ^ 0xA5]).unwrap();
    }
    IkpirServer::new(store, FrodoConfig::default())
}

/// Build a 4-ary `IkpirServer` populated with 16 keys (`0..16`).
fn build_server_4() -> IkpirServer<Segmented4aryScheme, FrodoPirBackend> {
    let mut store = Segmented4aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    for k in 0u32..16 {
        store.insert(k.to_le_bytes(), &[k as u8 ^ 0xA5]).unwrap();
    }
    IkpirServer::new(store, FrodoConfig::default())
}

/// Arity-generic body of the "setup → query → answer → decode returns
/// the inserted value" test.
fn setup_then_answer_inner<S>(server: IkpirServer<S, FrodoPirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    let bundle: ServerSetupBundle<FrodoPirBackend> = server.setup();
    assert_eq!(bundle.epoch, 0);
    assert_eq!(bundle.backend_params.len(), bundle.params.arity());
    assert_eq!(bundle.hints.len(), bundle.params.arity());

    let mut client = IkpirClient::<FrodoPirBackend>::from_setup(bundle);

    for k in 0u32..16 {
        let key = k.to_le_bytes();
        let q = client.build_query(&key);
        let resp = server.answer(&q).expect("answer must succeed");
        let got = client
            .decode(&key, &q, &resp)
            .expect("no error")
            .expect("key present");
        assert_eq!(got, vec![k as u8 ^ 0xA5], "value mismatch for key {k}");
    }
}

/// Arity-generic body of the "stale-epoch query is rejected with
/// `IkpirError::StaleEpoch`" test.
fn stale_epoch_inner<S>(server: IkpirServer<S, FrodoPirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    let mut client = IkpirClient::<FrodoPirBackend>::from_setup(server.setup());
    let mut q = client.build_query(&0u32.to_le_bytes());
    q.epoch = 99;
    match server.answer(&q) {
        Err(IkpirError::StaleEpoch {
            expected: 0,
            got: 99,
        }) => {}
        Err(e) => panic!("expected StaleEpoch(0, 99), got error {e:?}"),
        Ok(_) => panic!("expected StaleEpoch(0, 99), got Ok"),
    }
}

/// Arity-generic body of the "`full_rebuild` increments `epoch` and
/// emits a bundle structurally matching the original setup" test.
fn full_rebuild_increments_epoch_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    let initial = server.setup();
    assert_eq!(initial.epoch, 0);

    let rebuilt = server.full_rebuild();
    assert_eq!(rebuilt.epoch, 1);
    assert_eq!(server.epoch(), 1);
    assert_eq!(rebuilt.params, initial.params);
    assert_eq!(rebuilt.backend_params.len(), initial.backend_params.len());
    assert_eq!(rebuilt.hints.len(), initial.hints.len());
}

/// Setup → query → answer → decode round-trips every populated key on
/// the 2-ary server.
#[test]
fn setup_then_answer_returns_correct_row() {
    setup_then_answer_inner(build_server_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn setup_then_answer_returns_correct_row_3ary() {
    setup_then_answer_inner(build_server_3());
}
/// Same as 2-ary, on the 4-ary server.
#[test]
fn setup_then_answer_returns_correct_row_4ary() {
    setup_then_answer_inner(build_server_4());
}

/// A stale-epoch query is rejected with `IkpirError::StaleEpoch`
/// (2-ary).
#[test]
fn stale_epoch_rejected() {
    stale_epoch_inner(build_server_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn stale_epoch_rejected_3ary() {
    stale_epoch_inner(build_server_3());
}
/// Same as 2-ary, on the 4-ary server.
#[test]
fn stale_epoch_rejected_4ary() {
    stale_epoch_inner(build_server_4());
}

/// `full_rebuild` advances the server epoch from 0 to 1 and emits a
/// bundle whose shape matches the original setup (2-ary).
#[test]
fn full_rebuild_increments_epoch_and_matches_setup() {
    full_rebuild_increments_epoch_inner(build_server_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn full_rebuild_increments_epoch_and_matches_setup_3ary() {
    full_rebuild_increments_epoch_inner(build_server_3());
}
/// Same as 2-ary, on the 4-ary server.
#[test]
fn full_rebuild_increments_epoch_and_matches_setup_4ary() {
    full_rebuild_increments_epoch_inner(build_server_4());
}
