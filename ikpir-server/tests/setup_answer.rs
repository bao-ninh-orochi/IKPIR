use ikpir_client::IkpirClient;
use ikpir_server::{FrodoConfig, FrodoPirBackend, IkpirError, IkpirServer, ServerSetupBundle};
use segmented_cuckoo::{
    IndexScheme, SchemeMeta,
    Segmented2aryCuckooKVStore, Segmented2aryScheme,
    Segmented3aryCuckooKVStore, Segmented3aryScheme,
    Segmented4aryCuckooKVStore, Segmented4aryScheme,
};

fn build_server_2() -> IkpirServer<Segmented2aryScheme, FrodoPirBackend> {
    let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    for k in 0u32..16 {
        store.insert(k.to_le_bytes(), &[k as u8 ^ 0xA5]).unwrap();
    }
    IkpirServer::new(store, FrodoConfig::default())
}

fn build_server_3() -> IkpirServer<Segmented3aryScheme, FrodoPirBackend> {
    // num_buckets must be 3 * 2^t. 12 = 3*4, capacity = 48 slots.
    let mut store = Segmented3aryCuckooKVStore::new(12, 4, 12, 8, 8).unwrap();
    for k in 0u32..16 {
        store.insert(k.to_le_bytes(), &[k as u8 ^ 0xA5]).unwrap();
    }
    IkpirServer::new(store, FrodoConfig::default())
}

fn build_server_4() -> IkpirServer<Segmented4aryScheme, FrodoPirBackend> {
    let mut store = Segmented4aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    for k in 0u32..16 {
        store.insert(k.to_le_bytes(), &[k as u8 ^ 0xA5]).unwrap();
    }
    IkpirServer::new(store, FrodoConfig::default())
}

fn setup_then_answer_inner<S>(server: IkpirServer<S, FrodoPirBackend>)
where S: IndexScheme + SchemeMeta + 'static {
    let bundle: ServerSetupBundle<FrodoPirBackend> = server.setup();
    assert_eq!(bundle.epoch, 0);
    assert_eq!(bundle.backend_params.len(), bundle.params.arity());
    assert_eq!(bundle.hints.len(), bundle.params.arity());

    let mut client = IkpirClient::<FrodoPirBackend>::from_setup(bundle);

    for k in 0u32..16 {
        let key  = k.to_le_bytes();
        let q    = client.build_query(&key);
        let resp = server.answer(&q).expect("answer must succeed");
        let got  = client.decode(&key, &resp).expect("no error").expect("key present");
        assert_eq!(got, vec![k as u8 ^ 0xA5], "value mismatch for key {k}");
    }
}

fn stale_epoch_inner<S>(server: IkpirServer<S, FrodoPirBackend>)
where S: IndexScheme + SchemeMeta + 'static {
    let mut client = IkpirClient::<FrodoPirBackend>::from_setup(server.setup());
    let mut q      = client.build_query(&0u32.to_le_bytes());
    q.epoch = 99;
    match server.answer(&q) {
        Err(IkpirError::StaleEpoch { expected: 0, got: 99 }) => {}
        Err(e) => panic!("expected StaleEpoch(0, 99), got error {e:?}"),
        Ok(_)  => panic!("expected StaleEpoch(0, 99), got Ok"),
    }
}

fn full_rebuild_increments_epoch_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
where S: IndexScheme + SchemeMeta + 'static {
    let initial = server.setup();
    assert_eq!(initial.epoch, 0);

    let rebuilt = server.full_rebuild();
    assert_eq!(rebuilt.epoch, 1);
    assert_eq!(server.epoch(), 1);
    assert_eq!(rebuilt.params, initial.params);
    assert_eq!(rebuilt.backend_params.len(), initial.backend_params.len());
    assert_eq!(rebuilt.hints.len(), initial.hints.len());
}

#[test]
fn setup_then_answer_returns_correct_row()       { setup_then_answer_inner(build_server_2()); }
#[test]
fn setup_then_answer_returns_correct_row_3ary()  { setup_then_answer_inner(build_server_3()); }
#[test]
fn setup_then_answer_returns_correct_row_4ary()  { setup_then_answer_inner(build_server_4()); }

#[test]
fn stale_epoch_rejected()       { stale_epoch_inner(build_server_2()); }
#[test]
fn stale_epoch_rejected_3ary()  { stale_epoch_inner(build_server_3()); }
#[test]
fn stale_epoch_rejected_4ary()  { stale_epoch_inner(build_server_4()); }

#[test]
fn full_rebuild_increments_epoch_and_matches_setup()       { full_rebuild_increments_epoch_inner(build_server_2()); }
#[test]
fn full_rebuild_increments_epoch_and_matches_setup_3ary()  { full_rebuild_increments_epoch_inner(build_server_3()); }
#[test]
fn full_rebuild_increments_epoch_and_matches_setup_4ary()  { full_rebuild_increments_epoch_inner(build_server_4()); }
