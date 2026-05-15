//! Integration test: `IkpirServer` composed with `FrodoPirBackend`.
//!
//! Verifies that the client's per-segment hint state stays in lock-step
//! with the server's after a sequence of insert / update / delete mutations.

use ikpir_server::{FrodoConfig, FrodoPirBackend, IkpirServer, IndexPirBackend, IncrementalPirBackend};
use ikpir_common::backend::frodo::FrodoClientState;
use segmented_cuckoo::{Segmented2aryCuckooKVStore, Segmented2aryScheme};

#[test]
fn smoke_ikpir_server_compose_with_frodo() {
    let store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    let mut server: IkpirServer<Segmented2aryScheme, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::default());

    let setup = server.setup();
    let backend_params = setup.backend_params.clone();
    let mut states: Vec<FrodoClientState> = backend_params
        .iter()
        .zip(setup.hints.iter())
        .map(|(p, h)| FrodoPirBackend::client_setup(p, h))
        .collect();

    let bundles = vec![
        server.insert(b"k1", b"a").unwrap(),
        server.insert(b"k2", b"b").unwrap(),
        server.update(b"k1", b"A").unwrap(),
        server.delete(b"k2").unwrap(),
    ];

    for bundle in &bundles {
        for (j, deltas) in bundle.per_segment_row_deltas.iter().enumerate() {
            if !deltas.is_empty() {
                FrodoPirBackend::client_patch_state(
                    &mut states[j],
                    &backend_params[j],
                    deltas,
                );
            }
        }
    }

    let final_setup = server.setup();
    for (st, srv) in states.iter().zip(final_setup.hints.iter()) {
        assert_eq!(st.hint.data, srv.data,
            "client patched state diverged from server hint");
    }
}
