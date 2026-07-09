//! Integration test: `IkpirServer` composed with `FrodoPirBackend`.
//!
//! Verifies that the client's per-segment hint state stays in lock-step
//! with the server's after a sequence of insert / update / delete
//! mutations — under **both** [`HintPatchMode`] realizations, which must
//! produce bit-identical state from the same delta stream.

use ikpir_common::backend::frodo::FrodoClientState;
use ikpir_server::{
    FrodoConfig, FrodoPirBackend, HintPatchMode, IkpirServer, IncrementalPirBackend,
    IndexPirBackend,
};
use segmented_cuckoo::{Segmented2aryCuckooKVStore, Segmented2aryScheme};

#[test]
fn smoke_ikpir_server_compose_with_frodo() {
    let store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    let mut server: IkpirServer<Segmented2aryScheme, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::default());

    let setup = server.setup();
    let make_states = || -> Vec<FrodoClientState> {
        setup
            .backend_params
            .iter()
            .zip(setup.hints.iter())
            .map(|(p, h)| FrodoPirBackend::client_setup(p, h))
            .collect()
    };
    let mut states_entry = make_states();
    let mut states_row = make_states();

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
                    &mut states_entry[j],
                    deltas,
                    HintPatchMode::EntryLevel,
                );
                FrodoPirBackend::client_patch_state(
                    &mut states_row[j],
                    deltas,
                    HintPatchMode::RowLevel,
                );
            }
        }
    }

    let final_setup = server.setup();
    for ((entry, row), srv) in states_entry
        .iter()
        .zip(states_row.iter())
        .zip(final_setup.hints.iter())
    {
        assert_eq!(
            entry.hint.data, srv.data,
            "entry-level patched state diverged from server hint"
        );
        assert_eq!(
            row.hint.data, srv.data,
            "row-level patched state diverged from server hint"
        );
    }
}
