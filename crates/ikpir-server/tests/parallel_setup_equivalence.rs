//! Integration test: the optimized setup path is a drop-in for the
//! reference one, at the *protocol* level.
//!
//! The backend unit tests already pin bit-exactness of the two kernels
//! (`compute_hint_parallel` / `sample_a_parallel` vs their reference
//! twins). What this file pins is the property callers actually rely
//! on: a server or client bootstrapped on either path interoperates
//! with a peer bootstrapped on **either** path, and keeps doing so
//! across mutations and a full rebuild.
//!
//! Both shipped backends are covered; the bodies are generic over `B`
//! so the two `#[test]`s below stay a one-line instantiation each.
//!
//! Note that a *direct* state comparison between a reference-built and
//! a parallel-built server is meaningless: `server_setup` samples a
//! fresh public seed per call, so two setups of the same store differ
//! whichever path they took. Equality is therefore asserted against the
//! seed each setup actually produced (the backend unit tests), and
//! interoperability is asserted here.

use ikpir_server::{
    FrodoConfig, FrodoPirBackend, IkpirServer, IncrementalPirBackend, IndexPirBackend,
    ParallelSetupBackend, SimpleConfig, SimplePirBackend,
};
use segmented_cuckoo::{Segmented2aryCuckooKVStore, Segmented2aryScheme};

use ikpir_client::IkpirClient;

/// Small but non-degenerate store: 64 buckets × 4 slots, 12-bit
/// fingerprints, 8-bit values, `plaintext_bits = 8`.
fn store() -> Segmented2aryCuckooKVStore {
    Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap()
}

/// Look `key` up end to end and assert it decodes to `expected`.
fn assert_lookup<B>(
    server: &IkpirServer<Segmented2aryScheme, B>,
    client: &mut IkpirClient<B>,
    key: &[u8],
    expected: &[u8],
) where
    B: IndexPirBackend,
    B::Query: Clone,
    B::Response: Clone,
{
    let q = client.build_query(key);
    let r = server.answer(&q).expect("answer");
    let got = client.decode(key, &r).expect("decode");
    assert_eq!(
        got.as_deref(),
        Some(expected),
        "lookup of {key:?} did not return {expected:?}"
    );
}

/// Every combination of `{reference, parallel}` server bootstrap ×
/// `{reference, parallel}` client bootstrap answers and decodes
/// correctly — including after a mutation stream and after a full
/// rebuild on the optimized path.
fn paths_interoperate<B>(config: B::Config)
where
    // `B: Clone` is what makes the wire bundles cloneable, so the same
    // delta / setup snapshot can be handed to both clients.
    B: ParallelSetupBackend + IncrementalPirBackend + Clone,
    B::Query: Clone,
    B::Response: Clone,
{
    // ── Server on the optimized path, clients on both ────────────────
    let mut server: IkpirServer<Segmented2aryScheme, B> =
        IkpirServer::new_parallel(store(), config.clone());
    server.insert(b"alpha", b"A").unwrap();
    server.insert(b"beta", b"B").unwrap();

    let mut reference_client = IkpirClient::<B>::from_setup(server.setup());
    let mut parallel_client = IkpirClient::<B>::from_setup_parallel(server.setup());
    assert_lookup(&server, &mut reference_client, b"alpha", b"A");
    assert_lookup(&server, &mut parallel_client, b"alpha", b"A");

    // ── Mutations keep both clients in lock-step with that server ────
    let deltas = [
        server.update(b"alpha", b"X").unwrap(),
        server.insert(b"gamma", b"C").unwrap(),
        server.delete(b"beta").unwrap(),
    ];
    for delta in deltas {
        reference_client.apply_delta(delta.clone()).unwrap();
        parallel_client.apply_delta(delta).unwrap();
    }
    assert_lookup(&server, &mut reference_client, b"alpha", b"X");
    assert_lookup(&server, &mut parallel_client, b"gamma", b"C");

    // ── A rebuild on the optimized path resyncs both clients ─────────
    let bundle = server.full_rebuild_parallel();
    reference_client.reset_from(bundle.clone());
    parallel_client.reset_from_parallel(bundle);
    assert_lookup(&server, &mut reference_client, b"gamma", b"C");
    assert_lookup(&server, &mut parallel_client, b"alpha", b"X");

    // ── Server on the reference path, client on the optimized one ────
    let mut reference_server: IkpirServer<Segmented2aryScheme, B> =
        IkpirServer::new(store(), config);
    reference_server.insert(b"delta", b"D").unwrap();
    let mut client = IkpirClient::<B>::from_setup_parallel(reference_server.setup());
    assert_lookup(&reference_server, &mut client, b"delta", b"D");
}

#[test]
fn frodo_setup_paths_interoperate() {
    paths_interoperate::<FrodoPirBackend>(FrodoConfig::default());
}

#[test]
fn simple_setup_paths_interoperate() {
    paths_interoperate::<SimplePirBackend>(SimpleConfig::default());
}
