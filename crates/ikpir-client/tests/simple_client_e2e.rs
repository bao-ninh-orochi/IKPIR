//! End-to-end integration tests for `IkpirClient` against a real
//! `IkpirServer` and `SimplePirBackend`.
//!
//! # Purpose
//!
//! Mirror of `client_e2e.rs`: pins the same client-side behaviours, but
//! against `SimplePirBackend` so the reshape + sparse-patch math gets
//! exercised end-to-end through the public client API.

use ikpir_client::{IkpirClient, IkpirClientError, SimpleConfig, SimplePirBackend};
use ikpir_server::IkpirServer;
use segmented_cuckoo::{
    IndexScheme, SchemeMeta, Segmented2aryCuckooKVStore, Segmented2aryScheme,
    Segmented3aryCuckooKVStore, Segmented3aryScheme, Segmented4aryCuckooKVStore,
    Segmented4aryScheme,
};

type Client = IkpirClient<SimplePirBackend>;

/// Reduced LWE dimension keeps the per-segment matvec fast in tests.
const fn test_config() -> SimpleConfig {
    SimpleConfig {
        lwe_dim: 256,
        sigma: 6.4,
    }
}

fn build_server_2() -> IkpirServer<Segmented2aryScheme, SimplePirBackend> {
    let store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, test_config())
}
fn build_server_3() -> IkpirServer<Segmented3aryScheme, SimplePirBackend> {
    let store = Segmented3aryCuckooKVStore::new(96, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, test_config())
}
fn build_server_4() -> IkpirServer<Segmented4aryScheme, SimplePirBackend> {
    let store = Segmented4aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, test_config())
}

fn fresh_client<S>(server: &IkpirServer<S, SimplePirBackend>) -> Client
where
    S: IndexScheme + SchemeMeta + 'static,
{
    IkpirClient::from_setup(server.setup())
}

const fn pair(k: u32) -> ([u8; 4], [u8; 1]) {
    (k.to_le_bytes(), [(k as u8) ^ 0xA5])
}

fn decode_returns_value_inner<S>(mut server: IkpirServer<S, SimplePirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    let mut inserted = Vec::new();
    for k in 0u32..50 {
        let (key, val) = pair(k);
        server.insert(&key, &val).expect("insert ok");
        inserted.push((key, val));
    }
    let mut client = fresh_client(&server);

    for (key, val) in &inserted {
        let q = client.build_query(key);
        let resp = server.answer(&q).expect("answer ok");
        let got = client.decode(key, &resp).expect("no error").expect("found");
        assert_eq!(got, val.to_vec(), "value mismatch for key {key:?}");
    }
}

fn decode_absent_inner<S>(mut server: IkpirServer<S, SimplePirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    for k in 0u32..20 {
        let (key, val) = pair(k);
        server.insert(&key, &val).unwrap();
    }
    let mut client = fresh_client(&server);

    let absent = 9_999u32.to_le_bytes();
    let q = client.build_query(&absent);
    let resp = server.answer(&q).expect("answer ok");
    assert_eq!(client.decode(&absent, &resp).expect("no error"), None);
}

fn decode_after_apply_delta_inner<S>(mut server: IkpirServer<S, SimplePirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    let mut inc: Client = fresh_client(&server);

    for k in 0u32..10 {
        let (key, val) = pair(k);
        let delta = server.insert(&key, &val).unwrap();
        inc.apply_delta(delta).unwrap();
    }

    let ops = [
        ("ins", 10u32, 0x10u8),
        ("ins", 11, 0x11),
        ("upd", 3, 0x99),
        ("del", 5u32, 0u8),
        ("ins", 12, 0x12),
        ("upd", 7, 0xAA),
        ("del", 2u32, 0u8),
        ("ins", 100, 0x42),
    ];
    for (op, k, v) in ops {
        let key = k.to_le_bytes();
        let delta = match op {
            "ins" => server.insert(&key, &[v]).unwrap(),
            "upd" => server.update(&key, &[v]).unwrap(),
            "del" => server.delete(&key).unwrap(),
            _ => unreachable!(),
        };
        inc.apply_delta(delta).unwrap();

        let mut oracle: Client = fresh_client(&server);

        for probe in 0u32..15 {
            let pkey = probe.to_le_bytes();

            let q_inc = inc.build_query(&pkey);
            let r_inc = server.answer(&q_inc).unwrap();
            let v_inc = inc.decode(&pkey, &r_inc).unwrap();

            let q_oracle = oracle.build_query(&pkey);
            let r_oracle = server.answer(&q_oracle).unwrap();
            let v_oracle = oracle.decode(&pkey, &r_oracle).unwrap();

            assert_eq!(
                v_inc, v_oracle,
                "decode diverged on probe {probe} after op {op}/{k}",
            );
        }
    }
}

fn precomputed_warm_path_inner<S>(mut server: IkpirServer<S, SimplePirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    let mut inserted = Vec::new();
    for k in 0u32..30 {
        let (key, val) = pair(k);
        server.insert(&key, &val).expect("insert ok");
        inserted.push((key, val));
    }
    let mut client = fresh_client(&server);

    let n = inserted.len() as u32;
    client.precompute_queries(n);
    client.precompute_decodes();

    let arity = client.params().arity();
    assert_eq!(
        client.prepared_per_segment(),
        vec![n as usize; arity],
        "precompute_queries(n) must fill n slots in every segment"
    );
    assert_eq!(client.in_flight_per_segment(), vec![0; arity]);

    for (i, (key, val)) in inserted.iter().enumerate() {
        let q = client.build_query(key);
        let resp = server.answer(&q).expect("answer ok");
        let got = client.decode(key, &resp).expect("no error").expect("found");
        assert_eq!(got, val.to_vec(), "value mismatch on lookup {i}");

        let remaining = (n - 1) - i as u32;
        assert_eq!(
            client.prepared_per_segment(),
            vec![remaining as usize; arity],
            "prepared count must decrement with each query (lookup {i})"
        );
    }
}

fn precomputed_survives_apply_delta_inner<S>(mut server: IkpirServer<S, SimplePirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    for k in 0u32..15 {
        let (key, val) = pair(k);
        server.insert(&key, &val).unwrap();
    }
    let mut client = fresh_client(&server);
    client.precompute_queries(20);
    client.precompute_decodes();

    let ops = [
        ("ins", 50u32, 0xCC),
        ("upd", 5, 0xDD),
        ("del", 2u32, 0u8),
        ("ins", 51, 0x77),
    ];
    for (op, k, v) in ops {
        let key = k.to_le_bytes();
        let delta = match op {
            "ins" => server.insert(&key, &[v]).unwrap(),
            "upd" => server.update(&key, &[v]).unwrap(),
            "del" => server.delete(&key).unwrap(),
            _ => unreachable!(),
        };
        client.apply_delta(delta).unwrap();
    }

    let mut oracle = fresh_client(&server);
    for k in [0u32, 5, 10, 50, 51, 999] {
        let key = k.to_le_bytes();
        let q_warm = client.build_query(&key);
        let r_warm = server.answer(&q_warm).unwrap();
        let v_warm = client.decode(&key, &r_warm).unwrap();

        let q_oracle = oracle.build_query(&key);
        let r_oracle = server.answer(&q_oracle).unwrap();
        let v_oracle = oracle.decode(&key, &r_oracle).unwrap();

        assert_eq!(v_warm, v_oracle, "warm-path decode diverged on key {k}");
    }
}

fn stale_delta_inner<S>(mut server: IkpirServer<S, SimplePirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    let (k1, v1) = pair(1);
    let (k2, v2) = pair(2);

    let delta1 = server.insert(&k1, &v1).unwrap();
    let _delta2 = server.insert(&k2, &v2).unwrap();

    let mut client = fresh_client(&server);
    let err = client.apply_delta(delta1).unwrap_err();
    assert!(
        matches!(err, IkpirClientError::StaleDelta { .. }),
        "got {err:?}"
    );
}

#[test]
fn decode_returns_value_for_inserted_key() {
    decode_returns_value_inner(build_server_2());
}
#[test]
fn decode_returns_value_for_inserted_key_3ary() {
    decode_returns_value_inner(build_server_3());
}
#[test]
fn decode_returns_value_for_inserted_key_4ary() {
    decode_returns_value_inner(build_server_4());
}

#[test]
fn decode_returns_none_for_absent_key() {
    decode_absent_inner(build_server_2());
}
#[test]
fn decode_returns_none_for_absent_key_3ary() {
    decode_absent_inner(build_server_3());
}
#[test]
fn decode_returns_none_for_absent_key_4ary() {
    decode_absent_inner(build_server_4());
}

#[test]
fn decode_after_apply_delta_matches_freshly_setup_client() {
    decode_after_apply_delta_inner(build_server_2());
}
#[test]
fn decode_after_apply_delta_matches_freshly_setup_client_3ary() {
    decode_after_apply_delta_inner(build_server_3());
}
#[test]
fn decode_after_apply_delta_matches_freshly_setup_client_4ary() {
    decode_after_apply_delta_inner(build_server_4());
}

#[test]
fn stale_delta_rejected() {
    stale_delta_inner(build_server_2());
}
#[test]
fn stale_delta_rejected_3ary() {
    stale_delta_inner(build_server_3());
}
#[test]
fn stale_delta_rejected_4ary() {
    stale_delta_inner(build_server_4());
}

#[test]
fn precomputed_warm_path() {
    precomputed_warm_path_inner(build_server_2());
}
#[test]
fn precomputed_warm_path_3ary() {
    precomputed_warm_path_inner(build_server_3());
}
#[test]
fn precomputed_warm_path_4ary() {
    precomputed_warm_path_inner(build_server_4());
}

#[test]
fn precomputed_survives_apply_delta() {
    precomputed_survives_apply_delta_inner(build_server_2());
}
#[test]
fn precomputed_survives_apply_delta_3ary() {
    precomputed_survives_apply_delta_inner(build_server_3());
}
#[test]
fn precomputed_survives_apply_delta_4ary() {
    precomputed_survives_apply_delta_inner(build_server_4());
}

/// Wire-size accounting: one mutation's HintDeltaBundle is dramatically
/// smaller than the corresponding ServerSetupBundle. SimplePIR-specific:
/// the setup bundle hint is larger than FrodoPIR's (square reshape), so
/// the ratio is even more pronounced.
#[test]
fn hint_delta_is_much_smaller_than_setup_bundle() {
    let mut server = build_server_2();
    for k in 0u32..30 {
        let (key, val) = pair(k);
        server.insert(&key, &val).unwrap();
    }
    let setup = server.setup();
    let setup_bytes = setup.wire_byte_size();

    let (key, val) = pair(100);
    let delta = server.insert(&key, &val).unwrap();
    let delta_bytes = delta.wire_byte_size();

    assert!(
        delta_bytes * 20 < setup_bytes,
        "hint delta ({delta_bytes} B) must be >20× smaller than setup ({setup_bytes} B)",
    );
}

/// Query/response wire-size accounting reflects the SimplePIR-specific
/// shape: query length = `reshape_rows`, response length =
/// `reshape_row_width = k · row_width`.
#[test]
fn query_response_wire_sizes_match_underlying_vectors() {
    let mut server = build_server_2();
    for k in 0u32..5 {
        let (key, val) = pair(k);
        server.insert(&key, &val).unwrap();
    }
    let setup = server.setup();
    // Compute the SimplePIR reshape dims off the first segment's server params.
    let sp0 = &setup.backend_params[0];
    let reshape_rows = sp0.reshape_rows as usize;
    let reshape_row_width = sp0.reshape_row_width as usize;
    let arity = setup.params.arity();

    let mut client = fresh_client(&server);
    let (key, _) = pair(2);

    let q = client.build_query(&key);
    let q_bytes = q.wire_byte_size();
    // 8 (epoch) + 4 (vec length prefix) + arity × reshape_rows × 4 (u32 each).
    let expected_q = 8 + 4 + arity * (reshape_rows * 4);
    assert_eq!(q_bytes, expected_q);

    let r = server.answer(&q).unwrap();
    let r_bytes = r.wire_byte_size();
    let expected_r = 8 + 4 + arity * (reshape_row_width * 4);
    assert_eq!(r_bytes, expected_r);
}

/// Constant-time decode visits all candidate slots — same coverage as
/// the FrodoPIR test but on SimplePIR. Every inserted key must decode,
/// regardless of which candidate slot the cuckoo placement chose.
#[test]
fn decode_visits_all_candidate_slots() {
    let mut server = build_server_2();
    let mut inserted = Vec::new();
    for k in 0u32..50 {
        let (key, val) = pair(k);
        if server.insert(&key, &val).is_ok() {
            inserted.push((key, val));
        }
    }
    let mut client = fresh_client(&server);
    for (key, val) in &inserted {
        let q = client.build_query(key);
        let resp = server.answer(&q).expect("answer ok");
        let got = client.decode(key, &resp).expect("no error").expect("found");
        assert_eq!(got, val.to_vec(), "value mismatch for key {key:?}");
    }
}

/// End-to-end round-trip at `fingerprint_bits = 64` — the newly widened
/// upper bound. SimplePIR mirror of the FrodoPIR test of the same name:
/// insert, update, delete, and query all flow through the public
/// client/server API with 64-bit fingerprints (dev-scale geometry, small
/// key count to keep the runtime tiny).
#[test]
fn decode_roundtrip_at_fingerprint_bits_64() {
    let store = Segmented2aryCuckooKVStore::new(64, 4, 64, 8, 8).unwrap();
    let mut server: IkpirServer<Segmented2aryScheme, SimplePirBackend> =
        IkpirServer::new(store, test_config());

    let mut inserted = Vec::new();
    for k in 0u32..10 {
        let (key, val) = pair(k);
        server.insert(&key, &val).expect("insert ok");
        inserted.push((key, val));
    }
    let mut client: Client = fresh_client(&server);

    // Insert: every inserted key round-trips through query/answer/decode.
    for (key, val) in &inserted {
        let q = client.build_query(key);
        let resp = server.answer(&q).expect("answer ok");
        let got = client.decode(key, &resp).expect("no error").expect("found");
        assert_eq!(got, val.to_vec(), "value mismatch for key {key:?}");
    }

    // Update: the client observes the new value after apply_delta.
    let (upd_key, _) = pair(3);
    let new_val = [0x77u8];
    let delta = server.update(&upd_key, &new_val).unwrap();
    client.apply_delta(delta).unwrap();
    let q = client.build_query(&upd_key);
    let resp = server.answer(&q).unwrap();
    assert_eq!(
        client.decode(&upd_key, &resp).unwrap(),
        Some(new_val.to_vec()),
        "update not observed at fingerprint_bits = 64"
    );

    // Delete: the client observes the key is gone after apply_delta.
    let (del_key, _) = pair(5);
    let delta = server.delete(&del_key).unwrap();
    client.apply_delta(delta).unwrap();
    let q = client.build_query(&del_key);
    let resp = server.answer(&q).unwrap();
    assert_eq!(
        client.decode(&del_key, &resp).unwrap(),
        None,
        "delete not observed at fingerprint_bits = 64"
    );
}
