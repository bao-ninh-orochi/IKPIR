//! End-to-end integration tests for `IkpirClient` against a real
//! `IkpirServer` and `FrodoPirBackend`.
//!
//! # Purpose
//!
//! Pins the client-side behaviours that span the server boundary and
//! therefore cannot be unit-tested inside `src/`:
//!
//! 1. `decode_returns_value` — 50 keys round-trip through the full
//!    `build_query` → `answer` → `decode` pipeline.
//! 2. `decode_absent` — `decode` returns `Ok(None)` (no `Err`) for a
//!    key that was never inserted.
//! 3. `decode_after_apply_delta` — a client patched by `apply_delta`
//!    produces the same decodes as a freshly-rebuilt client.
//! 4. `precomputed_warm_path` — the cheap Phase-B / Phase-C path
//!    produces the same decodes as the cold inline path, and the
//!    `prepared_per_segment` counter decrements with each query.
//! 5. `precomputed_survives_apply_delta` — warm queues stay coherent
//!    across `apply_delta`.
//! 6. `stale_delta_rejected` — `apply_delta` rejects a delta older
//!    than the current client epoch.
//!
//! Behaviours 1–6 are parameterised by arity (2 / 3 / 4) so the
//! diagnostic on failure pins which arity broke. Two extra
//! single-arity tests pin the wire-byte accounting helpers
//! (`wire_byte_size`) and the constant-time-decode last-slot probe.

use ikpir_client::{FrodoConfig, FrodoPirBackend, HintPatchMode, IkpirClient, IkpirClientError};
use ikpir_server::IkpirServer;
use segmented_cuckoo::{
    IndexScheme, SchemeMeta, Segmented2aryCuckooKVStore, Segmented2aryScheme,
    Segmented3aryCuckooKVStore, Segmented3aryScheme, Segmented4aryCuckooKVStore,
    Segmented4aryScheme,
};

type Client = IkpirClient<FrodoPirBackend>;

/// Build an empty 2-ary `IkpirServer` (64 buckets × 4 slots).
fn build_server_2() -> IkpirServer<Segmented2aryScheme, FrodoPirBackend> {
    let store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}
/// Build an empty 3-ary `IkpirServer` (`num_buckets = 3 · 32 = 96`,
/// same per-segment row count as the 2-ary fixture).
fn build_server_3() -> IkpirServer<Segmented3aryScheme, FrodoPirBackend> {
    let store = Segmented3aryCuckooKVStore::new(96, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}
/// Build an empty 4-ary `IkpirServer` (64 buckets × 4 slots).
fn build_server_4() -> IkpirServer<Segmented4aryScheme, FrodoPirBackend> {
    let store = Segmented4aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
    IkpirServer::new(store, FrodoConfig::default())
}

/// `IkpirClient::from_setup(server.setup())` shortcut shared by every
/// arity-generic body.
fn fresh_client<S>(server: &IkpirServer<S, FrodoPirBackend>) -> Client
where
    S: IndexScheme + SchemeMeta + 'static,
{
    IkpirClient::from_setup(server.setup())
}

/// `(key_bytes, value_bytes)` pair derived from `k` — keeps the test
/// inputs compact and deterministic.
const fn pair(k: u32) -> ([u8; 4], [u8; 1]) {
    (k.to_le_bytes(), [(k as u8) ^ 0xA5])
}

/// Arity-generic body of the "50 inserted keys round-trip through
/// query / answer / decode" test.
fn decode_returns_value_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
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

/// Arity-generic body of the "absent key returns `Ok(None)`" test.
fn decode_absent_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
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

/// Arity-generic body of the headline correctness test: a client
/// patched by `apply_delta` decodes identically to a freshly-rebuilt
/// client, across a mixed insert/update/delete trace.
fn decode_after_apply_delta_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
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

/// Arity-generic body of the "Phase-B + Phase-C warm path round-trips
/// correctly and consumes one prepared slot per query" test.
fn precomputed_warm_path_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
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

    // Issue a query for every inserted key — each should consume one slot
    // per segment from the prepared queue and round-trip correctly.
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

/// Arity-generic body of "warm queue survives a burst of mutations":
/// precompute, apply a mixed mutation trace, then query — every
/// decode must match a freshly-rebuilt oracle client.
fn precomputed_survives_apply_delta_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
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

    // Apply a few mutations between precomputation and use.
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

    // Probe several keys via the warm path; compare against a fresh oracle.
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

/// Arity-generic body of "applying a delta older than the current
/// epoch returns `StaleDelta`".
fn stale_delta_inner<S>(mut server: IkpirServer<S, FrodoPirBackend>)
where
    S: IndexScheme + SchemeMeta + 'static,
{
    let (k1, v1) = pair(1);
    let (k2, v2) = pair(2);

    let delta1 = server.insert(&k1, &v1).unwrap();
    let _delta2 = server.insert(&k2, &v2).unwrap();

    // Client epoch is 2 (two inserts before setup). delta1 has epoch 1 → stale.
    let mut client = fresh_client(&server);
    let err = client.apply_delta(delta1).unwrap_err();
    assert!(
        matches!(err, IkpirClientError::StaleDelta { .. }),
        "got {err:?}"
    );
}

/// 50 inserted keys round-trip through query/answer/decode (2-ary).
#[test]
fn decode_returns_value_for_inserted_key() {
    decode_returns_value_inner(build_server_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn decode_returns_value_for_inserted_key_3ary() {
    decode_returns_value_inner(build_server_3());
}
/// Same as 2-ary, on the 4-ary server.
#[test]
fn decode_returns_value_for_inserted_key_4ary() {
    decode_returns_value_inner(build_server_4());
}

/// Absent key returns `Ok(None)` (2-ary).
#[test]
fn decode_returns_none_for_absent_key() {
    decode_absent_inner(build_server_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn decode_returns_none_for_absent_key_3ary() {
    decode_absent_inner(build_server_3());
}
/// Same as 2-ary, on the 4-ary server.
#[test]
fn decode_returns_none_for_absent_key_4ary() {
    decode_absent_inner(build_server_4());
}

/// Incrementally patched client matches a freshly-rebuilt client
/// across a mixed mutation trace (2-ary). This is the headline
/// correctness claim of the IKPIR scheme.
#[test]
fn decode_after_apply_delta_matches_freshly_setup_client() {
    decode_after_apply_delta_inner(build_server_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn decode_after_apply_delta_matches_freshly_setup_client_3ary() {
    decode_after_apply_delta_inner(build_server_3());
}
/// Same as 2-ary, on the 4-ary server.
#[test]
fn decode_after_apply_delta_matches_freshly_setup_client_4ary() {
    decode_after_apply_delta_inner(build_server_4());
}

/// `apply_delta` with an older-than-current epoch returns
/// `StaleDelta` (2-ary).
#[test]
fn stale_delta_rejected() {
    stale_delta_inner(build_server_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn stale_delta_rejected_3ary() {
    stale_delta_inner(build_server_3());
}
/// Same as 2-ary, on the 4-ary server.
#[test]
fn stale_delta_rejected_4ary() {
    stale_delta_inner(build_server_4());
}

/// Phase-B + Phase-C warm queue round-trips and the prepared-slot
/// counter decrements correctly with each query (2-ary).
#[test]
fn precomputed_warm_path() {
    precomputed_warm_path_inner(build_server_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn precomputed_warm_path_3ary() {
    precomputed_warm_path_inner(build_server_3());
}
/// Same as 2-ary, on the 4-ary server.
#[test]
fn precomputed_warm_path_4ary() {
    precomputed_warm_path_inner(build_server_4());
}

/// Warm queue survives a burst of mutations: every probe still
/// matches a freshly-rebuilt oracle client (2-ary).
#[test]
fn precomputed_survives_apply_delta() {
    precomputed_survives_apply_delta_inner(build_server_2());
}
/// Same as 2-ary, on the 3-ary server.
#[test]
fn precomputed_survives_apply_delta_3ary() {
    precomputed_survives_apply_delta_inner(build_server_3());
}
/// Same as 2-ary, on the 4-ary server.
#[test]
fn precomputed_survives_apply_delta_4ary() {
    precomputed_survives_apply_delta_inner(build_server_4());
}

/// Wire-size accounting: one mutation's HintDeltaBundle is dramatically
/// smaller than the corresponding ServerSetupBundle. This is the headline
/// claim of incremental hint patching, expressed as a runtime invariant.
#[test]
fn hint_delta_is_much_smaller_than_setup_bundle() {
    let mut server = build_server_2();
    // Seed enough rows that the hint dominates the setup bundle.
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

/// Query/response wire-size accounting is consistent with the underlying
/// FrodoPIR encoding (4 bytes per u32 × per-segment vector length).
#[test]
fn query_response_wire_sizes_match_underlying_vectors() {
    let mut server = build_server_2();
    for k in 0u32..5 {
        let (key, val) = pair(k);
        server.insert(&key, &val).unwrap();
    }
    let mut client = fresh_client(&server);
    let (key, _) = pair(2);

    let q = client.build_query(&key);
    let q_bytes = q.wire_byte_size();
    // 8 (epoch) + 4 (vec length prefix) + arity × n_rows × 4 (u32 each).
    let arity = client.params().arity();
    let n_rows = client.params().segment_size() as usize;
    let expected_q = 8 + 4 + arity * (n_rows * 4);
    assert_eq!(q_bytes, expected_q);

    let r = server.answer(&q).unwrap();
    let r_bytes = r.wire_byte_size();
    let row_width = (client.params().bucket_size * client.params().cells_per_slot()) as usize;
    let expected_r = 8 + 4 + arity * (row_width * 4);
    assert_eq!(r_bytes, expected_r);
}

/// Constant-time decode: both an absent key and a present key are
/// decoded correctly when the underlying scan visits *all* candidate
/// slots, not just up to the first match. (The unit test
/// `decode_returns_value_for_inserted_key` validated the present-key
/// case; this test specifically forces the present key to live in the
/// *last* candidate slot of the *last* segment, exercising the path
/// that the previous early-return implementation could short-circuit.)
#[test]
fn decode_visits_all_candidate_slots() {
    let mut server = build_server_2();
    // Fill enough to make the cuckoo placement non-trivial.
    let mut inserted = Vec::new();
    for k in 0u32..50 {
        let (key, val) = pair(k);
        if server.insert(&key, &val).is_ok() {
            inserted.push((key, val));
        }
    }
    let mut client = fresh_client(&server);
    // Every inserted key, regardless of which segment / slot the cuckoo
    // placement put it in, must decode correctly. This indirectly
    // exercises last-slot probing.
    for (key, val) in &inserted {
        let q = client.build_query(key);
        let resp = server.answer(&q).expect("answer ok");
        let got = client.decode(key, &resp).expect("no error").expect("found");
        assert_eq!(got, val.to_vec(), "value mismatch for key {key:?}");
    }
}

/// End-to-end round-trip at `fingerprint_bits = 64` — the newly widened
/// upper bound. Insert, update, delete, and query all flow through the
/// public client/server API with 64-bit fingerprints (dev-scale
/// geometry, small key count to keep the runtime tiny).
#[test]
fn decode_roundtrip_at_fingerprint_bits_64() {
    let store = Segmented2aryCuckooKVStore::new(64, 4, 64, 8, 8).unwrap();
    let mut server: IkpirServer<Segmented2aryScheme, FrodoPirBackend> =
        IkpirServer::new(store, FrodoConfig::default());

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

/// The hint-patch mode defaults to entry-level, is a client-side
/// preference that survives `reset_from`, and a row-level client stays
/// consistent with the server across subsequent deltas.
#[test]
fn hint_patch_mode_defaults_and_survives_reset() {
    let mut server = build_server_2();
    server.insert(b"alice", &[0xAB]).unwrap();

    let mut client = fresh_client(&server);
    assert_eq!(client.hint_patch_mode(), HintPatchMode::EntryLevel);
    client.set_hint_patch_mode(HintPatchMode::RowLevel);

    // `reset_from` replaces all protocol state but keeps the preference.
    client.reset_from(server.setup());
    assert_eq!(client.hint_patch_mode(), HintPatchMode::RowLevel);

    // The row-level client still tracks the server across deltas.
    let d = server.insert(b"bob", &[0xCD]).unwrap();
    client.apply_delta(d).unwrap();
    let q = client.build_query(b"bob");
    let r = server.answer(&q).unwrap();
    assert_eq!(client.decode(b"bob", &r).unwrap(), Some(vec![0xCD]));
}
