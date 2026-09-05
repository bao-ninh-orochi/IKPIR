//! Differential proof that the mutation benches' replay harness measures
//! the same thing as a fresh setup.
//!
//! # Purpose
//!
//! `benches/server_mutation.rs` and `benches/client_mutation.rs` build one
//! `IkpirServer` per config and rewind it between mutation sequences with
//! `IkpirServer::reset_for_replay` instead of paying a full setup per
//! sequence. That is only a valid measurement if a rewound server produces
//! exactly what a freshly built one would. This file pins that, for both
//! shipped backends and every arity, at a tiny geometry:
//!
//! 1. `replay_matches_fresh_setup_*` — server A (fresh) and server B
//!    (after an unrelated sequence M′, then `reset_for_replay`) run the
//!    same update/delete sequence M: same epochs, element-wise identical
//!    `HintDeltaBundle`s (in memory and on the wire); a client bootstrapped
//!    from B's epoch-0 bundle and fed B's deltas decodes every present key
//!    and rejects every absent one, exactly like A's client. Strongest
//!    check: rewinding B a second time and replaying M leaves the hints
//!    **bit-identical** to the first replay (`B::Hint: PartialEq`, which
//!    both shipped hint types derive).
//! 2. `replay_inserts_agree_*` — inserts draw on `rand::rng()` for the
//!    cuckoo start bucket and the evicted slot, so their deltas differ run
//!    to run even from identical state. For an insert sequence the test
//!    therefore compares only success count, epoch, and the resulting key
//!    set, and checks that both clients decode every key.
//! 3. `replay_with_stale_hints_is_detected_*` — negative control: rewinding
//!    with hints captured *after* a mutation (the misuse
//!    `reset_for_replay`'s docs warn about) leaves the server's hints
//!    disagreeing with the honest replay, and a client that bootstraps from
//!    that server's post-replay `setup()` bundle decodes the stale key
//!    wrongly. This shows the checks above can fail.
//!
//! Servers A and B sample different public seeds, so their hints differ by
//! construction and are never compared across servers — only across
//! replays of one server.

use std::collections::BTreeMap;

use ikpir_client::{
    FrodoConfig, FrodoPirBackend, HintDeltaBundle, IncrementalPirBackend, IndexPirBackend,
    ResponseRewind, RewindClient, SimpleConfig, SimplePirBackend,
};
use ikpir_server::{IkpirError, IkpirServer};
use segmented_cuckoo::{
    CuckooKVStore, CuckooParams, IndexScheme, SchemeMeta, Segmented2aryScheme, Segmented3aryScheme,
    Segmented4aryScheme,
};

const BUCKET_SIZE: u32 = 4;
const FINGERPRINT_BITS: u32 = 64;
const VALUE_BITS: u32 = 64;
const PLAINTEXT_BITS: u32 = 9;
/// Reduced LWE dimension keeps setup and the per-query matvec fast; the
/// replay property does not depend on it.
const LWE_DIM: u32 = 256;
/// The kick budget the bench helpers restore after `from_cells` (which
/// resets it to 500).
const MAX_KICKS: u32 = 2_500;
/// Seed fill as a percentage of capacity: full enough that some inserts
/// kick, with room for the insert sequences to succeed.
const SEED_LOAD_PERCENT: u64 = 70;
/// Never-inserted keys, probed as absent by every client check.
const NEVER_PRESENT: [u32; 3] = [1_000_000, 1_000_001, 1_000_002];

const fn frodo_config() -> FrodoConfig {
    FrodoConfig { lwe_dim: LWE_DIM }
}
const fn simple_config() -> SimpleConfig {
    SimpleConfig {
        lwe_dim: LWE_DIM,
        sigma: 6.4,
    }
}

/// Per-arity store construction: `CuckooKVStore::new` / `from_cells` are
/// inherent per-scheme functions, not trait methods.
trait Scheme: IndexScheme + SchemeMeta + Sized + 'static {
    /// Bucket count of the fixture: 64 (2-ary), 96 = `3 · 2^5` (3-ary),
    /// 128 (4-ary), so every arity has 32 rows per segment.
    const NUM_BUCKETS: u32;
    fn new_store() -> CuckooKVStore<Self>;
    fn from_cells(cells: Vec<u32>, params: CuckooParams, num_items: u64) -> CuckooKVStore<Self>;
}

macro_rules! impl_scheme {
    ($s:ty, $nb:expr) => {
        impl Scheme for $s {
            const NUM_BUCKETS: u32 = $nb;
            fn new_store() -> CuckooKVStore<Self> {
                let mut store = CuckooKVStore::<Self>::new(
                    Self::NUM_BUCKETS,
                    BUCKET_SIZE,
                    FINGERPRINT_BITS,
                    VALUE_BITS,
                    PLAINTEXT_BITS,
                )
                .expect("fixture geometry");
                store.set_max_kicks(MAX_KICKS);
                store
            }
            fn from_cells(
                cells: Vec<u32>,
                params: CuckooParams,
                num_items: u64,
            ) -> CuckooKVStore<Self> {
                let mut store = CuckooKVStore::<Self>::from_cells(cells, params, num_items)
                    .expect("from_cells");
                store.set_max_kicks(MAX_KICKS);
                store
            }
        }
    };
}
impl_scheme!(Segmented2aryScheme, 64);
impl_scheme!(Segmented3aryScheme, 96);
impl_scheme!(Segmented4aryScheme, 128);

/// The key → value contents of a store, tracked alongside every mutation.
type Model = BTreeMap<u32, Vec<u8>>;

/// Epoch-0 snapshot: what `from_cells` needs, plus the model of what the
/// store holds.
struct Snapshot {
    cells: Vec<u32>,
    params: CuckooParams,
    num_items: u64,
    model: Model,
}

impl Snapshot {
    /// A fresh store holding exactly the snapshot cells.
    fn restore<S: Scheme>(&self) -> CuckooKVStore<S> {
        S::from_cells(self.cells.clone(), self.params, self.num_items)
    }
}

/// Deterministic 8-byte value for `key` under `salt`. Byte 0 is the salt
/// itself, so two salts that differ mod 256 always yield different values
/// — an update never degenerates into a no-op.
fn value_for(key: u32, salt: u32) -> Vec<u8> {
    let vsize = VALUE_BITS.div_ceil(8) as usize;
    (0..vsize)
        .map(|i| {
            if i == 0 {
                salt as u8
            } else {
                (key.wrapping_mul(17).wrapping_add(i as u32) & 0xFF) as u8 ^ salt as u8
            }
        })
        .collect()
}

/// Seed a fresh store to `SEED_LOAD_PERCENT` with keys `0..n` and snapshot it.
fn populate<S: Scheme>() -> Snapshot {
    let mut store = S::new_store();
    let capacity = u64::from(S::NUM_BUCKETS) * u64::from(BUCKET_SIZE);
    let target = capacity * SEED_LOAD_PERCENT / 100;
    let mut model = Model::new();
    for k in 0..target as u32 {
        let v = value_for(k, 17);
        store.insert(k.to_le_bytes(), &v).expect("seed insert");
        model.insert(k, v);
    }
    Snapshot {
        cells: store.snapshot_cells(),
        params: store.params(),
        num_items: store.num_items(),
        model,
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Insert(u32, u32),
    Update(u32, u32),
    Delete(u32),
}

/// M: updates and deletes only, on disjoint key sets — every delta is a
/// deterministic function of the cells. Keys 0..10, 20..30, 40..45 are
/// always present at the seed fill.
fn sequence_m() -> Vec<Op> {
    let mut ops = Vec::new();
    for i in 0..10 {
        ops.push(Op::Update(i, 47));
        ops.push(Op::Delete(20 + i));
    }
    for i in 0..5 {
        ops.push(Op::Update(40 + i, 49));
    }
    ops
}

/// M′: an unrelated sequence — different keys, plus a few inserts so the
/// server's cuckoo state genuinely diverges before the rewind.
fn sequence_m_prime(n: u32) -> Vec<Op> {
    let mut ops = Vec::new();
    for i in 0..6 {
        ops.push(Op::Update(50 + i, 91));
        ops.push(Op::Delete(60 + i));
    }
    for i in 0..4 {
        ops.push(Op::Insert(n + 500 + i, 91));
    }
    ops
}

/// Fresh keys past the seed range.
fn sequence_inserts(n: u32) -> Vec<Op> {
    (0..24).map(|i| Op::Insert(n + i, 31)).collect()
}

/// Apply `ops` to `server`, tracking `model`. Returns the deltas of the
/// successful ops and their count. `TableFull` is tolerated for inserts
/// only; any other error is a test failure.
fn apply<S, B>(
    server: &mut IkpirServer<S, B>,
    ops: &[Op],
    model: &mut Model,
) -> (Vec<HintDeltaBundle<B>>, usize)
where
    S: Scheme,
    B: IncrementalPirBackend,
{
    let mut deltas = Vec::with_capacity(ops.len());
    for &op in ops {
        let res = match op {
            Op::Insert(k, salt) => server.insert(&k.to_le_bytes(), &value_for(k, salt)),
            Op::Update(k, salt) => server.update(&k.to_le_bytes(), &value_for(k, salt)),
            Op::Delete(k) => server.delete(&k.to_le_bytes()),
        };
        match res {
            Ok(delta) => {
                match op {
                    Op::Insert(k, salt) | Op::Update(k, salt) => {
                        model.insert(k, value_for(k, salt));
                    }
                    Op::Delete(k) => {
                        model.remove(&k);
                    }
                }
                deltas.push(delta);
            }
            Err(IkpirError::TableFull) => {
                assert!(matches!(op, Op::Insert(..)), "TableFull from {op:?}");
            }
            Err(e) => panic!("{op:?}: {e:?}"),
        }
    }
    let n = deltas.len();
    (deltas, n)
}

/// Element-wise bundle equality: epoch, geometry, the in-memory sparse
/// deltas, and the v1 wire bytes.
fn assert_bundles_equal<B: IndexPirBackend>(
    a: &[HintDeltaBundle<B>],
    b: &[HintDeltaBundle<B>],
    what: &str,
) {
    assert_eq!(a.len(), b.len(), "{what}: delta count");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.epoch, y.epoch, "{what}: epoch of delta {i}");
        assert_eq!(x.params, y.params, "{what}: params of delta {i}");
        assert_eq!(
            x.per_segment_row_deltas, y.per_segment_row_deltas,
            "{what}: row deltas of delta {i}"
        );
        assert_eq!(x.encode(), y.encode(), "{what}: wire bytes of delta {i}");
    }
}

/// One `build_query` → `answer` → `decode` round trip.
fn lookup<S, B>(
    client: &mut RewindClient<B>,
    server: &IkpirServer<S, B>,
    key: u32,
) -> Option<Vec<u8>>
where
    S: Scheme,
    B: IncrementalPirBackend + ResponseRewind,
    B::Query: Clone,
    B::Response: Clone,
{
    let kb = key.to_le_bytes();
    let q = client.build_query(&kb);
    let r = server.answer(&q).expect("answer");
    client.decode(&kb, &q, &r).expect("decode")
}

/// Keys the sequence removed, plus the never-inserted probes.
fn absent_keys(before: &Model, after: &Model) -> Vec<u32> {
    let mut absent: Vec<u32> = before
        .keys()
        .filter(|k| !after.contains_key(k))
        .copied()
        .collect();
    absent.extend(NEVER_PRESENT);
    absent
}

/// Every key in `model` decodes to its value; every key in `absent` decodes
/// to `None`.
fn assert_client_tracks_model<S, B>(
    client: &mut RewindClient<B>,
    server: &IkpirServer<S, B>,
    model: &Model,
    absent: &[u32],
    what: &str,
) where
    S: Scheme,
    B: IncrementalPirBackend + ResponseRewind,
    B::Query: Clone,
    B::Response: Clone,
{
    assert_eq!(client.epoch(), server.epoch(), "{what}: epoch");
    for (&k, v) in model {
        assert_eq!(
            lookup(client, server, k).as_deref(),
            Some(v.as_slice()),
            "{what}: present key {k}"
        );
    }
    for &k in absent {
        assert_eq!(lookup(client, server, k), None, "{what}: absent key {k}");
    }
}

/// Body of `replay_matches_fresh_setup_*`.
fn replay_matches_fresh_setup<S, B>(config: B::Config)
where
    S: Scheme,
    B: IncrementalPirBackend + ResponseRewind,
    B::Hint: PartialEq,
    B::Query: Clone,
    B::Response: Clone,
{
    let snap = populate::<S>();
    let n = snap.num_items as u32;
    let m = sequence_m();
    let m_prime = sequence_m_prime(n);

    // Server A: fresh setup, then M.
    let mut a: IkpirServer<S, B> = IkpirServer::new(snap.restore(), config.clone());
    let bundle_a0 = a.setup();
    let mut model_a = snap.model.clone();
    let (deltas_a, ok_a) = apply(&mut a, &m, &mut model_a);
    assert_eq!(ok_a, m.len(), "M must succeed in full on the fresh server");

    // Server B: fresh setup, an unrelated M′, then rewind and M.
    let mut b: IkpirServer<S, B> = IkpirServer::new(snap.restore(), config);
    let bundle_b0 = b.setup();
    let hints0 = bundle_b0.hints.clone();
    let mut scratch = snap.model.clone();
    let _ = apply(&mut b, &m_prime, &mut scratch);
    assert!(b.epoch() > 0, "M′ must move B off epoch 0");
    b.reset_for_replay(snap.restore(), hints0.clone());
    assert_eq!(b.epoch(), 0, "reset_for_replay rewinds to epoch 0");
    let mut model_b = snap.model.clone();
    let (deltas_b, ok_b) = apply(&mut b, &m, &mut model_b);

    assert_eq!(a.epoch(), b.epoch(), "epoch after M");
    assert_eq!(ok_a, ok_b, "success count after M");
    assert_eq!(model_a, model_b, "key set after M");
    assert_bundles_equal(&deltas_a, &deltas_b, "fresh vs replay");
    let hints_b1 = b.setup().hints;

    // Strongest check: a second rewind + M reproduces the hints bit for bit.
    b.reset_for_replay(snap.restore(), hints0);
    let mut model_b2 = snap.model.clone();
    let (deltas_b2, _) = apply(&mut b, &m, &mut model_b2);
    assert_bundles_equal(&deltas_b, &deltas_b2, "replay vs replay");
    assert!(
        b.setup().hints == hints_b1,
        "hints after two replays of M are not bit-identical"
    );

    // Clients: C_A from A's epoch-0 bundle fed A's deltas; C_B from B's
    // epoch-0 bundle fed the deltas of B's current (second) replay.
    let absent = absent_keys(&snap.model, &model_a);
    let mut c_a = RewindClient::<B>::from_setup(bundle_a0);
    for d in deltas_a {
        c_a.accumulate_delta(d).expect("accumulate_delta A");
    }
    assert_client_tracks_model(&mut c_a, &a, &model_a, &absent, "client A");

    let mut c_b = RewindClient::<B>::from_setup(bundle_b0);
    for d in deltas_b2 {
        c_b.accumulate_delta(d).expect("accumulate_delta B");
    }
    assert_client_tracks_model(&mut c_b, &b, &model_b2, &absent, "client B");

    // Honest counterpart of the negative control below: a client that
    // bootstraps from the rewound server's *post-replay* bundle is right
    // too, because the replayed hints track the replayed cells.
    let mut c_b_fresh = RewindClient::<B>::from_setup(b.setup());
    assert_client_tracks_model(
        &mut c_b_fresh,
        &b,
        &model_b2,
        &absent,
        "client from B's post-replay bundle",
    );
}

/// Body of `replay_inserts_agree_*`.
fn replay_inserts_agree<S, B>(config: B::Config)
where
    S: Scheme,
    B: IncrementalPirBackend + ResponseRewind,
    B::Query: Clone,
    B::Response: Clone,
{
    let snap = populate::<S>();
    let n = snap.num_items as u32;
    let ins = sequence_inserts(n);
    let m_prime = sequence_m_prime(n);

    let mut a: IkpirServer<S, B> = IkpirServer::new(snap.restore(), config.clone());
    let bundle_a0 = a.setup();
    let mut model_a = snap.model.clone();
    let (deltas_a, ok_a) = apply(&mut a, &ins, &mut model_a);

    let mut b: IkpirServer<S, B> = IkpirServer::new(snap.restore(), config);
    let bundle_b0 = b.setup();
    let hints0 = bundle_b0.hints.clone();
    let mut scratch = snap.model.clone();
    let _ = apply(&mut b, &m_prime, &mut scratch);
    b.reset_for_replay(snap.restore(), hints0);
    let mut model_b = snap.model.clone();
    let (deltas_b, ok_b) = apply(&mut b, &ins, &mut model_b);

    assert_eq!(ok_a, ins.len(), "every insert fits at the seed fill");
    assert_eq!(ok_a, ok_b, "insert success count");
    assert_eq!(a.epoch(), b.epoch(), "epoch after inserts");
    assert_eq!(model_a, model_b, "key set after inserts");

    let absent = NEVER_PRESENT.to_vec();
    let mut c_a = RewindClient::<B>::from_setup(bundle_a0);
    for d in deltas_a {
        c_a.accumulate_delta(d).expect("accumulate_delta A");
    }
    assert_client_tracks_model(&mut c_a, &a, &model_a, &absent, "client A");

    let mut c_b = RewindClient::<B>::from_setup(bundle_b0);
    for d in deltas_b {
        c_b.accumulate_delta(d).expect("accumulate_delta B");
    }
    assert_client_tracks_model(&mut c_b, &b, &model_b, &absent, "client B");
}

/// Body of `replay_with_stale_hints_is_detected_*`.
fn replay_with_stale_hints_is_detected<S, B>(config: B::Config)
where
    S: Scheme,
    B: IncrementalPirBackend + ResponseRewind,
    B::Hint: PartialEq,
    B::Query: Clone,
    B::Response: Clone,
{
    /// Present at the seed fill and untouched by M.
    const STALE_KEY: u32 = 100;

    let snap = populate::<S>();
    let m = sequence_m();
    let stale_old = snap.model[&STALE_KEY].clone();

    let mut s: IkpirServer<S, B> = IkpirServer::new(snap.restore(), config);
    let bundle0 = s.setup();
    let hints0 = bundle0.hints.clone();

    // Honest reference: M from the fresh epoch-0 state.
    let mut model = snap.model.clone();
    let _ = apply(&mut s, &m, &mut model);
    let hints_honest = s.setup().hints;

    // Hints one mutation past epoch 0 — the wrong thing to rewind to.
    s.reset_for_replay(snap.restore(), hints0);
    s.update(&STALE_KEY.to_le_bytes(), &value_for(STALE_KEY, 200))
        .expect("stale update");
    assert_eq!(s.epoch(), 1);
    let hints_stale = s.setup().hints;

    // The misuse: snapshot cells (old value of STALE_KEY) under hints that
    // already fold in its new value.
    s.reset_for_replay(snap.restore(), hints_stale);
    let mut model_misuse = snap.model.clone();
    let (deltas, _) = apply(&mut s, &m, &mut model_misuse);
    assert_eq!(model_misuse, model);
    assert!(
        s.setup().hints != hints_honest,
        "a stale-hint replay must not reproduce the honest hints"
    );

    // Its observable consequence: a client that bootstraps from this
    // server's post-replay bundle holds a hint that does not track the
    // cells. The server answers from the true cells, the untouched
    // fingerprint cells still match, and the value cells decode as noise —
    // so the stale key comes back wrong (or not at all).
    let mut poisoned = RewindClient::<B>::from_setup(s.setup());
    assert_ne!(
        lookup(&mut poisoned, &s, STALE_KEY).as_deref(),
        Some(stale_old.as_slice()),
        "a client bootstrapped from the poisoned bundle decoded the stale key \
         correctly — the misuse went undetected"
    );

    // Whereas a client that stays in sync from the epoch-0 bundle through
    // the replayed deltas is unaffected: `answer` never reads the server's
    // hints, and deltas are folded from the cells alone. The misuse is
    // invisible to the delta-fed clients the benches time and visible only
    // through `setup()` — the bundle `setup_bundle_bytes` measures.
    let mut synced = RewindClient::<B>::from_setup(bundle0);
    for d in deltas {
        synced.accumulate_delta(d).expect("accumulate_delta");
    }
    assert_eq!(
        lookup(&mut synced, &s, STALE_KEY).as_deref(),
        Some(stale_old.as_slice()),
        "a delta-fed client must be unaffected by the server's stale hints"
    );
}

macro_rules! instantiate {
    ($($name:ident: $body:ident < $s:ty, $b:ty > ($cfg:expr);)*) => {
        $(
            #[test]
            fn $name() {
                $body::<$s, $b>($cfg);
            }
        )*
    };
}

instantiate! {
    replay_matches_fresh_setup_frodo_2ary:
        replay_matches_fresh_setup<Segmented2aryScheme, FrodoPirBackend>(frodo_config());
    replay_matches_fresh_setup_frodo_3ary:
        replay_matches_fresh_setup<Segmented3aryScheme, FrodoPirBackend>(frodo_config());
    replay_matches_fresh_setup_frodo_4ary:
        replay_matches_fresh_setup<Segmented4aryScheme, FrodoPirBackend>(frodo_config());
    replay_matches_fresh_setup_simple_2ary:
        replay_matches_fresh_setup<Segmented2aryScheme, SimplePirBackend>(simple_config());
    replay_matches_fresh_setup_simple_3ary:
        replay_matches_fresh_setup<Segmented3aryScheme, SimplePirBackend>(simple_config());
    replay_matches_fresh_setup_simple_4ary:
        replay_matches_fresh_setup<Segmented4aryScheme, SimplePirBackend>(simple_config());

    replay_inserts_agree_frodo_2ary:
        replay_inserts_agree<Segmented2aryScheme, FrodoPirBackend>(frodo_config());
    replay_inserts_agree_frodo_3ary:
        replay_inserts_agree<Segmented3aryScheme, FrodoPirBackend>(frodo_config());
    replay_inserts_agree_frodo_4ary:
        replay_inserts_agree<Segmented4aryScheme, FrodoPirBackend>(frodo_config());
    replay_inserts_agree_simple_2ary:
        replay_inserts_agree<Segmented2aryScheme, SimplePirBackend>(simple_config());
    replay_inserts_agree_simple_3ary:
        replay_inserts_agree<Segmented3aryScheme, SimplePirBackend>(simple_config());
    replay_inserts_agree_simple_4ary:
        replay_inserts_agree<Segmented4aryScheme, SimplePirBackend>(simple_config());

    replay_with_stale_hints_is_detected_frodo_2ary:
        replay_with_stale_hints_is_detected<Segmented2aryScheme, FrodoPirBackend>(frodo_config());
    replay_with_stale_hints_is_detected_frodo_3ary:
        replay_with_stale_hints_is_detected<Segmented3aryScheme, FrodoPirBackend>(frodo_config());
    replay_with_stale_hints_is_detected_frodo_4ary:
        replay_with_stale_hints_is_detected<Segmented4aryScheme, FrodoPirBackend>(frodo_config());
    replay_with_stale_hints_is_detected_simple_2ary:
        replay_with_stale_hints_is_detected<Segmented2aryScheme, SimplePirBackend>(simple_config());
    replay_with_stale_hints_is_detected_simple_3ary:
        replay_with_stale_hints_is_detected<Segmented3aryScheme, SimplePirBackend>(simple_config());
    replay_with_stale_hints_is_detected_simple_4ary:
        replay_with_stale_hints_is_detected<Segmented4aryScheme, SimplePirBackend>(simple_config());
}
