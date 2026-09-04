//! Response-rewind equivalence: the production `IkpirClient` (response-rewind,
//! the client's sole update strategy) decodes **bit-identically** to the
//! bench-only `HintPatchClient` comparator and to a fresh setup, for both
//! shipped backends and every arity. Feature-gated behind `hint-patch-bench`
//! (disabled by default) since it is the regression net the paper's §6.2
//! comparison relies on, not a production-path test.
//!
//! # What is pinned
//!
//! One server is built and a mixed mutation sequence M (updates, deletes, and
//! post-pin inserts) is applied, collecting the per-epoch `HintDeltaBundle`s.
//! Then three clients, all bootstrapped from the same epoch-0 bundle unless
//! noted, are brought to the server's head and queried for every key:
//!
//! - **R (rewind, the production client)** — `accumulate_delta` each delta,
//!   then `decode`. Its hint is never patched (`pin_epoch()` stays 0), yet it
//!   decodes the current database.
//! - **P (hint-patch, the bench comparator)** — `HintPatchClient::apply_delta`
//!   each delta, then `HintPatchClient::decode`.
//! - **F (fresh)** — a `HintPatchClient` bootstrapped from the server's
//!   *post-M* `setup()` bundle.
//!
//! Every present key must decode to its model value under all three; every
//! absent key to `None`. Then R garbage-collects and must still agree — proving
//! the fold-into-hint path (`collect_garbage`) matches.

use ikpir_client::{
    FrodoConfig, FrodoPirBackend, HintDeltaBundle, HintPatchClient, IkpirClient,
    IncrementalPirBackend, IndexPirBackend, PrecomputingPirBackend, ResponseRewind, SimpleConfig,
    SimplePirBackend,
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
/// Small LWE dimension keeps setup and the per-query matvec fast; the rewind
/// equivalence does not depend on it.
const LWE_DIM: u32 = 256;
const MAX_KICKS: u32 = 2_500;
const SEED_LOAD_PERCENT: u64 = 70;
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

trait Scheme: IndexScheme + SchemeMeta + Sized + 'static {
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

type Model = std::collections::BTreeMap<u32, Vec<u8>>;

struct Snapshot {
    cells: Vec<u32>,
    params: CuckooParams,
    num_items: u64,
    model: Model,
}
impl Snapshot {
    fn restore<S: Scheme>(&self) -> CuckooKVStore<S> {
        S::from_cells(self.cells.clone(), self.params, self.num_items)
    }
}

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

/// Updates, deletes, and post-pin inserts (keys past the seed range) — a mixed
/// run exercising every mutation kind and the created-after-pin path.
fn sequence_mixed(n: u32) -> Vec<Op> {
    let mut ops = Vec::new();
    for i in 0..8 {
        ops.push(Op::Update(i, 47));
        ops.push(Op::Delete(20 + i));
    }
    for i in 0..6 {
        ops.push(Op::Insert(n + i, 31));
    }
    for i in 0..4 {
        ops.push(Op::Update(40 + i, 49));
    }
    ops
}

fn apply<S, B>(
    server: &mut IkpirServer<S, B>,
    ops: &[Op],
    model: &mut Model,
) -> Vec<HintDeltaBundle<B>>
where
    S: Scheme,
    B: IncrementalPirBackend,
{
    let mut deltas = Vec::new();
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
            Err(IkpirError::TableFull) => assert!(matches!(op, Op::Insert(..)), "TableFull {op:?}"),
            Err(e) => panic!("{op:?}: {e:?}"),
        }
    }
    deltas
}

/// Rewind-mode lookup: `build_query` → `answer` → `decode(key, &q, &r)`.
fn lookup_rewind<S, B>(
    client: &mut IkpirClient<B>,
    server: &IkpirServer<S, B>,
    key: u32,
) -> Option<Vec<u8>>
where
    S: Scheme,
    B: IncrementalPirBackend + ResponseRewind + Clone,
    B::Query: Clone,
    B::Response: Clone,
{
    let kb = key.to_le_bytes();
    let q = client.build_query(&kb);
    let r = server.answer(&q).expect("answer");
    client.decode(&kb, &q, &r).expect("decode")
}

/// Hint-patch (bench comparator) lookup: `build_query` → `answer` →
/// `decode(key, &r)`.
fn lookup_patch<S, B>(
    client: &mut HintPatchClient<B>,
    server: &IkpirServer<S, B>,
    key: u32,
) -> Option<Vec<u8>>
where
    S: Scheme,
    B: IndexPirBackend,
    B::Query: Clone,
    B::Response: Clone,
{
    let kb = key.to_le_bytes();
    let q = client.build_query(&kb);
    let r = server.answer(&q).expect("answer");
    client.decode(&kb, &r).expect("decode")
}

fn absent_keys(before: &Model, after: &Model) -> Vec<u32> {
    let mut absent: Vec<u32> = before
        .keys()
        .filter(|k| !after.contains_key(k))
        .copied()
        .collect();
    absent.extend(NEVER_PRESENT);
    absent
}

/// The body: rewind == hint-patch == fresh, then GC still agrees.
fn rewind_matches_patch_and_fresh<S, B>(config: B::Config)
where
    S: Scheme,
    B: IncrementalPirBackend + ResponseRewind + Clone,
    B::Query: Clone,
    B::Response: Clone,
{
    let snap = populate::<S>();
    let n = snap.num_items as u32;
    let ops = sequence_mixed(n);

    let mut server: IkpirServer<S, B> = IkpirServer::new(snap.restore(), config);
    let bundle0 = server.setup();
    let mut model = snap.model.clone();
    let deltas = apply(&mut server, &ops, &mut model);
    assert!(!deltas.is_empty(), "M produced deltas");
    let head = server.epoch();
    let absent = absent_keys(&snap.model, &model);

    // R — rewind (the production client). Never patched; pin stays at 0.
    let mut r = IkpirClient::<B>::from_setup(bundle0.clone());
    for d in &deltas {
        r.accumulate_delta(d.clone()).expect("accumulate_delta");
    }
    assert_eq!(r.epoch(), head, "R head tracks the server");
    assert_eq!(r.pin_epoch(), 0, "R hint never re-pinned");
    assert!(r.pending_cells() > 0, "R accumulated a nonempty ΔD");

    // P — hint-patch (bench comparator).
    let mut p = HintPatchClient::<B>::from_setup(bundle0.clone());
    for d in &deltas {
        p.apply_delta(d.clone()).expect("apply_delta");
    }
    assert_eq!(p.epoch(), head);

    // F — fresh from the post-M bundle (hint-patch, no deltas to apply).
    let mut f = HintPatchClient::<B>::from_setup(server.setup());
    assert_eq!(f.epoch(), head);

    // Every key agrees across R, P, F and equals the model.
    for (&k, v) in &model {
        let vr = lookup_rewind(&mut r, &server, k);
        let vp = lookup_patch(&mut p, &server, k);
        let vf = lookup_patch(&mut f, &server, k);
        assert_eq!(vr.as_deref(), Some(v.as_slice()), "rewind present key {k}");
        assert_eq!(vp, vr, "patch == rewind, key {k}");
        assert_eq!(vf, vr, "fresh == rewind, key {k}");
    }
    for &k in &absent {
        assert_eq!(lookup_rewind(&mut r, &server, k), None, "rewind absent {k}");
        assert_eq!(lookup_patch(&mut p, &server, k), None, "patch absent {k}");
        assert_eq!(lookup_patch(&mut f, &server, k), None, "fresh absent {k}");
    }

    // GC folds ΔD into the hint and re-pins at the head; queries still agree.
    r.collect_garbage().expect("collect_garbage");
    assert_eq!(r.pin_epoch(), head, "GC advances the pin to the head");
    assert_eq!(r.pending_cells(), 0, "GC clears ΔD");
    assert_eq!(r.epoch(), head);
    for (&k, v) in &model {
        assert_eq!(
            lookup_rewind(&mut r, &server, k).as_deref(),
            Some(v.as_slice()),
            "post-GC rewind present key {k}"
        );
    }
    for &k in &absent {
        assert_eq!(
            lookup_rewind(&mut r, &server, k),
            None,
            "post-GC rewind absent {k}"
        );
    }
    // GC with nothing pending is a no-op.
    r.collect_garbage().expect("idempotent GC");
    assert_eq!(r.pending_cells(), 0);
}

/// Explicit control for the step-3-before-scan ordering. Key 0 is updated after
/// the pin; the `ΔD` add (step 3) is what turns the stale pinned value into the
/// current one, and it must precede the fingerprint scan. Without it — or if it
/// ran after the scan — the scan would accumulate the STALE value (an update
/// leaves the fingerprint unchanged, so the slot still matches). The main
/// equivalence test guards this implicitly; this isolates and names it.
fn step3_ordering_control<S, B>(config: B::Config)
where
    S: Scheme,
    B: IncrementalPirBackend + ResponseRewind + Clone,
    B::Query: Clone,
    B::Response: Clone,
{
    let snap = populate::<S>();
    let n = snap.num_items as u32;
    let mut server: IkpirServer<S, B> = IkpirServer::new(snap.restore(), config);
    let bundle0 = server.setup();
    let mut model = snap.model.clone();
    let deltas = apply(&mut server, &sequence_mixed(n), &mut model);

    let mut r = IkpirClient::<B>::from_setup(bundle0);
    for d in &deltas {
        r.accumulate_delta(d.clone()).expect("accumulate_delta");
    }

    // Key 0 was updated salt 17 -> 47 after the pin: a real value change.
    let new = value_for(0, 47);
    let old = value_for(0, 17);
    assert_ne!(new, old, "the post-pin update must change the value");

    let kb = 0u32.to_le_bytes();
    let q = r.build_query(&kb);
    let resp = server.answer(&q).expect("answer");
    assert_eq!(
        r.decode(&kb, &q, &resp).expect("decode").as_deref(),
        Some(new.as_slice()),
        "rewind must recover the post-pin-updated value (step-3 before the scan)"
    );
}

/// A **warm** (precomputed) rewind client stays correct, and garbage collection
/// keeps the Phase-C decode material (`c = sᵀ·H`) consistent with the folded
/// hint: correct decodes both before GC (against the pinned `H₀` + correction)
/// and after GC (against the folded hint, `ΔD` empty), on prepared slots.
fn warm_rewind_and_gc<S, B>(config: B::Config)
where
    S: Scheme,
    B: IncrementalPirBackend + ResponseRewind + PrecomputingPirBackend + Clone,
    B::Query: Clone,
    B::Response: Clone,
{
    let snap = populate::<S>();
    let n = snap.num_items as u32;
    let mut server: IkpirServer<S, B> = IkpirServer::new(snap.restore(), config);
    let bundle0 = server.setup();
    let mut model = snap.model.clone();
    let deltas = apply(&mut server, &sequence_mixed(n), &mut model);
    let absent = absent_keys(&snap.model, &model);
    let present: Vec<u32> = model.keys().copied().collect();

    let mut r = IkpirClient::<B>::from_setup(bundle0);
    // Warm the query + decode material against the pinned hint H₀.
    r.precompute_queries(present.len() as u32 + absent.len() as u32 + 8);
    r.precompute_decodes();
    for d in &deltas {
        r.accumulate_delta(d.clone()).expect("accumulate_delta");
    }
    // Warm decodes against H₀ + ΔD correction.
    for (&k, v) in &model {
        assert_eq!(
            lookup_rewind(&mut r, &server, k).as_deref(),
            Some(v.as_slice()),
            "warm rewind (pre-GC) present key {k}"
        );
    }

    // GC folds ΔD into the hint and, per the Phase-C contract, updates any
    // prepared/in-flight decode material. Refill and re-warm, then decode again.
    r.collect_garbage().expect("collect_garbage");
    assert_eq!(r.pending_cells(), 0);
    r.precompute_queries(present.len() as u32 + absent.len() as u32 + 8);
    r.precompute_decodes();
    for (&k, v) in &model {
        assert_eq!(
            lookup_rewind(&mut r, &server, k).as_deref(),
            Some(v.as_slice()),
            "warm rewind (post-GC) present key {k}"
        );
    }
    for &k in &absent {
        assert_eq!(
            lookup_rewind(&mut r, &server, k),
            None,
            "warm rewind absent {k}"
        );
    }
}

macro_rules! instantiate {
    ($($name:ident: $body:ident < $s:ty, $b:ty > ($cfg:expr);)*) => {
        $(
            #[test]
            fn $name() { $body::<$s, $b>($cfg); }
        )*
    };
}

instantiate! {
    rewind_matches_frodo_2ary: rewind_matches_patch_and_fresh<Segmented2aryScheme, FrodoPirBackend>(frodo_config());
    rewind_matches_frodo_3ary: rewind_matches_patch_and_fresh<Segmented3aryScheme, FrodoPirBackend>(frodo_config());
    rewind_matches_frodo_4ary: rewind_matches_patch_and_fresh<Segmented4aryScheme, FrodoPirBackend>(frodo_config());
    rewind_matches_simple_2ary: rewind_matches_patch_and_fresh<Segmented2aryScheme, SimplePirBackend>(simple_config());
    rewind_matches_simple_3ary: rewind_matches_patch_and_fresh<Segmented3aryScheme, SimplePirBackend>(simple_config());
    rewind_matches_simple_4ary: rewind_matches_patch_and_fresh<Segmented4aryScheme, SimplePirBackend>(simple_config());

    step3_ordering_frodo_2ary: step3_ordering_control<Segmented2aryScheme, FrodoPirBackend>(frodo_config());
    step3_ordering_simple_2ary: step3_ordering_control<Segmented2aryScheme, SimplePirBackend>(simple_config());

    warm_gc_frodo_2ary: warm_rewind_and_gc<Segmented2aryScheme, FrodoPirBackend>(frodo_config());
    warm_gc_simple_2ary: warm_rewind_and_gc<Segmented2aryScheme, SimplePirBackend>(simple_config());
    warm_gc_frodo_4ary: warm_rewind_and_gc<Segmented4aryScheme, FrodoPirBackend>(frodo_config());
}
