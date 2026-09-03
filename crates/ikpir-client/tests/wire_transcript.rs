//! Protocol-level tests for the `HintDeltaBundle` wire encoding
//! (`docs/hint-delta-wire-format.md`), exercised from OUTSIDE
//! `crates/ikpir-common/src/wire.rs` — through the public server/client
//! API and the `IndexPirBackend` trait family, the way a real deployment
//! would use the format.
//!
//! # What these tests prove that `wire.rs`'s own unit tests do not
//!
//! `wire.rs` pins that `decode(encode(b)) == b` *structurally*. That is
//! necessary but not sufficient: a serializer that round-trips the sparse
//! `(row, offset, delta)` shape but encodes `delta` one bit too narrow, or
//! reduces it modulo `p`, would still pass a structural round-trip check
//! for every `|delta| < p/2`-ish input and only show its bug once the
//! decoded rows are actually applied to a hint via
//! `IncrementalPirBackend::server_patch_hint` /
//! `IndexPirBackend::client_decode`. Every test below therefore proves
//! *functional* identity — patching from the wire form yields a
//! bit-identical hint / an identical decoded database — not just
//! structural equality, and each test's doc comment names the concrete
//! bug it would catch.
//!
//! # Why probe-set decode comparison instead of direct `ClientState` access
//!
//! `IkpirClient<B>` keeps its per-segment `B::ClientState` (and hence the
//! patched `B::Hint`) in a **private** field; this file is an integration
//! test (a separate compiled crate) and therefore cannot reach it, even
//! though the concrete `FrodoHint` / `SimpleHint` types happen to derive
//! `PartialEq`. So T1 establishes "client C1 (patched from the real
//! bundle) and client C2 (patched from the decoded bundle) are identical"
//! by comparing `IkpirClient::decode` outputs on a small **fixed probe
//! set** of keys through the *same* server response — if the two
//! clients' hints ever diverged, some probe would eventually decode
//! differently. T2/T3 instead bypass `IkpirClient` entirely and call
//! `IncrementalPirBackend::server_patch_hint` directly on two clones of
//! the same starting `B::Hint` (which *does* derive `PartialEq` for both
//! shipped backends), which is the more direct and more powerful check —
//! see each test's doc comment.
//!
//! # Fixture scale
//!
//! The task calls for a `d × value_bits × plaintext_bits` matrix across
//! both backends (24 configs total). To keep the whole file's debug-mode
//! runtime bounded (~30 s), every fixture uses a deliberately small
//! `num_buckets = d·32`, a small `lwe_dim` (`FrodoConfig` / `SimpleConfig`
//! enforce no minimum beyond `> 0`, and neither shipped backend's
//! correctness margin depends on `lwe_dim`; it is bounded by `row_width`
//! / the reshape row count instead — see `ikpir-common::pir_params`), a
//! short mixed mutation trace, and a small fixed probe set — never a
//! smaller config matrix. T2–T5
//! additionally skip populating the store at all: their property (does
//! patching from the original rows agree with patching from
//! decode(encode(rows))?) does not depend on what the hint starts as, so
//! an all-zero hint from an empty store is exactly as informative and
//! far cheaper than a populated one.

use std::collections::HashMap;

use ikpir_client::{
    FrodoConfig, FrodoPirBackend, HintDeltaBundle, HintPatchMode, IkpirClient, IkpirClientError,
    IncrementalPirBackend, IndexPirBackend, ParallelSetupBackend, ServerSetupBundle, SimpleConfig,
    SimplePirBackend,
};
use ikpir_common::{wire::DeltaWireLayout, SegmentRowDeltas};
use ikpir_server::IkpirServer;
use segmented_cuckoo::{
    CuckooError, CuckooKVStore, CuckooParams, IndexScheme, SchemeMeta, Segmented2aryScheme,
    Segmented3aryScheme, Segmented4aryScheme,
};

// ── Fixture scale knobs ──────────────────────────────────────────────────

/// Small LWE dimension shared by every fixture. Neither shipped backend's
/// decode-correctness margin depends on `lwe_dim` (FrodoPIR's ternary
/// error is a single per-row term; SimplePIR's bound is driven by the
/// reshape row count), so shrinking it only shrinks arithmetic cost, not
/// correctness — see `ikpir-common::pir_params`. Kept small (rather than
/// shrinking `ρ` below the task's specified `d·32`) because T1's
/// per-probe decode cost is `Θ(lwe_dim · row_width)` per segment and
/// dominates the file's debug-mode runtime.
const LWE_DIM: u32 = 12;
/// Fixed at 64 for every fixture per the task's paper-geometry matrix.
const FP_BITS: u32 = 64;
/// Seed load: well below `TableFull` (leaves headroom for the mutation
/// trace's inserts) but high enough to make kick chains routine.
const LOAD_FACTOR: f64 = 0.7;
/// Length of T1's mixed insert/update/delete trace. Kept well below the
/// task's illustrative "~40" to hold the whole file's debug-mode runtime
/// under budget — T1 pays `Θ(lwe_dim · row_width · arity)` per probed key
/// per step for the answer+decode round trip on *two* clients, which
/// dominates every other cost in this file by two orders of magnitude.
const N_OPS: usize = 24;

// ── Per-arity store construction (mirrors `benches/helpers.rs::MakeStore`,
// which lives in a bench target and so cannot be imported here) ─────────

trait MakeStore: IndexScheme + SchemeMeta + Sized + 'static {
    fn make_store(
        num_buckets: u32,
        bucket_size: u32,
        fingerprint_bits: u32,
        value_bits: u32,
        plaintext_bits: u32,
    ) -> Result<CuckooKVStore<Self>, CuckooError>;
}

impl MakeStore for Segmented2aryScheme {
    fn make_store(
        n: u32,
        bs: u32,
        fp: u32,
        vb: u32,
        pb: u32,
    ) -> Result<CuckooKVStore<Self>, CuckooError> {
        CuckooKVStore::<Self>::new(n, bs, fp, vb, pb)
    }
}
impl MakeStore for Segmented3aryScheme {
    fn make_store(
        n: u32,
        bs: u32,
        fp: u32,
        vb: u32,
        pb: u32,
    ) -> Result<CuckooKVStore<Self>, CuckooError> {
        CuckooKVStore::<Self>::new(n, bs, fp, vb, pb)
    }
}
impl MakeStore for Segmented4aryScheme {
    fn make_store(
        n: u32,
        bs: u32,
        fp: u32,
        vb: u32,
        pb: u32,
    ) -> Result<CuckooKVStore<Self>, CuckooError> {
        CuckooKVStore::<Self>::new(n, bs, fp, vb, pb)
    }
}

/// Backend-specific small-LWE-dimension config constructor, so the shared
/// `for_shape` driver can build a `B::Config` without knowing which
/// concrete backend it is instantiating.
trait SmallLweConfig: IndexPirBackend {
    fn small_config(lwe_dim: u32) -> Self::Config;
}
impl SmallLweConfig for FrodoPirBackend {
    fn small_config(lwe_dim: u32) -> FrodoConfig {
        FrodoConfig::with_lwe_dim(lwe_dim)
    }
}
impl SmallLweConfig for SimplePirBackend {
    fn small_config(lwe_dim: u32) -> SimpleConfig {
        SimpleConfig::with_lwe_dim(lwe_dim)
    }
}

/// Drive one `(backend, shape)` cell of the config matrix across
/// `value_bits ∈ {2048, 8192}` at `plaintext_bits = 9`, plus
/// `(value_bits, plaintext_bits) = (8192, 8)` when `extra_pb8` is set
/// (the two `p = 2^8` shapes the task calls out: `(2,4)` and `(3,2)`).
fn for_shape<B: SmallLweConfig>(
    shape_label: &'static str,
    num_buckets: u32,
    bucket_size: u32,
    extra_pb8: bool,
    per_config: fn(&str, u32, u32, u32, u32, B::Config),
) {
    for &(vb, pb) in &[(2048u32, 9u32), (8192u32, 9u32)] {
        per_config(
            shape_label,
            num_buckets,
            bucket_size,
            vb,
            pb,
            B::small_config(LWE_DIM),
        );
    }
    if extra_pb8 {
        per_config(
            shape_label,
            num_buckets,
            bucket_size,
            8192,
            8,
            B::small_config(LWE_DIM),
        );
    }
}

// ── Shared fixture helpers ───────────────────────────────────────────────

/// Deterministic value bytes for key `k`, `⌈value_bits/8⌉` bytes long.
fn value_bytes(k: u32, value_bits: u32) -> Vec<u8> {
    let n = (value_bits as usize).div_ceil(8);
    (0..n)
        .map(|i| (k.wrapping_mul(2_654_435_761).wrapping_add(i as u32) & 0xFF) as u8)
        .collect()
}

/// Seed a fresh store with keys `0..target` (`target = LOAD_FACTOR ×
/// capacity`), close enough to the threshold that later inserts in the
/// mutation trace routinely trigger kick chains.
fn populate<S: MakeStore>(
    num_buckets: u32,
    bucket_size: u32,
    value_bits: u32,
    plaintext_bits: u32,
) -> (CuckooKVStore<S>, u32) {
    let capacity = u64::from(num_buckets) * u64::from(bucket_size);
    let target = (capacity as f64 * LOAD_FACTOR).floor() as u32;
    let mut store = S::make_store(
        num_buckets,
        bucket_size,
        FP_BITS,
        value_bits,
        plaintext_bits,
    )
    .unwrap_or_else(|e| panic!("populate: make_store failed: {e:?}"));
    store.set_max_kicks(2_500);
    for k in 0..target {
        let key = k.to_le_bytes();
        store
            .insert(key, &value_bytes(k, value_bits))
            .unwrap_or_else(|e| panic!("populate: insert {k} failed: {e:?}"));
    }
    (store, target)
}

#[derive(Clone, Copy)]
enum Op {
    Insert(u32),
    Update(u32),
    Delete(u32),
}

/// A deterministic mixed trace of `n_ops` insert/update/delete calls: new
/// keys start at `seed_count`, updates and deletes target the live set.
/// Never lets the live set collapse below 5 keys, so deletes never
/// starve.
fn build_trace(seed_count: u32, n_ops: usize) -> Vec<Op> {
    let mut alive: Vec<u32> = (0..seed_count).collect();
    let mut next_key = seed_count;
    let mut ops = Vec::with_capacity(n_ops);
    for i in 0..n_ops {
        let kind = if alive.len() <= 4 { 0 } else { i % 3 };
        match kind {
            0 => {
                let k = next_key;
                next_key += 1;
                alive.push(k);
                ops.push(Op::Insert(k));
            }
            1 => {
                let idx = (i * 7 + 3) % alive.len();
                ops.push(Op::Update(alive[idx]));
            }
            _ => {
                let idx = (i * 5 + 1) % alive.len();
                let k = alive.swap_remove(idx);
                ops.push(Op::Delete(k));
            }
        }
    }
    ops
}

/// Apply one `Op` to the server, updating the plain-Rust `oracle` map of
/// true current values alongside it, and return the emitted delta.
fn apply_op<S, B>(
    server: &mut IkpirServer<S, B>,
    oracle: &mut HashMap<u32, Vec<u8>>,
    op: Op,
    value_bits: u32,
) -> HintDeltaBundle<B>
where
    S: IndexScheme + SchemeMeta,
    B: IncrementalPirBackend,
{
    match op {
        Op::Insert(k) => {
            let v = value_bytes(k, value_bits);
            let key = k.to_le_bytes();
            let d = server
                .insert(&key, &v)
                .unwrap_or_else(|e| panic!("apply_op: insert {k} failed: {e:?}"));
            oracle.insert(k, v);
            d
        }
        Op::Update(k) => {
            let v = value_bytes(k.wrapping_add(0x4000_0000), value_bits);
            let key = k.to_le_bytes();
            let d = server
                .update(&key, &v)
                .unwrap_or_else(|e| panic!("apply_op: update {k} failed: {e:?}"));
            oracle.insert(k, v);
            d
        }
        Op::Delete(k) => {
            let key = k.to_le_bytes();
            let d = server
                .delete(&key)
                .unwrap_or_else(|e| panic!("apply_op: delete {k} failed: {e:?}"));
            oracle.remove(&k);
            d
        }
    }
}

// ── T1 ────────────────────────────────────────────────────────────────────

/// **T1.** For every real `HintDeltaBundle` a live server emits across a
/// mixed mutation trace: `encode`/`decode` round-trip both structurally
/// (`decode(encode(b)) == b`, `encode().len() == wire_byte_size()`,
/// `wire_stats().nonzero_cells == |S|`) *and* functionally — a client
/// patched from the real bundle (`C1`) and a client patched from
/// `decode(encode(b))` (`C2`), both starting from the same
/// `ServerSetupBundle`, must decode identically on every subsequent
/// probe, and both must decode the *true* current database contents
/// (tracked independently in a plain `oracle` map, not re-derived from
/// the PIR path). This is the end-to-end regression net: it would catch
/// a delta field that is one bit too narrow (silently truncating the
/// high bit of `γ`), an off-by-one in the run-length / gap-merge logic
/// that shifts a cell to the wrong offset, or a canonicalisation bug that
/// makes `encode` emit a form `decode` accepts but that folds `A·D`
/// differently once actually patched — none of which would necessarily
/// break `decode(encode(b)) == b` on its own, since that equality only
/// pins the bundle's *sparse in-memory shape*, not what patching with it
/// does to a hint.
#[allow(clippy::too_many_arguments)]
fn run_t1<S, B>(
    shape_label: &str,
    num_buckets: u32,
    bucket_size: u32,
    value_bits: u32,
    plaintext_bits: u32,
    config: B::Config,
) where
    S: MakeStore,
    B: IncrementalPirBackend + ParallelSetupBackend + Clone,
    B::Query: Clone,
    B::Response: Clone,
{
    let label = format!("{shape_label} vb={value_bits} pb={plaintext_bits} [T1]");
    let (store, seed_count) = populate::<S>(num_buckets, bucket_size, value_bits, plaintext_bits);
    let mut server: IkpirServer<S, B> = IkpirServer::new_parallel(store, config);

    let mut oracle: HashMap<u32, Vec<u8>> = (0..seed_count)
        .map(|k| (k, value_bytes(k, value_bits)))
        .collect();

    let setup = server.setup();
    let params = setup.params;
    let mut c1: IkpirClient<B> = IkpirClient::from_setup(setup.clone());
    let mut c2: IkpirClient<B> = IkpirClient::from_setup(setup);

    let ops = build_trace(seed_count, N_OPS);
    let first_new_key = ops
        .iter()
        .find_map(|op| {
            if let Op::Insert(k) = *op {
                Some(k)
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("{label}: trace must contain at least one insert"));
    // Present-at-start (may later be updated/deleted by the trace),
    // becomes-present mid-trace, always-absent. Kept to three (rather than
    // "every present key") to hold the per-step cost bounded -- see the
    // module doc comment and the `N_OPS` / `LWE_DIM` rationale above.
    let probes = [0u32, first_new_key, 0xFFFF_FF00];

    for (i, &op) in ops.iter().enumerate() {
        let delta = apply_op(&mut server, &mut oracle, op, value_bits);

        let bytes = delta.encode();
        assert_eq!(
            bytes.len(),
            delta.wire_byte_size(),
            "{label} step {i}: encode().len() != wire_byte_size()"
        );
        let decoded = HintDeltaBundle::<B>::decode(&bytes, params)
            .unwrap_or_else(|e| panic!("{label} step {i}: decode(encode(b)) failed: {e}"));
        assert_eq!(decoded, delta, "{label} step {i}: decode(encode(b)) != b");

        let nonzero: u64 = delta
            .per_segment_row_deltas
            .iter()
            .flat_map(|seg| seg.iter())
            .map(|(_, cells)| cells.len() as u64)
            .sum();
        assert_eq!(
            delta.wire_stats().nonzero_cells,
            nonzero,
            "{label} step {i}: wire_stats().nonzero_cells != |(offset, delta)| in b"
        );

        c1.apply_delta(delta)
            .unwrap_or_else(|e| panic!("{label} step {i}: c1.apply_delta(real) failed: {e}"));
        c2.apply_delta(decoded)
            .unwrap_or_else(|e| panic!("{label} step {i}: c2.apply_delta(decoded) failed: {e}"));

        for &k in &probes {
            let key = k.to_le_bytes();

            let q1 = c1.build_query(&key);
            let r1 = server.answer(&q1).unwrap_or_else(|e| {
                panic!("{label} step {i} key {k}: server.answer(c1's query) failed: {e}")
            });
            let v1 = c1
                .decode(&key, &r1)
                .unwrap_or_else(|e| panic!("{label} step {i} key {k}: c1.decode failed: {e}"));

            let q2 = c2.build_query(&key);
            let r2 = server.answer(&q2).unwrap_or_else(|e| {
                panic!("{label} step {i} key {k}: server.answer(c2's query) failed: {e}")
            });
            let v2 = c2
                .decode(&key, &r2)
                .unwrap_or_else(|e| panic!("{label} step {i} key {k}: c2.decode failed: {e}"));

            assert_eq!(
                v1, v2,
                "{label} step {i} key {k}: C1 (patched from the real bundle) and C2 \
                 (patched from decode(encode(bundle))) diverged"
            );

            let expected = oracle.get(&k).cloned();
            assert_eq!(
                v1, expected,
                "{label} step {i} key {k}: decoded value != true database contents"
            );
        }
    }
}

#[test]
fn real_transcripts_round_trip_and_patch_identically() {
    for_shape::<FrodoPirBackend>(
        "Frodo:(2,4)",
        64,
        4,
        true,
        run_t1::<Segmented2aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(3,2)",
        96,
        2,
        true,
        run_t1::<Segmented3aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(3,3)",
        96,
        3,
        false,
        run_t1::<Segmented3aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(4,1)",
        128,
        1,
        false,
        run_t1::<Segmented4aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(4,2)",
        128,
        2,
        false,
        run_t1::<Segmented4aryScheme, FrodoPirBackend>,
    );

    for_shape::<SimplePirBackend>(
        "Simple:(2,4)",
        64,
        4,
        true,
        run_t1::<Segmented2aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(3,2)",
        96,
        2,
        true,
        run_t1::<Segmented3aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(3,3)",
        96,
        3,
        false,
        run_t1::<Segmented3aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(4,1)",
        128,
        1,
        false,
        run_t1::<Segmented4aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(4,2)",
        128,
        2,
        false,
        run_t1::<Segmented4aryScheme, SimplePirBackend>,
    );
}

// ── T2 / T3 shared core ──────────────────────────────────────────────────

/// Hand-built adversarial rows exercising every boundary
/// `docs/hint-delta-wire-format.md` §5 names, independent of any real
/// mutation: the segment/row/offset extremes (row 0), a gap of exactly
/// `max_gap + 1` zero cells (row 1 — must split into two runs), and a gap
/// of exactly `max_gap` zero cells (row 2 — must merge into one run with
/// literal zero deltas inside it). Deltas are placed at `±1` and
/// `±(p−1)` — the narrowest and widest values the `DB`-bit delta field
/// carries.
fn build_adversarial_rows(layout: &DeltaWireLayout, arity: usize) -> Vec<SegmentRowDeltas> {
    let rho = layout.segment_size;
    let omega = layout.row_width;
    let g = layout.max_gap;
    let p_minus_1 = (1i64 << layout.plaintext_bits) - 1;

    let mut segs: Vec<SegmentRowDeltas> = vec![Vec::new(); arity];

    // Segment 0 / row 0 / offset 0 lower boundary.
    segs[0].push((0u32, vec![(0u16, 1i64)]));
    // Last segment / last row / last offset upper boundary (a distinct
    // segment from segment 0 whenever arity >= 2, which it always is).
    segs[arity - 1].push((rho - 1, vec![((omega - 1) as u16, -1i64)]));

    // Two runs separated by exactly max_gap + 1 zero cells.
    let split_a = 5u32;
    let split_b = split_a + 1 + (g + 1);
    segs[0].push((
        1u32,
        vec![(split_a as u16, p_minus_1), (split_b as u16, -p_minus_1)],
    ));

    // One run bridging an interior zero stretch of exactly max_gap cells.
    let merge_a = 5u32;
    let merge_b = merge_a + 1 + g;
    segs[0].push((
        2u32,
        vec![(merge_a as u16, p_minus_1), (merge_b as u16, -p_minus_1)],
    ));

    segs
}

/// Core of T2/T3: patching a server hint from the **original** sparse
/// rows must be bit-identical, under **both** `HintPatchMode`s, to
/// patching from `decode(encode(rows))`. This is the test that would
/// catch a delta field one bit too narrow (silently truncating `±(p−1)`
/// to `±(p/2−1)` or wrapping) or an encoder that reduces `γ` modulo `p`:
/// either bug can still leave `decode(encode(b)) == b` looking plausible
/// for small `|γ|`, and only shows up once the rows are actually applied
/// arithmetically — which only a real `server_patch_hint` call, not a
/// structural comparison, can catch. Bypasses `fold_mutations_into_row_deltas`
/// entirely (the rows here do not correspond to any real DB), which is
/// sound because the patch is linear: identity of the two *results* is
/// the property under test, not their relationship to a real database.
fn assert_rows_patch_identically<B>(
    label: &str,
    params: CuckooParams,
    setup: &ServerSetupBundle<B>,
    segs: Vec<SegmentRowDeltas>,
) where
    B: IncrementalPirBackend,
    B::Hint: PartialEq + std::fmt::Debug,
{
    let bundle: HintDeltaBundle<B> = HintDeltaBundle::new(1, segs, params);
    let bytes = bundle.encode();
    let decoded = HintDeltaBundle::<B>::decode(&bytes, params)
        .unwrap_or_else(|e| panic!("{label}: decode(encode(bundle)) failed: {e}"));
    assert_eq!(
        decoded, bundle,
        "{label}: adversarial bundle did not round-trip structurally"
    );

    let arity = params.arity();
    for j in 0..arity {
        let rows_orig = &bundle.per_segment_row_deltas[j];
        let rows_dec = &decoded.per_segment_row_deltas[j];
        if rows_orig.is_empty() {
            continue;
        }
        let sp = &setup.backend_params[j];
        let material = B::expand_hint_material(sp);
        let hint0 = &setup.hints[j];

        for mode in [HintPatchMode::EntryLevel, HintPatchMode::RowLevel] {
            let mut h1 = hint0.clone();
            let mut h2 = hint0.clone();
            B::server_patch_hint(sp, &material, &mut h1, rows_orig, mode);
            B::server_patch_hint(sp, &material, &mut h2, rows_dec, mode);
            assert_eq!(
                h1, h2,
                "{label} segment {j} mode {mode:?}: server_patch_hint diverged between \
                 the original sparse rows and decode(encode(rows))"
            );
        }
    }
}

/// **T2.** Drives `assert_rows_patch_identically` at every config in the
/// matrix, on an empty (all-zero-hint) server — the property does not
/// depend on real DB content, only on whether decode(encode(rows))
/// carries the same arithmetic information as rows.
fn run_t2<S, B>(
    shape_label: &str,
    num_buckets: u32,
    bucket_size: u32,
    value_bits: u32,
    plaintext_bits: u32,
    config: B::Config,
) where
    S: MakeStore,
    B: IncrementalPirBackend + ParallelSetupBackend,
    B::Hint: PartialEq + std::fmt::Debug,
{
    let label = format!("{shape_label} vb={value_bits} pb={plaintext_bits} [T2]");
    let store = S::make_store(
        num_buckets,
        bucket_size,
        FP_BITS,
        value_bits,
        plaintext_bits,
    )
    .unwrap_or_else(|e| panic!("{label}: make_store failed: {e:?}"));
    let server: IkpirServer<S, B> = IkpirServer::new_parallel(store, config);
    let setup = server.setup();
    let params = setup.params;
    let layout = DeltaWireLayout::for_params(&params);

    let segs = build_adversarial_rows(&layout, params.arity());
    assert_rows_patch_identically::<B>(&label, params, &setup, segs);
}

#[test]
fn adversarial_synthetic_bundles_patch_identically() {
    for_shape::<FrodoPirBackend>(
        "Frodo:(2,4)",
        64,
        4,
        true,
        run_t2::<Segmented2aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(3,2)",
        96,
        2,
        true,
        run_t2::<Segmented3aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(3,3)",
        96,
        3,
        false,
        run_t2::<Segmented3aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(4,1)",
        128,
        1,
        false,
        run_t2::<Segmented4aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(4,2)",
        128,
        2,
        false,
        run_t2::<Segmented4aryScheme, FrodoPirBackend>,
    );

    for_shape::<SimplePirBackend>(
        "Simple:(2,4)",
        64,
        4,
        true,
        run_t2::<Segmented2aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(3,2)",
        96,
        2,
        true,
        run_t2::<Segmented3aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(3,3)",
        96,
        3,
        false,
        run_t2::<Segmented3aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(4,1)",
        128,
        1,
        false,
        run_t2::<Segmented4aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(4,2)",
        128,
        2,
        false,
        run_t2::<Segmented4aryScheme, SimplePirBackend>,
    );
}

// ── T3 ────────────────────────────────────────────────────────────────────

/// **T3.** At `plaintext_bits = 8` (`p = 2^8 = 256`), `DB = plaintext_bits
/// + 1 = 9`, *not* 8: `docs/hint-delta-wire-format.md` §2 derives the
/// delta width from `2p − 1` code points, not `p`, because `γ` ranges
/// over `(−p, p)`, not `[0, p)`. A backend that mistakenly sized the
/// delta field at `plaintext_bits` bits (8, matching the *cell* width)
/// would silently truncate `γ = ±255` — this is exactly the config where
/// that bug would first appear, since `p = 256` is a power of two and
/// `255` fits in 8 bits unsigned, so an off-by-one here is easy to miss
/// without an explicit boundary check. Runs only on the two `p = 2^8`
/// shapes the task specifies, `(2,4)` and `(3,2)`, on both backends.
fn run_t3<S, B>(shape_label: &str, num_buckets: u32, bucket_size: u32, config: B::Config)
where
    S: MakeStore,
    B: IncrementalPirBackend + ParallelSetupBackend,
    B::Hint: PartialEq + std::fmt::Debug,
{
    let value_bits = 8192u32;
    let plaintext_bits = 8u32;
    let label = format!("{shape_label} vb={value_bits} pb={plaintext_bits} [T3: p=2^8]");
    let store = S::make_store(
        num_buckets,
        bucket_size,
        FP_BITS,
        value_bits,
        plaintext_bits,
    )
    .unwrap_or_else(|e| panic!("{label}: make_store failed: {e:?}"));
    let server: IkpirServer<S, B> = IkpirServer::new_parallel(store, config);
    let setup = server.setup();
    let params = setup.params;
    let layout = DeltaWireLayout::for_params(&params);
    assert_eq!(
        layout.delta_bits, 9,
        "{label}: p = 2^8 must encode deltas at plaintext_bits + 1 = 9 bits, not 8"
    );

    // build_adversarial_rows places gamma = +-(p-1) = +-255 at several
    // cells; assert_rows_patch_identically's round-trip check therefore
    // also covers "gamma = +-255 round-trips" as part of the structural
    // check below.
    let segs = build_adversarial_rows(&layout, params.arity());
    assert_rows_patch_identically::<B>(&label, params, &setup, segs);
}

#[test]
fn p_equals_256_cells_use_nine_bit_deltas() {
    run_t3::<Segmented2aryScheme, FrodoPirBackend>(
        "Frodo:(2,4)",
        64,
        4,
        FrodoPirBackend::small_config(LWE_DIM),
    );
    run_t3::<Segmented3aryScheme, FrodoPirBackend>(
        "Frodo:(3,2)",
        96,
        2,
        FrodoPirBackend::small_config(LWE_DIM),
    );
    run_t3::<Segmented2aryScheme, SimplePirBackend>(
        "Simple:(2,4)",
        64,
        4,
        SimplePirBackend::small_config(LWE_DIM),
    );
    run_t3::<Segmented3aryScheme, SimplePirBackend>(
        "Simple:(3,2)",
        96,
        2,
        SimplePirBackend::small_config(LWE_DIM),
    );
}

// ── T4 ────────────────────────────────────────────────────────────────────

/// **T4.** `decode`d output under a receiver whose `CuckooParams` differ
/// (here, only in `plaintext_bits`, so field widths shift) must never
/// panic — `Ok` under the foreign geometry or a `WireError` are both
/// acceptable branches, since the input crosses a trust boundary
/// (`docs/hint-delta-wire-format.md` §7). Separately,
/// `IkpirClient::apply_delta` must reject a bundle carrying the client's
/// own real deltas but a different `params` with `MalformedBundle` —
/// catching a version where `apply_delta` forgot the `delta.params !=
/// self.params` check (or checked it after already mutating state,
/// rather than before).
#[allow(clippy::too_many_arguments)]
fn run_t4<S, B>(
    shape_label: &str,
    num_buckets: u32,
    bucket_size: u32,
    value_bits: u32,
    plaintext_bits: u32,
    config: B::Config,
) where
    S: MakeStore,
    B: IncrementalPirBackend + ParallelSetupBackend,
    B::Query: Clone,
    B::Response: Clone,
{
    let label = format!("{shape_label} vb={value_bits} pb={plaintext_bits} [T4]");
    let store = S::make_store(
        num_buckets,
        bucket_size,
        FP_BITS,
        value_bits,
        plaintext_bits,
    )
    .unwrap_or_else(|e| panic!("{label}: make_store failed: {e:?}"));
    let mut server: IkpirServer<S, B> = IkpirServer::new_parallel(store, config);
    let key = 12_345u32.to_le_bytes();
    let delta = server
        .insert(&key, &value_bytes(12_345, value_bits))
        .unwrap_or_else(|e| panic!("{label}: seed insert failed: {e:?}"));
    let real_params = delta.params;

    // (a) decode under a receiver whose plaintext_bits differs must not panic.
    let mut foreign = real_params;
    foreign.plaintext_bits = if real_params.plaintext_bits == 9 {
        8
    } else {
        9
    };
    let bytes = delta.encode();
    let _ = HintDeltaBundle::<B>::decode(&bytes, foreign);

    // (b) apply_delta rejects the client's own deltas under foreign params.
    let mut client: IkpirClient<B> = IkpirClient::from_setup(server.setup());
    let forged: HintDeltaBundle<B> = HintDeltaBundle::new(
        client.epoch() + 1,
        delta.per_segment_row_deltas.clone(),
        foreign,
    );
    let err = match client.apply_delta(forged) {
        Ok(()) => panic!("{label}: apply_delta accepted a bundle with foreign params"),
        Err(e) => e,
    };
    assert!(
        matches!(err, IkpirClientError::MalformedBundle),
        "{label}: apply_delta with foreign params must return MalformedBundle, got {err:?}"
    );
}

#[test]
fn client_rejects_bundle_with_foreign_params() {
    for_shape::<FrodoPirBackend>(
        "Frodo:(2,4)",
        64,
        4,
        true,
        run_t4::<Segmented2aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(3,2)",
        96,
        2,
        true,
        run_t4::<Segmented3aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(3,3)",
        96,
        3,
        false,
        run_t4::<Segmented3aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(4,1)",
        128,
        1,
        false,
        run_t4::<Segmented4aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(4,2)",
        128,
        2,
        false,
        run_t4::<Segmented4aryScheme, FrodoPirBackend>,
    );

    for_shape::<SimplePirBackend>(
        "Simple:(2,4)",
        64,
        4,
        true,
        run_t4::<Segmented2aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(3,2)",
        96,
        2,
        true,
        run_t4::<Segmented3aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(3,3)",
        96,
        3,
        false,
        run_t4::<Segmented3aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(4,1)",
        128,
        1,
        false,
        run_t4::<Segmented4aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(4,2)",
        128,
        2,
        false,
        run_t4::<Segmented4aryScheme, SimplePirBackend>,
    );
}

// ── T5 ────────────────────────────────────────────────────────────────────

/// **T5.** `decode` must never panic on adversarial input, regardless of
/// whether it accepts or rejects it — `docs/hint-delta-wire-format.md`
/// §7 requires bounds to be checked *before* any index is used, so a
/// fuzzed bucket / offset / run-length must be turned into a `WireError`,
/// never used to index out of bounds or to build an inconsistent `Vec`
/// length. And whenever a corrupted stream happens to decode to a
/// well-formed but *different* bundle (`Ok(b2)` with `b2 != b`), applying
/// `b2` to a client must not panic either — a wrong resulting hint is the
/// caller's epoch/params problem (rejected by `apply_delta`'s own checks,
/// or simply a hint that no longer matches the server), not the
/// decoder's, but `client_patch_state` must still process whatever
/// legal-per-`decode` row/offset data it was handed without an internal
/// bounds violation.
#[allow(clippy::too_many_arguments)]
fn run_t5<S, B>(
    shape_label: &str,
    num_buckets: u32,
    bucket_size: u32,
    value_bits: u32,
    plaintext_bits: u32,
    config: B::Config,
) where
    S: MakeStore,
    B: IncrementalPirBackend + ParallelSetupBackend + Clone,
    B::Query: Clone,
    B::Response: Clone,
{
    let label = format!("{shape_label} vb={value_bits} pb={plaintext_bits} [T5]");
    let store = S::make_store(
        num_buckets,
        bucket_size,
        FP_BITS,
        value_bits,
        plaintext_bits,
    )
    .unwrap_or_else(|e| panic!("{label}: make_store failed: {e:?}"));
    let mut server: IkpirServer<S, B> = IkpirServer::new_parallel(store, config);

    let pre_insert_setup = server.setup();
    let key0 = 0u32.to_le_bytes();
    let insert_delta = server
        .insert(&key0, &value_bytes(0, value_bits))
        .unwrap_or_else(|e| panic!("{label}: seed insert failed: {e:?}"));
    let pre_update_setup = server.setup();
    let update_delta = server
        .update(&key0, &value_bytes(0xDEAD_BEEF, value_bits))
        .unwrap_or_else(|e| panic!("{label}: seed update failed: {e:?}"));

    for (pre_setup, real) in [
        (&pre_insert_setup, &insert_delta),
        (&pre_update_setup, &update_delta),
    ] {
        let bytes = real.encode();
        let params = real.params;

        let mut variants: Vec<Vec<u8>> = Vec::new();
        let flip_bits = (bytes.len() * 8).min(128);
        for bit in 0..flip_bits {
            let mut b = bytes.clone();
            b[bit / 8] ^= 1 << (bit % 8);
            variants.push(b);
        }
        for len in 0..=bytes.len() {
            variants.push(bytes[..len].to_vec());
        }
        {
            let mut b = bytes.clone();
            b.push(0xAA);
            variants.push(b);
        }

        // One client, reused (and repeatedly mutated) across every variant
        // for this real encoding. Sound because the only property under
        // test is "does not panic" -- never post-apply correctness -- so a
        // client left in a nonsensical state by one corrupted apply is
        // still a valid target for the next "does this panic?" probe.
        // Rebuilding a fresh bootstrap per variant instead (as an earlier
        // version of this test did) is O(variants) setup calls and made
        // this test dominate the file's runtime for no additional
        // coverage.
        let mut client: IkpirClient<B> = IkpirClient::from_setup(pre_setup.clone());

        for cbytes in variants {
            match HintDeltaBundle::<B>::decode(&cbytes, params) {
                Err(_) => {} // rejected -- fine, this is the expected common case
                Ok(b2) => {
                    if b2 != *real {
                        let _ = client.apply_delta(b2);
                    }
                }
            }
        }
    }
}

#[test]
fn decode_never_panics_on_corrupted_input() {
    for_shape::<FrodoPirBackend>(
        "Frodo:(2,4)",
        64,
        4,
        true,
        run_t5::<Segmented2aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(3,2)",
        96,
        2,
        true,
        run_t5::<Segmented3aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(3,3)",
        96,
        3,
        false,
        run_t5::<Segmented3aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(4,1)",
        128,
        1,
        false,
        run_t5::<Segmented4aryScheme, FrodoPirBackend>,
    );
    for_shape::<FrodoPirBackend>(
        "Frodo:(4,2)",
        128,
        2,
        false,
        run_t5::<Segmented4aryScheme, FrodoPirBackend>,
    );

    for_shape::<SimplePirBackend>(
        "Simple:(2,4)",
        64,
        4,
        true,
        run_t5::<Segmented2aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(3,2)",
        96,
        2,
        true,
        run_t5::<Segmented3aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(3,3)",
        96,
        3,
        false,
        run_t5::<Segmented3aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(4,1)",
        128,
        1,
        false,
        run_t5::<Segmented4aryScheme, SimplePirBackend>,
    );
    for_shape::<SimplePirBackend>(
        "Simple:(4,2)",
        128,
        2,
        false,
        run_t5::<Segmented4aryScheme, SimplePirBackend>,
    );
}
