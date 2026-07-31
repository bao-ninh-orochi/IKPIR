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

use ikpir_common::backend::parallel::PAR_MIN_HINT_MACS;
use ikpir_server::{
    FrodoConfig, FrodoPirBackend, IkpirServer, IncrementalPirBackend, IndexPirBackend,
    ParallelSetupBackend, SimpleConfig, SimplePirBackend,
};
use segmented_cuckoo::{Segmented2aryCuckooKVStore, Segmented2aryScheme};

use ikpir_client::IkpirClient;

/// One store geometry to run the whole interop matrix at.
///
/// The fields are exactly the axes that move `row_width`, which is the
/// only channel through which the fingerprint width reaches the setup
/// kernels: `row_width = bucket_size · ⌈(fingerprint_bits +
/// value_bits)/plaintext_bits⌉`.
#[derive(Clone, Copy)]
struct Fixture {
    label: &'static str,
    num_buckets: u32,
    bucket_size: u32,
    fingerprint_bits: u32,
    value_bits: u32,
    plaintext_bits: u32,
}

impl Fixture {
    /// The `row_width` this fixture presents to the backends.
    fn row_width(&self) -> u64 {
        u64::from(self.bucket_size)
            * u64::from(self.fingerprint_bits + self.value_bits)
                .div_ceil(u64::from(self.plaintext_bits))
    }

    fn store(&self) -> Segmented2aryCuckooKVStore {
        Segmented2aryCuckooKVStore::new(
            self.num_buckets,
            self.bucket_size,
            self.fingerprint_bits,
            self.value_bits,
            self.plaintext_bits,
        )
        .unwrap()
    }
}

/// The geometries this file pins.
///
/// The first is the historical toy shape, kept because it is cheap and
/// exercises a `row_width` far below one cache line. The rest run at
/// **`fingerprint_bits = 64` and the plaintext widths the δ_cell-targeted
/// selector actually chooses**, reproducing real paper `row_width`s
/// (`ikpir_common::pir_params`, and the parameter memo's operating-point
/// table) rather than a shape no deployment uses:
///
/// | fixture | f | ℓ | pb | `row_width` | paper cell it reproduces |
/// |---|---|---|---|---|---|
/// | `toy` | 12 | 8 | 8 | 12 | none (historical) |
/// | `f64-narrow` | 64 | 2048 | 9 | 235 | (4,1) ℓ = 2048, both backends |
/// | `f64-wide` | 64 | 8192 | 9 | 918 | (4,1) ℓ = 8192, both backends |
/// | `f64-widest` | 64 | 8192 | 8 | 1032 | (2,4)/(3,2) ℓ = 8192, SimplePIR |
///
/// `num_buckets` is kept small deliberately: the fan-out gate is on
/// `segment_rows · lwe_dim · row_width`, and at these widths even a
/// 16-bucket store clears [`PAR_MIN_HINT_MACS`] by orders of magnitude
/// (asserted per fixture), so the wide cases cost a few million MACs
/// rather than the paper's hundreds of billions. What is being pinned is
/// the *schedule*, which depends on `lwe_dim` and `row_width`, not on
/// how many rows are stacked underneath.
const FIXTURES: &[Fixture] = &[
    Fixture {
        label: "toy f=12 ℓ=8 pb=8 (row_width 12)",
        num_buckets: 256,
        bucket_size: 4,
        fingerprint_bits: 12,
        value_bits: 8,
        plaintext_bits: 8,
    },
    Fixture {
        label: "f64-narrow f=64 ℓ=2048 pb=9 (row_width 235)",
        num_buckets: 64,
        bucket_size: 1,
        fingerprint_bits: 64,
        value_bits: 2048,
        plaintext_bits: 9,
    },
    Fixture {
        label: "f64-wide f=64 ℓ=8192 pb=9 (row_width 918)",
        num_buckets: 64,
        bucket_size: 1,
        fingerprint_bits: 64,
        value_bits: 8192,
        plaintext_bits: 9,
    },
    Fixture {
        label: "f64-widest f=64 ℓ=8192 pb=8 (row_width 1032)",
        num_buckets: 32,
        bucket_size: 1,
        fingerprint_bits: 64,
        value_bits: 8192,
        plaintext_bits: 8,
    },
];

/// Fail loudly if the fixture is too small to reach the parallel hint
/// kernel for `lwe_dim`.
///
/// The optimized entry points fall back to the reference schedule below
/// [`PAR_MIN_HINT_MACS`], and that fallback is silent — a fixture that
/// drifts under the threshold turns every test in this file into an
/// expensive way of testing the reference path twice. The backend unit
/// tests guard themselves the same way; this is the integration-level
/// counterpart.
///
/// `macs` mirrors what `compute_hint_parallel` computes: per-segment
/// rows × `lwe_dim` × `row_width`, on the *pre-reshape* dimensions that
/// both backends use for the decision.
fn assert_fixture_fans_out(fixture: &Fixture, lwe_dim: u32, backend: &str) {
    let segment_rows = u64::from(fixture.num_buckets) / 2; // arity 2
    let macs = segment_rows * u64::from(lwe_dim) * fixture.row_width();
    assert!(
        macs >= PAR_MIN_HINT_MACS,
        "{backend} fixture {} only reaches {macs} MACs, under the \
         PAR_MIN_HINT_MACS={PAR_MIN_HINT_MACS} fan-out threshold — this \
         file would silently test the sequential path; grow num_buckets",
        fixture.label,
    );
}

/// A value of exactly the fixture's width, deterministic in `seed`.
///
/// The store rejects any other length (`IkpirError::InvalidInput`), and
/// at `ℓ = 8192` that is a 1 KiB value — the paper's wide cell.
fn val(fixture: &Fixture, seed: u8) -> Vec<u8> {
    let len = (fixture.value_bits / 8) as usize;
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
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
fn paths_interoperate<B>(fixture: &Fixture, config: B::Config)
where
    // `B: Clone` is what makes the wire bundles cloneable, so the same
    // delta / setup snapshot can be handed to both clients.
    B: ParallelSetupBackend + IncrementalPirBackend + Clone,
    B::Query: Clone,
    B::Response: Clone,
{
    let (a, b, c, d, x) = (
        val(fixture, 0xA1),
        val(fixture, 0xB2),
        val(fixture, 0xC3),
        val(fixture, 0xD4),
        val(fixture, 0x58),
    );

    // ── Server on the optimized path, clients on both ────────────────
    let mut server: IkpirServer<Segmented2aryScheme, B> =
        IkpirServer::new_parallel(fixture.store(), config.clone());
    server.insert(b"alpha", &a).unwrap();
    server.insert(b"beta", &b).unwrap();

    let mut reference_client = IkpirClient::<B>::from_setup(server.setup());
    let mut parallel_client = IkpirClient::<B>::from_setup_parallel(server.setup());
    assert_lookup(&server, &mut reference_client, b"alpha", &a);
    assert_lookup(&server, &mut parallel_client, b"alpha", &a);

    // ── Mutations keep both clients in lock-step with that server ────
    let deltas = [
        server.update(b"alpha", &x).unwrap(),
        server.insert(b"gamma", &c).unwrap(),
        server.delete(b"beta").unwrap(),
    ];
    for delta in deltas {
        reference_client.apply_delta(delta.clone()).unwrap();
        parallel_client.apply_delta(delta).unwrap();
    }
    assert_lookup(&server, &mut reference_client, b"alpha", &x);
    assert_lookup(&server, &mut parallel_client, b"gamma", &c);

    // ── A rebuild on the optimized path resyncs both clients ─────────
    let bundle = server.full_rebuild_parallel();
    reference_client.reset_from(bundle.clone());
    parallel_client.reset_from_parallel(bundle);
    assert_lookup(&server, &mut reference_client, b"gamma", &c);
    assert_lookup(&server, &mut parallel_client, b"alpha", &x);

    // ── Server on the reference path, client on the optimized one ────
    let mut reference_server: IkpirServer<Segmented2aryScheme, B> =
        IkpirServer::new(fixture.store(), config);
    reference_server.insert(b"delta", &d).unwrap();
    let mut client = IkpirClient::<B>::from_setup_parallel(reference_server.setup());
    assert_lookup(&reference_server, &mut client, b"delta", &d);
}

#[test]
fn frodo_setup_paths_interoperate() {
    for fixture in FIXTURES {
        let config = FrodoConfig::default();
        assert_fixture_fans_out(fixture, config.lwe_dim, "frodo");
        println!("frodo: {}", fixture.label);
        paths_interoperate::<FrodoPirBackend>(fixture, config);
    }
}

#[test]
fn simple_setup_paths_interoperate() {
    for fixture in FIXTURES {
        let config = SimpleConfig::default();
        assert_fixture_fans_out(fixture, config.lwe_dim, "simple");
        println!("simple: {}", fixture.label);
        paths_interoperate::<SimplePirBackend>(fixture, config);
    }
}
