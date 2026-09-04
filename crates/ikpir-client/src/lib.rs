#![warn(missing_docs)]
//! Client side of the **Incremental Keyword PIR** (IKPIR) protocol —
//! with an LWE backend plugged in, the client half of the paper's
//! **RisePIR** (RisePIR-F over FrodoPIR, RisePIR-S over SimplePIR).
//!
//! Holds [`segmented_cuckoo::CuckooParams`] and per-segment
//! [`B::ClientState`](IndexPirBackend::ClientState) plus an epoch counter.
//! Translates user-level `(key, value)` operations into the wire-level
//! Index-PIR query/response bundles exposed by `ikpir-server`. The client
//! never owns a `CuckooKVStore`; its only persistent material from the
//! server is the [`ServerSetupBundle`] received at setup time.
//!
//! # Lifecycle
//!
//! The client's sole update strategy is **response-rewind**: it pins its
//! bootstrap hint `H₀` and never patches it, rolling the server's published
//! deltas forward as a `ΔD` correction instead — a factor-`n` cheaper
//! maintenance than patching the hint on every mutation
//! (`docs/rewind-client-mode.md`).
//!
//! ```text
//! from_setup(bundle)          — initialise from server's setup bundle
//!   └── (optional) precompute_queries(N)   — Phase B: amortise b = A·s + e
//!   └── (optional) precompute_decodes()    — Phase C: amortise c = sᵀ·H
//!   └── loop:
//!         build_query(key)       — one B::Query per segment (cheap if Phase B warm)
//!         server.answer(&q)      — server returns PirResponseBundle
//!         decode(key, &q, &resp) — rewind the response to H₀, decode, fp match
//!         accumulate_delta(delta) — roll the published ΔD forward (epoch+1)
//!   └── (optional) collect_garbage() — fold ΔD into H₀, reclaim the
//!         per-query correction cost
//!   └── on FutureDelta / after server full_rebuild:
//!         reset_from(new_bundle)  — replace all internal state
//! ```
//!
//! - [`accumulate_delta`](IkpirClient::accumulate_delta) is
//!   **strict-monotone**: only `delta.epoch == self.epoch + 1` is accepted;
//!   older deltas are [`StaleDelta`](IkpirClientError::StaleDelta), gaps are
//!   [`FutureDelta`](IkpirClientError::FutureDelta).
//! - [`decode`](IkpirClient::decode) requires `resp.epoch == self.epoch`;
//!   any mismatch is [`EpochMismatch`](IkpirClientError::EpochMismatch).
//!
//! The classical alternative — patching the hint on every delta — survives
//! only as a benchmark comparator behind the `hint-patch-bench` Cargo
//! feature (`HintPatchClient`, disabled by default); see
//! `bench_comparator` and `docs/rewind-client-mode.md`.
//!
//! # Quick start
//!
//! ```no_run
//! use ikpir_client::IkpirClient;
//! use ikpir_server::{FrodoConfig, FrodoPirBackend, Segmented2aryIkpirServer};
//! use segmented_cuckoo::Segmented2aryCuckooKVStore;
//!
//! let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
//! store.insert(b"alice", &[0xAB]).unwrap();
//!
//! let mut server: Segmented2aryIkpirServer<FrodoPirBackend> =
//!     Segmented2aryIkpirServer::new(store, FrodoConfig::default());
//! let mut client: IkpirClient<FrodoPirBackend> =
//!     IkpirClient::from_setup(server.setup());
//!
//! let q = client.build_query(b"alice");
//! let r = server.answer(&q).unwrap();
//! // `decode` threads the query back so the response can be rewound to the
//! // client's pinned hint before decoding.
//! let v = client.decode(b"alice", &q, &r).unwrap().expect("found");
//! assert_eq!(v, vec![0xAB]);
//! ```
//!
//! For systematic measurement of build_query / decode / accumulate_delta
//! across parameter ranges, see
//! `benches/{client_query,client_decode,client_mutation,client_rewind_staleness,headtohead_query,headtohead_decode}.rs`,
//! run via the `scripts/bench.sh <name>` runner at the workspace root.

mod client;
mod error;
mod pending;

#[cfg(feature = "hint-patch-bench")]
pub mod bench_comparator;

pub use client::{DeltaApplyOutcome, IkpirClient};
pub use error::IkpirClientError;

#[cfg(feature = "hint-patch-bench")]
pub use bench_comparator::HintPatchClient;

pub use ikpir_common::{
    BackendWireSize, DeltaWireLayout, DeltaWireStats, FrodoConfig, FrodoPirBackend,
    HintDeltaBundle, HintPatchMode, IkpirError, IncrementalPirBackend, IndexPirBackend,
    ParallelSetupBackend, PirQueryBundle, PirResponseBundle, PrecomputingPirBackend,
    ResponseRewind, SegmentRowDeltas, ServerSetupBundle, SimpleConfig, SimplePirBackend, WireError,
};
