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
//! This crate ships **two parallel client flows** over the same scheme,
//! both consuming the same server-published [`HintDeltaBundle`] stream:
//!
//! | flow | type | sync verb | decode signature | maintenance cost | notes |
//! |---|---|---|---|---|---|
//! | client-hint-patch | [`HintPatchClient`] | `apply_delta` | `decode(key, resp)` | `Θ(n·τ·ω)` per batch | patched hint, no per-query correction, selectable [`HintPatchMode`] |
//! | client-rewind | [`RewindClient`] (alias [`IkpirClient`]) | `accumulate_delta` | `decode(key, query, resp)` | `Θ(τ·ω)` per batch | pinned hint, per-query correction grows with \|ΔD\|, `collect_garbage` reclaims it |
//!
//! # Lifecycles
//!
//! **client-hint-patch** (`HintPatchClient`):
//!
//! ```text
//! from_setup(bundle)          — initialise from server's setup bundle
//!   └── (optional) precompute_queries(N)   — Phase B: amortise b = A·s + e
//!   └── (optional) precompute_decodes()    — Phase C: amortise c = sᵀ·H
//!   └── loop:
//!         build_query(key)       — one B::Query per segment (cheap if Phase B warm)
//!         server.answer(&q)      — server returns PirResponseBundle
//!         decode(key, &resp)     — fp match (cheap if Phase C warm)
//!         apply_delta(delta)     — fold incremental hint update (epoch+1)
//!                                    also patches Phase-C material in place
//!   └── on FutureDelta / after server full_rebuild:
//!         reset_from(new_bundle)  — replace all internal state
//! ```
//!
//! **client-rewind** (`RewindClient`):
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
//! # Epoch rules (both flows)
//!
//! - The sync verb ([`HintPatchClient::apply_delta`] /
//!   [`RewindClient::accumulate_delta`]) is **strict-monotone**: only
//!   `delta.epoch == self.epoch + 1` is accepted; older deltas are
//!   [`StaleDelta`](IkpirClientError::StaleDelta), gaps are
//!   [`FutureDelta`](IkpirClientError::FutureDelta).
//! - `decode` requires `resp.epoch == self.epoch` on both flows; any
//!   mismatch is [`EpochMismatch`](IkpirClientError::EpochMismatch).
//!
//! See `docs/rewind-client-mode.md` for the full comparison and
//! `tests/client_flow_parity.rs` for the pin that both flows decode
//! identically.
//!
//! # Quick start
//!
//! Client-rewind:
//!
//! ```no_run
//! use ikpir_client::RewindClient;
//! use ikpir_server::{FrodoConfig, FrodoPirBackend, Segmented2aryIkpirServer};
//! use segmented_cuckoo::Segmented2aryCuckooKVStore;
//!
//! let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
//! store.insert(b"alice", &[0xAB]).unwrap();
//!
//! let mut server: Segmented2aryIkpirServer<FrodoPirBackend> =
//!     Segmented2aryIkpirServer::new(store, FrodoConfig::default());
//! let mut client: RewindClient<FrodoPirBackend> =
//!     RewindClient::from_setup(server.setup());
//!
//! let q = client.build_query(b"alice");
//! let r = server.answer(&q).unwrap();
//! // `decode` threads the query back so the response can be rewound to the
//! // client's pinned hint before decoding.
//! let v = client.decode(b"alice", &q, &r).unwrap().expect("found");
//! assert_eq!(v, vec![0xAB]);
//! ```
//!
//! Client-hint-patch:
//!
//! ```no_run
//! use ikpir_client::HintPatchClient;
//! use ikpir_server::{FrodoConfig, FrodoPirBackend, Segmented2aryIkpirServer};
//! use segmented_cuckoo::Segmented2aryCuckooKVStore;
//!
//! let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
//! store.insert(b"alice", &[0xAB]).unwrap();
//!
//! let mut server: Segmented2aryIkpirServer<FrodoPirBackend> =
//!     Segmented2aryIkpirServer::new(store, FrodoConfig::default());
//! let mut client: HintPatchClient<FrodoPirBackend> =
//!     HintPatchClient::from_setup(server.setup());
//!
//! let q = client.build_query(b"alice");
//! let r = server.answer(&q).unwrap();
//! // The hint is patched immediately on every delta, so `decode` needs only
//! // the response — no query threading, no per-query correction.
//! let v = client.decode(b"alice", &r).unwrap().expect("found");
//! assert_eq!(v, vec![0xAB]);
//! ```
//!
//! For systematic measurement of build_query / decode / accumulate_delta
//! across parameter ranges, see
//! `benches/{client_query,client_decode,client_mutation,client_rewind_staleness,headtohead_query,headtohead_decode}.rs`,
//! run via the `scripts/bench.sh <name>` runner at the workspace root.

mod client_hint_patch;
mod client_rewind;
mod ct;
mod error;
mod outcome;
mod pending;

pub use client_hint_patch::HintPatchClient;
pub use client_rewind::RewindClient;
pub use error::IkpirClientError;
pub use outcome::DeltaApplyOutcome;

/// Alias of [`RewindClient`] kept for source compatibility; new code should
/// name the flow it uses.
pub type IkpirClient<B> = RewindClient<B>;

pub use ikpir_common::{
    BackendWireSize, DeltaWireLayout, DeltaWireStats, FrodoConfig, FrodoPirBackend,
    HintDeltaBundle, HintPatchMode, IkpirError, IncrementalPirBackend, IndexPirBackend,
    ParallelSetupBackend, PirQueryBundle, PirResponseBundle, PrecomputingPirBackend,
    ResponseRewind, SegmentRowDeltas, ServerSetupBundle, SimpleConfig, SimplePirBackend, WireError,
};
