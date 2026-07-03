#![warn(missing_docs)]
//! Client side of the **Incremental Keyword PIR** (IKPIR) protocol.
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
//! ```text
//! from_setup(bundle)          — initialise from server's setup bundle
//!   └── (optional) precompute_queries(N) — Phase B: amortise b = A·s + e
//!   └── (optional) precompute_decodes()  — Phase C: amortise c = sᵀ·H
//!   └── loop:
//!         build_query(key)    — one B::Query per segment (cheap if Phase B warm)
//!         server.answer(&q)   — server returns PirResponseBundle
//!         decode(key, &resp)  — fp match (cheap if Phase C warm)
//!         apply_delta(delta)  — fold incremental hint update (epoch+1)
//!                                also patches Phase-C material in place
//!   └── on FutureDelta / after server full_rebuild:
//!         reset_from(new_bundle)  — replace all internal state
//! ```
//!
//! - [`apply_delta`](IkpirClient::apply_delta) is **strict-monotone**: only
//!   `delta.epoch == self.epoch + 1` is accepted; older deltas are
//!   [`StaleDelta`](IkpirClientError::StaleDelta), gaps are
//!   [`FutureDelta`](IkpirClientError::FutureDelta).
//! - [`decode`](IkpirClient::decode) requires `resp.epoch == self.epoch`;
//!   any mismatch is [`EpochMismatch`](IkpirClientError::EpochMismatch).
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
//! let v = client.decode(b"alice", &r).unwrap().expect("found");
//! assert_eq!(v, vec![0xAB]);
//! ```
//!
//! For systematic measurement of build_query / decode / apply_delta
//! across parameter ranges, see `benches/{query,decode,apply_delta}_throughput.rs`
//! and the matching `scripts/plot.py`.

mod client;
mod error;

pub use client::{DeltaApplyOutcome, IkpirClient};
pub use error::IkpirClientError;

pub use ikpir_common::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, HintDeltaBundle, HintPatchMode, IkpirError,
    IncrementalPirBackend, IndexPirBackend, PirQueryBundle, PirResponseBundle,
    PrecomputingPirBackend, ServerSetupBundle, SimpleConfig, SimplePirBackend,
};
