#![warn(missing_docs)]
//! Server side of the **Incremental Keyword PIR** (IKPIR) protocol.
//!
//! Wraps a [`segmented_cuckoo::CuckooKVStore`] in per-segment Index-PIR
//! sub-databases and exposes the full server protocol — `setup`, `answer`,
//! `insert`, `update`, `delete`, and `full_rebuild`.
//!
//! # Design at a glance
//!
//! An arity-`k` Segmented Cuckoo Filter splits the underlying KV store into
//! `k` independent Index-PIR sub-databases. A query for a key targets row
//! `indices[j] % segment_size` in segment `j`, so `k` Index-PIR queries
//! suffice to recover the slot — for any arity. The filter geometry is in
//! [`segmented_cuckoo::CuckooParams`]; the per-segment dispatch lives in
//! [`IkpirServer`].
//!
//! # Incremental hint patching
//!
//! After every successful `insert` / `update` / `delete`, the server drains
//! the [`segmented_cuckoo::SlotMutation`] log and folds the entries into a
//! sparse [`HintDeltaBundle`]. The active backend's
//! [`IncrementalPirBackend::server_patch_hint`] applies the same delta to
//! its own preprocessing matrix, so the wire bandwidth is proportional to
//! the mutation footprint, not the database size.
//!
//! Each successful mutation strictly increments the server `epoch`. The
//! client uses the epoch monotonicity to ensure deltas are applied in
//! order; see [`ikpir_client`](https://docs.rs/ikpir-client) for the
//! corresponding state machine.
//!
//! # Backend selection
//!
//! The shipped backend is [`FrodoPirBackend`] (LWE-based, post-quantum,
//! incremental). New backends implement [`IndexPirBackend`] and optionally
//! [`IncrementalPirBackend`]; see the trait docs for the contract.
//!
//! # Quick start
//!
//! ```
//! use ikpir_server::{FrodoConfig, FrodoPirBackend, Segmented2aryIkpirServer};
//! use segmented_cuckoo::Segmented2aryCuckooKVStore;
//!
//! let mut store = Segmented2aryCuckooKVStore::new(64, 4, 12, 8, 8).unwrap();
//! store.insert(b"alice", &[0xAB]).unwrap();
//!
//! let mut server: Segmented2aryIkpirServer<FrodoPirBackend> =
//!     Segmented2aryIkpirServer::new(store, FrodoConfig::default());
//! let bundle = server.setup();
//! assert_eq!(bundle.epoch, 0);
//! ```
//!
//! For systematic measurement of setup / answer / incremental-vs-rebuild
//! across parameter ranges, see `benches/{setup,answer,incremental_vs_rebuild}_throughput.rs`
//! and the matching `scripts/plot.py`.

pub mod backend;
mod hint_patch;
mod server;
mod wire;

pub use backend::{
    BackendWireSize, IndexPirBackend, IncrementalPirBackend, PrecomputingPirBackend,
};
pub use backend::{FrodoConfig, FrodoPirBackend};
pub use server::{
    IkpirServer,
    Segmented2aryIkpirServer, Segmented3aryIkpirServer, Segmented4aryIkpirServer,
};
pub use wire::{HintDeltaBundle, PirQueryBundle, PirResponseBundle, ServerSetupBundle};

/// Errors returned by [`IkpirServer`] methods.
///
/// All variants are recoverable in the sense that they leave the server's
/// internal state consistent: the mutation log is drained on insert/update
/// failure to prevent leaked deltas (see
/// `tests/incremental_correctness.rs::mutation_log_drained_on_failure`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IkpirError {
    /// The query carries an epoch that does not match the server's current
    /// epoch. Returned by [`IkpirServer::answer`].
    StaleEpoch {
        /// Current server epoch.
        expected: u64,
        /// Epoch the client query announced.
        got: u64,
    },
    /// The query bundle has the wrong number of segments for the active arity.
    /// Returned by [`IkpirServer::answer`].
    MalformedQuery,
    /// The cuckoo kicking budget was exhausted on `insert`. The store and
    /// hint are rolled back to the pre-insert state.
    ///
    /// Recovery requires rebuilding the server with a larger `num_buckets`
    /// from an externally-held `(key, value)` map — the SCF store discards
    /// keys after hashing, so IKPIR cannot self-recover. Clients then call
    /// `IkpirClient::reset_from` on the new bundle. See [`IkpirServer::insert`]
    /// for the full recovery recipe.
    TableFull,
    /// `update` or `delete` was called with a key that is not present.
    NotFound,
    /// Caller-provided value width does not match the store's `value_size_in_bytes()`.
    InvalidInput,
}
