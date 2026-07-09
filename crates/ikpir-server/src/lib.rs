#![warn(missing_docs)]
//! Server side of the **Incremental Keyword PIR** (IKPIR) protocol —
//! with an LWE backend plugged in, the server half of the paper's
//! **RisePIR** (RisePIR-F over FrodoPIR, RisePIR-S over SimplePIR).
//!
//! Wraps a [`segmented_cuckoo::CuckooKVStore`] in per-segment Index-PIR
//! sub-databases and exposes the full server protocol — `setup`, `answer`,
//! `insert`, `update`, `delete`, and `full_rebuild`.
//!
//! # Design at a glance
//!
//! An arity-`d` Segmented Cuckoo Filter splits the underlying KV store into
//! `d` independent Index-PIR sub-databases. A query for a key targets row
//! `indices[j] % segment_size` in segment `j`, so `d` Index-PIR queries
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
//! Two backends ship today, both LWE-based and post-quantum:
//! [`FrodoPirBackend`] (ternary errors, tall-skinny per-segment matrix,
//! default `lwe_dim = 1566`) and [`SimplePirBackend`] (discrete-Gaussian
//! errors, square-ish reshape, default `lwe_dim = 1275`). They are
//! drop-in alternatives at the `B: IndexPirBackend` type parameter on
//! [`IkpirServer`]. New backends implement [`IndexPirBackend`] and
//! optionally [`IncrementalPirBackend`]; see the trait docs for the
//! contract.
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
//! For systematic measurement of setup / answer / mutation throughput
//! across parameter ranges, see
//! `benches/{server_setup,server_answer,server_mutation,headtohead_answer}.rs`,
//! run via the `scripts/bench.sh <name>` runner at the workspace root.

mod hint_patch;
mod server;

// Re-export the shared protocol surface from ikpir-common so existing
// `use ikpir_server::{IndexPirBackend, FrodoConfig, ServerSetupBundle, ...}`
// call sites continue to resolve unchanged.
pub use ikpir_common::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, HintDeltaBundle, HintPatchMode, IkpirError,
    IncrementalPirBackend, IndexPirBackend, PirQueryBundle, PirResponseBundle,
    PrecomputingPirBackend, ServerSetupBundle, SimpleConfig, SimplePirBackend,
};

pub use server::{
    IkpirServer, Segmented2aryIkpirServer, Segmented3aryIkpirServer, Segmented4aryIkpirServer,
};
