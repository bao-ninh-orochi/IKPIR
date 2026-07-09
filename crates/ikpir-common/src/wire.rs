//! Wire-format bundles exchanged between `IkpirServer`
//! and `IkpirClient`.
//!
//! # Purpose
//!
//! Names the four data shapes that cross the server/client boundary:
//! setup, query, response, and incremental hint delta. Each carries the
//! current `epoch` so the receiver can detect drift.
//!
//! # Design / architecture
//!
//! These types are **plain data**: no I/O, no serialisation. The crate
//! keeps the wire format unstable on purpose — bundles are passed by
//! value within a process in tests and examples, and any production
//! deployment is expected to layer its own serialisation strategy on top.
//! `wire_byte_size` reports the *minimum* on-wire footprint under a
//! fixed-width little-endian encoding so deployments can compare
//! parameter choices without committing to a specific serializer.
//!
//! ## Bundle taxonomy
//!
//! | Bundle | Direction | Carries |
//! |---|---|---|
//! | [`ServerSetupBundle`]    | server → client | full preprocessing material (`Hint`, `ServerParams`, `CuckooParams`, epoch) |
//! | [`PirQueryBundle`]       | client → server | one `B::Query` per segment + epoch |
//! | [`PirResponseBundle`]    | server → client | one `B::Response` per segment + epoch |
//! | [`HintDeltaBundle`]      | server → client | sparse per-segment row deltas + epoch (after a single mutation) |
//!
//! # Related files
//!
//! - `server.rs` in `ikpir-server` — emits all four bundles; consumes `PirQueryBundle` in
//!   `answer`.
//! - `hint_patch.rs` in `ikpir-server` — `fold_mutations_into_row_deltas` produces the
//!   `per_segment_row_deltas` inside `HintDeltaBundle`.
//! - `IkpirClient` in `ikpir-client` — sole consumer on the client side.

use std::marker::PhantomData;

use crate::backend::{BackendWireSize, IndexPirBackend};
use segmented_cuckoo::CuckooParams;

/// Minimum on-wire encoding of a [`CuckooParams`] under the convention
/// used by [`BackendWireSize`]: every scalar at its declared width.
/// `CuckooParams` is `{ scheme_kind, num_buckets, bucket_size,
/// fingerprint_bits, value_bits, plaintext_bits }`: 1 enum byte + 5×u32.
const CUCKOO_PARAMS_BYTES: usize = 1 + 5 * 4;

/// Per-segment sparse row deltas: `Vec<(row_in_segment, Vec<(cell_offset_in_row, delta)>)>`.
///
/// Each tuple is `(row_index_within_segment, edits)`. Each edit is
/// `(cell_offset_within_row, signed_delta_mod_2^plaintext_bits)`. Empty
/// segments and zero deltas are dropped at fold time.
pub type SegmentRowDeltas = Vec<(u32, Vec<(u16, i64)>)>;

/// Snapshot of the server's full preprocessing state, sent to a fresh client.
///
/// # Purpose
///
/// Bootstraps `IkpirClient`: the client materialises one
/// `B::ClientState` per segment
/// from `(backend_params[j], hints[j])`, caches the [`CuckooParams`],
/// and adopts the server's `epoch`.
///
/// # Constraints
///
/// Once a client has been initialised from a `ServerSetupBundle`, every
/// subsequent bundle for that client instance — whether a full re-setup
/// or a [`HintDeltaBundle`] — must carry identical
/// `(scheme_kind, num_buckets, bucket_size, fingerprint_bits, value_bits,
/// plaintext_bits)`. Mid-flight changes are undefined; the upgrade path
/// is to rebuild a fresh client via `IkpirClient::reset_from(new_bundle)`.
///
/// # Rationale
///
/// If support for changing these parameters mid-flight is ever needed,
/// both this bundle and `HintDeltaBundle` would need to carry a
/// parameter-identity fingerprint that the client asserts on every
/// patch. Today the IKPIR design assumes static geometry across a client
/// lifetime.
#[derive(Clone)]
pub struct ServerSetupBundle<B: IndexPirBackend> {
    /// Geometry of the underlying SCF KV store (arity, bucket size, fingerprint
    /// width, value width, plaintext width).
    pub params: CuckooParams,
    /// One [`B::ServerParams`](IndexPirBackend::ServerParams) per segment;
    /// length equals `params.arity()`.
    pub backend_params: Vec<B::ServerParams>,
    /// One [`B::Hint`](IndexPirBackend::Hint) per segment; length equals
    /// `params.arity()`.
    pub hints: Vec<B::Hint>,
    /// Server epoch at the moment the bundle was emitted. Strictly monotone
    /// across the server's lifetime.
    pub epoch: u64,
}

impl<B: BackendWireSize> ServerSetupBundle<B> {
    /// Minimum on-wire byte size of this setup bundle.
    ///
    /// # Returns
    ///
    /// Sum of [`CuckooParams`] (21 bytes) plus epoch (8 bytes) plus per-segment
    /// [`B::server_params_byte_size`](BackendWireSize::server_params_byte_size)
    /// plus per-segment [`B::hint_byte_size`](BackendWireSize::hint_byte_size),
    /// plus 4-byte length prefixes for each of the two `Vec`s.
    ///
    /// # Complexity
    ///
    /// `O(arity)` (one pass over `backend_params` and `hints`).
    pub fn wire_byte_size(&self) -> usize {
        let mut bytes = CUCKOO_PARAMS_BYTES + 8; // params + epoch
        bytes += 4; // backend_params length prefix
        for p in &self.backend_params {
            bytes += B::server_params_byte_size(p);
        }
        bytes += 4; // hints length prefix
        for h in &self.hints {
            bytes += B::hint_byte_size(h);
        }
        bytes
    }
}

/// One PIR query per segment, addressed to a specific server epoch.
///
/// # Purpose
///
/// Client → server message: carries the per-segment LWE-encrypted row
/// requests plus the epoch they were built against, so the server can
/// reject any query that targets a stale view of the database.
///
/// # Constraints
///
/// `queries.len()` must equal `params.arity()` of the
/// [`ServerSetupBundle`] used to build the client.
/// `IkpirServer::answer` rejects with
/// `IkpirError::StaleEpoch` if `epoch`
/// is not the server's current epoch.
#[derive(Clone)]
pub struct PirQueryBundle<B: IndexPirBackend>
where
    B::Query: Clone,
{
    /// Epoch at which the client built the queries. Must equal the server's
    /// current epoch when `IkpirServer::answer` runs.
    pub epoch: u64,
    /// One [`B::Query`](IndexPirBackend::Query) per segment.
    pub queries: Vec<B::Query>,
}

impl<B: BackendWireSize> PirQueryBundle<B>
where
    B::Query: Clone,
{
    /// Minimum on-wire byte size of this query bundle.
    ///
    /// # Returns
    ///
    /// `8 (epoch) + 4 (vec length prefix) + Σ`
    /// [`B::query_byte_size`](BackendWireSize::query_byte_size).
    ///
    /// # Complexity
    ///
    /// `O(arity)`.
    pub fn wire_byte_size(&self) -> usize {
        let mut bytes = 8 + 4;
        for q in &self.queries {
            bytes += B::query_byte_size(q);
        }
        bytes
    }
}

/// One PIR response per segment, paired with the epoch that produced it.
///
/// # Purpose
///
/// Server → client message: carries one
/// [`B::Response`](IndexPirBackend::Response) per segment plus the epoch
/// the response was computed under. The client uses the epoch to detect
/// races (a successful mutation between query and answer would shift
/// the response's row vs. the client's view).
///
/// # Constraints
///
/// `responses.len()` equals `params.arity()`. `IkpirClient::decode`
/// rejects with `EpochMismatch` if `resp.epoch` differs from the
/// client's cached epoch.
#[derive(Clone)]
pub struct PirResponseBundle<B: IndexPirBackend>
where
    B::Response: Clone,
{
    /// Epoch under which the server produced these responses.
    pub epoch: u64,
    /// One [`B::Response`](IndexPirBackend::Response) per segment.
    pub responses: Vec<B::Response>,
}

impl<B: BackendWireSize> PirResponseBundle<B>
where
    B::Response: Clone,
{
    /// Minimum on-wire byte size of this response bundle.
    ///
    /// # Returns
    ///
    /// `8 (epoch) + 4 (vec length prefix) + Σ`
    /// [`B::response_byte_size`](BackendWireSize::response_byte_size).
    ///
    /// # Complexity
    ///
    /// `O(arity)`.
    pub fn wire_byte_size(&self) -> usize {
        let mut bytes = 8 + 4;
        for r in &self.responses {
            bytes += B::response_byte_size(r);
        }
        bytes
    }
}

/// Sparse hint patch produced by a single successful mutation.
///
/// # Purpose
///
/// Lets the IKPIR server propagate the effect of one
/// `insert` / `update` / `delete` to every client without re-sending the
/// full hint. The client folds the delta with
/// `IkpirClient::apply_delta`, which is strict-monotone (only
/// `delta.epoch == client.epoch + 1` is accepted).
///
/// # Rationale
///
/// Wire size is proportional to the mutation footprint (one mutated row
/// per touched bucket × `cells_per_slot` cells), not the database size.
/// Per-mutation deltas inherit the cell-width geometry of the
/// [`ServerSetupBundle`] they patch; `plaintext_bits` (and the rest of
/// the SCF geometry) must remain constant across the bundle stream. See
/// [`ServerSetupBundle`] for the invariant statement and upgrade path.
///
/// The bundle does **not** encode a
/// [`HintPatchMode`](crate::HintPatchMode): the transcript is the same
/// sparse cell-delta set under either realization, and either
/// realization folds it into the same post-patch state — so the
/// row-level vs entry-level choice stays local to each side and never
/// crosses the wire.
pub struct HintDeltaBundle<B: IndexPirBackend> {
    /// Server epoch *after* the mutation that produced this delta. Strictly
    /// `previous_epoch + 1`.
    pub epoch: u64,
    /// Per-segment list of touched-row sparse cell deltas. `vec[j]` is the
    /// edits for segment `j`; length equals `params.arity()`. Empty
    /// segments are represented by an empty inner `Vec`.
    pub per_segment_row_deltas: Vec<SegmentRowDeltas>,
    _marker: PhantomData<B>,
}

impl<B: IndexPirBackend> HintDeltaBundle<B> {
    /// Internal constructor used by `IkpirServer::commit_mutations`; end users
    /// receive the bundle from `insert` / `update` / `delete` rather
    /// than build it themselves.
    #[doc(hidden)]
    pub const fn new(epoch: u64, per_segment_row_deltas: Vec<SegmentRowDeltas>) -> Self {
        Self {
            epoch,
            per_segment_row_deltas,
            _marker: PhantomData,
        }
    }

    /// Minimum on-wire byte size of this delta bundle.
    ///
    /// # Rationale
    ///
    /// Backend-agnostic: the delta contents are plain integers, no
    /// backend ciphertext involved.
    ///
    /// # Returns
    ///
    /// Bytes under fixed-width little-endian encoding: 8 (epoch) +
    /// 4 (per-segment length prefix) + per segment: 4 (segment-internal
    /// length prefix) + per row: 4 (row index) + 4 (cells length prefix)
    /// + 10 bytes per cell (`u16` offset + `i64` signed delta).
    ///
    /// # Complexity
    ///
    /// `O(touched_rows + touched_cells)`.
    pub fn wire_byte_size(&self) -> usize {
        let mut bytes = 8; // epoch
        bytes += 4; // per_segment_row_deltas length prefix
        for seg in &self.per_segment_row_deltas {
            bytes += 4; // segment-internal length prefix
            for (_row, cells) in seg {
                bytes += 4; // row index
                bytes += 4; // cells length prefix
                bytes += cells.len() * (2 + 8); // (cell_offset: u16, delta: i64)
            }
        }
        bytes
    }
}

impl<B: IndexPirBackend> Clone for HintDeltaBundle<B> {
    fn clone(&self) -> Self {
        Self {
            epoch: self.epoch,
            per_segment_row_deltas: self.per_segment_row_deltas.clone(),
            _marker: PhantomData,
        }
    }
}
