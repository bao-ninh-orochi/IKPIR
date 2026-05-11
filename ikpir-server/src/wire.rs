//! Wire-format bundles exchanged between [`IkpirServer`](crate::IkpirServer)
//! and `IkpirClient`.
//!
//! These types are plain data: no I/O, no serialisation. The crate keeps the
//! format unstable on purpose — bundles are passed by value within a process
//! in tests and examples, and any production deployment is expected to layer
//! its own serialisation strategy on top.
//!
//! # Bundle taxonomy
//!
//! | Bundle | Direction | Carries |
//! |---|---|---|
//! | [`ServerSetupBundle`]    | server → client | full preprocessing material (`Hint`, `ServerParams`, `CuckooParams`, epoch) |
//! | [`PirQueryBundle`]       | client → server | one [`B::Query`](crate::IndexPirBackend::Query) per segment + epoch |
//! | [`PirResponseBundle`]    | server → client | one [`B::Response`](crate::IndexPirBackend::Response) per segment + epoch |
//! | [`HintDeltaBundle`]      | server → client | sparse per-segment row deltas + epoch (after a single mutation) |

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
pub(crate) type SegmentRowDeltas = Vec<(u32, Vec<(u16, i64)>)>;

/// Snapshot of the server's full preprocessing state, sent to a fresh client.
///
/// The client materialises one [`B::ClientState`](crate::IndexPirBackend::ClientState)
/// per segment from `(backend_params[j], hints[j])`, caches the
/// [`CuckooParams`], and adopts the server's `epoch`.
///
/// # Invariants
///
/// Once a client has been initialised from a `ServerSetupBundle`, every
/// subsequent bundle for that client instance — whether a full re-setup or a
/// [`HintDeltaBundle`] — must carry identical
/// `(scheme_kind, num_buckets, bucket_size, fingerprint_bits, value_bits,
/// plaintext_bits)`. Mid-flight changes are undefined; the upgrade path is to
/// rebuild a fresh client via `IkpirClient::reset_from(new_bundle)`.
/// If support for changing these parameters mid-flight is ever needed, both
/// this bundle and `HintDeltaBundle` would need to carry a parameter-identity
/// fingerprint that the client asserts on every patch.
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
    /// Minimum on-wire byte size of this setup bundle. Sums
    /// [`CuckooParams`] + epoch + per-segment
    /// [`B::server_params_byte_size`](BackendWireSize::server_params_byte_size)
    /// + per-segment [`B::hint_byte_size`](BackendWireSize::hint_byte_size).
    pub fn wire_byte_size(&self) -> usize {
        let mut bytes = CUCKOO_PARAMS_BYTES + 8; // params + epoch
        bytes += 4; // backend_params length prefix
        for p in &self.backend_params { bytes += B::server_params_byte_size(p); }
        bytes += 4; // hints length prefix
        for h in &self.hints           { bytes += B::hint_byte_size(h); }
        bytes
    }
}

/// One PIR query per segment, addressed to a specific server epoch.
///
/// Reject by [`IkpirServer::answer`](crate::IkpirServer::answer) with
/// [`IkpirError::StaleEpoch`](crate::IkpirError::StaleEpoch) if `epoch`
/// is not the server's current epoch.
#[derive(Clone)]
pub struct PirQueryBundle<B: IndexPirBackend>
where
    B::Query: Clone,
{
    /// Epoch at which the client built the queries. Must equal the server's
    /// current epoch when [`IkpirServer::answer`](crate::IkpirServer::answer)
    /// runs.
    pub epoch: u64,
    /// One [`B::Query`](IndexPirBackend::Query) per segment.
    pub queries: Vec<B::Query>,
}

impl<B: BackendWireSize> PirQueryBundle<B>
where
    B::Query: Clone,
{
    /// Minimum on-wire byte size: 8 (epoch) + 4 (vec length prefix) +
    /// Σ [`B::query_byte_size`](BackendWireSize::query_byte_size).
    pub fn wire_byte_size(&self) -> usize {
        let mut bytes = 8 + 4;
        for q in &self.queries { bytes += B::query_byte_size(q); }
        bytes
    }
}

/// One PIR response per segment, paired with the epoch that produced it.
///
/// Reject by `IkpirClient::decode` with `EpochMismatch` if `resp.epoch`
/// differs from the client's cached epoch — the server moved between query
/// and answer.
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
    /// Minimum on-wire byte size: 8 (epoch) + 4 (vec length prefix) +
    /// Σ [`B::response_byte_size`](BackendWireSize::response_byte_size).
    pub fn wire_byte_size(&self) -> usize {
        let mut bytes = 8 + 4;
        for r in &self.responses { bytes += B::response_byte_size(r); }
        bytes
    }
}

/// Sparse hint patch produced by a single successful mutation.
///
/// Wire size is proportional to the mutation footprint (one mutated row per
/// touched bucket × `cells_per_slot` cells), not the database size. The
/// client folds the delta into its per-segment state with `apply_delta`,
/// which is strict-monotone: only `delta.epoch == client.epoch + 1` is
/// accepted.
///
/// Per-mutation deltas inherit the cell-width geometry of the
/// [`ServerSetupBundle`] they patch; `plaintext_bits` (and the rest of the
/// SCF geometry) must remain constant across the bundle stream. See
/// [`ServerSetupBundle`] for the invariant statement and upgrade path.
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
    pub(crate) fn new(epoch: u64, per_segment_row_deltas: Vec<SegmentRowDeltas>) -> Self {
        Self { epoch, per_segment_row_deltas, _marker: PhantomData }
    }

    /// Minimum on-wire byte size of this delta bundle under fixed-width
    /// little-endian encoding (8-byte epoch, 4-byte length prefix per
    /// segment, 4-byte row index + 4-byte length prefix per row, 2-byte
    /// cell offset + 8-byte signed delta per cell).
    ///
    /// Backend-agnostic: the delta contents are plain integers, no
    /// backend ciphertext involved.
    pub fn wire_byte_size(&self) -> usize {
        let mut bytes = 8; // epoch
        bytes += 4;        // per_segment_row_deltas length prefix
        for seg in &self.per_segment_row_deltas {
            bytes += 4;    // segment-internal length prefix
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
            epoch:                  self.epoch,
            per_segment_row_deltas: self.per_segment_row_deltas.clone(),
            _marker:                PhantomData,
        }
    }
}
