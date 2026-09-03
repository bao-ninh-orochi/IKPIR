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
//! [`HintDeltaBundle`] has a normative, specified on-wire encoding —
//! `docs/hint-delta-wire-format.md` — implemented by
//! [`HintDeltaBundle::encode`] / [`HintDeltaBundle::decode`] in this file
//! and enforced by the `#[cfg(test)]` module below. `wire_byte_size` on
//! that bundle equals `encode().len()` **by construction** (both derive
//! from the same [`DeltaWireLayout`] and the same run-splitting rule) and
//! **by test** (`encode().len() == wire_byte_size()` is checked on every
//! hand-built and property-tested bundle).
//!
//! The other three bundles below (`ServerSetupBundle`, `PirQueryBundle`,
//! `PirResponseBundle`) are unchanged: **plain data, no I/O, no
//! serialisation**. Their `wire_byte_size` reports the *minimum* on-wire
//! footprint under a fixed-width little-endian encoding so deployments can
//! compare parameter choices without committing to a specific serializer —
//! the wire format for those three stays unstable on purpose; only
//! `HintDeltaBundle` has a specified encoding.
//!
//! ## Bundle taxonomy
//!
//! | Bundle | Direction | Carries |
//! |---|---|---|
//! | [`ServerSetupBundle`]    | server → client | full preprocessing material (`Hint`, `ServerParams`, `CuckooParams`, epoch) — accounting only, no specified encoding |
//! | [`PirQueryBundle`]       | client → server | one `B::Query` per segment + epoch — accounting only, no specified encoding |
//! | [`PirResponseBundle`]    | server → client | one `B::Response` per segment + epoch — accounting only, no specified encoding |
//! | [`HintDeltaBundle`]      | server → client | sparse per-segment row deltas + epoch (after a single mutation) — the **only** bundle with a specified bit-packed wire encoding, `docs/hint-delta-wire-format.md` |
//!
//! # Related files
//!
//! - `docs/hint-delta-wire-format.md` — normative specification of
//!   [`HintDeltaBundle`]'s byte encoding; this file implements it.
//! - `server.rs` in `ikpir-server` — emits all four bundles; consumes `PirQueryBundle` in
//!   `answer`.
//! - `hint_patch.rs` in `ikpir-server` — `fold_mutations_into_row_deltas` produces the
//!   `per_segment_row_deltas` inside `HintDeltaBundle`.
//! - `IkpirClient` in `ikpir-client` — sole consumer on the client side.

use std::fmt;
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

// ─── HintDeltaBundle wire format ──────────────────────────────────────────

/// Number of bits needed to write `x` as an index: `0` if `x == 0`, else
/// `⌊log₂ x⌋ + 1`. `docs/hint-delta-wire-format.md` §2, `bitlen`.
const fn bitlen(x: u64) -> u32 {
    if x == 0 {
        0
    } else {
        64 - x.leading_zeros()
    }
}

/// Derived bit/byte widths of the [`HintDeltaBundle`] wire encoding for one
/// [`CuckooParams`] geometry.
///
/// # Purpose
///
/// Single source of truth for the four widths `docs/hint-delta-wire-format.md`
/// §2 defines (`IB`, `OW`, `DB`, `G`), plus the geometry scalars they derive
/// from. [`HintDeltaBundle::encode`], [`HintDeltaBundle::decode`], and
/// [`HintDeltaBundle::wire_stats`] all build one from `self.params` and
/// share it, so the three can never compute different widths for the same
/// bundle.
///
/// # Rationale
///
/// A width may be `0` (an index domain with one element needs no bits to
/// name its sole member); the bit writer/reader treat a `0`-width field as
/// writing/reading nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaWireLayout {
    /// `n_b = P.num_buckets` — total buckets in the store.
    pub num_buckets: u32,
    /// `ρ = P.segment_size()` — rows per segment.
    pub segment_size: u32,
    /// `ω = P.bucket_size · P.cells_per_slot()` — cells per row.
    pub row_width: u32,
    /// `P.plaintext_bits` — the cell modulus is `p = 2^plaintext_bits`.
    pub plaintext_bits: u32,
    /// `IB = bitlen(n_b - 1)` — width of a global bucket index.
    pub bucket_bits: u32,
    /// `OW = bitlen(ω - 1)` — width of a cell offset and of a run length
    /// minus one.
    pub offset_bits: u32,
    /// `DB = plaintext_bits + 1` — width of one delta code point.
    pub delta_bits: u32,
    /// `G = (2·OW + 1) div DB` — longest run of zero cells one run may
    /// bridge.
    pub max_gap: u32,
}

impl DeltaWireLayout {
    /// Derive the wire widths for one geometry.
    ///
    /// # Purpose
    ///
    /// Implements `docs/hint-delta-wire-format.md` §2 exactly: `IB`, `OW`,
    /// `DB`, `G` from `P.num_buckets`, `P.segment_size()`,
    /// `P.bucket_size · P.cells_per_slot()`, and `P.plaintext_bits`.
    ///
    /// # Returns
    ///
    /// A [`DeltaWireLayout`] with every field populated.
    pub fn for_params(params: &CuckooParams) -> Self {
        let num_buckets = params.num_buckets;
        let segment_size = params.segment_size();
        let row_width = params.bucket_size * params.cells_per_slot();
        let plaintext_bits = params.plaintext_bits;
        let bucket_bits = bitlen(u64::from(num_buckets.saturating_sub(1)));
        let offset_bits = bitlen(u64::from(row_width.saturating_sub(1)));
        let delta_bits = plaintext_bits + 1;
        let max_gap = (2 * offset_bits + 1) / delta_bits;
        Self {
            num_buckets,
            segment_size,
            row_width,
            plaintext_bits,
            bucket_bits,
            offset_bits,
            delta_bits,
            max_gap,
        }
    }

    /// Total encoded length in bits for the given counts, before byte
    /// padding.
    ///
    /// # Purpose
    ///
    /// The closed form of `docs/hint-delta-wire-format.md` §6.
    ///
    /// # Arguments
    ///
    /// - `rows`  — touched rows across every segment.
    /// - `runs`  — emitted runs across every touched row.
    /// - `cells` — delta literals across every run (run lengths summed,
    ///   interior zeros included).
    ///
    /// # Returns
    ///
    /// `64 + 1 + rows·(1 + IB) + runs·(2·OW + 1) + cells·DB`.
    pub const fn bits(&self, rows: u64, runs: u64, cells: u64) -> u64 {
        64 + 1
            + rows * (1 + self.bucket_bits as u64)
            + runs * (2 * self.offset_bits as u64 + 1)
            + cells * self.delta_bits as u64
    }

    /// Total encoded length in bytes for the given counts.
    ///
    /// # Returns
    ///
    /// `⌈self.bits(rows, runs, cells) / 8⌉`.
    pub const fn bytes(&self, rows: u64, runs: u64, cells: u64) -> u64 {
        self.bits(rows, runs, cells).div_ceil(8)
    }
}

/// Row/run/cell accounting for one [`HintDeltaBundle`], produced by
/// [`HintDeltaBundle::wire_stats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaWireStats {
    /// Touched rows across every segment.
    pub rows: u64,
    /// Emitted runs across every touched row.
    pub runs: u64,
    /// Delta literals across every run — run lengths summed, interior
    /// zeros (bridged gaps `<= max_gap`) included.
    pub cells: u64,
    /// Nonzero cells — the size of the sparse edit set `S`; the `Θ(τ·w)`
    /// quantity the paper's asymptotics count.
    pub nonzero_cells: u64,
    /// Total encoded length in bits, before byte padding.
    pub bits: u64,
    /// `⌈bits / 8⌉`; equals [`HintDeltaBundle::encode`]`().len()`.
    pub bytes: usize,
}

/// Rejection reasons for [`HintDeltaBundle::decode`].
///
/// # Purpose
///
/// One variant per validation class in `docs/hint-delta-wire-format.md`
/// §7. The offending value is carried where doing so is cheap (it was
/// already in hand at the point of rejection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireError {
    /// The bit stream ended before a field could be read in full.
    Truncated,
    /// A row's `bucket` field was `>= n_b`.
    BucketOutOfRange {
        /// The out-of-range bucket value read from the stream.
        bucket: u32,
    },
    /// A run's `start` was `>= ω`, or `start + len` exceeded `ω`.
    OffsetOutOfRange {
        /// The out-of-range offset (either `start` or `start + len`).
        offset: u32,
    },
    /// A delta code point was the reserved value `2p - 1` (all `DB` bits
    /// set).
    DeltaOutOfRange {
        /// The invalid code point.
        code: u64,
    },
    /// Rows were not in strictly ascending `bucket` order (includes a
    /// repeated bucket).
    NonCanonicalRows,
    /// A row's runs violated §5 rule 2: not strictly ascending, closer
    /// together than `G + 1` zero cells, a run's first or last delta was
    /// zero, or an interior zero stretch exceeded `G`.
    NonCanonicalRuns,
    /// A padding bit after the last field was set.
    NonZeroPadding,
    /// Bytes remained in the input after the padded encoding.
    TrailingBytes,
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => {
                write!(
                    f,
                    "hint-delta wire stream ended before a field was complete"
                )
            }
            Self::BucketOutOfRange { bucket } => {
                write!(f, "hint-delta wire: bucket {bucket} out of range")
            }
            Self::OffsetOutOfRange { offset } => {
                write!(f, "hint-delta wire: offset {offset} out of range")
            }
            Self::DeltaOutOfRange { code } => {
                write!(
                    f,
                    "hint-delta wire: delta code point {code} is the reserved value"
                )
            }
            Self::NonCanonicalRows => {
                write!(f, "hint-delta wire: rows are not in canonical order")
            }
            Self::NonCanonicalRuns => {
                write!(f, "hint-delta wire: runs are not in canonical form")
            }
            Self::NonZeroPadding => {
                write!(
                    f,
                    "hint-delta wire: non-zero padding bit after the last field"
                )
            }
            Self::TrailingBytes => {
                write!(
                    f,
                    "hint-delta wire: trailing bytes after the padded encoding"
                )
            }
        }
    }
}

impl std::error::Error for WireError {}

/// LSB-first bit-packing writer over a pre-sized, zero-initialised byte
/// buffer (`docs/hint-delta-wire-format.md` §3).
struct BitWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}

impl BitWriter {
    /// Allocate a zeroed buffer of exactly `byte_len` bytes. Padding bits
    /// are therefore zero by construction, with nothing further to write.
    fn new(byte_len: usize) -> Self {
        Self {
            bytes: vec![0u8; byte_len],
            bit_pos: 0,
        }
    }

    /// Write the low `width` bits of `value`, LSB first; a `width` of `0`
    /// writes nothing.
    fn write_bits(&mut self, value: u64, width: u32) {
        for j in 0..width {
            if (value >> j) & 1 == 1 {
                let idx = self.bit_pos + j as usize;
                self.bytes[idx / 8] |= 1 << (idx % 8);
            }
        }
        self.bit_pos += width as usize;
    }

    /// Write a single flag bit.
    fn write_bit(&mut self, bit: bool) {
        self.write_bits(u64::from(bit), 1);
    }

    /// Consume the writer, returning the packed buffer.
    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// LSB-first bit-packing reader over a borrowed byte slice
/// (`docs/hint-delta-wire-format.md` §3). Every read is bounds-checked
/// against the slice before any byte is indexed.
struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
    bit_len: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            bit_pos: 0,
            bit_len: bytes.len() * 8,
        }
    }

    /// Read `width` bits (LSB first) as a `u64`; a `width` of `0` reads
    /// nothing and returns `0`. Bounds are checked before any index into
    /// `self.bytes` is computed.
    fn read_bits(&mut self, width: u32) -> Result<u64, WireError> {
        if width == 0 {
            return Ok(0);
        }
        if self.bit_pos + width as usize > self.bit_len {
            return Err(WireError::Truncated);
        }
        let mut v: u64 = 0;
        for j in 0..width {
            let idx = self.bit_pos + j as usize;
            let bit = (self.bytes[idx / 8] >> (idx % 8)) & 1;
            v |= u64::from(bit) << j;
        }
        self.bit_pos += width as usize;
        Ok(v)
    }
}

/// Split one row's sorted, validated nonzero cells into the maximal runs
/// `docs/hint-delta-wire-format.md` §5 rule 2 defines: consecutive cells
/// whose offsets are at most `max_gap + 1` apart merge into one run.
/// Shared by [`HintDeltaBundle::encode`] and [`HintDeltaBundle::wire_stats`]
/// so the two can never disagree on where a run starts or ends.
///
/// `cells` must already be canonical (strictly ascending offsets, all
/// nonzero, in range) — callers validate that first via
/// [`HintDeltaBundle::validate`].
///
/// # Returns
///
/// One `(start_offset, run_len, nonzero_slice)` per run, where
/// `nonzero_slice` is the sub-slice of `cells` the run covers (`run_len`
/// may exceed `nonzero_slice.len()` when the run bridges interior zero
/// gaps).
fn row_runs(cells: &[(u16, i64)], max_gap: u32) -> impl Iterator<Item = (u16, u16, &[(u16, i64)])> {
    cells
        .chunk_by(move |a: &(u16, i64), b: &(u16, i64)| {
            u32::from(b.0) - u32::from(a.0) - 1 <= max_gap
        })
        .map(|slice| {
            let start = slice[0].0;
            let end = slice[slice.len() - 1].0;
            (start, end - start + 1, slice)
        })
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
/// # Design / architecture
///
/// This bundle has a normative, bit-packed on-wire encoding —
/// `docs/hint-delta-wire-format.md` — via [`Self::encode`] / [`Self::decode`].
/// [`Self::wire_byte_size`] equals `self.encode().len()` by construction:
/// both are computed from the same [`DeltaWireLayout`] and the same
/// `row_runs` splitting rule.
///
/// # Rationale
///
/// Wire size is proportional to the mutation footprint (one mutated row
/// per touched bucket × its touched cells, run-length-encoded), not the
/// database size. Per-mutation deltas inherit the cell-width geometry of
/// the [`ServerSetupBundle`] they patch; `plaintext_bits` (and the rest of
/// the SCF geometry) must remain constant across the bundle stream — see
/// [`ServerSetupBundle`] for the invariant statement and upgrade path.
/// That geometry now travels with the bundle itself, in `params`, purely
/// so [`Self::encode`] / [`Self::wire_stats`] can derive the same
/// [`DeltaWireLayout`] the receiver's `decode` call is given explicitly;
/// `params` is never itself part of the wire encoding (`docs/hint-delta-wire-format.md`
/// §2).
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
    /// Geometry this bundle was folded under — the source of the wire
    /// widths in [`Self::layout`]. Never itself on the wire: `encode`
    /// derives the widths from it, and `decode` takes the receiver's own
    /// copy as an argument instead of reading one back.
    pub params: CuckooParams,
    _marker: PhantomData<B>,
}

impl<B: IndexPirBackend> HintDeltaBundle<B> {
    /// Internal constructor used by `IkpirServer::commit_mutations`; end
    /// users receive the bundle from `insert` / `update` / `delete` rather
    /// than build it themselves.
    #[doc(hidden)]
    pub const fn new(
        epoch: u64,
        per_segment_row_deltas: Vec<SegmentRowDeltas>,
        params: CuckooParams,
    ) -> Self {
        Self {
            epoch,
            per_segment_row_deltas,
            params,
            _marker: PhantomData,
        }
    }

    /// The wire-encoding widths this bundle's geometry implies.
    ///
    /// # Purpose
    ///
    /// Single source of truth for `IB` / `OW` / `DB` / `G`
    /// (`docs/hint-delta-wire-format.md` §2), derived from `self.params`.
    ///
    /// # Returns
    ///
    /// A [`DeltaWireLayout`] with every width field populated; a field is
    /// `0` exactly when its domain has one element.
    pub fn layout(&self) -> DeltaWireLayout {
        DeltaWireLayout::for_params(&self.params)
    }

    /// Check the bundle is in the canonical form [`Self::encode`] requires
    /// (`docs/hint-delta-wire-format.md` §5 and §8).
    ///
    /// # Panics
    ///
    /// Panics, naming the offending field, if any of: the segment count
    /// doesn't equal `params.arity()`; a row is `>= segment_size`; rows
    /// within a segment are not strictly ascending; a row carries no
    /// cells; an offset is `>= row_width`; offsets within a row are not
    /// strictly ascending; a cell's delta is `0`; or `|delta| >=
    /// 2^plaintext_bits`.
    fn validate(&self, layout: &DeltaWireLayout) {
        let arity = self.params.arity();
        assert_eq!(
            self.per_segment_row_deltas.len(),
            arity,
            "HintDeltaBundle::encode: per_segment_row_deltas.len() ({}) != arity ({arity})",
            self.per_segment_row_deltas.len()
        );
        let p_minus_1: i64 = (1i64 << layout.plaintext_bits) - 1;
        for (j, seg) in self.per_segment_row_deltas.iter().enumerate() {
            let mut prev_row: Option<u32> = None;
            for (row, cells) in seg {
                assert!(
                    *row < layout.segment_size,
                    "HintDeltaBundle::encode: row {row} in segment {j} out of range (segment_size = {})",
                    layout.segment_size
                );
                if let Some(pr) = prev_row {
                    assert!(
                        *row > pr,
                        "HintDeltaBundle::encode: rows not strictly ascending in segment {j}: row {row} after row {pr}"
                    );
                }
                prev_row = Some(*row);
                assert!(
                    !cells.is_empty(),
                    "HintDeltaBundle::encode: row {row} in segment {j} has no cells (empty rows must be dropped by the fold)"
                );
                let mut prev_off: Option<u16> = None;
                for (offset, delta) in cells {
                    assert!(
                        u32::from(*offset) < layout.row_width,
                        "HintDeltaBundle::encode: offset {offset} in row {row} segment {j} out of range (row_width = {})",
                        layout.row_width
                    );
                    if let Some(po) = prev_off {
                        assert!(
                            *offset > po,
                            "HintDeltaBundle::encode: offsets not strictly ascending in row {row} segment {j}: offset {offset} after offset {po}"
                        );
                    }
                    prev_off = Some(*offset);
                    assert_ne!(
                        *delta, 0,
                        "HintDeltaBundle::encode: zero delta at (segment {j}, row {row}, offset {offset}) — the fold must drop zero deltas"
                    );
                    assert!(
                        delta.unsigned_abs() <= p_minus_1.unsigned_abs(),
                        "HintDeltaBundle::encode: delta γ = {delta} at (segment {j}, row {row}, offset {offset}) out of range (|γ| < {})",
                        1u64 << layout.plaintext_bits
                    );
                }
            }
        }
    }

    /// Row/run/cell accounting and exact wire size for this bundle.
    ///
    /// # Purpose
    ///
    /// Walks the bundle with the same run-splitting rule [`Self::encode`]
    /// uses (`row_runs`, `docs/hint-delta-wire-format.md` §5 rule 2) and
    /// reports the resulting counts plus the closed-form bit/byte length
    /// from §6.
    ///
    /// # Returns
    ///
    /// A [`DeltaWireStats`] with `bytes == self.encode().len()`.
    ///
    /// # Panics
    ///
    /// See `Self::validate`.
    pub fn wire_stats(&self) -> DeltaWireStats {
        let layout = self.layout();
        self.validate(&layout);
        let mut rows: u64 = 0;
        let mut runs: u64 = 0;
        let mut cells: u64 = 0;
        let mut nonzero_cells: u64 = 0;
        for seg in &self.per_segment_row_deltas {
            for (_row, row_cells) in seg {
                rows += 1;
                nonzero_cells += row_cells.len() as u64;
                for (_start, len, _slice) in row_runs(row_cells, layout.max_gap) {
                    runs += 1;
                    cells += u64::from(len);
                }
            }
        }
        let bits = layout.bits(rows, runs, cells);
        let bytes = layout.bytes(rows, runs, cells) as usize;
        DeltaWireStats {
            rows,
            runs,
            cells,
            nonzero_cells,
            bits,
            bytes,
        }
    }

    /// Exact on-wire byte size of `self.encode()`.
    ///
    /// # Returns
    ///
    /// `self.wire_stats().bytes`.
    ///
    /// # Panics
    ///
    /// See `Self::validate`.
    pub fn wire_byte_size(&self) -> usize {
        self.wire_stats().bytes
    }

    /// Encode this bundle per `docs/hint-delta-wire-format.md` §3–§5, §8.
    ///
    /// # Returns
    ///
    /// The bit-packed, LSB-first byte encoding — exactly
    /// `self.wire_stats().bytes` long.
    ///
    /// # Panics
    ///
    /// See `Self::validate` (`docs/hint-delta-wire-format.md` §8): a
    /// violation is a bug in the caller (the fold) and must be loud, never
    /// silently truncated or wrapped.
    pub fn encode(&self) -> Vec<u8> {
        let stats = self.wire_stats();
        let layout = self.layout();
        let p_minus_1: i64 = (1i64 << layout.plaintext_bits) - 1;

        let mut w = BitWriter::new(stats.bytes);
        w.write_bits(self.epoch, 64);
        for (j, seg) in self.per_segment_row_deltas.iter().enumerate() {
            for (row, cells) in seg {
                w.write_bit(true);
                let bucket = j as u32 * layout.segment_size + row;
                w.write_bits(u64::from(bucket), layout.bucket_bits);

                let mut first_run = true;
                for (start, len, slice) in row_runs(cells, layout.max_gap) {
                    if !first_run {
                        w.write_bit(true);
                    }
                    first_run = false;
                    w.write_bits(u64::from(start), layout.offset_bits);
                    w.write_bits(u64::from(len - 1), layout.offset_bits);

                    let mut idx = 0usize;
                    for k in 0..len {
                        let offset = start + k;
                        let delta = if idx < slice.len() && slice[idx].0 == offset {
                            let d = slice[idx].1;
                            idx += 1;
                            d
                        } else {
                            0
                        };
                        let code = (delta + p_minus_1) as u64;
                        w.write_bits(code, layout.delta_bits);
                    }
                }
                w.write_bit(false); // end this row's run list
            }
        }
        w.write_bit(false); // end the bundle's row list

        let bytes = w.finish();
        debug_assert_eq!(
            bytes.len(),
            stats.bytes,
            "HintDeltaBundle::encode: encoded length must equal wire_stats().bytes"
        );
        bytes
    }

    /// Decode a bundle previously produced by [`Self::encode`] under the
    /// same `params`.
    ///
    /// # Arguments
    ///
    /// - `bytes`  — the encoded byte stream; a trust boundary
    ///   (`docs/hint-delta-wire-format.md` §7).
    /// - `params` — the receiver's geometry; supplies the same widths the
    ///   encoder used. Carried unchanged into the returned bundle's
    ///   `params` field.
    ///
    /// # Returns
    ///
    /// `Ok(bundle)` with literal zero deltas dropped, so the result is the
    /// sparse in-memory form [`Self::encode`] was built from.
    ///
    /// # Errors
    ///
    /// See [`WireError`] — every check from §7 is applied, with bounds
    /// checked before any index into `bytes` is used. Row and run counts
    /// are never trusted ahead of the bits that justify them: each row
    /// costs at least `1 + IB` bits and each run at least `2·OW + 1`, so
    /// the amount of state this function can build is bounded by
    /// `bytes.len()`, not by an untrusted length field.
    pub fn decode(bytes: &[u8], params: CuckooParams) -> Result<Self, WireError> {
        let layout = DeltaWireLayout::for_params(&params);
        let arity = params.arity();
        let p: u64 = 1u64 << layout.plaintext_bits;
        let max_code = 2 * p - 2;
        let p_minus_1: i64 = (p - 1) as i64;

        let mut r = BitReader::new(bytes);
        let epoch = r.read_bits(64)?;

        let mut per_segment_row_deltas: Vec<SegmentRowDeltas> =
            (0..arity).map(|_| Vec::new()).collect();
        let mut prev_bucket: Option<u32> = None;

        loop {
            if r.read_bits(1)? == 0 {
                break; // end of bundle's row list
            }

            let bucket = r.read_bits(layout.bucket_bits)? as u32;
            if bucket >= layout.num_buckets {
                return Err(WireError::BucketOutOfRange { bucket });
            }
            if let Some(prev) = prev_bucket {
                if bucket <= prev {
                    return Err(WireError::NonCanonicalRows);
                }
            }
            prev_bucket = Some(bucket);

            let seg = (bucket / layout.segment_size) as usize;
            let row = bucket % layout.segment_size;

            let mut cells: Vec<(u16, i64)> = Vec::new();
            let mut prev_run_end_excl: Option<u32> = None;

            loop {
                let start = r.read_bits(layout.offset_bits)? as u32;
                if start >= layout.row_width {
                    return Err(WireError::OffsetOutOfRange { offset: start });
                }
                let len_minus_1 = r.read_bits(layout.offset_bits)? as u32;
                let len = len_minus_1 + 1;
                let end_excl = start + len;
                if end_excl > layout.row_width {
                    return Err(WireError::OffsetOutOfRange { offset: end_excl });
                }
                if let Some(prev_end) = prev_run_end_excl {
                    if start < prev_end {
                        return Err(WireError::NonCanonicalRuns);
                    }
                    let gap = start - prev_end;
                    if gap <= layout.max_gap {
                        return Err(WireError::NonCanonicalRuns);
                    }
                }
                prev_run_end_excl = Some(end_excl);

                let mut zero_run: u32 = 0;
                let mut max_zero_run: u32 = 0;
                let mut boundary_zero = false;
                for k in 0..len {
                    let code = r.read_bits(layout.delta_bits)?;
                    if code > max_code {
                        return Err(WireError::DeltaOutOfRange { code });
                    }
                    let delta = code as i64 - p_minus_1;
                    if delta == 0 {
                        if k == 0 || k == len - 1 {
                            boundary_zero = true;
                        }
                        zero_run += 1;
                        max_zero_run = max_zero_run.max(zero_run);
                    } else {
                        zero_run = 0;
                        let offset = start + k;
                        // `ω` fits a `u16` for every geometry the fold can
                        // produce; a wider one would have been an encoder-side
                        // panic, so reject rather than truncate here.
                        let offset = u16::try_from(offset)
                            .map_err(|_| WireError::OffsetOutOfRange { offset })?;
                        cells.push((offset, delta));
                    }
                }
                if boundary_zero {
                    return Err(WireError::NonCanonicalRuns);
                }
                if max_zero_run > layout.max_gap {
                    return Err(WireError::NonCanonicalRuns);
                }

                if r.read_bits(1)? == 0 {
                    break; // end this row's run list
                }
            }

            per_segment_row_deltas[seg].push((row, cells));
        }

        // Padding: zero bits out to the next byte boundary, then no
        // trailing bytes.
        let pad = (8 - (r.bit_pos % 8)) % 8;
        for _ in 0..pad {
            if r.read_bits(1)? != 0 {
                return Err(WireError::NonZeroPadding);
            }
        }
        if r.bit_pos / 8 != bytes.len() {
            return Err(WireError::TrailingBytes);
        }

        Ok(Self {
            epoch,
            per_segment_row_deltas,
            params,
            _marker: PhantomData,
        })
    }
}

impl<B: IndexPirBackend> fmt::Debug for HintDeltaBundle<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HintDeltaBundle")
            .field("epoch", &self.epoch)
            .field("per_segment_row_deltas", &self.per_segment_row_deltas)
            .field("params", &self.params)
            .finish()
    }
}

impl<B: IndexPirBackend> PartialEq for HintDeltaBundle<B> {
    fn eq(&self, other: &Self) -> bool {
        self.epoch == other.epoch
            && self.per_segment_row_deltas == other.per_segment_row_deltas
            && self.params == other.params
    }
}

impl<B: IndexPirBackend> Clone for HintDeltaBundle<B> {
    fn clone(&self) -> Self {
        Self {
            epoch: self.epoch,
            per_segment_row_deltas: self.per_segment_row_deltas.clone(),
            params: self.params,
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit + property tests pinning `docs/hint-delta-wire-format.md`.
    //!
    //! Each test is named for the property or bug it targets, per the
    //! implementation plan: width table (§2), hand-built round trips
    //! (§3–§6), delta boundary/off-by-one behaviour (§4, §8), encoder
    //! preconditions (§8), decoder rejections (§7), a randomized
    //! round-trip property, and a hand-computed closed-form check (§6).

    use super::*;
    use crate::FrodoPirBackend;
    use proptest::prelude::*;
    use segmented_cuckoo::SchemeKind;

    type Bundle = HintDeltaBundle<FrodoPirBackend>;

    fn build(params: CuckooParams, epoch: u64, segs: Vec<SegmentRowDeltas>) -> Bundle {
        Bundle::new(epoch, segs, params)
    }

    fn delta_code(delta: i64, params: &CuckooParams) -> u64 {
        (delta + (1i64 << params.plaintext_bits) - 1) as u64
    }

    /// Hand-pack a sequence of `(value, width)` fields into a byte stream,
    /// zero-padded to a byte boundary — used to craft intentionally
    /// invalid streams `encode` could never itself produce.
    fn raw_pack(fields: &[(u64, u32)]) -> Vec<u8> {
        let total_bits: usize = fields.iter().map(|(_, w)| *w as usize).sum();
        let byte_len = total_bits.div_ceil(8);
        let mut w = BitWriter::new(byte_len);
        for &(value, width) in fields {
            w.write_bits(value, width);
        }
        w.finish()
    }

    /// `SchemeKind` is `#[non_exhaustive]`, so dispatch on it from outside
    /// `segmented-cuckoo` via `CuckooParams::arity()` rather than a local
    /// match (which the compiler would then require a wildcard arm for).
    fn arity_of(scheme: SchemeKind) -> u32 {
        CuckooParams {
            scheme_kind: scheme,
            num_buckets: 0,
            bucket_size: 0,
            fingerprint_bits: 0,
            value_bits: 0,
            plaintext_bits: 0,
        }
        .arity() as u32
    }

    /// A small fixture with every ω a power of two (ρ = 4, ω = 4, OW = 2,
    /// DB = 5, G = 1) — used by the hand-built round-trip / precondition
    /// tests, where exact-power-of-two widths are harmless.
    fn small_params(scheme: SchemeKind) -> CuckooParams {
        let arity = arity_of(scheme);
        CuckooParams {
            scheme_kind: scheme,
            num_buckets: arity * 4, // segment_size == 4 for every arity
            bucket_size: 2,
            fingerprint_bits: 4,
            value_bits: 4,
            plaintext_bits: 4,
        }
    }

    /// A fixture whose `n_b` and `ω` are deliberately *not* powers of two,
    /// so their bit-field widths leave room for an out-of-range code point
    /// — needed to craft `BucketOutOfRange` / `OffsetOutOfRange` streams.
    /// `G == 0` here, which also gives the simplest possible "runs too
    /// close" fixture.
    fn odd_params() -> CuckooParams {
        CuckooParams {
            scheme_kind: SchemeKind::Segmented2ary,
            num_buckets: 6, // segment_size = 3, IB = bitlen(5) = 3 (2^3 = 8 > 6)
            bucket_size: 3, // cells_per_slot = 1 => ω = 3, OW = bitlen(2) = 2 (2^2 = 4 > 3)
            fingerprint_bits: 4,
            value_bits: 4,
            plaintext_bits: 8, // DB = 9, G = (2*2+1)/9 = 0
        }
    }

    fn params_with_pb(pb: u32) -> CuckooParams {
        CuckooParams {
            scheme_kind: SchemeKind::Segmented2ary,
            num_buckets: 8,
            bucket_size: 2,
            fingerprint_bits: 8,
            value_bits: 8,
            plaintext_bits: pb,
        }
    }

    // ── 1. Width table pinned at paper geometry ────────────────────────

    #[test]
    fn width_table_matches_paper_geometry() {
        struct Row {
            arity: SchemeKind,
            num_buckets: u32,
            bucket_size: u32,
            value_bits: u32,
            plaintext_bits: u32,
            expect_ib: u32,
            expect_ow: u32,
            expect_db: u32,
            expect_g: u32,
        }
        let fingerprint_bits = 64u32;
        let rows = [
            Row {
                arity: SchemeKind::Segmented2ary,
                num_buckets: 1 << 18,
                bucket_size: 4,
                value_bits: 2048,
                plaintext_bits: 9,
                expect_ib: 18,
                expect_ow: 10,
                expect_db: 10,
                expect_g: 2,
            },
            Row {
                arity: SchemeKind::Segmented2ary,
                num_buckets: 1 << 18,
                bucket_size: 4,
                value_bits: 8192,
                plaintext_bits: 9,
                expect_ib: 18,
                expect_ow: 12,
                expect_db: 10,
                expect_g: 2,
            },
            Row {
                // RisePIR-S, ℓ = 1 kB: ω = 4128 -> OW = 13, G = 3.
                arity: SchemeKind::Segmented2ary,
                num_buckets: 1 << 18,
                bucket_size: 4,
                value_bits: 8192,
                plaintext_bits: 8,
                expect_ib: 18,
                expect_ow: 13,
                expect_db: 9,
                expect_g: 3,
            },
            Row {
                arity: SchemeKind::Segmented4ary,
                num_buckets: 1 << 20,
                bucket_size: 1,
                value_bits: 2048,
                plaintext_bits: 9,
                expect_ib: 20,
                expect_ow: 8,
                expect_db: 10,
                expect_g: 1,
            },
            Row {
                arity: SchemeKind::Segmented4ary,
                num_buckets: 1 << 19,
                bucket_size: 2,
                value_bits: 2048,
                plaintext_bits: 9,
                expect_ib: 19,
                expect_ow: 9,
                expect_db: 10,
                expect_g: 1,
            },
            Row {
                arity: SchemeKind::Segmented3ary,
                num_buckets: 3 << 18,
                bucket_size: 2,
                value_bits: 2048,
                plaintext_bits: 9,
                expect_ib: 20,
                expect_ow: 9,
                expect_db: 10,
                expect_g: 1,
            },
            Row {
                arity: SchemeKind::Segmented3ary,
                num_buckets: 3 << 17,
                bucket_size: 3,
                value_bits: 2048,
                plaintext_bits: 9,
                expect_ib: 19,
                expect_ow: 10,
                expect_db: 10,
                expect_g: 2,
            },
        ];
        for r in rows {
            let params = CuckooParams {
                scheme_kind: r.arity,
                num_buckets: r.num_buckets,
                bucket_size: r.bucket_size,
                fingerprint_bits,
                value_bits: r.value_bits,
                plaintext_bits: r.plaintext_bits,
            };
            let layout = DeltaWireLayout::for_params(&params);
            assert_eq!(
                layout.bucket_bits, r.expect_ib,
                "IB mismatch for {params:?}"
            );
            assert_eq!(
                layout.offset_bits, r.expect_ow,
                "OW mismatch for {params:?}"
            );
            assert_eq!(layout.delta_bits, r.expect_db, "DB mismatch for {params:?}");
            assert_eq!(layout.max_gap, r.expect_g, "G mismatch for {params:?}");
        }
    }

    // ── 2. Hand-built round trips ───────────────────────────────────────

    #[test]
    fn empty_bundle_round_trips_and_is_nine_bytes() {
        let p = small_params(SchemeKind::Segmented2ary);
        for epoch in [0u64, 1, u64::MAX] {
            let b = build(p, epoch, vec![Vec::new(); p.arity()]);
            let bytes = b.encode();
            assert_eq!(
                bytes.len(),
                9,
                "epoch(64) + terminator(1), padded, is 9 bytes"
            );
            assert_eq!(b.wire_byte_size(), 9);
            let decoded = Bundle::decode(&bytes, p).unwrap();
            assert_eq!(decoded, b);
            assert_eq!(decoded.epoch, epoch);
        }
    }

    #[test]
    fn single_row_single_run_round_trips() {
        let p = small_params(SchemeKind::Segmented2ary);
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((1u32, vec![(0u16, 3i64), (1u16, -2i64)]));
        let b = build(p, 5, segs);
        let bytes = b.encode();
        assert_eq!(bytes.len(), b.wire_byte_size());
        assert_eq!(Bundle::decode(&bytes, p).unwrap(), b);
    }

    #[test]
    fn gap_equal_to_max_gap_merges_into_one_run() {
        let p = small_params(SchemeKind::Segmented2ary);
        let layout = DeltaWireLayout::for_params(&p);
        assert_eq!(
            layout.max_gap, 1,
            "fixture sanity: this test targets G == 1"
        );
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((0u32, vec![(0u16, 3i64), (2u16, -2i64)])); // gap = 2-0-1 = 1 = G
        let b = build(p, 1, segs);
        let stats = b.wire_stats();
        assert_eq!(stats.runs, 1, "gap == G must merge into a single run");
        assert_eq!(
            stats.cells, 3,
            "run covers offsets 0,1,2: 2 nonzero + 1 zero literal"
        );
        assert_eq!(stats.nonzero_cells, 2);
        let bytes = b.encode();
        assert_eq!(bytes.len(), b.wire_byte_size());
        assert_eq!(Bundle::decode(&bytes, p).unwrap(), b);
    }

    #[test]
    fn gap_greater_than_max_gap_splits_into_two_runs() {
        let p = small_params(SchemeKind::Segmented2ary);
        let layout = DeltaWireLayout::for_params(&p);
        assert_eq!(
            layout.max_gap, 1,
            "fixture sanity: this test targets G == 1"
        );
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((0u32, vec![(0u16, 3i64), (3u16, -2i64)])); // gap = 3-0-1 = 2 > G
        let b = build(p, 1, segs);
        let stats = b.wire_stats();
        assert_eq!(stats.runs, 2, "gap > G must split into two runs");
        assert_eq!(stats.cells, 2);
        let bytes = b.encode();
        assert_eq!(bytes.len(), b.wire_byte_size());
        assert_eq!(Bundle::decode(&bytes, p).unwrap(), b);
    }

    fn round_trip_boundaries(p: CuckooParams) {
        let arity = p.arity();
        let layout = DeltaWireLayout::for_params(&p);
        let rho = layout.segment_size;
        let omega = layout.row_width;
        let mut segs = vec![Vec::new(); arity];
        segs[0].push((0u32, vec![(0u16, 1i64)]));
        segs[0].push((rho - 1, vec![((omega - 1) as u16, -1i64)]));
        segs[arity - 1].push((0u32, vec![(0u16, 2i64), (1u16, -3i64)]));
        segs[arity - 1].push((rho - 1, vec![((omega - 1) as u16, 1i64)]));
        let b = build(p, 42, segs);
        let bytes = b.encode();
        assert_eq!(bytes.len(), b.wire_byte_size());
        assert_eq!(Bundle::decode(&bytes, p).unwrap(), b);
    }

    #[test]
    fn round_trip_boundary_rows_and_offsets_2ary() {
        round_trip_boundaries(small_params(SchemeKind::Segmented2ary));
    }

    #[test]
    fn round_trip_boundary_rows_and_offsets_3ary() {
        round_trip_boundaries(small_params(SchemeKind::Segmented3ary));
    }

    #[test]
    fn round_trip_boundary_rows_and_offsets_4ary() {
        round_trip_boundaries(small_params(SchemeKind::Segmented4ary));
    }

    // ── 3. Off-by-one delta boundary behaviour ──────────────────────────

    #[test]
    fn delta_boundary_values_round_trip_at_several_plaintext_bits() {
        for pb in [8u32, 9, 10] {
            let p = params_with_pb(pb);
            let layout = DeltaWireLayout::for_params(&p);
            assert_eq!(layout.delta_bits, pb + 1);
            let modulus = 1i64 << pb;
            for gamma in [-(modulus - 1), -1, 1, modulus - 1] {
                let mut segs = vec![Vec::new(); p.arity()];
                segs[0].push((0u32, vec![(0u16, gamma)]));
                let b = build(p, 1, segs);
                let bytes = b.encode();
                assert_eq!(
                    Bundle::decode(&bytes, p).unwrap(),
                    b,
                    "gamma = {gamma} at pb = {pb} must round-trip"
                );
            }
        }
    }

    #[test]
    #[should_panic]
    fn encode_panics_on_delta_equal_to_plus_p() {
        let p = params_with_pb(9);
        let modulus = 1i64 << 9;
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((0u32, vec![(0u16, modulus)]));
        build(p, 1, segs).encode();
    }

    #[test]
    #[should_panic]
    fn encode_panics_on_delta_equal_to_minus_p() {
        let p = params_with_pb(9);
        let modulus = 1i64 << 9;
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((0u32, vec![(0u16, -modulus)]));
        build(p, 1, segs).encode();
    }

    #[test]
    fn decode_rejects_reserved_delta_code_point() {
        let p = params_with_pb(9);
        let layout = DeltaWireLayout::for_params(&p);
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((0u32, vec![(0u16, 1i64)]));
        let bytes = build(p, 1, segs).encode();
        let mut bytes = bytes;

        // Patch the single delta field to the reserved code point
        // u = 2p - 1 (all DB bits set). Layout up to that field:
        // epoch(64) row_flag(1) bucket(IB) start(OW) len_minus_1(OW).
        let delta_bit_offset =
            64 + 1 + layout.bucket_bits as usize + 2 * layout.offset_bits as usize;
        let db = layout.delta_bits as usize;
        for j in 0..db {
            let idx = delta_bit_offset + j;
            bytes[idx / 8] |= 1 << (idx % 8);
        }
        let err = Bundle::decode(&bytes, p).unwrap_err();
        let expected_code = (1u64 << db) - 1;
        assert_eq!(
            err,
            WireError::DeltaOutOfRange {
                code: expected_code
            }
        );
    }

    // ── 4. Encoder preconditions ─────────────────────────────────────────

    #[test]
    #[should_panic]
    fn encode_panics_on_zero_delta_in_sparse_input() {
        let p = small_params(SchemeKind::Segmented2ary);
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((0u32, vec![(0u16, 0i64)]));
        build(p, 1, segs).encode();
    }

    #[test]
    #[should_panic]
    fn encode_panics_on_unsorted_rows() {
        let p = small_params(SchemeKind::Segmented2ary);
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((2u32, vec![(0u16, 1i64)]));
        segs[0].push((1u32, vec![(0u16, 1i64)]));
        build(p, 1, segs).encode();
    }

    #[test]
    #[should_panic]
    fn encode_panics_on_duplicate_row() {
        let p = small_params(SchemeKind::Segmented2ary);
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((1u32, vec![(0u16, 1i64)]));
        segs[0].push((1u32, vec![(1u16, 1i64)]));
        build(p, 1, segs).encode();
    }

    #[test]
    #[should_panic]
    fn encode_panics_on_unsorted_offsets() {
        let p = small_params(SchemeKind::Segmented2ary);
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((0u32, vec![(2u16, 1i64), (1u16, 1i64)]));
        build(p, 1, segs).encode();
    }

    #[test]
    #[should_panic]
    fn encode_panics_on_duplicate_offset() {
        let p = small_params(SchemeKind::Segmented2ary);
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((0u32, vec![(1u16, 1i64), (1u16, 2i64)]));
        build(p, 1, segs).encode();
    }

    #[test]
    #[should_panic]
    fn encode_panics_on_offset_out_of_range() {
        let p = small_params(SchemeKind::Segmented2ary);
        let layout = DeltaWireLayout::for_params(&p);
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((0u32, vec![(layout.row_width as u16, 1i64)])); // == omega
        build(p, 1, segs).encode();
    }

    #[test]
    #[should_panic]
    fn encode_panics_on_row_out_of_range() {
        let p = small_params(SchemeKind::Segmented2ary);
        let layout = DeltaWireLayout::for_params(&p);
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((layout.segment_size, vec![(0u16, 1i64)])); // == rho
        build(p, 1, segs).encode();
    }

    #[test]
    #[should_panic]
    fn encode_panics_on_wrong_segment_count() {
        let p = small_params(SchemeKind::Segmented2ary);
        let segs = vec![Vec::new(); p.arity() + 1];
        build(p, 1, segs).encode();
    }

    // ── 5. Decoder rejections ────────────────────────────────────────────

    #[test]
    fn decode_rejects_bucket_out_of_range() {
        let p = odd_params();
        let layout = DeltaWireLayout::for_params(&p);
        let bytes = raw_pack(&[
            (0, 64),
            (1, 1),
            (u64::from(layout.num_buckets), layout.bucket_bits), // == n_b
        ]);
        let err = Bundle::decode(&bytes, p).unwrap_err();
        assert_eq!(
            err,
            WireError::BucketOutOfRange {
                bucket: layout.num_buckets
            }
        );
    }

    #[test]
    fn decode_rejects_start_out_of_range() {
        let p = odd_params();
        let layout = DeltaWireLayout::for_params(&p);
        let bytes = raw_pack(&[
            (0, 64),
            (1, 1),
            (0, layout.bucket_bits),
            (u64::from(layout.row_width), layout.offset_bits), // == omega
        ]);
        let err = Bundle::decode(&bytes, p).unwrap_err();
        assert_eq!(
            err,
            WireError::OffsetOutOfRange {
                offset: layout.row_width
            }
        );
    }

    #[test]
    fn decode_rejects_run_extending_past_row_width() {
        let p = odd_params();
        let layout = DeltaWireLayout::for_params(&p);
        // start = 1 (valid, omega = 3), len - 1 = 2 => len = 3, start+len = 4 > 3.
        let bytes = raw_pack(&[
            (0, 64),
            (1, 1),
            (0, layout.bucket_bits),
            (1, layout.offset_bits),
            (2, layout.offset_bits),
        ]);
        let err = Bundle::decode(&bytes, p).unwrap_err();
        assert_eq!(err, WireError::OffsetOutOfRange { offset: 4 });
    }

    #[test]
    fn decode_rejects_rows_not_strictly_ascending() {
        let p = odd_params();
        let layout = DeltaWireLayout::for_params(&p);
        let bytes = raw_pack(&[
            (0, 64),
            (1, 1),
            (1, layout.bucket_bits), // row1: bucket = 1
            (0, layout.offset_bits),
            (0, layout.offset_bits),
            (delta_code(1, &p), layout.delta_bits),
            (0, 1),                  // end row1's runs
            (1, 1),                  // another row follows
            (1, layout.bucket_bits), // row2: bucket = 1, not > 1
        ]);
        let err = Bundle::decode(&bytes, p).unwrap_err();
        assert_eq!(err, WireError::NonCanonicalRows);
    }

    #[test]
    fn decode_rejects_runs_closer_than_max_gap_plus_one() {
        let p = odd_params();
        let layout = DeltaWireLayout::for_params(&p);
        assert_eq!(
            layout.max_gap, 0,
            "fixture sanity: this test targets G == 0"
        );
        let bytes = raw_pack(&[
            (0, 64),
            (1, 1),
            (0, layout.bucket_bits),
            (0, layout.offset_bits), // run1 start = 0
            (0, layout.offset_bits), // run1 len - 1 = 0 (covers offset 0)
            (delta_code(1, &p), layout.delta_bits),
            (1, 1),                  // another run follows
            (1, layout.offset_bits), // run2 start = 1 (gap = 1 - 1 = 0 <= G)
            (0, layout.offset_bits), // run2 len - 1 = 0
        ]);
        let err = Bundle::decode(&bytes, p).unwrap_err();
        assert_eq!(err, WireError::NonCanonicalRuns);
    }

    #[test]
    fn decode_rejects_run_with_zero_boundary_delta() {
        let p = odd_params();
        let layout = DeltaWireLayout::for_params(&p);
        let bytes = raw_pack(&[
            (0, 64),
            (1, 1),
            (0, layout.bucket_bits),
            (0, layout.offset_bits),
            (0, layout.offset_bits),                // single-cell run
            (delta_code(0, &p), layout.delta_bits), // delta = 0
            (0, 1),
            (0, 1),
        ]);
        let err = Bundle::decode(&bytes, p).unwrap_err();
        assert_eq!(err, WireError::NonCanonicalRuns);
    }

    #[test]
    fn decode_rejects_interior_zero_stretch_longer_than_max_gap() {
        let p = small_params(SchemeKind::Segmented2ary);
        let layout = DeltaWireLayout::for_params(&p);
        assert_eq!(
            layout.max_gap, 1,
            "fixture sanity: this test targets G == 1"
        );
        let bytes = raw_pack(&[
            (0, 64),
            (1, 1),
            (0, layout.bucket_bits),
            (0, layout.offset_bits),                // start = 0
            (3, layout.offset_bits),                // len - 1 = 3 => len = 4, covers offsets 0..3
            (delta_code(1, &p), layout.delta_bits), // offset 0: nonzero
            (delta_code(0, &p), layout.delta_bits), // offset 1: zero
            (delta_code(0, &p), layout.delta_bits), // offset 2: zero (2 zeros > G = 1)
            (delta_code(1, &p), layout.delta_bits), // offset 3: nonzero
            (0, 1),
            (0, 1),
        ]);
        let err = Bundle::decode(&bytes, p).unwrap_err();
        assert_eq!(err, WireError::NonCanonicalRuns);
    }

    #[test]
    fn decode_rejects_nonzero_padding_bit() {
        let p = small_params(SchemeKind::Segmented2ary);
        let b = build(p, 5, vec![Vec::new(); p.arity()]);
        let mut bytes = b.encode();
        assert_eq!(bytes.len(), 9);
        // Bit 65 is the first padding bit after epoch(64) + terminator(1).
        let bit_index = 65;
        bytes[bit_index / 8] |= 1 << (bit_index % 8);
        let err = Bundle::decode(&bytes, p).unwrap_err();
        assert_eq!(err, WireError::NonZeroPadding);
    }

    #[test]
    fn decode_rejects_trailing_bytes() {
        let p = small_params(SchemeKind::Segmented2ary);
        let b = build(p, 5, vec![Vec::new(); p.arity()]);
        let mut bytes = b.encode();
        bytes.push(0);
        let err = Bundle::decode(&bytes, p).unwrap_err();
        assert_eq!(err, WireError::TrailingBytes);
    }

    #[test]
    fn decode_rejects_every_truncated_prefix() {
        let p = small_params(SchemeKind::Segmented2ary);
        let mut segs = vec![Vec::new(); p.arity()];
        segs[0].push((0u32, vec![(0u16, 1i64)]));
        segs[0].push((3u32, vec![(0u16, -1i64), (2u16, 1i64)]));
        segs[p.arity() - 1].push((1u32, vec![(3u16, 2i64)]));
        let b = build(p, 7, segs);
        let bytes = b.encode();
        assert!(
            bytes.len() > 8,
            "fixture must be non-trivial for a meaningful prefix sweep"
        );
        for len in 0..bytes.len() {
            let prefix = &bytes[..len];
            assert!(
                Bundle::decode(prefix, p).is_err(),
                "prefix of length {len} (full length {}) must be rejected",
                bytes.len()
            );
        }
    }

    // ── 6. Property test: random params + random canonical bundles ──────

    fn arb_scheme() -> impl Strategy<Value = SchemeKind> {
        prop_oneof![
            Just(SchemeKind::Segmented2ary),
            Just(SchemeKind::Segmented3ary),
            Just(SchemeKind::Segmented4ary),
        ]
    }

    fn arb_params() -> impl Strategy<Value = CuckooParams> {
        (
            arb_scheme(),
            1u32..=4,
            prop_oneof![Just(12u32), Just(32), Just(64)],
            prop_oneof![Just(8u32), Just(64), Just(2048)],
            8u32..=12,
            3u32..=8,
        )
            .prop_map(
                |(scheme_kind, bucket_size, fingerprint_bits, value_bits, plaintext_bits, k)| {
                    let arity = arity_of(scheme_kind);
                    CuckooParams {
                        scheme_kind,
                        num_buckets: arity * (1u32 << k),
                        bucket_size,
                        fingerprint_bits,
                        value_bits,
                        plaintext_bits,
                    }
                },
            )
    }

    fn arb_row_cells(omega: u32, p: i64) -> impl Strategy<Value = Vec<(u16, i64)>> {
        let max_cells = (omega as usize).clamp(1, 6);
        let delta = prop_oneof![1i64..p, (-(p - 1))..=-1i64];
        proptest::collection::btree_map(0u32..omega, delta, 1..=max_cells)
            .prop_map(|m| m.into_iter().map(|(o, d)| (o as u16, d)).collect())
    }

    fn arb_segment(rho: u32, omega: u32, p: i64) -> impl Strategy<Value = SegmentRowDeltas> {
        let max_rows = (rho as usize).min(6);
        proptest::collection::btree_map(0u32..rho, arb_row_cells(omega, p), 0..=max_rows)
            .prop_map(|m| m.into_iter().collect())
    }

    fn arb_params_and_bundle() -> impl Strategy<Value = (CuckooParams, Bundle)> {
        arb_params().prop_flat_map(|params| {
            let arity = params.arity();
            let rho = params.segment_size();
            let omega = params.bucket_size * params.cells_per_slot();
            let p = 1i64 << params.plaintext_bits;
            let epoch = any::<u64>();
            let segs = proptest::collection::vec(arb_segment(rho, omega, p), arity..=arity);
            (epoch, segs).prop_map(move |(epoch, segs)| (params, Bundle::new(epoch, segs, params)))
        })
    }

    proptest! {
        #[test]
        fn prop_encode_decode_round_trip((params, bundle) in arb_params_and_bundle()) {
            let stats = bundle.wire_stats();
            let bytes = bundle.encode();
            prop_assert_eq!(bytes.len(), bundle.wire_byte_size());
            prop_assert_eq!(bytes.len(), stats.bytes);

            let decoded = Bundle::decode(&bytes, params).map_err(|e| {
                TestCaseError::fail(format!("decode failed on a bundle `encode` just produced: {e}"))
            })?;

            let expected_nonzero: u64 = bundle
                .per_segment_row_deltas
                .iter()
                .flat_map(|seg| seg.iter())
                .map(|(_, cells)| cells.len() as u64)
                .sum();
            prop_assert_eq!(stats.nonzero_cells, expected_nonzero);
            prop_assert_eq!(decoded, bundle);
        }
    }

    // ── 7. Closed form vs. hand count ────────────────────────────────────

    #[test]
    fn wire_stats_bits_match_hand_computed_closed_form() {
        let p = small_params(SchemeKind::Segmented2ary); // rho=4, omega=4, IB=3, OW=2, DB=5, G=1
        let layout = DeltaWireLayout::for_params(&p);
        let mut segs = vec![Vec::new(); p.arity()];
        // segment 0, row 0: offsets {0, 2}, gap = 1 = G -> merges into one run (len 3).
        segs[0].push((0u32, vec![(0u16, 1i64), (2u16, -1i64)]));
        // segment 1, row 3: single-cell run.
        segs[1].push((3u32, vec![(3u16, 2i64)]));
        let b = build(p, 9, segs);
        let stats = b.wire_stats();

        let rows = 2u64;
        let runs = 2u64;
        let cells = 4u64; // 3 (merged run, incl. 1 zero literal) + 1
        let hand_bits = 64
            + 1
            + rows * (1 + u64::from(layout.bucket_bits))
            + runs * (2 * u64::from(layout.offset_bits) + 1)
            + cells * u64::from(layout.delta_bits);

        assert_eq!(stats.rows, rows);
        assert_eq!(stats.runs, runs);
        assert_eq!(stats.cells, cells);
        assert_eq!(stats.bits, hand_bits);
        assert_eq!(stats.bytes, hand_bits.div_ceil(8) as usize);
    }
}
