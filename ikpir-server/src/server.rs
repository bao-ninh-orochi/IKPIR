//! [`IkpirServer`] — the server-side state machine.
//!
//! # Purpose
//!
//! Wraps a [`segmented_cuckoo::CuckooKVStore`] in per-segment Index-PIR
//! sub-databases and exposes the protocol surface
//! (`setup` / `answer` / `insert` / `update` / `delete` / `full_rebuild`).
//!
//! # Design / architecture
//!
//! - **Ownership.** The server owns the only authoritative copy of the
//!   cell array, mutation log, and hint matrices. The client carries
//!   only preprocessing material derived from the [`ServerSetupBundle`].
//! - **Per-segment partition.** An arity-`k` SCF splits the store into
//!   `k` independent Index-PIR sub-databases. `answer` runs the active
//!   backend's `server_answer` once per segment over a contiguous slice
//!   of `self.store.as_cells()`.
//! - **Mutation log → incremental hint.** Every successful mutation
//!   drains the SCF mutation log, folds it via
//!   [`fold_mutations_into_row_deltas`] into sparse per-segment row
//!   deltas, applies them in-place via
//!   [`IncrementalPirBackend::server_patch_hint`], and bumps the
//!   strict-monotone `epoch`.
//! - **`backend_config` is persistent.** Captured at construction time
//!   and re-used by every `full_rebuild`, so hint dimensions stay stable
//!   across the server's lifetime.
//!
//! # Related files
//!
//! - `lib.rs` — re-exports `IkpirServer` and the segmented type aliases.
//! - `wire.rs` — the four bundle types the server emits and consumes.
//! - `hint_patch.rs` — incremental-hint folding.
//! - `backend/mod.rs` — `IndexPirBackend` / `IncrementalPirBackend`
//!   trait contract.

use segmented_cuckoo::{
    CuckooError, CuckooKVStore, CuckooParams, IndexScheme, SchemeMeta, Segmented2aryScheme,
    Segmented3aryScheme, Segmented4aryScheme,
};

use ikpir_common::{HintDeltaBundle, IkpirError, IndexPirBackend, IncrementalPirBackend,
    PirQueryBundle, PirResponseBundle, ServerSetupBundle};
use crate::hint_patch::fold_mutations_into_row_deltas;

/// Server-side IKPIR engine, generic over the SCF scheme `S` and PIR
/// backend `B`.
///
/// # Purpose
///
/// Holds the authoritative server state: the `CuckooKVStore`, one
/// `B::ServerParams` and `B::Hint` per segment, the captured
/// `B::Config`, and the strict-monotone `epoch`. Drives the IKPIR
/// protocol surface end-to-end.
///
/// # Rationale
///
/// Prefer the type aliases [`Segmented2aryIkpirServer`],
/// [`Segmented3aryIkpirServer`], and [`Segmented4aryIkpirServer`] over
/// instantiating this type directly — the scheme parameter is fixed at
/// the SCF level, so users rarely need to spell it out.
///
/// # Threading
///
/// All methods are synchronous and take `&mut self` for any state-changing
/// operation. Wrap in `Mutex` / `RwLock` if exposing across threads.
pub struct IkpirServer<S: IndexScheme + SchemeMeta, B: IndexPirBackend> {
    store:                 CuckooKVStore<S>,
    params:                CuckooParams,
    backend_config:        B::Config,
    backend_params:        Vec<B::ServerParams>,
    /// Per-segment seed-derived material (e.g. the LWE matrix `A`). May be
    /// `None` after [`IkpirServer::drop_hint_material`]; transparently
    /// re-expanded on the next mutation via
    /// [`IndexPirBackend::expand_hint_material`].
    backend_hint_material: Vec<Option<B::HintMaterial>>,
    hints:                 Vec<B::Hint>,
    epoch:                 u64,
}

impl<S: IndexScheme + SchemeMeta, B: IndexPirBackend> IkpirServer<S, B> {
    /// Build a server from a populated `CuckooKVStore` and a backend
    /// config.
    ///
    /// # Purpose
    ///
    /// One-shot constructor that runs `B::server_setup` once per segment
    /// to produce the initial hints, enables the store's mutation log,
    /// and sets `epoch = 0`.
    ///
    /// # Arguments
    ///
    /// - `store`          — populated SCF KV store; ownership moves into
    ///   the server.
    /// - `backend_config` — backend-specific tunables (e.g.
    ///   `FrodoConfig::default()` for FrodoPIR's 128-bit security
    ///   `lwe_dim = 1566`). Persisted on the server and re-used by every
    ///   future [`Self::full_rebuild`] call.
    ///
    /// # Constraints
    ///
    /// `store.params().plaintext_bits` (and the rest of the SCF geometry
    /// — `scheme_kind`, `num_buckets`, `bucket_size`,
    /// `fingerprint_bits`, `value_bits`) is fixed for the lifetime of
    /// the `IkpirServer` / `IkpirClient` pair sharing this setup.
    /// Mutation deltas and incremental hint patches assume a constant
    /// cell width across the wire. If support for changing
    /// `plaintext_bits` across rebuilds is ever needed, tag both
    /// [`ServerSetupBundle`] and [`HintDeltaBundle`] with a
    /// parameter-identity fingerprint (e.g. a hash of
    /// `(scheme_kind, num_buckets, bucket_size, fingerprint_bits,
    /// value_bits, plaintext_bits)`) and assert match on
    /// `IkpirClient::apply_delta` / `IkpirClient::reset_from`.
    ///
    /// # Rationale
    ///
    /// Persisting `backend_config` (rather than re-asking the caller on
    /// every rebuild) keeps `setup` ↔ `full_rebuild` symmetric: a client
    /// can always rebuild against the same dimensions.
    ///
    /// # Returns
    ///
    /// An `IkpirServer` at `epoch = 0` with the mutation log enabled and
    /// drained.
    ///
    /// # Complexity
    ///
    /// `O(arity × n_rows × lwe_dim × row_width)` arithmetic plus a
    /// per-segment seed sample. This is the hot path during cold start;
    /// for FrodoPIR with `lwe_dim = 1566` and a 2-ary 16k-bucket store
    /// it dominates the first-query latency.
    pub fn new(store: CuckooKVStore<S>, backend_config: B::Config) -> Self {
        let params       = store.params();
        let arity        = params.arity();
        let segment_size = params.segment_size();
        let row_width    = params.bucket_size * params.cells_per_slot();
        let pb           = params.plaintext_bits;
        let seg_cells    = segment_size as usize * row_width as usize;

        let mut backend_params        = Vec::with_capacity(arity);
        let mut backend_hint_material = Vec::with_capacity(arity);
        let mut hints                 = Vec::with_capacity(arity);
        {
            let cells = store.as_cells();
            for j in 0..arity {
                let start = j * seg_cells;
                let (sp, mat, h) = B::server_setup(
                    &backend_config,
                    &cells[start..start + seg_cells],
                    segment_size,
                    row_width,
                    pb,
                );
                backend_params.push(sp);
                backend_hint_material.push(Some(mat));
                hints.push(h);
            }
        }
        let mut s = Self {
            store, params, backend_config, backend_params, backend_hint_material, hints, epoch: 0,
        };
        s.store.enable_mutation_log();
        let _ = s.store.drain_mutations();
        s
    }

    /// Backend-config snapshot held by this server.
    ///
    /// # Rationale
    ///
    /// Useful for benchmarks and tests that need to construct an oracle
    /// server (or a second client) with the same backend dimensions
    /// without re-deriving them from configuration.
    pub fn backend_config(&self) -> &B::Config { &self.backend_config }

    /// Snapshot the full preprocessing state for a fresh client.
    ///
    /// # Purpose
    ///
    /// First-time client bootstrap; also used to recover a client that
    /// has fallen out of epoch sync (caller drives
    /// `IkpirClient::reset_from`).
    ///
    /// # Returns
    ///
    /// A [`ServerSetupBundle`] holding clones of `params`,
    /// `backend_params`, `hints`, and the current `epoch`.
    ///
    /// # Complexity
    ///
    /// `O(arity)` shallow clones of `Vec<B::ServerParams>` /
    /// `Vec<B::Hint>` plus the deep clone the backend types perform.
    /// Cheap relative to [`Self::new`] / [`Self::full_rebuild`].
    pub fn setup(&self) -> ServerSetupBundle<B> {
        ServerSetupBundle {
            params:         self.params,
            backend_params: self.backend_params.clone(),
            hints:          self.hints.clone(),
            epoch:          self.epoch,
        }
    }

    /// Answer a per-segment PIR query bundle.
    ///
    /// # Purpose
    ///
    /// Server hot path: runs `B::server_answer` once per segment over
    /// `self.store.as_cells()` and returns the per-segment responses.
    ///
    /// # Arguments
    ///
    /// - `q` — query bundle from the client.
    ///
    /// # Constraints
    ///
    /// - `q.epoch` must equal `self.epoch` (the client is on the
    ///   server's current view of the database).
    /// - `q.queries.len()` must equal `params.arity()`.
    ///
    /// # Returns
    ///
    /// - `Ok(PirResponseBundle)` carrying the per-segment responses and
    ///   the current epoch.
    /// - `Err(IkpirError::StaleEpoch)` if `q.epoch != self.epoch` — the
    ///   client has missed at least one mutation.
    /// - `Err(IkpirError::MalformedQuery)` if `q.queries.len()` doesn't
    ///   match arity.
    ///
    /// # Complexity
    ///
    /// `O(arity × n_rows × row_width × lwe_dim)` — dominates server CPU
    /// at steady state. For FrodoPIR this is the matvec a `--bench
    /// answer_throughput` run isolates.
    pub fn answer(&self, q: &PirQueryBundle<B>) -> Result<PirResponseBundle<B>, IkpirError>
    where
        B::Query:    Clone,
        B::Response: Clone,
    {
        if q.epoch != self.epoch {
            return Err(IkpirError::StaleEpoch { expected: self.epoch, got: q.epoch });
        }
        let arity = self.params.arity();
        if q.queries.len() != arity {
            return Err(IkpirError::MalformedQuery);
        }
        let segment_size = self.params.segment_size();
        let row_width    = self.params.bucket_size * self.params.cells_per_slot();
        let cells        = self.store.as_cells();
        let seg_cells    = segment_size as usize * row_width as usize;

        let mut responses = Vec::with_capacity(arity);
        for j in 0..arity {
            let start = j * seg_cells;
            let r = B::server_answer(
                &self.backend_params[j],
                &cells[start..start + seg_cells],
                segment_size,
                row_width,
                &q.queries[j],
            );
            responses.push(r);
        }
        Ok(PirResponseBundle { epoch: self.epoch, responses })
    }

    /// Recompute every per-segment hint from scratch and increment
    /// `epoch`.
    ///
    /// # Purpose
    ///
    /// Escape hatch when the incremental hint patches have become more
    /// expensive than a fresh setup, or when the server wants to flip
    /// every connected client into a forced resync (each client's next
    /// `apply_delta` will hit `FutureDelta`).
    ///
    /// # Rationale
    ///
    /// Re-uses the [`backend_config`](Self::backend_config) the server
    /// was constructed with, so hint dimensions stay stable across the
    /// rebuild. The mutation log is drained as part of the rebuild
    /// (its deltas are now subsumed in the new hints) and the epoch is
    /// bumped, so all in-flight client queries become stale.
    ///
    /// # Returns
    ///
    /// A [`ServerSetupBundle`] mirroring the new internal state — same
    /// shape as [`Self::setup`].
    ///
    /// # Complexity
    ///
    /// Same as [`Self::new`]:
    /// `O(arity × n_rows × lwe_dim × row_width)`.
    pub fn full_rebuild(&mut self) -> ServerSetupBundle<B> {
        let arity        = self.params.arity();
        let segment_size = self.params.segment_size();
        let row_width    = self.params.bucket_size * self.params.cells_per_slot();
        let pb           = self.params.plaintext_bits;
        let seg_cells    = segment_size as usize * row_width as usize;
        // Snapshot to avoid a borrow conflict with self.backend_params / self.hints writes.
        let cells: Vec<u32> = self.store.as_cells().to_vec();

        let mut backend_params        = Vec::with_capacity(arity);
        let mut backend_hint_material = Vec::with_capacity(arity);
        let mut hints                 = Vec::with_capacity(arity);
        for j in 0..arity {
            let start = j * seg_cells;
            let (sp, mat, h) = B::server_setup(
                &self.backend_config,
                &cells[start..start + seg_cells],
                segment_size,
                row_width,
                pb,
            );
            backend_params.push(sp);
            backend_hint_material.push(Some(mat));
            hints.push(h);
        }
        self.backend_params        = backend_params;
        self.backend_hint_material = backend_hint_material;
        self.hints                 = hints;
        let _ = self.store.drain_mutations();
        self.epoch                += 1;
        self.setup()
    }

    /// Free the per-segment seed-derived [`HintMaterial`](IndexPirBackend::HintMaterial).
    ///
    /// # Purpose
    ///
    /// Lets read-only deployments (and benches that only sample queries
    /// after setup) reclaim the LWE public matrix `A` from RAM. The next
    /// mutation transparently re-expands the affected segments from the
    /// seed inside [`B::ServerParams`](IndexPirBackend::ServerParams);
    /// callers observe nothing different other than a one-time
    /// first-mutation re-expansion cost.
    ///
    /// # Constraints
    ///
    /// Safe to call at any time. Calling it twice in a row is a no-op on
    /// the second invocation. Production read-write services that mutate
    /// frequently should generally **not** call this — they will pay the
    /// re-expansion cost on every drop / mutate cycle.
    ///
    /// # Complexity
    ///
    /// `O(arity)` `Option::take` operations.
    pub fn drop_hint_material(&mut self) {
        for slot in &mut self.backend_hint_material {
            *slot = None;
        }
    }

    /// Current server epoch. Strictly monotone across mutation and rebuild.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// SCF geometry parameters of the wrapped store. Same shape as
    /// `ServerSetupBundle::params`.
    pub fn params(&self) -> CuckooParams {
        self.params
    }
}

impl<S, B> IkpirServer<S, B>
where
    S: IndexScheme + SchemeMeta,
    B: IncrementalPirBackend,
{
    /// Insert a `(key, value)` pair, patching every affected segment
    /// hint in place.
    ///
    /// # Arguments
    ///
    /// - `key`   — keyword bytes; hashed by the SCF to derive
    ///   fingerprint + candidate buckets.
    /// - `value` — raw bytes; length must equal `params.value_size_in_bytes()`.
    ///
    /// # Constraints
    ///
    /// Mutation log must be enabled (it is by default after
    /// [`Self::new`]). Disabling the log breaks incremental correctness
    /// — see [`CuckooKVStore::disable_mutation_log`].
    ///
    /// # Returns
    ///
    /// - `Ok(HintDeltaBundle)` on success. `self.epoch` is incremented
    ///   by 1; the delta carries the new epoch and the sparse row
    ///   patches the client must apply.
    /// - `Err(IkpirError::TableFull)` if the cuckoo kick budget is
    ///   exhausted; the SCF rolls back and `self.epoch` is **not**
    ///   advanced. The mutation log is drained to prevent leaked
    ///   deltas (see
    ///   `tests/incremental_correctness.rs::mutation_log_drained_on_failure`).
    /// - `Err(IkpirError::InvalidInput)` if `value.len()` is wrong.
    ///
    /// # On `IkpirError::TableFull`
    ///
    /// Recovery requires a full rebuild at a larger `num_buckets`, but
    /// [`segmented_cuckoo::CuckooKVStore`] discards keys after hashing
    /// — only fingerprints and packed values are retained — so IKPIR
    /// cannot rebuild itself from internal state. The application
    /// layer is expected to hold an authoritative `(key, value)` map
    /// alongside the server and, on `TableFull`, reconstruct a new
    /// [`IkpirServer`] from that map with a larger `num_buckets`. The
    /// resulting fresh setup bundle must be delivered to clients via
    /// [`IkpirClient::reset_from`](https://docs.rs/ikpir-client/latest/ikpir_client/struct.IkpirClient.html#method.reset_from);
    /// `apply_delta` cannot bridge a `num_buckets` change.
    ///
    /// # Complexity
    ///
    /// SCF insert (`O(value_size_in_cells)` direct, up to
    /// `O(max_kicks · value_size_in_cells)` with kicks) + sparse hint
    /// patch (`O(touched_rows · cells_per_slot · lwe_dim)`).
    pub fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<HintDeltaBundle<B>, IkpirError> {
        self.store.insert(key, value).map_err(map_cuckoo_err)?;
        Ok(self.commit_mutations())
    }

    /// Update the value for an existing key, patching every affected
    /// segment hint.
    ///
    /// # Arguments
    ///
    /// - `key`   — key to update.
    /// - `value` — replacement bytes; length must equal
    ///   `params.value_size_in_bytes()`.
    ///
    /// # Returns
    ///
    /// - `Ok(HintDeltaBundle)` on success; `self.epoch` is incremented.
    /// - `Err(IkpirError::NotFound)` if the key is absent; server
    ///   state and epoch are unchanged.
    /// - `Err(IkpirError::InvalidInput)` if `value.len()` is wrong.
    ///
    /// # Complexity
    ///
    /// SCF update (`O(arity · bucket_size + value_size_in_cells)`) +
    /// sparse hint patch on one row per arity.
    pub fn update(&mut self, key: &[u8], value: &[u8]) -> Result<HintDeltaBundle<B>, IkpirError> {
        self.store.update(key, value).map_err(map_cuckoo_err)?;
        Ok(self.commit_mutations())
    }

    /// Delete an existing key, patching every affected segment hint.
    ///
    /// # Arguments
    ///
    /// - `key` — key to remove.
    ///
    /// # Returns
    ///
    /// - `Ok(HintDeltaBundle)` on success; `self.epoch` is incremented.
    /// - `Err(IkpirError::NotFound)` if the key is absent; server
    ///   state and epoch are unchanged.
    ///
    /// # Complexity
    ///
    /// SCF delete (`O(arity · bucket_size · ⌈fingerprint_bits / plaintext_bits⌉)`)
    /// + sparse hint patch on one row per arity.
    pub fn delete(&mut self, key: &[u8]) -> Result<HintDeltaBundle<B>, IkpirError> {
        self.store.delete(key).map_err(map_cuckoo_err)?;
        Ok(self.commit_mutations())
    }

    fn commit_mutations(&mut self) -> HintDeltaBundle<B> {
        let muts       = self.store.drain_mutations();
        let row_deltas = fold_mutations_into_row_deltas(&muts, &self.params);

        // Phase 1: re-expand any dropped hint-material we are about to
        // touch. Holds `&mut self.backend_hint_material` only, no overlap
        // with `self.hints` writes.
        for (j, deltas) in row_deltas.iter().enumerate() {
            if !deltas.is_empty() && self.backend_hint_material[j].is_none() {
                self.backend_hint_material[j] =
                    Some(B::expand_hint_material(&self.backend_params[j]));
            }
        }
        // Phase 2: split-borrow params / material / hints by index — three
        // disjoint fields, the borrow checker is happy.
        for (j, deltas) in row_deltas.iter().enumerate() {
            if !deltas.is_empty() {
                let material = self.backend_hint_material[j]
                    .as_ref()
                    .expect("expanded in Phase 1");
                B::server_patch_hint(
                    &self.backend_params[j],
                    material,
                    &mut self.hints[j],
                    deltas,
                );
            }
        }
        self.epoch += 1;
        HintDeltaBundle::new(self.epoch, row_deltas)
    }
}

fn map_cuckoo_err(e: CuckooError) -> IkpirError {
    match e {
        CuckooError::TableFull        => IkpirError::TableFull,
        CuckooError::NotFound         => IkpirError::NotFound,
        CuckooError::InvalidParams(_) => IkpirError::InvalidInput,
    }
}

/// Server typed for a 2-ary Segmented Cuckoo KV store.
pub type Segmented2aryIkpirServer<B> = IkpirServer<Segmented2aryScheme, B>;
/// Server typed for a 3-ary Segmented Cuckoo KV store.
pub type Segmented3aryIkpirServer<B> = IkpirServer<Segmented3aryScheme, B>;
/// Server typed for a 4-ary Segmented Cuckoo KV store.
pub type Segmented4aryIkpirServer<B> = IkpirServer<Segmented4aryScheme, B>;
