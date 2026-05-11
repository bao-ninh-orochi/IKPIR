//! [`IkpirServer`] — the server-side state machine.
//!
//! Wraps a [`segmented_cuckoo::CuckooKVStore`] in per-segment Index-PIR
//! sub-databases. The server owns the only authoritative copy of the cell
//! array, mutation log, and hint matrices; the client carries only
//! preprocessing material derived from the [`ServerSetupBundle`].

use segmented_cuckoo::{
    CuckooError, CuckooKVStore, CuckooParams, IndexScheme, SchemeMeta, Segmented2aryScheme,
    Segmented3aryScheme, Segmented4aryScheme,
};

use crate::backend::{IndexPirBackend, IncrementalPirBackend};
use crate::hint_patch::fold_mutations_into_row_deltas;
use crate::wire::{HintDeltaBundle, PirQueryBundle, PirResponseBundle, ServerSetupBundle};
use crate::IkpirError;

/// Server-side IKPIR engine, generic over the SCF scheme `S` and PIR backend `B`.
///
/// Prefer the type aliases [`Segmented2aryIkpirServer`],
/// [`Segmented3aryIkpirServer`], and [`Segmented4aryIkpirServer`] over
/// instantiating this type directly.
///
/// # Threading
///
/// All methods are synchronous and take `&mut self` for any state-changing
/// operation. Wrap in `Mutex` / `RwLock` if exposing across threads.
pub struct IkpirServer<S: IndexScheme + SchemeMeta, B: IndexPirBackend> {
    store:          CuckooKVStore<S>,
    params:         CuckooParams,
    backend_config: B::Config,
    backend_params: Vec<B::ServerParams>,
    hints:          Vec<B::Hint>,
    epoch:          u64,
}

impl<S: IndexScheme + SchemeMeta, B: IndexPirBackend> IkpirServer<S, B> {
    /// Build a server from a populated `CuckooKVStore` and a backend config.
    ///
    /// Runs `B::server_setup` once per segment to produce the initial hints,
    /// enables the store's mutation log, and sets `epoch = 0`. The supplied
    /// `backend_config` is persisted on the server and re-used by every
    /// future [`Self::full_rebuild`] call so all hints share the same
    /// dimensions across the server's lifetime.
    ///
    /// `O(arity × n_rows × lwe_dim × row_width)` arithmetic plus a per-segment
    /// random `seed` sample.
    ///
    /// Pass `B::Config::default()` if you don't need to override any backend
    /// knobs (e.g. `FrodoConfig::default()` selects FrodoPIR's 128-bit
    /// security `lwe_dim = 1774`).
    ///
    /// # Invariants
    ///
    /// `store.params().plaintext_bits` (and the rest of the SCF geometry —
    /// `scheme_kind`, `num_buckets`, `bucket_size`, `fingerprint_bits`,
    /// `value_bits`) is fixed for the lifetime of the `IkpirServer` /
    /// `IkpirClient` pair sharing this setup. Mutation deltas and incremental
    /// hint patches assume a constant cell width across the wire. If support
    /// for changing `plaintext_bits` across rebuilds is ever needed, tag both
    /// [`ServerSetupBundle`] and [`HintDeltaBundle`] with a parameter-identity
    /// fingerprint (e.g. a hash of
    /// `(scheme_kind, num_buckets, bucket_size, fingerprint_bits, value_bits,
    /// plaintext_bits)`) and assert match on `IkpirClient::apply_delta` /
    /// `IkpirClient::reset_from`.
    pub fn new(store: CuckooKVStore<S>, backend_config: B::Config) -> Self {
        let params       = store.params();
        let arity        = params.arity();
        let segment_size = params.segment_size();
        let row_width    = params.bucket_size * params.cells_per_slot();
        let pb           = params.plaintext_bits;
        let seg_cells    = segment_size as usize * row_width as usize;

        let mut backend_params = Vec::with_capacity(arity);
        let mut hints          = Vec::with_capacity(arity);
        {
            let cells = store.as_cells();
            for j in 0..arity {
                let start = j * seg_cells;
                let (sp, h) = B::server_setup(
                    &backend_config,
                    &cells[start..start + seg_cells],
                    segment_size,
                    row_width,
                    pb,
                );
                backend_params.push(sp);
                hints.push(h);
            }
        }
        let mut s = Self { store, params, backend_config, backend_params, hints, epoch: 0 };
        s.store.enable_mutation_log();
        let _ = s.store.drain_mutations();
        s
    }

    /// Backend-config snapshot held by this server. Useful for benchmarks
    /// and tests that need to construct an oracle server with the same
    /// dimensions.
    pub fn backend_config(&self) -> &B::Config { &self.backend_config }

    /// Snapshot the full preprocessing state for a fresh client.
    ///
    /// Cheap: just clones the `ServerParams` and `Hint` vectors. The client
    /// drives [`IkpirClient::from_setup`](https://docs.rs/ikpir-client/latest/ikpir_client/struct.IkpirClient.html#method.from_setup)
    /// off the result.
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
    /// Calls `B::server_answer` once per segment over `self.store.as_cells()`.
    ///
    /// # Errors
    ///
    /// - [`IkpirError::StaleEpoch`] if `q.epoch != self.epoch`.
    /// - [`IkpirError::MalformedQuery`] if `q.queries.len()` doesn't match arity.
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

    /// Recompute every per-segment hint from scratch and increment `epoch`.
    ///
    /// Re-uses the [`backend_config`](Self::backend_config) the server was
    /// constructed with, so dimensions stay stable across rebuilds. Cost
    /// equals the original [`Self::new`] setup cost. Use this after large
    /// mutation bursts (when the cumulative incremental patch cost would
    /// exceed a fresh setup), or to recover a client that has fallen out of
    /// epoch sync.
    pub fn full_rebuild(&mut self) -> ServerSetupBundle<B> {
        let arity        = self.params.arity();
        let segment_size = self.params.segment_size();
        let row_width    = self.params.bucket_size * self.params.cells_per_slot();
        let pb           = self.params.plaintext_bits;
        let seg_cells    = segment_size as usize * row_width as usize;
        // Snapshot to avoid a borrow conflict with self.backend_params / self.hints writes.
        let cells: Vec<u32> = self.store.as_cells().to_vec();

        let mut backend_params = Vec::with_capacity(arity);
        let mut hints          = Vec::with_capacity(arity);
        for j in 0..arity {
            let start = j * seg_cells;
            let (sp, h) = B::server_setup(
                &self.backend_config,
                &cells[start..start + seg_cells],
                segment_size,
                row_width,
                pb,
            );
            backend_params.push(sp);
            hints.push(h);
        }
        self.backend_params = backend_params;
        self.hints          = hints;
        let _ = self.store.drain_mutations();
        self.epoch         += 1;
        self.setup()
    }

    /// Current server epoch. Strictly monotone across mutation and rebuild.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// SCF geometry parameters of the wrapped store.
    pub fn params(&self) -> CuckooParams {
        self.params
    }
}

impl<S, B> IkpirServer<S, B>
where
    S: IndexScheme + SchemeMeta,
    B: IncrementalPirBackend,
{
    /// Insert a `(key, value)` pair, then patch every affected segment hint
    /// in place. Returns the resulting [`HintDeltaBundle`] for the client to
    /// apply.
    ///
    /// On [`IkpirError::TableFull`] the store and hints are restored to the
    /// pre-call state and `epoch` is *not* advanced.
    ///
    /// # On `IkpirError::TableFull`
    ///
    /// The SCF rolls back internally and the server's state is unchanged.
    /// Recovery requires a full rebuild at a larger `num_buckets`, but
    /// [`segmented_cuckoo::CuckooKVStore`] discards keys after hashing — only
    /// fingerprints and packed values are retained — so IKPIR cannot rebuild
    /// itself from internal state. The application layer is expected to hold
    /// an authoritative `(key, value)` map alongside the server and, on
    /// `TableFull`, reconstruct a new [`IkpirServer`] from that map with a
    /// larger `num_buckets`. The resulting fresh setup bundle must be
    /// delivered to clients via
    /// [`IkpirClient::reset_from`](https://docs.rs/ikpir-client/latest/ikpir_client/struct.IkpirClient.html#method.reset_from);
    /// `apply_delta` cannot bridge a `num_buckets` change.
    pub fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<HintDeltaBundle<B>, IkpirError> {
        self.store.insert(key, value).map_err(map_cuckoo_err)?;
        Ok(self.commit_mutations())
    }

    /// Update the value for an existing key, then patch every affected segment hint.
    /// Returns [`IkpirError::NotFound`] if the key is absent.
    pub fn update(&mut self, key: &[u8], value: &[u8]) -> Result<HintDeltaBundle<B>, IkpirError> {
        self.store.update(key, value).map_err(map_cuckoo_err)?;
        Ok(self.commit_mutations())
    }

    /// Delete an existing key, then patch every affected segment hint.
    /// Returns [`IkpirError::NotFound`] if the key is absent.
    pub fn delete(&mut self, key: &[u8]) -> Result<HintDeltaBundle<B>, IkpirError> {
        self.store.delete(key).map_err(map_cuckoo_err)?;
        Ok(self.commit_mutations())
    }

    fn commit_mutations(&mut self) -> HintDeltaBundle<B> {
        let muts       = self.store.drain_mutations();
        let row_deltas = fold_mutations_into_row_deltas(&muts, &self.params);

        for (j, deltas) in row_deltas.iter().enumerate() {
            if !deltas.is_empty() {
                B::server_patch_hint(&self.backend_params[j], &mut self.hints[j], deltas);
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
