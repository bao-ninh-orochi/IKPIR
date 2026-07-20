//! Scoped-thread fan-out behind the *optimized* setup path.
//!
//! # Purpose
//!
//! [`ParallelSetupBackend`](crate::ParallelSetupBackend) needs to split
//! two embarrassingly parallel setup kernels across cores:
//!
//! - the hint precompute `H = Aᵀ·D`, partitioned by **bands of `H`
//!   rows** (each band is a disjoint output region, so no reduction is
//!   ever needed), and
//! - the ChaCha20 expansion of the public matrix `A`, partitioned by
//!   **runs of output words** (each run re-seeks the same keystream).
//!
//! Both partitions are static and uniform — every task does the same
//! amount of work — so this module deliberately provides *only* a
//! fixed-partition scoped fan-out, not a work-stealing pool.
//!
//! # Design / architecture
//!
//! - **`std::thread::scope`, no dependency.** The tasks borrow `&A` and
//!   `&D` and write into disjoint `&mut` sub-slices of the output.
//!   Scoped threads express exactly that, so the optimized path costs
//!   the crate zero new dependencies and no `Arc`/`Mutex` machinery.
//!   A work-stealing runtime (rayon) would buy nothing here: the
//!   partition is already balanced.
//! - **Bit-exactness by construction.** A task owns a contiguous slice
//!   of the *output* and no two tasks touch the same cell, so every
//!   output cell accumulates the same terms in the same order as the
//!   single-threaded reference. No floating point, no reduction tree,
//!   nothing order-sensitive. This is what lets
//!   `ParallelSetupBackend` promise bit-identical results.
//! - **Public shapes only.** Thread count and chunk length are derived
//!   from `(n_rows, row_width, lwe_dim)` and the machine's core count —
//!   never from database contents or from any secret. The schedule
//!   therefore leaks nothing that the parameters do not already
//!   publish.
//! - **Reproducible thread count.** [`setup_threads`] reads
//!   `IKPIR_SETUP_THREADS` first and falls back to
//!   `std::thread::available_parallelism`. Setting the variable to `1`
//!   makes every optimized entry point degrade to the reference
//!   schedule, which is handy when bisecting a result mismatch.
//!
//! # Related files
//!
//! - `backend/mod.rs` — [`ParallelSetupBackend`](crate::ParallelSetupBackend),
//!   the trait these primitives implement.
//! - `frodo/backend.rs`, `simple/backend.rs` — `compute_hint_parallel`.
//! - `frodo/sampler.rs`, `simple/sampler.rs` — `sample_a_parallel`.

/// Environment variable overriding the optimized path's worker count.
///
/// Parsed as a `usize ≥ 1`, then clamped to [`MAX_SETUP_THREADS`];
/// anything else is ignored. `1` forces the sequential schedule
/// everywhere in the optimized path.
pub const SETUP_THREADS_ENV: &str = "IKPIR_SETUP_THREADS";

/// Upper bound on the optimized path's worker count.
///
/// # Rationale
///
/// The fan-out spawns **one thread per chunk**, and chunk count follows
/// the requested worker count, not the machine. Without a ceiling, a
/// mistyped `IKPIR_SETUP_THREADS=1000000` would ask
/// `par_chunks_mut` (below) for a million chunks of the keystream — the
/// keystream partition's `unit` is only 64 words, so nothing else caps
/// it — and the loop would issue ~10⁵ `spawn` calls before the first
/// join, failing on thread exhaustion rather than on anything the
/// caller could diagnose. (The hint partition is accidentally immune:
/// its `unit` is a whole hint row, so it can never exceed `lwe_dim`
/// chunks.)
///
/// The value is far above any real core count, so it constrains typos
/// and nothing else; deliberate oversubscription up to this bound still
/// works.
pub const MAX_SETUP_THREADS: usize = 1024;

/// Minimum multiply-accumulate count worth fanning the hint precompute
/// out over. Below it, `compute_hint_parallel` runs the reference
/// schedule instead — thread spawn would dominate the arithmetic at
/// unit-test shapes.
///
/// Single source of truth for both backends. Exposed (hidden) so tests
/// outside this crate can assert that their fixture is large enough to
/// actually exercise the parallel path; a test that quietly falls back
/// pins nothing.
#[doc(hidden)]
pub const PAR_MIN_HINT_MACS: u64 = 1 << 20;

/// Minimum keystream length, in `u32` words, worth fanning the public
/// matrix expansion out over. Below it, `sample_a_parallel` runs the
/// reference schedule. Single source of truth for both backends; see
/// [`PAR_MIN_HINT_MACS`] for why it is reachable from outside.
#[doc(hidden)]
pub const PAR_MIN_WORDS: usize = 1 << 18;

/// ChaCha20's refill-buffer size in `u32` words — the indivisible unit
/// of the keystream partition, so that every worker's `set_word_pos`
/// seek lands on a buffer boundary.
pub(crate) const CHACHA_BUFFER_WORDS: usize = 64;

/// Worker count the optimized setup path fans out to.
///
/// `IKPIR_SETUP_THREADS` wins if it parses to a positive integer;
/// otherwise `std::thread::available_parallelism()`, itself falling
/// back to `1` on platforms that cannot report it.
///
/// # Rationale
///
/// Reading the machine's parallelism rather than hardcoding a count
/// keeps the optimized path honest under cgroup / taskset limits (CI
/// containers, `nproc`-restricted benchmark runs); the env override
/// exists so a run can be pinned for reproducibility.
pub fn setup_threads() -> usize {
    if let Ok(raw) = std::env::var(SETUP_THREADS_ENV) {
        if let Ok(n) = raw.trim().parse::<usize>() {
            if n >= 1 {
                // Clamped, not rejected: a too-large value is a typo, and
                // silently doing sane work beats spawning a thread per
                // keystream buffer. See `MAX_SETUP_THREADS`.
                return n.min(MAX_SETUP_THREADS);
            }
        }
    }
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(MAX_SETUP_THREADS)
}

/// Chunk length that splits `total` elements into **at most** `parts`
/// contiguous chunks, each a whole multiple of `unit` elements.
///
/// # Arguments
///
/// - `total` — number of elements to partition.
/// - `unit`  — indivisible group size (an `H` row, a ChaCha block); the
///   returned length is always a multiple of it, so chunk boundaries
///   never fall inside a group. Treated as `1` if zero.
/// - `parts` — desired number of chunks (the worker count). Treated as
///   `1` if zero.
///
/// # Returns
///
/// The chunk length in elements, at least `unit`. The actual chunk
/// count is `total.div_ceil(len) ≤ parts`.
pub(crate) const fn balanced_chunk_len(total: usize, unit: usize, parts: usize) -> usize {
    let unit = if unit == 0 { 1 } else { unit };
    let parts = if parts == 0 { 1 } else { parts };
    total.div_ceil(parts).div_ceil(unit) * unit
}

/// Apply `f` to each `chunk_len`-sized contiguous chunk of `data`, one
/// scoped thread per chunk.
///
/// `f(offset, chunk)` receives the index of the chunk's first element
/// within `data` alongside the chunk itself, so a task can recover its
/// absolute position (which `H` row band it owns, which keystream word
/// it starts at).
///
/// # Constraints
///
/// - Chunks are disjoint, so `f` must not need to observe another
///   chunk's writes. Every caller in this crate satisfies that by
///   construction (see the module docs).
/// - A single chunk (or `chunk_len == 0`) runs inline on the calling
///   thread — no thread is spawned for work that cannot be split.
///
/// # Panics
///
/// Propagates a panic from any task after all tasks have been joined
/// (`std::thread::scope` semantics).
pub(crate) fn par_chunks_mut<T, F>(data: &mut [T], chunk_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    if chunk_len == 0 || data.len() <= chunk_len {
        f(0, data);
        return;
    }
    std::thread::scope(|scope| {
        for (i, chunk) in data.chunks_mut(chunk_len).enumerate() {
            let f = &f;
            scope.spawn(move || f(i * chunk_len, chunk));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_chunk_len_respects_unit_and_parts() {
        // 1000 elements, rows of 7, 4 workers → 250 → rounded up to 252 (36·7).
        assert_eq!(balanced_chunk_len(1000, 7, 4), 252);
        assert_eq!(1000usize.div_ceil(252), 4);
        // Never below one unit, even with more workers than units.
        assert_eq!(balanced_chunk_len(10, 7, 64), 7);
        // Degenerate inputs are clamped, not panics.
        assert_eq!(balanced_chunk_len(10, 0, 0), 10);
        assert_eq!(balanced_chunk_len(0, 4, 8), 0);
    }

    #[test]
    fn par_chunks_mut_visits_every_element_with_absolute_offsets() {
        for chunk_len in [0usize, 1, 3, 7, 64, 1000] {
            let mut data = vec![0usize; 100];
            par_chunks_mut(&mut data, chunk_len, |offset, chunk| {
                for (i, cell) in chunk.iter_mut().enumerate() {
                    *cell = offset + i;
                }
            });
            assert_eq!(data, (0..100).collect::<Vec<_>>(), "chunk_len={chunk_len}");
        }
    }

    #[test]
    fn setup_threads_is_at_least_one() {
        assert!(setup_threads() >= 1);
    }
}
