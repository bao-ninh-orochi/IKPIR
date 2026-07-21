//! Threading policy for the whole crate: the worker count, the chunk
//! arithmetic, and the two fan-out mechanisms.
//!
//! # Purpose
//!
//! Two things need to split work across cores, and they want different
//! machinery:
//!
//! - **Setup**, called a handful of times per process, splits the hint
//!   precompute `H = Aᵀ·D` into **bands of `H` rows** and the ChaCha20
//!   expansion of `A` into **runs of output words**. Both partitions
//!   are static, uniform, and huge; thread-spawn cost is noise.
//! - **The online kernels** (`backend/matvec.rs`, `backend/gemm.rs`,
//!   `backend/prg.rs`, both backends' hint patches), which run per
//!   query and per mutation and cannot afford a thread spawn each
//!   time. These live behind the default-on `parallel` feature and use
//!   rayon's persistent pool.
//!
//! # Design / architecture
//!
//! - **Two mechanisms, one policy.** [`setup_threads`] is the single
//!   worker count and `balanced_chunk_len` the single chunk rule;
//!   both mechanisms read them, so `IKPIR_SETUP_THREADS=1` degrades
//!   *every* optimized path in the crate to its reference schedule —
//!   the first thing to try when bisecting a result mismatch.
//!   `kernels_parallel` is the rayon side's gate on that knob, cached
//!   because it sits inside hot loops.
//! - **`std::thread::scope` when rayon is absent.** With the `parallel`
//!   feature off, `par_chunks_mut` is how `ParallelSetupBackend`
//!   still reaches every core: the tasks borrow `&A` and `&D` and write
//!   into disjoint `&mut` sub-slices, which scoped threads express
//!   without an `Arc`/`Mutex` in sight. With the feature on, the
//!   setup twins delegate to the reference entry points — those are
//!   *already* the rayon kernels — so a second fan-out would only
//!   oversubscribe, and this primitive is not compiled at all.
//! - **Bit-exactness by construction.** Every partition here splits the
//!   *output*: a task owns a contiguous slice no other task touches, so
//!   each output cell accumulates the same terms in the same order as
//!   the single-threaded reference. No floating point, no reduction
//!   tree, nothing order-sensitive. (`matvec.rs`'s row partition does
//!   reduce, and leans on `u32` wrapping addition being associative and
//!   commutative — see its own docs.)
//! - **Public shapes only.** Thread count and chunk length are derived
//!   from `(n_rows, row_width, lwe_dim)` and the machine's core count —
//!   never from database contents or from any secret. The schedule
//!   therefore leaks nothing that the parameters do not already
//!   publish.
//!
//! # Related files
//!
//! - `backend/mod.rs` — [`ParallelSetupBackend`](crate::ParallelSetupBackend),
//!   the trait these primitives implement.
//! - `frodo/backend.rs`, `simple/backend.rs` — `compute_hint_parallel`
//!   and the banded hint patches.
//! - `frodo/sampler.rs`, `simple/sampler.rs` — `sample_a_parallel`.
//! - `backend/matvec.rs`, `backend/gemm.rs`, `backend/prg.rs` — the
//!   rayon kernels.

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

/// Number of rayon tasks the online kernels split their work into.
///
/// # Rationale
///
/// Four tasks per worker rather than one: the kernels run inside a pool
/// the caller may already be sharing (a server answering several
/// queries, a bench driving both backends), and oversubscribing the
/// partition is what gives rayon's scheduler something to steal. The
/// partitions themselves stay balanced, so this costs only a handful of
/// extra task headers.
#[cfg(feature = "parallel")]
pub(crate) fn kernel_tasks() -> usize {
    rayon::current_num_threads() * 4
}

/// Whether the rayon kernels may fan out at all.
///
/// `IKPIR_SETUP_THREADS=1` forces the reference schedule everywhere in
/// the crate, the online kernels included — so a mismatch can be
/// bisected against a single-threaded run without rebuilding. A
/// single-worker rayon pool is treated the same way, since fanning out
/// would then be pure overhead.
///
/// # Rationale
///
/// Cached: these gates sit inside hot loops, and both inputs are fixed
/// for the life of the process (rayon's global pool cannot shrink, and
/// re-reading the environment per matvec would be a syscall per query).
#[cfg(feature = "parallel")]
pub(crate) fn kernels_parallel() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| setup_threads() > 1 && rayon::current_num_threads() > 1)
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
/// Only compiled with the `parallel` feature **off**: with it on, the
/// setup twins delegate to the rayon kernels instead of banding around
/// them (see the module docs).
///
/// # Panics
///
/// Propagates a panic from any task after all tasks have been joined
/// (`std::thread::scope` semantics).
#[cfg(not(feature = "parallel"))]
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

    #[cfg(not(feature = "parallel"))]
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
