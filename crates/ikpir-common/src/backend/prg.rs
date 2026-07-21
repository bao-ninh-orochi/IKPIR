//! Chunk-parallel ChaCha20 word-stream expansion — the shared engine
//! behind both backends' `sample_a` / `sample_a_parallel`.
//!
//! # Purpose
//!
//! Expanding the LWE public matrix `A` from its 16-byte seed is a pure
//! ChaCha20 stream read (`n_rows · lwe_dim` successive `next_u32`
//! words) and, once the matrix kernels are blocked and parallelised, it
//! dominates FrodoPIR's `server_setup` and *is* the entire cost of
//! every `expand_hint_material` / `client_setup` call.
//!
//! # Design / architecture
//!
//! `ChaCha20Rng` is seekable at 32-bit-word granularity
//! (`set_word_pos`): the word at stream position `w` is a pure function
//! of `(seed, w)`. Each `next_u32` consumes exactly one word, so
//! splitting the output at *any* word boundary and generating each
//! chunk from a fresh RNG seeked to the chunk's start reproduces the
//! sequential stream **byte-for-byte** — the determinism contract on
//! `expand_hint_material` (server and client independently re-expand
//! the same `A` from the wire seed) is preserved exactly. Chunk starts
//! are rounded to ChaCha20's 64-word refill buffer
//! ([`parallel::CHACHA_BUFFER_WORDS`]) so no task regenerates a partial
//! buffer.
//!
//! Two realizations, same bytes:
//!
//! - [`chacha20_fill_words`] — the default path. With the `parallel`
//!   feature on (the default here) it fans out over rayon's persistent
//!   pool, so `sample_a` — and therefore every setup and every client
//!   bootstrap — is already multi-threaded.
//! - `chacha20_fill_words_scoped` — the `--no-default-features` build's
//!   fan-out for `sample_a_parallel`, on `std::thread::scope`. With the
//!   `parallel` feature on it is not compiled: `sample_a` is already
//!   parallel, so the twin simply calls it.
//!
//! # Related files
//!
//! - `frodo/sampler.rs` / `simple/sampler.rs` — `sample_a` callers
//!   (both pad the same 16-byte seed the same way; the padding stays in
//!   the per-backend samplers, mirroring the `arith.rs` convention).
//! - `backend/parallel.rs` — the worker count, the chunk rule, and the
//!   scoped fan-out.
//! - `matvec.rs` / `gemm.rs` — the same shared-kernel rationale: this
//!   is backend-agnostic machinery whose tuning must never diverge
//!   between backends.

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::backend::parallel;

/// Fill `out` with the first `out.len()` words of `ChaCha20(seed)` —
/// `seed` is the already-padded 32-byte RNG seed.
///
/// # Constraints
///
/// Byte-identical to `for cell in out { *cell = rng.next_u32() }` on a
/// fresh `ChaCha20Rng::from_seed(seed)`, regardless of thread count —
/// pinned by the unit tests. The chunk schedule depends only on
/// `out.len()` and the public worker count.
pub(crate) fn chacha20_fill_words(seed: [u8; 32], out: &mut [u32]) {
    #[cfg(feature = "parallel")]
    if out.len() >= parallel::PAR_MIN_WORDS && parallel::kernels_parallel() {
        use rayon::prelude::*;
        let chunk = parallel::balanced_chunk_len(
            out.len(),
            parallel::CHACHA_BUFFER_WORDS,
            parallel::kernel_tasks(),
        );
        out.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, part)| {
                fill_words_seq(seed, part, (ci * chunk) as u128);
            });
        return;
    }
    fill_words_seq(seed, out, 0);
}

/// Scoped-thread twin of [`chacha20_fill_words`] — **byte-identical
/// output**, and the `--no-default-features` build's only fan-out.
///
/// Falls back to the sequential fill on a single worker or below
/// [`parallel::PAR_MIN_WORDS`].
#[cfg(not(feature = "parallel"))]
pub(crate) fn chacha20_fill_words_scoped(seed: [u8; 32], out: &mut [u32]) {
    let threads = parallel::setup_threads();
    if threads <= 1 || out.len() < parallel::PAR_MIN_WORDS {
        fill_words_seq(seed, out, 0);
        return;
    }
    let chunk = parallel::balanced_chunk_len(out.len(), parallel::CHACHA_BUFFER_WORDS, threads);
    parallel::par_chunks_mut(out, chunk, |offset, part| {
        fill_words_seq(seed, part, offset as u128);
    });
}

/// Sequential fill starting at stream word `word_pos`.
fn fill_words_seq(seed: [u8; 32], out: &mut [u32], word_pos: u128) {
    let mut rng = ChaCha20Rng::from_seed(seed);
    if word_pos != 0 {
        rng.set_word_pos(word_pos);
    }
    for cell in out {
        *cell = rng.next_u32();
    }
}

#[cfg(test)]
mod tests {
    //! Pins the one contract that matters: chunked generation is
    //! byte-identical to the sequential stream at every boundary shape.

    use super::*;

    fn sequential(seed: [u8; 32], len: usize) -> Vec<u32> {
        let mut out = vec![0u32; len];
        fill_words_seq(seed, &mut out, 0);
        out
    }

    /// Forced chunk sizes — buffer-aligned, unaligned, larger than the
    /// buffer — all reproduce the sequential stream exactly. This is the
    /// primitive both fan-outs rest on, so it is pinned independently of
    /// whichever one the current feature set compiles.
    #[test]
    fn chunked_matches_sequential() {
        let seed = [0x5Au8; 32];
        for len in [1usize, 15, 16, 17, 1000, 4096, 100_003] {
            let expected = sequential(seed, len);
            for chunk in [16usize, 48, 64, 1024, 1 << 20] {
                let mut got = vec![0u32; len];
                for (ci, part) in got.chunks_mut(chunk).enumerate() {
                    fill_words_seq(seed, part, (ci * chunk) as u128);
                }
                assert_eq!(got, expected, "len={len} chunk={chunk}");
            }
        }
    }

    /// The public entry (whatever path it picks) reproduces the
    /// sequential stream on both sides of the parallel gate.
    #[test]
    fn entry_matches_sequential_across_gate() {
        let seed = [0xC3u8; 32];
        for len in [1000usize, parallel::PAR_MIN_WORDS + 37] {
            let expected = sequential(seed, len);
            let mut got = vec![0u32; len];
            chacha20_fill_words(seed, &mut got);
            assert_eq!(got, expected, "len={len}");
        }
    }

    /// The scoped twin agrees too, on both sides of its own gate.
    #[cfg(not(feature = "parallel"))]
    #[test]
    fn scoped_matches_sequential_across_gate() {
        let seed = [0x2Bu8; 32];
        for len in [1000usize, parallel::PAR_MIN_WORDS + 37] {
            let expected = sequential(seed, len);
            let mut got = vec![0u32; len];
            chacha20_fill_words_scoped(seed, &mut got);
            assert_eq!(got, expected, "len={len}");
        }
    }

    /// Seeking to a mid-stream word position matches skipping there by
    /// consuming words (the primitive the chunked fill relies on).
    #[test]
    fn seek_matches_skip() {
        let seed = [0x11u8; 32];
        let full = sequential(seed, 200);
        for start in [1usize, 16, 33, 100] {
            let mut got = vec![0u32; 200 - start];
            fill_words_seq(seed, &mut got, start as u128);
            assert_eq!(got, full[start..], "start={start}");
        }
    }
}
