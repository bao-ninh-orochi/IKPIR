//! Chunk-parallel ChaCha20 word-stream expansion — the shared engine
//! behind both backends' `sample_a`.
//!
//! # Purpose
//!
//! Expanding the LWE public matrix `A` from its 16-byte seed is a pure
//! ChaCha20 stream read (`n_rows · lwe_dim` successive `next_u32`
//! words) and, after the matrix kernels were blocked and parallelised,
//! it dominates FrodoPIR's `server_setup` (~220 ms of a ~250 ms setup
//! at the paper's mid shape on M1) and every `expand_hint_material` /
//! `client_setup` call.
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
//! are rounded to the 16-word ChaCha block size so no task recomputes a
//! partial block.
//!
//! # Related files
//!
//! - `frodo/sampler.rs` / `simple/sampler.rs` — `sample_a` callers
//!   (both pad the same 16-byte seed the same way; the padding stays in
//!   the per-backend samplers, mirroring the `arith.rs` convention).
//! - `matvec.rs` / `gemm.rs` — the same shared-kernel rationale: this
//!   is backend-agnostic machinery whose tuning must never diverge
//!   between backends.

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// ChaCha20 outputs 16 `u32` words per block; chunk boundaries snap to
/// this so no parallel task regenerates a partial block.
#[cfg(feature = "parallel")]
const CHACHA_BLOCK_WORDS: usize = 16;

/// Minimum output words before fanning out to rayon. A ~1 MiB fill
/// takes ~500 µs single-threaded — well past fork/join overhead — while
/// SimplePIR's smallest reshaped `A` stays comfortably above this too.
#[cfg(feature = "parallel")]
const PAR_MIN_WORDS: usize = 1 << 18;

/// Fill `out` with the first `out.len()` words of `ChaCha20(seed)` —
/// `seed` is the already-padded 32-byte RNG seed.
///
/// # Constraints
///
/// Byte-identical to `for cell in out { *cell = rng.next_u32() }` on a
/// fresh `ChaCha20Rng::from_seed(seed)`, regardless of thread count —
/// pinned by the unit tests. The chunk schedule depends only on
/// `out.len()`.
pub(crate) fn chacha20_fill_words(seed: [u8; 32], out: &mut [u32]) {
    #[cfg(feature = "parallel")]
    if out.len() >= PAR_MIN_WORDS {
        let tasks = rayon::current_num_threads() * 4;
        let chunk = out.len().div_ceil(tasks).div_ceil(CHACHA_BLOCK_WORDS) * CHACHA_BLOCK_WORDS;
        fill_words_chunked(seed, out, chunk);
        return;
    }
    fill_words_seq(seed, out, 0);
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

/// Parallel fill: `chunk`-word tasks, each seeked to its own offset.
#[cfg(feature = "parallel")]
fn fill_words_chunked(seed: [u8; 32], out: &mut [u32], chunk: usize) {
    use rayon::prelude::*;
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, oc)| {
        fill_words_seq(seed, oc, (ci * chunk) as u128);
    });
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

    /// Forced chunk sizes — block-aligned, unaligned, larger than the
    /// buffer — all reproduce the sequential stream exactly.
    #[cfg(feature = "parallel")]
    #[test]
    fn chunked_matches_sequential() {
        let seed = [0x5Au8; 32];
        for len in [1usize, 15, 16, 17, 1000, 4096, 100_003] {
            let expected = sequential(seed, len);
            for chunk in [16usize, 48, 1024, 1 << 20] {
                let mut got = vec![0u32; len];
                fill_words_chunked(seed, &mut got, chunk);
                assert_eq!(got, expected, "len={len} chunk={chunk}");
            }
        }
    }

    /// The public entry (whatever path it picks) reproduces the
    /// sequential stream on both sides of the parallel gate.
    #[test]
    fn entry_matches_sequential_across_gate() {
        let seed = [0xC3u8; 32];
        #[cfg(feature = "parallel")]
        let lens = [1000usize, PAR_MIN_WORDS + 37];
        #[cfg(not(feature = "parallel"))]
        let lens = [1000usize, (1 << 18) + 37];
        for len in lens {
            let expected = sequential(seed, len);
            let mut got = vec![0u32; len];
            chacha20_fill_words(seed, &mut got);
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
