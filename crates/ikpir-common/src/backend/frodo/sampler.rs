//! Deterministic LWE sampling primitives for the FrodoPIR backend.
//!
//! # Purpose
//!
//! Two deterministic samplers shared by setup and query:
//! [`sample_a`] expands the public LWE matrix `A` from a 16-byte seed,
//! and [`sample_ternary_into`] fills a slice with uniform `{-1, 0, +1}`
//! samples drawn from a caller-supplied RNG.
//!
//! # Design / architecture
//!
//! Both samplers use ChaCha20 as the underlying PRG:
//! - `sample_a` is keyed by the per-segment public seed; same seed +
//!   same `(n_rows, lwe_dim)` always produces byte-identical `A`. This
//!   is what lets `server_answer` and `client_setup` recover `A`
//!   without shipping it explicitly.
//! - `sample_ternary_into` consumes the RNG in 32-bit words, taking
//!   2-bit chunks and rejecting `0b11` to keep the distribution uniform
//!   on `{0, 1, 2}` (mapped to `{0, +1, -1}`).
//!
//! [`sample_a_parallel`] is the optimized-setup twin of [`sample_a`]:
//! same keystream, same bytes, split across cores by seeking each
//! worker to its own offset in the ChaCha20 stream.
//!
//! # Related files
//!
//! - `params.rs` — `FrodoParams::seed` is the input to `sample_a`.
//! - `backend.rs` — sole caller for both samplers.
//! - `backend/parallel.rs` — fan-out used by [`sample_a_parallel`].

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::backend::parallel;

/// Build a ChaCha20Rng from a 16-byte seed (zero-padded to 32 bytes).
///
/// # Rationale
///
/// FrodoPIR specifies a `λ = 128`-bit public seed (16 bytes); ChaCha20
/// takes a 32-byte seed. We zero-pad to bridge — equivalent to using
/// the first 128 bits of a 256-bit seed and is documented in
/// `frodo/mod.rs`'s math summary.
fn rng_from_seed(seed: &[u8; 16]) -> ChaCha20Rng {
    let mut padded = [0u8; 32];
    padded[..16].copy_from_slice(seed);
    ChaCha20Rng::from_seed(padded)
}

/// Sample the public matrix `A` in row-major shape
/// `(n_rows × lwe_dim)`.
///
/// # Arguments
///
/// - `seed`     — 16-byte public seed (per-segment in FrodoPIR).
/// - `n_rows`   — number of database rows per segment.
/// - `lwe_dim`  — LWE dimension `n`.
///
/// # Constraints
///
/// `n_rows · lwe_dim` must fit in `usize`.
///
/// # Returns
///
/// `Vec<u32>` of length `n_rows · lwe_dim` filled with successive 32-bit
/// outputs of `ChaCha20(seed)`. Determinism: same `(seed, n_rows,
/// lwe_dim)` always produces byte-identical output.
///
/// # Complexity
///
/// `O(n_rows · lwe_dim)` ChaCha20 outputs.
pub fn sample_a(seed: &[u8; 16], n_rows: u32, lwe_dim: u32) -> Vec<u32> {
    let len = (n_rows as usize)
        .checked_mul(lwe_dim as usize)
        .expect("A dimensions overflow usize");
    let mut rng = rng_from_seed(seed);
    let mut out = vec![0u32; len];
    for cell in &mut out {
        *cell = rng.next_u32();
    }
    out
}

/// Multi-threaded twin of [`sample_a`] — **byte-identical output**.
///
/// # Purpose
///
/// The optimized setup path's `A` expansion (see
/// [`ParallelSetupBackend`](crate::ParallelSetupBackend)). At paper
/// scale `A` is `n_rows × lwe_dim` words — gigabytes of keystream — and
/// a client's `client_setup` does nothing else.
///
/// # Rationale
///
/// `ChaCha20Rng` is seekable: `set_word_pos(i)` positions the stream at
/// its `i`-th 32-bit output. So worker `t`, owning output words
/// `[o, o + len)`, seeds an RNG from the *same* seed, seeks to `o`, and
/// fills its slice — reproducing exactly the words [`sample_a`] would
/// have written there. This is what keeps the server's and client's
/// independently expanded copies of `A` identical regardless of which
/// path either side used.
///
/// Falls back to [`sample_a`] on a single core or below
/// [`parallel::PAR_MIN_WORDS`].
///
/// # Complexity
///
/// Same `O(n_rows · lwe_dim)` ChaCha20 words as [`sample_a`], spread
/// over [`parallel::setup_threads`] workers.
pub fn sample_a_parallel(seed: &[u8; 16], n_rows: u32, lwe_dim: u32) -> Vec<u32> {
    let len = (n_rows as usize)
        .checked_mul(lwe_dim as usize)
        .expect("A dimensions overflow usize");
    let threads = parallel::setup_threads();
    if threads <= 1 || len < parallel::PAR_MIN_WORDS {
        return sample_a(seed, n_rows, lwe_dim);
    }

    let mut out = vec![0u32; len];
    let chunk = parallel::balanced_chunk_len(len, parallel::CHACHA_BUFFER_WORDS, threads);
    parallel::par_chunks_mut(&mut out, chunk, |offset, part| {
        let mut rng = rng_from_seed(seed);
        rng.set_word_pos(offset as u128);
        for cell in part {
            *cell = rng.next_u32();
        }
    });
    out
}

/// Fill `dst` with uniform ternary `{-1, 0, +1}` samples.
///
/// # Purpose
///
/// LWE secret and error sampling per FrodoPIR §4.1 (Assumption 1).
///
/// # Arguments
///
/// - `rng` — any `RngCore`; the caller controls determinism by passing
///   a seeded RNG.
/// - `dst` — output buffer; entries are written as `0`, `1`, or
///   `u32::MAX` (the latter encodes `-1` via `u32` wraparound, which
///   makes the matvec accumulator's arithmetic work in `Z_{2^32}`
///   without sign-extension).
///
/// # Rationale
///
/// Pulls 2 bits per sample and rejects `0b11` so the three remaining
/// outcomes are equiprobable. Wastes ~25% of the PRG bandwidth in
/// exchange for unbiased uniform sampling — cheap relative to the
/// downstream matvec.
///
/// # Complexity
///
/// Expected `O(dst.len() · 8/3 bits)` PRG output; worst-case bounded
/// loop because each `next_u32` always produces at least one accepted
/// sample.
pub fn sample_ternary_into<R: RngCore>(rng: &mut R, dst: &mut [u32]) {
    let mut idx = 0;
    while idx < dst.len() {
        let mut word = rng.next_u32();
        for _ in 0..16 {
            if idx == dst.len() {
                return;
            }
            match word & 0b11 {
                0 => {
                    dst[idx] = 0;
                    idx += 1;
                }
                1 => {
                    dst[idx] = 1;
                    idx += 1;
                }
                2 => {
                    dst[idx] = u32::MAX; // -1 mod 2^32
                    idx += 1;
                }
                _ => { /* 0b11 → reject */ }
            }
            word >>= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LWE_DIM: u32 = crate::backend::frodo::FrodoParams::DEFAULT_LWE_DIM;

    #[test]
    fn sample_a_determinism() {
        let seed = [0x42u8; 16];
        let a1 = sample_a(&seed, 4, LWE_DIM);
        let a2 = sample_a(&seed, 4, LWE_DIM);
        assert_eq!(a1, a2);
    }

    #[test]
    fn sample_a_distinctness() {
        let seed1 = [0x11u8; 16];
        let seed2 = [0x22u8; 16];
        let a1 = sample_a(&seed1, 4, LWE_DIM);
        let a2 = sample_a(&seed2, 4, LWE_DIM);
        let differ = a1.iter().zip(a2.iter()).filter(|(x, y)| x != y).count();
        assert!(
            differ * 100 >= a1.len() * 99,
            "fewer than 99% cells differ: {differ}/{}",
            a1.len()
        );
    }

    #[test]
    fn sample_a_shape() {
        let seed = [0u8; 16];
        let n_rows = 4u32;
        let a = sample_a(&seed, n_rows, LWE_DIM);
        assert_eq!(a.len(), (n_rows * LWE_DIM) as usize);
    }

    /// The optimized path is byte-identical to the reference. Shaped
    /// above `parallel::PAR_MIN_WORDS` so the fan-out actually runs.
    #[test]
    fn sample_a_parallel_matches_sequential() {
        let seed = [0x5Au8; 16];
        let n_rows = 256u32;
        assert!(
            (n_rows as usize) * (LWE_DIM as usize) >= parallel::PAR_MIN_WORDS,
            "test shape must exceed the parallel threshold"
        );
        assert_eq!(
            sample_a_parallel(&seed, n_rows, LWE_DIM),
            sample_a(&seed, n_rows, LWE_DIM)
        );
    }

    /// Pins the primitive `sample_a_parallel` rests on: a fresh RNG
    /// seeked to word `o` emits exactly the words the sequential stream
    /// has at `o` — for chunk lengths on and off the 64-word buffer
    /// edge, and for a chunk longer than the whole output.
    #[test]
    fn chacha_seek_reproduces_stream_at_every_chunking() {
        let seed = [0xC3u8; 16];
        let expected = sample_a(&seed, 8, 512);
        for chunk in [1usize, 15, 16, 63, 64, 100, 4096, 9000] {
            let mut got = vec![0u32; expected.len()];
            for (i, part) in got.chunks_mut(chunk).enumerate() {
                let mut rng = rng_from_seed(&seed);
                rng.set_word_pos((i * chunk) as u128);
                for cell in part {
                    *cell = rng.next_u32();
                }
            }
            assert_eq!(got, expected, "chunk={chunk}");
        }
    }

    #[test]
    fn ternary_distribution() {
        let n: usize = 100_000;
        let mut rng = rng_from_seed(&[0xABu8; 16]);
        let mut dst = vec![0u32; n];
        sample_ternary_into(&mut rng, &mut dst);

        let count_zero = dst.iter().filter(|&&x| x == 0).count();
        let count_one = dst.iter().filter(|&&x| x == 1).count();
        let count_neg = dst.iter().filter(|&&x| x == u32::MAX).count();

        // σ = sqrt(2N/9) for each bucket under uniform multinomial
        let sigma = ((2.0 * n as f64) / 9.0).sqrt();
        let lo = 5.0f64.mul_add(-sigma, n as f64 / 3.0) as usize;
        let hi = 5.0f64.mul_add(sigma, n as f64 / 3.0) as usize;

        assert!(
            (lo..=hi).contains(&count_zero),
            "zero count {count_zero} out of [{lo}, {hi}]"
        );
        assert!(
            (lo..=hi).contains(&count_one),
            "one count {count_one} out of [{lo}, {hi}]"
        );
        assert!(
            (lo..=hi).contains(&count_neg),
            "neg count {count_neg} out of [{lo}, {hi}]"
        );
    }

    #[test]
    fn ternary_determinism() {
        let mut rng1 = rng_from_seed(&[0x77u8; 16]);
        let mut rng2 = rng_from_seed(&[0x77u8; 16]);
        let mut dst1 = vec![0u32; LWE_DIM as usize];
        let mut dst2 = vec![0u32; LWE_DIM as usize];
        sample_ternary_into(&mut rng1, &mut dst1);
        sample_ternary_into(&mut rng2, &mut dst2);
        assert_eq!(dst1, dst2);
    }

    #[test]
    fn ternary_support() {
        let mut rng = rng_from_seed(&[0x55u8; 16]);
        let mut dst = vec![0u32; LWE_DIM as usize];
        sample_ternary_into(&mut rng, &mut dst);
        for &v in &dst {
            assert!(
                v == 0 || v == 1 || v == u32::MAX,
                "unexpected ternary value: {v}"
            );
        }
    }
}
