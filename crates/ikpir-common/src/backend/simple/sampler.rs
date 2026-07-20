//! Deterministic LWE sampling primitives for the SimplePIR backend.
//!
//! # Purpose
//!
//! Three samplers shared by setup and query:
//! [`sample_a`] expands the public LWE matrix `A` from a 16-byte seed,
//! [`sample_uniform_zq_into`] fills a slice with uniform `Z_q` samples
//! (the SimplePIR secret distribution), and
//! [`sample_discrete_gaussian_into`] fills a slice with discrete-Gaussian
//! samples of standard deviation `σ` (the SimplePIR error distribution).
//!
//! # Design / architecture
//!
//! All three samplers use ChaCha20 as the underlying PRG:
//! - `sample_a` is keyed by the per-segment public seed; same seed +
//!   same `(n_rows, lwe_dim)` always produces byte-identical `A`. This
//!   is what lets `server_answer` and `client_setup` recover `A`
//!   without shipping it explicitly. Identical implementation to
//!   `frodo::sample_a`, duplicated per the project rule "don't refactor
//!   beyond what the task requires".
//! - `sample_uniform_zq_into` consumes the RNG in 32-bit words and
//!   writes them as-is — `next_u32` is already uniform over `Z_{2^32}`.
//! - `sample_discrete_gaussian_into` uses the Box–Muller transform with
//!   saturating cast to `i32` so pathological RNG output cannot wrap the
//!   `f64 → i32` cast in an undefined-behaviour way. For
//!   SimplePIR-relevant `σ ≤ 16` the saturation branch is statistically
//!   never taken (the tail probability beyond `±i32::MAX` is on the
//!   order of `exp(-(2³¹/16)² / 2)`, i.e. unobservably small).
//!
//! [`sample_a_parallel`] is the optimized-setup twin of [`sample_a`]:
//! same keystream, same bytes, split across cores by seeking each
//! worker to its own offset in the ChaCha20 stream. Duplicated from
//! `frodo/sampler.rs` under the same rule as [`sample_a`] itself.
//!
//! # Related files
//!
//! - `params.rs` — `SimpleParams::seed` is the input to `sample_a`;
//!   `SimpleParams::sigma` is the width fed to the Gaussian sampler.
//! - `backend.rs` — sole caller for all three samplers.
//! - `backend/parallel.rs` — fan-out used by [`sample_a_parallel`].

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::backend::parallel;

/// Build a ChaCha20Rng from a 16-byte seed (zero-padded to 32 bytes).
///
/// # Rationale
///
/// SimplePIR specifies a `λ = 128`-bit public seed (16 bytes); ChaCha20
/// takes a 32-byte seed. We zero-pad to bridge — same trick as
/// `frodo/sampler.rs`.
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
/// - `seed`     — 16-byte public seed (per-segment in SimplePIR).
/// - `n_rows`   — number of database rows per segment (in SimplePIR this
///   is the reshape row count `R`, not the original segment size).
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

/// Words `ChaCha20Rng` regenerates per internal refill (4 blocks ×
/// 16 words). Chunk boundaries are snapped to it so a worker's seek
/// lands on a buffer edge and costs no extra block generation.
const CHACHA_BUFFER_WORDS: usize = 64;

/// Below this many words, expanding `A` on one thread beats paying for
/// thread spawn + seek. ~1 MiB of output; the crossover is flat around
/// it, so the exact value is not critical.
const PAR_MIN_WORDS: usize = 1 << 18;

/// Multi-threaded twin of [`sample_a`] — **byte-identical output**.
///
/// # Purpose
///
/// The optimized setup path's `A` expansion (see
/// [`ParallelSetupBackend`](crate::ParallelSetupBackend)). At paper
/// scale `A` is `reshape_rows × lwe_dim` words — gigabytes of keystream
/// — and a client's `client_setup` does nothing else.
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
/// [`PAR_MIN_WORDS`].
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
    if threads <= 1 || len < PAR_MIN_WORDS {
        return sample_a(seed, n_rows, lwe_dim);
    }

    let mut out = vec![0u32; len];
    let chunk = parallel::balanced_chunk_len(len, CHACHA_BUFFER_WORDS, threads);
    parallel::par_chunks_mut(&mut out, chunk, |offset, part| {
        let mut rng = rng_from_seed(seed);
        rng.set_word_pos(offset as u128);
        for cell in part {
            *cell = rng.next_u32();
        }
    });
    out
}

/// Fill `dst` with uniform `Z_q` samples (`q = 2³²`).
///
/// # Purpose
///
/// LWE secret sampling per the SimplePIR LWE assumption (§3.1, Fig. 2):
/// `s ←R Z_q^n` uniformly. `next_u32` is already uniform on
/// `0..=2³² − 1`, so no rejection is needed.
///
/// # Arguments
///
/// - `rng` — any `RngCore`; the caller controls determinism by passing
///   a seeded RNG.
/// - `dst` — output buffer; each entry receives an unbiased `u32`.
///
/// # Rationale
///
/// SimplePIR §3.1 / Fig. 2 fix the secret distribution to uniform
/// `Z_q^n` as part of the underlying LWE assumption. §4.2's parameter
/// analysis (`n = 1024`, `q = 2³²`, error σ = 6.4, correctness error
/// `δ = 2⁻⁴⁰`) is performed against that same uniform-secret LWE
/// instance. The reference implementation at
/// `github.com/ahenzinger/simplepir` matches.
///
/// # Complexity
///
/// `O(dst.len())` PRG outputs (1 `u32` per sample).
pub fn sample_uniform_zq_into<R: RngCore>(rng: &mut R, dst: &mut [u32]) {
    for cell in dst.iter_mut() {
        *cell = rng.next_u32();
    }
}

/// Fill `dst` with discrete-Gaussian samples of standard deviation `sigma`,
/// encoded as two's-complement `u32` (so that `dst[i]` interpreted as
/// `i32` is the signed value, and wrapping arithmetic over `Z_{2^32}`
/// does the right thing in the LWE matvec).
///
/// # Purpose
///
/// LWE error sampling per SimplePIR §4.2 (`σ = 6.4` is the default).
///
/// # Arguments
///
/// - `rng`   — any `RngCore`; the caller controls determinism by passing
///   a seeded RNG.
/// - `sigma` — standard deviation of the underlying continuous Gaussian;
///   must be `> 0` and finite.
/// - `dst`   — output buffer; each entry receives one rounded
///   Gaussian sample (sign-extended `i32` cast to `u32`).
///
/// # Rationale
///
/// Box–Muller transform on pairs of `[0, 1)` uniforms drawn from
/// `next_u32`. Output is rounded to nearest integer with saturating cast
/// to `i32` — at `σ = 6.4` the saturation branch is statistically never
/// taken (tail probability beyond `±i32::MAX` is on the order of
/// `exp(-(2³¹/6.4)² / 2)`). The saturation is defensive against
/// pathological RNG output that would otherwise be undefined behaviour
/// in the `f64 → i32` cast.
///
/// # Complexity
///
/// `O(dst.len())` PRG outputs (2 `u32`s per Box–Muller pair, producing
/// 2 samples).
pub fn sample_discrete_gaussian_into<R: RngCore>(rng: &mut R, sigma: f64, dst: &mut [u32]) {
    debug_assert!(sigma > 0.0 && sigma.is_finite());
    let two_pi = std::f64::consts::TAU;
    let mut i = 0;
    while i < dst.len() {
        // Two uniform (0, 1] doubles — open at zero so ln(u1) is finite.
        let u1 = uniform_unit_open(rng);
        let u2 = uniform_unit(rng);
        let radius = sigma * (-2.0 * u1.ln()).sqrt();
        let theta = two_pi * u2;
        let z1 = radius * theta.cos();
        let z2 = radius * theta.sin();

        dst[i] = f64_round_to_i32(z1) as u32;
        i += 1;
        if i < dst.len() {
            dst[i] = f64_round_to_i32(z2) as u32;
            i += 1;
        }
    }
}

/// Draw a uniform `[0, 1)` double from `rng`. Uses the high 53 bits of
/// one `next_u64` for full mantissa precision.
#[inline]
fn uniform_unit<R: RngCore>(rng: &mut R) -> f64 {
    let u = rng.next_u64() >> 11; // 53 bits
    (u as f64) * (1.0_f64 / ((1u64 << 53) as f64))
}

/// Draw a uniform `(0, 1)` double from `rng`. Differs from
/// [`uniform_unit`] by ensuring the result is strictly positive — needed
/// for the `ln(u1)` factor in Box–Muller.
#[inline]
fn uniform_unit_open<R: RngCore>(rng: &mut R) -> f64 {
    loop {
        let u = rng.next_u64() >> 11;
        if u != 0 {
            return (u as f64) * (1.0_f64 / ((1u64 << 53) as f64));
        }
    }
}

/// Round a finite `f64` to the nearest `i32`, saturating at
/// `i32::{MIN, MAX}` outside the representable range.
///
/// # Rationale
///
/// A raw `f64 as i32` cast on out-of-range values is undefined behaviour
/// in older Rust editions and platform-specific in current ones. The
/// saturating cast is defensive — for SimplePIR-relevant σ the branch
/// is never taken, but we want a deterministic well-defined fallback.
#[inline]
fn f64_round_to_i32(x: f64) -> i32 {
    let r = x.round();
    if r >= i32::MAX as f64 {
        i32::MAX
    } else if r <= i32::MIN as f64 {
        i32::MIN
    } else {
        r as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LWE_DIM: u32 = crate::backend::simple::SimpleParams::DEFAULT_LWE_DIM;

    #[test]
    fn sample_a_determinism() {
        let seed = [0x42u8; 16];
        let a1 = sample_a(&seed, 4, LWE_DIM);
        let a2 = sample_a(&seed, 4, LWE_DIM);
        assert_eq!(a1, a2);
    }

    /// The optimized path is byte-identical to the reference. Shaped
    /// above `PAR_MIN_WORDS` so the fan-out actually runs.
    #[test]
    fn sample_a_parallel_matches_sequential() {
        let seed = [0x5Au8; 16];
        let n_rows = 256u32;
        assert!(
            (n_rows as usize) * (LWE_DIM as usize) >= PAR_MIN_WORDS,
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

    #[test]
    fn uniform_zq_determinism() {
        let mut rng1 = rng_from_seed(&[0x77u8; 16]);
        let mut rng2 = rng_from_seed(&[0x77u8; 16]);
        let mut dst1 = vec![0u32; LWE_DIM as usize];
        let mut dst2 = vec![0u32; LWE_DIM as usize];
        sample_uniform_zq_into(&mut rng1, &mut dst1);
        sample_uniform_zq_into(&mut rng2, &mut dst2);
        assert_eq!(dst1, dst2);
    }

    /// Mean of `n` uniform `u32` samples (interpreted as `f64`) should
    /// be approximately `2³¹`, within several standard deviations.
    #[test]
    fn uniform_zq_mean() {
        let n = 50_000usize;
        let mut rng = rng_from_seed(&[0xABu8; 16]);
        let mut dst = vec![0u32; n];
        sample_uniform_zq_into(&mut rng, &mut dst);
        let mean: f64 = dst.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let expected = (1u64 << 31) as f64; // ≈ 2³¹
                                            // σ of uniform on [0, 2³²) is 2³²/√12 ≈ 1.24·10⁹, so σ_mean ≈ that/√n ≈ 5.5·10⁶.
                                            // Allow a generous 5σ window.
        let sigma_mean = (1u64 << 32) as f64 / 12.0_f64.sqrt() / (n as f64).sqrt();
        assert!(
            (mean - expected).abs() < 5.0 * sigma_mean,
            "uniform Z_q mean = {mean}, expected ≈ {expected} (±{:.0})",
            5.0 * sigma_mean
        );
    }

    /// Gaussian sample mean ≈ 0, variance ≈ σ². Use generous error windows
    /// because each test pulls only a few thousand samples.
    #[test]
    fn gaussian_first_two_moments() {
        let sigma = 6.4_f64;
        let n = 20_000usize;
        let mut rng = rng_from_seed(&[0xCDu8; 16]);
        let mut dst = vec![0u32; n];
        sample_discrete_gaussian_into(&mut rng, sigma, &mut dst);

        let signed: Vec<f64> = dst.iter().map(|&x| (x as i32) as f64).collect();
        let mean: f64 = signed.iter().sum::<f64>() / n as f64;
        let var: f64 = signed.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        // σ of the sample mean is sigma/√n. Allow 5σ window.
        let sigma_mean = sigma / (n as f64).sqrt();
        assert!(
            mean.abs() < 5.0 * sigma_mean,
            "gaussian mean = {mean}, expected ≈ 0 (±{:.3})",
            5.0 * sigma_mean
        );

        // Sample variance for a normal has std ≈ σ²·√(2/n).
        let var_std = sigma * sigma * (2.0 / n as f64).sqrt();
        let expected_var = sigma * sigma;
        assert!(
            (var - expected_var).abs() < 5.0 * var_std,
            "gaussian variance = {var}, expected ≈ {expected_var} (±{:.3})",
            5.0 * var_std
        );
    }

    /// Gaussian samples are bounded in practice; for σ=6.4 nothing should
    /// land near i32::{MIN, MAX}. Sanity-check that the saturation branch
    /// stays untaken.
    #[test]
    fn gaussian_saturation_branch_untaken() {
        let sigma = 6.4_f64;
        let n = 50_000usize;
        let mut rng = rng_from_seed(&[0xEFu8; 16]);
        let mut dst = vec![0u32; n];
        sample_discrete_gaussian_into(&mut rng, sigma, &mut dst);
        for &x in &dst {
            let s = x as i32;
            // Anything beyond ±200 (≈30σ) on σ=6.4 is impossible in practice.
            assert!(
                s.abs() < 200,
                "gaussian sample {s} out of plausible σ=6.4 range"
            );
        }
    }

    #[test]
    fn gaussian_determinism() {
        let sigma = 6.4_f64;
        let mut rng1 = rng_from_seed(&[0x33u8; 16]);
        let mut rng2 = rng_from_seed(&[0x33u8; 16]);
        let mut d1 = vec![0u32; 256];
        let mut d2 = vec![0u32; 256];
        sample_discrete_gaussian_into(&mut rng1, sigma, &mut d1);
        sample_discrete_gaussian_into(&mut rng2, sigma, &mut d2);
        assert_eq!(d1, d2);
    }
}
