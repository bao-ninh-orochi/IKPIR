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
//! - `sample_discrete_gaussian_into` draws a *true* discrete Gaussian
//!   `D_σ` over `ℤ` (`P(X = x) ∝ exp(−x²/(2σ²))`) via the weight-table
//!   rejection sampler of the canonical SimplePIR reference
//!   (`ahenzinger/simplepir`, `pir/gauss.go`), generalized from that
//!   reference's hardcoded `σ = 6.4` table to a runtime `σ`.
//!
//! [`sample_a_parallel`] is the optimized-setup twin of [`sample_a`]:
//! same keystream, same bytes, split across cores by seeking each
//! worker to its own offset in the ChaCha20 stream. On this branch
//! [`sample_a`] itself already fans out (see `backend/prg.rs`), so the
//! twin is the same function unless the `parallel` feature is off.
//!
//! # Related files
//!
//! - `params.rs` — `SimpleParams::seed` is the input to `sample_a`;
//!   `SimpleParams::sigma` is the width fed to the Gaussian sampler.
//! - `backend.rs` — sole caller for all three samplers.
//! - `backend/prg.rs` — the shared ChaCha20 expansion engine.

use rand::RngCore;
#[cfg(test)]
use rand::SeedableRng;
#[cfg(test)]
use rand_chacha::ChaCha20Rng;

use crate::backend::prg;

/// Zero-pad SimplePIR's 16-byte public seed to ChaCha20's 32.
///
/// # Rationale
///
/// SimplePIR specifies a `λ = 128`-bit public seed (16 bytes); ChaCha20
/// takes a 32-byte seed. We zero-pad to bridge — same trick as
/// `frodo/sampler.rs`, and kept per-backend for the same reason.
const fn padded_seed(seed: &[u8; 16]) -> [u8; 32] {
    let mut padded = [0u8; 32];
    let mut i = 0;
    while i < 16 {
        padded[i] = seed[i];
        i += 1;
    }
    padded
}

/// Build a ChaCha20Rng from a 16-byte seed — test-only, since the
/// production expansion goes through `backend/prg.rs`.
#[cfg(test)]
fn rng_from_seed(seed: &[u8; 16]) -> ChaCha20Rng {
    ChaCha20Rng::from_seed(padded_seed(seed))
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
    let mut out = vec![0u32; len];
    // Chunk-parallel, byte-identical to the sequential stream — see
    // backend/prg.rs for the seekability argument and pinning tests.
    prg::chacha20_fill_words(padded_seed(seed), &mut out);
    out
}

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
/// With the default `parallel` feature this **is** [`sample_a`] — that
/// path already fans out over rayon, and wrapping a second partition
/// around it would only oversubscribe. `--no-default-features` selects
/// the scoped-thread expansion instead, falling back to the sequential
/// stream on a single worker or below
/// [`parallel::PAR_MIN_WORDS`](crate::backend::parallel::PAR_MIN_WORDS).
///
/// # Complexity
///
/// Same `O(n_rows · lwe_dim)` ChaCha20 words as [`sample_a`], spread
/// over
/// [`parallel::setup_threads`](crate::backend::parallel::setup_threads)
/// workers.
pub fn sample_a_parallel(seed: &[u8; 16], n_rows: u32, lwe_dim: u32) -> Vec<u32> {
    #[cfg(feature = "parallel")]
    {
        sample_a(seed, n_rows, lwe_dim)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let len = (n_rows as usize)
            .checked_mul(lwe_dim as usize)
            .expect("A dimensions overflow usize");
        let mut out = vec![0u32; len];
        prg::chacha20_fill_words_scoped(padded_seed(seed), &mut out);
        out
    }
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

/// Fill `dst` with true discrete-Gaussian samples `D_σ` over `ℤ`,
/// `P(X = x) ∝ exp(−x²/(2σ²))`, encoded as two's-complement `u32` (so
/// `dst[i] as i32` is the signed value and wrapping `Z_{2^32}` arithmetic
/// is correct in the matvec). SimplePIR error distribution (§4.2, default
/// `σ = 6.4`); `σ` is a runtime parameter here, not baked into the
/// sampler.
///
/// # Algorithm
///
/// A table rejection sampler modeled on the canonical SimplePIR
/// reference (`ahenzinger/simplepir`, `pir/gauss.go`), itself modeled on
/// Martin Albrecht's `dgs` discrete-Gaussian sampler. The support is cut
/// to `x ∈ [−t, t]` with tail cut `t = ⌈20σ⌉` (the truncated mass is below
/// `e^{−t²/(2σ²)} ≈ e^{−200}` at `σ = 6.4`, negligible against the
/// `2⁻⁴⁰` per-cell decode budget `KAPPA` enforced by
/// [`simple_max_plaintext_bits`](crate::pir_params::simple_max_plaintext_bits)).
/// An acceptance-weight table `w` is built once per call: `w[0] = 1/2`
/// and `w[x] = exp(−x²/(2σ²))` for `1 ≤ x ≤ t` — the halved `w[0]`
/// compensates for `+0` and `−0` collapsing to the single value `0` under
/// the sign flip below, so the sampled law has `P(0) : P(±x)` exactly
/// `1 : exp(−x²/(2σ²))`. Each output draws an unbiased magnitude
/// `x ← Uniform{0, …, t}` (by rejection sampling on `next_u64`, never
/// plain modulo, to avoid modulo bias) and `y ← Uniform[0, 1)`, accepting
/// `x` iff `y < w[x]`; the accepted magnitude is then negated with
/// probability `1/2`. Expected iterations of the accept/reject loop are
/// `≈ (t + 1) / (σ√(π/2)) ≈ 16` at `σ = 6.4` (the halved `w[0]` makes the
/// table sum exactly `σ√(π/2)` up to exponentially small terms).
pub fn sample_discrete_gaussian_into<R: RngCore>(rng: &mut R, sigma: f64, dst: &mut [u32]) {
    debug_assert!(sigma > 0.0 && sigma.is_finite());
    if dst.is_empty() {
        return;
    }

    let t = (20.0 * sigma).ceil() as usize;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut weights = Vec::with_capacity(t + 1);
    weights.push(0.5);
    weights.extend((1..=t).map(|x| (-((x * x) as f64) / two_sigma_sq).exp()));

    // Unbiased magnitude draw m <- Uniform{0, ..., t}: `next_u64() % base`
    // alone is biased whenever `base` does not divide 2^64, so instead
    // reject draws >= zone, the largest multiple of `base` that fits in a
    // u64 — the remaining range is then an exact multiple of `base`, so
    // `% base` is exactly uniform over it. The rejection probability is at
    // most `base / 2^64`, astronomically small for any realistic tail cut.
    let base = (t as u64) + 1;
    let zone = u64::MAX - (u64::MAX % base);

    for cell in dst.iter_mut() {
        let magnitude = loop {
            let candidate = loop {
                let u = rng.next_u64();
                if u < zone {
                    break (u % base) as usize;
                }
            };
            let y = uniform_unit(rng);
            if y < weights[candidate] {
                break candidate;
            }
        };
        let negate = rng.next_u64() & 1 == 1;
        let signed = if negate {
            -(magnitude as i32)
        } else {
            magnitude as i32
        };
        *cell = signed as u32;
    }
}

/// Draw a uniform `[0, 1)` double from `rng`. Uses the high 53 bits of
/// one `next_u64` for full mantissa precision.
#[inline]
fn uniform_unit<R: RngCore>(rng: &mut R) -> f64 {
    let u = rng.next_u64() >> 11; // 53 bits
    (u as f64) * (1.0_f64 / ((1u64 << 53) as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::parallel;

    const LWE_DIM: u32 = crate::backend::simple::SimpleParams::DEFAULT_LWE_DIM;

    #[test]
    fn sample_a_determinism() {
        let seed = [0x42u8; 16];
        let a1 = sample_a(&seed, 4, LWE_DIM);
        let a2 = sample_a(&seed, 4, LWE_DIM);
        assert_eq!(a1, a2);
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

        // The true D_sigma has variance sigma^2 up to an O(e^{-2 pi^2
        // sigma^2}) correction (negligible at sigma = 6.4) -- unlike the
        // rounded continuous Gaussian this sampler replaced, whose
        // variance was sigma^2 + 1/12 (Sheppard's correction). Sample
        // variance for a normal has std ≈ σ²·√(2/n).
        let var_std = sigma * sigma * (2.0 / n as f64).sqrt();
        let expected_var = sigma * sigma;
        assert!(
            (var - expected_var).abs() < 5.0 * var_std,
            "gaussian variance = {var}, expected ≈ {expected_var} (±{:.3})",
            5.0 * var_std
        );
    }

    #[test]
    fn gaussian_support_is_bounded() {
        let sigma = 6.4_f64;
        let t = (20.0 * sigma).ceil() as i32; // 128
        let mut rng = rng_from_seed(&[0x11u8; 16]);
        let mut dst = vec![0u32; 50_000];
        sample_discrete_gaussian_into(&mut rng, sigma, &mut dst);

        let mut max_abs = 0i32;
        for &raw in &dst {
            let x = raw as i32;
            assert!(x.abs() <= t, "sample {x} exceeds tail cut t={t}");
            max_abs = max_abs.max(x.abs());
        }
        assert!(
            (max_abs as f64) > 3.0 * sigma,
            "max |x| = {max_abs} never exceeded 3*sigma = {}; tail cut may be too tight",
            3.0 * sigma
        );

        // A small sigma: t shrinks to 10 and the mode (0) should dominate.
        let sigma_small = 0.5_f64;
        let t_small = (20.0 * sigma_small).ceil() as i32; // 10
        let mut rng_small = rng_from_seed(&[0x12u8; 16]);
        let mut dst_small = vec![0u32; 20_000];
        sample_discrete_gaussian_into(&mut rng_small, sigma_small, &mut dst_small);
        let mut zeros = 0usize;
        for &raw in &dst_small {
            let x = raw as i32;
            assert!(
                x.abs() <= t_small,
                "sample {x} exceeds tail cut t={t_small} at sigma={sigma_small}"
            );
            if x == 0 {
                zeros += 1;
            }
        }
        assert!(
            zeros > dst_small.len() / 2,
            "expected most samples to be 0 at sigma={sigma_small}, got {zeros}/{}",
            dst_small.len()
        );
    }

    #[test]
    fn gaussian_determinism() {
        let sigma = 6.4_f64;
        let n = 2_000usize;

        let seed_a = [0x33u8; 32];
        let mut rng_a1 = ChaCha20Rng::from_seed(seed_a);
        let mut rng_a2 = ChaCha20Rng::from_seed(seed_a);
        let mut dst_a1 = vec![0u32; n];
        let mut dst_a2 = vec![0u32; n];
        sample_discrete_gaussian_into(&mut rng_a1, sigma, &mut dst_a1);
        sample_discrete_gaussian_into(&mut rng_a2, sigma, &mut dst_a2);
        assert_eq!(dst_a1, dst_a2, "same seed must give identical output");

        let seed_b = [0x44u8; 32];
        let mut rng_b = ChaCha20Rng::from_seed(seed_b);
        let mut dst_b = vec![0u32; n];
        sample_discrete_gaussian_into(&mut rng_b, sigma, &mut dst_b);
        assert_ne!(
            dst_a1, dst_b,
            "different seeds must (overwhelmingly likely) differ"
        );
    }

    #[test]
    fn gaussian_matches_discrete_gaussian_law() {
        let sigma = 6.4_f64;
        let t = (20.0 * sigma).ceil() as i64; // 128

        // 200_000 rather than 400_000: the accept/reject sampler burns ~2
        // next_u64 draws per loop iteration (~16 iterations/sample), which
        // is slow enough unoptimized that 400_000 samples took ~9.6s in a
        // debug build; 200_000 keeps this test in the low single-digit
        // seconds while (see below) still leaving >6 SE of margin on every
        // assertion.
        let n = 200_000usize;
        let mut rng = rng_from_seed(&[0x55u8; 16]);
        let mut dst = vec![0u32; n];
        sample_discrete_gaussian_into(&mut rng, sigma, &mut dst);

        let idx = |x: i64| (x + t) as usize;
        let mut counts = vec![0u64; (2 * t + 1) as usize];
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for &raw in &dst {
            let x = (raw as i32) as i64;
            counts[idx(x)] += 1;
            sum += x as f64;
            sum_sq += (x as f64) * (x as f64);
        }
        let c = |x: i64| counts[idx(x)] as f64;

        // (a) log-ratio of the weight table's shape, for x = 1..=floor(2*sigma)
        // = 12. The multinomial standard error of ln(c[x]/c[0]) is
        // sqrt(1/c[x] + 1/c[0]); at x=12, N=200_000 the expected counts are
        // c[12] ~ 2.1e3 and c[0] ~ 1.2e4, giving SE ~ 0.023, so the 0.15
        // tolerance below is ~6.4 SE -- thin but above the 6-SE floor.
        // (b) symmetry between +x and -x: |c[x]-c[-x]| is approximately
        // Normal(0, c[x]+c[-x]), so 6*sqrt(c[x]+c[-x]) is a 6-SE bound.
        let bound = (2.0 * sigma) as i64; // 12
        for x in 1..=bound {
            let expected = -((x * x) as f64) / (2.0 * sigma * sigma);
            let observed_pos = (c(x) / c(0)).ln();
            let observed_neg = (c(-x) / c(0)).ln();
            assert!(
                (observed_pos - expected).abs() <= 0.15,
                "x={x}: ln(c[x]/c[0])={observed_pos} vs expected {expected}"
            );
            assert!(
                (observed_neg - expected).abs() <= 0.15,
                "x=-{x}: ln(c[-x]/c[0])={observed_neg} vs expected {expected}"
            );

            let diff = (c(x) - c(-x)).abs();
            let se = (c(x) + c(-x)).sqrt();
            assert!(
                diff <= 6.0 * se,
                "x={x}: |c[x]-c[-x]|={diff} exceeds 6*SE={}",
                6.0 * se
            );
        }

        // (c) first two moments, 6-SE bounds (mean SE = sigma/sqrt(N);
        // variance relative SE ~= sqrt(2/N)).
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        let mean_se = sigma / (n as f64).sqrt();
        assert!(
            mean.abs() <= 6.0 * mean_se,
            "mean {mean} exceeds 6*SE={}",
            6.0 * mean_se
        );
        let var_se = (2.0 / n as f64).sqrt() * sigma * sigma;
        assert!(
            (var - sigma * sigma).abs() <= 6.0 * var_se,
            "var {var} exceeds 6*SE={} from sigma^2={}",
            6.0 * var_se,
            sigma * sigma
        );

        // (d) P(0) against the exact normalizer of the (tail-cut) target law.
        let mut z = 0.0f64;
        for x in -t..=t {
            z += (-((x * x) as f64) / (2.0 * sigma * sigma)).exp();
        }
        let p0_exact = 1.0 / z;
        let p0_empirical = c(0) / n as f64;
        let p0_se = (p0_exact * (1.0 - p0_exact) / n as f64).sqrt();
        assert!(
            (p0_empirical - p0_exact).abs() <= 6.0 * p0_se,
            "empirical P(0)={p0_empirical} vs exact {p0_exact}, 6*SE={}",
            6.0 * p0_se
        );
    }

    #[test]
    fn gaussian_runtime_sigma() {
        let n = 100_000usize;
        for (i, &sigma) in [2.0_f64, 12.0_f64].iter().enumerate() {
            let t = (20.0 * sigma).ceil() as i32;
            let mut rng = rng_from_seed(&[0x66u8 + i as u8; 16]);
            let mut dst = vec![0u32; n];
            sample_discrete_gaussian_into(&mut rng, sigma, &mut dst);

            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            for &raw in &dst {
                let x = raw as i32;
                assert!(
                    x.abs() <= t,
                    "sigma={sigma}: sample {x} exceeds tail cut t={t}"
                );
                sum += x as f64;
                sum_sq += (x as f64) * (x as f64);
            }
            let mean = sum / n as f64;
            let var = sum_sq / n as f64 - mean * mean;
            let var_se = (2.0 / n as f64).sqrt() * sigma * sigma;
            assert!(
                (var - sigma * sigma).abs() <= 6.0 * var_se,
                "sigma={sigma}: var {var} exceeds 6*SE={} from sigma^2={}",
                6.0 * var_se,
                sigma * sigma
            );
        }
    }
}
