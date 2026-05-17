//! SimplePIR parameter types.
//!
//! # Purpose
//!
//! Two types, mirroring the FrodoPIR split:
//!
//! - [`SimpleParams`] — per-segment runtime values
//!   (`lwe_dim`, `plaintext_bits`, `sigma`, public seed). Built once per
//!   `server_setup` call and threaded into the matvec routines.
//! - [`SimpleConfig`] — user-facing tunable knobs persisted by
//!   `IkpirServer` and re-used across every `full_rebuild`.
//!
//! # Design / architecture
//!
//! `SimpleConfig` is the small, stable surface a deployment commits to;
//! `SimpleParams` is the larger object actually consumed by the math.
//! The split keeps the IKPIR server holding only the user-facing knobs
//! (`lwe_dim`, `sigma`) and assembling the runtime params per-segment.
//!
//! # Related files
//!
//! - `backend.rs` — sole consumer; reads `SimpleParams` per call and
//!   `SimpleConfig` once per setup.
//! - `ikpir-server::IkpirServer` — persists `SimpleConfig` for the
//!   server's lifetime.

/// SimplePIR LWE parameters.
///
/// # Purpose
///
/// Per-segment runtime values needed by the SimplePIR matvec
/// (`lwe_dim`, `plaintext_bits`, `sigma`, public seed). One instance is
/// built per `server_setup` call.
///
/// # Rationale
///
/// `q = 2^32` is implicit (native `u32` wraparound; matches the
/// `Vec<u32>` hint shape required by the trait family). The secret is
/// sampled uniformly over `Z_q` (matches the reference repo at
/// `github.com/ahenzinger/simplepir`); the error is a discrete Gaussian
/// of width `sigma`, defaulting to the SimplePIR §4.2 value `σ = 6.4`.
#[derive(Clone, Debug)]
pub struct SimpleParams {
    /// LWE dimension `n` (default 1024 for 128-bit security per SimplePIR §4.2).
    pub lwe_dim: u32,
    /// log₂ of the plaintext modulus `p`. Set per-call from the trait's
    /// `plaintext_bits` argument. Required: `1 ≤ plaintext_bits ≤ 31`.
    pub plaintext_bits: u32,
    /// Standard deviation of the discrete-Gaussian error distribution.
    /// SimplePIR §4.2 specifies `σ = 6.4`.
    pub sigma: f64,
    /// Public seed for sampling `A` (paper's λ-bit μ; we use 128 bits to
    /// match FrodoPIR's seed width).
    pub seed: [u8; 16],
}

impl SimpleParams {
    /// Default LWE dimension for 128-bit security (SimplePIR §4.2,
    /// `n = 2^10 = 1024`).
    pub const DEFAULT_LWE_DIM: u32 = 1024;

    /// Default error standard deviation (SimplePIR §4.2).
    pub const DEFAULT_SIGMA: f64 = 6.4;

    /// Construct LWE parameters with explicit dimension, plaintext bit
    /// width, error σ, and 128-bit seed used to sample the public
    /// matrix `A`.
    ///
    /// # Arguments
    ///
    /// - `lwe_dim`        — LWE dimension `n`; must be positive.
    /// - `plaintext_bits` — log₂ of the plaintext modulus; in `1..=31`.
    /// - `sigma`          — discrete-Gaussian error σ; must be `> 0`.
    /// - `seed`           — 128-bit public seed for the `A` sampler.
    ///
    /// # Constraints
    ///
    /// Panics if `plaintext_bits` is outside `1..=31`, if `lwe_dim == 0`,
    /// or if `sigma <= 0`.
    pub fn new(lwe_dim: u32, plaintext_bits: u32, sigma: f64, seed: [u8; 16]) -> Self {
        assert!(
            (1..=31).contains(&plaintext_bits),
            "plaintext_bits must be in 1..=31, got {plaintext_bits}"
        );
        assert!(lwe_dim > 0, "lwe_dim must be positive");
        assert!(sigma > 0.0 && sigma.is_finite(), "sigma must be positive and finite, got {sigma}");
        Self { lwe_dim, plaintext_bits, sigma, seed }
    }
}

/// User-facing tunable knobs for [`SimplePirBackend`](crate::SimplePirBackend).
///
/// # Purpose
///
/// The small, stable backend surface persisted by `IkpirServer`. Threaded
/// into every `B::server_setup` call (initial setup *and* every
/// `full_rebuild`), so a single config controls the LWE dimension and
/// error width across the server's lifetime.
///
/// # Rationale
///
/// Use [`SimpleConfig::default`] for the SimplePIR §4.2 128-bit security
/// setting (`lwe_dim = 1024`, `sigma = 6.4`). Use
/// [`SimpleConfig::with_lwe_dim`] / [`SimpleConfig::with_sigma`] /
/// [`SimpleConfig::new`] to override (e.g. for benchmarking a reduced
/// parameter set).
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleConfig {
    /// LWE dimension `n`. Larger values give stronger security at higher
    /// preprocessing/query cost. Must be > 0.
    pub lwe_dim: u32,
    /// Discrete-Gaussian error σ. Must be `> 0` and finite. The paper's
    /// default is `6.4`.
    pub sigma: f64,
}

impl SimpleConfig {
    /// New config with both knobs explicit.
    ///
    /// # Arguments
    ///
    /// - `lwe_dim` — LWE dimension `n`; must be positive.
    /// - `sigma`   — discrete-Gaussian error σ; must be `> 0` and finite.
    ///
    /// # Constraints
    ///
    /// Panics if `lwe_dim == 0` or `sigma <= 0`.
    pub fn new(lwe_dim: u32, sigma: f64) -> Self {
        assert!(lwe_dim > 0, "lwe_dim must be positive");
        assert!(sigma > 0.0 && sigma.is_finite(), "sigma must be positive and finite, got {sigma}");
        Self { lwe_dim, sigma }
    }

    /// New config overriding only `lwe_dim`; `sigma` stays at the default
    /// `6.4`.
    ///
    /// # Constraints
    ///
    /// Panics if `lwe_dim == 0`.
    pub fn with_lwe_dim(lwe_dim: u32) -> Self {
        Self::new(lwe_dim, SimpleParams::DEFAULT_SIGMA)
    }

    /// New config overriding only `sigma`; `lwe_dim` stays at the default
    /// `1024`.
    ///
    /// # Constraints
    ///
    /// Panics if `sigma <= 0` or not finite.
    pub fn with_sigma(sigma: f64) -> Self {
        Self::new(SimpleParams::DEFAULT_LWE_DIM, sigma)
    }
}

impl Default for SimpleConfig {
    /// 128-bit security default (`lwe_dim = 1024`, `sigma = 6.4`,
    /// SimplePIR §4.2).
    fn default() -> Self {
        Self { lwe_dim: SimpleParams::DEFAULT_LWE_DIM, sigma: SimpleParams::DEFAULT_SIGMA }
    }
}
