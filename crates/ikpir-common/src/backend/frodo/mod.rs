//! FrodoPIR backend — LWE Index-PIR with ternary errors.
//!
//! ## Math summary
//! q = 2^32 (implicit), p = 2^plaintext_bits, Δ = q/p. A is sampled from
//! a per-segment seed; H = Aᵀ·D is the hint. Per-query: b = A·s + e + Δ·u_row;
//! response a = bᵀ·D; decode rounds a − sᵀ·H to nearest Δ-multiple.
//!
//! ## Concurrency contract
//! `client_query` mutates `ClientState::last_secret` in place. `client_decode`
//! reads it. Single-in-flight-query per ClientState; if a caller issues two
//! queries before decoding the first, the first's secret is overwritten and
//! the first decode will produce garbage. The IKPIR protocol issues one query
//! per segment per round, so this is the natural cadence.

mod arith;
mod backend;
mod params;
mod sampler;

pub use backend::{
    FrodoClientState, FrodoHint, FrodoHintMaterial, FrodoPirBackend, FrodoQuery, FrodoResponse,
    FrodoServerParams,
};
pub use params::{FrodoConfig, FrodoParams};

pub(crate) use arith::{round_p_to_q, round_q_to_p};
pub(crate) use sampler::{sample_a, sample_a_parallel, sample_ternary_into};
