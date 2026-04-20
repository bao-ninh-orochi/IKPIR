#![warn(missing_docs)]
//! Shared primitives for the Incremental-Keyword-PIR scheme.
//!
//! This crate holds everything that both the client and the server need
//! to agree on: concrete LWE parameters, the Segmented-Cuckoo-Filter →
//! row-position encoding, matrix arithmetic over `Z_{2^32}`, the small LWE
//! toolkit, and the wire format.
//!
//! The module split mirrors the Phase B plan:
//!
//! | Module            | Responsibility                                       |
//! |-------------------|------------------------------------------------------|
//! | [`params`]        | LWE dimension, arity, plaintext layout, sizing rules |
//! | [`hash`]          | SCF candidate dispatcher + ChaCha20 PRG for `A`      |
//! | [`matrix`]        | Row-major `u32` matrix & vector ops over `Z_{2^32}`  |
//! | [`lwe`]           | Ternary sampling, encode / decode / round-scaled     |
//! | [`cuckoo_store`]  | Segmented cuckoo map: `(key, value)` placement + bucket row encoding |
//! | [`serialization`] | Versioned tight wire format for params / matrices    |
//!
//! See `docs/PIR-INTEGRATION.md` for the end-to-end data flow and
//! `docs/THREAT-MODEL.md` for the security claims.

pub mod cuckoo_store;
pub mod hash;
pub mod lwe;
pub mod matrix;
pub mod params;
pub mod serialization;

pub use params::{Arity, FilterParams, ParamError, LWE_DIMENSION, SEED_LEN};
