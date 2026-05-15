#![warn(missing_docs)]
//! Shared building blocks of the IKPIR protocol.
//!
//! Holds the pluggable Index-PIR backend trait family, the shipped
//! FrodoPIR backend, the wire-format bundles exchanged between
//! `IkpirServer` and `IkpirClient`, and the `IkpirError` enum.
//!
//! Production callers should not depend on this crate directly — they
//! use the re-exports from `ikpir-server` (server-side) or
//! `ikpir-client` (client-side), which expose every item under the
//! same path as before the `ikpir-common` extraction.

pub mod backend;
pub mod wire;
mod error;

pub use backend::{
    BackendWireSize, FrodoConfig, FrodoPirBackend,
    IndexPirBackend, IncrementalPirBackend, PrecomputingPirBackend,
};
pub use wire::{HintDeltaBundle, PirQueryBundle, PirResponseBundle, SegmentRowDeltas, ServerSetupBundle};
pub use error::IkpirError;
