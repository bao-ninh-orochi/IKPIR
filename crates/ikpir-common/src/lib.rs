#![warn(missing_docs)]
//! Shared building blocks of the IKPIR protocol.
//!
//! Holds the pluggable Index-PIR backend trait family (the paper's
//! **UIPIR** interface), the two shipped LWE backends (FrodoPIR and
//! SimplePIR — instantiating the paper's **RisePIR-F** and
//! **RisePIR-S**), the wire-format bundles exchanged between
//! `IkpirServer` and `IkpirClient`, the operating-point selection in
//! [`pir_params`], and the `IkpirError` enum.
//!
//! Production callers should not depend on this crate directly — they
//! use the re-exports from `ikpir-server` (server-side) or
//! `ikpir-client` (client-side), which expose every item under the
//! same path as before the `ikpir-common` extraction.

pub mod backend;
mod error;
pub mod pir_params;
pub mod wire;

pub use backend::{
    BackendWireSize, FrodoConfig, FrodoPirBackend, HintPatchMode, IncrementalPirBackend,
    IndexPirBackend, ParallelSetupBackend, PrecomputingPirBackend, ResponseRewind, SimpleConfig,
    SimplePirBackend,
};
pub use error::IkpirError;
pub use wire::{
    DeltaWireLayout, DeltaWireStats, HintDeltaBundle, PirQueryBundle, PirResponseBundle,
    SegmentRowDeltas, ServerSetupBundle, WireError,
};
