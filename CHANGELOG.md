# Changelog

All notable changes to this workspace are documented here. Crate-level changes
are tracked in each crate's own `CHANGELOG.md`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial workspace scaffold.
- Phase A: `segmented-cuckoo-filter` library (6 schemes across 2 arities × 3
  arity classes), custom CSV-emitting bench harnesses, property-tested delete
  correctness, and a seed CSV in `results/paper/scf/`.
- Phase B/C scaffolding: `ikpir-common`, `ikpir-client`, `ikpir-server` crates
  with module stubs and placeholder APIs.
- Full reproducibility pipeline: `Justfile`, `scripts/plot.py`,
  `scripts/verify_results.py`, committed CSVs in `results/paper/`.

## [0.1.0] — TBD

The first tagged release will ship with Phase A populated and Phase B/C as
placeholder modules.
