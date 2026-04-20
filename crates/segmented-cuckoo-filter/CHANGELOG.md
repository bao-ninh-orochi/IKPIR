# Changelog

All notable changes to this crate are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 6 cuckoo filter variants: Standard and Segmented, each in 2-ary, 3-ary,
  and 4-ary forms. Public type aliases: `StandardCuckooFilter`,
  `SegmentedCuckooFilter`, `Standard3aryCuckooFilter`,
  `Segmented3aryCuckooFilter`, `Standard4aryCuckooFilter`,
  `Segmented4aryCuckooFilter`.
- `IndexScheme` trait abstracting the k-ary / segmented distinction; six
  scheme structs implement it.
- Bit-packed `TagTable` for arbitrary fingerprint widths.
- Custom CSV-emitting bench harnesses for load factor, insert / lookup /
  delete throughput, false-positive rate, eviction chains, and degree
  distribution.
- Property-based tests for delete round-trip correctness (proptest).

## [0.1.0] — TBD

First tagged release alongside paper submission.
