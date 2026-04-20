# Development notes

Scratch pad for collaboration notes that aren't part of the paper narrative
but are worth versioning so future-you (or a co-author) has context. Keep
entries short; dated; appended to the bottom.

## Layout

- **Phase A** (this release) ports the SCF library from the research playground
  in `../Filters/Experiments/Segmented-Cuckoo-Filter` and establishes the
  workspace scaffolding, CI, bench harnesses, and reproducibility pipeline.
- **Phase B** implements the PIR core. See `docs/PIR-INTEGRATION.md` for the
  contract and `crates/ikpir-{common,client,server}/src/*` for the per-file
  TODO markers.
- **Phase C** implements the novel incremental-update protocol. See
  `docs/INCREMENTAL-UPDATE.md`.

## Reference repositories

- **SCF source:** `/Users/admin/Documents/Filters/Experiments/Segmented-Cuckoo-Filter/`
  The research playground the SCF crate was ported from. Treat as archival.
- **ChalametPIR:** <https://github.com/itzmeanjan/ChalametPIR>. Reference
  *only* for PIR data flow and wire shapes. All IKPIR code is clean-room.

## Open questions

_Append-only log. Date each entry._

- **<TODO-DATE>** — Concrete LWE parameters. Track down whether the
  ChalametPIR parameters still match current lattice-estimator output.
- **<TODO-DATE>** — Merkle commitment for patch verification in Phase C.
  Does the commitment need to be in-band (server embeds root in each
  patch) or out-of-band (published on a separate channel)?
- **<TODO-DATE>** — Is the `Segmented4aryScheme` the right SCF variant to
  wire into Phase B? `Segmented3aryScheme` has smaller `arity` → smaller
  response, but 3-ary has a segment-count restriction that complicates
  bucket sizing.

## Bench hardware history

_Append-only. Record when a paper number was regenerated on new hardware._

- <TODO-DATE> — initial SCF numbers on <TODO-CPU>, <TODO-OS>.
