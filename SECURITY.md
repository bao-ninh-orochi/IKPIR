# Security Policy

## Scope

**This is research code, not a production library.** It accompanies an
academic paper. The implementations have not been:

- independently audited,
- hardened against side-channel attacks,
- or verified against formal correctness / security proofs.

Do not deploy this code unmodified in any setting where the confidentiality
or integrity of user data matters.

## Threat model (summary)

The PIR construction inherits its security argument from the LWE assumption
and from the ChalametPIR framework. The incremental-update extension
introduces additional leakage surfaces analyzed in
[`docs/THREAT-MODEL.md`](docs/THREAT-MODEL.md), including:

- Timing channels on insert / delete (a caller observing response times
  can distinguish "fast incremental update" from "full rebuild").
- Update-size leakage to passive network observers.
- Client-linkability concerns across query rounds when the server state has
  mutated between them.

The segmented cuckoo filter itself uses **non-cryptographic hashing**
(xxHash3). It is not suitable for adversarial inputs; see the crate-level
`src/lib.rs` security notes.

## Reporting a vulnerability

If you believe you have found a security-relevant bug:

1. **Do not open a public issue.** Email `<TODO-SECURITY-EMAIL>` instead.
2. Include a proof-of-concept if possible and the commit SHA you tested.
3. We will acknowledge receipt within 7 days and, for confirmed issues,
   coordinate disclosure before any public fix.

## Known limitations

- No constant-time guarantees on filter operations.
- `delete` removes any fingerprint that matches — deleting an item never
  inserted can silently remove a colliding item's fingerprint. Document
  your call-site assumptions.
- No wire-format versioning yet; clients and servers must use the same
  workspace revision.
