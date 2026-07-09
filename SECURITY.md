# Security Policy

## Scope and status

This repository is the research-prototype artifact for the CANS 2026 paper
*"Incremental Keyword Private Information Retrieval from d-ary Segmented
Cuckoo Filters"*. It has **not** been independently audited and is not yet
intended for production deployments handling real user data.

Known, deliberate limitations of the prototype's side-channel posture:

- The LWE matrix-vector products avoid data-dependent shortcuts on secret
  values (skips keyed only on the public matrix `A` are fine), and
  `IkpirClient::decode` scans fingerprints with a branchless compare as
  best-effort hardening — but a fully constant-time-audited decode path is
  explicitly out of scope for this prototype.
- The wire bundles are plain in-process data with no serialisation layer;
  any deployment must layer (and harden) its own encoding.

## Reporting a vulnerability

Please **do not open a public issue** for security-relevant findings.
Instead, use one of:

- GitHub's private vulnerability reporting on
  [`orochi-network/IKPIR`](https://github.com/orochi-network/IKPIR/security/advisories)
  ("Report a vulnerability"), or
- email the maintainer: <bao.ninh@orochi.network>.

## Supported versions

Pre-1.0: only the `main` branch receives fixes. Response and remediation are
best-effort; there is no SLA while the project is a paper artifact.
