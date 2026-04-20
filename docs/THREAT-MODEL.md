# Threat model

## Baseline: ChalametPIR security

IKPIR inherits ChalametPIR's security argument. The PIR property — the
server learns nothing about which key the client queried — reduces to
LWE-IND-CPA under standard parameters.

We assume:

- **Honest-but-curious server** for the query-privacy property. The server
  follows the protocol but may observe all messages and keep arbitrary
  state over time.
- **Active network adversary** for the channel. We do not provide channel
  confidentiality or integrity; callers should wrap IKPIR in TLS.
- **Public database schema.** The set of keys and the parameters
  (`FilterParams`, LWE params) are public. Secrecy is only over *which
  key* the client retrieves.

## What the filter does NOT provide

- **Adversarial robustness.** SCF uses xxHash3 (non-cryptographic). An
  attacker who can pick keys and sees the SCF layout can craft inputs that
  all collide in few buckets, causing (a) artificially high false-positive
  rates or (b) TableFull conditions. Deployments exposed to untrusted key
  sources must hash keys with a keyed HMAC or similar before feeding them
  to the filter.
- **Constant-time operations.** Neither SCF nor the current LWE code is
  constant-time. Side-channel adversaries with fine-grained timing access
  can learn which bucket a kick chain landed in.

## Delta over ChalametPIR: updateability leakage

Incremental updates expose new observable state that static PIR does not:

### 1. Patch size leakage

A patch has size proportional to the SCF kick-chain length. Kick chains
are shorter early in the load curve and grow as the filter fills. An
adversary observing a sequence of patches therefore learns a noisy estimate
of how full the filter is.

**Mitigation:** pad each patch to a small number of fixed bucket sizes
(e.g. 16, 64, 256 column-writes). Choose the smallest bucket that fits.
Revealing the bucket leaks only log of the kick-chain length.

### 2. Patch-timing leakage

An incremental patch completes in `O(polylog n)` time; a full rebuild
(triggered when the filter would exceed its load ceiling) takes `O(n)`.
An observer timing the server's response can distinguish these.

**Mitigation:** constant-time response window. The server delays every
response to a fixed deadline. Throughput suffers in exchange for
indistinguishability.

### 3. Cross-patch linkability

If the server emits distinct patches per client (e.g., because of server
sharding), a malicious server can probe which client saw which patch.

**Mitigation:** all patches are broadcast identically. A published
Merkle-root commitment lets clients verify they saw the same patch.

## What we do NOT claim

- **Query-unlinkability across mutation epochs.** If client C queries for
  `k` at epoch `t_0`, then at epoch `t_1` the filter has been mutated, a
  network observer correlating `t_0` and `t_1` queries may learn that the
  two were against the same `k` if the kick-chain information from `t_1`
  touched exactly C's bucket set. Providing full query-unlinkability
  across mutation requires rerandomising each client's LWE secret per
  epoch; we leave this as future work.
- **Active-server security.** A malicious server can supply a
  fresh-looking hint matrix that is not consistent with its database.
  This results in the client receiving incorrect plaintexts, not a
  privacy break, but deployments should include an integrity layer
  (e.g., Merkle proofs of the DB commitment).
- **Post-quantum parameter selection.** The LWE parameters baked into
  Phase B are inherited from ChalametPIR; they are *not* tuned against
  the latest lattice-estimator output. Re-tune before deploying.

## Security test vectors (Phase B/C)

We will commit in `crates/ikpir-common/tests/vectors.rs`:

- A fixed (seed, key-set, hint-matrix, query, response) tuple.
- Assertion that decrypt returns the expected record.
- Assertion that the hint matrix is the **exact** output of
  `Server::new` under the seed.

These let reviewers verify that implementation drift hasn't silently
broken the protocol.
