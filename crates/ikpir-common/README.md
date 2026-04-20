# ikpir-common

Shared primitives used by both the Incremental-Keyword-PIR client and server:

- `params`       — LWE parameters, filter arity, concrete instantiations.
- `encoding`     — SCF-based keyword-to-position encoding (the SCF ↔ PIR bridge).
- `matrix`       — matrix types and arithmetic over `Z_q`.
- `lwe`          — noise sampling, secret keygen, (encrypt/decrypt for vectors).
- `hash`         — extra hashing beyond what SCF provides.
- `serialization` — wire format for params, hint matrix, queries, responses.

This crate is not intended for direct use; see the top-level workspace README
for the full architecture. Phase B of the project implements this crate;
until then, each module exposes a `placeholder()` fn that returns
`unimplemented!()`.
