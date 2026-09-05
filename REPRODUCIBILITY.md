# Reproducibility

This repository backs two papers. Each reports a different **client flow** of the
same scheme, and the code ships both as first-class, parallel clients
(`docs/rewind-client-mode.md`):

| Paper | Client flow | Type | Sync verb | Decode |
|---|---|---|---|---|
| CANS 2026 camera-ready, *Incremental Keyword Private Information Retrieval from d-ary Segmented Cuckoo Filters* | **client-hint-patch** | `HintPatchClient<B>` | `apply_delta` | `decode(key, resp)` |
| Extended (full) version of the same paper | **client-rewind** | `RewindClient<B>` (alias `IkpirClient<B>`) | `accumulate_delta` | `decode(key, query, resp)` |

Every server-side number (answer latency, mutation transcript, setup) and every
filter number is flow-independent and shared by both papers. Only the client
columns differ, and each flow has its own benchmark binary writing its own CSV:
**benchmark data of the two flows is never merged into one file.** Every row
of a client CSV carries a `flow` column naming the flow that produced it.

## One parameter set

Both flows run under the repository's single parameter set. There is no
"submission" set and no "camera-ready" set:

- LWE dimension `n = 1566` (RisePIR-F, FrodoPIR backend) and `n = 1275`
  (RisePIR-S, SimplePIR backend), 128-bit security under the ADPS16 cost model;
- fingerprint width `f = 64`;
- `plaintext_bits` derived per `(backend, geometry, value width)` by
  `ikpir_common::pir_params` from an explicit per-cell decode-failure budget
  `δ_cell ≤ 2⁻⁴¹ / (d · row_width)` (`κ = 40`), which `scripts/bench.sh` applies
  automatically and records in every CSV row;
- the paper's config matrix (`scripts/lib.sh`, the `PAPER_*` variables): five
  `(arity, bucket_size)` shapes × value widths 256 B / 1 kB × both backends,
  fill 0.90 and `τ = 1 %` of the slots for the mutation and setup tables,
  `m = 10⁶` keys for the online table at arity 2 and 4.

The camera-ready's numbers moved relative to the submitted version because the
correctness bound was tightened at the reviewer's request; the paper explains
that in its own text. The code carries only the tightened set.

## Environment

- Rust **1.85.0**, pinned by `rust-toolchain.toml` (rustup selects it).
- The paper's numbers were measured on the machine described in
  `docs/server-specs.txt`. Every timed path is single-threaded and non-SIMD;
  the setup preambles use all cores without changing any measured value
  (`CLAUDE.md`, *Setup in the benches*).
- Results land under `results/<crate>/` (git-ignored; one row is appended per
  invocation, so delete a stale file before a fresh sweep). Paper-scale sweeps
  take hours per table; the seconds-scale checks at the end of this file
  verify the wiring first.

## Reproduce a paper's numbers

Run every command from the workspace root. Each table script resolves the
paper matrix from `scripts/lib.sh` and loops `scripts/bench.sh` over it;
`--flow` selects which client flow's benchmark runs (the flow-independent server
and query benches run once either way). `--arity`, `--bucket-size`,
`--value-bits` and `--backend` narrow a sweep to some cells.

### CANS 2026 camera-ready → client-hint-patch

| Table | Command | Output file(s) |
|---|---|---|
| Table 2 — filter: load factor and insert / lookup / delete throughput, SCF vs standard | `./scripts/table2.sh` | `results/segmented-cuckoo/cuckoo_filter_{load_factor,insert_throughput,lookup_throughput,delete_throughput}.csv` |
| Table 3 — online: query size, response size, answer latency | `./scripts/table3.sh --flow client-hint-patch` | `results/ikpir-server/ikpir_headtohead_server_answer.csv`, `results/ikpir-client/ikpir_headtohead_client_query.csv`, `results/ikpir-client/ikpir_headtohead_client_hint_patch_decode.csv` |
| Table 4 — mutation: client `HintUpdate` throughput per insert / update / delete, transcript bytes | `./scripts/table4.sh --flow client-hint-patch` | `results/ikpir-server/ikpir_server_mutation.csv`, `results/ikpir-client/ikpir_client_hint_patch_mutation.csv` |
| Table 5 — setup: the static rebuild cost | `./scripts/table5.sh` | `results/ikpir-server/ikpir_server_setup.csv` |

The `HintUpdate` columns are the `entry` rows of
`ikpir_client_hint_patch_mutation.csv`; the `row` rows are the row-level
realization the paper prints in brackets.

### Extended (full) version → client-rewind

| Table | Command | Output file(s) |
|---|---|---|
| Filter table | `./scripts/table2.sh` | as above (flow-independent) |
| Online table | `./scripts/table3.sh --flow client-rewind` | `results/ikpir-server/ikpir_headtohead_server_answer.csv`, `results/ikpir-client/ikpir_headtohead_client_query.csv`, `results/ikpir-client/ikpir_headtohead_client_rewind_decode.csv` |
| Mutation table — client maintenance | `./scripts/table4.sh --flow client-rewind` | `results/ikpir-server/ikpir_server_mutation.csv`, `results/ikpir-client/ikpir_client_rewind_mutation.csv` |
| Client-maintenance gap (client-rewind ÷ client-hint-patch, `update` kind) | `./scripts/table4.sh --flow all` | `results/ikpir-client/ikpir_client_rewind_mutation.csv` and `results/ikpir-client/ikpir_client_hint_patch_mutation.csv` — one file per flow; the ratio is computed across the two files, never inside one |
| Staleness curve — decode latency against \|ΔD\|, then after `collect_garbage` | `for cell in 2:4 4:1 4:2; do for vb in 2048 8192; do for be in frodo simple; do ./scripts/bench.sh client_rewind_staleness --arity ${cell%%:*} --bucket-size ${cell##*:} --num-buckets $(( 1048576 / ${cell##*:} )) --value-bits $vb --backend $be --batch-size 2000 --staleness-steps 10 --queries 200; done; done; done` | `results/ikpir-client/ikpir_client_rewind_staleness.csv` |
| Setup table | `./scripts/table5.sh` | as above (flow-independent) |

The full version's arity-3 cells are part of the matrix; `--arity 3` narrows a
sweep to them.

## Checks that take seconds

| Purpose | Command |
|---|---|
| Both flows return the same query result on the same database and key, and equal a fresh client at the head (both backends, arities 2 / 3 / 4, fixed and random mutation traces) | `cargo test -p ikpir-client --test client_flow_parity` |
| Each flow's own end-to-end tests | `cargo test -p ikpir-client --test client_hint_patch_e2e --test client_hint_patch_simple_e2e` and `cargo test -p ikpir-client --test client_rewind_e2e --test client_rewind_simple_e2e` |
| Every PIR bench at a tiny config, both backends, both flows | `./scripts/smoke.sh` |
| One bench at one dev-scale config, either flow | `./scripts/bench.sh client_hint_patch_decode --backend simple` or `./scripts/bench.sh client_rewind_mutation --arity 4` |
| A table script at dev scale (one cell, tiny geometry; the extra flags override the paper geometry) | `./scripts/table4.sh --flow client-hint-patch --arity 2 --value-bits 2048 --backend frodo --num-buckets 256` or `./scripts/table3.sh --flow client-rewind --arity 2 --value-bits 2048 --backend frodo --num-buckets 256 --num-keys 900` |

## Which benchmark measures what

| Bench | Flow | Times |
|---|---|---|
| `client_hint_patch_mutation` | client-hint-patch | `HintPatchClient::apply_delta` per batch, entry- and row-level realizations (`--patch-mode`) |
| `client_rewind_mutation` | client-rewind | `RewindClient::accumulate_delta` per batch, plus the final \|ΔD\| (`pending_cells`) |
| `client_hint_patch_decode`, `headtohead_hint_patch_decode` | client-hint-patch | `HintPatchClient::decode(key, resp)`, warm |
| `client_rewind_decode`, `headtohead_rewind_decode` | client-rewind | `RewindClient::decode(key, query, resp)` at empty ΔD, warm |
| `client_rewind_staleness` | client-rewind | `RewindClient::decode` as \|ΔD\| grows, then after `collect_garbage` |
| `client_query`, `headtohead_query` | both (flow-independent) | `build_query`, the same code in both flows |
| `server_answer`, `headtohead_answer`, `server_mutation`, `server_setup` | both (flow-independent) | the server |
| `cuckoo_filter_*`, `kv_store_*` | both (flow-independent) | the filter and the KV store |

The per-bench flags and CSV columns are documented in
`crates/ikpir-client/README.md`, `crates/ikpir-server/README.md` and
`crates/segmented-cuckoo/README.md`; the sweep scripts describe themselves with
`-h`.
