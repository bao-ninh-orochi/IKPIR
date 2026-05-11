# CLAUDE.md — ikpir-client crate

## 1. Crate purpose

Client-side IKPIR: holds `CuckooParams` and per-segment `B::ClientState`
plus an epoch counter. Translates user-level `(key, query)` operations into
wire-level Index-PIR query/response bundles from `ikpir-server`. The client
**never** owns a `CuckooKVStore`; its only persistent material from the
server is the setup bundle.

## 2. File map

| File | Role |
|---|---|
| `src/lib.rs` | Re-exports `IkpirClient`, `IkpirClientError`, and wire / backend types from `ikpir-server` |
| `src/client.rs` | `IkpirClient<B>` generic + 7 public methods |
| `src/error.rs` | `IkpirClientError` enum (4 protocol variants + `Server(IkpirError)` forward) |

## 3. Key design decisions (the WHY)

- **Params-only state** — the client stores `CuckooParams` and per-segment
  `ClientState`. Cells, mutation log, and the KV store are server-side.
  Lookup geometry (`candidate_buckets`) is public, not secret; the client
  re-derives it on every query without any privacy cost.

- **Parallel per-segment queries** — `build_query` emits one `B::Query`
  per segment (j-th query targets row `indices[j] % segment_size` in
  segment j). The server processes each segment independently in `answer`.

- **`fp` re-derivation in `decode`** — `decode(key, resp)` re-runs
  `candidate_buckets(key)` to recover `fp` deterministically. No query
  IDs, no stashed state, no extra wire field. There is no privacy reason
  to hide `fp` from the client itself.

- **Dual-path recovery** — `apply_delta` for the steady state (strict
  monotone epoch+1 patch), `reset_from` after `full_rebuild` or after a
  `FutureDelta` gap that cannot be bridged incrementally.

## 4. Epoch state machine

```
           apply_delta(delta.epoch == self.epoch + 1)
               ┌──────────────────────────────────────┐
               │                                      ▼
 [from_setup] epoch=E  ──────────────────────▶  epoch=E+1  ···
               │
               │  apply_delta(epoch ≤ self.epoch)   → StaleDelta
               │  apply_delta(epoch > self.epoch+1) → FutureDelta
               │                                        ↓ caller must:
               └──────────────────────────────▶  reset_from(new_bundle)
```

- `apply_delta` is strict-monotone: only `delta.epoch == self.epoch + 1`.
- `decode` requires `resp.epoch == self.epoch`; mismatch → `EpochMismatch`.

## 5. Failure-mode table

| Variant | Source | Meaning |
|---|---|---|
| `StaleDelta` | `apply_delta` | `delta.epoch ≤ self.epoch` |
| `FutureDelta` | `apply_delta` | `delta.epoch > self.epoch + 1` (gap) |
| `EpochMismatch` | `decode` | server moved between query and answer |
| `MalformedBundle` | `apply_delta` / `decode` | wrong segment count or row width |
| `Server(IkpirError)` | forward | for synchronous in-process composition |

## 6. Entry points and test taxonomy

| Task | Where to look |
|---|---|
| Build a fresh client | `client.rs::IkpirClient::from_setup` |
| Issue a query | `client.rs::IkpirClient::build_query` |
| Decode a response | `client.rs::IkpirClient::decode` |
| Apply an incremental delta | `client.rs::IkpirClient::apply_delta` |
| Recover from a gap | `client.rs::IkpirClient::reset_from` |
| Debug a fingerprint mismatch | `client.rs::IkpirClient::decode` — check `candidate_buckets` + `unpack_slot_cells` |
| Integration tests | `tests/client_e2e.rs` (4 tests) |
| Benches | `benches/query_throughput.rs`, `benches/decode_throughput.rs`, `benches/apply_delta_throughput.rs`, `benches/preprocess_throughput.rs`, `benches/client_setup_latency.rs`, `benches/client_memory_footprint.rs` |

### Bench layer (under `benches/`)

- Each bench is `harness = false` and parses CLI via `clap` (see helpers
  `parse_cli` / `parse_cli_with_matches`). Per-arity dispatch happens
  through `MakeStore` / `CloneStore`; the typed scheme is picked once in
  `main` based on `--arity`.
- One invocation = one CSV row (append-mode writer); the orchestrator
  is responsible for `rm`-ing the CSV before sweeping.
- Shared helpers in `benches/helpers.rs` (deliberately duplicated across
  crates — same content lives in `ikpir-server/benches/helpers.rs`):
    - `default_num_buckets_for_arity(arity)` — academic-scale defaults
      (2-ary → 16384, 3-ary → 24576, 4-ary → 16384).
    - `populate_until_full::<S>(…)` / `populate_to_load::<S>(load_factor, …)`
      — seed a `CuckooKVStore<S>` to `TableFull` or to a target load.
    - `print_preamble(name, knobs, store_state, geom)` — the standard
      `=== <bench> ===` / Parameters / KV store / Geometry banner.
    - `run_criterion_throughput(label, elems, body)` and
      `run_criterion_throughput_batched(label, elems, setup, routine)` —
      criterion wrappers. The batched form uses `iter_custom` with an
      inner per-call `Instant` so per-iteration setup is excluded from
      the timing bracket; used by `query_throughput`, `decode_throughput`,
      `apply_delta_throughput`, `preprocess_throughput`.
- The four criterion microbenches also emit
  `target/criterion/{query_throughput,decode_throughput,apply_delta_throughput,preprocess_phase_b,preprocess_phase_c}/`
  for browsable HTML/JSON. `client_setup_latency` and
  `client_memory_footprint` stay on manual timing / closed-form math.

**Per-segment data flow (client annotations):**

```
                     ┌──────── arity-2 SCF ────────┐
key  ──candidate_buckets──▶  (fp, [b0, b1])
                                    │      │
                            seg 0   │  seg 1
                      row = b0%N ◀──┘  b1%N ──▶ row
                               │           │
build_query: B::client_query   ▼           ▼
                              Q[0]        Q[1]    ── PirQueryBundle
                               │           │
       (server processes)      ▼           ▼
                              R[0]        R[1]    ── PirResponseBundle
                               │           │
decode: B::client_decode       ▼           ▼
                          row_cells    row_cells
           slot scan: unpack_slot_cells → fp match? → value
```
