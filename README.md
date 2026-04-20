# Incremental-Keyword-PIR

**IKPIR** is an updatable keyword-PIR scheme that swaps the binary fuse filter
of [ChalametPIR] for a **Segmented Cuckoo Filter (SCF)**. The substitution gives
two wins:

1. **Better space efficiency.** SCF reaches load factors above 0.94 at arity 4,
   so the server's hint matrix is smaller per stored item.
2. **Native updateability.** Cuckoo filters support `insert`, `delete`, and
   `update` in place — something binary fuse filters cannot offer without a
   full rebuild. We extend this to the *preprocessing matrix* with an
   incremental update algorithm whose cost is logarithmic in the DB size.

The artefact is two contributions in one workspace:

| Crate | Role |
|---|---|
| [`segmented-cuckoo-filter`](crates/segmented-cuckoo-filter/) | Stand-alone SCF library (publishable to crates.io) |
| [`ikpir-common`](crates/ikpir-common/) | Shared PIR primitives (LWE, matrix, keyword encoding) |
| [`ikpir-client`](crates/ikpir-client/) | Client: setup, query, decrypt |
| [`ikpir-server`](crates/ikpir-server/) | Server: setup, respond, **insert/delete/update** |

[ChalametPIR]: https://eprint.iacr.org/2024/092

## 60-second smoke test

```bash
git clone https://github.com/<TODO-GITHUB-USER>/incremental-keyword-pir
cd incremental-keyword-pir
just repro-smoke          # build + test + tiny bench + regenerate plots
```

`just repro-smoke` is the entry point CI uses; if it passes on your box, the
full reproduction pipeline (`just repro-all`) will too.

## Repo map

```
incremental-keyword-pir/
├── crates/
│   ├── segmented-cuckoo-filter/   # Phase A — SCF library
│   ├── ikpir-common/              # Phase B — shared PIR types
│   ├── ikpir-client/              # Phase B — client
│   └── ikpir-server/              # Phase B + C — server (+incremental update)
├── docs/
│   ├── ARCHITECTURE.md            # system diagram, module map
│   ├── SCF-DESIGN.md              # SCF technical deep-dive
│   ├── PIR-INTEGRATION.md         # how SCF replaces BFF
│   ├── INCREMENTAL-UPDATE.md      # the update protocol (novel)
│   ├── THREAT-MODEL.md            # PIR security + updateability leakage
│   ├── BENCHMARKS.md              # methodology
│   └── PLOTS.md                   # figure-by-figure guide
├── scripts/                       # plot.py, verify_results.py, reproduce.sh
├── papers/                        # our paper + cited references
├── results/
│   ├── paper/                     # committed CSVs backing every figure
│   └── plots/                     # committed PNG/SVG figures
└── Justfile                       # one-command build / bench / plot / repro
```

## Paper

> *(Title TBD.)* <TODO-AUTHOR>. <TODO-VENUE>, <TODO-YEAR>. `<TODO-DOI-OR-ARXIV>`.

See [`papers/ours/paper.pdf`](papers/ours/) (published with the repo) and
[`CITATION.cff`](CITATION.cff) for BibTeX.

```bibtex
@inproceedings{ikpir<TODO-YEAR>,
  title     = {<TODO-TITLE>},
  author    = {<TODO-AUTHOR>},
  booktitle = {<TODO-VENUE>},
  year      = {<TODO-YEAR>},
}
```

## Reproducing

See [REPRODUCING.md](REPRODUCING.md) for hardware, wall-clock expectations per
bench, and the exact command that regenerates each paper figure.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Related work

- ChalametPIR: single-round keyword PIR using binary fuse filters.
  <https://eprint.iacr.org/2024/092>. Our hint-matrix construction follows
  their design; we replace the filter and add incremental updates.
- Reference implementation: <https://github.com/itzmeanjan/ChalametPIR>. We do
  not fork; all code in this repo is written clean-room.

## Disclaimer

This is **research code** accompanying an academic paper. It has not been
audited for production use. See [SECURITY.md](SECURITY.md).
