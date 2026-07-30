//! CLI front-end for [`ikpir_common::pir_params`] — prints the largest
//! `plaintext_bits` a backend decodes correctly at a given SCF geometry.
//!
//! # Purpose
//!
//! Single source of truth for the bench runner:
//! `scripts/lib.sh::backend_plaintext_bits` (sourced by `scripts/bench.sh`)
//! shells out to this example instead of maintaining a hand-computed lookup
//! table that can (and did) drift from the backends' actual per-segment shapes.
//!
//! # Usage
//!
//! ```text
//! cargo run -q --release -p ikpir-common --example max_plaintext_bits -- \
//!     <frodo|simple> --arity D --num-buckets NB --bucket-size B \
//!     --value-bits V [--fingerprint-bits 64] [--sigma 6.4]
//! ```
//!
//! Prints a single integer (the `plaintext_bits` to pass to the
//! benches) on stdout. The per-segment row count is derived as
//! `num_buckets / arity` (must divide exactly — the SCF constructors
//! enforce the same). Both backends now target the same per-cell
//! correctness budget (`ikpir_common::pir_params` module docs), so
//! every flag matters for both: `--arity`, `--bucket-size`,
//! `--value-bits`, and `--fingerprint-bits` all feed the per-cell
//! `row_width` / union-bound term for `frodo` too, not only `simple`.
//! The one flag still accepted-but-ignored for `frodo` is `--sigma`
//! (the FrodoPIR bound has no discrete-Gaussian error term), kept so
//! callers can use one uniform invocation for both backends.
//!
//! # Related files
//!
//! - `src/pir_params.rs` — the two selection functions and the full
//!   derivation of both correctness bounds.
//! - `scripts/lib.sh` — the bench-side caller (via `scripts/bench.sh`).

use std::process::exit;

use ikpir_common::pir_params::{frodo_max_plaintext_bits, simple_max_plaintext_bits};

const USAGE: &str = "usage: max_plaintext_bits <frodo|simple> \
     --arity D --num-buckets NB --bucket-size B --value-bits V \
     [--fingerprint-bits 64] [--sigma 6.4]";

fn die(msg: &str) -> ! {
    eprintln!("max_plaintext_bits: {msg}\n{USAGE}");
    exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((backend, flags)) = args.split_first() else {
        die("missing backend argument");
    };
    if !matches!(backend.as_str(), "frodo" | "simple") {
        die(&format!(
            "unknown backend '{backend}' (valid: frodo, simple)"
        ));
    }

    let (mut arity, mut num_buckets, mut bucket_size, mut value_bits) =
        (None::<u32>, None::<u32>, None::<u32>, None::<u32>);
    let mut fingerprint_bits: u32 = 64;
    let mut sigma: f64 = 6.4;

    let mut it = flags.iter();
    while let Some(flag) = it.next() {
        let Some(raw) = it.next() else {
            die(&format!("flag '{flag}' is missing a value"));
        };
        let parse_u32 = || -> u32 {
            raw.parse()
                .unwrap_or_else(|_| die(&format!("'{raw}' is not a valid integer for {flag}")))
        };
        match flag.as_str() {
            "--arity" => arity = Some(parse_u32()),
            "--num-buckets" => num_buckets = Some(parse_u32()),
            "--bucket-size" => bucket_size = Some(parse_u32()),
            "--value-bits" => value_bits = Some(parse_u32()),
            "--fingerprint-bits" => fingerprint_bits = parse_u32(),
            "--sigma" => {
                sigma = raw
                    .parse()
                    .unwrap_or_else(|_| die(&format!("'{raw}' is not a valid float for --sigma")));
            }
            other => die(&format!("unknown flag '{other}'")),
        }
    }

    let arity = arity.unwrap_or_else(|| die("--arity is required"));
    let num_buckets = num_buckets.unwrap_or_else(|| die("--num-buckets is required"));
    let bucket_size = bucket_size.unwrap_or_else(|| die("--bucket-size is required"));
    let value_bits = value_bits.unwrap_or_else(|| die("--value-bits is required"));

    if arity == 0 || num_buckets == 0 {
        die("--arity and --num-buckets must be positive");
    }
    if num_buckets % arity != 0 {
        die(&format!(
            "num_buckets ({num_buckets}) must be divisible by arity ({arity})"
        ));
    }
    let segment_rows = num_buckets / arity;

    let pb = match backend.as_str() {
        "frodo" => frodo_max_plaintext_bits(
            arity,
            segment_rows,
            bucket_size,
            fingerprint_bits,
            value_bits,
        ),
        "simple" => simple_max_plaintext_bits(
            arity,
            segment_rows,
            bucket_size,
            fingerprint_bits,
            value_bits,
            sigma,
        ),
        _ => unreachable!("backend validated above"),
    };
    println!("{pb}");
}
