//! Thin binary for the **client-rewind** head-to-head decode bench. All
//! measurement logic lives in `flow_headtohead_decode_body.rs` (`mod
//! flow_headtohead_decode_body;`, included the same way `mod helpers;` is);
//! this file only names the flow's client type and wires up the
//! arity/backend dispatch. See `flow_headtohead_decode_body.rs` for intent,
//! method, CLI, and CSV documentation.
//!
//! **Output:** `results/ikpir_headtohead_client_rewind_decode.csv`

mod flow_headtohead_decode_body;
mod helpers;

use flow_headtohead_decode_body::Cli;
use helpers::Backend;
use ikpir_client::{FrodoConfig, FrodoPirBackend, RewindClient, SimpleConfig, SimplePirBackend};
use segmented_cuckoo::{Segmented2aryScheme, Segmented3aryScheme, Segmented4aryScheme};

fn dispatch_backend<S: helpers::MakeStore>(cli: &Cli, arity: u32, num_buckets: u32) {
    let lwe_dim = flow_headtohead_decode_body::effective_lwe_dim(cli);
    match cli.backend {
        Backend::Frodo => flow_headtohead_decode_body::run_one::<
            S,
            FrodoPirBackend,
            RewindClient<FrodoPirBackend>,
        >(cli, arity, num_buckets, FrodoConfig::with_lwe_dim(lwe_dim)),
        Backend::Simple => flow_headtohead_decode_body::run_one::<
            S,
            SimplePirBackend,
            RewindClient<SimplePirBackend>,
        >(cli, arity, num_buckets, SimpleConfig::with_lwe_dim(lwe_dim)),
    }
}

fn main() {
    if helpers::skip_when_cargo_test() {
        return;
    }
    let (cli, matches) = helpers::parse_cli_with_matches::<Cli>();
    let num_buckets =
        if matches.value_source("num_buckets") == Some(clap::parser::ValueSource::CommandLine) {
            cli.num_buckets
        } else {
            helpers::default_num_buckets_for_arity(cli.arity)
        };

    match cli.arity {
        2 => dispatch_backend::<Segmented2aryScheme>(&cli, 2, num_buckets),
        3 => dispatch_backend::<Segmented3aryScheme>(&cli, 3, num_buckets),
        4 => dispatch_backend::<Segmented4aryScheme>(&cli, 4, num_buckets),
        _ => unreachable!("clap value_parser bounds arity to 2..=4"),
    }
}
