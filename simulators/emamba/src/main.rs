//! eMamba simulator binary.
//!
//! Usage:
//!     emamba -c <config.yaml> -t <trace.trace>
//!
//! Mirrors `ramulator2/src/main.cpp:81-112`. We don't bring in clap to avoid
//! pulling its proc-macro dependencies — a 30-line hand parser does the job
//! for two named flags.

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut config: Option<PathBuf> = None;
    let mut trace: Option<PathBuf> = None;

    let mut iter = args.iter().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "-c" | "--config" => config = iter.next().map(PathBuf::from),
            "-t" | "--trace" => trace = iter.next().map(PathBuf::from),
            "-h" | "--help" => {
                print_help();
                return;
            }
            other => {
                eprintln!("unknown flag: {}", other);
                print_help();
                std::process::exit(2);
            }
        }
    }

    let config = match config {
        Some(p) => p,
        None => {
            eprintln!("missing -c <config.yaml>");
            std::process::exit(2);
        }
    };
    let trace = match trace {
        Some(p) => p,
        None => {
            eprintln!("missing -t <trace.trace>");
            std::process::exit(2);
        }
    };

    let result = match emamba::run(&config, &trace) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("emamba: {}", e);
            std::process::exit(1);
        }
    };

    print!("{}", result.stats.to_yaml_string());
}

fn print_help() {
    println!("emamba — eMamba cycle-accurate simulator");
    println!();
    println!("Usage:");
    println!("  emamba -c <config.yaml> -t <trace.trace>");
    println!();
    println!("Flags:");
    println!("  -c, --config <path>   YAML config (currently informational)");
    println!("  -t, --trace  <path>   op trace file");
    println!("  -h, --help            show this message");
}
