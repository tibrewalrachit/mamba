//! `run(config_path, trace_path)` — top-level helper. Used by `main.rs` and
//! by integration tests / snapshots.

use emamba_core::stats::StatTree;
use emamba_memsys::AcceleratorChip;
use emamba_trace::frontend::OpTraceFrontend;
use std::path::Path;

#[derive(Debug)]
pub struct RunResult {
    pub stats: StatTree,
}

#[derive(Debug)]
pub enum RunError {
    Io(std::io::Error),
    Parse(String),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Io(e) => write!(f, "io error: {}", e),
            RunError::Parse(msg) => write!(f, "parse error: {}", msg),
        }
    }
}

impl std::error::Error for RunError {}

impl From<std::io::Error> for RunError {
    fn from(e: std::io::Error) -> Self {
        RunError::Io(e)
    }
}

pub fn run(_config_path: &Path, trace_path: &Path) -> Result<RunResult, RunError> {
    let trace_text = std::fs::read_to_string(trace_path)?;

    let frontend =
        OpTraceFrontend::from_trace_str(&trace_text).map_err(|e| RunError::Parse(e.to_string()))?;
    let chip = AcceleratorChip::default_mars();

    let mut sim = chip.into_simulation(frontend);
    sim.run_to_completion();

    let mut stats = StatTree::new();
    sim.write_stats(&mut stats, &trace_path.display().to_string());

    Ok(RunResult { stats })
}
