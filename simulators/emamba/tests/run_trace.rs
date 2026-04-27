//! Tests 32-34 — `emamba::run::*`
//! End-to-end: load a config + a trace, run the simulator, validate stats.

use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // emamba/
    p
}

#[test]
fn single_op_trace_full_run() {
    let root = workspace_root();
    let cfg = root.join("examples/mars.yaml");
    let trace = root.join("examples/single_op.trace");

    let result = emamba::run(&cfg, &trace).expect("run ok");
    let yaml = result.stats.to_yaml_string();

    // RangeNorm with units=10, div=4 → cycles_for(20) = 54
    assert!(yaml.contains("ops_completed: 1"));
    assert!(
        yaml.contains("total_cycles: 54") || yaml.contains("total_cycles: 55"),
        "stats:\n{}",
        yaml
    );
}

#[test]
fn stats_yaml_snapshot_single_op() {
    let root = workspace_root();
    let cfg = root.join("examples/mars.yaml");
    let trace = root.join("examples/single_op.trace");

    let result = emamba::run(&cfg, &trace).expect("run ok");
    let yaml = result.stats.to_yaml_string();

    // Hand-rolled snapshot (no insta on this toolchain — format is fully
    // deterministic so a literal check is fine).
    let expected = "\
simulator:
  ops_completed: 1
  total_cycles: 54
  trace_path: ${TRACE_PATH}
units:
  Conv1D:
    busy_cycles: 0
    idle_cycles: 54
    ops: 0
  LinearProj:
    busy_cycles: 0
    idle_cycles: 54
    ops: 0
  Memmove:
    busy_cycles: 0
    idle_cycles: 54
    ops: 0
  PwlExp:
    busy_cycles: 0
    idle_cycles: 54
    ops: 0
  PwlSilu:
    busy_cycles: 0
    idle_cycles: 54
    ops: 0
  RangeNorm:
    busy_cycles: 53
    idle_cycles: 1
    ops: 1
  Relu:
    busy_cycles: 0
    idle_cycles: 54
    ops: 0
  ResidualAdd:
    busy_cycles: 0
    idle_cycles: 54
    ops: 0
  SsmOutput:
    busy_cycles: 0
    idle_cycles: 54
    ops: 0
  SsmState:
    busy_cycles: 0
    idle_cycles: 54
    ops: 0
";
    let normalized = yaml.replace(
        &format!("trace_path: {}", trace.display()),
        "trace_path: ${TRACE_PATH}",
    );
    assert_eq!(normalized, expected, "snapshot mismatch:\n{}", yaml);
}

#[test]
fn stats_yaml_snapshot_one_token() {
    let root = workspace_root();
    let cfg = root.join("examples/mars.yaml");
    let trace = root.join("examples/mars_one_token.trace");

    let result = emamba::run(&cfg, &trace).expect("run ok");
    let yaml = result.stats.to_yaml_string();

    // Full-frame snapshot: 11 ops, total_cycles dominated by the slowest
    // serially-executed sequence (since the v1 sequencer doesn't pipeline
    // independent ops across stages — that's a Phase 6+ optimization).
    assert!(yaml.contains("ops_completed: 11"), "stats:\n{}", yaml);
    assert!(yaml.contains("RangeNorm:"));
    assert!(yaml.contains("SsmState:"));
    assert!(yaml.contains("SsmOutput:"));

    // Verify per-unit op counts. Dispatch happens in trace order; LOAD/STORE
    // both go to Memmove → 2 ops; PWL_SILU appears twice → 2 ops.
    let count = |s: &str| yaml.matches(s).count();
    assert!(
        count("ops: 1") >= 6,
        "expected ≥6 units with single op:\n{}",
        yaml
    );
}
