//! Test 30 — `core::stats::record_and_dump_yaml_stable`
//! Stats path uses BTreeMap so YAML output is byte-identical across runs.

use emamba_core::stats::{StatSink, StatTree};

#[test]
fn record_and_dump_yaml_stable() {
    let mut s = StatTree::new();
    s.record_u64(&["simulator", "total_cycles"], 1643);
    s.record_u64(&["simulator", "ops_completed"], 11);
    s.record_str(&["simulator", "trace_path"], "examples/single_op.trace");
    s.record_u64(&["units", "range_norm", "ops"], 1);
    s.record_u64(&["units", "range_norm", "busy_cycles"], 54);
    s.record_u64(&["units", "conv1d", "ops"], 1);

    let a = s.to_yaml_string();
    let b = s.to_yaml_string();
    assert_eq!(a, b, "two dumps in a row must be byte-identical");

    // BTreeMap ordering: keys are sorted alphabetically.
    let pos_simulator = a.find("simulator:").unwrap();
    let pos_units = a.find("units:").unwrap();
    assert!(
        pos_simulator < pos_units,
        "alphabetical key order: simulator < units"
    );
}

#[test]
fn deterministic_across_insertion_order() {
    let mut a = StatTree::new();
    a.record_u64(&["b"], 2);
    a.record_u64(&["a"], 1);

    let mut b = StatTree::new();
    b.record_u64(&["a"], 1);
    b.record_u64(&["b"], 2);

    assert_eq!(a.to_yaml_string(), b.to_yaml_string());
}

#[test]
fn nested_paths_create_subtrees() {
    let mut s = StatTree::new();
    s.record_u64(&["memory", "scratchpad", "bank_0", "reads"], 3);
    s.record_u64(&["memory", "scratchpad", "bank_0", "writes"], 2);

    let yaml = s.to_yaml_string();
    assert!(yaml.contains("bank_0:"));
    assert!(yaml.contains("reads: 3"));
    assert!(yaml.contains("writes: 2"));
}
