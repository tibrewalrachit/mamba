//! Test 5 — `core::config::parse_minimal_yaml`
//! Parse the YAML schema described in the plan §5: `Frontend: { impl: ... }`.
//! Each subtree carries an `impl: <name>` field plus a free-form rest. Unknown
//! shapes must produce a clear error tied to the YAML path.

use emamba_core::config::{parse_root, ConfigError};

#[test]
fn parses_frontend_impl_field() {
    let yaml = r#"
Frontend:
  impl: OpTrace
  trace: examples/single_op.trace
"#;

    let cfg = parse_root(yaml).expect("yaml must parse");
    let frontend = cfg.section("Frontend").expect("Frontend section present");
    assert_eq!(frontend.impl_name(), "OpTrace");
    assert_eq!(
        frontend.field_str("trace").unwrap(),
        "examples/single_op.trace"
    );
}

#[test]
fn parses_nested_memory_system() {
    let yaml = r#"
Frontend:
  impl: OpTrace
MemorySystem:
  impl: AcceleratorChip
  Sequencer:
    impl: PipelinedSequencer
    pipeline_depth: 8
"#;

    let cfg = parse_root(yaml).expect("yaml must parse");
    let memsys = cfg.section("MemorySystem").unwrap();
    assert_eq!(memsys.impl_name(), "AcceleratorChip");

    let sequencer = memsys.subsection("Sequencer").unwrap();
    assert_eq!(sequencer.impl_name(), "PipelinedSequencer");
    assert_eq!(sequencer.field_u64("pipeline_depth").unwrap(), 8);
}

#[test]
fn missing_impl_field_errors() {
    let yaml = r#"
Frontend:
  trace: foo.trace
"#;

    let err = parse_root(yaml).err().expect("must error on missing impl");
    matches!(err, ConfigError::MissingImpl { .. });
}

#[test]
fn malformed_yaml_errors() {
    let yaml = "Frontend: {[ broken";
    let err = parse_root(yaml).err().expect("must error on bad yaml");
    matches!(err, ConfigError::Yaml(_));
}
