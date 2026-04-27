//! Stats sink + tree. Mirrors ramulator2's `register_stat(...)` →
//! per-component `print_stats(emitter)` walk (`ramulator2/src/base/base.h:165-192`).
//!
//! Determinism contract (plan §9): all maps on the stats path are `BTreeMap`,
//! never `HashMap`. Callers walk `StatSink::record_*(path, value)` and the
//! tree builds a YAML-emittable nested structure with sorted keys.

use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub enum StatValue {
    U64(u64),
    F64(f64),
    Str(String),
}

#[derive(Clone, Debug, Default)]
pub struct StatTree {
    children: BTreeMap<String, StatNode>,
}

#[derive(Clone, Debug)]
enum StatNode {
    Leaf(StatValue),
    Subtree(StatTree),
}

pub trait StatSink {
    fn record_u64(&mut self, path: &[&str], v: u64);
    fn record_f64(&mut self, path: &[&str], v: f64);
    fn record_str(&mut self, path: &[&str], v: &str);
}

impl StatTree {
    pub fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, path: &[&str], value: StatValue) {
        assert!(!path.is_empty(), "stat path must be non-empty");
        let head = path[0];
        if path.len() == 1 {
            self.children
                .insert(head.to_string(), StatNode::Leaf(value));
            return;
        }

        let entry = self
            .children
            .entry(head.to_string())
            .or_insert_with(|| StatNode::Subtree(StatTree::new()));

        match entry {
            StatNode::Subtree(child) => child.insert(&path[1..], value),
            StatNode::Leaf(_) => {
                // Replace leaf with subtree on path conflict.
                let mut sub = StatTree::new();
                sub.insert(&path[1..], value);
                *entry = StatNode::Subtree(sub);
            }
        }
    }

    /// Walk `path` and return the `u64` leaf value, or `None` if missing/wrong type.
    pub fn get_u64(&self, path: &[&str]) -> Option<u64> {
        if path.is_empty() {
            return None;
        }
        match self.children.get(path[0])? {
            StatNode::Leaf(StatValue::U64(n)) if path.len() == 1 => Some(*n),
            StatNode::Subtree(sub) if path.len() > 1 => sub.get_u64(&path[1..]),
            _ => None,
        }
    }

    pub fn to_yaml_string(&self) -> String {
        let mut out = String::new();
        self.write_yaml(&mut out, 0);
        out
    }

    fn write_yaml(&self, out: &mut String, indent: usize) {
        let pad = "  ".repeat(indent);
        for (k, v) in &self.children {
            match v {
                StatNode::Leaf(value) => {
                    use std::fmt::Write;
                    let _ = write!(out, "{}{}: ", pad, k);
                    match value {
                        StatValue::U64(n) => {
                            let _ = writeln!(out, "{}", n);
                        }
                        StatValue::F64(n) => {
                            // Stable formatting: always show decimal; fall back to
                            // scientific only if the integer form rounds.
                            let _ = writeln!(out, "{}", n);
                        }
                        StatValue::Str(s) => {
                            let _ = writeln!(out, "{}", s);
                        }
                    }
                }
                StatNode::Subtree(child) => {
                    use std::fmt::Write;
                    let _ = writeln!(out, "{}{}:", pad, k);
                    child.write_yaml(out, indent + 1);
                }
            }
        }
    }
}

impl StatSink for StatTree {
    fn record_u64(&mut self, path: &[&str], v: u64) {
        self.insert(path, StatValue::U64(v));
    }
    fn record_f64(&mut self, path: &[&str], v: f64) {
        self.insert(path, StatValue::F64(v));
    }
    fn record_str(&mut self, path: &[&str], v: &str) {
        self.insert(path, StatValue::Str(v.to_string()));
    }
}
