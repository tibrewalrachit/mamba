//! Test 4 — `core::registry::register_and_lookup`
//! Plays the role of ramulator2's static-init Factory (RD §4): an explicit
//! registry where each impl declares itself and gets looked up by
//! (interface_name, impl_name).

use emamba_core::registry::{Registry, RegistryError};

trait Foo {
    fn whoami(&self) -> &'static str;
}

struct Alpha;
impl Foo for Alpha {
    fn whoami(&self) -> &'static str {
        "alpha"
    }
}

#[test]
fn register_and_lookup_returns_constructor() {
    let mut reg = Registry::new();
    reg.add::<Box<dyn Foo>>("Foo", "Alpha", |_yaml| Box::new(Alpha));

    let ctor = reg.lookup::<Box<dyn Foo>>("Foo", "Alpha").unwrap();
    let instance = ctor(&serde_yaml_stub());
    assert_eq!(instance.whoami(), "alpha");
}

#[test]
fn lookup_unknown_iface_errors() {
    let reg = Registry::new();
    let err = reg
        .lookup::<Box<dyn Foo>>("Bogus", "Alpha")
        .err()
        .expect("must error on unknown iface");
    matches!(err, RegistryError::UnknownInterface(_));
}

#[test]
fn lookup_unknown_impl_errors() {
    let mut reg = Registry::new();
    reg.add::<Box<dyn Foo>>("Foo", "Alpha", |_| Box::new(Alpha));

    let err = reg
        .lookup::<Box<dyn Foo>>("Foo", "Beta")
        .err()
        .expect("must error on unknown impl");
    matches!(err, RegistryError::UnknownImpl { .. });
}

// We don't depend on serde_yaml in core yet; use a unit type as the YAML node.
fn serde_yaml_stub() -> emamba_core::registry::ConfigNode {
    emamba_core::registry::ConfigNode::default()
}
