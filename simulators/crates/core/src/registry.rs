//! Explicit plugin registry. Replaces ramulator2's static-init `Factory`
//! (`ramulator2/src/base/factory.h:26-86`). RD §4 picked the explicit-registry
//! path: each impl is registered by a top-level `register_all(reg)` call
//! rather than via linker-time inventory tricks.
//!
//! Type erasure: constructors are stored as `Box<dyn Any>` keyed by
//! `(TypeId<T>, iface_name, impl_name)`, so a single `Registry` holds
//! constructors for every trait-object type.

use std::any::{Any, TypeId};
use std::collections::HashMap;

#[derive(Debug, Default, Clone)]
pub struct ConfigNode;

pub type Ctor<T> = fn(&ConfigNode) -> T;

#[derive(Debug)]
pub enum RegistryError {
    UnknownInterface(String),
    UnknownImpl { iface: String, name: String },
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::UnknownInterface(i) => write!(f, "unknown interface: {}", i),
            RegistryError::UnknownImpl { iface, name } => {
                write!(f, "unknown impl {} for interface {}", name, iface)
            }
        }
    }
}
impl std::error::Error for RegistryError {}

#[derive(Default)]
pub struct Registry {
    entries: HashMap<(TypeId, &'static str, &'static str), Box<dyn Any>>,
    known_ifaces: HashMap<TypeId, Vec<&'static str>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add<T: 'static>(
        &mut self,
        iface: &'static str,
        impl_name: &'static str,
        ctor: Ctor<T>,
    ) {
        let key = (TypeId::of::<T>(), iface, impl_name);
        self.entries.insert(key, Box::new(ctor));
        let known = self.known_ifaces.entry(TypeId::of::<T>()).or_default();
        if !known.contains(&iface) {
            known.push(iface);
        }
    }

    pub fn lookup<T: 'static>(
        &self,
        iface: &'static str,
        impl_name: &'static str,
    ) -> Result<Ctor<T>, RegistryError> {
        let tid = TypeId::of::<T>();

        let iface_known = self
            .known_ifaces
            .get(&tid)
            .map(|v| v.contains(&iface))
            .unwrap_or(false);

        if !iface_known {
            return Err(RegistryError::UnknownInterface(iface.to_string()));
        }

        let entry = self
            .entries
            .get(&(tid, iface, impl_name))
            .ok_or_else(|| RegistryError::UnknownImpl {
                iface: iface.to_string(),
                name: impl_name.to_string(),
            })?;

        let ctor = entry
            .downcast_ref::<Ctor<T>>()
            .copied()
            .expect("registry stored mismatched type for key");
        Ok(ctor)
    }
}
