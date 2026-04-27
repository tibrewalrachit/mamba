//! YAML config tree. Mirrors ramulator2's nested `impl:` schema
//! (`ramulator2/example_config.yaml`). Each section names an interface (e.g.
//! `Frontend`) and selects an impl via `impl: <name>`. Subsections recurse.
//!
//! We don't decode strongly-typed config structs at this layer — that's each
//! impl's job (RD §8). Here we only expose the tree as `Section`s with
//! `impl_name()` and `field_*` accessors, plus the YAML-error → `ConfigError`
//! adapter.

use serde_yaml::Value;
use std::collections::HashMap;

#[derive(Debug)]
pub enum ConfigError {
    Yaml(serde_yaml::Error),
    NotAMapping(String),
    MissingImpl { path: String },
    BadType { path: String, expected: &'static str },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Yaml(e) => write!(f, "yaml parse error: {}", e),
            ConfigError::NotAMapping(p) => write!(f, "config at {} is not a mapping", p),
            ConfigError::MissingImpl { path } => {
                write!(f, "config section {} is missing required `impl` field", path)
            }
            ConfigError::BadType { path, expected } => {
                write!(f, "config field {} expected {}", path, expected)
            }
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<serde_yaml::Error> for ConfigError {
    fn from(e: serde_yaml::Error) -> Self {
        ConfigError::Yaml(e)
    }
}

/// The root of a parsed config: a map of top-level interface names → sections.
#[derive(Debug, Clone)]
pub struct Config {
    sections: HashMap<String, Section>,
}

impl Config {
    pub fn section(&self, name: &str) -> Option<&Section> {
        self.sections.get(name)
    }
}

/// One interface subtree. Carries its impl name + the rest as a YAML mapping.
#[derive(Debug, Clone)]
pub struct Section {
    impl_name: String,
    fields: HashMap<String, Value>,
    path: String,
}

impl Section {
    pub fn impl_name(&self) -> &str {
        &self.impl_name
    }

    pub fn field_str(&self, key: &str) -> Option<&str> {
        self.fields.get(key).and_then(|v| v.as_str())
    }

    pub fn field_u64(&self, key: &str) -> Option<u64> {
        self.fields.get(key).and_then(|v| v.as_u64())
    }

    pub fn field_bool(&self, key: &str) -> Option<bool> {
        self.fields.get(key).and_then(|v| v.as_bool())
    }

    pub fn raw(&self, key: &str) -> Option<&Value> {
        self.fields.get(key)
    }

    pub fn subsection(&self, key: &str) -> Option<Section> {
        let v = self.fields.get(key)?;
        section_from_value(v, &format!("{}.{}", self.path, key)).ok()
    }
}

pub fn parse_root(yaml: &str) -> Result<Config, ConfigError> {
    let value: Value = serde_yaml::from_str(yaml)?;
    let map = value
        .as_mapping()
        .ok_or_else(|| ConfigError::NotAMapping("<root>".into()))?;

    let mut sections = HashMap::new();
    for (k, v) in map {
        let key = k
            .as_str()
            .ok_or_else(|| ConfigError::BadType {
                path: "<root key>".into(),
                expected: "string",
            })?
            .to_string();
        let section = section_from_value(v, &key)?;
        sections.insert(key, section);
    }
    Ok(Config { sections })
}

fn section_from_value(v: &Value, path: &str) -> Result<Section, ConfigError> {
    let map = v
        .as_mapping()
        .ok_or_else(|| ConfigError::NotAMapping(path.to_string()))?;

    let mut fields = HashMap::new();
    let mut impl_name: Option<String> = None;

    for (k, val) in map {
        let key = k
            .as_str()
            .ok_or_else(|| ConfigError::BadType {
                path: path.to_string(),
                expected: "string-keyed mapping",
            })?
            .to_string();
        if key == "impl" {
            impl_name = Some(
                val.as_str()
                    .ok_or_else(|| ConfigError::BadType {
                        path: format!("{}.impl", path),
                        expected: "string",
                    })?
                    .to_string(),
            );
        } else {
            fields.insert(key, val.clone());
        }
    }

    let impl_name = impl_name.ok_or_else(|| ConfigError::MissingImpl {
        path: path.to_string(),
    })?;

    Ok(Section {
        impl_name,
        fields,
        path: path.to_string(),
    })
}
