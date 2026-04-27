//! emamba-core: domain-free primitives shared by every other crate.

pub mod config;
pub mod dtype;
pub mod op;
pub mod registry;
pub mod tensor;
pub mod tick;

pub use dtype::Dtype;
pub use op::{Op, OpKind, OpParams};
pub use tensor::{Addr, MemSpace, TensorDesc};
pub use tick::{Cycle, Tick};
