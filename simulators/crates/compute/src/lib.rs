//! emamba-compute: per-stage compute units.
//!
//! Each unit is one pipeline stage of the eMamba accelerator. They share the
//! `ComputeUnit` trait — analog of ramulator2's `IDramController` — which
//! exposes `enqueue` / `tick` / `drain` and a per-op cycle-count formula
//! (paper §4–5). Tests pin formulas, not magic numbers.

pub mod conv1d;
pub mod linear;
pub mod pwl;
pub mod range_norm;
pub mod relu;
pub mod residual;
pub mod ssm;

pub use conv1d::Conv1DUnit;
pub use linear::LinearProjUnit;
pub use pwl::{PwlExpUnit, PwlSiluUnit};
pub use range_norm::RangeNormUnit;
pub use relu::ReluUnit;
pub use residual::ResidualAddUnit;
pub use ssm::{SsmOutputUnit, SsmStateUnit};
