//! ComputeUnit-trait-impl wrappers around the pure cycle-formula structs.
//!
//! Each wrapper holds the formula struct plus a `PipelineSlot`. They impl
//! `ComputeUnit` (trait) + `Tick` so the sequencer can drive them.

pub mod range_norm;
pub mod conv1d;
pub mod ssm;
pub mod pwl;
pub mod relu;
pub mod linear;
pub mod residual;
pub mod memmove;

pub use range_norm::RangeNormStage;
pub use conv1d::Conv1DStage;
pub use ssm::{SsmOutputStage, SsmStateStage};
pub use pwl::{PwlExpStage, PwlSiluStage};
pub use relu::ReluStage;
pub use linear::LinearProjStage;
pub use residual::ResidualAddStage;
pub use memmove::MemmoveStage;
