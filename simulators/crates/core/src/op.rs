//! Op — what flows through the simulator. Mirrors ramulator2's `Request`
//! (`ramulator2/src/base/request.h:12-44`), retargeted from DRAM commands to
//! accelerator operators.
//!
//! Cycle counting is the goal; the simulator never holds element values. Each
//! op carries tensor descriptors (shape + dtype + base addr + memory space)
//! plus operator-specific parameters via the typed `OpParams` enum.

use crate::TensorDesc;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum OpKind {
    RangeNorm,
    Conv1D,
    SsmStateUpdate,
    SsmOutput,
    PwlSilu,
    PwlExp,
    Relu,
    LinearProj,
    ResidualAdd,
    Load,
    Store,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OpParams {
    /// RANGE_NORM, PWL_SILU, PWL_EXP, RELU, RESIDUAL — single shape parameter D.
    Elementwise { d: u32 },
    /// CONV1D — D-vector with kernel size K.
    Conv1D { d: u32, k: u32 },
    /// SSM_STATE / SSM_OUTPUT — D-vector, N-state, expansion factor E.
    Ssm { d: u32, n: u32, e: u32 },
    /// LINEAR — in_dim × out_dim matrix-vector multiply.
    Linear { in_dim: u32, out_dim: u32 },
    /// LOAD / STORE — explicit byte count moved between SRAM and DRAM.
    Memmove { bytes: u32 },
}

pub struct Op {
    pub id: u64,
    pub layer: u32,
    pub token: u32,
    pub kind: OpKind,
    pub inputs: Vec<TensorDesc>,
    pub outputs: Vec<TensorDesc>,
    pub params: OpParams,
    pub scratch: [i32; 4],
    pub callback: Option<Box<dyn FnOnce(&Op) + Send>>,
}

impl std::fmt::Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Op")
            .field("id", &self.id)
            .field("layer", &self.layer)
            .field("token", &self.token)
            .field("kind", &self.kind)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("params", &self.params)
            .field("scratch", &self.scratch)
            .field("callback", &self.callback.as_ref().map(|_| "<fn>"))
            .finish()
    }
}
