//! Tensor descriptors. The simulator never holds element values — only shape,
//! dtype, base address, and memory space — because cycle counting is the goal.

use crate::Dtype;

pub type Addr = u64;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MemSpace {
    Sram,
    Dram,
}

#[derive(Clone, Debug)]
pub struct TensorDesc {
    pub base: Addr,
    pub space: MemSpace,
    pub shape: Vec<u32>,
    pub dtype: Dtype,
}

impl TensorDesc {
    pub fn bytes(&self) -> u64 {
        let elements: u64 = self.shape.iter().map(|&d| d as u64).product();
        elements * self.dtype.bytes() as u64
    }
}
