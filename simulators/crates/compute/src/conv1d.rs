//! `Conv1DUnit` — paper §4.3.
//!
//! 1D convolution slides a kernel of width `K` over a D-vector. With a fully
//! pipelined multiplier+adder chain, throughput is one element per cycle once
//! the pipeline is filled.
//!
//! `cycles_for(D) = D + (K - 1) + PIPELINE_FILL`
//!
//! The `(K - 1)` term reflects the kernel's reach; `PIPELINE_FILL` is the
//! initial multiply-add fill (default 2).

#[derive(Clone, Debug)]
pub struct Conv1DUnit {
    kernel_size: u32,
    pipeline_fill: u32,
}

impl Conv1DUnit {
    pub fn new(kernel_size: u32, pipeline_fill: u32) -> Self {
        assert!(kernel_size >= 1, "Conv1DUnit kernel must be at least 1");
        Self {
            kernel_size,
            pipeline_fill,
        }
    }

    pub fn cycles_for(&self, d: u32) -> u32 {
        d + (self.kernel_size - 1) + self.pipeline_fill
    }

    pub fn kernel_size(&self) -> u32 {
        self.kernel_size
    }
}
